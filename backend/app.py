from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_project_env
from openrouter import OpenRouter
from retrieval.hybrid_search import _load_bm25_bundle, bm25_search, qdrant_search
from retrieval.product_dictionary import enrich_query_with_product_names
from retrieval.rag_generate import build_context_str, build_system_prompt
from retrieval.rrf import reciprocal_rank_fusion
from groq import Groq

load_project_env()

app = FastAPI(title="Gloriapets RAG API")

# CORS for local/dev frontends (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://gloriapets-chatbot.innovin.win",
    "http://localhost:3000",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    history: list[dict[str, str]] | None = None
    top_k: int | None = None
    top_n: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    rewritten_query: str | None = None
    enriched_query: str | None = None


@lru_cache(maxsize=1)
def _get_bm25_bundle() -> dict[str, Any]:
    return _load_bm25_bundle(Path("data/index/bm25.pkl"))


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


@lru_cache(maxsize=1)
def _get_embedder() -> Any:
    model_name = os.getenv("EMBEDDING_MODEL", os.getenv("HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name, device="cpu")


def _call_openrouter(system_prompt: str, query: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    with OpenRouter(
        http_referer="ragbot-poc",
        x_open_router_title="ragbot-poc",
        api_key=api_key,
    ) as open_router:
        res = open_router.chat.send(
            model=os.getenv("OPENROUTER_MODEL", "qwen/qwen-plus"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            stream=False,
            temperature=0.0,
        )

    if hasattr(res, "choices") and res.choices:
        return res.choices[0].message.content
    return str(res)


def _openrouter_chat(messages: list[dict[str, str]], *, model: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    with OpenRouter(
        http_referer="ragbot-poc",
        x_open_router_title="ragbot-poc",
        api_key=api_key,
    ) as open_router:
        res = open_router.chat.send(
            model=model,
            messages=messages,
            stream=False,
            temperature=0.0,
        )

    if hasattr(res, "choices") and res.choices:
        return res.choices[0].message.content
    return str(res)


def _get_langfuse_client():
    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    if not (pub and sec):
        return None
    # Some Langfuse SDK versions read LANGFUSE_HOST; keep compatibility with LANGFUSE_BASE_URL.
    if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL", "")
    try:
        from langfuse import get_client  # type: ignore

        return get_client()
    except Exception:
        return None


def _log_langfuse_status_once() -> None:
    # Print a single startup line so it's obvious in Uvicorn logs.
    try:
        enabled = _langfuse_enabled()
        host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or ""
        if enabled:
            print(f"[Langfuse] enabled ({host})" if host else "[Langfuse] enabled")
        else:
            print("[Langfuse] disabled (missing keys, dependency not installed, or client init failed)")
    except Exception:
        # Never crash the app because of logging.
        pass


def _langfuse_enabled() -> bool:
    lf = _get_langfuse_client()
    return lf is not None and hasattr(lf, "start_observation")


@app.on_event("startup")
def _on_startup() -> None:
    _log_langfuse_status_once()


_SKU_PATTERN = r"\b[A-Z]{2,}[A-Z0-9/*.-]{1,}\b"


def _looks_like_sku(value: str) -> bool:
    if not value:
        return False
    return any(ch.isalpha() for ch in value) and any(ch.isdigit() for ch in value)


def _extract_skus_from_query(query: str) -> list[str]:
    import re

    found: list[str] = []
    for candidate in re.findall(_SKU_PATTERN, (query or "").upper()):
        if _looks_like_sku(candidate) and candidate not in found:
            found.append(candidate)
    return found


def _extract_retrieved_pages(retrieved_chunks: list[dict[str, Any]]) -> list[str]:
    pages = set()
    for c in retrieved_chunks:
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else None
        p = None
        if meta:
            p = meta.get("physical_page_number", meta.get("page_number"))
        if p is None:
            p = c.get("physical_page_number", c.get("page_number"))
        if p is None:
            continue
        pages.add(str(p))
    return sorted(pages)


def _extract_openrouter_stream_delta(event: Any) -> str:
    # The OpenRouter SDK returns a mix of event types depending on mode/version:
    # - raw strings
    # - objects with .choices[0].delta.content
    # - dicts with ["choices"][0]["delta"]["content"]
    if isinstance(event, str):
        return event
    if hasattr(event, "choices") and getattr(event, "choices", None):
        try:
            delta = event.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                return str(delta.content)
        except Exception:
            pass
    if isinstance(event, dict) and "choices" in event:
        try:
            delta = event["choices"][0].get("delta", {})
            if isinstance(delta, dict) and "content" in delta and delta["content"]:
                return str(delta["content"])
        except Exception:
            pass
    return ""


def _sse(obj: dict[str, Any]) -> str:
    import json

    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _rewrite_query_with_history(query: str, history: list[dict[str, str]]) -> str:
    """
    LLM query rewrite layer for contextual follow-ups.

    Returns a standalone retrieval query string (no extra commentary).
    """
    rewrite_model = os.getenv("OPENROUTER_REWRITE_MODEL", "qwen/qwen-turbo")
    max_history = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))

    # Keep only {role, content} and only the last N turns.
    cleaned: list[dict[str, str]] = []
    for m in history[-max_history:]:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})

    system = (
        "You rewrite a user's message into a standalone search query for a product catalog RAG system.\n"
        "Rules:\n"
        "- Output ONLY the rewritten query text (no quotes, no bullets, no explanations).\n"
        "- Preserve SKUs, barcodes, and exact product names.\n"
        "- Resolve pronouns and references using the conversation history.\n"
        "- Keep the user's language.\n"
        "- If the message is a broad category/browse request, append a few short synonym/alternate terms "
        "that the catalog might use for the same concept (3-6 terms max). Do not add unrelated categories.\n"
        "- If the user message is already standalone, return it unchanged.\n"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if cleaned:
        messages.extend(cleaned)
    messages.append({"role": "user", "content": query})

    rewritten = _openrouter_chat(messages, model=rewrite_model)
    rewritten = (rewritten or "").strip().strip('"').strip()
    return rewritten or query


def _source_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Return a best-effort metadata dict for a retrieval payload.

    - BM25 payloads are {id, text, meta}.
    - Qdrant payloads are usually already "flat" (chunk_id, physical_page_number, ...).
    - Some callers may already provide {chunk_id, text, metadata}.
    """
    if isinstance(payload.get("metadata"), dict):
        return dict(payload.get("metadata") or {})
    if isinstance(payload.get("meta"), dict):
        return dict(payload.get("meta") or {})
    # Assume flat payload.
    return dict(payload)


def _source_chunk_type(payload: dict[str, Any]) -> str | None:
    meta = _source_meta(payload)
    ct = meta.get("chunk_type") or payload.get("chunk_type")
    return str(ct) if ct else None


def _source_chunk_id(payload: dict[str, Any]) -> str | None:
    meta = _source_meta(payload)
    cid = meta.get("chunk_id") or payload.get("chunk_id") or payload.get("id")
    return str(cid) if cid else None


def _normalize_source(payload: dict[str, Any], *, score: float) -> dict[str, Any]:
    meta = _source_meta(payload)
    chunk_id = _source_chunk_id(payload) or ""
    text = payload.get("text") or ""
    # Keep qdrant ids / debug fields in metadata so the UI can show them if desired.
    for k in ("qdrant_point_id",):
        if k in payload and k not in meta:
            meta[k] = payload[k]
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": meta,
        "score": round(float(score), 6),
    }


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict[str, str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    client = Groq(api_key=api_key)
    transcription = client.audio.transcriptions.create(
        file=(file.filename, content),
        model="whisper-large-v3-turbo",
        response_format="text",
    )
    return {"text": str(transcription).strip()}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "langfuse": "enabled" if _langfuse_enabled() else "disabled"}


@app.get("/warmup")
def warmup() -> dict[str, Any]:
    """
    Best-effort warmup endpoint to initialize caches:
    - BM25 bundle load
    - embedding model load
    - Qdrant client init

    This helps cold-start latency on small servers.
    """
    bm25_ok = False
    embed_ok = False
    qdrant_ok = False
    try:
        _get_bm25_bundle()
        bm25_ok = True
    except Exception:
        bm25_ok = False

    try:
        _get_embedder()
        embed_ok = True
    except Exception:
        embed_ok = False

    try:
        _get_qdrant_client()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return {
        "status": "ok",
        "bm25_loaded": bm25_ok,
        "embedder_loaded": embed_ok,
        "qdrant_client_ready": qdrant_ok,
        "langfuse": "enabled" if _langfuse_enabled() else "disabled",
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    langfuse = _get_langfuse_client()
    span = None
    if langfuse is not None:
        try:
            span = langfuse.start_observation(as_type="span", name="api_chat")
        except Exception:
            span = None

    bm25_bundle = _get_bm25_bundle()
    bm25 = bm25_bundle["bm25"]
    bm25_chunks = bm25_bundle["chunks"]
    product_dictionary = bm25_bundle.get("product_dictionary", {}) or {}

    top_k = req.top_k or 8
    top_n = req.top_n or 4

    client = _get_qdrant_client()
    model = _get_embedder()

    history = req.history or []
    search_query = _rewrite_query_with_history(query, history)
    enriched_query = enrich_query_with_product_names(search_query, product_dictionary)

    if span is not None:
        try:
            span.update(
                input={
                    "query": query,
                    "rewritten_query": search_query,
                    "enriched_query": enriched_query,
                    "history_len": len(history),
                    "skus": _extract_skus_from_query(search_query),
                }
            )
        except Exception:
            pass

    vec = qdrant_search(
        client,
        os.getenv("QDRANT_COLLECTION", "catalog_es"),
        model,
        enriched_query,
        top_k=top_k,
    )
    kw = bm25_search(bm25, bm25_chunks, enriched_query, top_k=top_k)
    fused = reciprocal_rank_fusion(vec, kw, top_n=top_n)

    skus_in_query = _extract_skus_from_query(enriched_query)

    # If we didn't pull enough concrete product rows for a broad query, do a second pass
    # restricted to product SKU rows and merge results (generic improvement; not domain-specific).
    if not skus_in_query:
        product_rows = 0
        for it in fused:
            payload = dict(it.payload or {})
            if _source_chunk_type(payload) == "product_sku_row":
                product_rows += 1
        if product_rows < max(2, top_n // 2):
            try:
                from qdrant_client.http.models import FieldCondition, Filter, MatchValue

                flt = Filter(must=[FieldCondition(key="chunk_type", match=MatchValue(value="product_sku_row"))])
                vec2 = qdrant_search(
                    client,
                    os.getenv("QDRANT_COLLECTION", "catalog_es"),
                    model,
                    enriched_query,
                    top_k=top_k,
                    qdrant_filter=flt,
                )
            except Exception:
                vec2 = []

            kw2 = [it for it in bm25_search(bm25, bm25_chunks, enriched_query, top_k=top_k * 4) if _source_chunk_type(dict(it.payload or {})) == "product_sku_row"][:top_k]
            fused2 = reciprocal_rank_fusion(vec2, kw2, top_n=top_n * 2)

            seen: set[str] = set()
            merged: list[Any] = []
            for it in list(fused) + list(fused2):
                cid = _source_chunk_id(dict(it.payload or {})) or it.id
                if cid in seen:
                    continue
                seen.add(cid)
                merged.append(it)
                if len(merged) >= top_n:
                    break
            fused = merged

    retrieved_chunks = [_normalize_source(dict(item.payload or {}), score=float(item.score)) for item in fused]
    context_str = build_context_str(retrieved_chunks)
    system_prompt = build_system_prompt(context_str)

    llm_model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-plus")
    max_history = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
    cleaned_history: list[dict[str, str]] = []
    for m in history[-max_history:]:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned_history.append({"role": role, "content": content})

    answer_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    answer_messages.extend(cleaned_history)
    answer_messages.append({"role": "user", "content": query})
    answer = _openrouter_chat(answer_messages, model=llm_model)

    if span is not None:
        try:
            span.update(
                output={
                    "top_k": top_k,
                    "top_n": top_n,
                    "retrieved_count": len(retrieved_chunks),
                    "retrieved_pages": _extract_retrieved_pages(retrieved_chunks),
                    "retrieved_skus": sorted(
                        {str((c.get("metadata") or {}).get("sku")) for c in retrieved_chunks if (c.get("metadata") or {}).get("sku")}
                    ),
                }
            )
        except Exception:
            pass
        try:
            span.end()
        except Exception:
            pass

    if langfuse:
        try:
            langfuse.flush()
        except Exception:
            pass

    return ChatResponse(
        answer=answer,
        sources=retrieved_chunks,
        rewritten_query=search_query,
        enriched_query=enriched_query,
    )


@app.post("/api/chat_stream")
def chat_stream(req: ChatRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    def _gen():
        langfuse = _get_langfuse_client()
        span = None
        if langfuse is not None:
            try:
                span = langfuse.start_observation(as_type="span", name="api_chat_stream")
            except Exception:
                span = None
        try:
            bm25_bundle = _get_bm25_bundle()
            bm25 = bm25_bundle["bm25"]
            bm25_chunks = bm25_bundle["chunks"]
            product_dictionary = bm25_bundle.get("product_dictionary", {}) or {}

            top_k = req.top_k or 8
            top_n = req.top_n or 4

            history = req.history or []

            yield _sse({"type": "status", "message": "Rewriting your query…"})
            search_query = _rewrite_query_with_history(query, history)
            yield _sse({"type": "rewrite", "rewritten_query": search_query})

            enriched_query = enrich_query_with_product_names(search_query, product_dictionary)
            yield _sse({"type": "enrich", "enriched_query": enriched_query})

            if span is not None:
                try:
                    span.update(
                        input={
                            "query": query,
                            "rewritten_query": search_query,
                            "enriched_query": enriched_query,
                            "history_len": len(history),
                            "skus": _extract_skus_from_query(search_query),
                        }
                    )
                except Exception:
                    pass

            yield _sse({"type": "status", "message": "Searching source documents…"})
            client = _get_qdrant_client()
            embedder = _get_embedder()
            vec = qdrant_search(
                client,
                os.getenv("QDRANT_COLLECTION", "catalog_es"),
                embedder,
                enriched_query,
                top_k=top_k,
            )
            kw = bm25_search(bm25, bm25_chunks, enriched_query, top_k=top_k)

            yield _sse({"type": "status", "message": "Fusing results…"})
            fused = reciprocal_rank_fusion(vec, kw, top_n=top_n)

            skus_in_query = _extract_skus_from_query(enriched_query)

            if not skus_in_query:
                product_rows = 0
                for it in fused:
                    payload = dict(it.payload or {})
                    if _source_chunk_type(payload) == "product_sku_row":
                        product_rows += 1
                if product_rows < max(2, top_n // 2):
                    try:
                        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

                        flt = Filter(must=[FieldCondition(key="chunk_type", match=MatchValue(value="product_sku_row"))])
                        vec2 = qdrant_search(
                            client,
                            os.getenv("QDRANT_COLLECTION", "catalog_es"),
                            embedder,
                            enriched_query,
                            top_k=top_k,
                            qdrant_filter=flt,
                        )
                    except Exception:
                        vec2 = []

                    kw2 = [
                        it
                        for it in bm25_search(bm25, bm25_chunks, enriched_query, top_k=top_k * 4)
                        if _source_chunk_type(dict(it.payload or {})) == "product_sku_row"
                    ][:top_k]
                    fused2 = reciprocal_rank_fusion(vec2, kw2, top_n=top_n * 2)

                    seen: set[str] = set()
                    merged: list[Any] = []
                    for it in list(fused) + list(fused2):
                        cid = _source_chunk_id(dict(it.payload or {})) or it.id
                        if cid in seen:
                            continue
                        seen.add(cid)
                        merged.append(it)
                        if len(merged) >= top_n:
                            break
                    fused = merged

            retrieved_chunks = [_normalize_source(dict(item.payload or {}), score=float(item.score)) for item in fused]

            context_str = build_context_str(retrieved_chunks)
            system_prompt = build_system_prompt(context_str)

            yield _sse({"type": "status", "message": "Generating answer…"})
            llm_model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-plus")
            max_history = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
            cleaned_history: list[dict[str, str]] = []
            for m in history[-max_history:]:
                role = (m.get("role") or "").strip()
                content = (m.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    cleaned_history.append({"role": role, "content": content})

            messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
            messages.extend(cleaned_history)
            messages.append({"role": "user", "content": query})

            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                yield _sse({"type": "error", "message": "OPENROUTER_API_KEY not set"})
                return

            full = ""
            with OpenRouter(
                http_referer="ragbot-poc",
                x_open_router_title="ragbot-poc",
                api_key=api_key,
            ) as open_router:
                res = open_router.chat.send(
                    model=llm_model,
                    messages=messages,
                    stream=True,
                    temperature=0.0,
                )
                with res as event_stream:
                    for event in event_stream:
                        delta = _extract_openrouter_stream_delta(event)
                        if not delta:
                            continue
                        full += delta
                        yield _sse({"type": "token", "delta": delta})

            if span is not None:
                try:
                    span.update(
                        output={
                            "top_k": top_k,
                            "top_n": top_n,
                            "retrieved_count": len(retrieved_chunks),
                            "retrieved_pages": _extract_retrieved_pages(retrieved_chunks),
                            "retrieved_skus": sorted(
                                {str((c.get("metadata") or {}).get("sku")) for c in retrieved_chunks if (c.get("metadata") or {}).get("sku")}
                            ),
                        }
                    )
                except Exception:
                    pass
                try:
                    span.end()
                except Exception:
                    pass

            if langfuse:
                try:
                    langfuse.flush()
                except Exception:
                    pass

            yield _sse(
                {
                    "type": "done",
                    "answer": full,
                    "sources": retrieved_chunks,
                    "rewritten_query": search_query,
                    "enriched_query": enriched_query,
                }
            )
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(_gen(), media_type="text/event-stream")
