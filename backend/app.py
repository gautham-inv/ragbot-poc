from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ragbot")

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
from backend.tools import build_tool_system_prompt, render_compare_products_markdown, run_tool_loop, run_tool_loop_stream
from backend.db import (
    ensure_conversation_id,
    get_conversation_messages,
    init_chat_schema,
    insert_message,
    list_conversations,
)

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
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[dict[str, Any]]
    sources_total: int | None = None
    rewritten_query: str | None = None
    enriched_query: str | None = None
    # Compact product cards derived from retrieved SKUs (deduped, with image URLs
    # attached from data/sku_image_map.json). Empty list if no products match.
    products: list[dict[str, Any]] = []


class ConversationSummary(BaseModel):
    id: str
    created_at: str | None = None
    updated_at: str | None = None
    first_user_message: str | None = None
    message_count: int = 0


class ConversationMessage(BaseModel):
    id: int
    role: str
    content: str
    metadata: dict[str, Any] | None = None
    created_at: str | None = None


class ConversationDetail(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    _log_langfuse_status_once()
    
    # Best-effort: enabled only when CHAT_DATABASE_URL is set.
    try:
        init_chat_schema()
    except Exception as e:
        logger.warning("[chat_db] init failed: %s: %s", type(e).__name__, e)
    
    yield


app = FastAPI(title="Gloriapets RAG API", lifespan=lifespan)


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

    def _provider_pref(*, env_key: str) -> dict[str, Any] | None:
        # Purpose-specific keys can explicitly disable provider forcing by being present but empty, e.g.
        # OPENROUTER_REWRITE_PROVIDER_ONLY=
        if env_key in os.environ:
            raw = (os.getenv(env_key) or "").strip()
            if not raw:
                return None
            only = [p.strip() for p in raw.split(",") if p.strip()]
            return {"only": only} if only else None

        raw = (os.getenv("OPENROUTER_PROVIDER_ONLY") or "").strip()
        if not raw:
            return None
        only = [p.strip() for p in raw.split(",") if p.strip()]
        return {"only": only} if only else None

    provider = _provider_pref(env_key="OPENROUTER_ANSWER_PROVIDER_ONLY")

    with OpenRouter(
        http_referer="ragbot-poc",
        x_open_router_title="ragbot-poc",
        api_key=api_key,
    ) as open_router:
        send_kwargs: dict[str, Any] = {
            "model": os.getenv("OPENROUTER_MODEL", "qwen/qwen-plus"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "temperature": 0.0,
        }
        if provider is not None:
            send_kwargs["provider"] = provider
        res = open_router.chat.send(**send_kwargs)

    if hasattr(res, "choices") and res.choices:
        return res.choices[0].message.content
    return str(res)


def _openrouter_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    provider_env_key: str = "OPENROUTER_PROVIDER_ONLY",
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    def _provider_pref(*, env_key: str) -> dict[str, Any] | None:
        if env_key in os.environ:
            raw = (os.getenv(env_key) or "").strip()
            if not raw:
                return None
            only = [p.strip() for p in raw.split(",") if p.strip()]
            return {"only": only} if only else None

        raw = (os.getenv("OPENROUTER_PROVIDER_ONLY") or "").strip()
        if not raw:
            return None
        only = [p.strip() for p in raw.split(",") if p.strip()]
        return {"only": only} if only else None

    provider = _provider_pref(env_key=provider_env_key)

    with OpenRouter(
        http_referer="ragbot-poc",
        x_open_router_title="ragbot-poc",
        api_key=api_key,
    ) as open_router:
        send_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.0,
        }
        if provider is not None:
            send_kwargs["provider"] = provider
        res = open_router.chat.send(**send_kwargs)

    if hasattr(res, "choices") and res.choices:
        return res.choices[0].message.content
    return str(res)


_langfuse_init_logged = False


def _get_langfuse_client():
    global _langfuse_init_logged
    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    if not (pub and sec):
        if not _langfuse_init_logged:
            logger.warning(
                "[langfuse] keys missing: LANGFUSE_PUBLIC_KEY=%s LANGFUSE_SECRET_KEY=%s — no traces will be sent",
                bool(pub), bool(sec),
            )
            _langfuse_init_logged = True
        return None
    # Some Langfuse SDK versions read LANGFUSE_HOST; keep compatibility with LANGFUSE_BASE_URL.
    if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL", "")
    try:
        from langfuse import get_client  # type: ignore

        client = get_client()
        if not _langfuse_init_logged:
            host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "<default>"
            logger.info(
                "[langfuse] client initialized: type=%s host=%s has_start_observation=%s",
                type(client).__name__, host, hasattr(client, "start_observation"),
            )
            _langfuse_init_logged = True
        return client
    except ImportError as e:
        if not _langfuse_init_logged:
            logger.error("[langfuse] SDK not installed: %s", e)
            _langfuse_init_logged = True
        return None
    except Exception as e:
        if not _langfuse_init_logged:
            logger.error("[langfuse] get_client() failed: %s: %s", type(e).__name__, e, exc_info=True)
            _langfuse_init_logged = True
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


def _langfuse_add_trace_tags(langfuse: Any, *, trace_id: str | None, tags: list[str]) -> None:
    if not langfuse or not trace_id or not tags:
        return

    # Langfuse tags are used for trace-level filtering; keep them small, unique, and stable.
    deduped: list[str] = []
    for t in tags:
        t = (t or "").strip()
        if not t or t in deduped:
            continue
        deduped.append(t)
        if len(deduped) >= 50:
            break

    if not deduped:
        return

    # Preferred (SDK >=4): update trace tags via ingestion API.
    try:
        fn = getattr(langfuse, "_create_trace_tags_via_ingestion", None)
        if callable(fn):
            fn(trace_id=trace_id, tags=deduped)
            return
    except Exception:
        pass

    # Fallback: best-effort propagation (no-op if tracing context isn't active).
    try:
        from langfuse import propagate_attributes  # type: ignore

        with propagate_attributes(tags=deduped):
            pass
    except Exception:
        pass




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


def _chunk_meta(c: dict[str, Any]) -> dict[str, Any]:
    meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else None
    return meta if meta else c


def _extract_retrieved_brands(retrieved_chunks: list[dict[str, Any]]) -> dict[str, int]:
    """Per-trace counts of brands across the retrieved chunks, e.g. {'KONG': 3, 'HUNTER': 1}."""
    counts: dict[str, int] = {}
    for c in retrieved_chunks:
        meta = _chunk_meta(c)
        brand = meta.get("brand") or c.get("brand")
        if not brand:
            continue
        key = str(brand).strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _extract_retrieved_categories(
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, int]:
    """Per-trace counts of categories across the retrieved chunks, e.g. {'accessories': 2, 'toys': 1}."""
    counts: dict[str, int] = {}
    for c in retrieved_chunks:
        meta = _chunk_meta(c)
        cat = meta.get("category") or c.get("category")
        if not cat:
            continue
        key = str(cat).strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _extract_retrieved_subcategories(
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, int]:
    """Per-trace counts of category/subcategory pairs, e.g. {'accessories/collar': 2, 'toys/ball': 1}."""
    counts: dict[str, int] = {}
    for c in retrieved_chunks:
        meta = _chunk_meta(c)
        cat = meta.get("category") or c.get("category")
        sub = meta.get("subcategory") or c.get("subcategory")
        if not cat and not sub:
            continue
        cat_s = str(cat).strip() if cat else ""
        sub_s = str(sub).strip() if sub else ""
        if cat_s and sub_s:
            key = f"{cat_s}/{sub_s}"
        elif cat_s:
            key = cat_s
        else:
            key = sub_s
        counts[key] = counts.get(key, 0) + 1
    return counts


def _sku_counts_in_text(text: str, skus: list[str]) -> dict[str, int]:
    import re

    hay = (text or "").upper()
    out: dict[str, int] = {}
    for sku in skus:
        s = (sku or "").strip().upper()
        if not s:
            continue
        # Count exact substring occurrences (SKU tokens can contain punctuation so word-boundaries are unreliable).
        out[s] = len(re.findall(re.escape(s), hay))
    return out


def _sku_product_names_from_chunks(retrieved_chunks: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in retrieved_chunks or []:
        meta = (c or {}).get("metadata") or {}
        sku_raw = meta.get("sku")
        sku = str(sku_raw).strip() if sku_raw else ""
        if not sku:
            continue

        # Prefer a stable, human-friendly name if available.
        name_raw = meta.get("product_name") or meta.get("product_name_es") or meta.get("name") or meta.get("title")
        name = str(name_raw).strip() if name_raw else ""
        if not name:
            continue

        out.setdefault(sku, name)
    return out


def _sanitize_sku_for_score_name(sku: str) -> str:
    import re

    s = (sku or "").strip().upper()
    if not s:
        return "UNKNOWN"

    # Score names should be stable and safe across Langfuse backends/exports.
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "UNKNOWN"
    return s[:80]


def _langfuse_score_sku_counts(span: Any, *, query: str, answer: str, skus: list[str]) -> None:
    if span is None or not skus:
        return

    score_fn = getattr(span, "score_trace", None)
    if not callable(score_fn):
        return

    query_counts = _sku_counts_in_text(query, skus)
    answer_counts = _sku_counts_in_text(answer, skus)

    total_query = 0
    total_answer = 0
    for sku in skus:
        sku_norm = (sku or "").strip().upper()
        if not sku_norm:
            continue
        safe = _sanitize_sku_for_score_name(sku_norm)
        qn = int(query_counts.get(sku_norm, 0))
        an = int(answer_counts.get(sku_norm, 0))
        total_query += qn
        total_answer += an
        try:
            score_fn(
                name=f"sku_query_count__{safe}",
                value=float(qn),
                data_type="NUMERIC",
                metadata={"sku": sku_norm},
            )
        except Exception:
            pass
        try:
            score_fn(
                name=f"sku_answer_count__{safe}",
                value=float(an),
                data_type="NUMERIC",
                metadata={"sku": sku_norm},
            )
        except Exception:
            pass

    try:
        score_fn(name="sku_query_count__TOTAL", value=float(total_query), data_type="NUMERIC")
    except Exception:
        pass
    try:
        score_fn(name="sku_answer_count__TOTAL", value=float(total_answer), data_type="NUMERIC")
    except Exception:
        pass


def _langfuse_score_numeric(span: Any, name: str, value: float) -> None:
    if span is None:
        return
    score_fn = getattr(span, "score_trace", None)
    if not callable(score_fn):
        return
    try:
        score_fn(name=name, value=float(value), data_type="NUMERIC")
    except Exception:
        pass


_DEFAULT_ALLOWED_INTENTS = [
    "barcode_lookup",
    "product_search",
    "product_recommendation",
    "price_compare",
    "basket_build",
    "order_status",
    "general_qa",
]


def _get_allowed_intents() -> list[str]:
    import json

    raw = (os.getenv("ALLOWED_INTENTS") or "").strip()
    if not raw:
        return list(_DEFAULT_ALLOWED_INTENTS)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            out: list[str] = []
            for v in data:
                if isinstance(v, str) and v.strip() and v.strip() not in out:
                    out.append(v.strip())
            return out or list(_DEFAULT_ALLOWED_INTENTS)
    except Exception:
        pass
    return list(_DEFAULT_ALLOWED_INTENTS)


def _route_intent_and_language(query: str) -> tuple[str, str, float]:
    """
    Returns (intent, language_code, confidence).

    Best-effort: if routing fails, returns ("unknown", "unknown", 0.0).
    """
    import json

    allowed_intents = _get_allowed_intents()
    model = os.getenv(
        "OPENROUTER_INTENT_MODEL",
        os.getenv(
            "OPENROUTER_REWRITE_MODEL",
            os.getenv("OPENROUTER_MODEL", "qwen/qwen-turbo"),
        ),
    )

    system = (
        "You are a classifier for a product-catalog assistant.\n"
        "Return ONLY valid JSON with keys: intent, language, confidence.\n"
        f"Allowed intents (choose exactly one): {json.dumps(allowed_intents, ensure_ascii=False)}\n"
        "language: 2-letter ISO code if possible (e.g. en, es, hi). Use 'unknown' if unsure.\n"
        "confidence: number from 0.0 to 1.0.\n"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": (query or "").strip()},
    ]

    raw = _openrouter_chat(
        messages,
        model=model,
        provider_env_key="OPENROUTER_INTENT_PROVIDER_ONLY",
    )
    raw = (raw or "").strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("intent router output not a dict")
        intent = str(data.get("intent") or "").strip()
        language = str(data.get("language") or "").strip().lower()
        confidence = float(data.get("confidence") or 0.0)
        if intent not in allowed_intents:
            intent = "unknown"
        if not language or len(language) > 8:
            language = "unknown"
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0
        return intent, language, confidence
    except Exception:
        return "unknown", "unknown", 0.0


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
    rewrite_model = os.getenv(
        "OPENROUTER_REWRITE_MODEL",
        os.getenv("OPENROUTER_MODEL", "qwen/qwen-turbo"),
    )
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

    rewritten = _openrouter_chat(
        messages,
        model=rewrite_model,
        provider_env_key="OPENROUTER_REWRITE_PROVIDER_ONLY",
    )
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


def _products_from_sources(
    sources: list[dict[str, Any]],
    *,
    max_products: int = 12,
) -> list[dict[str, Any]]:
    """
    Build deduped product cards from a `sources` list (shape produced by either
    `/api/chat` or `/api/chat_tools`). Attaches Cloudinary image URLs from the
    SKU → image map (written by `ingestion/upload_product_images.py`), drops any
    entry without a SKU, and preserves first-appearance order.
    """
    from backend.image_map import get_images  # local import keeps startup light

    out: list[dict[str, Any]] = []
    seen_skus: set[str] = set()

    for src in sources:
        if not isinstance(src, dict):
            continue
        meta = src.get("metadata") or {}
        if not isinstance(meta, dict):
            continue

        sku_raw = meta.get("sku")
        if not sku_raw:
            continue
        sku = str(sku_raw).strip()
        if not sku or sku in seen_skus:
            continue

        # metadata may already contain image fields (set by tools._product_summary
        # after batch 1). Fall back to the JSON map otherwise.
        primary = meta.get("primary_image") or ""
        thumbnail = meta.get("thumbnail") or ""
        images = list(meta.get("images") or [])
        thumbnails = list(meta.get("thumbnails") or [])
        if not primary and not images:
            rec = get_images(sku)
            if rec:
                primary = rec.get("primary_image") or ""
                thumbnail = rec.get("thumbnail") or ""
                images = list(rec.get("images") or [])
                thumbnails = list(rec.get("thumbnails") or [])

        # Product name prefers the product-summary fields, falls back to payload keys.
        name = (
            meta.get("name_es")
            or meta.get("name")
            or meta.get("product_name")
            or (meta.get("names") or {}).get("es")
            or ""
        )

        card = {
            "sku": sku,
            "brand": meta.get("brand") or "",
            "name": name,
            "category": meta.get("category") or "",
            "subcategory": meta.get("subcategory") or "",
            "price_pvpr": meta.get("price_pvpr") or meta.get("price_eur"),
            "price_per_unit": meta.get("price_per_unit"),
            "min_purchase_qty": meta.get("min_purchase_qty") or meta.get("min_order"),
            "primary_image": primary,
            "thumbnail": thumbnail,
            "images": images,
            "thumbnails": thumbnails,
            "catalog_pages": meta.get("catalog_pages"),
            "primary_page": meta.get("primary_page"),
        }
        # Only include cards that actually have at least one image — otherwise
        # the frontend has nothing useful to show.
        if not (card["primary_image"] or card["images"]):
            continue

        seen_skus.add(sku)
        out.append(card)
        if len(out) >= max_products:
            break

    return out


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


@app.get("/api/conversations", response_model=list[ConversationSummary])
def api_list_conversations(limit: int = 50) -> list[ConversationSummary]:
    try:
        rows = list_conversations(limit=limit)
        return [ConversationSummary(**row) for row in rows]
    except Exception as e:
        logger.exception("failed to list conversations: %s", e)
        raise HTTPException(status_code=500, detail="failed to list conversations")


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def api_get_conversation(conversation_id: str) -> ConversationDetail:
    try:
        uuid.UUID(conversation_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid conversation_id")

    try:
        rows = get_conversation_messages(conversation_id)
    except Exception as e:
        logger.exception("failed to load conversation %s: %s", conversation_id, e)
        raise HTTPException(status_code=500, detail="failed to load conversation")

    if not rows:
        raise HTTPException(status_code=404, detail="conversation not found")

    return ConversationDetail(
        conversation_id=conversation_id,
        messages=[ConversationMessage(**row) for row in rows],
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    conversation_id = ensure_conversation_id(req.conversation_id)
    try:
        insert_message(
            conversation_id=conversation_id,
            role="user",
            content=query,
            metadata={"endpoint": "/api/chat"},
        )
    except Exception:
        pass

    rid = uuid.uuid4().hex[:8]
    t_req = time.perf_counter()
    logger.info("[%s] /api/chat query=%r history_len=%d", rid, query, len(req.history or []))

    langfuse = _get_langfuse_client()
    span = None
    if langfuse is not None:
        try:
            span = langfuse.start_observation(as_type="span", name="api_chat")
        except Exception:
            span = None

    intent = "unknown"
    language = "unknown"
    intent_confidence = 0.0
    retrieval_latency_ms = 0.0
    retrieval_success = False
    fallback_triggered = False
    fallback_reason: str | None = None
    no_result = False
    basics_scored = False

    top_k = req.top_k or 8
    top_n = req.top_n or 4
    history = req.history or []
    search_query = ""
    enriched_query = ""
    retrieved_chunks: list[dict[str, Any]] = []
    retrieved_skus: list[str] = []
    answer = ""

    try:
        try:
            intent, language, intent_confidence = _route_intent_and_language(query)
        except Exception:
            intent, language, intent_confidence = "unknown", "unknown", 0.0
        logger.info("[%s] intent=%s lang=%s conf=%.2f", rid, intent, language, intent_confidence)

        bm25_bundle = _get_bm25_bundle()
        bm25 = bm25_bundle["bm25"]
        bm25_chunks = bm25_bundle["chunks"]
        product_dictionary = bm25_bundle.get("product_dictionary", {}) or {}

        client = _get_qdrant_client()
        model = _get_embedder()

        search_query = _rewrite_query_with_history(query, history)
        logger.info("[%s] rewritten=%r", rid, search_query)
        enriched_query = enrich_query_with_product_names(search_query, product_dictionary)
        logger.info("[%s] enriched=%r", rid, enriched_query)

        if span is not None:
            try:
                skus_in_user_query = _extract_skus_from_query(query)
                span.update(
                    input={
                        "query": query,
                        "intent": intent,
                        "intent_confidence": intent_confidence,
                        "user_language": language,
                        "rewritten_query": search_query,
                        "enriched_query": enriched_query,
                        "history_len": len(history),
                        "skus": _extract_skus_from_query(search_query),
                        "skus_in_user_query": skus_in_user_query,
                        "sku_counts_in_user_query": _sku_counts_in_text(query, skus_in_user_query),
                    }
                )
            except Exception:
                pass

        t0 = time.perf_counter()
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
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_skus = sorted(
            {str((c.get("metadata") or {}).get("sku")) for c in retrieved_chunks if (c.get("metadata") or {}).get("sku")}
        )
        sku_product_names = _sku_product_names_from_chunks(retrieved_chunks)
        retrieval_success = len(retrieved_chunks) > 0
        logger.info(
            "[%s] retrieval top_k=%d top_n=%d latency_ms=%.1f hits=%d skus=%s",
            rid, top_k, top_n, retrieval_latency_ms, len(retrieved_chunks), retrieved_skus,
        )

        if not retrieval_success:
            fallback_triggered = True
            fallback_reason = "no_results"
            no_result = True
            logger.info("[%s] fallback reason=%s", rid, fallback_reason)

        context_str = build_context_str(retrieved_chunks)
        system_prompt = build_system_prompt(context_str, user_language=language)

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
        logger.info("[%s] llm call model=%s messages=%d", rid, llm_model, len(answer_messages))
        t_llm = time.perf_counter()
        answer = _openrouter_chat(
            answer_messages,
            model=llm_model,
            provider_env_key="OPENROUTER_ANSWER_PROVIDER_ONLY",
        )
        logger.info(
            "[%s] llm done latency_ms=%.1f answer_len=%d total_ms=%.1f",
            rid, (time.perf_counter() - t_llm) * 1000.0, len(answer or ""),
            (time.perf_counter() - t_req) * 1000.0,
        )

        if span is not None:
            try:
                span.update(
                    output={
                        "top_k": top_k,
                        "top_n": top_n,
                        "retrieval_latency_ms": round(float(retrieval_latency_ms), 3),
                        "retrieval_success": bool(retrieval_success),
                        "retrieved_count": len(retrieved_chunks),
                        "retrieved_pages": _extract_retrieved_pages(retrieved_chunks),
                        "retrieved_brands": _extract_retrieved_brands(retrieved_chunks),
                        "retrieved_categories": _extract_retrieved_categories(retrieved_chunks),
                        "retrieved_subcategories": _extract_retrieved_subcategories(retrieved_chunks),
                        "retrieved_skus": retrieved_skus,
                        "sku_product_names": sku_product_names,
                        "answer": (answer or "")[:20000],
                        "answer_truncated": len((answer or "")) > 20000,
                        "sku_counts_in_answer": _sku_counts_in_text(answer, retrieved_skus),
                        "fallback_triggered": bool(fallback_triggered),
                        "fallback_reason": fallback_reason,
                        "no_result": bool(no_result),
                    }
                )
            except Exception:
                pass
            try:
                _langfuse_score_sku_counts(span, query=query, answer=answer, skus=retrieved_skus)
            except Exception:
                pass
            _langfuse_score_numeric(span, "retrieval_count", float(len(retrieved_chunks)))
            _langfuse_score_numeric(span, "retrieval_latency_ms", float(retrieval_latency_ms))
            _langfuse_score_numeric(span, "fallback_triggered", 1.0 if fallback_triggered else 0.0)
            _langfuse_score_numeric(span, "no_result", 1.0 if no_result else 0.0)
            basics_scored = True
    except Exception as e:
        fallback_triggered = True
        fallback_reason = "system_error"
        no_result = True
        retrieval_success = False
        logger.exception("[%s] /api/chat failed: %s", rid, e)
        raise
    finally:
        if span is not None:
            try:
                if not basics_scored:
                    _langfuse_score_numeric(span, "fallback_triggered", 1.0 if fallback_triggered else 0.0)
                    _langfuse_score_numeric(span, "no_result", 1.0 if no_result else 0.0)
                    basics_scored = True
                tags = [
                    f"intent:{intent or 'unknown'}",
                    f"lang:{language or 'unknown'}",
                    f"fallback:{'true' if fallback_triggered else 'false'}",
                    f"no_result:{'true' if no_result else 'false'}",
                    f"retrieval_success:{'true' if retrieval_success else 'false'}",
                ]
                if fallback_triggered and fallback_reason:
                    tags.append(f"fallback_reason:{fallback_reason}")
                _langfuse_add_trace_tags(langfuse, trace_id=getattr(span, "trace_id", None), tags=tags)
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

    products = _products_from_sources(retrieved_chunks)
    try:
        insert_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            metadata={
                "endpoint": "/api/chat",
                "sources_count": len(retrieved_chunks),
                "products_count": len(products or []),
                "products": products or [],
                "rewritten_query": search_query,
                "enriched_query": enriched_query,
            },
        )
    except Exception as e:
        logger.warning("[chat_db] insert assistant message failed (/api/chat): %s: %s", type(e).__name__, e)
    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        sources=retrieved_chunks,
        rewritten_query=search_query,
        enriched_query=enriched_query,
        products=products,
    )


@app.post("/api/chat_stream")
def chat_stream(req: ChatRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    conversation_id = ensure_conversation_id(req.conversation_id)
    try:
        insert_message(
            conversation_id=conversation_id,
            role="user",
            content=query,
            metadata={"endpoint": "/api/chat_stream"},
        )
    except Exception as e:
        logger.warning("[chat_db] insert user message failed (/api/chat_stream): %s: %s", type(e).__name__, e)

    rid = uuid.uuid4().hex[:8]
    t_req = time.perf_counter()
    logger.info("[%s] /api/chat_stream query=%r history_len=%d", rid, query, len(req.history or []))

    def _gen():
        langfuse = _get_langfuse_client()
        span = None
        if langfuse is not None:
            try:
                span = langfuse.start_observation(as_type="span", name="api_chat_stream")
            except Exception:
                span = None

        intent = "unknown"
        language = "unknown"
        intent_confidence = 0.0
        retrieval_latency_ms = 0.0
        retrieval_success = False
        fallback_triggered = False
        fallback_reason: str | None = None
        no_result = False
        try:
            bm25_bundle = _get_bm25_bundle()
            bm25 = bm25_bundle["bm25"]
            bm25_chunks = bm25_bundle["chunks"]
            product_dictionary = bm25_bundle.get("product_dictionary", {}) or {}

            top_k = req.top_k or 8
            top_n = req.top_n or 4

            history = req.history or []

            try:
                intent, language, intent_confidence = _route_intent_and_language(query)
            except Exception:
                intent, language, intent_confidence = "unknown", "unknown", 0.0
            logger.info("[%s] intent=%s lang=%s conf=%.2f", rid, intent, language, intent_confidence)

            yield _sse({"type": "status", "message": "Rewriting your query..."})
            search_query = _rewrite_query_with_history(query, history)
            logger.info("[%s] rewritten=%r", rid, search_query)
            yield _sse({"type": "rewrite", "rewritten_query": search_query})

            enriched_query = enrich_query_with_product_names(search_query, product_dictionary)
            logger.info("[%s] enriched=%r", rid, enriched_query)
            yield _sse({"type": "enrich", "enriched_query": enriched_query})

            if span is not None:
                try:
                    skus_in_user_query = _extract_skus_from_query(query)
                    span.update(
                        input={
                            "query": query,
                            "intent": intent,
                            "intent_confidence": intent_confidence,
                            "user_language": language,
                            "rewritten_query": search_query,
                            "enriched_query": enriched_query,
                            "history_len": len(history),
                            "skus": _extract_skus_from_query(search_query),
                            "skus_in_user_query": skus_in_user_query,
                            "sku_counts_in_user_query": _sku_counts_in_text(query, skus_in_user_query),
                        }
                    )
                except Exception:
                    pass

            yield _sse({"type": "status", "message": "Searching source documents..."})
            client = _get_qdrant_client()
            embedder = _get_embedder()
            t0 = time.perf_counter()
            vec = qdrant_search(
                client,
                os.getenv("QDRANT_COLLECTION", "catalog_es"),
                embedder,
                enriched_query,
                top_k=top_k,
            )
            kw = bm25_search(bm25, bm25_chunks, enriched_query, top_k=top_k)

            yield _sse({"type": "status", "message": "Fusing results..."})
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
            retrieval_latency_ms = (time.perf_counter() - t0) * 1000.0
            retrieval_success = len(retrieved_chunks) > 0
            if not retrieval_success:
                fallback_triggered = True
                fallback_reason = "no_results"
                no_result = True
            retrieved_skus = sorted({str((c.get("metadata") or {}).get("sku")) for c in retrieved_chunks if (c.get("metadata") or {}).get("sku")})
            sku_product_names = _sku_product_names_from_chunks(retrieved_chunks)
            logger.info(
                "[%s] retrieval top_k=%d top_n=%d latency_ms=%.1f hits=%d skus=%s",
                rid, top_k, top_n, retrieval_latency_ms, len(retrieved_chunks), retrieved_skus,
            )
            if not retrieval_success:
                logger.info("[%s] fallback reason=%s", rid, fallback_reason)

            context_str = build_context_str(retrieved_chunks)
            system_prompt = build_system_prompt(context_str, user_language=language)

            yield _sse({"type": "status", "message": "Generating answer..."})
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
            provider = None
            if "OPENROUTER_ANSWER_PROVIDER_ONLY" in os.environ:
                raw = (os.getenv("OPENROUTER_ANSWER_PROVIDER_ONLY") or "").strip()
                if raw:
                    only = [p.strip() for p in raw.split(",") if p.strip()]
                    provider = {"only": only} if only else None
            else:
                raw = (os.getenv("OPENROUTER_PROVIDER_ONLY") or "").strip()
                if raw:
                    only = [p.strip() for p in raw.split(",") if p.strip()]
                    provider = {"only": only} if only else None
            logger.info("[%s] llm stream model=%s messages=%d", rid, llm_model, len(messages))
            t_llm = time.perf_counter()
            with OpenRouter(
                http_referer="ragbot-poc",
                x_open_router_title="ragbot-poc",
                api_key=api_key,
            ) as open_router:
                send_kwargs: dict[str, Any] = {
                    "model": llm_model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.0,
                }
                if provider is not None:
                    send_kwargs["provider"] = provider
                res = open_router.chat.send(**send_kwargs)
                with res as event_stream:
                    for event in event_stream:
                        delta = _extract_openrouter_stream_delta(event)
                        if not delta:
                            continue
                        full += delta
                        yield _sse({"type": "token", "delta": delta})
            logger.info(
                "[%s] llm stream done latency_ms=%.1f answer_len=%d total_ms=%.1f",
                rid, (time.perf_counter() - t_llm) * 1000.0, len(full),
                (time.perf_counter() - t_req) * 1000.0,
            )

            if span is not None:
                try:
                        span.update(
                            output={
                                "top_k": top_k,
                                "top_n": top_n,
                                "retrieval_latency_ms": round(float(retrieval_latency_ms), 3),
                                "retrieval_success": bool(retrieval_success),
                                "retrieved_count": len(retrieved_chunks),
                                "retrieved_pages": _extract_retrieved_pages(retrieved_chunks),
                                "retrieved_brands": _extract_retrieved_brands(retrieved_chunks),
                                "retrieved_categories": _extract_retrieved_categories(retrieved_chunks),
                                "retrieved_subcategories": _extract_retrieved_subcategories(retrieved_chunks),
                                "retrieved_skus": retrieved_skus,
                                "sku_product_names": sku_product_names,
                                "fallback_triggered": bool(fallback_triggered),
                                "fallback_reason": fallback_reason,
                                "no_result": bool(no_result),
                            }
                        )
                except Exception:
                    pass
                try:
                    span.update(
                        output={
                            "answer": full[:20000],
                            "answer_truncated": len(full) > 20000,
                            "sku_counts_in_answer": _sku_counts_in_text(full, retrieved_skus),
                            "sku_product_names": sku_product_names,
                        }
                    )
                except Exception:
                    pass
                    try:
                        _langfuse_score_sku_counts(span, query=query, answer=full, skus=retrieved_skus)
                    except Exception:
                        pass
                    try:
                        _langfuse_score_numeric(span, "retrieval_count", float(len(retrieved_chunks)))
                        _langfuse_score_numeric(span, "retrieval_latency_ms", float(retrieval_latency_ms))
                        _langfuse_score_numeric(span, "fallback_triggered", 1.0 if fallback_triggered else 0.0)
                        _langfuse_score_numeric(span, "no_result", 1.0 if no_result else 0.0)
                    except Exception:
                        pass
                    try:
                        tags = [
                            f"intent:{intent or 'unknown'}",
                            f"lang:{language or 'unknown'}",
                            f"fallback:{'true' if fallback_triggered else 'false'}",
                            f"no_result:{'true' if no_result else 'false'}",
                            f"retrieval_success:{'true' if retrieval_success else 'false'}",
                        ]
                        if fallback_triggered and fallback_reason:
                            tags.append(f"fallback_reason:{fallback_reason}")
                        _langfuse_add_trace_tags(langfuse, trace_id=getattr(span, "trace_id", None), tags=tags)
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

            products = _products_from_sources(retrieved_chunks)
            if products:
                yield _sse({"type": "products", "items": products})
            try:
                insert_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full,
                    metadata={
                        "endpoint": "/api/chat_stream",
                        "sources_count": len(retrieved_chunks),
                        "products_count": len(products or []),
                "products": products or [],
                        "rewritten_query": search_query,
                        "enriched_query": enriched_query,
                    },
                )
            except Exception:
                pass
            yield _sse(
                {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "answer": full,
                    "sources": retrieved_chunks,
                    "rewritten_query": search_query,
                    "enriched_query": enriched_query,
                    "products": products,
                }
            )
        except Exception as e:
            fallback_triggered = True
            fallback_reason = "system_error"
            no_result = True
            retrieval_success = False
            logger.exception("[%s] /api/chat_stream failed: %s", rid, e)
            if span is not None:
                try:
                    _langfuse_score_numeric(span, "fallback_triggered", 1.0)
                    _langfuse_score_numeric(span, "no_result", 1.0)
                except Exception:
                    pass
                try:
                    tags = [
                        f"intent:{intent or 'unknown'}",
                        f"lang:{language or 'unknown'}",
                        "fallback:true",
                        "no_result:true",
                        "retrieval_success:false",
                        "fallback_reason:system_error",
                    ]
                    _langfuse_add_trace_tags(langfuse, trace_id=getattr(span, "trace_id", None), tags=tags)
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
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Tool-calling chat endpoint
# ---------------------------------------------------------------------------
@app.post("/api/chat_tools", response_model=ChatResponse)
def chat_tools(req: ChatRequest) -> ChatResponse:
    """
    LLM-with-tools chat. The model is given the catalog schema + 5 tools
    (semantic_search, filter_scroll, count_products, get_product,
    list_distinct_values) and decides which to call. Handles aggregation and
    enumeration questions that pure RAG cannot answer.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    conversation_id = ensure_conversation_id(req.conversation_id)
    try:
        insert_message(
            conversation_id=conversation_id,
            role="user",
            content=query,
            metadata={"endpoint": "/api/chat_tools"},
        )
    except Exception as e:
        logger.warning("[chat_db] insert user message failed (/api/chat_tools): %s: %s", type(e).__name__, e)

    langfuse = _get_langfuse_client()
    span = None
    if langfuse is not None:
        try:
            span = langfuse.start_observation(as_type="span", name="api_chat_tools")
        except Exception:
            span = None

    intent = "unknown"
    language = "unknown"
    intent_confidence = 0.0
    tool_trace: list[dict[str, Any]] = []
    retrieved_products: list[dict[str, Any]] = []
    answer = ""
    sources_total: int | None = None
    compare_result: dict[str, Any] | None = None
    error: str | None = None

    try:
        try:
            intent, language, intent_confidence = _route_intent_and_language(query)
        except Exception:
            pass

        bm25_bundle = _get_bm25_bundle()
        client = _get_qdrant_client()
        embedder = _get_embedder()
        collection = os.getenv("QDRANT_COLLECTION", "catalog_es")

        if span is not None:
            try:
                span.update(
                    input={
                        "query": query,
                        "intent": intent,
                        "intent_confidence": intent_confidence,
                        "user_language": language,
                        "history_len": len(req.history or []),
                        "skus_in_user_query": _extract_skus_from_query(query),
                    }
                )
            except Exception:
                pass

        result = run_tool_loop(
            user_query=query,
            history=req.history,
            system_prompt=build_tool_system_prompt(language),
            qdrant=client,
            collection=collection,
            embedder=embedder,
            bm25=bm25_bundle["bm25"],
            bm25_chunks=bm25_bundle["chunks"],
        )
        answer = result["answer"]
        tool_trace = result["tool_trace"]
        retrieved_products = result["retrieved_products"]
        sources_total = result.get("sources_total")
        cr = result.get("compare_result")
        compare_result = cr if isinstance(cr, dict) else None
        if not isinstance(sources_total, int) or sources_total < 0:
            sources_total = None

        if compare_result:
            table_md = render_compare_products_markdown(compare_result, user_language=language)
            if table_md:
                answer = f"{table_md}\n\n{answer}".strip()

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        answer = "No tengo esa información en el catálogo actual."
        if (language or "").strip().lower() == "en":
            answer = "I don't have that information in the current catalog."

    # Shape sources to match ChatResponse / frontend expectations.
    sources: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for p in retrieved_products:
        cid = str(p.get("id") or p.get("chunk_id") or "")
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)
        sources.append({
            "chunk_id": cid,
            "text": p.get("text") or "",
            "metadata": p,
            "score": 1.0,
        })

    # Langfuse output + scoring
    if span is not None:
        try:
            span.update(
                output={
                    "answer": (answer or "")[:20000],
                    "answer_truncated": len((answer or "")) > 20000,
                    "model": (result if not error else {}).get("model") if not error else None,
                    "rounds": (result if not error else {}).get("rounds") if not error else 0,
                    "tool_trace": tool_trace,
                    "retrieved_count": len(sources),
                    "retrieved_brands": _extract_retrieved_brands(sources),
                    "retrieved_categories": _extract_retrieved_categories(sources),
                    "retrieved_subcategories": _extract_retrieved_subcategories(sources),
                    "error": error,
                }
            )
        except Exception:
            pass
        try:
            tags = [
                f"intent:{intent or 'unknown'}",
                f"lang:{language or 'unknown'}",
                f"path:tools",
            ]
            for t in tool_trace:
                tags.append(f"tool:{t.get('name')}")
            if error:
                tags.append(f"fallback_reason:tool_loop_error")
            _langfuse_add_trace_tags(langfuse, trace_id=getattr(span, "trace_id", None), tags=tags)
        except Exception:
            pass
        try:
            _langfuse_score_numeric(span, "retrieval_count", float(len(sources)))
            _langfuse_score_numeric(span, "tool_rounds", float(len(tool_trace)))
            _langfuse_score_numeric(span, "fallback_triggered", 1.0 if error else 0.0)
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

    products = _products_from_sources(sources)
    try:
        insert_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            metadata={
                "endpoint": "/api/chat_tools",
                "sources_count": len(sources),
                "products_count": len(products or []),
                "products": products or [],
            },
        )
    except Exception as e:
        logger.warning("[chat_db] insert assistant message failed (/api/chat_tools): %s: %s", type(e).__name__, e)
    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        sources=sources,
        sources_total=sources_total,
        products=products,
    )


@app.post("/api/chat_tools_stream")
def chat_tools_stream(req: ChatRequest):
    """
    Streaming (SSE) variant of /api/chat_tools.

    Emits events compatible with the frontend SSE parser:
      - {type: "status", message: "..."}
      - {type: "done", answer, sources}
      - {type: "error", message}

    Full Langfuse tracing is wired here — matches /api/chat_tools output shape
    so the admin dashboard sees traces from either endpoint.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    conversation_id = ensure_conversation_id(req.conversation_id)
    try:
        insert_message(
            conversation_id=conversation_id,
            role="user",
            content=query,
            metadata={"endpoint": "/api/chat_tools_stream"},
        )
    except Exception as e:
        logger.warning("[chat_db] insert user message failed (/api/chat_tools_stream): %s: %s", type(e).__name__, e)

    # --- Langfuse: create the span OUTSIDE the generator so input is logged
    # even if the generator crashes before producing its first event. ---
    langfuse = _get_langfuse_client()
    span = None
    if langfuse is not None:
        try:
            span = langfuse.start_observation(as_type="span", name="api_chat_tools_stream")
            logger.info("[chat_tools_stream] span started trace_id=%s", getattr(span, "trace_id", None))
        except Exception as e:
            logger.error(
                "[chat_tools_stream] start_observation failed: %s: %s",
                type(e).__name__, e, exc_info=True,
            )
            span = None
    else:
        logger.warning("[chat_tools_stream] langfuse client is None — no trace will be sent")

    def _gen():
        intent = "unknown"
        language = "unknown"
        intent_confidence = 0.0
        tool_trace: list[dict[str, Any]] = []
        answer = ""
        sources_out: list[dict[str, Any]] = []
        retrieved_skus: list[str] = []
        sku_product_names: dict[str, str] = {}
        error: str | None = None

        try:
            yield _sse({"type": "phase", "phase": "understanding"})

            try:
                intent, language, intent_confidence = _route_intent_and_language(query)
            except Exception:
                pass

            yield _sse(
                {
                    "type": "intent",
                    "intent": intent,
                    "language": language,
                    "confidence": intent_confidence,
                }
            )

            # Log INPUT to Langfuse as early as possible.
            if span is not None:
                try:
                    span.update(
                        input={
                            "query": query,
                            "intent": intent,
                            "intent_confidence": intent_confidence,
                            "user_language": language,
                            "history_len": len(req.history or []),
                            "skus_in_user_query": list(_extract_skus_from_query(query) or []),
                        }
                    )
                    logger.info(
                        "[chat_tools_stream] input logged: intent=%s lang=%s conf=%.2f",
                        intent, language, intent_confidence,
                    )
                except Exception as e:
                    logger.error(
                        "[chat_tools_stream] span.update(input) failed: %s: %s",
                        type(e).__name__, e, exc_info=True,
                    )

            yield _sse({"type": "phase", "phase": "loading_catalog"})

            bm25_bundle = _get_bm25_bundle()
            client = _get_qdrant_client()
            embedder = _get_embedder()
            collection = os.getenv("QDRANT_COLLECTION", "catalog_es")

            for evt in run_tool_loop_stream(
                user_query=query,
                history=req.history,
                system_prompt=build_tool_system_prompt(language),
                qdrant=client,
                collection=collection,
                embedder=embedder,
                bm25=bm25_bundle["bm25"],
                bm25_chunks=bm25_bundle["chunks"],
            ):
                et = (evt or {}).get("type")
                if et in {"phase", "tool_start", "tool_end"}:
                    yield _sse(evt)
                    continue
                if et == "status":
                    # Backwards-compatible: tool loop may emit textual status updates.
                    # Don't surface internal tool call text to the UI.
                    continue

                if et == "error":
                    error = str(evt.get("message") or "Error.")
                    logger.warning("[chat_tools_stream] tool loop error: %s", error)
                    yield _sse({"type": "error", "message": error})
                    return

                if et == "done_raw":
                    answer = str(evt.get("answer") or "").strip() or "No response."
                    tool_trace = list(evt.get("tool_trace") or [])
                    retrieved_products = list(evt.get("retrieved_products") or [])
                    sources_total = evt.get("sources_total")
                    if not isinstance(sources_total, int) or sources_total < 0:
                        sources_total = None
                    compare_result = evt.get("compare_result")
                    if not isinstance(compare_result, dict):
                        compare_result = None

                    if compare_result:
                        table_md = render_compare_products_markdown(compare_result, user_language=language)
                        if table_md:
                            answer = f"{table_md}\n\n{answer}".strip()

                    # Shape sources to match ChatResponse / frontend expectations.
                    seen_ids: set[str] = set()
                    for p in retrieved_products:
                        cid = str(p.get("id") or p.get("chunk_id") or "")
                        if cid and cid in seen_ids:
                            continue
                        if cid:
                            seen_ids.add(cid)
                        sources_out.append(
                            {
                                "chunk_id": cid,
                                "text": p.get("text") or "",
                                "metadata": p,
                                "score": 1.0,
                            }
                        )

                    # Enrich trace output so the admin dashboard can render "Top SKUs retrieved"
                    # with product names, even for tools-stream traces.
                    try:
                        sku_product_names = {}
                        for s in sources_out:
                            md = (s or {}).get("metadata") or {}
                            if not isinstance(md, dict):
                                continue
                            sku = str(md.get("sku") or "").strip()
                            if not sku:
                                continue
                            name = (
                                str(md.get("name_es") or "").strip()
                                or str(md.get("name") or "").strip()
                                or str(md.get("product") or "").strip()
                            )
                            if sku and name and sku not in sku_product_names:
                                sku_product_names[sku] = name
                        retrieved_skus = sorted(sku_product_names.keys())
                    except Exception:
                        retrieved_skus = []
                        sku_product_names = {}

                    yield _sse({"type": "phase", "phase": "finalizing"})
                    products = _products_from_sources(sources_out)
                    if products:
                        yield _sse({"type": "products", "items": products})
                    try:
                        insert_message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=answer,
                            metadata={
                                "endpoint": "/api/chat_tools_stream",
                                "sources_count": len(sources_out),
                                "products_count": len(products or []),
                                "products": products or [],
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            "[chat_db] insert assistant message failed (/api/chat_tools_stream): %s: %s",
                            type(e).__name__,
                            e,
                        )
                    yield _sse({
                        "type": "done",
                        "conversation_id": conversation_id,
                        "answer": answer,
                        "sources": sources_out,
                        "sources_total": sources_total,
                        "products": products,
                    })
                    return

            error = "Tool loop ended unexpectedly."
            logger.warning("[chat_tools_stream] %s", error)
            yield _sse({"type": "error", "message": error})
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.error("[chat_tools_stream] unhandled: %s", error, exc_info=True)
            yield _sse({"type": "error", "message": error})
        finally:
            # --- Log OUTPUT + tags + scores, then end + flush. ---
            if span is not None:
                try:
                    span.update(
                        output={
                            "answer": (answer or "")[:20000],
                            "answer_truncated": len(answer or "") > 20000,
                            "tool_trace": tool_trace,
                            "rounds": len(tool_trace),
                            "retrieved_count": len(sources_out),
                            "retrieved_skus": retrieved_skus,
                            "sku_product_names": sku_product_names,
                            "sku_counts_in_answer": _sku_counts_in_text(answer, retrieved_skus),
                            "retrieved_brands": _extract_retrieved_brands(sources_out),
                            "retrieved_categories": _extract_retrieved_categories(sources_out),
                            "retrieved_subcategories": _extract_retrieved_subcategories(sources_out),
                            "no_result": len(sources_out) == 0,
                            "fallback_triggered": bool(error),
                            "fallback_reason": error,
                            "error": error,
                        }
                    )
                    logger.info(
                        "[chat_tools_stream] output logged: answer_len=%d tools=%d sources=%d error=%s",
                        len(answer or ""), len(tool_trace), len(sources_out), bool(error),
                    )
                except Exception as e:
                    logger.error(
                        "[chat_tools_stream] span.update(output) failed: %s: %s",
                        type(e).__name__, e, exc_info=True,
                    )

                try:
                    tags = [
                        f"intent:{intent or 'unknown'}",
                        f"lang:{language or 'unknown'}",
                        "path:tools_stream",
                        f"no_result:{'true' if len(sources_out) == 0 else 'false'}",
                    ]
                    for t in tool_trace:
                        nm = (t or {}).get("name")
                        if nm:
                            tags.append(f"tool:{nm}")
                    if error:
                        tags.append("fallback_reason:stream_error")
                    _langfuse_add_trace_tags(
                        langfuse,
                        trace_id=getattr(span, "trace_id", None),
                        tags=tags,
                    )
                    logger.info("[chat_tools_stream] tags added: %s", tags)
                except Exception as e:
                    logger.error(
                        "[chat_tools_stream] add_trace_tags failed: %s: %s",
                        type(e).__name__, e,
                    )

                try:
                    _langfuse_score_numeric(span, "retrieval_count", float(len(sources_out)))
                    _langfuse_score_numeric(span, "tool_rounds", float(len(tool_trace)))
                    _langfuse_score_numeric(span, "no_result", 1.0 if len(sources_out) == 0 else 0.0)
                    _langfuse_score_numeric(span, "fallback_triggered", 1.0 if error else 0.0)
                except Exception as e:
                    logger.error(
                        "[chat_tools_stream] scores failed: %s: %s",
                        type(e).__name__, e,
                    )

                try:
                    span.end()
                    logger.info("[chat_tools_stream] span ended")
                except Exception as e:
                    logger.error(
                        "[chat_tools_stream] span.end failed: %s: %s",
                        type(e).__name__, e,
                    )
            if langfuse:
                try:
                    langfuse.flush()
                    logger.info("[chat_tools_stream] flushed to langfuse cloud")
                except Exception as e:
                    logger.error(
                        "[chat_tools_stream] flush failed: %s: %s",
                        type(e).__name__, e,
                    )

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Langfuse diagnostic endpoint — run a synchronous end-to-end test to isolate
# where the pipeline breaks (keys, SDK, init, observation, update, flush).
# Curl: `curl -s http://127.0.0.1:8000/api/langfuse_diagnose | python3 -m json.tool`
# ---------------------------------------------------------------------------
@app.get("/api/langfuse_diagnose")
def langfuse_diagnose() -> dict[str, Any]:
    report: dict[str, Any] = {
        "env": {
            "LANGFUSE_PUBLIC_KEY_set": bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
            "LANGFUSE_SECRET_KEY_set": bool(os.getenv("LANGFUSE_SECRET_KEY")),
            "LANGFUSE_BASE_URL": os.getenv("LANGFUSE_BASE_URL"),
            "LANGFUSE_HOST": os.getenv("LANGFUSE_HOST"),
        },
        "steps": {},
    }

    # Step 1: SDK import
    try:
        import langfuse as _lf  # type: ignore
        report["steps"]["sdk_import"] = {
            "ok": True,
            "version": getattr(_lf, "__version__", "unknown"),
        }
    except Exception as e:
        report["steps"]["sdk_import"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        report["final"] = "SDK not importable"
        return report

    # Step 2: get_client
    client = _get_langfuse_client()
    if client is None:
        report["steps"]["get_client"] = {
            "ok": False,
            "error": "Client is None (missing keys, SDK issue, or init error). Check server log.",
        }
        report["final"] = "Client init failed"
        return report
    report["steps"]["get_client"] = {
        "ok": True,
        "class": type(client).__name__,
        "has_start_observation": hasattr(client, "start_observation"),
    }

    # Step 3: start_observation
    span = None
    try:
        span = client.start_observation(as_type="span", name="langfuse_diagnose")
        report["steps"]["start_observation"] = {
            "ok": True,
            "trace_id": getattr(span, "trace_id", None),
        }
    except Exception as e:
        report["steps"]["start_observation"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        report["final"] = "start_observation failed"
        return report

    # Step 4: update input
    try:
        span.update(input={"diagnose": "ping", "timestamp": time.time()})
        report["steps"]["update_input"] = {"ok": True}
    except Exception as e:
        report["steps"]["update_input"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Step 5: update output
    try:
        span.update(output={"diagnose": "pong", "answer": "synthetic diagnostic response"})
        report["steps"]["update_output"] = {"ok": True}
    except Exception as e:
        report["steps"]["update_output"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Step 6: trace tags
    try:
        _langfuse_add_trace_tags(
            client, trace_id=getattr(span, "trace_id", None), tags=["diagnose:true"]
        )
        report["steps"]["add_tags"] = {"ok": True}
    except Exception as e:
        report["steps"]["add_tags"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Step 7: end
    try:
        span.end()
        report["steps"]["end_span"] = {"ok": True}
    except Exception as e:
        report["steps"]["end_span"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Step 8: flush
    try:
        client.flush()
        report["steps"]["flush"] = {"ok": True}
    except Exception as e:
        report["steps"]["flush"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    all_ok = all(isinstance(v, dict) and v.get("ok") for v in report["steps"].values())
    report["final"] = "ALL OK — check Langfuse UI for trace 'langfuse_diagnose'" if all_ok else "Some steps failed — see per-step errors"
    return report

