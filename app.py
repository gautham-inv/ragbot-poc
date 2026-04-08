import os
import sys
from pathlib import Path

import streamlit as st

# Setup project path for internal imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_project_env
from openrouter import OpenRouter
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from retrieval.hybrid_search import _load_bm25_bundle, bm25_search, qdrant_search
from retrieval.product_dictionary import enrich_query_with_product_names
from retrieval.rag_generate import build_context_str, build_system_prompt
from retrieval.rrf import reciprocal_rank_fusion

load_project_env()

# Prevent noisy HF warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import warnings

warnings.filterwarnings("ignore")


def _format_openrouter_error(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]

    status_code = getattr(exc, "status_code", None)
    if status_code:
        parts.append(f"status={status_code}")

    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status and not status_code:
            parts.append(f"status={resp_status}")
        text = getattr(response, "text", None)
        if text:
            parts.append(text[:500])
        else:
            content = getattr(response, "content", None)
            if content:
                try:
                    decoded = content.decode("utf-8", errors="ignore")
                except Exception:
                    decoded = str(content)
                parts.append(decoded[:500])

    body = getattr(exc, "body", None)
    if body:
        parts.append(str(body)[:500])

    return " | ".join(parts)


TOP_K_SEARCH = 8
TOP_N_FUSE = 4
MAX_HISTORY_MESSAGES = 8
REWRITE_MODEL = "qwen/qwen-turbo"

st.set_page_config(page_title="Gloriapets RAG Bot")

st.title("Gloriapets RAG Bot")
st.markdown(
    """
### Chat with your product catalog
Searching through thousands of SKUs, prices, and tables across 44 pages of data.
"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_retrievers():
    bm25_path = Path("data/index/bm25.pkl")
    model_name = os.getenv("EMBEDDING_MODEL", os.getenv("HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    collection = os.getenv("QDRANT_COLLECTION", "catalog_es")

    bm25_bundle = _load_bm25_bundle(bm25_path)
    bm25 = bm25_bundle["bm25"]
    bm25_chunks = bm25_bundle["chunks"]
    product_dictionary = bm25_bundle.get("product_dictionary", {})
    model = SentenceTransformer(model_name, device="cpu")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    return {
        "bm25": bm25,
        "bm25_chunks": bm25_chunks,
        "product_dictionary": product_dictionary,
        "model": model,
        "client": client,
        "collection": collection,
    }


with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "OpenRouter API Key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
    )

    st.divider()
    st.caption(
        f"Embedding model: `{os.getenv('EMBEDDING_MODEL', os.getenv('HF_EMBEDDING_MODEL', 'intfloat/multilingual-e5-small'))}`"
    )
    st.caption(f"Qdrant collection: `{os.getenv('QDRANT_COLLECTION', 'catalog_es')}`")

    st.divider()

    model_choice = st.selectbox("LLM Model", ["qwen/qwen-plus", "qwen/qwen-max", "openai/gpt-4o"])

    st.divider()

    try:
        get_retrievers()
        st.success("Hybrid Index Loaded")
    except Exception as e:
        st.error(f"Failed to load Index: {e}")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Source Documents"):
                for s in message["sources"]:
                    st.info(f"Page {s.get('page_number', '?')} - Match Score: {s.get('score', '?')}")
                    st.code(s.get("text", "")[:300] + "...")


if prompt := st.chat_input("Ask about a product, SKU, or price..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    print(f"\n[Server] Received query: {prompt}")
    previous_messages = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        progress_placeholder = st.empty()
        response_placeholder = st.empty()
        retrieved_chunks = []
        full_response = ""

        if not api_key:
            st.warning("Please provide an OpenRouter API Key in the sidebar.")
        else:
            try:
                progress_placeholder.markdown("_Analyzing your question..._")
                r = get_retrievers()
                search_query = prompt

                if len(previous_messages) >= 2:
                    progress_placeholder.markdown("_Analyzing conversation context..._")
                    print("[Server] Evaluating conversation context for query rewrite...")
                    try:
                        rewrite_sys_prompt = (
                            "Given the conversation history and the user's new follow-up question, "
                            "rewrite the question as a standalone search query. "
                            "It must include the product names or concepts being discussed. "
                            "Keep it very short, like a search engine query. "
                            "If the question is already standalone or is a simple greeting, return it as-is. "
                            "Do NOT answer the question. ONLY output the rewritten search query text."
                        )

                        history_for_rewrite = previous_messages[-6:]
                        history_str = ""
                        for msg in history_for_rewrite:
                            content_trunc = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                            history_str += f"{msg['role'].capitalize()}: {content_trunc}\n"
                        history_str += f"\nNew user message: {prompt}"

                        with OpenRouter(
                            http_referer="ragbot-poc",
                            x_open_router_title="Gloriapets-RAG",
                            api_key=api_key,
                        ) as open_router:
                            res_rw = open_router.chat.send(
                                model=REWRITE_MODEL,
                                messages=[
                                    {"role": "system", "content": rewrite_sys_prompt},
                                    {"role": "user", "content": history_str},
                                ],
                                stream=True,
                                temperature=0.0,
                            )

                            rewritten_query = ""
                            with res_rw as rewrite_stream:
                                for event in rewrite_stream:
                                    if isinstance(event, str):
                                        rewritten_query += event
                                    elif hasattr(event, "choices") and event.choices:
                                        delta = event.choices[0].delta
                                        if hasattr(delta, "content") and delta.content:
                                            rewritten_query += delta.content
                                    elif isinstance(event, dict) and "choices" in event:
                                        delta = event["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            rewritten_query += delta["content"]

                                    preview = rewritten_query.strip().replace('"', "")
                                    if preview:
                                        progress_placeholder.markdown(
                                            f"_Understanding context... searching for:_ `{preview[:120]}`"
                                        )

                            search_query = rewritten_query.strip().replace('"', "") or search_query
                            print(f"[Server] Rewrote contextual query to: '{search_query}'")
                    except Exception as e:
                        print(f"[Server] Query rewriting failed: {e}")

                search_query = enrich_query_with_product_names(search_query, r["product_dictionary"])
                progress_placeholder.markdown("_Searching the catalog..._")

                print(f"[Server] Performing Semantic Search (top_k={TOP_K_SEARCH})...")
                vec = qdrant_search(r["client"], r["collection"], r["model"], search_query, top_k=TOP_K_SEARCH)

                print(f"[Server] Performing Keyword Search (top_k={TOP_K_SEARCH})...")
                kw = bm25_search(r["bm25"], r["bm25_chunks"], search_query, top_k=TOP_K_SEARCH)

                progress_placeholder.markdown("_Synthesizing results..._")
                print(f"[Server] Fusing results (top_n={TOP_N_FUSE})...")
                fused = reciprocal_rank_fusion(vec, kw, top_n=TOP_N_FUSE)

                retrieved_chunks = [item.payload for item in fused]
                for i, item in enumerate(fused):
                    retrieved_chunks[i]["score"] = round(item.score, 4)
                    print(
                        f"[Server] Retrieved Doc {i+1}: Page {item.payload.get('page_number', '?')} "
                        f"(Score: {retrieved_chunks[i]['score']})"
                    )

                progress_placeholder.markdown("_Information retrieved. Thinking..._")
                print("[Server] Sending context to LLM for response generation...")

                context_str = build_context_str(retrieved_chunks)
                system_prompt = build_system_prompt(context_str)

                with OpenRouter(
                    http_referer="ragbot-poc",
                    x_open_router_title="Gloriapets-RAG",
                    api_key=api_key,
                ) as open_router:
                    llm_messages = [{"role": "system", "content": system_prompt}]
                    recent_history = previous_messages[-MAX_HISTORY_MESSAGES:]
                    for msg in recent_history:
                        llm_messages.append({"role": msg["role"], "content": msg["content"]})

                    llm_messages.append({"role": "user", "content": prompt})

                    res = open_router.chat.send(
                        model=model_choice,
                        messages=llm_messages,
                        stream=True,
                        temperature=0.0,
                    )

                    with res as event_stream:
                        for event in event_stream:
                            progress_placeholder.empty()
                            if isinstance(event, str):
                                full_response += event
                            elif hasattr(event, "choices") and event.choices:
                                delta = event.choices[0].delta
                                if hasattr(delta, "content") and delta.content:
                                    full_response += delta.content
                            elif isinstance(event, dict) and "choices" in event:
                                delta = event["choices"][0].get("delta", {})
                                if "content" in delta:
                                    full_response += delta["content"]
                            response_placeholder.markdown(full_response + "â–Œ")

                    response_placeholder.markdown(full_response)
                    print("[Server] LLM response generated successfully.")
            except Exception as e:
                error_detail = _format_openrouter_error(e)
                full_response = f"Error triggering LLM: {error_detail}"
                print(f"[Server] Error triggering LLM: {error_detail}")
                progress_placeholder.empty()
                response_placeholder.error(full_response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": retrieved_chunks,
                }
            )

            with st.expander("Source Documents"):
                for s in retrieved_chunks:
                    st.info(f"Page {s.get('page_number', '?')} - Match Score: {s.get('score', '?')}")
                    st.code(s.get("text", "")[:300] + "...")
