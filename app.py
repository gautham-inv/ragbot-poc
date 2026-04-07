import streamlit as st
import os
import sys
from pathlib import Path

# Setup project path for internal imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openrouter import OpenRouter

from retrieval.hybrid_search import _load_bm25, qdrant_search, bm25_search
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.rag_generate import build_context_str, build_system_prompt

# Prevent noisy HF warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_K_SEARCH = 8          # documents to retrieve from each search method
TOP_N_FUSE = 4            # documents to keep after RRF fusion
MAX_HISTORY_MESSAGES = 8  # cap conversation history sent to LLM (4 turns)
REWRITE_MODEL = "qwen/qwen-turbo"  # lightweight model for query rewriting

# App Configuration
st.set_page_config(page_title="Gloriapets RAG Bot")

# App Title & Intro
st.title("Gloriapets RAG Bot")
st.markdown("""
### Chat with your product catalog
Searching through thousands of SKUs, prices, and tables across 44 pages of data.
""")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load models and index only once
@st.cache_resource
def get_retrievers():
    bm25_path = Path("data/index/bm25.pkl")
    model_name = "intfloat/multilingual-e5-small"
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    collection = "catalog_es"
    
    # Load BM25
    bm25, bm25_chunks = _load_bm25(bm25_path)
    # Load Embedder
    model = SentenceTransformer(model_name)
    # Connect Qdrant
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    return {
        "bm25": bm25,
        "bm25_chunks": bm25_chunks,
        "model": model,
        "client": client,
        "collection": collection
    }

# Sidebar for Config & Status
with st.sidebar:
    st.header("Configuration")
    
    # Check for API Key
    api_key = st.text_input("OpenRouter API Key", 
                          value=os.getenv("OPENROUTER_API_KEY", ""), 
                          type="password")
    
    st.divider()
    
    # Search Params
    model_choice = st.selectbox("LLM Model", 
                              ["qwen/qwen-plus", "qwen/qwen-max", "openai/gpt-4o"])
    
    st.divider()
    
    try:
        retrievers = get_retrievers()
        st.success("Hybrid Index Loaded")
    except Exception as e:
        st.error(f"Failed to load Index: {e}")
    
    st.divider()
    
    # Clear Chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Loop — render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Source Documents"):
                for s in message["sources"]:
                    st.info(f"Page {s.get('page_number', '?')} - Match Score: {s.get('score', '?')}")
                    st.code(s.get("text", "")[:300] + "...")

# Chat Input
if prompt := st.chat_input("Ask about a product, SKU, or price..."):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    print(f"\n[Server] Received query: {prompt}")

    # ------------------------------------------------------------------
    # RETRIEVAL & PROCESSING
    # ------------------------------------------------------------------
    with st.status("Analyzing your question...", expanded=False) as status:
        r = get_retrievers()
        search_query = prompt
        
        # Contextual query rewriting
        previous_messages = st.session_state.messages[:-1]
        if len(previous_messages) >= 2 and api_key:
            status.update(label="Analyzing conversation context...", state="running")
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
                    content_trunc = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                    history_str += f"{msg['role'].capitalize()}: {content_trunc}\n"
                history_str += f"\nNew user message: {prompt}"
                
                with OpenRouter(
                    http_referer="ragbot-poc",
                    x_open_router_title="Gloriapets-RAG",
                    api_key=api_key
                ) as open_router:
                    res_rw = open_router.chat.send(
                        model=REWRITE_MODEL,
                        messages=[
                            {"role": "system", "content": rewrite_sys_prompt},
                            {"role": "user", "content": history_str}
                        ],
                        stream=False,
                        temperature=0.0
                    )
                    
                    if hasattr(res_rw, "choices") and res_rw.choices:
                        search_query = res_rw.choices[0].message.content.strip().replace('"', '')
                        print(f"[Server] Rewrote contextual query to: '{search_query}'")
            except Exception as e:
                print(f"[Server] Query rewriting failed: {e}")

        # Search step
        status.update(label="Searching product catalog...", state="running")
        print(f"[Server] Performing Semantic Search (top_k={TOP_K_SEARCH})...")
        vec = qdrant_search(r["client"], r["collection"], r["model"], search_query, top_k=TOP_K_SEARCH)
        
        print(f"[Server] Performing Keyword Search (top_k={TOP_K_SEARCH})...")
        kw = bm25_search(r["bm25"], r["bm25_chunks"], search_query, top_k=TOP_K_SEARCH)
        
        status.update(label="Synthesizing results...", state="running")
        print(f"[Server] Fusing results (top_n={TOP_N_FUSE})...")
        fused = reciprocal_rank_fusion(vec, kw, top_n=TOP_N_FUSE)
        
        retrieved_chunks = [item.payload for item in fused]
        for i, item in enumerate(fused):
            retrieved_chunks[i]["score"] = round(item.score, 4)
            print(f"[Server] Retrieved Doc {i+1}: Page {item.payload.get('page_number', '?')} (Score: {retrieved_chunks[i]['score']})")
        
        status.update(label="Information retrieved!", state="complete", expanded=False)

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("Please provide an OpenRouter API Key in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                print("[Server] Sending context to LLM for response generation...")
                response_placeholder = st.empty()
                full_response = ""
                
                # Build system prompt using shared guardrails
                context_str = build_context_str(retrieved_chunks)
                system_prompt = build_system_prompt(context_str)
                
                try:
                    with OpenRouter(
                        http_referer="ragbot-poc",
                        x_open_router_title="Gloriapets-RAG",
                        api_key=api_key
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
                            temperature=0.0
                        )
                        
                        with res as event_stream:
                            for event in event_stream:
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
                                response_placeholder.markdown(full_response + "▌")
                                
                        response_placeholder.markdown(full_response)
                        print("[Server] LLM response generated successfully.")
                except Exception as e:
                    full_response = f"Error triggering LLM: {e}"
                    print(f"[Server] Error triggering LLM: {e}")
                    response_placeholder.error(full_response)

            # Store assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_chunks
            })
            
            # Display Sources on fresh response
            with st.expander("Source Documents"):
                for s in retrieved_chunks:
                    st.info(f"Page {s.get('page_number', '?')} - Match Score: {s.get('score', '?')}")
                    st.code(s.get("text", "")[:300] + "...")

