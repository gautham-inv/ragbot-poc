"""Standalone script to build Qdrant + BM25 indexes. 
Avoids PowerShell stderr issues with the module runner."""
from __future__ import annotations

import os
import pickle
import sys
import uuid
import traceback

# Suppress HF warnings that break PowerShell
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_project_env
from indexing.build_index import _load_pages, build_chunks
from retrieval.tokenize_es import tokenize_es

load_project_env()


def main() -> int:
    pages_dir = Path("data/ingested/pages")
    collection = "catalog_es"
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    bm25_out = Path("data/index/bm25.pkl")
    model_name = os.getenv("EMBEDDING_MODEL", os.getenv("HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    batch_size = 32

    print(f"[1/5] Loading pages from {pages_dir}...", flush=True)
    pages = _load_pages(pages_dir)
    chunks = build_chunks(pages)
    if not chunks:
        print("ERROR: No chunks built!")
        return 1
    print(f"       Built {len(chunks)} product-level chunks from {len(pages)} pages", flush=True)

    print(f"[2/5] Loading embedding model: {model_name}...", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device="cpu")
        dim = model.get_sentence_embedding_dimension()
        print(f"       Model loaded, embedding dim = {dim}", flush=True)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        return 1

    print(f"[3/5] Creating Qdrant collection '{collection}'...", flush=True)
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        # Delete if exists
        try:
            client.delete_collection(collection)
        except Exception:
            pass
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # Create payload indexes for filtered fields
        for field_name in ["chunk_type", "brand", "sku"]:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        print(f"       Collection created", flush=True)
    except Exception as e:
        print(f"ERROR with Qdrant: {e}")
        traceback.print_exc()
        return 1

    print(f"[4/5] Embedding & upserting {len(chunks)} chunks (batch_size={batch_size})...", flush=True)
    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [f"passage: {c.text}" for c in batch]
            embs = model.encode(texts, normalize_embeddings=True).tolist()
            client.upsert(
                collection_name=collection,
                points=[
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, batch[j].chunk_id)),
                        "vector": embs[j],
                        "payload": {"text": batch[j].text, **batch[j].meta},
                    }
                    for j in range(len(batch))
                ],
            )
            done = min(i + batch_size, len(chunks))
            print(f"       Upserted {done}/{len(chunks)}", flush=True)
    except Exception as e:
        print(f"ERROR during embed/upsert: {e}")
        traceback.print_exc()
        return 1

    print(f"[5/5] Building BM25 index...", flush=True)
    try:
        from rank_bm25 import BM25Okapi
        bm25_out.parent.mkdir(parents=True, exist_ok=True)
        tokenized = [tokenize_es(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        with bm25_out.open("wb") as f:
            pickle.dump(
                {
                    "bm25": bm25,
                    "chunks": [{"id": c.chunk_id, "text": c.text, "meta": c.meta} for c in chunks],
                },
                f,
            )
        print(f"       BM25 index saved to {bm25_out}", flush=True)
    except Exception as e:
        print(f"ERROR building BM25: {e}")
        traceback.print_exc()
        return 1

    # Verify
    info = client.get_collection(collection)
    print(f"\n=== DONE ===", flush=True)
    print(f"Qdrant collection '{collection}': {info.points_count} points", flush=True)
    print(f"BM25 index: {len(chunks)} chunks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
