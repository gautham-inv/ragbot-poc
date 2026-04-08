from __future__ import annotations

import argparse
import json
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retrieval.product_dictionary import build_product_dictionary, enrich_text_with_product_names, product_chunks
from retrieval.tokenize import tokenize_es


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: dict[str, Any]


def _load_pages(pages_dir: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for p in sorted(pages_dir.glob("page_*.json")):
        pages.append(json.loads(p.read_text(encoding="utf-8")))
    return pages


def _simple_token_chunk(text: str, *, target_tokens: int = 650, overlap: int = 100) -> list[str]:
    # Tokenization approximation (word tokens) that’s stable offline.
    toks = text.split()
    if not toks:
        return []
    out: list[str] = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + target_tokens)
        out.append(" ".join(toks[i:j]))
        if j == len(toks):
            break
        i = max(0, j - overlap)
    return out


def build_chunks(pages: list[dict[str, Any]]) -> list[Chunk]:
    product_dictionary = build_product_dictionary(pages)
    chunks: list[Chunk] = []
    for page in pages:
        page_num = int(page.get("page_number") or 0)
        source_file = page.get("source_file") or "unknown"
        text = enrich_text_with_product_names(page.get("text") or "", product_dictionary)
        page_chunks = product_chunks(text)
        if len(page_chunks) == 1 and page_chunks[0][1] is None:
            page_chunks = [(chunk, None) for chunk in _simple_token_chunk(text)]
        for idx, (ch, product_name) in enumerate(page_chunks, start=0):
            chunk_id = f"{source_file}:p{page_num}:c{idx}"
            meta: dict[str, Any] = {
                "source_file": source_file,
                "page_number": page_num,
                "chunk_index": idx,
                "chunk_id": chunk_id,
            }
            if product_name:
                meta["product_name"] = product_name
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=ch,
                    meta=meta,
                )
            )
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Qdrant + BM25 indexes from ingested pages.")
    ap.add_argument("--pages-dir", default="data/ingested/pages", type=Path)
    ap.add_argument("--collection", default="catalog_es")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-api-key", default=None)
    ap.add_argument("--bm25-out", default="data/index/bm25.pkl", type=Path)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    pages = _load_pages(args.pages_dir)
    chunks = build_chunks(pages)
    if not chunks:
        raise SystemExit(f"No chunks built from: {args.pages_dir}")

    args.bm25_out.parent.mkdir(parents=True, exist_ok=True)

    # Embeddings (E5 uses prefixes).
    model = SentenceTransformer(args.model, device="cpu")

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    dim = model.get_sentence_embedding_dimension()
    try:
        client.get_collection(args.collection)
    except Exception:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    for i in tqdm(range(0, len(chunks), args.batch_size), desc="Embedding+upsert"):
        batch = chunks[i : i + args.batch_size]
        texts = [f"passage: {c.text}" for c in batch]
        embs = model.encode(texts, normalize_embeddings=True).tolist()

        client.upsert(
            collection_name=args.collection,
            points=[
                {
                    # Qdrant point IDs must be int or UUID. We derive a stable UUID
                    # from our human-readable chunk_id, and store chunk_id in payload.
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, batch[j].chunk_id)),
                    "vector": embs[j],
                    "payload": {"text": batch[j].text, **batch[j].meta},
                }
                for j in range(len(batch))
            ],
        )

    # BM25
    tokenized = [tokenize_es(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    product_dictionary = build_product_dictionary(pages)
    with args.bm25_out.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": [{"id": c.chunk_id, "text": c.text, "meta": c.meta} for c in chunks],
                "product_dictionary": product_dictionary,
            },
            f,
        )

    print(f"Upserted {len(chunks)} chunks to Qdrant collection {args.collection}")
    print(f"Wrote BM25 index to {args.bm25_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
