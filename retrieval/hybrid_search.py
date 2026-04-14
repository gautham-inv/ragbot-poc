from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from retrieval.rrf import RankedItem, reciprocal_rank_fusion
from retrieval.tokenize_es import tokenize_es


def _load_bm25(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    data = _load_bm25_bundle(path)
    return data["bm25"], data["chunks"]


def _load_bm25_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def bm25_search(bm25: Any, chunks: list[dict[str, Any]], query: str, top_k: int = 10) -> list[RankedItem]:
    toks = tokenize_es(query)
    scores = bm25.get_scores(toks)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    out: list[RankedItem] = []
    for i in idxs:
        c = chunks[i]
        out.append(RankedItem(id=c["id"], payload=c, score=float(scores[i])))
    return out


def qdrant_search(
    client: QdrantClient,
    collection: str,
    model: Any,
    query: str,
    top_k: int = 10,
    *,
    qdrant_filter: Filter | None = None,
) -> list[RankedItem]:
    q = model.encode([f"query: {query}"], normalize_embeddings=True).tolist()[0]
    if hasattr(client, "search"):
        hits = client.search(
            collection_name=collection,
            query_vector=q,
            query_filter=qdrant_filter,
            limit=top_k,
        )
    else:
        result = client.query_points(
            collection_name=collection,
            query=q,
            query_filter=qdrant_filter,
            limit=top_k,
        )
        hits = result.points
    out: list[RankedItem] = []
    for h in hits:
        payload = dict(h.payload or {})
        # Use the same identifier as BM25 (chunk_id) so RRF merges correctly.
        chunk_id = payload.get("chunk_id") or str(h.id)
        payload["qdrant_point_id"] = str(h.id)
        out.append(RankedItem(id=str(chunk_id), payload=payload, score=float(h.score)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid search (Qdrant + BM25 + RRF).")
    ap.add_argument("--query", required=True)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-api-key", default=None)
    ap.add_argument("--collection", default="catalog_es")
    ap.add_argument("--bm25", default="data/index/bm25.pkl", type=Path)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    args = ap.parse_args()

    bm25, chunks = _load_bm25(args.bm25)
    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(args.model, device="cpu")

    vec = qdrant_search(client, args.collection, model, args.query, top_k=10)
    kw = bm25_search(bm25, chunks, args.query, top_k=10)
    fused = reciprocal_rank_fusion(vec, kw, top_n=5)

    for i, item in enumerate(fused, start=1):
        p = item.payload
        print(f"{i}. {item.id} rrf={item.score:.4f} page={p.get('page_number')} ref={p.get('ref','')}")
        print(p.get("text", "")[:300].replace("\n", " ") + "...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
