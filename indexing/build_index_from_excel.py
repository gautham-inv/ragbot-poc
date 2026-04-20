"""
Build Qdrant + BM25 indexes from the cleaned Excel-derived JSONL.

This mirrors indexing/run_build.py (OCR flow) but skips all OCR/table-parsing:
the input is /Users/dias/Documents/products_normalized.jsonl, produced by the
Excel cleaning pipeline. Each record is already one product (one chunk).

Differences vs run_build.py (the "rewrite" answer):
    1. No page loading, no table parsing, no brand/SKU inference — input is already structured.
    2. Text-to-embed = record["soft_text"] (mechanical, no hallucination) instead of the
       comma-joined flat sentence built from OCR tables.
    3. chunk_id = f"excel:{brand}:{sku}" so re-ingest is idempotent.
    4. chunk_type = "product_sku_row" on every record — the backend filter
       (backend/app.py) already keys on this value, so retrieval keeps working
       with zero changes.
    5. Payload carries backend-compat aliases (`price_eur`, `product_name`,
       `barcode_norm`, `min_order`, `weight_g`) AND the full expanded field
       set (species, subcategory, neck_min_cm/max, body_*, chest_*,
       dog/cat_weight_*, length/width/height/depth/thickness_cm, leash_length_m,
       dureza, tipo, scent, breed_suitability, etc.).
    6. Payload indexes expanded from 3 (chunk_type/brand/sku) to ~30 keyword +
       float + integer fields.

Kept identical to run_build.py:
    - SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    - "passage: " prefix, normalize_embeddings=True, cosine distance
    - Single dense vector in the collection (no hybrid named vectors)
    - uuid.uuid5(uuid.NAMESPACE_URL, chunk_id) for point ids
    - Batch upsert (batch_size=32)
    - BM25Okapi pickle written alongside to data/index/bm25.pkl

Env vars:
    QDRANT_URL            (required)  e.g. https://<cluster>.cloud.qdrant.io
    QDRANT_API_KEY        (required for cloud)
    QDRANT_COLLECTION     default: excel_dataset
    EMBEDDING_MODEL       default: intfloat/multilingual-e5-small
    PRODUCTS_JSONL        default: /Users/dias/Documents/products_normalized.jsonl
    BM25_OUT              default: data/index/bm25.pkl
    RECREATE              if "1", delete + recreate the collection

Run:
    export QDRANT_URL=https://...qdrant.io
    export QDRANT_API_KEY=...
    python indexing/build_index_from_excel.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import traceback
import uuid
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_project_env  # noqa: E402
from retrieval.tokenize_es import tokenize_es  # noqa: E402

load_project_env()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QDRANT_URL       = os.getenv("QDRANT_URL")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY") or None
COLLECTION       = os.getenv("QDRANT_COLLECTION", "excel_dataset")
MODEL_NAME       = os.getenv("EMBEDDING_MODEL",
                             os.getenv("HF_EMBEDDING_MODEL",
                                       "intfloat/multilingual-e5-small"))
JSONL_PATH       = Path(os.getenv("PRODUCTS_JSONL",
                                  "/Users/dias/Documents/products_normalized.jsonl"))
BM25_OUT         = Path(os.getenv("BM25_OUT", "data/index/bm25.pkl"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "32"))
RECREATE         = os.getenv("RECREATE") == "1"


# ---------------------------------------------------------------------------
# Payload-index schema (keyword / integer / float)
# ---------------------------------------------------------------------------
KEYWORD_FIELDS = [
    # backend-compat keys
    "chunk_type", "brand", "sku", "product_name", "barcode_norm",
    # new keys (aligned with doc.txt system prompt)
    "category", "subcategory", "species", "change_flag",
    "size_label", "color", "scent", "dureza", "tipo",
    "source_tab", "ean",
]

INTEGER_FIELDS = [
    "min_purchase_qty",
    "min_order",          # alias
    "source_row",
]

FLOAT_FIELDS = [
    "price_pvpr", "price_per_unit",
    "price_eur",           # alias
    "neck_min_cm", "neck_max_cm",
    "body_min_cm", "body_max_cm",
    "chest_min_cm", "chest_max_cm",
    "dog_weight_min_kg", "dog_weight_max_kg",
    "cat_weight_min_kg", "cat_weight_max_kg",
    "length_cm", "width_cm", "height_cm", "depth_cm", "thickness_cm",
    "leash_length_m", "weight_g",
    "cut_length_mm",
    "size_cm",             # alias (max of dimensions) for backend compat
]


# ---------------------------------------------------------------------------
# Build one chunk per product
# ---------------------------------------------------------------------------
def _to_number(v: Any) -> float | int | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def record_to_chunk(rec: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Return (chunk_id, text_to_embed, payload)."""
    p = rec.get("payload") or {}
    names = rec.get("names") or {}
    raw_attrs = rec.get("raw_attributes") or {}

    brand = p.get("brand") or ""
    sku = p.get("sku") or ""
    chunk_id = f"excel:{brand}:{sku}"

    text = (rec.get("soft_text") or "").strip()
    if not text:
        text = f"{brand} product"

    # Flat payload = every field the retrieval / UI might want.
    payload: dict[str, Any] = {
        "text": text,
        "chunk_id": chunk_id,
        "chunk_type": "product_sku_row",        # backend filter expects this

        # ------- backend-compat aliases (used by existing retrieval code) -------
        "brand": brand,
        "sku": sku,
        "product_name": names.get("es") or names.get("en"),
        "barcode_norm": p.get("ean"),
        "price_eur": _to_number(p.get("price_pvpr")),
        "min_order": _to_number(p.get("min_purchase_qty")),
        "weight_g": _to_number(p.get("weight_g")),

        # ------- cleaned identity / classification -------
        "id": rec.get("id"),
        "ean": p.get("ean"),
        "category": p.get("category"),
        "subcategory": p.get("subcategory"),
        "species": p.get("species") or [],
        "change_flag": p.get("change_flag"),

        # ------- pricing / ordering -------
        "price_pvpr": _to_number(p.get("price_pvpr")),
        "price_per_unit": _to_number(p.get("price_per_unit")),
        "min_purchase_qty": _to_number(p.get("min_purchase_qty")),

        # ------- sizing (keywords + ranges) -------
        "size_label": p.get("size_label"),
        "size_raw": p.get("size_raw"),
        "neck_min_cm": _to_number(p.get("neck_min_cm")),
        "neck_max_cm": _to_number(p.get("neck_max_cm")),
        "body_min_cm": _to_number(p.get("body_min_cm")),
        "body_max_cm": _to_number(p.get("body_max_cm")),
        "chest_min_cm": _to_number(p.get("chest_min_cm")),
        "chest_max_cm": _to_number(p.get("chest_max_cm")),
        "dog_weight_min_kg": _to_number(p.get("dog_weight_min_kg")),
        "dog_weight_max_kg": _to_number(p.get("dog_weight_max_kg")),
        "cat_weight_min_kg": _to_number(p.get("cat_weight_min_kg")),
        "cat_weight_max_kg": _to_number(p.get("cat_weight_max_kg")),

        # ------- physical dimensions -------
        "length_cm": _to_number(p.get("length_cm")),
        "width_cm": _to_number(p.get("width_cm")),
        "height_cm": _to_number(p.get("height_cm")),
        "depth_cm": _to_number(p.get("depth_cm")),
        "thickness_cm": _to_number(p.get("thickness_cm")),
        "leash_length_m": _to_number(p.get("leash_length_m")),

        # ------- product-specific attributes -------
        "color": p.get("color"),
        "scent": p.get("scent"),
        "breed_suitability": p.get("breed_suitability"),
        "dureza": p.get("dureza"),
        "tipo": p.get("tipo"),
        "capacity_raw": p.get("capacity_raw"),

        # ------- ANDIS-only -------
        "watts": p.get("watts"),
        "hz": p.get("hz"),
        "cut_length_mm": _to_number(p.get("cut_length_mm")),
        "recambio": p.get("recambio"),

        # ------- stored-only / debug -------
        "photo_ref": p.get("photo_ref"),
        "observaciones": p.get("observaciones"),
        "estado": p.get("estado"),
        "fecha": p.get("fecha"),
        "source_tab": p.get("source_tab"),
        "source_row": _to_number(p.get("source_row")),

        # ------- multilingual display names -------
        "name_es": names.get("es"),
        "name_en": names.get("en"),
        "name_fr": names.get("fr"),
        "name_pt": names.get("pt"),
        "name_it": names.get("it"),

        # ------- lossless fallback -------
        "raw_attributes": raw_attrs,
    }

    # dimensions_cm / size_cm aliases (backend uses these for size queries).
    dims = [v for v in (payload["length_cm"], payload["width_cm"], payload["height_cm"])
            if isinstance(v, (int, float))]
    if dims:
        payload["dimensions_cm"] = dims
        payload["size_cm"] = max(dims)

    # Drop None-valued keys to keep payload small (lists and empty dicts are fine).
    payload = {k: v for k, v in payload.items() if v is not None}
    return chunk_id, text, payload


def build_chunks(records: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    chunks: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for r in records:
        chunk_id, text, payload = record_to_chunk(r)
        if chunk_id in seen:
            # Duplicate (brand, sku) — disambiguate with source_row
            chunk_id = f"{chunk_id}:row{payload.get('source_row', 0)}"
            payload["chunk_id"] = chunk_id
        seen.add(chunk_id)
        chunks.append((chunk_id, text, payload))
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not QDRANT_URL:
        print("ERROR: QDRANT_URL not set", file=sys.stderr)
        return 1
    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} not found — run the cleaning pipeline first.",
              file=sys.stderr)
        return 1

    print(f"[1/5] Loading {JSONL_PATH} ...", flush=True)
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"       Loaded {len(records)} records", flush=True)

    chunks = build_chunks(records)
    print(f"       Built {len(chunks)} product chunks (one product = one chunk)",
          flush=True)

    print(f"[2/5] Loading embedding model: {MODEL_NAME} ...", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        dim = model.get_sentence_embedding_dimension()
        print(f"       Model loaded, embedding dim = {dim}", flush=True)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        return 1

    print(f"[3/5] Preparing Qdrant collection {COLLECTION!r} ...", flush=True)
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import (
            Distance, VectorParams, PayloadSchemaType,
        )

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        if RECREATE:
            try:
                client.delete_collection(COLLECTION)
                print(f"       Deleted existing collection (RECREATE=1)", flush=True)
            except Exception:
                pass

        try:
            client.get_collection(COLLECTION)
            print(f"       Collection {COLLECTION!r} already exists; upserting in place",
                  flush=True)
        except Exception:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"       Created collection {COLLECTION!r} (dim={dim}, cosine)",
                  flush=True)

        created = 0
        for name in KEYWORD_FIELDS:
            try:
                client.create_payload_index(COLLECTION, name, PayloadSchemaType.KEYWORD)
                created += 1
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"       warn: keyword index {name!r}: {e}")
        for name in INTEGER_FIELDS:
            try:
                client.create_payload_index(COLLECTION, name, PayloadSchemaType.INTEGER)
                created += 1
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"       warn: integer index {name!r}: {e}")
        for name in FLOAT_FIELDS:
            try:
                client.create_payload_index(COLLECTION, name, PayloadSchemaType.FLOAT)
                created += 1
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"       warn: float index {name!r}: {e}")
        total = len(KEYWORD_FIELDS) + len(INTEGER_FIELDS) + len(FLOAT_FIELDS)
        print(f"       Payload indexes: {created} new / {total} total", flush=True)
    except Exception as e:
        print(f"ERROR with Qdrant: {e}")
        traceback.print_exc()
        return 1

    print(f"[4/5] Embedding + upserting {len(chunks)} chunks (batch_size={BATCH_SIZE}) ...",
          flush=True)
    try:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [f"passage: {c[1]}" for c in batch]
            embs = model.encode(texts, normalize_embeddings=True).tolist()
            client.upsert(
                collection_name=COLLECTION,
                points=[
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, batch[j][0])),
                        "vector": embs[j],
                        "payload": batch[j][2],
                    }
                    for j in range(len(batch))
                ],
            )
            done = min(i + BATCH_SIZE, len(chunks))
            print(f"       Upserted {done}/{len(chunks)}", flush=True)
    except Exception as e:
        print(f"ERROR during embed/upsert: {e}")
        traceback.print_exc()
        return 1

    print(f"[5/5] Building BM25 pickle at {BM25_OUT} ...", flush=True)
    try:
        from rank_bm25 import BM25Okapi
        BM25_OUT.parent.mkdir(parents=True, exist_ok=True)
        tokenized = [tokenize_es(c[1]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        with BM25_OUT.open("wb") as f:
            pickle.dump(
                {
                    "bm25": bm25,
                    "chunks": [
                        {"id": c[0], "text": c[1], "meta": c[2]}
                        for c in chunks
                    ],
                },
                f,
            )
        print(f"       BM25 index saved ({len(chunks)} chunks)", flush=True)
    except Exception as e:
        print(f"ERROR building BM25: {e}")
        traceback.print_exc()
        return 1

    info = client.get_collection(COLLECTION)
    print()
    print("=== DONE ===", flush=True)
    print(f"Qdrant collection {COLLECTION!r}: {info.points_count} points", flush=True)
    print(f"BM25 pickle: {BM25_OUT} ({len(chunks)} chunks)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
