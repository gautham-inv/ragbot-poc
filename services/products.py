"""
Product CRUD service. Qdrant is the source of truth.

Reuses indexing/build_index_from_excel.py:record_to_chunk() so the payload shape
written by the admin UI matches the bulk-indexed payload byte-for-byte. Image
uploads are delegated to services/cloudinary_upload.py.

Responsibilities:
    - create_product / update_product / delete_product
    - add_product_image / remove_product_image
    - get_product / list_products
    - rebuild_bm25() — pulls from Qdrant scroll, regenerates data/index/bm25.pkl

Out of scope here:
    - HTTP layer (lives in backend/admin_api.py)
    - Auth (admin_api.py gates routes via Better Auth session)
"""
from __future__ import annotations

import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable

# Make sibling packages importable when this module is imported from FastAPI.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import load_project_env
from indexing.build_index_from_excel import record_to_chunk
from services.cloudinary_upload import upload_bytes as _cloudinary_upload
from services.cloudinary_upload import delete_image as _cloudinary_delete

load_project_env()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "excel_dataset")
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    os.getenv("HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
)


# ---------------------------------------------------------------------------
# Lazy singletons (Qdrant client, embedding model)
# ---------------------------------------------------------------------------
_client = None
_client_lock = threading.Lock()

_model = None
_model_lock = threading.Lock()


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            from qdrant_client import QdrantClient
            if not QDRANT_URL:
                raise RuntimeError("QDRANT_URL not set")
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
class ProductValidationError(ValueError):
    pass


class ProductNotFoundError(LookupError):
    pass


_REQUIRED_FIELDS = ("sku", "brand")


def _validate_create_input(p: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_FIELDS if not (p.get(k) and str(p[k]).strip())]
    if missing:
        raise ProductValidationError(f"missing required fields: {missing}")


# ---------------------------------------------------------------------------
# Soft-text builder for admin-created products
# ---------------------------------------------------------------------------
def _compose_soft_text(p: dict[str, Any]) -> str:
    """Build the embedding-input string from a flat admin payload, mirroring the
    style produced by the Excel cleaning pipeline (Spanish name + brand + tags +
    species + price). Kept short to match the typical product-row signal."""
    parts: list[str] = []
    name_es = p.get("name_es") or p.get("name_en")
    if name_es:
        parts.append(str(name_es).strip())
    brand = p.get("brand")
    if brand:
        parts.append(f"by {brand}")
    tag_bits = [p.get("category"), p.get("subcategory")]
    tags = [str(t).strip() for t in tag_bits if t]
    if tags:
        parts.append(", ".join(tags))
    species = p.get("species")
    if isinstance(species, list) and species:
        parts.append("For " + " and ".join(str(s) for s in species))
    elif isinstance(species, str) and species:
        parts.append(f"For {species}")
    price = p.get("price_pvpr")
    if price is not None:
        parts.append(f"Price {price}€")
    return ". ".join(parts)


def _to_jsonl_record(p: dict[str, Any]) -> dict[str, Any]:
    """Reshape a flat admin payload into the {payload, names, soft_text, raw_attributes}
    structure record_to_chunk() expects."""
    names = {
        lang: p.get(f"name_{lang}")
        for lang in ("es", "en", "fr", "pt", "it")
        if p.get(f"name_{lang}")
    }
    inner_payload = {
        k: v for k, v in p.items()
        if k not in ("raw_attributes",) and not k.startswith("name_")
    }
    inner_payload.setdefault("name_es", p.get("name_es"))
    inner_payload.setdefault("name_en", p.get("name_en"))
    return {
        "id": p.get("id") or str(uuid.uuid4()),
        "payload": inner_payload,
        "names": names,
        "soft_text": p.get("soft_text") or _compose_soft_text(p),
        "raw_attributes": p.get("raw_attributes") or {},
    }


# ---------------------------------------------------------------------------
# Point-id helpers
# ---------------------------------------------------------------------------
def _chunk_id(brand: str, sku: str) -> str:
    return f"excel:{brand}:{sku}"


def _point_id(brand: str, sku: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, _chunk_id(brand, sku)))


def _embed(text: str) -> list[float]:
    model = _get_model()
    vec = model.encode([f"passage: {text}"], normalize_embeddings=True).tolist()
    return vec[0]


# ---------------------------------------------------------------------------
# Filter helpers (look up by SKU when brand is unknown)
# ---------------------------------------------------------------------------
def _find_point_by_sku(sku: str) -> tuple[str, dict[str, Any]] | None:
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    client = _get_client()
    flt = Filter(must=[FieldCondition(key="sku", match=MatchValue(value=sku))])
    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    p = points[0]
    return str(p.id), dict(p.payload or {})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_product(sku: str) -> dict[str, Any]:
    found = _find_point_by_sku(sku)
    if not found:
        raise ProductNotFoundError(sku)
    _, payload = found
    return payload


def list_products(*, limit: int = 50, offset: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
    client = _get_client()
    points, next_offset = client.scroll(
        collection_name=COLLECTION,
        limit=max(1, min(int(limit), 200)),
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    rows = [dict(p.payload or {}) for p in points]
    return rows, (str(next_offset) if next_offset is not None else None)


def create_product(p: dict[str, Any]) -> dict[str, Any]:
    _validate_create_input(p)
    sku = str(p["sku"]).strip()
    brand = str(p["brand"]).strip()

    # Reject if SKU already indexed.
    if _find_point_by_sku(sku):
        raise ProductValidationError(f"SKU {sku!r} already exists")

    record = _to_jsonl_record(p)
    chunk_id, text, payload = record_to_chunk(record)
    vector = _embed(text)
    point_id = _point_id(brand, sku)

    client = _get_client()
    client.upsert(
        collection_name=COLLECTION,
        points=[{"id": point_id, "vector": vector, "payload": payload}],
    )
    return payload


def update_product(sku: str, patch: dict[str, Any]) -> dict[str, Any]:
    found = _find_point_by_sku(sku)
    if not found:
        raise ProductNotFoundError(sku)
    point_id, existing = found

    # Merge: caller's fields win, but `images`/`thumbnails`/`primary_image`/
    # `thumbnail` stay unless explicitly overwritten (admins typically edit text).
    merged = dict(existing)
    for k, v in patch.items():
        if v is None:
            merged.pop(k, None)
        else:
            merged[k] = v

    brand = merged.get("brand") or existing.get("brand")
    if not brand:
        raise ProductValidationError("brand missing on existing product")

    record = _to_jsonl_record(merged)
    _, text, payload = record_to_chunk(record)

    # Preserve image fields if patch didn't touch them and record_to_chunk dropped them
    # (e.g. SKU not in the JSON image map).
    for k in ("primary_image", "thumbnail", "images", "thumbnails"):
        if k not in payload and existing.get(k) is not None:
            payload[k] = existing[k]

    vector = _embed(text)
    client = _get_client()
    client.upsert(
        collection_name=COLLECTION,
        points=[{"id": point_id, "vector": vector, "payload": payload}],
    )
    return payload


def delete_product(sku: str, *, also_delete_images: bool = True) -> None:
    found = _find_point_by_sku(sku)
    if not found:
        raise ProductNotFoundError(sku)
    point_id, existing = found

    if also_delete_images:
        for entry in (existing.get("image_assets") or []):
            pid = (entry or {}).get("public_id")
            if pid:
                try:
                    _cloudinary_delete(pid)
                except Exception as e:
                    print(f"[products] cloudinary delete failed for {pid}: {e}")

    client = _get_client()
    from qdrant_client.http.models import PointIdsList
    client.delete(collection_name=COLLECTION, points_selector=PointIdsList(points=[point_id]))


# ---------------------------------------------------------------------------
# Image management
# ---------------------------------------------------------------------------
def _next_image_position(existing: dict[str, Any]) -> int:
    assets = existing.get("image_assets") or []
    if not assets:
        return 0
    return max(int(a.get("position", 0)) for a in assets) + 1


def add_product_image(sku: str, file_bytes: bytes) -> dict[str, Any]:
    found = _find_point_by_sku(sku)
    if not found:
        raise ProductNotFoundError(sku)
    point_id, existing = found

    pos = _next_image_position(existing)
    upload = _cloudinary_upload(sku=sku, position=pos, file_bytes=file_bytes)

    images = list(existing.get("images") or [])
    thumbnails = list(existing.get("thumbnails") or [])
    assets = list(existing.get("image_assets") or [])

    images.append(upload["image_url"])
    thumbnails.append(upload["thumbnail"])
    assets.append({
        "public_id": upload["public_id"],
        "position": upload["position"],
        "image_url": upload["image_url"],
        "thumbnail": upload["thumbnail"],
    })

    payload_patch = {
        "primary_image": existing.get("primary_image") or upload["image_url"],
        "thumbnail": existing.get("thumbnail") or upload["thumbnail"],
        "images": images,
        "thumbnails": thumbnails,
        "image_assets": assets,
    }
    client = _get_client()
    client.set_payload(
        collection_name=COLLECTION,
        payload=payload_patch,
        points=[point_id],
    )
    return upload


def remove_product_image(sku: str, public_id: str) -> None:
    found = _find_point_by_sku(sku)
    if not found:
        raise ProductNotFoundError(sku)
    point_id, existing = found

    assets = [a for a in (existing.get("image_assets") or []) if a.get("public_id") != public_id]
    images = [a["image_url"] for a in assets]
    thumbnails = [a["thumbnail"] for a in assets]

    try:
        _cloudinary_delete(public_id)
    except Exception as e:
        print(f"[products] cloudinary delete failed for {public_id}: {e}")

    payload_patch = {
        "image_assets": assets,
        "images": images,
        "thumbnails": thumbnails,
        "primary_image": images[0] if images else None,
        "thumbnail": thumbnails[0] if thumbnails else None,
    }
    # Drop None-valued keys — set_payload would store nulls otherwise.
    payload_patch = {k: v for k, v in payload_patch.items() if v is not None}

    client = _get_client()
    client.set_payload(
        collection_name=COLLECTION,
        payload=payload_patch,
        points=[point_id],
    )


# ---------------------------------------------------------------------------
# BM25 rebuild (called after every write)
# ---------------------------------------------------------------------------
_BM25_OUT = Path(os.getenv("BM25_OUT", "data/index/bm25.pkl"))
_bm25_lock = threading.Lock()


def _scroll_all_payloads() -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield (point_id, payload) for every point in the collection."""
    client = _get_client()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            yield str(p.id), dict(p.payload or {})
        if offset is None:
            break


def rebuild_bm25() -> int:
    """Regenerate data/index/bm25.pkl from Qdrant. Returns chunk count.

    Serialized via a process-local lock so concurrent admin writes can't race
    on the pickle file. (Cross-process safety would need a file lock — not
    needed today since the API runs as one container.)"""
    import pickle
    from rank_bm25 import BM25Okapi
    from retrieval.tokenize_es import tokenize_es

    with _bm25_lock:
        chunks: list[dict[str, Any]] = []
        for pid, payload in _scroll_all_payloads():
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            chunks.append({
                "id": payload.get("chunk_id") or pid,
                "text": text,
                "meta": payload,
            })
        if not chunks:
            return 0

        tokenized = [tokenize_es(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)

        _BM25_OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = _BM25_OUT.with_suffix(_BM25_OUT.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        tmp.replace(_BM25_OUT)
        return len(chunks)
