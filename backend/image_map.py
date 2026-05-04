"""
SKU → image URLs lookup.

The map is produced by `ingestion/upload_product_images.py` and stored at
`data/sku_image_map.json`. Loaded once per process; empty dict if the file is
absent so the bot keeps working pre-upload.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "sku_image_map.json"


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, Any]]:
    path = Path(os.getenv("SKU_IMAGE_MAP_PATH", str(_DEFAULT_PATH)))
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        # Never fail chatbot requests because of this optional asset.
        pass
    return {}


def get_images(sku: str | None) -> dict[str, Any] | None:
    """Return the image record for a SKU or None. Keys: primary_image, thumbnail, images[], thumbnails[]."""
    if not sku:
        return None
    m = _load()
    key = str(sku).strip()
    return m.get(key) or m.get(key.upper()) or m.get(key.lower())


def attach_images(product: dict[str, Any]) -> dict[str, Any]:
    """Mutate+return a product summary with image fields (or empty strings if no entry).

    Order of precedence:
      1. Image fields already on the product dict (Qdrant payload — populated by
         indexing/build_index_from_excel.py and admin add-product writes).
      2. data/sku_image_map.json fallback (back-compat for any point not yet
         migrated by indexing/backfill_image_payload.py).
    """
    has_payload_image = bool(product.get("primary_image") or product.get("images"))
    if not has_payload_image:
        rec = get_images(product.get("sku"))
        if rec:
            product["primary_image"] = rec.get("primary_image") or ""
            product["thumbnail"] = rec.get("thumbnail") or ""
            product["images"] = rec.get("images") or []
            product["thumbnails"] = rec.get("thumbnails") or []

    product.setdefault("primary_image", "")
    product.setdefault("thumbnail", "")
    product.setdefault("images", [])
    product.setdefault("thumbnails", [])
    return product
