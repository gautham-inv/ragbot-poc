"""
Cloudinary upload service for the admin add-product flow.

Reuses the same cloud_name / API key / secret as ingestion/upload_product_images.py
and the same folder convention (`products/`) so URLs match the existing
data/sku_image_map.json shape.

public_id convention is sku-N (e.g. GOOD3P201-0, GOOD3P201-1) so URLs are
predictable and re-uploads to the same slot overwrite cleanly.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any

import cloudinary
import cloudinary.uploader

from config import load_project_env

load_project_env()


CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
API_KEY = os.getenv("CLOUDINARY_API_KEY", "").strip()
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "").strip()

UPLOAD_FOLDER = "products"
CDN_BASE = f"https://res.cloudinary.com/{CLOUD_NAME}/image/upload" if CLOUD_NAME else ""

_TRANSFORM_FULL = "w_800,q_auto,f_auto"
_TRANSFORM_THUMB = "w_400,q_auto,f_auto"

_configure_lock = threading.Lock()
_configured = False


def _configure() -> None:
    """Idempotent. Errors loudly if creds are missing — admin write paths must fail
    fast instead of silently uploading nowhere."""
    global _configured
    if _configured:
        return
    with _configure_lock:
        if _configured:
            return
        if not (CLOUD_NAME and API_KEY and API_SECRET):
            raise RuntimeError(
                "Cloudinary credentials missing. Set CLOUDINARY_CLOUD_NAME / "
                "CLOUDINARY_API_KEY / CLOUDINARY_API_SECRET in .env."
            )
        cloudinary.config(
            cloud_name=CLOUD_NAME,
            api_key=API_KEY,
            api_secret=API_SECRET,
            secure=True,
        )
        _configured = True


def _safe_public_id(sku: str, position: int) -> str:
    # Match the sanitization rule used by ingestion/upload_product_images.py:public_id_for
    # so existing URLs remain reachable if an admin replaces an image.
    stem = f"{sku}-{position}"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-")
    return stem or f"image-{position}"


def _full_url(public_id: str) -> str:
    return f"{CDN_BASE}/{_TRANSFORM_FULL}/{UPLOAD_FOLDER}/{public_id}"


def _thumb_url(public_id: str) -> str:
    return f"{CDN_BASE}/{_TRANSFORM_THUMB}/{UPLOAD_FOLDER}/{public_id}"


def upload_bytes(*, sku: str, position: int, file_bytes: bytes) -> dict[str, Any]:
    """
    Upload raw image bytes for a given SKU + position.

    Returns:
        {
          "public_id": "GOOD3P201-0",
          "secure_url": "<raw upload URL Cloudinary returned>",
          "image_url":  "<CDN URL with w_800 transform>",
          "thumbnail":  "<CDN URL with w_400 transform>",
          "position":   0,
        }

    Raises RuntimeError if Cloudinary rejects the upload.
    """
    _configure()
    public_id = _safe_public_id(sku, position)

    res = cloudinary.uploader.upload(
        file_bytes,
        public_id=public_id,
        folder=UPLOAD_FOLDER,
        overwrite=True,
        unique_filename=False,
        resource_type="image",
    )
    secure_url = res.get("secure_url") or res.get("url")
    if not secure_url:
        raise RuntimeError(f"Cloudinary returned empty URL for {public_id!r}")

    return {
        "public_id": public_id,
        "secure_url": secure_url,
        "image_url": _full_url(public_id),
        "thumbnail": _thumb_url(public_id),
        "position": int(position),
    }


def delete_image(public_id: str) -> bool:
    """Delete an image by public_id (no folder prefix). Returns True if Cloudinary
    confirmed deletion, False if it reported the asset wasn't found."""
    _configure()
    full_id = f"{UPLOAD_FOLDER}/{public_id}"
    res = cloudinary.uploader.destroy(full_id, resource_type="image", invalidate=True)
    return res.get("result") == "ok"
