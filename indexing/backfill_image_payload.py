"""
One-time migration: fold data/sku_image_map.json into the existing Qdrant
payload so points carry primary_image / thumbnail / images / thumbnails
without re-embedding.

Run once after deploying the record_to_chunk() change. From this point onward
the JSON file is no longer the runtime source — Qdrant is. Future re-ingests
already preserve these fields (see ADMIN_PRESERVED_FIELDS in
indexing/build_index_from_excel.py).

Env:
    QDRANT_URL, QDRANT_API_KEY      (required, same as build_index_from_excel)
    QDRANT_COLLECTION               default: excel_dataset
    SKU_IMAGE_MAP_PATH              default: data/sku_image_map.json
    DRY_RUN                         set to "1" to log without writing

Run:
    python indexing/backfill_image_payload.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_project_env  # noqa: E402

load_project_env()


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "excel_dataset")
IMAGE_MAP_PATH = Path(os.getenv("SKU_IMAGE_MAP_PATH", "data/sku_image_map.json"))
DRY_RUN = os.getenv("DRY_RUN") == "1"

IMAGE_FIELDS = ("primary_image", "thumbnail", "images", "thumbnails")


def main() -> int:
    if not QDRANT_URL:
        print("ERROR: QDRANT_URL not set", file=sys.stderr)
        return 1
    if not IMAGE_MAP_PATH.exists():
        print(f"ERROR: {IMAGE_MAP_PATH} not found", file=sys.stderr)
        return 1

    image_map_raw = json.loads(IMAGE_MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(image_map_raw, dict):
        print(f"ERROR: {IMAGE_MAP_PATH} is not a dict", file=sys.stderr)
        return 1

    # Case-insensitive lookup table.
    image_map: dict[str, dict] = {}
    for k, v in image_map_raw.items():
        if not isinstance(v, dict):
            continue
        image_map[str(k).strip()] = v
        image_map[str(k).strip().upper()] = v
        image_map[str(k).strip().lower()] = v
    print(f"[backfill] {len(image_map_raw)} SKUs in image map")

    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # 1) Scroll the whole collection once, collect (point_id, sku) pairs.
    print(f"[backfill] scrolling collection {COLLECTION!r} ...")
    pairs: list[tuple[str, str]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=512,
            offset=offset,
            with_payload=["sku"],
            with_vectors=False,
        )
        for p in points:
            sku = (p.payload or {}).get("sku")
            if sku:
                pairs.append((str(p.id), str(sku)))
        if offset is None:
            break
    print(f"[backfill] {len(pairs)} points scrolled")

    # 2) For each point whose SKU is in the image map, set_payload (merge).
    updated = 0
    skipped_no_match = 0
    skipped_no_fields = 0
    for point_id, sku in pairs:
        entry = image_map.get(sku.strip()) or image_map.get(sku.strip().upper()) or image_map.get(sku.strip().lower())
        if not entry:
            skipped_no_match += 1
            continue

        payload_patch = {k: entry.get(k) for k in IMAGE_FIELDS if entry.get(k)}
        if not payload_patch:
            skipped_no_fields += 1
            continue

        if DRY_RUN:
            updated += 1
            continue

        try:
            client.set_payload(
                collection_name=COLLECTION,
                payload=payload_patch,
                points=[point_id],
            )
            updated += 1
            if updated % 500 == 0:
                print(f"       patched {updated} points...", flush=True)
        except Exception as e:
            print(f"       warn: {sku} ({point_id}): {e}")

    print()
    print("=== DONE ===")
    print(f"  Patched              : {updated}")
    print(f"  Skipped (no SKU match): {skipped_no_match}")
    print(f"  Skipped (empty entry) : {skipped_no_fields}")
    if DRY_RUN:
        print("  (dry-run — nothing actually written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
