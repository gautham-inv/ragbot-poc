"""
Patch Qdrant payloads in-place with `catalog_pages` and `primary_page` from
data/index/sku_pages.json — NO re-embedding, NO vector recomputation.

Uses Qdrant's `set_payload` API, which only updates the payload of existing
points. The dense vectors (e5-small, 384-d) stay untouched.

Usage (env-driven, matches indexing/build_index_from_excel.py):
    export QDRANT_URL=...
    export QDRANT_API_KEY=...
    export QDRANT_COLLECTION=excel_dataset
    python indexing/patch_qdrant_pages.py

Options:
    --dry-run     Print what would be patched, don't touch Qdrant.
    --map PATH    Override default data/index/sku_pages.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_project_env  # noqa: E402

load_project_env()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, default=Path("data/index/sku_pages.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY") or None
    coll = os.getenv("QDRANT_COLLECTION", "excel_dataset")
    if not url:
        print("ERROR: QDRANT_URL not set", file=sys.stderr)
        return 1

    if not args.map.is_file():
        print(f"ERROR: map not found: {args.map} — run indexing/build_sku_page_map.py first",
              file=sys.stderr)
        return 1

    mapping: dict[str, dict] = json.loads(args.map.read_text(encoding="utf-8"))
    print(f"Loaded page mapping for {len(mapping)} SKUs from {args.map}")

    if args.dry_run:
        print("--dry-run → will not call Qdrant.")

    from qdrant_client import QdrantClient
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    client = QdrantClient(url=url, api_key=key)

    # Scroll ALL points in the collection; match each by sku in payload.
    offset = None
    total = 0
    patched = 0
    unmatched_samples: list[str] = []

    flt = Filter(must=[
        FieldCondition(key="chunk_type", match=MatchValue(value="product_sku_row"))
    ])

    # Group patches by (pages tuple, primary_page) so we issue one set_payload
    # per unique patch rather than per-point. Qdrant happily accepts a list of
    # point IDs for one payload.
    patches_by_payload: dict[tuple, list[str]] = {}

    while True:
        points, offset = client.scroll(
            collection_name=coll,
            scroll_filter=flt,
            limit=args.batch_size,
            with_payload=["sku"],
            offset=offset,
        )
        if not points:
            break

        for p in points:
            total += 1
            sku = (p.payload or {}).get("sku")
            if not sku:
                continue
            sku_key = str(sku).strip().upper()
            entry = mapping.get(sku_key)
            if not entry:
                if len(unmatched_samples) < 10:
                    unmatched_samples.append(sku_key)
                continue

            pages = tuple(entry.get("pages") or [])
            primary = entry.get("primary_page")
            key_tuple = (pages, primary)
            patches_by_payload.setdefault(key_tuple, []).append(str(p.id))
            patched += 1

        if offset is None:
            break

    print(f"Scanned {total} points; {patched} matched, "
          f"{total - patched} had no PDF mapping (normal for new 2026 SKUs).")
    if unmatched_samples:
        print(f"  examples without page match: {unmatched_samples}")

    if args.dry_run:
        print(f"Would issue {len(patches_by_payload)} set_payload calls.")
        return 0

    # Issue batched set_payload calls.
    for (pages, primary), point_ids in patches_by_payload.items():
        payload_patch = {
            "catalog_pages": list(pages),
            "primary_page": primary,
        }
        # set_payload with a list of point IDs performs the same payload merge on all of them.
        client.set_payload(
            collection_name=coll,
            payload=payload_patch,
            points=point_ids,
            wait=False,
        )

    print(f"set_payload issued on {patched} points across {len(patches_by_payload)} unique patches.")
    print("Done. Vectors unchanged; only payloads were enriched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
