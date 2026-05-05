"""
Diagnostic: cross-reference Qdrant points against data/sku_image_map.json to
find image coverage gaps.

Output:
  - data/audit_image_coverage.json    full per-SKU breakdown
  - data/audit_no_image.csv           Qdrant SKUs with no image, ranked by whether
                                      the JSON map could supply one
  - stdout summary

Run:
  python indexing/audit_image_coverage.py
"""
from __future__ import annotations

import csv
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

DATA_DIR = Path("data")
OUT_JSON = DATA_DIR / "audit_image_coverage.json"
OUT_CSV = DATA_DIR / "audit_no_image.csv"


def main() -> int:
    if not QDRANT_URL:
        print("ERROR: QDRANT_URL not set", file=sys.stderr)
        return 1
    if not IMAGE_MAP_PATH.exists():
        print(f"ERROR: {IMAGE_MAP_PATH} not found", file=sys.stderr)
        return 1

    raw_map = json.loads(IMAGE_MAP_PATH.read_text(encoding="utf-8"))
    image_map: dict[str, dict] = {}
    for k, v in raw_map.items():
        if not isinstance(v, dict):
            continue
        ks = str(k).strip()
        image_map[ks] = v
        image_map[ks.upper()] = v
        image_map[ks.lower()] = v

    def map_lookup(sku: str) -> dict | None:
        s = sku.strip()
        return image_map.get(s) or image_map.get(s.upper()) or image_map.get(s.lower())

    print(f"[audit] image map: {len(raw_map)} unique SKUs")

    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"[audit] scrolling collection {COLLECTION!r} ...")

    rows: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=512,
            offset=offset,
            with_payload=[
                "sku", "brand", "name_es", "name_en",
                "primary_image", "thumbnail", "images", "thumbnails",
            ],
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            sku = pl.get("sku")
            if not sku:
                continue
            qdrant_imgs = list(pl.get("images") or [])
            qdrant_primary = pl.get("primary_image") or ""
            map_entry = map_lookup(str(sku)) or {}
            map_imgs = list(map_entry.get("images") or [])

            rows.append({
                "sku": sku,
                "brand": pl.get("brand"),
                "name": pl.get("name_es") or pl.get("name_en"),
                "qdrant_image_count": len(qdrant_imgs),
                "qdrant_primary": qdrant_primary,
                "map_image_count": len(map_imgs),
                "map_primary": map_entry.get("primary_image") or "",
                "in_map": bool(map_entry),
            })
        if offset is None:
            break

    print(f"[audit] {len(rows)} Qdrant points read")

    # --- Categorize ---
    has_qd_image      = [r for r in rows if r["qdrant_image_count"] > 0]
    no_qd_image       = [r for r in rows if r["qdrant_image_count"] == 0]
    no_qd_but_in_map  = [r for r in no_qd_image if r["in_map"] and r["map_image_count"] > 0]
    no_qd_no_map      = [r for r in no_qd_image if not r["in_map"] or r["map_image_count"] == 0]

    # SKUs in image map but NOT in Qdrant at all.
    qd_sku_set = {r["sku"] for r in rows}
    qd_sku_upper = {s.upper() for s in qd_sku_set}
    map_only: list[str] = []
    for k in raw_map.keys():
        ks = str(k).strip()
        if ks not in qd_sku_set and ks.upper() not in qd_sku_upper:
            map_only.append(ks)

    # --- Write outputs ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "qdrant_total": len(rows),
        "qdrant_with_image": len(has_qd_image),
        "qdrant_without_image": len(no_qd_image),
        "would_be_fixed_by_map": len(no_qd_but_in_map),
        "truly_missing": len(no_qd_no_map),
        "map_total_skus": len(raw_map),
        "map_skus_not_in_qdrant": len(map_only),
        "map_total_image_urls": sum(len(v.get("images") or []) for v in raw_map.values() if isinstance(v, dict)),
    }

    OUT_JSON.write_text(
        json.dumps({
            "summary": summary,
            "no_qd_but_in_map_sample": no_qd_but_in_map[:50],
            "no_qd_no_map_sample": no_qd_no_map[:50],
            "map_only_sample": map_only[:50],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sku", "brand", "name", "would_fix_by_map", "map_primary"])
        # First: the actionable ones (would be fixed by re-running backfill).
        for r in no_qd_but_in_map:
            w.writerow([r["sku"], r["brand"] or "", r["name"] or "", "yes", r["map_primary"]])
        for r in no_qd_no_map:
            w.writerow([r["sku"], r["brand"] or "", r["name"] or "", "no", ""])

    # --- Print summary ---
    print()
    print("================ Image coverage audit ================")
    print(f"  Qdrant points                : {summary['qdrant_total']}")
    print(f"    with at least 1 image      : {summary['qdrant_with_image']}")
    print(f"    without any image          : {summary['qdrant_without_image']}")
    print(f"      → fixable via image map  : {summary['would_be_fixed_by_map']}")
    print(f"      → truly missing          : {summary['truly_missing']}")
    print()
    print(f"  Image map SKUs               : {summary['map_total_skus']}")
    print(f"    not in Qdrant              : {summary['map_skus_not_in_qdrant']}")
    print(f"    total image URLs           : {summary['map_total_image_urls']}")
    print()
    print(f"  Reports written:")
    print(f"    {OUT_JSON}")
    print(f"    {OUT_CSV}  ({summary['qdrant_without_image']} rows)")

    if summary["would_be_fixed_by_map"] > 0:
        print()
        print(f"  ⚠  {summary['would_be_fixed_by_map']} Qdrant SKUs have NO image but the")
        print(f"     JSON map has one. Re-run `python indexing/backfill_image_payload.py`")
        print(f"     to patch those points.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
