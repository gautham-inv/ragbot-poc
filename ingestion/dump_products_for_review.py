"""
Dump every product row from Qdrant into a flat JSONL with just the fields
needed for manual re-classification (sku, brand, name, current labels,
description snippet). Also writes a `priority_queue.jsonl` ordered by suspicion
so the most-likely-mislabeled products surface first.

Usage:
  python ingestion/dump_products_for_review.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from qdrant_client import QdrantClient


SUSPICIOUS_NAME_TOKENS = {
    # Plush / soft toy markers — if name has these but subcategory != plush, flag.
    "plush":          {"plush", "softies", "cuddler", "snuggle", "stuffie", "fuzzy",
                       "peluche", "soft", "snuzzles", "knots", "vibez", "cozie"},
    # Ball markers — if subcategory == ball but name lacks these, flag.
    "ball_marker":    {"ball", "pelota", "sphere", "orb"},
    # Cat-only product line markers.
    "cat_only":       {"cat softies", "kickeroo", "kitty", "catnip", "cat toy",
                       "purrsuit", "feline", "scratcher", "scratch post"},
    # Dog-only line markers.
    "dog_only":       {"para perro", "for dogs", "dog only", "puppy"},
}


def suspicion_score(p: dict) -> int:
    """Higher = more suspect. Used to order the priority queue."""
    score = 0
    name = (p.get("product_name") or "").lower()
    sub = (p.get("subcategory") or "").lower()
    cat = (p.get("category") or "").lower()
    species = set(p.get("species") or [])

    # Plush-like name but not labeled plush.
    if any(t in name for t in SUSPICIOUS_NAME_TOKENS["plush"]) and sub != "plush":
        score += 5

    # Sub == ball but name doesn't say ball.
    if sub == "ball" and not any(t in name for t in SUSPICIOUS_NAME_TOKENS["ball_marker"]):
        score += 4

    # Generic catch-all subcategory.
    if sub in {"toy", "", None}:
        score += 2

    # Cat-only line marked as dual species.
    if any(t in name for t in SUSPICIOUS_NAME_TOKENS["cat_only"]) and species != {"cat"}:
        score += 4

    # Dog-only marker but listed for cats.
    if any(t in name for t in SUSPICIOUS_NAME_TOKENS["dog_only"]) and "cat" in species:
        score += 3

    # No species at all.
    if not species:
        score += 2

    return score


def main() -> int:
    out_dir = Path("data/recategorize")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / "products_dump.jsonl"
    queue_path = out_dir / "priority_queue.jsonl"
    inventory_path = out_dir / "inventory.txt"

    qclient = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    collection = os.getenv("QDRANT_COLLECTION", "excel_dataset")

    rows: list[dict] = []
    nxt = None
    while True:
        pts, nxt = qclient.scroll(
            collection_name=collection,
            limit=512,
            with_payload=True,
            with_vectors=False,
            offset=nxt,
        )
        if not pts:
            break
        for p in pts:
            pl = p.payload or {}
            if pl.get("chunk_type") != "product_sku_row":
                continue
            text = (pl.get("text") or pl.get("soft_text") or "").strip()
            text = re.sub(r"\s+", " ", text)[:600]
            rows.append({
                "point_id": str(p.id),
                "chunk_id": pl.get("chunk_id"),
                "sku": pl.get("sku"),
                "brand": pl.get("brand"),
                "product_name": pl.get("product_name") or pl.get("name") or "",
                "category": pl.get("category"),
                "subcategory": pl.get("subcategory"),
                "species": pl.get("species") or [],
                "size_label": pl.get("size_label"),
                "description": text,
            })
        if nxt is None:
            break

    rows.sort(key=lambda r: (r.get("brand") or "", r.get("product_name") or ""))
    with dump_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    suspect_rows = sorted(
        ({**r, "_suspicion": suspicion_score(r)} for r in rows),
        key=lambda r: -r["_suspicion"],
    )
    with queue_path.open("w") as f:
        for r in suspect_rows:
            if r["_suspicion"] > 0:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_suspect = sum(1 for r in suspect_rows if r["_suspicion"] > 0)

    inventory = (
        f"Products dumped:        {len(rows)}\n"
        f"Priority-queue size:    {n_suspect}  (heuristically suspicious)\n"
        f"Files:\n"
        f"  {dump_path}\n"
        f"  {queue_path}\n"
    )
    inventory_path.write_text(inventory)
    print(inventory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
