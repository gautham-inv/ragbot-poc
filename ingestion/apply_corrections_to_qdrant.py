"""
Apply manual_corrections.jsonl to Qdrant cloud.

Each correction patches category / subcategory / species in the point's payload,
keyed by chunk_id. Uses set_payload (merge, not overwrite) so other payload
fields (text, soft_text, brand, sku, names, dimensions, etc.) stay intact.
The vector is unchanged — no re-embed needed.

Usage:
  # Validate without writing:
  python ingestion/apply_corrections_to_qdrant.py --dry-run

  # Apply for real:
  python ingestion/apply_corrections_to_qdrant.py --apply

  # Limit for testing:
  python ingestion/apply_corrections_to_qdrant.py --apply --limit 5

Env vars: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION (default 'excel_dataset').
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

CORRECTIONS = Path("data/recategorize/manual_corrections.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="Verify chunk_ids exist; do not write.")
    g.add_argument("--apply", action="store_true",
                   help="Actually write payload updates to Qdrant.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N corrections (for testing).")
    args = ap.parse_args()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "excel_dataset")
    if not qdrant_url:
        print("QDRANT_URL not set", file=sys.stderr)
        return 2

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    corrections = [
        json.loads(line)
        for line in CORRECTIONS.read_text().splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        corrections = corrections[: args.limit]

    print(f"corrections to process: {len(corrections)}")
    print(f"collection: {collection}")
    print(f"mode: {'DRY-RUN (verify only)' if args.dry_run else 'APPLY (mutates Qdrant)'}")
    if args.apply:
        print("starting in 3 seconds — Ctrl+C to abort…")
        time.sleep(3)

    # Build chunk_id → point_id map in one scroll so we don't need a payload index.
    print("loading chunk_id → point_id map from Qdrant…")
    chunk_to_point: dict[str, str] = {}
    nxt = None
    n_scanned = 0
    while True:
        pts, nxt = client.scroll(
            collection_name=collection,
            limit=512,
            with_payload=["chunk_id"],
            with_vectors=False,
            offset=nxt,
        )
        if not pts:
            break
        for p in pts:
            cid = (p.payload or {}).get("chunk_id")
            if cid:
                chunk_to_point[cid] = str(p.id)
        n_scanned += len(pts)
        if nxt is None:
            break
    print(f"  mapped {len(chunk_to_point)} chunk_ids from {n_scanned} points")

    n_ok = n_err = n_skip = 0
    n_changed_field: dict[str, int] = {"category": 0, "subcategory": 0, "species": 0}
    sample_diffs: list[str] = []
    not_found: list[str] = []
    write_errors: list[tuple[str, str]] = []

    t0 = time.perf_counter()
    for i, c in enumerate(corrections):
        chunk_id = c.get("chunk_id")
        new = c.get("new") or {}
        old = c.get("old") or {}
        sku = c.get("sku")

        if not chunk_id:
            n_skip += 1
            continue

        # Build merge payload — only the fields that actually change.
        payload: dict = {}
        if new.get("category") is not None and new.get("category") != old.get("category"):
            payload["category"] = new["category"]
            n_changed_field["category"] += 1
        if new.get("subcategory") is not None and new.get("subcategory") != old.get("subcategory"):
            payload["subcategory"] = new["subcategory"]
            n_changed_field["subcategory"] += 1
        if new.get("species") is not None and set(new.get("species") or []) != set(old.get("species") or []):
            payload["species"] = new["species"]
            n_changed_field["species"] += 1

        if not payload:
            n_skip += 1
            continue

        point_id = chunk_to_point.get(chunk_id)
        if point_id is None:
            not_found.append(chunk_id)
            n_err += 1
            continue

        if args.dry_run:
            n_ok += 1
            if len(sample_diffs) < 5:
                sample_diffs.append(f"  {sku} ({chunk_id} → {point_id}): {old} → {payload}")
        else:
            try:
                client.set_payload(
                    collection_name=collection,
                    payload=payload,
                    points=[point_id],
                )
                n_ok += 1
            except Exception as e:
                write_errors.append((chunk_id, f"{type(e).__name__}: {e}"))
                n_err += 1

        if (i + 1) % 50 == 0 or i + 1 == len(corrections):
            rate = (i + 1) / max(0.001, time.perf_counter() - t0)
            print(f"  …{i + 1}/{len(corrections)}  ok={n_ok}  err={n_err}  ({rate:.1f}/s)")

    elapsed = time.perf_counter() - t0
    print()
    print(f"=== summary ===")
    print(f"applied:        {n_ok}")
    print(f"errors:         {n_err}")
    print(f"no-op skipped:  {n_skip}")
    print(f"field changes:  category={n_changed_field['category']}  subcategory={n_changed_field['subcategory']}  species={n_changed_field['species']}")
    print(f"elapsed:        {elapsed:.1f}s")

    if sample_diffs:
        print("\nsample diffs (dry-run):")
        for d in sample_diffs:
            print(d)

    if not_found:
        print(f"\nchunk_ids NOT FOUND in Qdrant ({len(not_found)}, first 10):")
        for cid in not_found[:10]:
            print(f"  {cid}")

    if write_errors:
        print(f"\nwrite errors (first 10):")
        for cid, msg in write_errors[:10]:
            print(f"  {cid}: {msg}")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
