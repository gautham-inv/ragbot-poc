"""
Merge corrected category/subcategory/species/soft_text from
data/products_corrected.jsonl into the master products_normalized.jsonl
that feeds indexing/build_index_from_excel.py.

- Matches by `id`
- Converts pipe-joined species ("dog|cat") back to a list (["dog", "cat"])
- Writes corrected fields into BOTH the top-level row and row["payload"]
  (the indexer reads from row["payload"]; soft_text is top-level)
- Writes to <input>.corrected.jsonl by default, leaves the source untouched
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NORMALIZED = Path("/Users/dias/Documents/products_normalized.jsonl")
DEFAULT_CORRECTIONS = ROOT / "data/corrections.jsonl"

CORRECTABLE_FIELDS = ("category", "subcategory", "species", "soft_text")


def _normalize_species(value):
    """Accepts list, pipe-joined string, single string, or None → list[str]."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(s).strip() for s in value if str(s).strip()]
    if isinstance(value, str):
        return [s.strip() for s in value.split("|") if s.strip()]
    return [str(value)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    ap.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path. Defaults to <normalized>.corrected.jsonl")
    args = ap.parse_args()

    out_path = args.out or args.normalized.with_suffix(".corrected.jsonl")

    corrections: dict[str, dict] = {}
    with args.corrections.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            corrections[r["id"]] = r

    print(f"[corrections] loaded {len(corrections)} rows from {args.corrections}")

    n_total = 0
    n_matched = 0
    n_changed = 0
    field_changes = {f: 0 for f in CORRECTABLE_FIELDS}
    species_normalized = 0

    with args.normalized.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            row = json.loads(line)
            payload = row.setdefault("payload", {})

            # Always normalize species in payload to list (even if no correction
            # row exists — defensive against any pipe-joined leftovers).
            cur_sp = payload.get("species")
            cur_sp_list = _normalize_species(cur_sp)
            if cur_sp != cur_sp_list:
                payload["species"] = cur_sp_list
                species_normalized += 1

            row_id = row.get("id") or payload.get("id")
            corr = corrections.get(row_id)
            if not corr:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            n_matched += 1
            row_changed = False

            for field in CORRECTABLE_FIELDS:
                if field not in corr:
                    continue
                new_val = corr[field]
                if field == "species":
                    new_val = _normalize_species(new_val)
                    if payload.get("species") != new_val:
                        payload["species"] = new_val
                        field_changes["species"] += 1
                        row_changed = True
                elif field == "soft_text":
                    if row.get("soft_text") != new_val:
                        row["soft_text"] = new_val
                        field_changes["soft_text"] += 1
                        row_changed = True
                else:
                    if payload.get(field) != new_val:
                        payload[field] = new_val
                        field_changes[field] += 1
                        row_changed = True

            if row_changed:
                n_changed += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    unmatched_ids = sorted(set(corrections) - {
        json.loads(line)["id"]
        for line in args.normalized.read_text().splitlines() if line.strip()
    })

    print(f"[merge] total rows: {n_total}")
    print(f"[merge] matched corrections: {n_matched}/{len(corrections)}")
    print(f"[merge] rows changed: {n_changed}")
    print(f"[merge] field changes: {field_changes}")
    print(f"[merge] species lists normalized (separate from corrections): {species_normalized}")
    if unmatched_ids:
        print(f"[merge] WARNING: {len(unmatched_ids)} corrections had no matching id (first 10):")
        for cid in unmatched_ids[:10]:
            print(f"          {cid}")
    print(f"[merge] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
