"""
Build a {sku -> {pages: [int], primary_page: int}} mapping from the OCR'd PDF
catalog pages in data/pages/.

- Source: data/pages/page_*.json (each JSON has page_number + text)
- Whitelist of valid SKUs: products_normalized.jsonl
- Output: data/index/sku_pages.json

Run locally after a new PDF catalog is OCR'd. Used by:
  - indexing/build_index_from_excel.py    (baked into new full ingests)
  - indexing/patch_qdrant_pages.py        (patches existing collection without re-embed)

Usage:
    python indexing/build_sku_page_map.py
    # optional:
    python indexing/build_sku_page_map.py \
        --pages-dir data/pages \
        --jsonl /Users/dias/Documents/products_normalized.jsonl \
        --out data/index/sku_pages.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# Broad SKU candidate pattern — letters + digits, 5+ chars, may contain
# punctuation used by the catalog. We then intersect with the Excel SKU set
# to eliminate garbage tokens.
_SKU_CANDIDATE_RE = re.compile(r"\b[A-Z][A-Z0-9][A-Z0-9*/.\-]{3,}\b")


def load_sku_whitelist(jsonl_path: Path) -> set[str]:
    skus: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sku = (rec.get("payload") or {}).get("sku")
            if sku:
                skus.add(str(sku).strip().upper())
    return skus


def extract_candidates(text: str) -> set[str]:
    return {m.group(0).upper() for m in _SKU_CANDIDATE_RE.finditer(text or "")}


def build_map(pages_dir: Path, sku_whitelist: set[str]) -> dict[str, dict]:
    """Return {sku: {"pages": sorted_list, "primary_page": earliest}}."""
    by_sku: dict[str, set[int]] = defaultdict(set)

    page_files = sorted(pages_dir.glob("page_*.json"))
    if not page_files:
        print(f"WARNING: no page_*.json found in {pages_dir}", file=sys.stderr)

    for pf in page_files:
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARNING: could not parse {pf.name}: {e}", file=sys.stderr)
            continue

        page_num = data.get("page_number")
        if not isinstance(page_num, int):
            # Fall back to parsing the filename: page_0072.json -> 72
            m = re.match(r"page_(\d+)\.json$", pf.name)
            page_num = int(m.group(1)) if m else None
        if page_num is None:
            continue

        text = data.get("text") or ""
        candidates = extract_candidates(text)
        hits = candidates & sku_whitelist
        for sku in hits:
            by_sku[sku].add(int(page_num))

    result: dict[str, dict] = {}
    for sku, pages in by_sku.items():
        sorted_pages = sorted(pages)
        result[sku] = {
            "pages": sorted_pages,
            "primary_page": sorted_pages[0],
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-dir", type=Path, default=Path("data/pages"))
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=Path("/Users/dias/Documents/products_normalized.jsonl"),
    )
    ap.add_argument("--out", type=Path, default=Path("data/index/sku_pages.json"))
    args = ap.parse_args()

    if not args.pages_dir.is_dir():
        print(f"ERROR: pages dir not found: {args.pages_dir}", file=sys.stderr)
        return 1
    if not args.jsonl.is_file():
        print(f"ERROR: jsonl not found: {args.jsonl}", file=sys.stderr)
        return 1

    sku_whitelist = load_sku_whitelist(args.jsonl)
    print(f"Loaded {len(sku_whitelist)} unique SKUs from {args.jsonl.name}")

    mapping = build_map(args.pages_dir, sku_whitelist)
    print(f"Matched {len(mapping)} / {len(sku_whitelist)} SKUs to at least one page")

    # Coverage stats
    one_page = sum(1 for v in mapping.values() if len(v["pages"]) == 1)
    multi_page = sum(1 for v in mapping.values() if len(v["pages"]) > 1)
    max_pages = max((len(v["pages"]) for v in mapping.values()), default=0)
    print(f"  single-page SKUs: {one_page}")
    print(f"  multi-page SKUs:  {multi_page} (max occurrences: {max_pages})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")

    # Show 3 samples so it's visually obvious the mapping is sensible.
    samples = list(mapping.items())[:3]
    for sku, v in samples:
        print(f"  sample: {sku} -> primary_page={v['primary_page']} total={len(v['pages'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
