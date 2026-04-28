"""
LLM-assisted attribute correction for the cleaned product catalog.

Reads the master xlsx, sends each product (multilingual names + current
attributes) to an LLM, and writes a normalized JSONL with verified
category/subcategory/species and regenerated soft_text. Original fields are
preserved; only the four target fields ever change.

Usage:
    python3 ingestion/llm_correct_attributes.py
    # options:
    python3 ingestion/llm_correct_attributes.py \
        --xlsx web/public/GLORIA_2026_CLEANED.corrected.v8.xlsx \
        --sheet products \
        --out data/products_corrected.jsonl \
        --concurrency 12 \
        --limit 50

The script is idempotent: every LLM response is cached at
data/llm_correction_cache.json by SKU. Re-running picks up where it left
off and only calls the LLM for new SKUs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_project_env  # noqa: E402

load_project_env()


# --------------------------------------------------------------------------
# Pooled OpenRouter client (self-contained — avoids backend/* coupling).
# --------------------------------------------------------------------------
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_HTTP = httpx.Client(
    timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
)


def post_openrouter_chat(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "ragbot-poc",
        "X-Title": "ragbot-poc",
        "Content-Type": "application/json",
    }
    r = _HTTP.post(_OPENROUTER_URL, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
ALLOWED_CATEGORIES = [
    "grooming", "accessories", "nutrition", "hygiene", "training",
    "toys", "housing", "healthcare", "apparel", "equipment",
]
ALLOWED_SPECIES = ["dog", "cat", "bird", "rabbit", "ferret", "rodent", "reptile", "horse"]
DEFAULT_MODEL = os.getenv("OPENROUTER_CORRECTOR_MODEL", "openai/gpt-4o-mini")

DEFAULT_XLSX = ROOT / "web/public/GLORIA_2026_CLEANED.corrected.v8.xlsx"
DEFAULT_OUT = ROOT / "data/products_corrected.jsonl"
DEFAULT_CACHE = ROOT / "data/llm_correction_cache.json"
DEFAULT_LOG = ROOT / "data/llm_corrections_log.json"


# --------------------------------------------------------------------------
# Excel → records
# --------------------------------------------------------------------------
NAME_FIELDS = ["name_es", "name_en", "name_fr", "name_pt", "name_it"]
CARRY_FIELDS = [
    "id", "brand", "sku", "ean", "price_pvpr", "price_per_unit", "min_purchase_qty",
    "size_label", "size_raw", "color", "scent", "dureza", "tipo", "capacity_raw",
    "neck_min_cm", "neck_max_cm", "body_min_cm", "body_max_cm",
    "chest_min_cm", "chest_max_cm",
    "dog_weight_min_kg", "dog_weight_max_kg",
    "cat_weight_min_kg", "cat_weight_max_kg",
    "length_cm", "width_cm", "height_cm", "depth_cm", "thickness_cm",
    "leash_length_m", "weight_g", "watts", "hz", "cut_length_mm",
    "recambio", "breed_suitability", "change_flag",
    "photo_ref", "observaciones", "estado", "fecha",
]


def read_products(xlsx_path: Path, sheet: str) -> list[dict[str, Any]]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = ws.iter_rows(values_only=True)
    headers = [str(h).strip() if h is not None else "" for h in next(rows)]

    def at(row: tuple[Any, ...], name: str) -> Any:
        try:
            return row[headers.index(name)]
        except ValueError:
            return None

    out: list[dict[str, Any]] = []
    for row in rows:
        if not any(c is not None for c in row):
            continue
        sku = at(row, "sku")
        if sku is None or str(sku).strip() == "":
            continue
        rec: dict[str, Any] = {f: at(row, f) for f in CARRY_FIELDS}
        rec["category"] = at(row, "category")
        rec["subcategory"] = at(row, "subcategory")
        rec["species"] = at(row, "species")
        rec["soft_text"] = at(row, "soft_text")
        for nf in NAME_FIELDS:
            rec[nf] = at(row, nf)
        raw = at(row, "raw_attributes_json")
        try:
            rec["raw_attributes"] = json.loads(raw) if isinstance(raw, str) and raw else {}
        except Exception:
            rec["raw_attributes"] = {}
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# LLM correction
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You verify and correct product attributes for a pet-supplies wholesale "
    "catalog (Gloriapets). For each product you receive its multilingual names "
    "and current attributes. Your job:\n"
    "1. Decide if `category` is correct given the names. Allowed values: "
    + ", ".join(ALLOWED_CATEGORIES) + ".\n"
    "2. Decide if `subcategory` is plausible (free-form, lowercase, snake_case "
    "preferred — e.g. 'spray', 'collar', 'chew_toy').\n"
    "3. Decide if `species` is correct. Use a pipe-joined list from these tokens: "
    + ", ".join(ALLOWED_SPECIES) + ". Examples: 'dog', 'dog|cat', 'cat|bird'.\n"
    "4. Regenerate `soft_text` in English: 1–2 sentences covering brand, what "
    "the product is, key attribute (size/color/capacity if known), and species. "
    "Aim for ~120-200 chars. Search-friendly, no marketing fluff.\n\n"
    "Be conservative — change a field ONLY if the names clearly contradict it. "
    "Always regenerate `soft_text` (it gets indexed for search).\n\n"
    "Return ONLY valid JSON with these keys (no prose, no markdown):\n"
    '{"category": str, "subcategory": str, "species": str, '
    '"soft_text": str, "changed": [str], "reason": str}'
)


def build_user_prompt(rec: dict[str, Any]) -> str:
    names = {f: rec.get(f) for f in NAME_FIELDS if rec.get(f)}
    extras = {k: rec.get(k) for k in (
        "color", "scent", "size_label", "capacity_raw", "weight_g",
        "length_cm", "width_cm", "height_cm",
    ) if rec.get(k) not in (None, "")}
    payload = {
        "sku": rec.get("sku"),
        "brand": rec.get("brand"),
        "names": names,
        "current_category": rec.get("category"),
        "current_subcategory": rec.get("subcategory"),
        "current_species": rec.get("species"),
        "current_soft_text": rec.get("soft_text"),
        "physical_attributes": extras,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm(rec: dict[str, Any], *, model: str, max_retries: int = 3) -> dict[str, Any]:
    user = build_user_prompt(rec)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            res = post_openrouter_chat({
                "model": model,
                "messages": msgs,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })
            content = (res.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            if not content.strip():
                raise RuntimeError("empty LLM response")
            data = json.loads(content)
            return data
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"LLM failed after {max_retries} retries: {last_err}")


# --------------------------------------------------------------------------
# Output assembly
# --------------------------------------------------------------------------
def _is_clean_string(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def merge_correction(rec: dict[str, Any], llm: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Apply LLM corrections conservatively. Track which fields actually changed.
    """
    changed: list[str] = []
    out = dict(rec)

    # category — only accept if it's in the allowed enum
    new_cat = llm.get("category")
    if _is_clean_string(new_cat) and new_cat in ALLOWED_CATEGORIES and new_cat != rec.get("category"):
        out["category"] = new_cat
        changed.append("category")

    # subcategory — accept any plausible string
    new_sub = llm.get("subcategory")
    if _is_clean_string(new_sub) and new_sub != rec.get("subcategory"):
        out["subcategory"] = new_sub
        changed.append("subcategory")

    # species — accept if every token is in allowed set
    new_sp = llm.get("species")
    if _is_clean_string(new_sp):
        tokens = [t.strip() for t in new_sp.split("|") if t.strip()]
        if tokens and all(t in ALLOWED_SPECIES for t in tokens):
            normalized = "|".join(tokens)
            if normalized != rec.get("species"):
                out["species"] = normalized
                changed.append("species")

    # soft_text — always overwrite if the LLM produced something usable
    new_st = llm.get("soft_text")
    if _is_clean_string(new_st):
        if new_st.strip() != (rec.get("soft_text") or "").strip():
            out["soft_text"] = new_st.strip()
            changed.append("soft_text")
    return out, changed


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--sheet", default="products")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="process first N rows (0=all)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM, just emit JSONL of cleaned source rows")
    args = ap.parse_args()

    print(f"[xlsx] reading {args.xlsx}")
    products = read_products(args.xlsx, args.sheet)
    print(f"       loaded {len(products)} rows")
    if args.limit:
        products = products[: args.limit]

    cache = load_cache(args.cache)
    print(f"[cache] {len(cache)} cached LLM responses at {args.cache}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    todo = [p for p in products if str(p.get("sku") or "") not in cache]
    print(f"[plan] todo={len(todo)} cached={len(products) - len(todo)} model={args.model}")

    failures: list[dict[str, Any]] = []
    t0 = time.time()

    if args.no_llm:
        print("[llm] --no-llm set; skipping LLM calls")
    elif todo:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(call_llm, p, model=args.model): p for p in todo}
            for fut in as_completed(futs):
                p = futs[fut]
                sku = str(p.get("sku") or "")
                try:
                    llm_out = fut.result()
                    cache[sku] = llm_out
                    completed += 1
                except Exception as e:
                    failures.append({"sku": sku, "error": str(e)})
                if completed and completed % 50 == 0:
                    save_cache(args.cache, cache)
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - completed) / rate if rate > 0 else 0
                    print(f"  [{completed}/{len(todo)}] rate={rate:.1f}/s eta={eta:.0f}s "
                          f"failures={len(failures)}")
        save_cache(args.cache, cache)

    # Assemble JSONL
    print(f"[write] {args.out}")
    changes_summary: dict[str, int] = {"category": 0, "subcategory": 0, "species": 0, "soft_text": 0}
    total_changed_rows = 0
    log_rows: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8") as fout:
        for rec in products:
            sku = str(rec.get("sku") or "")
            llm = cache.get(sku) if not args.no_llm else None
            if llm:
                merged, changed = merge_correction(rec, llm)
                if changed:
                    total_changed_rows += 1
                    for c in changed:
                        changes_summary[c] = changes_summary.get(c, 0) + 1
                    log_rows.append({
                        "sku": sku,
                        "changed": changed,
                        "reason": llm.get("reason"),
                        "before": {k: rec.get(k) for k in changed},
                        "after": {k: merged.get(k) for k in changed},
                    })
            else:
                merged = dict(rec)
                changed = []
            # Drop helper keys we don't want emitted, then write.
            line = {
                "id": merged.get("id"),
                "sku": merged.get("sku"),
                "brand": merged.get("brand"),
                "ean": merged.get("ean"),
                "category": merged.get("category"),
                "subcategory": merged.get("subcategory"),
                "species": merged.get("species"),
                "names": {nf: merged.get(nf) for nf in NAME_FIELDS if merged.get(nf)},
                "soft_text": merged.get("soft_text"),
                "attributes": {
                    k: merged.get(k) for k in CARRY_FIELDS
                    if k not in {"id", "sku", "brand", "ean"} and merged.get(k) not in (None, "")
                },
                "raw_attributes": merged.get("raw_attributes") or {},
                "_corrections": {"changed": changed} if changed else {},
            }
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps({
        "model": args.model,
        "input_rows": len(products),
        "rows_changed": total_changed_rows,
        "field_change_counts": changes_summary,
        "failures": failures,
        "elapsed_sec": round(time.time() - t0, 1),
        "examples": log_rows[:50],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("================ Summary ================")
    print(f"  Input rows               : {len(products)}")
    print(f"  Rows with corrections    : {total_changed_rows}")
    print(f"  Field change counts      : {changes_summary}")
    print(f"  LLM failures             : {len(failures)}")
    print(f"  Output JSONL             : {args.out}")
    print(f"  Cache (SKU → LLM resp)   : {args.cache}")
    print(f"  Detailed log             : {args.log}")
    print(f"  Elapsed                  : {round(time.time() - t0, 1)}s")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
