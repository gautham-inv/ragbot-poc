"""
Re-classify category / subcategory / species for every product in Qdrant
using Claude (via OpenRouter), and write three review files:

  data/recategorize/corrections.jsonl      — confidence >= threshold and labels changed
  data/recategorize/low_confidence.jsonl   — confidence < threshold (need web research)
  data/recategorize/new_taxonomy.json      — values the LLM wanted to add to the vocab
  data/recategorize/unchanged.jsonl        — LLM agrees with current labels (audit log)
  data/recategorize/summary.txt            — counts + top-changing buckets

This script does NOT modify Qdrant. After review, the corrections are merged
into the source JSONL and re-ingested via indexing/build_index_from_excel.py.

Usage:
  # Pilot run (validates prompt + cost on a small random sample)
  python ingestion/llm_recategorize.py --sample 30

  # Full run
  python ingestion/llm_recategorize.py

Env vars:
  OPENROUTER_API_KEY    required
  RECAT_MODEL           default: anthropic/claude-haiku-4-5
  RECAT_RETRY_MODEL     default: anthropic/claude-sonnet-4-6 (used for low-conf rows)
  RECAT_THRESHOLD       default: 0.85
  RECAT_WORKERS         default: 8
  QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION as in the rest of the project
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient

# --- Catalog-aligned controlled vocabulary --------------------------------
# Mirror of the values in backend/app.py's vision prompt. Keep in sync.

CATEGORIES = [
    "accessories", "toys", "grooming", "nutrition", "healthcare",
    "hygiene", "training", "housing", "apparel", "equipment",
]

SUBCATEGORIES_BY_CATEGORY: dict[str, list[str]] = {
    "accessories": ["collar", "harness", "leash", "muzzle", "bowl", "water_dispenser",
                    "dispenser", "doormat", "seat_cover", "scoop", "pouch", "carrier",
                    "leash_coupler", "food_bag", "car_restraint", "reflector",
                    "grooming_mitt", "travel_mat", "car_barrier", "training_line", "refill"],
    "toys":        ["chew_toy", "ball", "toy", "cat_toy", "plush", "lick_mat",
                    "chew_toy_pack", "catnip", "play_mat"],
    "grooming":    ["shampoo", "fragrance", "brush", "balm", "conditioner", "scissors",
                    "nail_clipper", "grooming_mitt", "spray", "mousse", "deshedder",
                    "ear_care", "lotion", "towel", "flea_comb", "cleaning",
                    "antistatic_spray", "hoof_care"],
    "nutrition":   ["dental_chew", "wet_food", "treat", "yogurt", "ice_cream",
                    "edible_chew", "wet_treat", "milk", "kefir"],
    "healthcare":  ["parasiticide", "pheromone", "repellent", "skin_care", "dental_care",
                    "recovery_shirt", "calming", "supplement", "antiparasitic",
                    "eye_care", "wound_care", "tick_remover", "liniment"],
    "hygiene":     ["wipes", "poop_bags", "litter", "cleaner", "diaper", "training_pad",
                    "odor_eliminator", "litter_filter", "stain_remover", "detergent"],
    "training":    ["tug_toy", "bite_pillow", "dumbbell", "tug", "ball", "clicker",
                    "treat_dummy", "whistle", "treat_bag", "behavior_corrector",
                    "reward_toy", "deterrent", "dummy", "control_rod", "training_aid",
                    "training_bells", "target_stick"],
    "housing":     ["bed", "litter_box", "mattress", "carrier"],
    "apparel":     ["coat"],
    "equipment":   ["blade", "clipper"],
}

SPECIES = ["dog", "cat", "reptile", "bird", "rabbit", "horse", "rodent", "ferret"]

# --- Prompt ---------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a senior pet-product taxonomist auditing an e-commerce catalog.

Your job: given one product's name + brand + free-text description, decide the
correct CATEGORY, SUBCATEGORY, and SPECIES. The catalog already has labels but
they are unreliable (e.g. KONG plush squeak toys mislabeled as "ball"; cat-only
toys mislabeled as dual species).

Controlled vocabulary (use these whenever they fit):

CATEGORIES (10): {', '.join(CATEGORIES)}

SUBCATEGORIES per category:
{chr(10).join(f"  {cat}: {', '.join(subs)}" for cat, subs in SUBCATEGORIES_BY_CATEGORY.items())}

SPECIES (allowed): {', '.join(SPECIES)}

Rules:
1. Always prefer an existing vocab value over inventing a new one.
2. Only propose a new category/subcategory/species when NOTHING in the list
   reasonably fits. If you do, set the matching `is_new_*` flag to true.
   New values must be lowercase English snake_case slugs.
3. The current labels in the catalog are unreliable — do not anchor on them.
   Decide from the product name + description.
4. Subcategory must be specific to the actual product type:
   - Plush / soft / squeaky stuffed toys → "plush" (not "ball" or "toy")
   - Rubber chewing toys → "chew_toy"
   - Round throwable toys → "ball"
   - Generic "toy" should be a last resort.
5. Species must list every animal the product is genuinely intended for.
   Cat-only product lines (Cat Softies, Kickeroo Cuddler, cat toys) → ["cat"].
   Dog-only product lines → ["dog"]. Don't default to ["dog","cat"] for everything.
6. Confidence: 0.0-1.0, your honest read on whether the product name+description
   uniquely identifies the type. Vague names ("KONG toy assorted") get LOW
   confidence; clearly-named products ("KONG Ballistic Vibez Plush Llama") get HIGH.

Return ONLY a JSON object with these EXACT keys:
{{
  "category": "<one of the 10, or new slug>",
  "subcategory": "<from the per-category list, or new slug>",
  "species": ["<list of species>"],
  "confidence": 0.0,
  "reasoning": "<one short sentence — what about the product name decided this>",
  "is_new_category": false,
  "is_new_subcategory": false,
  "is_new_species": false
}}
No prose, no markdown fences."""

# --- OpenRouter call ------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _classify_one(client: httpx.Client, model: str, product: dict[str, Any], api_key: str) -> dict[str, Any]:
    user_block = json.dumps({
        "sku": product.get("sku") or "",
        "brand": product.get("brand") or "",
        "product_name": product.get("product_name") or product.get("name") or "",
        "current_category": product.get("category") or "",
        "current_subcategory": product.get("subcategory") or "",
        "current_species": product.get("species") or [],
        "description_snippet": (product.get("text") or product.get("soft_text") or "")[:1200],
    }, ensure_ascii=False, indent=2)

    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Product:\n{user_block}\n\nReturn the JSON now."},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "ragbot-poc",
        "X-Title": "ragbot-recategorize",
    }
    r = client.post(OPENROUTER_URL, headers=headers, json=payload)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"] or "{}"
    if raw.strip().startswith("```"):
        # Defensive: strip code fences if a model wraps despite json_object mode.
        raw = raw.strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


# --- Qdrant pull ----------------------------------------------------------

def load_products(qclient: QdrantClient, collection: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
            out.append({
                "point_id": str(p.id),
                "chunk_id": pl.get("chunk_id"),
                "sku": pl.get("sku"),
                "brand": pl.get("brand"),
                "product_name": pl.get("product_name") or pl.get("name"),
                "category": pl.get("category"),
                "subcategory": pl.get("subcategory"),
                "species": pl.get("species") or [],
                "text": pl.get("text") or pl.get("soft_text") or "",
            })
        if nxt is None:
            break
    return out


# --- Main -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, classify only N random products (pilot mode).")
    ap.add_argument("--threshold", type=float,
                    default=float(os.getenv("RECAT_THRESHOLD", "0.85")))
    ap.add_argument("--workers", type=int,
                    default=int(os.getenv("RECAT_WORKERS", "8")))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/recategorize"))
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2
    model = os.getenv("RECAT_MODEL", "anthropic/claude-haiku-4-5")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "excel_dataset")
    if not qdrant_url:
        print("QDRANT_URL not set", file=sys.stderr)
        return 2
    qclient = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    print(f"[recat] loading products from {collection}…", flush=True)
    products = load_products(qclient, collection)
    print(f"[recat] loaded {len(products)} product rows", flush=True)

    if args.sample > 0:
        random.seed(42)
        products = random.sample(products, min(args.sample, len(products)))
        print(f"[recat] PILOT: classifying {len(products)} random products", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    corrections_path = args.out_dir / "corrections.jsonl"
    low_conf_path = args.out_dir / "low_confidence.jsonl"
    unchanged_path = args.out_dir / "unchanged.jsonl"
    new_taxonomy_path = args.out_dir / "new_taxonomy.json"
    summary_path = args.out_dir / "summary.txt"
    failures_path = args.out_dir / "failures.jsonl"

    classified: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    def task(p: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
        with httpx.Client(timeout=60.0) as c:
            for attempt in range(3):
                try:
                    res = _classify_one(c, model, p, api_key)
                    return p, res, None
                except (httpx.HTTPError, json.JSONDecodeError) as e:
                    if attempt == 2:
                        return p, None, f"{type(e).__name__}: {e}"
                    time.sleep(1.5 * (attempt + 1))
        return p, None, "unreachable"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(task, p) for p in products]
        for i, fut in enumerate(as_completed(futures), 1):
            p, res, err = fut.result()
            if err or res is None:
                failures.append({"product": p, "error": err})
            else:
                classified.append((p, res))
            if i % 50 == 0 or i == len(products):
                rate = i / max(1.0, time.perf_counter() - t_start)
                print(f"[recat] {i}/{len(products)} ({rate:.1f}/s)", flush=True)

    # --- Sort outputs into buckets ---
    def changed(old: dict[str, Any], new: dict[str, Any]) -> bool:
        return (
            (old.get("category") or "") != (new.get("category") or "")
            or (old.get("subcategory") or "") != (new.get("subcategory") or "")
            or set(old.get("species") or []) != set(new.get("species") or [])
        )

    new_cats: dict[str, list[str]] = {}
    new_subs: dict[str, list[str]] = {}
    new_species: dict[str, list[str]] = {}

    n_corr = n_low = n_same = 0
    with corrections_path.open("w") as f_corr, \
         low_conf_path.open("w") as f_low, \
         unchanged_path.open("w") as f_un:
        for p, res in classified:
            conf = float(res.get("confidence") or 0.0)
            row = {
                "sku": p.get("sku"),
                "brand": p.get("brand"),
                "product_name": p.get("product_name"),
                "old": {"category": p.get("category"), "subcategory": p.get("subcategory"), "species": p.get("species")},
                "new": {"category": res.get("category"), "subcategory": res.get("subcategory"), "species": res.get("species")},
                "confidence": conf,
                "reasoning": res.get("reasoning"),
                "is_new_category": bool(res.get("is_new_category")),
                "is_new_subcategory": bool(res.get("is_new_subcategory")),
                "is_new_species": bool(res.get("is_new_species")),
                "point_id": p.get("point_id"),
                "chunk_id": p.get("chunk_id"),
            }
            if res.get("is_new_category"):
                new_cats.setdefault(res.get("category") or "?", []).append(p.get("sku") or "")
            if res.get("is_new_subcategory"):
                new_subs.setdefault(res.get("subcategory") or "?", []).append(p.get("sku") or "")
            if res.get("is_new_species"):
                for sp in (res.get("species") or []):
                    if sp not in SPECIES:
                        new_species.setdefault(sp, []).append(p.get("sku") or "")

            if conf < args.threshold:
                f_low.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_low += 1
            elif changed(p, res):
                f_corr.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_corr += 1
            else:
                f_un.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_same += 1

    if failures:
        with failures_path.open("w") as f:
            for x in failures:
                f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")

    new_taxonomy = {"categories": new_cats, "subcategories": new_subs, "species": new_species}
    with new_taxonomy_path.open("w") as f:
        json.dump(new_taxonomy, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - t_start
    summary = (
        f"Re-categorization summary\n"
        f"=========================\n"
        f"model:        {model}\n"
        f"threshold:    {args.threshold}\n"
        f"workers:      {args.workers}\n"
        f"products:     {len(products)}\n"
        f"classified:   {len(classified)}\n"
        f"failures:     {len(failures)}\n"
        f"  corrections (confident, label-change):   {n_corr}\n"
        f"  unchanged   (confident, agrees w/ old):  {n_same}\n"
        f"  low_confidence (needs web research):     {n_low}\n"
        f"  proposed new categories:                 {len(new_cats)}\n"
        f"  proposed new subcategories:              {len(new_subs)}\n"
        f"  proposed new species:                    {len(new_species)}\n"
        f"elapsed:      {elapsed:.1f}s\n"
        f"output dir:   {args.out_dir}\n"
    )
    summary_path.write_text(summary)
    print("\n" + summary, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
