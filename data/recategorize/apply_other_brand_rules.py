"""
Species-narrowing rules for CEVA / CHIEN CHIC / COCOSI.

CEVA: Adaptil = DOG, Feliway = CAT, Vectra 3D = DOG, Vectra Felis = CAT.
CHIEN CHIC: 'Chien' (French for dog) — all DOG.
COCOSI: dog food brand; products without 'para gato' in name are DOG; tofu/cereal litters are CAT.
"""
from __future__ import annotations

import json
from pathlib import Path

REMAINING = Path("/Users/gautham/Downloads/ragbot-poc/data/recategorize/remaining_for_review.jsonl")
CORRECTIONS = Path("/Users/gautham/Downloads/ragbot-poc/data/recategorize/manual_corrections.jsonl")


def classify(brand: str, sku: str, name: str) -> tuple[list[str] | None, float, str]:
    nl = name.lower()

    if brand == "CEVA":
        if sku.startswith("ADAP"):
            return (["dog"], 0.95, "Adaptil — dog appeasing pheromone product line.")
        if sku.startswith("FELI"):
            return (["cat"], 0.95, "Feliway — cat facial pheromone product line.")
        # Vectra already correct.
        return (None, 0.0, "")

    if brand == "CHIEN CHIC":
        # 'Chien' = French for dog; brand is exclusively dog perfumes.
        return (["dog"], 0.9, "Chien Chic — French dog-perfume brand.")

    if brand == "COCOSI":
        if "para gato" in nl or "gatos" in nl:
            return (["cat"], 0.9, "COCOSI cat product — name says 'para gato(s)'.")
        if "para perro" in nl or " perros" in nl or "para puppy" in nl:
            return (["dog"], 0.9, "COCOSI dog product — name says 'para perro' or 'puppy'.")
        # Litter products for cats (no species in name).
        if "arena" in nl or "lecho" in nl or "tofu" in nl:
            return (["cat"], 0.85, "COCOSI litter (arena/tofu/lecho) — cat-only.")
        # Default: COCOSI is a Spanish dog wet-food brand; without 'gato' marker, infer dog.
        if any(t in nl for t in ("trocitos", "tarrina", "albóndigas", "lata", "estofado",
                                  "guiso", "ragú", "marmitako", "cocido", "salchich", "real food")):
            return (["dog"], 0.8, "COCOSI wet-food line — dog-only by brand convention; no 'gato' marker in name.")
        return (None, 0.0, "")

    return (None, 0.0, "")


def main() -> int:
    rows = [json.loads(l) for l in REMAINING.read_text().splitlines() if l.strip()]
    target = [r for r in rows if r.get("brand") in {"CEVA", "CHIEN CHIC", "COCOSI"}]

    corrections: list[dict] = []

    for r in target:
        sku = r.get("sku") or ""
        name = r.get("product_name") or ""
        brand = r.get("brand")
        old = {
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "species": r.get("species") or [],
        }

        new_species, conf, reason = classify(brand, sku, name)
        if new_species is None:
            continue
        if set(old["species"]) == set(new_species):
            continue

        corrections.append({
            "sku": sku,
            "chunk_id": r.get("chunk_id"),
            "product_name": name,
            "old": old,
            "new": {
                "category": old["category"],
                "subcategory": old["subcategory"],
                "species": new_species,
            },
            "reasoning": reason,
            "confidence": conf,
        })

    with CORRECTIONS.open("a") as f:
        for c in corrections:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"target rows: {len(target)}")
    print(f"  corrections: {len(corrections)}")
    by_brand: dict[str, int] = {}
    for c in corrections:
        # extract brand by chunk_id prefix or product
        bn = c["product_name"]
        brand = c.get("brand") or ""
        # we don't carry brand on corrections; just count by reasoning prefix
        key = c["reasoning"].split("—")[0].strip()
        by_brand[key] = by_brand.get(key, 0) + 1
    for k, n in by_brand.items():
        print(f"  {n:4d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
