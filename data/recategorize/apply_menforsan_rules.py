"""
MEN FOR SAN species correction — leverages the SKU prefix.

  MFP*  → perro  (dog)
  MFG*  → gato   (cat)
  MFE*  → equino (horse — proposes NEW species)
  MFA*  → ave    (bird)
  MFR*  → roedor (rodent — NEW)
  MFH*  → hurón/conejo (ferret/rabbit — NEW)
  MFL*  → limpieza products (keep dual [dog,cat] unless name says otherwise)
  SNM*  → animal-feed line, species inferred from product name
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REMAINING = Path("/Users/gautham/Downloads/ragbot-poc/data/recategorize/remaining_for_review.jsonl")
CORRECTIONS = Path("/Users/gautham/Downloads/ragbot-poc/data/recategorize/manual_corrections.jsonl")
LOW_CONF = Path("/Users/gautham/Downloads/ragbot-poc/data/recategorize/low_confidence.jsonl")

NEW_SPECIES = {"horse", "rodent", "rabbit", "ferret"}


def infer_species(sku: str, name: str) -> tuple[list[str] | None, float, str]:
    """Returns (species_list_or_None, confidence, reasoning)."""
    nl = name.lower()
    # Name-driven first — most reliable.
    has_dog = any(t in nl for t in ("para perros", "perros y gatos", "para perro y gato", " perros"))
    has_cat = any(t in nl for t in ("para gatos", "perros y gatos", "para perro y gato", " gatos"))
    has_horse = any(t in nl for t in ("caballos", "para caballo", "equinos", "para equino"))
    has_bird = any(t in nl for t in ("para aves", " aves", "para canarios", "para pájaros"))
    has_reptile = any(t in nl for t in ("para reptiles", "terrarios"))
    has_rodent = any(t in nl for t in ("para roedores", "roedores"))
    has_ferret = "hurones" in nl or "hurón" in nl
    has_rabbit = "conejos" in nl or "conejo" in nl

    if has_dog and not has_cat and not has_horse and not has_bird and not has_reptile:
        return (["dog"], 0.9, "Name says 'perros' only.")
    if has_cat and not has_dog and not has_horse:
        return (["cat"], 0.9, "Name says 'gatos' only.")
    if has_dog and has_cat:
        return (["dog", "cat"], 0.95, "Name says 'perros y gatos'.")
    if has_horse:
        return (["horse"], 0.85, "Name mentions 'caballos' / 'equinos' — NEW species.")
    if has_bird and not has_dog and not has_cat:
        return (["bird"], 0.9, "Name mentions 'aves'.")
    if has_reptile:
        return (["reptile"], 0.9, "Name mentions 'reptiles' / 'terrarios'.")
    multi_small = []
    if has_rodent: multi_small.append("rodent")
    if has_ferret: multi_small.append("ferret")
    if has_rabbit: multi_small.append("rabbit")
    if multi_small:
        return (multi_small, 0.8, f"Name mentions small mammals: {multi_small}. NEW species.")

    # Fall back to SKU prefix when name is generic ("Champú aloe vera" — no species mentioned).
    if sku.startswith("MFP"):
        return (["dog"], 0.7, "MFP* prefix → 'perro' (dog) line in MEN FOR SAN's SKU scheme.")
    if sku.startswith("MFG"):
        return (["cat"], 0.7, "MFG* prefix → 'gato' (cat) line.")
    if sku.startswith("MFE"):
        return (["horse"], 0.7, "MFE* prefix → 'equino' (horse) line. NEW species.")
    if sku.startswith("MFA"):
        return (["bird"], 0.7, "MFA* prefix → 'aves' (bird) line.")
    if sku.startswith("MFR"):
        return (["rodent"], 0.65, "MFR* prefix → 'roedores' (rodent) line. NEW species.")
    if sku.startswith("MFH"):
        return (["ferret", "rabbit", "rodent"], 0.6, "MFH* prefix → small mammals (ferret/rabbit/rodent). NEW species.")
    if sku.startswith("MFL"):
        return (["dog", "cat"], 0.6, "MFL* prefix → cleaning/odor products, dual species.")
    return (None, 0.0, "")


def main() -> int:
    rows = [json.loads(l) for l in REMAINING.read_text().splitlines() if l.strip()]
    target = [r for r in rows if r.get("brand") == "MEN FOR SAN"]

    corrections: list[dict] = []
    low_conf: list[dict] = []

    for r in target:
        sku = r.get("sku") or ""
        name = r.get("product_name") or ""
        old = {
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "species": r.get("species") or [],
        }

        new_species, conf, reason = infer_species(sku, name)
        if new_species is None:
            continue

        # Skip when no actual change.
        if set(old["species"]) == set(new_species):
            continue

        introduces_new = sorted(set(new_species) & NEW_SPECIES)

        new = {
            "category": old["category"],
            "subcategory": old["subcategory"],
            "species": new_species,
        }
        row = {
            "sku": sku,
            "chunk_id": r.get("chunk_id"),
            "product_name": name,
            "old": old,
            "new": new,
            "reasoning": reason,
            "confidence": conf,
        }
        if introduces_new:
            row["introduces_new_species"] = introduces_new

        if conf >= 0.7:
            corrections.append(row)
        else:
            row["needs_web_check"] = True
            low_conf.append(row)

    with CORRECTIONS.open("a") as f:
        for c in corrections:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with LOW_CONF.open("a") as f:
        for c in low_conf:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"MEN FOR SAN remaining: {len(target)}")
    print(f"  corrections written:  {len(corrections)}")
    print(f"  low-confidence:       {len(low_conf)}")
    print(f"  no change / no infer: {len(target) - len(corrections) - len(low_conf)}")

    new_species_counts: dict[str, int] = {}
    for c in corrections + low_conf:
        for sp in c["new"]["species"]:
            if sp in NEW_SPECIES:
                new_species_counts[sp] = new_species_counts.get(sp, 0) + 1
    if new_species_counts:
        print(f"\nNew species values introduced (count):")
        for sp, n in sorted(new_species_counts.items()):
            print(f"  {sp:10s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
