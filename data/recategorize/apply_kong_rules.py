"""
Apply pattern-based corrections to remaining KONG toys (292 SKUs).

Each rule = (substring or regex matched against product_name, target_subcategory,
target_species, confidence, reasoning). First matching rule wins.

Output is appended to manual_corrections.jsonl. Products that don't match any
rule are skipped (kept as-is) — that's intentional; only confident families get
corrected.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REMAINING = Path("/Users/gautham/Downloads/ragbot-poc/data/recategorize/remaining_for_review.jsonl")
CORRECTIONS = Path("/Users/gautham/Downloads/ragbot-poc/data/recategorize/manual_corrections.jsonl")

# Each rule: (regex_pattern, new_subcategory, new_species, confidence, reasoning)
# Order matters — more specific first.
RULES: list[tuple[str, str | None, list[str] | None, str | None, float, str]] = [
    # ------- KONG Cat-only lines -------
    # Existing cat_toy with chew_toy mislabel — change subcategory
    (r"^KONG Cat Active Wild Tails", "cat_toy", ["cat"], "toys", 0.9, "KONG Cat Active line — cat-only."),
    (r"^KONG Cat Better Buzz Bee", "cat_toy", ["cat"], "toys", 0.9, "KONG Cat Better Buzz — cat batting toy."),
    (r"^KONG Cat Kitten Teddy Bear", "cat_toy", ["cat"], "toys", 0.9, "KONG Cat Kitten line — cat-only kitten toy."),
    (r"^KONG Cat Kitty\b", "cat_toy", ["cat"], "toys", 0.9, "KONG Cat Kitty — cat plush."),
    (r"^KONG Cat Purrsuit", "cat_toy", ["cat"], "toys", 0.9, "KONG Cat Purrsuit — cat puzzle/track toy."),
    (r"^KONG Kitten Mice", "cat_toy", ["cat"], "toys", 0.95, "Kitten mice — cat-only toy."),
    (r"^KONG Infused.*Cat", "cat_toy", ["cat"], "toys", 0.9, "KONG Infused Cat — catnip-infused cat toy."),
    (r"^KONG Naturals.*Teaser", "cat_toy", ["cat"], "toys", 0.9, "KONG Naturals Teaser — cat teaser toy."),
    (r"^KONG Play Spaces", "cat_toy", ["cat"], "toys", 0.9, "KONG Play Spaces — cat play tent/cave."),
    (r"^KONG Pull-?a-?partz", "cat_toy", ["cat"], "toys", 0.9, "KONG Pull-a-partz — cat velcro toy line."),
    (r"^KONG Pull-A-Partz sushi", "cat_toy", ["cat"], "toys", 0.9, "KONG Pull-A-Partz sushi — cat velcro toy."),
    (r"^KONG Refillables", "cat_toy", ["cat"], "toys", 0.85, "KONG Refillables — refillable catnip cat toy line."),
    (r"^KONG Scrattles", "cat_toy", ["cat"], "toys", 0.9, "KONG Scrattles — cat-only line."),
    (r"^KONG Teaser ", "cat_toy", ["cat"], "toys", 0.9, "KONG Teaser line — cat teaser toys."),
    (r"^KONG Tennishues", "cat_toy", ["cat"], "toys", 0.85, "KONG Tennishues — cat tennis-shoe-shaped toy."),
    (r"^KONG Connects.*Window Teaser", "cat_toy", ["cat"], "toys", 0.9, "KONG Connects Window Teaser — cat suction toy."),
    (r"^KONG Active Bubble Ball", "ball", ["cat"], "toys", 0.85, "KONG Active Bubble Ball — cat ball."),
    (r"^KONG Artz\b", "cat_toy", ["cat"], "toys", 0.75, "KONG Artz — cat cardboard scratcher line."),

    # ------- KONG Plush dog lines (subcategory change) -------
    (r"^KONG Cuteseas (Octopus|Whale)", "plush", ["dog"], "toys", 0.9, "KONG Cuteseas — sea-creature plush dog toy."),
    (r"^KONG Comfort\b", "plush", ["dog"], "toys", 0.85, "KONG Comfort line — soft plush dog toys."),
    (r"^KONG Cozies?\b|^KONG Cozie\b", "plush", ["dog"], "toys", 0.85, "KONG Cozies — soft plush dog toys."),
    (r"^KONG Daily Newspaper", "plush", ["dog"], "toys", 0.85, "KONG Daily Newspaper — plush newspaper-shape dog toy."),
    (r"^KONG Dynos", "plush", ["dog"], "toys", 0.85, "KONG Dynos — plush dragon dog toys."),
    (r"^KONG Enchanted", "plush", ["dog"], "toys", 0.85, "KONG Enchanted — plush dog toy line."),
    (r"^KONG Flingaroo", "plush", ["dog"], "toys", 0.8, "KONG Flingaroo — plush flying disc-style dog toy."),
    (r"^KONG Jungle Jamz", "plush", ["dog"], "toys", 0.9, "KONG Jungle Jamz — plush dog toy line."),
    (r"^KONG Layerz Forage", "plush", ["dog"], "toys", 0.8, "KONG Layerz Forage — plush snuffle/forage dog toy."),
    (r"^KONG Low Stuff", "plush", ["dog"], "toys", 0.85, "KONG Low Stuff — low-stuffing plush dog toys."),
    (r"^KONG Wild Low Stuff", "plush", ["dog"], "toys", 0.85, "KONG Wild Low Stuff — plush dog toy."),
    (r"^KONG Maxx ", "plush", ["dog"], "toys", 0.85, "KONG Maxx — plush squeak dog toy."),
    (r"^KONG Phatz", "plush", ["dog"], "toys", 0.85, "KONG Phatz — plush dog toy."),
    (r"^Kong Ringaroos", "plush", ["dog"], "toys", 0.85, "KONG Ringaroos — plush ring dog toy."),
    (r"^KONG Scampers", "plush", ["dog"], "toys", 0.85, "KONG Scampers — plush dog toy."),
    (r"^KONG Scruffs", "plush", ["dog"], "toys", 0.85, "KONG Scruffs — plush dog toy."),
    (r"^Kong scrumplez|^KONG Scrumplez", "plush", ["dog"], "toys", 0.85, "KONG Scrumplez — plush dog toy."),
    (r"^KONG Shakers", "plush", ["dog"], "toys", 0.85, "KONG Shakers — plush squeaky dog toy line."),
    (r"^KONG Shells", "plush", ["dog"], "toys", 0.85, "KONG Shells — plush dog toy."),
    (r"^KONG Sherps", "plush", ["dog"], "toys", 0.85, "KONG Sherps — plush dog toy line."),
    (r"^KONG Shieldz", "plush", ["dog"], "toys", 0.8, "KONG Shieldz — plush dog toy."),
    (r"^KONG Wild Shieldz", "plush", ["dog"], "toys", 0.8, "KONG Wild Shieldz — plush dog toy."),
    (r"^KONG Snuzzles", "plush", ["dog"], "toys", 0.9, "KONG Snuzzles — plush dog toy line."),
    (r"^KONG Stretchezz", "plush", ["dog"], "toys", 0.85, "KONG Stretchezz — stretchy plush dog toy."),
    (r"^KONG Toughz", "plush", ["dog"], "toys", 0.8, "KONG Toughz — durable plush dog toy."),
    (r"^KONG Tropics Pals", "plush", ["dog"], "toys", 0.8, "KONG Tropics Pals — plush dog toy."),
    (r"^KONG Tuggz", "plush", ["dog"], "toys", 0.8, "KONG Tuggz — plush tug dog toy."),
    (r"^KONG Winder", "plush", ["dog"], "toys", 0.8, "KONG Winder — plush wind-up-style dog toy."),
    (r"^KONG Woozles", "plush", ["dog"], "toys", 0.85, "KONG Woozles — plush dog toy."),
    (r"^KONG Wrangler", "plush", ["dog"], "toys", 0.85, "KONG Wrangler — plush dog toy line."),
    (r"^KONG Yarnimals", "plush", ["dog"], "toys", 0.85, "KONG Yarnimals — plush yarn-style dog toy."),

    # KONG Ballistic line — non-ball variants are plush (ball variants stay ball)
    (r"^KONG Ballistic.*(Bird|Alligator|Giraffe|Llama)\b(?!.*Ball)", "plush", ["dog"], "toys", 0.85, "KONG Ballistic plush — dog plush squeak toy."),
    (r"^KONG Ballistic.*Vbez", "plush", ["dog"], "toys", 0.85, "KONG Ballistic Vibez — dog plush line."),

    # KONG Wubba Ballistic / Wubba Friends Ballistic (currently ball) → plush
    (r"^KONG (Halloween )?Wubba.*Ballistic", "plush", ["dog"], "toys", 0.8, "KONG Wubba Ballistic — Wubba dog plush variant."),
    (r"^KONG Wubba Friends Ballistic", "plush", ["dog"], "toys", 0.8, "KONG Wubba Friends Ballistic — Wubba dog plush."),

    # ------- KONG Lick mats -------
    (r"^KONG Licks(?! Kitty)\b", "lick_mat", ["dog"], "toys", 0.85, "KONG Licks line — dog lick mat."),
    (r"^KONG Fill Or Freeze", "lick_mat", ["dog"], "toys", 0.85, "KONG Fill Or Freeze — dog lick/freeze tray."),

    # ------- KONG Dog rubber chew toys (keep chew_toy, narrow species) -------
    (r"^KONG Classic\b", "chew_toy", ["dog"], "toys", 0.95, "KONG Classic — iconic dog rubber chew toy."),
    (r"^KONG Extreme\b", "chew_toy", ["dog"], "toys", 0.95, "KONG Extreme — dog rubber chew toy."),
    (r"^KONG Senior Rubber", "chew_toy", ["dog"], "toys", 0.95, "KONG Senior — dog rubber chew toy."),
    (r"^KONG Aqua\b", "chew_toy", ["dog"], "toys", 0.9, "KONG Aqua — water-floating dog chew toy."),
    (r"^KONG Dental with rope", "chew_toy", ["dog"], "toys", 0.9, "KONG Dental Rope — dog dental rope."),
    (r"^KONG Goodie Ribbon", "chew_toy", ["dog"], "toys", 0.9, "KONG Goodie Ribbon — dog stuffable rope."),
    (r"^KONG Gyro\b", "chew_toy", ["dog"], "toys", 0.9, "KONG Gyro — dog rubber puzzle toy."),
    (r"^KONG Ogee Stick", "chew_toy", ["dog"], "toys", 0.85, "KONG Ogee Stick — dog rubber stick."),
    (r"^KONG Reflex stick", "chew_toy", ["dog"], "toys", 0.85, "KONG Reflex stick — dog rubber chew."),
    (r"^KONG Rewards (Shell|Tennis|Tinker|Wally)", "chew_toy", ["dog"], "toys", 0.85, "KONG Rewards line — dog treat-dispensing toy."),
    (r"^KONG Ring\b", "chew_toy", ["dog"], "toys", 0.9, "KONG Ring — dog rubber ring."),
    (r"^KONG Twistz Ring", "chew_toy", ["dog"], "toys", 0.85, "KONG Twistz Ring — dog rubber ring."),
    (r"^KONG Twistz High-Viz Ring", "chew_toy", ["dog"], "toys", 0.85, "KONG Twistz Ring — dog rubber ring."),
    (r"^KONG Flyer\b|^KONG Flyer Extreme|^KONG Puppy Flyer", "chew_toy", ["dog"], "toys", 0.9, "KONG Flyer — dog rubber frisbee."),
    (r"^KONG Flyangle", "chew_toy", ["dog"], "toys", 0.85, "KONG Flyangle — dog rubber flying toy."),
    (r"^KONG Squeezz\b", "chew_toy", ["dog"], "toys", 0.9, "KONG Squeezz — dog rubber squeaky chew."),
    (r"^KONG Squiggles", "chew_toy", ["dog"], "toys", 0.85, "KONG Squiggles — dog rubber squiggle toy."),
    (r"^KONG Wobbler", "chew_toy", ["dog"], "toys", 0.95, "KONG Wobbler — dog treat-dispensing toy."),
    (r"^KONG Switcheroos", "chew_toy", ["dog"], "toys", 0.8, "KONG Switcheroos — dog interactive toy."),
    (r"^KONG Signature Stick|^KONG Signature Rope|^KONG Signature Dynos", "chew_toy", ["dog"], "toys", 0.85, "KONG Signature stick/rope/dynos — dog chew toy."),
    (r"^KONG Small Animal", "chew_toy", ["dog"], "toys", 0.8, "KONG Small Animal — dog rubber chew."),

    # ------- KONG Dog balls (keep ball, narrow species) -------
    (r"^KONG AirDog", "ball", ["dog"], "toys", 0.9, "KONG AirDog — dog tennis-ball line."),
    (r"^KONG Ball\b|^KONG Ball with rope|^KONG Ball W/Hole", "ball", ["dog"], "toys", 0.9, "KONG Ball — dog ball variant."),
    (r"^KONG Bamboo Feeder Ball", "ball", ["dog"], "toys", 0.85, "KONG Bamboo Feeder Ball — dog feeder ball."),
    (r"^KONG Bunji.*Ball|^KONG Bunji.*Bumper", "ball", ["dog"], "toys", 0.8, "KONG Bunji ball/bumper — dog fetch."),
    (r"^KONG CoreStrength Ball", "ball", ["dog"], "toys", 0.9, "KONG CoreStrength Ball — dog dental ball."),
    (r"^KONG Crunch Air Ball", "ball", ["dog"], "toys", 0.9, "KONG Crunch Air Ball — dog tennis ball variant."),
    (r"^KONG Duramax Ball", "ball", ["dog"], "toys", 0.9, "KONG Duramax Ball — dog ball."),
    (r"^KONG Extreme Ball", "ball", ["dog"], "toys", 0.9, "KONG Extreme Ball — dog ball."),
    (r"^Kong Flexball|^KONG Flexball", "ball", ["dog"], "toys", 0.9, "KONG Flexball — dog ball."),
    (r"^KONG Jumbler", "ball", ["dog"], "toys", 0.9, "KONG Jumbler — dog ball/football."),
    (r"^KONG Reflex (ball|Football)", "ball", ["dog"], "toys", 0.9, "KONG Reflex — dog ball."),
    (r"^KONG Rewards Ball", "ball", ["dog"], "toys", 0.9, "KONG Rewards Ball — dog treat ball."),
    (r"^KONG Signature Balls?", "ball", ["dog"], "toys", 0.9, "KONG Signature Balls — dog ball line."),
    (r"^KONG Signature Sport", "ball", ["dog"], "toys", 0.9, "KONG Signature Sport — dog ball."),
    (r"^KONG SqueakAir", "ball", ["dog"], "toys", 0.95, "KONG SqueakAir — dog squeaky tennis ball line."),
    (r"^KONG Squeezz Ball|^KONG Squeezz Goomz Ball", "ball", ["dog"], "toys", 0.9, "KONG Squeezz Ball — dog ball."),
    (r"^KONG Stuff-A-Ball", "ball", ["dog"], "toys", 0.95, "KONG Stuff-A-Ball — dog stuffable ball."),
    (r"^KONG Twistz High-Viz Ball", "ball", ["dog"], "toys", 0.9, "KONG Twistz Ball — dog ball."),

    # ------- KONG Ziggies — actually a dental chew, not a toy -------
    (r"^KONG Ziggies", "dental_chew", ["dog"], "nutrition", 0.85, "KONG Ziggies — dental dog treat, not a toy."),
]


def find_match(name: str) -> tuple[int, str | None, list[str] | None, str | None, float, str] | None:
    for idx, rule in enumerate(RULES):
        pat = rule[0]
        if re.search(pat, name, flags=re.IGNORECASE):
            return (idx, rule[1], rule[2], rule[3], rule[4], rule[5])
    return None


def main() -> int:
    rows = [json.loads(l) for l in REMAINING.read_text().splitlines() if l.strip()]
    kong_toys = [r for r in rows if r.get("brand") == "KONG" and r.get("category") == "toys"]

    new_corrections: list[dict] = []
    skipped_no_match: list[dict] = []
    rule_hits: dict[int, int] = {}

    for r in kong_toys:
        name = r.get("product_name") or ""
        old = {
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "species": r.get("species") or [],
        }

        m = find_match(name)
        if m is None:
            skipped_no_match.append(r)
            continue

        rule_idx, new_sub, new_species, new_cat, conf, reason = m
        rule_hits[rule_idx] = rule_hits.get(rule_idx, 0) + 1

        new = {
            "category": new_cat or old["category"],
            "subcategory": new_sub if new_sub is not None else old["subcategory"],
            "species": new_species if new_species is not None else old["species"],
        }

        # Only emit if at least one field actually changes.
        if (old["category"] == new["category"]
            and old["subcategory"] == new["subcategory"]
            and set(old["species"]) == set(new["species"])):
            continue

        new_corrections.append({
            "sku": r.get("sku"),
            "chunk_id": r.get("chunk_id"),
            "product_name": name,
            "old": old,
            "new": new,
            "reasoning": reason,
            "confidence": conf,
        })

    with CORRECTIONS.open("a") as f:
        for c in new_corrections:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"KONG toys remaining: {len(kong_toys)}")
    print(f"  matched & corrected: {len(new_corrections)}")
    print(f"  matched-but-noop:    {len(kong_toys) - len(new_corrections) - len(skipped_no_match)}")
    print(f"  no rule matched:     {len(skipped_no_match)}")
    if skipped_no_match:
        print("\nUnmatched (sample 30):")
        for r in skipped_no_match[:30]:
            print(f"  {r.get('sku'):14s} sub={r.get('subcategory'):12s} sp={','.join(r.get('species') or []):9s} {r.get('product_name')}")

    print("\nRule hits:")
    for idx in sorted(rule_hits, key=lambda i: -rule_hits[i]):
        print(f"  {rule_hits[idx]:4d}  {RULES[idx][0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
