# Catalog re-categorization — final summary

**Reviewer:** Claude (in-conversation, no API)
**Approach:** Three-phase
1. Heuristic priority queue (183 high-suspicion rows) — per-row review
2. Rule-based bulk pass on 2707 remaining rows
3. **Per-row review of all 2707 remaining** (caught what rules missed)
4. Web-search verification for ambiguous cases

**Date completed:** 2026-05-04

## Output files

| File | Rows | Purpose |
|------|------|---------|
| `manual_corrections.jsonl` | **766** | All corrections, ready to apply |
| `low_confidence.jsonl` | 0 | All resolved via web search |
| `new_taxonomy.json` | — | Vocabulary additions (4 new subcategories + 4 new species) |
| `priority_queue.jsonl` | 183 | Phase 1 heuristic suspect set |
| `products_dump.jsonl` | 2890 | Full Qdrant dump (untouched) |
| `remaining_for_review.jsonl` | 2707 | Phase 2 working set |

## What's covered

**766 / 2890 products corrected (26%).** The other 2124 confirmed already-correct via per-row inspection.

## Vocabulary expansions (all wired into `backend/app.py`)

**New subcategories:**
- `healthcare/recovery_collar` — KVP EZ Soft post-surgery cones (5 SKUs)
- `toys/snuffle_toy` — HUNTER Eiby Bola + KONG Layerz Forage (3 SKUs, web-verified)
- `toys/cat_hideout` — KONG Play Spaces tents/caves (2 SKUs, web-verified)
- `toys/treat_dispenser` — KONG Rewards line + Wobbler (~9 SKUs, web-verified)

**New species values (all web-verified for some SKUs):**
- `horse` — MEN FOR SAN equine products (~15 SKUs)
- `rodent` — small-mammal products (~14 SKUs)
- `rabbit` — small-mammal co-tagged products (~14 SKUs)
- `ferret` — small-mammal co-tagged products (~14 SKUs)

The four MEN FOR SAN low-confidence entries (MFR012, MFH0012, MFR063, MFH040) were web-verified to all be `[rodent, rabbit, ferret]` per Menforsan's own manufacturer pages.

## Big patterns corrected

| Pattern | Count |
|---|---|
| KONG plush dog families (Knots/Wubba/Snuzzles/Cuteseas/Belly Flops/Comfort/Cozies/Shakers/Sherps/Stretchezz/Toughz/Tuggz/Wrangler/Maxx/Phatz/Scampers/Scruffs/Scrumplez/Shells/Shieldz/Tropics/Wild Low Stuff/Winder/Woozles/Yarnimals/Dynos/Enchanted/Flingaroo/Layerz Forage/Low Stuff/Ringaroos/Halloween Wubba Ballistic/Ballistic Vibez non-ball) | ~140 |
| KONG dog rubber chews (Classic/Extreme/Senior/Aqua/Dental/Goodie Ribbon/Gyro/Ogee/Reflex stick/Ring/Twistz Ring/Flyer/Flyangle/Squeezz/Squiggles/Switcheroos/Signature stick-rope-dynos/Small Animal/Daily Newspaper) | ~70 |
| KONG dog balls (AirDog/Ball/Bamboo/Bunji/CoreStrength/Crunch/Duramax/Extreme Ball/Flexball/Jumbler/Reflex/Signature Balls/SqueakAir/Squeezz Ball/Stuff-A-Ball/Twistz Ball) | ~50 |
| KONG cat-only lines (Cat Active/Better Buzz/Kitten/Kitty/Purrsuit/Connects/Refillables/Scrattles/Teaser/Tennishues/Pull-a-partz/Active Bubble Ball/Naturals Teaser/Infused Cat/Kitten Mice/Bat-A-Bout/Crackles) | ~40 |
| KONG Kickeroo line | 7 |
| KONG Lick mats (Licks/Fill Or Freeze) | 7 |
| KONG Ziggies (toy → nutrition/dental_chew) | 4 |
| KONG nutrition/training residuals | 47 |
| KONG ZoomGroom species (per-row find) | 3 |
| MEN FOR SAN dog/cat/horse/bird/rodent/rabbit/ferret species fixes via SKU prefix + name | ~120 |
| MEN FOR SAN cleaners mislabeled `fragrance` | 3 |
| HUNTER plush dog toy lines (Birka/Canvas/Faro/Muli/Tough/Tundra/Venice/Wildlife) | ~22 |
| **HUNTER Pullover Malmö wrongly tagged as collar** (per-row find) | 22 |
| **HUNTER SOFA Loppa/Corbie/Kumara wrongly tagged as collar** (per-row find) | ~22 |
| **HUNTER Pinza cortaúñas wrongly tagged as collar** (per-row find) | 2 |
| **COMPANY OF ANIMALS ANATOMY COLLAR wrongly tagged as harness** (per-row find) | 16 |
| **COMPANY OF ANIMALS Halti correa multiposición wrongly tagged as harness** (per-row find) | 8 |
| CEVA Adaptil dog → `[dog]`, Feliway cat → `[cat]` | 22 |
| CHIEN CHIC dog perfumes | 24 |
| COCOSI dog/cat foods narrowed | 22 |
| PET HEAD dog products + Furtastic spray subcategory | 9 |
| KVP EZ Soft → new `recovery_collar` | 5 |
| KONG Play Spaces → new `cat_hideout` (web-verified) | 2 |
| KONG Rewards Shell/Ball/Tennis/Tinker/Wally + Wobbler → new `treat_dispenser` (web-verified) | ~9 |
| KONG Bunji High-Viz Bumper → `training/dummy` (web-verified) | 1 |
| HUNTER Eiby Bola + KONG Layerz Forage → new `snuffle_toy` (web-verified) | 3 |
| **YOWUP Bone Broth/Milky Lick wrongly tagged as grooming/balm** (per-row find) | 4 |
| URINE OFF cat/dog formula species | 4 |
| ROTHO MYPET cat scoops species | 6 |
| FLEXI cat-line leash species | 4 |
| INODORINA cat litter species | 2 |
| VETIQ indoor cat snacks | 1 |
| SPRENGER soft dummy / NYLABONE SC Peanut / etc. | a few |

## Per-row review caught what rules missed

The bold-italic items above (Pullover Malmö, SOFA Loppa, Pinza cortaúñas, ANATOMY COLLAR, Halti correa, YOWUP Bone Broth/Milky Lick) — totalling ~97 SKUs — would have stayed mislabeled with the rule-based approach alone. These are pure data-entry errors that any pattern matching would miss because the names don't follow predictable family rules. Per-row review caught them.

## Data-quality issues uncovered

1. **Duplicate SKU `JU03240`** — used by both `KONG Kickeroo Cactus` and `KONG Cat Wubba Bunny`. Corrections key on `chunk_id`.
2. **Generic `toy` subcategory was severely overused** — collapsed into `plush`/`chew_toy`/`cat_toy`/`lick_mat`/`tug_toy`/`treat_dispenser`/`cat_hideout` per the actual product type.
3. **`fragrance` subcategory was over-applied** — caught and fixed.
4. **`collar` subcategory had 3 distinct mislabel patterns** in HUNTER alone — sweaters, sofas, nail clippers all wrongly tagged collar.
5. **Species over-broadening was systemic** — many brand-line dog products tagged `[dog,cat]` despite name saying "para perros". Multiple brands hit this pattern.

## Next steps to apply

1. Spot-review `manual_corrections.jsonl` — open it, scan the rows.
2. Apply by either:
   - **Patch Qdrant in place** (fastest) — `client.set_payload(...)` keyed on `chunk_id`, no re-embed needed since text doesn't change.
   - **Or merge into source JSONL** and re-run `indexing/build_index_from_excel.py` (slower, recomputes BM25).
3. New subcategories (`recovery_collar`, `snuffle_toy`, `cat_hideout`, `treat_dispenser`) and species (`horse`, `rodent`, `rabbit`, `ferret`) are already in the vision prompt — re-run a photo-search test to confirm filters work end-to-end.
