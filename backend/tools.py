"""
Tool-calling layer for the Gloriapets RAG chatbot.

Exposes five tools to the LLM (via OpenRouter's OpenAI-compatible API):

  1. semantic_search       — hybrid dense + BM25 retrieval (same as the old /api/chat path)
  2. filter_scroll         — exact-filter enumeration of products ("all KONG harnesses under 20€")
  3. count_products        — aggregate counts grouped by brand / category / subcategory / species
  4. get_product           — single-product lookup by SKU or EAN
  5. list_distinct_values  — distinct values of a field, optionally scoped by a brand

The LLM decides which tool(s) to call per user turn. Tool results are fed back
as role=tool messages until the model finishes (finish_reason == "stop").

Design notes
------------
- Pure functions, no global state beyond the passed-in Qdrant/BM25/embedder handles.
- All range/float filters are applied via Qdrant payload filters (we index them in
  build_index_from_excel.py).
- count_products + list_distinct_values scroll the whole collection (2,890 points)
  and aggregate in Python — fast enough, no Qdrant facet API dependency.
- Scroll batch size kept small (512) to avoid oversized responses.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Iterator

import httpx
from backend.image_map import attach_images
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)

# --------------------------------------------------------------------------
# Schema used both for LLM prompt documentation and argument validation.
# --------------------------------------------------------------------------

BRAND_ENUM = [
    "SKINNIA", "KVP", "PET HEAD", "CEVA", "CHIEN CHIC", "COACHI",
    "COMPANY OF ANIMALS", "DGS", "3 CLAVELES", "ANDIS", "VAN CAT",
    "MEN FOR SAN", "YOWUP", "LICKIMAT", "SPRENGER", "RED DINGO",
    "CHEWLLAGEN", "KONG", "LICENCIADOS KONG", "FLEXI", "NYLABONE",
    "ROTHO MYPET", "UNITED PETS", "EARTH RATED", "URINE OFF",
    "VETIQ", "HUNTER", "INODORINA", "COCOSI", "WHIMZEES",
]

CATEGORY_ENUM = [
    "grooming", "accessories", "nutrition", "hygiene", "training",
    "toys", "housing", "healthcare", "apparel", "equipment",
]

SPECIES_ENUM = ["dog", "cat", "bird", "horse", "reptile", "rabbit", "ferret", "rodent"]

GROUP_BY_ENUM = ["brand", "category", "subcategory", "species", "change_flag"]

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Find products matching a free-text query. Use this for descriptive lookups "
                "('collar rojo para perro mediano', 'KONG chew toys', 'cat litter with aroma'). "
                "Returns the top-K most relevant product chunks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The semantic query in the user's language.",
                    },
                    "brand": {"type": "string", "enum": BRAND_ENUM,
                              "description": "Optional brand filter."},
                    "category": {"type": "string", "enum": CATEGORY_ENUM,
                                 "description": "Optional category filter."},
                    "species": {"type": "string", "enum": SPECIES_ENUM,
                                "description": "Optional species filter."},
                    "price_min": {"type": "number", "description": "€ minimum PVPR."},
                    "price_max": {"type": "number", "description": "€ maximum PVPR."},
                    "limit": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_scroll",
            "description": (
                "List products matching strict filter criteria, without any semantic ranking. "
                "Use for enumeration questions like 'show me all KONG chew toys' or "
                "'list harnesses under 30€'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "brand": {"type": "string", "enum": BRAND_ENUM},
                    "category": {"type": "string", "enum": CATEGORY_ENUM},
                    "subcategory": {"type": "string"},
                    "species": {"type": "string", "enum": SPECIES_ENUM},
                    "price_min": {"type": "number"},
                    "price_max": {"type": "number"},
                    "size_label": {"type": "string"},
                    "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_products",
            "description": (
                "Count products grouped by a single field. Use for aggregation questions "
                "like 'which brand has the most products', 'how many categories', "
                "'how many dog vs cat products'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string", "enum": GROUP_BY_ENUM,
                                 "description": "Field to group counts by."},
                    "brand": {"type": "string", "enum": BRAND_ENUM},
                    "category": {"type": "string", "enum": CATEGORY_ENUM},
                    "species": {"type": "string", "enum": SPECIES_ENUM},
                    "top_n": {"type": "integer", "default": 30, "minimum": 1, "maximum": 100},
                },
                "required": ["group_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Fetch a single product by SKU or EAN (barcode).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "ean": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_distinct_values",
            "description": (
                "List all distinct values of a field in the catalog, optionally scoped by brand. "
                "Use for questions like 'what colors does RED DINGO offer', 'what subcategories exist "
                "under toys'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": [
                        "brand", "category", "subcategory", "species", "change_flag",
                        "color", "scent", "size_label", "dureza", "tipo", "recambio",
                    ]},
                    "brand": {"type": "string", "enum": BRAND_ENUM},
                    "category": {"type": "string", "enum": CATEGORY_ENUM},
                    "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 500},
                },
                "required": ["field"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fit_search",
            "description": (
                "Find products that physically fit a given measurement. Uses range-containment: "
                "a 35 cm neck fits a collar iff neck_min_cm <= 35 <= neck_max_cm. "
                "Use for queries like 'collar for a 35 cm neck', 'harness for a 12 kg dog', "
                "'arnés para perro de 8 kg'. Combine with brand/category/species/price when given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "neck_cm": {"type": "number",
                                "description": "Neck circumference in cm. Filters on neck_min_cm / neck_max_cm (KVP, RED DINGO)."},
                    "chest_cm": {"type": "number",
                                 "description": "Chest girth in cm. Filters on chest_min_cm / chest_max_cm (RED DINGO)."},
                    "body_cm": {"type": "number",
                                "description": "Body length in cm. Filters on body_min_cm / body_max_cm (RED DINGO)."},
                    "dog_weight_kg": {"type": "number",
                                      "description": "Dog weight in kg. Filters on dog_weight_min_kg / dog_weight_max_kg (KVP, CEVA, FLEXI, NYLABONE)."},
                    "cat_weight_kg": {"type": "number",
                                      "description": "Cat weight in kg. Filters on cat_weight_min_kg / cat_weight_max_kg (CEVA, UNITED PETS)."},
                    "brand": {"type": "string", "enum": BRAND_ENUM},
                    "category": {"type": "string", "enum": CATEGORY_ENUM},
                    "species": {"type": "string", "enum": SPECIES_ENUM},
                    "price_min": {"type": "number"},
                    "price_max": {"type": "number"},
                    "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 30},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": (
                "Fetch multiple products by SKU and return them side-by-side for structured "
                "comparison (brand, price, size, weight, colors, etc.). Use when the user wants "
                "to compare specific products, e.g. 'compara KONG Classic vs NYLABONE Extreme'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SKUs to compare (2-10).",
                        "minItems": 2,
                        "maxItems": 10,
                    },
                },
                "required": ["skus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_budget_basket",
            "description": (
                "Compose a shopping basket that fits within a TOTAL budget, respecting each "
                "product's minimum order quantity. Returns a pre-computed basket with per-line "
                "totals and a running total. ALWAYS use this tool when the user gives a total "
                "budget (e.g. 'qué me llevo por 50€ de COCOSI', 'a basket of €200 across 5 "
                "categories'). Never do budget math in your own reply — this tool is "
                "deterministic and composes correctly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_eur": {
                        "type": "number",
                        "description": "Total budget in euros (required).",
                        "minimum": 1,
                    },
                    "brand": {"type": "string", "enum": BRAND_ENUM},
                    "category": {"type": "string", "enum": CATEGORY_ENUM},
                    "subcategory": {"type": "string"},
                    "species": {"type": "string", "enum": SPECIES_ENUM},
                    "min_items": {"type": "integer", "default": 3, "minimum": 1, "maximum": 20},
                    "max_items": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
                    "diversity": {
                        "type": "string",
                        "enum": ["subcategory", "brand", "none"],
                        "default": "subcategory",
                        "description": "Spread across this axis when multiple items would fit.",
                    },
                },
                "required": ["budget_eur"],
            },
        },
    },
]


# --------------------------------------------------------------------------
# Shared helpers — Qdrant filter construction
# --------------------------------------------------------------------------
def _build_filter(
    *,
    brand: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    species: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    size_label: str | None = None,
    change_flag: str | None = None,
    exclude_deleted: bool = True,
    chunk_type: str | None = "product_sku_row",
) -> Filter | None:
    must: list[FieldCondition] = []
    must_not: list[FieldCondition] = []

    if chunk_type:
        must.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
    if brand:
        must.append(FieldCondition(key="brand", match=MatchValue(value=brand)))
    if category:
        must.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if subcategory:
        must.append(FieldCondition(key="subcategory", match=MatchValue(value=subcategory)))
    if species:
        must.append(FieldCondition(key="species", match=MatchAny(any=[species])))
    if size_label:
        must.append(FieldCondition(key="size_label", match=MatchValue(value=size_label)))
    if change_flag:
        must.append(FieldCondition(key="change_flag", match=MatchValue(value=change_flag)))
    elif exclude_deleted:
        must_not.append(FieldCondition(
            key="change_flag",
            match=MatchAny(any=["ELIMINAR", "producto eliminado"]),
        ))
    if price_min is not None or price_max is not None:
        r = Range(
            gte=float(price_min) if price_min is not None else None,
            lte=float(price_max) if price_max is not None else None,
        )
        must.append(FieldCondition(key="price_pvpr", range=r))

    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def _product_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Compact summary of a product payload — fed back to the LLM as tool output."""
    names = payload.get("names") or {}
    name = payload.get("name_es") or names.get("es") or payload.get("product_name")

    price_pvpr = payload.get("price_pvpr") or payload.get("price_eur")
    price_per_unit = payload.get("price_per_unit")
    min_qty = payload.get("min_purchase_qty") or payload.get("min_order")

    # Derived metrics — make the LLM's job easier and remove a common miscalculation.
    # price_total_min_order = what the buyer actually pays for a minimum-compliant order.
    try:
        price_total_min_order = (
            round(float(price_pvpr) * float(min_qty), 2)
            if price_pvpr is not None and min_qty is not None
            else None
        )
    except (TypeError, ValueError):
        price_total_min_order = None

    # effective_price_per_unit — what the LLM should use when comparing VALUE between products.
    # Prefer explicit price_per_unit if set; else treat price_pvpr as per-unit (typical case).
    try:
        effective_unit_price = (
            float(price_per_unit) if price_per_unit is not None else
            (float(price_pvpr) if price_pvpr is not None else None)
        )
        if effective_unit_price is not None:
            effective_unit_price = round(effective_unit_price, 2)
    except (TypeError, ValueError):
        effective_unit_price = None

    product = {
        "id": payload.get("id") or payload.get("chunk_id"),
        "brand": payload.get("brand"),
        "sku": payload.get("sku"),
        "ean": payload.get("ean"),
        "name_es": name,
        "category": payload.get("category"),
        "subcategory": payload.get("subcategory"),
        "species": payload.get("species"),
        "price_pvpr": price_pvpr,
        "price_per_unit": price_per_unit,
        "min_purchase_qty": min_qty,
        # Derived — the LLM should use these directly for budgets & comparisons.
        "price_total_min_order": price_total_min_order,
        "effective_unit_price": effective_unit_price,
        "has_price": price_pvpr is not None,
        # PDF catalog cross-reference (populated by indexing/patch_qdrant_pages.py).
        "catalog_pages": payload.get("catalog_pages"),
        "primary_page": payload.get("primary_page"),
        "size_label": payload.get("size_label"),
        "color": payload.get("color"),
        "scent": payload.get("scent"),
        "capacity_raw": payload.get("capacity_raw"),
        "change_flag": payload.get("change_flag"),
        # Fit ranges (only set when present on the brand).
        "neck_cm": _range(payload.get("neck_min_cm"), payload.get("neck_max_cm")),
        "body_cm": _range(payload.get("body_min_cm"), payload.get("body_max_cm")),
        "chest_cm": _range(payload.get("chest_min_cm"), payload.get("chest_max_cm")),
        "dog_weight_kg": _range(payload.get("dog_weight_min_kg"), payload.get("dog_weight_max_kg")),
        "cat_weight_kg": _range(payload.get("cat_weight_min_kg"), payload.get("cat_weight_max_kg")),
        "length_cm": payload.get("length_cm"),
        "width_cm": payload.get("width_cm"),
        "height_cm": payload.get("height_cm"),
        "weight_g": payload.get("weight_g"),
        "text": payload.get("text"),
    }
    return attach_images(product)


def _range(lo: Any, hi: Any) -> list[float] | None:
    if lo is None and hi is None:
        return None
    return [lo, hi]


# --------------------------------------------------------------------------
# Cross-encoder reranker (applied inside semantic_search)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_reranker() -> Any:
    """Load the cross-encoder reranker once. Multilingual, ~120 MB, fast on CPU."""
    from sentence_transformers import CrossEncoder  # type: ignore

    model_name = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    return CrossEncoder(model_name, max_length=256, device="cpu")


def _rerank_products(
    query: str,
    products: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    """
    Rerank `products` by (query, passage) relevance using the cross-encoder,
    return the top-N. Falls back to the original order if the reranker is
    disabled (RERANKER_ENABLED=0) or fails to load.
    """
    if not products:
        return products
    if os.getenv("RERANKER_ENABLED", "1") == "0":
        return products[:top_n]

    try:
        reranker = _get_reranker()
    except Exception:
        return products[:top_n]

    pairs: list[list[str]] = []
    for p in products:
        passage = (p.get("text") or "").strip()
        if not passage:
            passage = " ".join(filter(None, [
                str(p.get("brand") or ""),
                str(p.get("name_es") or ""),
                str(p.get("category") or ""),
                str(p.get("subcategory") or ""),
            ]))
        pairs.append([query, passage])

    try:
        scores = reranker.predict(pairs)
    except Exception:
        return products[:top_n]

    scored = list(zip((float(s) for s in scores), products))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_n]]


# --------------------------------------------------------------------------
# System prompt for the tool-calling LLM
# --------------------------------------------------------------------------
def build_tool_system_prompt(user_language: str | None = None) -> str:
    lang = (user_language or "").strip().lower()
    if not lang or len(lang) > 8:
        lang = "unknown"
    if lang == "unknown":
        lang = "es"
    language_name = {
        "en": "English",
        "es": "Spanish",
        "hi": "Hindi",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
    }.get(lang, lang)

    return (
        "You are a warm, helpful sales assistant for Gloriapets, a wholesale pet products "
        "distributor. You have tool access to query a catalog of ~2,890 product variants across "
        "30 brands, stored in Qdrant. **Every factual claim about a product (name, price, size, "
        "min order, availability) must come from a tool result.** Do not invent.\n\n"

        "## LANGUAGE\n"
        f"- Detected user language (ISO code): {lang}.\n"
        f"- Write your reply in {language_name} to match the user.\n"
        "- If the user writes the NEXT message in a different language, switch to that language "
        "for that reply. You support every language the user speaks in — never tell the user "
        "'I only work in X'. Product names and category labels may stay in their catalog language "
        "(usually Spanish); translate them inline if helpful.\n\n"

        "## TONE\n"
        "- Conversational and warm, not clinical. Use friendly openers when natural "
        "('Te dejo algunas opciones…', 'Here are a few picks for you…', 'Voilà quelques options…').\n"
        "- Keep facts rigorous: prices, SKUs, and min-order figures must be exact and sourced from tools.\n"
        "- Concise over verbose. One product per line when listing.\n\n"

        "## PERSONALIZATION & MEMORY\n"
        "The conversation history is in your context. Before each reply, silently extract and apply any "
        "user-stated attributes:\n"
        "  - species, breed, weight, age\n"
        "  - budget, min-order tolerance\n"
        "  - preferred brand, preferred material/type\n"
        "  - prior rejections ('no me gusta FLEXI', 'ya tengo collar', 'sin perfume')\n"
        "Apply them to every subsequent tool call WITHOUT asking again. When context is applied, mention it "
        "briefly so the user knows you remembered: 'basándome en tu perro de 10 kg…', 'manteniéndonos bajo "
        "tu presupuesto de 20€…'.\n"
        "If conflicting signals appear (user says 12 kg now, 10 kg earlier), use the most recent and "
        "briefly note the change.\n\n"

        "## MANDATORY TOOL-USE DISCIPLINE\n"
        "NEVER say 'no hay', 'nothing available', 'we don't carry that', or similar without FIRST "
        "calling a tool AND seeing an empty result. If your intuition says 'probably doesn't exist', "
        "call the tool anyway — the catalog has 2,890 products and many queries seem narrow but hit matches.\n\n"

        "If your first tool call returns ZERO results:\n"
        "  a) Retry with relaxed filters: drop price range, drop species, drop color, widen subcategory.\n"
        "  b) Try `semantic_search` with a broader phrasing (strip adjectives like 'húmedas', 'soft', 'grande').\n"
        "  c) Only after those retries come back empty, say the catalog doesn't have it.\n\n"

        "Do NOT refuse a match because the user used an adjective that isn't in the product name. "
        "Example: 'toallitas húmedas' — products in the catalog are just 'Toallitas' (wet wipes). "
        "List the matching 'Toallitas' products; the user can refine.\n\n"

        "This rule applies to ADVICE/EDUCATIONAL queries too. If the user asks 'how do I…', "
        "'qué le doy a…', 'what's best for…', 'diet for…', 'cómo entrenar…', you MUST call a tool "
        "to retrieve relevant products before replying. This is a catalog assistant, not a generic "
        "vet — do not default to pure knowledge-mode answers when the catalog could plausibly help.\n\n"

        "## WHEN TO CALL WHICH TOOL\n"
        "- Semantic / descriptive ('red collar for small dog', 'champús para cachorro') → `semantic_search`.\n"
        "- Advisory with implied product need ('diet for senior shepherd', 'juguete para aburrimiento', "
        "'cómo cuidar el pelaje', 'algo para la ansiedad de mi gato') → `semantic_search` with a "
        "need-phrased query (not the user's literal words), optionally combined with a category filter "
        "(`nutrition`, `healthcare`, `grooming`, `toys`, `training`).\n"
        "- Strict enumeration ('list all KONG chew toys', 'show products under 20€') → `filter_scroll`.\n"
        "- Aggregation / counting ('which brand has the most products', 'how many dog items') → `count_products`.\n"
        "- Fit queries with explicit numbers ('collar for 35 cm neck', 'harness for 12 kg dog') → `fit_search`.\n"
        "  ALWAYS prefer fit_search over semantic_search when the user gives an exact measurement.\n"
        "- Product comparison ('compare KONG Classic vs NYLABONE Extreme', 'cuál es más barato') → `compare_products`.\n"
        "- TOTAL-budget queries ('qué me llevo por 50€ de COCOSI', 'basket for €200 across 5 "
        "categories', 'productos hasta 100€') → `build_budget_basket`. NEVER do budget math in your own "
        "reply; the tool composes the basket deterministically. Pass `brand`, `category`, `subcategory`, "
        "or `species` when the user specifies them. The tool auto-scales the item cap to the budget — "
        "you don't need to pass `max_items` unless the user explicitly asks for 'exactly N items'.\n"
        "- Exact SKU or EAN lookup → `get_product`.\n"
        "- Catalog facets ('what colors does RED DINGO sell') → `list_distinct_values`.\n"
        "You may call multiple tools in sequence. Prefer one well-targeted call over many.\n\n"

        "## ENUMERATION PREFERENCE (IMPORTANT)\n"
        "For narrow-category browse queries where the user explicitly names a subcategory or an "
        "unambiguous category, USE `filter_scroll` with the exact subcategory (and category when "
        "explicit). Do NOT use `semantic_search` for these — it returns semantic neighbours (harnesses "
        "when asked for leashes, collars when asked for harnesses), bloating results with irrelevant items.\n"
        "Use `semantic_search` only for descriptive queries where no subcategory is clearly named.\n\n"

        "## SUBCATEGORY DISCIPLINE (VERY IMPORTANT)\n"
        "Product subcategories are strict and NOT interchangeable:\n"
        "  - correa / leash = subcategory 'leash'       (lead rope, any length)\n"
        "  - arnés / petral / harness = subcategory 'harness' (chest strap)\n"
        "  - collar / collier = subcategory 'collar'    (neck band only)\n"
        "  - bozal / muzzle = subcategory 'muzzle'\n"
        "  - cama / bed = subcategory 'bed'\n"
        "  - transportín / carrier = subcategory 'carrier'\n"
        "  - champú / shampoo = subcategory 'shampoo'\n"
        "  - acondicionador / conditioner = subcategory 'conditioner'\n"
        "  - bolsa de caca / poop bag = subcategory 'poop_bags'\n"
        "  - arena / litter = subcategory 'litter'\n"
        "If the user names one of these, restrict retrieval to that EXACT subcategory via filter_scroll.\n"
        "NEVER recommend a harness when asked for a leash. Never recommend a collar when asked for a "
        "harness. They are distinct products with different uses.\n\n"

        "## CATEGORY HANDLING\n"
        "Apply `category=X` as a filter ONLY when the user uses an explicit category word or a tight synonym.\n"
        "Otherwise omit the category filter and let subcategory + semantic rerank handle the narrowing.\n\n"
        "Explicit category words → filter:\n"
        "  - juguetes / juego / toys / peluche                    → toys\n"
        "  - comida / alimento / pienso / food / treats           → nutrition\n"
        "  - adiestramiento / entrenamiento / training            → training\n"
        "  - vacuna / medicina / suplemento / healthcare          → healthcare\n"
        "  - cama / transportín / jaula / carrier                 → housing\n"
        "  - ropa / abrigo / impermeable / chaqueta / apparel     → apparel\n"
        "  - máquina de cortar / cuchilla / clipper blade         → equipment\n\n"
        "AMBIGUOUS words that span categories — DO NOT apply a category filter; use subcategory or free "
        "semantic instead:\n"
        "  - aseo / limpieza / bañar / hygienic   (→ grooming OR hygiene)\n"
        "  - accesorio / accessory                 (→ many subcategories)\n"
        "  - toallita / wipe                       (→ hygiene/wipes OR grooming/wipes)\n"
        "  - cosmético / care                      (→ grooming OR healthcare)\n\n"
        "CATEGORY FALLBACK: if a strict category filter returns ZERO results, retry the same query "
        "without the category filter before declaring nothing is available.\n\n"

        "## HANDLING BROAD QUERIES\n"
        "A broad query names a category but no constraint (e.g. 'correas para perro', 'champú', "
        "'un juguete para mi gato'). 300+ products match. DO NOT dump 5 random options. Choose ONE of:\n"
        "  (1) Ask ONE short clarifying question that will most narrow the search. Pick the single most "
        "      impactful axis:\n"
        "        - leash:   '¿Fija o extensible, y qué tamaño de perro?'\n"
        "        - shampoo: '¿Para perro o gato? ¿Piel sensible?'\n"
        "        - food:    '¿Especie, edad, y alguna restricción dietética?'\n"
        "        - toy:     '¿Mordedor fuerte, cachorro, o suave?'\n"
        "        - collar:  '¿Tamaño de cuello y estilo (liso, luminoso, diseño)?'\n"
        "  (2) OR give 3 baseline picks spanning the main sub-types + one sentence inviting refinement:\n"
        "        'Te dejo tres opciones populares: [A] tape fija, [B] extensible, [C] cordón — dime "
        "         tamaño o presupuesto si quieres afinar.'\n"
        "Never do both. Pick (1) for very vague queries; (2) when the user clearly wants to browse.\n"
        "If personalization memory already tells you the size/species/budget, skip clarifying and go "
        "straight to 3–5 targeted picks.\n\n"

        "## ADVISORY / EDUCATIONAL QUERIES\n"
        "Advice questions — diet, training, grooming, health, behaviour, enrichment, life-stage care — "
        "ALWAYS require a catalog call before answering. Trigger words include: *diet / dieta, cómo "
        "alimentar, best food for, qué le doy a, training / adiestrar / entrenar, cómo bañar, joint / "
        "articulaciones, ansiedad, aburrimiento, pelaje, piel sensible, cachorro, senior, sarro, pulgas, "
        "viaje, castración, embarazo*.\n"
        "Flow:\n"
        "  1. Build a need-phrased query from the implied product category, NOT the user's literal "
        "     words. Example: 'Diet for a senior German Shepherd?' → "
        "     `semantic_search(query='senior large breed dog food joint support omega-3', "
        "     species='dog', category='nutrition')`. "
        "     'Mi perro se aburre en casa' → `semantic_search(query='interactive chew toy mental "
        "     stimulation dog', species='dog', category='toys')`.\n"
        "  2. Reply with a short advice tip (2–4 bullets or 2–4 sentences) covering the key principles, "
        "     THEN 'Del catálogo te recomiendo:' / 'From our catalog I recommend:' with 3 products in "
        "     the standard `Brand · SKU · name · price/ud` line format.\n"
        "  3. Never end an advisory reply with only generic knowledge — always include catalog picks "
        "     unless retrieval genuinely returned nothing relevant.\n\n"

        "## MUST-SURFACE FIELDS IN EVERY RECOMMENDATION\n"
        "Every product you recommend MUST include, in the user's language:\n"
        "  1. Brand · SKU · name_es\n"
        "  2. Price (`price_pvpr`) in euros. If `price_pvpr` is null, do NOT recommend that product "
        "     for a purchase query — choose another. It's acceptable to mention it only if the user is "
        "     asking about availability, not price.\n"
        "  3. Minimum order quantity: if `min_purchase_qty` is present and > 1, state it explicitly "
        "     (e.g. 'mínimo 8 unidades', 'min. order: 3 units'). This is a required wholesale constraint — "
        "     never say 'no minimum' unless `min_purchase_qty` is null or 1.\n"
        "  4. When `min_purchase_qty > 1` and `price_total_min_order` is set, also show the total cost of a "
        "     minimum-compliant order (e.g. '6.80 €/ud × 8 = 54.40 €').\n\n"

        "## RECOMMENDATION STRUCTURE\n"
        "When listing products:\n"
        "  - 3–5 options, never more, unless user asks for 'all' / 'todos'.\n"
        "  - Spread across at least 2 brands when possible — don't monoculture the result.\n"
        "  - Each line format:\n"
        "      `Brand · SKU · name_es · price/ud (min. N uds = total €)` followed by a short "
        "      6–12-word rationale noting size range, notable feature, or suitability.\n"
        "      Example:\n"
        "        HUNTER · HU68934 · Correa Convenience · 20.72€/ud · 120 cm, ideal para paseos diarios con perros medianos.\n"
        "        FLEXI · CR04021AZ · Correa New Classic cinta · 21.14€/ud · 5 m extensible para perros hasta 15 kg.\n"
        "  - End the reply with a single short invitation to refine: "
        "    '¿Lo ajusto al tamaño de tu perro o a un presupuesto?'\n\n"
        "Advisory variant (when the query is advice/educational): lead with a short tip "
        "(2–4 bullets or sentences), then 'Del catálogo te recomiendo:' followed by 3 products in the "
        "same line format. Tip first, products second — never skip either.\n\n"

        "## COMPARISONS — ALWAYS COMPUTE VALUE METRICS\n"
        "When the user asks 'which is better value' / 'cuál es mejor precio' etc.:\n"
        "  - Use `effective_unit_price` from each product (already the right per-unit price).\n"
        "  - If `capacity_raw` contains a count like '60 unds', '300 bolsas', '5 L', compute "
        "    price_per_physical_unit = price_pvpr / count and state it.\n"
        "  - Present as a Markdown table with columns per SKU and rows for: price, min order, "
        "    effective unit price, price per physical unit (if derivable), weight/size, key attributes.\n"
        "  - Use '—' for missing values.\n"
        "  - End with a 1–2 sentence verdict: which is cheaper per unit, which is larger, who wins overall.\n\n"

        "## BUDGET QUERIES (use the dedicated tool)\n"
        "When the user gives a total budget — anything shaped like 'por 50€', 'con 200€', 'up to €100', "
        "'budget of X€' — DO NOT compose the basket yourself. Call `build_budget_basket(budget_eur=X, "
        "brand=…, category=…, subcategory=…, species=…)` and then format the returned basket.\n"
        "The tool returns:\n"
        "  - `basket`: list of {brand, sku, name_es, unit_price, min_qty, line_total, category, subcategory}\n"
        "  - `total_cost`, `budget_used_pct`, `remaining_budget`, `reason`, `items_considered`\n"
        "Rendering rules:\n"
        "  - Format each basket line exactly as `Brand · SKU · name_es · unit_price €/ud × min_qty uds = line_total €` "
        "    followed by a short 6–10-word rationale (subcategory / use-case).\n"
        "  - After the bullets, show: 'Total: {total_cost} € ({budget_used_pct}% del presupuesto)'.\n"
        "  - If `budget_used_pct < 90`, surface the `reason` verbatim so the user understands why it couldn't fill.\n"
        "  - Never add a product that wasn't in the returned `basket`. The tool already enforces min-order and budget math.\n"
        "  - Never re-do arithmetic — trust the tool's numbers exactly.\n\n"

        "## FIT QUERIES\n"
        "Range-containment: A 35 cm neck fits iff neck_cm[0] ≤ 35 ≤ neck_cm[1]. A null range on a chunk "
        "means that product doesn't specify that measurement — say so explicitly; don't guess fit.\n\n"

        "## FILTERING HYGIENE\n"
        "- Discontinued products (change_flag ∈ {ELIMINAR, 'producto eliminado'}) are filtered at the "
        "tool level by default. Don't mention them unless the user asks for deleted items.\n"
        "- Never cite by page number — the catalog has none.\n"
        "- If the question is completely off-topic (weather, news, math), politely decline and redirect.\n\n"

        "## PERMISSIBLE FILTER VALUES\n"
        "- brand: " + ", ".join(BRAND_ENUM) + "\n"
        "- category: " + ", ".join(CATEGORY_ENUM) + "\n"
        "- species: " + ", ".join(SPECIES_ENUM) + "\n"
        "- group_by (for count_products): " + ", ".join(GROUP_BY_ENUM) + "\n"
    )


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------
def semantic_search(
    *,
    query: str,
    brand: str | None = None,
    category: str | None = None,
    species: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    limit: int = 8,
    qdrant: QdrantClient,
    collection: str,
    embedder: Any,
    bm25: Any,
    bm25_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    from retrieval.hybrid_search import bm25_search, qdrant_search
    from retrieval.rrf import reciprocal_rank_fusion

    flt = _build_filter(brand=brand, category=category, species=species,
                        price_min=price_min, price_max=price_max)

    # Over-fetch candidates so the reranker can promote strong matches that
    # hybrid RRF ranked lower. Trimmed back to `limit` after reranking.
    initial_k = max(limit * 3, 20)
    vec = qdrant_search(qdrant, collection, embedder, query, top_k=initial_k, qdrant_filter=flt)
    kw = bm25_search(bm25, bm25_chunks, query, top_k=initial_k)
    fused = reciprocal_rank_fusion(vec, kw, top_n=initial_k)

    candidates = [_product_summary(dict(item.payload or {})) for item in fused]
    products = _rerank_products(query, candidates, top_n=limit)
    return {
        "tool": "semantic_search",
        "query": query,
        "filters": {k: v for k, v in {
            "brand": brand, "category": category, "species": species,
            "price_min": price_min, "price_max": price_max,
        }.items() if v is not None},
        "count": len(products),
        "reranked": os.getenv("RERANKER_ENABLED", "1") != "0",
        "products": products,
    }


def filter_scroll(
    *,
    brand: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    species: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    size_label: str | None = None,
    limit: int = 20,
    qdrant: QdrantClient,
    collection: str,
) -> dict[str, Any]:
    flt = _build_filter(brand=brand, category=category, subcategory=subcategory,
                        species=species, price_min=price_min, price_max=price_max,
                        size_label=size_label)
    total_count: int | None = None
    try:
        if flt is None:
            cr = qdrant.count(collection_name=collection, exact=True)
        else:
            cr = qdrant.count(collection_name=collection, count_filter=flt, exact=True)
        total_count = int(getattr(cr, "count", cr.get("count") if isinstance(cr, dict) else 0))
    except Exception:
        total_count = None

    points, _ = qdrant.scroll(
        collection_name=collection,
        scroll_filter=flt,
        limit=limit,
        with_payload=True,
    )
    products = [_product_summary(dict(p.payload or {})) for p in points]
    return {
        "tool": "filter_scroll",
        "filters": {k: v for k, v in {
            "brand": brand, "category": category, "subcategory": subcategory,
            "species": species, "price_min": price_min, "price_max": price_max,
            "size_label": size_label,
        }.items() if v is not None},
        "total_count": total_count,
        "count": len(products),
        "products": products,
    }


def count_products(
    *,
    group_by: str,
    brand: str | None = None,
    category: str | None = None,
    species: str | None = None,
    top_n: int = 30,
    qdrant: QdrantClient,
    collection: str,
) -> dict[str, Any]:
    flt = _build_filter(brand=brand, category=category, species=species)

    counts: dict[str, int] = {}
    offset = None
    scanned = 0
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=512,
            with_payload=[group_by, "chunk_type"],
            offset=offset,
        )
        if not points:
            break
        for p in points:
            payload = dict(p.payload or {})
            # Respect the product_sku_row scope even when no filter was built
            if payload.get("chunk_type") and payload.get("chunk_type") != "product_sku_row":
                continue
            value = payload.get(group_by)
            if value is None:
                key = "<null>"
            elif isinstance(value, list):
                for v in value:
                    k = str(v)
                    counts[k] = counts.get(k, 0) + 1
                scanned += 1
                continue
            else:
                key = str(value)
            counts[key] = counts.get(key, 0) + 1
        scanned += len(points)
        if offset is None:
            break

    ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    return {
        "tool": "count_products",
        "group_by": group_by,
        "filters": {k: v for k, v in {
            "brand": brand, "category": category, "species": species,
        }.items() if v is not None},
        "total_scanned": scanned,
        "distinct_values": len(counts),
        "counts": [{"value": k, "count": v} for k, v in ordered],
    }


def get_product(
    *,
    sku: str | None = None,
    ean: str | None = None,
    qdrant: QdrantClient,
    collection: str,
) -> dict[str, Any]:
    if not sku and not ean:
        return {"tool": "get_product", "error": "Provide sku or ean."}
    must = [FieldCondition(key="chunk_type", match=MatchValue(value="product_sku_row"))]
    if sku:
        must.append(FieldCondition(key="sku", match=MatchValue(value=sku.strip())))
    elif ean:
        must.append(FieldCondition(key="ean", match=MatchValue(value=ean.strip())))
    flt = Filter(must=must)
    points, _ = qdrant.scroll(
        collection_name=collection,
        scroll_filter=flt,
        limit=5,
        with_payload=True,
    )
    products = [_product_summary(dict(p.payload or {})) for p in points]
    return {
        "tool": "get_product",
        "lookup": {"sku": sku, "ean": ean},
        "count": len(products),
        "products": products,
    }


def list_distinct_values(
    *,
    field: str,
    brand: str | None = None,
    category: str | None = None,
    limit: int = 100,
    qdrant: QdrantClient,
    collection: str,
) -> dict[str, Any]:
    flt = _build_filter(brand=brand, category=category)
    values: dict[str, int] = {}
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=512,
            with_payload=[field, "chunk_type"],
            offset=offset,
        )
        if not points:
            break
        for p in points:
            payload = dict(p.payload or {})
            if payload.get("chunk_type") and payload.get("chunk_type") != "product_sku_row":
                continue
            v = payload.get(field)
            if v is None:
                continue
            if isinstance(v, list):
                for x in v:
                    k = str(x)
                    values[k] = values.get(k, 0) + 1
            else:
                k = str(v)
                values[k] = values.get(k, 0) + 1
        if offset is None:
            break
    ordered = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return {
        "tool": "list_distinct_values",
        "field": field,
        "scope": {k: v for k, v in {"brand": brand, "category": category}.items() if v is not None},
        "distinct_count": len(values),
        "values": [{"value": k, "count": v} for k, v in ordered],
    }


def fit_search(
    *,
    neck_cm: float | None = None,
    chest_cm: float | None = None,
    body_cm: float | None = None,
    dog_weight_kg: float | None = None,
    cat_weight_kg: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    species: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    limit: int = 10,
    qdrant: QdrantClient,
    collection: str,
) -> dict[str, Any]:
    """
    Range-containment search: returns products whose fit window *contains* the
    given measurement. A 35 cm neck matches a collar iff neck_min_cm <= 35 and
    neck_max_cm >= 35.
    """
    must: list[FieldCondition] = [
        FieldCondition(key="chunk_type", match=MatchValue(value="product_sku_row"))
    ]

    def _add_range(min_key: str, max_key: str, value: float) -> None:
        must.append(FieldCondition(key=min_key, range=Range(lte=float(value))))
        must.append(FieldCondition(key=max_key, range=Range(gte=float(value))))

    if neck_cm is not None:
        _add_range("neck_min_cm", "neck_max_cm", neck_cm)
    if chest_cm is not None:
        _add_range("chest_min_cm", "chest_max_cm", chest_cm)
    if body_cm is not None:
        _add_range("body_min_cm", "body_max_cm", body_cm)
    if dog_weight_kg is not None:
        _add_range("dog_weight_min_kg", "dog_weight_max_kg", dog_weight_kg)
    if cat_weight_kg is not None:
        _add_range("cat_weight_min_kg", "cat_weight_max_kg", cat_weight_kg)

    if brand:
        must.append(FieldCondition(key="brand", match=MatchValue(value=brand)))
    if category:
        must.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if species:
        must.append(FieldCondition(key="species", match=MatchAny(any=[species])))
    if price_min is not None or price_max is not None:
        must.append(FieldCondition(
            key="price_pvpr",
            range=Range(
                gte=float(price_min) if price_min is not None else None,
                lte=float(price_max) if price_max is not None else None,
            ),
        ))

    must_not = [FieldCondition(
        key="change_flag",
        match=MatchAny(any=["ELIMINAR", "producto eliminado"]),
    )]

    fit_criteria = {k: v for k, v in {
        "neck_cm": neck_cm, "chest_cm": chest_cm, "body_cm": body_cm,
        "dog_weight_kg": dog_weight_kg, "cat_weight_kg": cat_weight_kg,
    }.items() if v is not None}

    if not fit_criteria:
        return {
            "tool": "fit_search",
            "error": "No measurement provided. Give at least one of neck_cm, chest_cm, body_cm, dog_weight_kg, cat_weight_kg.",
            "products": [],
            "count": 0,
        }

    flt = Filter(must=must, must_not=must_not)
    points, _ = qdrant.scroll(
        collection_name=collection,
        scroll_filter=flt,
        limit=int(limit),
        with_payload=True,
    )
    products = [_product_summary(dict(p.payload or {})) for p in points]
    return {
        "tool": "fit_search",
        "fit_criteria": fit_criteria,
        "filters": {k: v for k, v in {
            "brand": brand, "category": category, "species": species,
            "price_min": price_min, "price_max": price_max,
        }.items() if v is not None},
        "count": len(products),
        "products": products,
    }


def compare_products(
    *,
    skus: list[str],
    qdrant: QdrantClient,
    collection: str,
) -> dict[str, Any]:
    """
    Fetch a set of products by SKU and return them side-by-side, plus a
    `comparison` map of `field -> [value_for_each_sku]` across attributes
    that most often differentiate products (price, weight, size, color, etc.).
    """
    results: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    for sku in (skus or [])[:10]:
        s = (sku or "").strip()
        if not s:
            continue
        flt = Filter(must=[
            FieldCondition(key="chunk_type", match=MatchValue(value="product_sku_row")),
            FieldCondition(key="sku", match=MatchValue(value=s)),
        ])
        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=1,
            with_payload=True,
        )
        if points:
            results.append(_product_summary(dict(points[0].payload or {})))
        else:
            errors[s] = "not_found"
            results.append({"sku": s, "error": "not_found"})

    # Side-by-side comparison matrix.
    compare_keys = [
        "brand", "category", "subcategory", "species", "change_flag",
        "price_pvpr", "price_per_unit", "min_purchase_qty",
        "size_label", "color", "scent",
        "weight_g", "length_cm", "width_cm", "height_cm",
        "neck_cm", "body_cm", "chest_cm",
        "dog_weight_kg", "cat_weight_kg",
    ]
    comparison: dict[str, list[Any]] = {}
    for k in compare_keys:
        values = [p.get(k) for p in results]
        if any(v is not None for v in values):
            comparison[k] = values

    return {
        "tool": "compare_products",
        "skus": [(p.get("sku") or "") for p in results],
        "count": len(results),
        "errors": errors,
        "products": results,
        "comparison": comparison,
    }


def render_compare_products_markdown(compare_result: dict[str, Any], *, user_language: str | None = None) -> str:
    """
    Deterministically render the output of `compare_products()` as a Markdown table.
    """
    lang = (user_language or "").strip().lower()
    if not lang or len(lang) > 8:
        lang = "unknown"
    if lang == "unknown":
        lang = "es"

    labels_es = {
        "field": "Campo",
        "name_es": "Nombre",
        "ean": "EAN",
        "category": "Categoría",
        "subcategory": "Subcategoría",
        "species": "Especie",
        "price_pvpr": "Precio (PVPR)",
        "price_per_unit": "Precio por unidad",
        "min_purchase_qty": "Compra mínima",
        "size_label": "Talla",
        "color": "Color",
        "scent": "Aroma",
        "weight_g": "Peso (g)",
        "length_cm": "Largo (cm)",
        "width_cm": "Ancho (cm)",
        "height_cm": "Alto (cm)",
        "neck_cm": "Cuello (cm)",
        "chest_cm": "Pecho (cm)",
        "body_cm": "Cuerpo (cm)",
        "dog_weight_kg": "Peso perro (kg)",
        "cat_weight_kg": "Peso gato (kg)",
        "change_flag": "Estado",
    }
    labels_en = {
        "field": "Field",
        "name_es": "Name",
        "ean": "EAN",
        "category": "Category",
        "subcategory": "Subcategory",
        "species": "Species",
        "price_pvpr": "Price (PVPR)",
        "price_per_unit": "Price per unit",
        "min_purchase_qty": "Min purchase qty",
        "size_label": "Size",
        "color": "Color",
        "scent": "Scent",
        "weight_g": "Weight (g)",
        "length_cm": "Length (cm)",
        "width_cm": "Width (cm)",
        "height_cm": "Height (cm)",
        "neck_cm": "Neck (cm)",
        "chest_cm": "Chest (cm)",
        "body_cm": "Body (cm)",
        "dog_weight_kg": "Dog weight (kg)",
        "cat_weight_kg": "Cat weight (kg)",
        "change_flag": "Status",
    }
    labels = labels_es if lang.startswith("es") else labels_en

    products = compare_result.get("products") or []
    if not isinstance(products, list) or not products:
        return ""

    def _escape_md(v: str) -> str:
        return (v or "").replace("|", "\\|").replace("\n", " ").strip()

    def _fmt_range(v: Any) -> str:
        if not (isinstance(v, list) and len(v) == 2):
            return ""
        lo, hi = v[0], v[1]
        if lo is None and hi is None:
            return "—"
        if lo is None:
            return f"≤ {hi}"
        if hi is None:
            return f"≥ {lo}"
        if lo == hi:
            return str(lo)
        return f"{lo}–{hi}"

    def _fmt_value(key: str, v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, str):
            s = v.strip()
            return _escape_md(s) if s else "—"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if key in {"price_pvpr", "price_per_unit"}:
                return f"€{v:.2f}"
            return f"{v:.2f}".rstrip("0").rstrip(".")
        if isinstance(v, list):
            if len(v) == 2 and all((isinstance(x, (int, float)) or x is None) for x in v):
                return _fmt_range(v)
            return _escape_md(", ".join(str(x) for x in v if x is not None)) or "—"
        return _escape_md(str(v)) or "—"

    headers: list[str] = [labels["field"]]
    for p in products:
        brand = str(p.get("brand") or "").strip()
        sku = str(p.get("sku") or "").strip()
        if brand and sku:
            headers.append(_escape_md(f"{brand} · {sku}"))
        elif sku:
            headers.append(_escape_md(sku))
        else:
            headers.append("—")

    row_keys = [
        "name_es",
        "ean",
        "category",
        "subcategory",
        "species",
        "price_pvpr",
        "price_per_unit",
        "min_purchase_qty",
        "size_label",
        "color",
        "scent",
        "weight_g",
        "length_cm",
        "width_cm",
        "height_cm",
        "neck_cm",
        "chest_cm",
        "body_cm",
        "dog_weight_kg",
        "cat_weight_kg",
        "change_flag",
    ]

    rows: list[list[str]] = []
    for k in row_keys:
        vals = [_fmt_value(k, p.get(k)) for p in products]
        if all(v == "—" for v in vals):
            continue
        rows.append([_escape_md(labels.get(k, k))] + vals)

    if not rows:
        return ""

    out_lines = []
    out_lines.append("| " + " | ".join(headers) + " |")
    out_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(out_lines)


def build_budget_basket(
    *,
    budget_eur: float,
    brand: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    species: str | None = None,
    min_items: int = 3,
    max_items: int = 8,
    diversity: str = "subcategory",
    qdrant: QdrantClient,
    collection: str,
) -> dict[str, Any]:
    """
    Compose a shopping basket that fits within `budget_eur`, respecting each
    product's minimum purchase quantity.

      1. Scroll Qdrant with the given filters (discontinued items excluded).
      2. Keep only items with non-null price_pvpr AND min_purchase_qty.
      3. Compute line_total = price_pvpr × min_purchase_qty.
      4. Drop items whose line_total alone exceeds the budget.
      5. Pass 1 — ascending greedy with diversity constraint; buys variety fast
         and satisfies `min_items`.
      6. Pass 2 — best-fit-decreasing top-up: sort remaining items descending by
         line_total and keep adding the largest one that still fits residual
         budget. Avoids the cheapest-first stall that leaves large budgets
         half-empty.

    `max_items` is auto-scaled from the budget (`effective_max_items`) so
    callers don't have to guess a ceiling proportional to spend.
    """
    try:
        budget_eur = float(budget_eur)
    except (TypeError, ValueError):
        return {"tool": "build_budget_basket", "error": "budget_eur must be a number.", "basket": [], "total_cost": 0}
    if budget_eur <= 0:
        return {"tool": "build_budget_basket", "error": "budget_eur must be positive.", "basket": [], "total_cost": 0}

    try:
        min_items = max(1, int(min_items))
        max_items = max(min_items, int(max_items))
    except (TypeError, ValueError):
        min_items, max_items = 3, 8
    if diversity not in {"subcategory", "brand", "none"}:
        diversity = "subcategory"

    # Auto-scale the item cap from the budget. Small baskets (€50) keep the
    # caller's cap; large baskets (€200, €600) need more slots or ascending
    # cheapest-first fill stalls long before reaching 90% budget use.
    effective_max_items = min(max(max_items, int(budget_eur // 8), min_items), 30)

    flt = _build_filter(
        brand=brand, category=category, subcategory=subcategory, species=species,
    )

    # 1. Fetch all candidates matching filters.
    candidates: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=512,
            with_payload=True,
            offset=offset,
        )
        if not points:
            break
        for p in points:
            candidates.append(dict(p.payload or {}))
        if offset is None:
            break

    # 2–4. Keep only items with numeric price+qty and line_total ≤ budget.
    items: list[dict[str, Any]] = []
    for c in candidates:
        price = c.get("price_pvpr") if c.get("price_pvpr") is not None else c.get("price_eur")
        qty = c.get("min_purchase_qty") if c.get("min_purchase_qty") is not None else c.get("min_order")
        if price is None or qty is None:
            continue
        try:
            price_f = float(price)
            qty_i = int(qty)
        except (TypeError, ValueError):
            continue
        # Skip items with no real price signal. `price_pvpr: 0.0` is used in the
        # catalog for "price TBD"; including it in a budget basket is misleading.
        if price_f <= 0:
            continue
        if qty_i < 1:
            qty_i = 1
        line_total = round(price_f * qty_i, 2)
        if line_total > budget_eur:
            continue
        items.append({
            "brand": c.get("brand"),
            "sku": c.get("sku"),
            "name_es": c.get("name_es") or c.get("product_name") or (c.get("names") or {}).get("es"),
            "category": c.get("category"),
            "subcategory": c.get("subcategory"),
            "unit_price": price_f,
            "min_qty": qty_i,
            "line_total": line_total,
        })

    items_considered = len(items)
    if not items:
        return {
            "tool": "build_budget_basket",
            "budget_eur": round(budget_eur, 2),
            "items_considered": 0,
            "basket": [],
            "total_cost": 0.0,
            "budget_used_pct": 0.0,
            "remaining_budget": round(budget_eur, 2),
            "reason": "No products fit within the budget after applying minimum order quantities.",
            "filters_applied": {k: v for k, v in {
                "brand": brand, "category": category, "subcategory": subcategory,
                "species": species, "diversity": diversity,
            }.items() if v is not None},
        }

    # 5. Sort ascending by line_total.
    items.sort(key=lambda x: x["line_total"])

    # 6. Pass 1 — diversity-respecting greedy.
    basket: list[dict[str, Any]] = []
    running = 0.0
    used_subcats: set[str] = set()
    used_brands: set[str] = set()
    chosen_skus: set[str] = set()

    def _diversity_blocks(item: dict[str, Any]) -> bool:
        if diversity == "subcategory":
            key = str(item.get("subcategory") or "")
            return bool(key) and key in used_subcats
        if diversity == "brand":
            key = str(item.get("brand") or "")
            return bool(key) and key in used_brands
        return False

    for item in items:
        if len(basket) >= effective_max_items:
            break
        if running + item["line_total"] > budget_eur + 1e-6:
            continue
        if _diversity_blocks(item):
            continue
        basket.append(item)
        running = round(running + item["line_total"], 2)
        chosen_skus.add(str(item["sku"] or ""))
        if item.get("subcategory"):
            used_subcats.add(str(item["subcategory"]))
        if item.get("brand"):
            used_brands.add(str(item["brand"]))

    # 7. Pass 2 — best-fit-decreasing top-up. If budget is underfilled, sort
    # remaining items descending by line_total and greedily add the largest
    # one that still fits. Repeat until no unused item fits the residual
    # budget. This kills the classic "cheapest-first fills with tiny items
    # then stalls" failure mode on large budgets.
    relaxed = False
    if running < budget_eur * 0.9 and len(basket) < effective_max_items:
        remaining_items = [
            it for it in items if str(it["sku"] or "") not in chosen_skus
        ]
        remaining_items.sort(key=lambda x: x["line_total"], reverse=True)
        while len(basket) < effective_max_items:
            picked = None
            for item in remaining_items:
                if str(item["sku"] or "") in chosen_skus:
                    continue
                if running + item["line_total"] > budget_eur + 1e-6:
                    continue
                picked = item
                break
            if picked is None:
                break
            basket.append(picked)
            running = round(running + picked["line_total"], 2)
            chosen_skus.add(str(picked["sku"] or ""))
            relaxed = True

    used_pct = round(100.0 * running / budget_eur, 1) if budget_eur > 0 else 0.0
    if used_pct >= 90:
        reason = "Greedy composition; budget filled to ≥90%."
    elif relaxed:
        reason = f"Budget partially filled ({used_pct}%); diversity was relaxed but no cheaper item fits the remaining {round(budget_eur - running, 2)}€."
    elif len(basket) < min_items:
        reason = (
            f"Only {len(basket)} item(s) could be fit within {budget_eur}€ given minimum "
            f"order quantities; falls short of min_items={min_items}. Consider a higher budget."
        )
    else:
        reason = f"Budget filled to {used_pct}%; no further item fits the remaining {round(budget_eur - running, 2)}€."

    return {
        "tool": "build_budget_basket",
        "budget_eur": round(budget_eur, 2),
        "items_considered": items_considered,
        "basket": basket,
        "total_cost": round(running, 2),
        "budget_used_pct": used_pct,
        "remaining_budget": round(budget_eur - running, 2),
        "reason": reason,
        "filters_applied": {k: v for k, v in {
            "brand": brand, "category": category, "subcategory": subcategory,
            "species": species, "diversity": diversity,
        }.items() if v is not None},
    }


# --------------------------------------------------------------------------
# Tool dispatcher
# --------------------------------------------------------------------------
def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    qdrant: QdrantClient,
    collection: str,
    embedder: Any,
    bm25: Any,
    bm25_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        if name == "semantic_search":
            return semantic_search(qdrant=qdrant, collection=collection,
                                   embedder=embedder, bm25=bm25,
                                   bm25_chunks=bm25_chunks, **arguments)
        if name == "filter_scroll":
            return filter_scroll(qdrant=qdrant, collection=collection, **arguments)
        if name == "count_products":
            return count_products(qdrant=qdrant, collection=collection, **arguments)
        if name == "get_product":
            return get_product(qdrant=qdrant, collection=collection, **arguments)
        if name == "list_distinct_values":
            return list_distinct_values(qdrant=qdrant, collection=collection, **arguments)
        if name == "fit_search":
            return fit_search(qdrant=qdrant, collection=collection, **arguments)
        if name == "compare_products":
            return compare_products(qdrant=qdrant, collection=collection, **arguments)
        if name == "build_budget_basket":
            return build_budget_basket(qdrant=qdrant, collection=collection, **arguments)
        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tool": name, "arguments": arguments}


# --------------------------------------------------------------------------
# Tool-calling LLM loop (OpenAI-compatible via OpenRouter HTTP API)
# --------------------------------------------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _openrouter_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    model: str,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "ragbot-poc",
        "X-Title": "ragbot-poc",
    }
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
    }
    with httpx.Client(timeout=timeout) as c:
        r = c.post(OPENROUTER_URL, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


def run_tool_loop(
    *,
    user_query: str,
    history: list[dict[str, str]] | None,
    system_prompt: str,
    qdrant: QdrantClient,
    collection: str,
    embedder: Any,
    bm25: Any,
    bm25_chunks: list[dict[str, Any]],
    model: str | None = None,
    max_rounds: int = 4,
) -> dict[str, Any]:
    """Run the tool-calling loop. Returns {answer, tool_trace, retrieved_products}.

    The LLM sees the schema of each tool plus the user query, chooses one (or
    more in sequence), the backend executes it, the result is fed back as a
    role=tool message, and the loop continues until the model stops.
    """
    model = model or os.getenv("OPENROUTER_TOOLS_MODEL",
                                os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in (history or [])[-10:]:
        if m.get("role") in {"user", "assistant"} and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_query})

    tool_trace: list[dict[str, Any]] = []
    retrieved_products: list[dict[str, Any]] = []
    answer = ""

    for round_idx in range(max_rounds):
        response = _openrouter_with_tools(messages, TOOL_SCHEMAS, model=model)
        choice = (response.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        finish = choice.get("finish_reason") or ""

        # Always record the assistant message so tool-results can be attached to it.
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.get("content"):
            assistant_msg["content"] = msg["content"]
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            answer = msg.get("content") or ""
            break

        for tc in tool_calls:
            name = (tc.get("function") or {}).get("name") or ""
            args_raw = (tc.get("function") or {}).get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except json.JSONDecodeError:
                args = {}

            result = dispatch_tool(name, args,
                                   qdrant=qdrant, collection=collection,
                                   embedder=embedder, bm25=bm25,
                                   bm25_chunks=bm25_chunks)

            if name == "compare_products" and isinstance(result, dict):
                compare_result = result

            tc_total = result.get("total_count")
            if isinstance(tc_total, int) and tc_total >= 0:
                sources_total = tc_total if sources_total is None else max(sources_total, tc_total)

            tool_trace.append({
                "round": round_idx,
                "name": name,
                "arguments": args,
                "result_summary": {
                    "count": result.get("count"),
                    "total_count": result.get("total_count"),
                    "distinct_values": result.get("distinct_values"),
                    "tool": result.get("tool"),
                    "error": result.get("error"),
                },
            })

            # Collect products for the UI "sources" panel.
            for p in (result.get("products") or [])[:8]:
                retrieved_products.append(p)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or "",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

        if finish == "stop":
            break

    return {
        "answer": answer,
        "tool_trace": tool_trace,
        "retrieved_products": retrieved_products,
        "sources_total": sources_total,
        "compare_result": compare_result,
        "model": model,
        "rounds": len(tool_trace),
    }


def run_tool_loop_stream(
    *,
    user_query: str,
    history: list[dict[str, str]] | None,
    system_prompt: str,
    qdrant: QdrantClient,
    collection: str,
    embedder: Any,
    bm25: Any,
    bm25_chunks: list[dict[str, Any]],
    model: str | None = None,
    max_rounds: int = 4,
) -> Iterator[dict[str, Any]]:
    """
    Streaming-friendly variant of `run_tool_loop`.

    Yields events of shape:
      - {type: "status", message: str}
      - {type: "phase", phase: str, ...}
      - {type: "tool_start", tool: str}
      - {type: "tool_end", tool: str, count?: int, total_count?: int, error?: str}
      - {type: "done_raw", answer: str, tool_trace: [...], retrieved_products: [...], model: str, rounds: int}
      - {type: "error", message: str}
    """
    model = model or os.getenv("OPENROUTER_TOOLS_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in (history or [])[-10:]:
        if m.get("role") in {"user", "assistant"} and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_query})

    tool_trace: list[dict[str, Any]] = []
    retrieved_products: list[dict[str, Any]] = []
    answer = ""
    sources_total: int | None = None
    compare_result: dict[str, Any] | None = None

    try:
        for round_idx in range(max_rounds):
            yield {"type": "phase", "phase": "planning", "round": round_idx + 1, "max_rounds": max_rounds}
            response = _openrouter_with_tools(messages, TOOL_SCHEMAS, model=model)
            choice = (response.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            finish = choice.get("finish_reason") or ""

            # Always record the assistant message so tool-results can be attached to it.
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if msg.get("content"):
                assistant_msg["content"] = msg["content"]
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                answer = msg.get("content") or ""
                break

            for tc in tool_calls:
                name = (tc.get("function") or {}).get("name") or ""
                if name:
                    yield {"type": "tool_start", "tool": name}

                args_raw = (tc.get("function") or {}).get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except json.JSONDecodeError:
                    args = {}

                result = dispatch_tool(
                    name,
                    args,
                    qdrant=qdrant,
                    collection=collection,
                    embedder=embedder,
                    bm25=bm25,
                    bm25_chunks=bm25_chunks,
                )

                if name == "compare_products" and isinstance(result, dict):
                    compare_result = result

                tc_total = result.get("total_count")
                if isinstance(tc_total, int) and tc_total >= 0:
                    sources_total = tc_total if sources_total is None else max(sources_total, tc_total)

                tool_trace.append(
                    {
                        "round": round_idx,
                        "name": name,
                        "arguments": args,
                        "result_summary": {
                            "count": result.get("count"),
                            "total_count": result.get("total_count"),
                            "distinct_values": result.get("distinct_values"),
                            "tool": result.get("tool"),
                            "error": result.get("error"),
                        },
                    }
                )

                # Collect products for the UI "sources" panel.
                products = (result.get("products") or [])[:8]
                for p in products:
                    retrieved_products.append(p)

                count = result.get("count")
                tool_end: dict[str, Any] = {"type": "tool_end", "tool": name}
                if isinstance(count, int):
                    tool_end["count"] = count
                if isinstance(tc_total, int) and tc_total >= 0:
                    tool_end["total_count"] = tc_total
                if result.get("error"):
                    tool_end["error"] = str(result.get("error"))
                if name:
                    yield tool_end

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id") or "",
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

            if finish == "stop":
                break

        yield {"type": "phase", "phase": "finalizing"}
        yield {
            "type": "done_raw",
            "answer": answer,
            "tool_trace": tool_trace,
            "retrieved_products": retrieved_products,
            "sources_total": sources_total,
            "compare_result": compare_result,
            "model": model,
            "rounds": len(tool_trace),
        }
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
