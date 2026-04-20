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
from typing import Any

import httpx
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

SPECIES_ENUM = ["dog", "cat"]

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
    return {
        "id": payload.get("id") or payload.get("chunk_id"),
        "brand": payload.get("brand"),
        "sku": payload.get("sku"),
        "ean": payload.get("ean"),
        "name_es": name,
        "category": payload.get("category"),
        "subcategory": payload.get("subcategory"),
        "species": payload.get("species"),
        "price_pvpr": payload.get("price_pvpr") or payload.get("price_eur"),
        "price_per_unit": payload.get("price_per_unit"),
        "min_purchase_qty": payload.get("min_purchase_qty") or payload.get("min_order"),
        "size_label": payload.get("size_label"),
        "color": payload.get("color"),
        "scent": payload.get("scent"),
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
def build_tool_system_prompt() -> str:
    return (
        "You are a sales assistant for Gloriapets, a wholesale pet products distributor. "
        "You have tool access to query a catalog of ~2,890 product variants across 30 brands, "
        "stored in Qdrant. **Every factual claim must come from a tool call result.** "
        "If a tool returns zero results, say so — do not invent products, SKUs, prices, or attributes.\n\n"

        "## When to call which tool\n"
        "- Semantic / descriptive lookup ('red collar for small dog', 'KONG chew toys', "
        "'champús sin fragancia') → `semantic_search`.\n"
        "- Strict enumeration ('list all KONG chew toys', 'show products under 20€') → `filter_scroll`.\n"
        "- Aggregation / counting ('which brand has the most products', 'how many dog vs cat items', "
        "'número de productos de KONG') → `count_products`.\n"
        "- Fit queries with explicit measurements ('collar for 35 cm neck', 'harness for 12 kg dog', "
        "'arnés para perro de 8 kg') → `fit_search`. ALWAYS prefer fit_search over semantic_search "
        "when the user gives an exact neck / chest / body / weight number — it does true "
        "range-containment filtering instead of guessing.\n"
        "- Product comparison ('compare KONG Classic vs NYLABONE Extreme', "
        "'cuál es más barato, X o Y') → `compare_products`.\n"
        "- Exact lookup by SKU or EAN/barcode → `get_product`.\n"
        "- Catalog facets ('what colors does RED DINGO sell', 'list all subcategories under toys') "
        "→ `list_distinct_values`.\n"
        "You may call multiple tools in sequence. Prefer one well-targeted call over many.\n\n"

        "## Answering rules\n"
        "1. Respond in the user's language.\n"
        "2. Cite products as `Brand · SKU · name_es · €price`. Never cite by page number — the Excel "
        "catalog has no page numbers.\n"
        "3. One product per line when listing.\n"
        "4. For fit queries (neck, dog/cat weight, chest, body): use the range fields in tool output. "
        "A 35 cm neck fits iff neck_cm[0] ≤ 35 ≤ neck_cm[1]. A null range means the attribute "
        "doesn't apply to that product — say so explicitly rather than guessing.\n"
        "5. Discontinued products (change_flag ∈ {ELIMINAR, 'producto eliminado'}) are filtered out "
        "at the tool level by default. Don't mention them unless the user asks for deleted items.\n"
        "6. Keep answers concise; avoid marketing fluff.\n"
        "7. If the question is completely off-topic (weather, news, math), decline and redirect.\n\n"

        "## Permissible filter values\n"
        "- brand: " + ", ".join(BRAND_ENUM) + "\n"
        "- category: " + ", ".join(CATEGORY_ENUM) + "\n"
        "- species: dog, cat\n"
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

            tool_trace.append({
                "round": round_idx,
                "name": name,
                "arguments": args,
                "result_summary": {
                    "count": result.get("count"),
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
        "model": model,
        "rounds": len(tool_trace),
    }
