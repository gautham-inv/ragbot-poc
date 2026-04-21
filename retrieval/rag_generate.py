from __future__ import annotations

import argparse
import contextlib
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when executed as a script, and avoid
# shadowing stdlib modules by the script directory (e.g. stale __pycache__/tokenize*.pyc).
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
while str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))

from config import load_project_env
from openrouter import OpenRouter
from qdrant_client import QdrantClient

from retrieval.hybrid_search import _load_bm25_bundle, bm25_search, qdrant_search
from retrieval.product_dictionary import enrich_query_with_product_names
from retrieval.rrf import reciprocal_rank_fusion

load_project_env()

# Prevent noisy HF warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

_SKU_PATTERN = r"\b[A-Z]{2,}[A-Z0-9/*.-]{1,}\b"


def _get_langfuse_client():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client  # type: ignore

        return get_client()
    except Exception:
        return None


def _langfuse_add_trace_tags(langfuse, *, trace_id: str | None, tags: list[str]) -> None:
    if not langfuse or not trace_id or not tags:
        return

    deduped: list[str] = []
    for t in tags:
        t = (t or "").strip()
        if not t or t in deduped:
            continue
        deduped.append(t)
        if len(deduped) >= 50:
            break

    if not deduped:
        return

    try:
        fn = getattr(langfuse, "_create_trace_tags_via_ingestion", None)
        if callable(fn):
            fn(trace_id=trace_id, tags=deduped)
            return
    except Exception:
        pass

    try:
        from langfuse import propagate_attributes  # type: ignore

        with propagate_attributes(tags=deduped):
            pass
    except Exception:
        pass


def build_context_str(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    context_str = ""
    for i, c in enumerate(chunks, 1):
        meta = {}
        if isinstance(c.get("metadata"), dict):
            meta = c.get("metadata") or {}
        elif isinstance(c.get("meta"), dict):
            meta = c.get("meta") or {}
        elif isinstance(c, dict):
            # Qdrant payloads are often already "flat".
            meta = c

        def _get(k: str) -> Any:
            return meta.get(k) if meta.get(k) is not None else c.get(k)

        chunk_type = _get("chunk_type") or "unknown"
        sku = _get("sku")
        brand = _get("brand")
        product_name = _get("product_name") or _get("name_es") or _get("name_en")
        category = _get("category")
        subcategory = _get("subcategory")
        species = _get("species")
        price_eur = _get("price_eur") if _get("price_eur") is not None else _get("price_pvpr")
        size_label = _get("size_label")
        chunk_id = _get("chunk_id") or c.get("id")

        # OCR-era fallbacks (still present on pre-Excel collections):
        physical_page = _get("physical_page_number") or _get("page_number")
        sub_page = _get("sub_page_number")

        header_parts = [f"type={chunk_type}"]
        if brand:
            header_parts.append(f"Brand {brand}")
        if sku:
            header_parts.append(f"SKU {sku}")
        if product_name:
            header_parts.append(f"Product {product_name}")
        if category:
            if subcategory:
                header_parts.append(f"Category {category}/{subcategory}")
            else:
                header_parts.append(f"Category {category}")
        if species:
            species_str = "/".join(species) if isinstance(species, list) else str(species)
            header_parts.append(f"Species {species_str}")
        if size_label:
            header_parts.append(f"Size {size_label}")
        if price_eur is not None:
            header_parts.append(f"Price {price_eur}€")
        if physical_page is not None:
            header_parts.append(f"Physical Page {physical_page}")
            if sub_page is not None:
                header_parts.append(f"Sub-page {sub_page}")
        if chunk_id:
            header_parts.append(f"id {chunk_id}")

        context_str += f"--- Chunk {i} ({' | '.join(header_parts)}) ---\n"
        context_str += (c.get("text") or "") + "\n\n"
    return context_str


def _looks_like_sku(value: str) -> bool:
    if not value:
        return False
    return any(ch.isalpha() for ch in value) and any(ch.isdigit() for ch in value)


def _extract_skus_from_query(query: str) -> set[str]:
    import re

    out: set[str] = set()
    for candidate in re.findall(_SKU_PATTERN, (query or "").upper()):
        if _looks_like_sku(candidate):
            out.add(candidate)
    return out


_BRAND_ENUM = {
    "SKINNIA", "KVP", "PET HEAD", "CEVA", "CHIEN CHIC", "COACHI",
    "COMPANY OF ANIMALS", "DGS", "3 CLAVELES", "ANDIS", "VAN CAT",
    "MEN FOR SAN", "YOWUP", "LICKIMAT", "SPRENGER", "RED DINGO",
    "CHEWLLAGEN", "KONG", "LICENCIADOS KONG", "FLEXI", "NYLABONE",
    "ROTHO MYPET", "UNITED PETS", "EARTH RATED", "URINE OFF",
    "VETIQ", "HUNTER", "INODORINA", "COCOSI", "WHIMZEES",
}


def _extract_brand_from_query(query: str) -> str | None:
    q = (query or "").upper()
    # Longest-match first so "LICENCIADOS KONG" beats "KONG".
    for brand in sorted(_BRAND_ENUM, key=len, reverse=True):
        if brand in q:
            return brand
    return None


def _build_qdrant_filter_for_query(query: str):
    from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

    conditions = []
    skus = _extract_skus_from_query(query)
    if skus:
        # For SKU queries, only search SKU-anchored chunks.
        conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value="product_sku_row")))
    brand = _extract_brand_from_query(query)
    if brand:
        conditions.append(FieldCondition(key="brand", match=MatchValue(value=brand)))
    # Always exclude soft-deleted products unless the query explicitly asks for them.
    q_lower = (query or "").lower()
    wants_deleted = any(tok in q_lower for tok in ("eliminar", "eliminado", "deleted", "removed", "discontinued"))
    base_filter = Filter(must=conditions) if conditions else None
    if not wants_deleted:
        must_not = [FieldCondition(key="change_flag", match=MatchAny(any=["ELIMINAR", "producto eliminado"]))]
        if base_filter is None:
            return Filter(must_not=must_not)
        base_filter.must_not = must_not
    return base_filter


def _apply_local_payload_filters(payloads: list[dict], *, require_chunk_type: str | None, require_brand: str | None) -> list[dict]:
    out: list[dict] = []
    for p in payloads:
        if require_chunk_type and p.get("chunk_type") != require_chunk_type:
            continue
        if require_brand and p.get("brand") != require_brand:
            continue
        out.append(p)
    return out


def _qdrant_scroll_exact(
    client: QdrantClient,
    collection: str,
    *,
    skus: set[str] | None = None,
    brand: str | None = None,
    limit: int = 8,
) -> list[dict]:
    """
    Exact payload retrieval without embeddings (useful in offline mode).
    """
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    must = []
    must.append(FieldCondition(key="chunk_type", match=MatchValue(value="product_sku_row")))
    if brand:
        must.append(FieldCondition(key="brand", match=MatchValue(value=brand)))
    if skus:
        # Qdrant match-any varies by version; use multiple must-should filters by scrolling each SKU.
        payloads: list[dict] = []
        for sku in list(skus)[:3]:
            flt = Filter(must=must + [FieldCondition(key="sku", match=MatchValue(value=sku))])
            points, _ = client.scroll(collection_name=collection, scroll_filter=flt, limit=limit, with_payload=True)
            for p in points:
                payloads.append(dict(p.payload or {}))
        return payloads[:limit]

    flt = Filter(must=must) if must else None
    points, _ = client.scroll(collection_name=collection, scroll_filter=flt, limit=limit, with_payload=True)
    return [dict(p.payload or {}) for p in points]


def build_system_prompt(context_str: str, *, user_language: str | None = None) -> str:
    """Build the unified system prompt with strong guardrails.

    This is the single source of truth for the system prompt used by
    both the Streamlit app and the CLI/evaluation scripts.
    """
    lang = (user_language or "").strip().lower()
    if not lang or len(lang) > 8:
        lang = "unknown"
    # Default to Spanish for a Spanish company when routing cannot confidently detect a language.
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

    fallback_exact = {
        "en": "I don't have that information in the current catalog.",
        "es": "No tengo esa información en el catálogo actual.",
    }.get(lang)

    if fallback_exact:
        fallback_rule = f"2. If the answer isn't in the context, say exactly: '{fallback_exact}'\n"
    else:
        fallback_rule = (
            "2. If the answer isn't in the context, say exactly: 'No tengo esa información en el catálogo actual.' "
            "(translated to the user's language).\n"
        )

    return (
        "You are a sales assistant for Gloriapets, a wholesale pet products distributor.\n"
        "The catalog is the cleaned 2026 Excel (one record per product variant, ~2,890 rows, 30 brands).\n\n"

        "## OUTPUT LANGUAGE\n"
        f"- Detected user language (ISO code): {lang}\n"
        f"- You MUST write the entire answer in {language_name} only.\n\n"

        "## STRICT RULES (never override these):\n"
        "1. Answer ONLY using the CONTEXT. Never use outside knowledge.\n"
        f"{fallback_rule}"
        "3. NEVER invent or guess product names, prices, SKUs, barcodes, dimensions, or stock. "
        "If a field is missing from the chunk, say so — don't fabricate.\n"
        "4. Off-topic questions (weather, news, math) → decline and redirect to catalog questions.\n"
        "5. Cite by Brand + SKU exactly as they appear in the chunk header (e.g. 'KONG · KNG0001'). "
        "Do NOT cite by page number — the Excel catalog has no page numbering.\n"
        "6. When listing products, use one line per item: Brand · SKU · name_es · price_pvpr € (if known).\n"
        "7. Respond in the OUTPUT LANGUAGE specified above.\n"
        "8. Use conversation history for follow-ups; ground every fact in the CONTEXT.\n"
        "9. Prefer chunks with type=product_sku_row for concrete product answers.\n"
        "10. Products with change_flag ∈ {ELIMINAR, 'producto eliminado'} are discontinued and "
        "already filtered out at retrieval time. Do not recommend them unless the user explicitly asks for deleted items.\n"
        "11. Keep answers concise; avoid marketing fluff unless asked.\n\n"

        "## CATALOG SCHEMA — FIELDS YOU CAN RELY ON\n"
        "Every chunk is one product variant with a flat payload. Fields marked 'optional' are null "
        "when the attribute doesn't apply to that brand — **that is not missing data, it's by design**. "
        "Do not assume a null field means 'unknown'; for a field that's null on a brand that never has it, "
        "say the attribute doesn't apply to that product.\n\n"

        "### Required fields (present on every chunk)\n"
        "- brand (keyword). Permissible values (exactly these 30):\n"
        "    SKINNIA, KVP, PET HEAD, CEVA, CHIEN CHIC, COACHI, COMPANY OF ANIMALS, DGS, 3 CLAVELES, ANDIS,\n"
        "    VAN CAT, MEN FOR SAN, YOWUP, LICKIMAT, SPRENGER, RED DINGO, CHEWLLAGEN, KONG, LICENCIADOS KONG,\n"
        "    FLEXI, NYLABONE, ROTHO MYPET, UNITED PETS, EARTH RATED, URINE OFF, VETIQ, HUNTER, INODORINA,\n"
        "    COCOSI, WHIMZEES.\n"
        "- sku (keyword) — REFERENCIA column value.\n"
        "- ean (keyword) — barcode, 8–14 digits.\n"
        "- category (keyword). Permissible values:\n"
        "    grooming, accessories, nutrition, hygiene, training, toys, housing, healthcare, apparel, equipment.\n"
        "- subcategory (keyword, free-text but from a limited set). Common values:\n"
        "    collar, harness, leash, muzzle, shampoo, conditioner, fragrance, mousse, balm, brush, scissors,\n"
        "    nail_clipper, deshedder, grooming_mitt, spray, cleaning, wipes, poop_bags, litter, training_pad,\n"
        "    diaper, cleaner, lint_roller, litter_filter, chew_toy, ball, toy, plush, lick_mat, cat_toy,\n"
        "    launcher, bowl, water_dispenser, food_bag, pouch, car_restraint, car_barrier, seat_cover,\n"
        "    travel_mat, doormat, dispenser, blade, clipper, case, comb_set, whistle, clicker, target_stick,\n"
        "    dumbbell, tug_toy, treat_pouch, corrector, toilet_bell, bite_pillow, control_rod, pheromone,\n"
        "    diffuser, parasiticide, repellent, skin_care, supplement, dental_care, dental_chew, treat,\n"
        "    wet_food, wet_treat, multi_pack, display, coat, shirt, bed, litter_box, carrier, scoop,\n"
        "    reflector, refill, light, set.\n"
        "- species (list[keyword]) — one or two of: 'dog', 'cat'.\n"
        "- change_flag (keyword). Permissible: active, NUEVOS PRODUCTOS, MODIFICACIÓN, CAMBIOS HECHOS "
        "(ELIMINAR / 'producto eliminado' are pre-filtered at retrieval).\n"
        "- price_pvpr (float, €) — retail price.\n"
        "- min_purchase_qty (integer) — minimum order quantity.\n\n"

        "### Optional fields — FIT & SIZING (null on brands that don't have the attribute)\n"
        "- size_label (free text). NOT a clean enum: includes XS/S/M/L/XL/XXL *and* numeric codes "
        "('35', '40'…) *and* raw dimension strings ('6,35 x 6,35 x 6,35 cm'). Treat as free text; "
        "filter only when the user types an exact value.\n"
        "- neck_min_cm / neck_max_cm (float) — only on KVP, RED DINGO.\n"
        "- body_min_cm / body_max_cm (float) — only on RED DINGO.\n"
        "- chest_min_cm / chest_max_cm (float) — only on RED DINGO.\n"
        "- dog_weight_min_kg / dog_weight_max_kg (float) — only on KVP, CEVA, FLEXI, NYLABONE.\n"
        "- cat_weight_min_kg / cat_weight_max_kg (float) — only on CEVA, UNITED PETS.\n\n"

        "### Optional fields — PHYSICAL DIMENSIONS\n"
        "- length_cm, width_cm, height_cm (float) — present on COACHI, DGS, COMPANY OF ANIMALS, "
        "RED DINGO, HUNTER, SPRENGER, LICKIMAT, COCOSI, WHIMZEES, INODORINA (partial).\n"
        "- depth_cm (float) — KVP only.\n"
        "- thickness_cm (float) — HUNTER only.\n"
        "- leash_length_m (float) — FLEXI, COACHI.\n"
        "- weight_g (float) — PET HEAD, 3 CLAVELES, ANDIS, VAN CAT, YOWUP, KONG, LICENCIADOS KONG, "
        "RED DINGO, HUNTER, EARTH RATED, URINE OFF, INODORINA, COCOSI, WHIMZEES, COMPANY OF ANIMALS, DGS.\n\n"

        "### Optional fields — PRODUCT ATTRIBUTES\n"
        "- color (free text) — KVP, COACHI, DGS, LICKIMAT, ROTHO MYPET, HUNTER, UNITED PETS, KONG. "
        "Values are multilingual ('Rojo', 'Red', 'Negro'). Use semantic match; filter only on exact value.\n"
        "- scent (free text) — CHIEN CHIC, MEN FOR SAN, INODORINA, EARTH RATED.\n"
        "- breed_suitability (free text) — RED DINGO only. Don't filter on exact match; use semantic search.\n"
        "- dureza (keyword) — KONG only. Values: 'Suave', 'Duro'.\n"
        "- tipo (keyword) — INODORINA only. Values: 'Pelo Corto', 'Pelo Largo', 'Todas las razas'.\n"
        "- capacity_raw (free text, unparsed, e.g. '5 L', '600 ml') — RED DINGO, ROTHO MYPET, HUNTER, "
        "UNITED PETS, PET HEAD, CEVA, KONG. Not a filter; cite as-is.\n"
        "- price_per_unit (float, €) — unit price when sold in packs; present on KVP, KONG, CHEWLLAGEN, "
        "YOWUP, INODORINA.\n\n"

        "### Optional fields — ANDIS ONLY (grooming equipment)\n"
        "- watts (text, e.g. '100-240 V'), hz (text, e.g. '50/60 Hz'), cut_length_mm (float), "
        "recambio (keyword: 'Batería', 'Cable', 'Cargador').\n\n"

        "### Fit-query playbook\n"
        "- 'collar for a 35 cm neck' → match products where neck_min_cm ≤ 35 ≤ neck_max_cm. "
        "If neck_min_cm is null on a chunk, that product's fit is unknown — don't recommend it for fit.\n"
        "- 'for a 12 kg dog' → dog_weight_min_kg ≤ 12 ≤ dog_weight_max_kg.\n"
        "- 'for a 4 kg cat' → cat_weight_min_kg ≤ 4 ≤ cat_weight_max_kg.\n"
        "- If the user asks a fit question but the relevant range field is null on every retrieved chunk, "
        "state that the available products don't specify a fit range for that measurement.\n\n"

        "### Multilingual names\n"
        "Each chunk carries name_es (always), and name_en/name_fr/name_pt/name_it where the brand provides them. "
        "For display and citation, prefer name_es; mention an EN/FR/PT/IT translation only if it exists on the chunk.\n\n"

        "## CONTEXT:\n"
        + context_str
    )


def generate_answer(query: str, chunks: list[dict], openrouter_api_key: str) -> None:
    """Generates an answer using Qwen over OpenRouter, grounded in the retrieved chunks."""
    context_str = build_context_str(chunks)
    system_prompt = build_system_prompt(context_str)

    print("\n\n" + "=" * 50)
    print("Generating answer...")
    print("=" * 50 + "\n")

    with OpenRouter(
        http_referer="ragbot-poc",
        x_open_router_title="ragbot-poc",
        api_key=openrouter_api_key,
    ) as open_router:
        res = open_router.chat.send(
            model="qwen/qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            stream=False,
            temperature=0.0,
        )

        if hasattr(res, "choices") and len(res.choices) > 0:
            print(res.choices[0].message.content)
        else:
            print(res)
        print("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG Generation (Qdrant + BM25 + Qwen Plus via OpenRouter).")
    ap.add_argument("--query", required=True)
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", None))
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "catalog_es"))
    ap.add_argument("--bm25", default="data/index/bm25.pkl", type=Path)
    ap.add_argument(
        "--model",
        default=os.getenv(
            "EMBEDDING_MODEL",
            os.getenv("HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        ),
    )
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: The OPENROUTER_API_KEY environment variable is not defined.")
        print("Use: $env:OPENROUTER_API_KEY='your_key' in PowerShell.")
        return 1

    langfuse = _get_langfuse_client()
    if langfuse:
        print("[Langfuse] enabled")
    else:
        print("[Langfuse] disabled (missing keys or client)")

    print("[1/3] Loading search models...")
    bm25_bundle = _load_bm25_bundle(args.bm25)
    bm25 = bm25_bundle["bm25"]
    bm25_chunks = bm25_bundle["chunks"]
    product_dictionary = bm25_bundle.get("product_dictionary", {})
    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)

    model = None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        # If the model isn't already cached locally, this may attempt a download.
        # We fall back to BM25 and/or exact Qdrant payload lookup for SKU queries.
        model = SentenceTransformer(args.model, device="cpu")
    except Exception as e:
        print(f"WARNING: Could not load embedding model ({args.model}). Falling back to BM25/exact SKU lookup.")
        print(f"         Error: {e}")

    enriched_query = enrich_query_with_product_names(args.query, product_dictionary)
    skus_in_query = _extract_skus_from_query(enriched_query)
    gloria_context = _query_mentions_gloria_context(enriched_query)

    root_obs = (
        langfuse.start_as_current_observation(as_type="span", name="rag_query")  # type: ignore[attr-defined]
        if langfuse
        else contextlib.nullcontext()
    )
    with root_obs as root_span:
        if root_span is not None:
            try:
                root_span.update(
                    input={
                        "query": args.query,
                        "enriched_query": enriched_query,
                        "skus": sorted(skus_in_query),
                        "gloria_context": gloria_context,
                    }
                )
            except Exception:
                pass

        top_k = 16 if skus_in_query else 8
        top_n = 6 if skus_in_query else 4

        print(f"[2/3] Searching information for: '{enriched_query}'...")
        # NOTE: We apply chunk_type/brand filters locally to avoid requiring Qdrant payload indexes.
        require_chunk_type = "product_sku_row" if skus_in_query else None
        require_brand = "GLORIA" if gloria_context else None
        qdrant_filter = None

        retrieval_obs = (
            langfuse.start_as_current_observation(as_type="span", name="retrieval")  # type: ignore[attr-defined]
            if langfuse
            else contextlib.nullcontext()
        )
        with retrieval_obs as retrieval_span:
            vec = []
            if model is not None:
                vec = qdrant_search(client, args.collection, model, enriched_query, top_k=top_k, qdrant_filter=qdrant_filter)
            kw = bm25_search(bm25, bm25_chunks, enriched_query, top_k=top_k)
            fused = reciprocal_rank_fusion(vec, kw, top_n=top_n)

            retrieved_chunks = [item.payload for item in fused]
            if model is None and skus_in_query:
                brand = "GLORIA" if gloria_context else None
                exact = _qdrant_scroll_exact(client, args.collection, skus=skus_in_query, brand=brand, limit=4)
                if exact:
                    retrieved_chunks = exact
            retrieved_chunks = _apply_local_payload_filters(
                retrieved_chunks,
                require_chunk_type=require_chunk_type,
                require_brand=require_brand,
            ) or retrieved_chunks

            retrieved_skus = sorted({str(c.get("sku")) for c in retrieved_chunks if c.get("sku")})
            if root_span is not None:
                _langfuse_add_trace_tags(langfuse, trace_id=getattr(root_span, "trace_id", None), tags=retrieved_skus)

            if retrieval_span is not None:
                try:
                    retrieval_span.update(
                        output={
                            "qdrant_filter": str(qdrant_filter) if qdrant_filter else None,
                            "bm25_top_k": top_k,
                            "vec_top_k": top_k if model is not None else 0,
                            "retrieved_count": len(retrieved_chunks),
                            "retrieved_skus": retrieved_skus,
                            "retrieved_pages": sorted(
                                {c.get("physical_page_number", c.get("page_number")) for c in retrieved_chunks}
                            ),
                        }
                    )
                except Exception:
                    pass

        print("[3/3] Sending context to LLM...")
        generate_answer(args.query, retrieved_chunks, api_key)

    if langfuse:
        try:
            langfuse.flush()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
