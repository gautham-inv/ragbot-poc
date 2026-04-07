from __future__ import annotations

import argparse
import json
import os
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retrieval.es_tokenizer import tokenize_es


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: dict[str, Any]


def _load_pages(pages_dir: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for p in sorted(pages_dir.glob("page_*.json")):
        pages.append(json.loads(p.read_text(encoding="utf-8")))
    return pages


def _simple_token_chunk(text: str, *, target_tokens: int = 650, overlap: int = 100) -> list[str]:
    """Fallback: sliding-window token chunking for pages with no heading structure."""
    toks = text.split()
    if not toks:
        return []
    out: list[str] = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + target_tokens)
        out.append(" ".join(toks[i:j]))
        if j == len(toks):
            break
        i = max(0, j - overlap)
    return out


# ---------------------------------------------------------------------------
# Product-aware chunking
# ---------------------------------------------------------------------------
# The catalog markdown follows a consistent pattern:
#   [optional preamble / brand intro]
#   # Product Name
#   multilingual descriptions…
#   | Referencia | … | PVPR/MSRP |   <-- table (after inlining)
#   SKU code
#   barcode
#   # Next Product Name
#   …
#
# We split on top-level headings (lines starting with "# ") so each product
# heading + its body (descriptions, tables, SKU codes) forms one indivisible
# chunk. This guarantees the table data (prices, sizes, references) stays
# bound to the product it belongs to.
# ---------------------------------------------------------------------------

_HEADING_RE = __import__("re").compile(r"^(#{1,2})\s+", __import__("re").MULTILINE)


def _split_into_product_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Split page markdown into (heading, body) sections.

    Returns a list of ``(heading_text | None, section_body)`` tuples.
    The first element may have ``heading_text=None`` if the page starts with
    preamble text before the first ``#`` heading.
    """
    lines = text.split("\n")
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Detect top-level or second-level headings
        if stripped.startswith("# ") or stripped.startswith("## "):
            # Flush previous section
            body = "\n".join(current_lines).strip()
            if body or current_heading is not None:
                sections.append((current_heading, body))
            current_heading = stripped.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    body = "\n".join(current_lines).strip()
    if body or current_heading is not None:
        sections.append((current_heading, body))

    return sections


def _product_chunk(page_text: str) -> list[tuple[str, str | None]]:
    """
    Chunk a page into product-level chunks.

    Returns list of ``(chunk_text, product_heading | None)``.
    Falls back to token chunking if no headings are found.
    """
    sections = _split_into_product_sections(page_text)

    # No heading structure → fall back to token chunking
    if len(sections) <= 1 and sections[0][0] is None:
        return [(ch, None) for ch in _simple_token_chunk(page_text)]

    results: list[tuple[str, str | None]] = []

    # Grab any preamble (brand intro, page header) before the first product
    if sections and sections[0][0] is None:
        preamble = sections[0][1].strip()
        if preamble:
            results.append((preamble, None))
        sections = sections[1:]

    for heading, body in sections:
        # Combine heading + body into one chunk
        chunk_text = f"# {heading}\n\n{body}".strip() if heading else body.strip()
        if chunk_text:
            results.append((chunk_text, heading))

    return results if results else [(ch, None) for ch in _simple_token_chunk(page_text)]


def _build_sku_dictionary(pages: list[dict[str, Any]]) -> dict[str, str]:
    import re
    sku_dict = {}
    for page in pages:
        text = page.get("text") or ""
        sections = _split_into_product_sections(text)
        for heading, body in sections:
            if not heading:
                continue
            for line in body.split("\n"):
                if line.startswith("|"):
                    cells = [c.strip() for c in line.split("|")[1:-1]]
                    if cells and len(cells) > 0 and "Referencia" not in cells[0] and "---" not in cells[0]:
                        sku = cells[0]
                        # Capture viable SKUs
                        if len(sku) >= 3 and any(c.isalpha() for c in sku) and any(c.isdigit() for c in sku):
                            if sku not in sku_dict:
                                sku_dict[sku] = heading
    return sku_dict

def _inject_sku_names(pages: list[dict[str, Any]], sku_dict: dict[str, str]):
    import re
    if not sku_dict:
        return
    # Order by length descending to match longest SKUs first
    sorted_skus = sorted(sku_dict.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted_skus)) + r")\b")
    
    for page in pages:
        text = page.get("text") or ""
        if not text:
            continue
        def replacer(match):
            m = match.group(1)
            return f"{m} [Contexto: {sku_dict[m]}]"
        # Run replacement safely
        page["text"] = pattern.sub(replacer, text)

def build_chunks(pages: list[dict[str, Any]]) -> list[Chunk]:
    sku_dictionary = _build_sku_dictionary(pages)
    _inject_sku_names(pages, sku_dictionary)

    chunks: list[Chunk] = []
    for page in pages:
        page_num = int(page.get("page_number") or 0)
        source_file = page.get("source_file") or "unknown"
        text = page.get("text") or ""

        product_chunks = _product_chunk(text)
        for idx, (ch_text, product_heading) in enumerate(product_chunks):
            chunk_id = f"{source_file}:p{page_num}:c{idx}"
            meta: dict[str, Any] = {
                "source_file": source_file,
                "page_number": page_num,
                "chunk_index": idx,
                "chunk_id": chunk_id,
            }
            if product_heading:
                meta["product_name"] = product_heading
            chunks.append(
                Chunk(chunk_id=chunk_id, text=ch_text, meta=meta)
            )
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Qdrant + BM25 indexes from ingested pages.")
    ap.add_argument("--pages-dir", default="data/ingested/pages", type=Path)
    ap.add_argument("--collection", default="catalog_es")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", None))
    ap.add_argument("--bm25-out", default="data/index/bm25.pkl", type=Path)
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    pages = _load_pages(args.pages_dir)
    chunks = build_chunks(pages)
    if not chunks:
        raise SystemExit(f"No chunks built from: {args.pages_dir}")

    args.bm25_out.parent.mkdir(parents=True, exist_ok=True)

    # Embeddings (E5 uses prefixes).
    model = SentenceTransformer(args.model)

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    dim = model.get_sentence_embedding_dimension()
    try:
        client.get_collection(args.collection)
    except Exception:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    for i in tqdm(range(0, len(chunks), args.batch_size), desc="Embedding+upsert"):
        batch = chunks[i : i + args.batch_size]
        texts = [f"passage: {c.text}" for c in batch]
        embs = model.encode(texts, normalize_embeddings=True).tolist()

        client.upsert(
            collection_name=args.collection,
            points=[
                {
                    # Qdrant point IDs must be int or UUID. We derive a stable UUID
                    # from our human-readable chunk_id, and store chunk_id in payload.
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, batch[j].chunk_id)),
                    "vector": embs[j],
                    "payload": {"text": batch[j].text, **batch[j].meta},
                }
                for j in range(len(batch))
            ],
        )

    # BM25
    tokenized = [tokenize_es(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with args.bm25_out.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": [{"id": c.chunk_id, "text": c.text, "meta": c.meta} for c in chunks],
            },
            f,
        )

    print(f"Upserted {len(chunks)} chunks to Qdrant collection {args.collection}")
    print(f"Wrote BM25 index to {args.bm25_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

