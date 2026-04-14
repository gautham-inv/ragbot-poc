from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Ensure project root is on sys.path when executed as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.product_dictionary import build_product_dictionary
from retrieval.tokenize_es import tokenize_es


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: dict[str, Any]


_SOURCE_RANGE_RE = re.compile(r"^(?P<start>\d+)-(?P<end>\d+)(?:\.pdf)?$", re.IGNORECASE)
_PAGE_DIR_RE = re.compile(r"(?:^|[\\/])page-(\d+)(?:$|[\\/])", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_ALL_CAPS_LINE_RE = re.compile(r"^[A-Z0-9&][A-Z0-9&'’ .+/-]{1,28}$")
_CATEGORY_RE = re.compile(r"^\s*([A-ZÁÉÍÓÚÜÑ]{3,})(?:\s*[·•\\-|/]|\\s*$)")
_SKU_RE = re.compile(r"^[A-Z]{2,}[A-Z0-9/*.-]{1,}$")
_BARCODE_RE = re.compile(r"^[0-9][0-9 *-]{6,}$")
_BARCODE_IN_TEXT_RE = re.compile(r"\b(\d{8,18})\b")

_GLORIA_SUBCATALOG_START = 250
_GLORIA_SUBCATALOG_END = 396


def _load_pages(pages_dir: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for p in sorted(pages_dir.glob("page_*.json")):
        pages.append(json.loads(p.read_text(encoding="utf-8")))
    return pages


def _simple_token_chunk(text: str, *, target_tokens: int = 650, overlap: int = 100) -> list[str]:
    # Tokenization approximation (word tokens) that's stable offline.
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


def _utc_compact_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    out = text
    out = re.sub(r"(?i)gl\*ria", "Gloria", out)
    out = out.replace("\u00a0", " ")
    out = re.sub(r"[ \t]+", " ", out)
    return out


def _normalize_sku(value: str) -> str:
    value = (value or "").strip().upper()
    value = value.replace(" ", "")
    return value


def _normalize_barcode(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    cleaned = re.sub(r"[ *-]+", "", raw)
    if not cleaned.isdigit():
        return None
    if len(cleaned) < 8 or len(cleaned) > 18:
        return None
    return cleaned


def _find_barcode_in_text(value: str) -> str | None:
    if not value:
        return None
    for match in _BARCODE_IN_TEXT_RE.finditer(value):
        bc = _normalize_barcode(match.group(1))
        if bc:
            return bc
    return None


def _infer_sub_page_number(page: dict[str, Any]) -> int | None:
    meta = page.get("meta") or {}
    page_dir = str(meta.get("page_dir") or "")
    match = _PAGE_DIR_RE.search(page_dir)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _infer_physical_page_number(page: dict[str, Any]) -> tuple[int, int | None, int | None]:
    """
    Returns (physical_page_number, sub_page_number, ocr_page_number_if_overridden).

    Some OCR batches reset numbering (page-1, page-2, ...) under a range-named folder like
    101-200.pdf. If we detect that pattern, we recompute physical page from range start + subpage.
    """
    ocr_page_number = int(page.get("page_number") or 0)
    sub_page_number = _infer_sub_page_number(page)

    source_file = str(page.get("source_file") or "")
    match = _SOURCE_RANGE_RE.match(source_file)
    range_start = int(match.group("start")) if match else None

    if range_start and sub_page_number and ocr_page_number == sub_page_number and range_start != 1:
        return (range_start + sub_page_number - 1, sub_page_number, ocr_page_number)

    return (ocr_page_number, sub_page_number, None)


def _parse_md_table_row(line: str) -> list[str]:
    return [p.strip() for p in line.strip().split("|")[1:-1]]


def _is_md_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    cells = _parse_md_table_row(stripped)
    if not cells:
        return False
    for cell in cells:
        candidate = cell.replace(":", "").replace("-", "").strip()
        if candidate:
            return False
    return True


def _looks_like_sku(value: str) -> bool:
    candidate = _normalize_sku(value)
    if not candidate or not _SKU_RE.match(candidate):
        return False
    return any(ch.isalpha() for ch in candidate) and any(ch.isdigit() for ch in candidate)


def _infer_category(text: str) -> str | None:
    for line in text.splitlines()[:12]:
        stripped = line.strip()
        if not stripped or stripped.startswith("![") or stripped.startswith("|"):
            continue
        match = _CATEGORY_RE.match(stripped.upper())
        if match:
            return match.group(1)
    return None


def _infer_brand_from_top(text: str) -> str | None:
    for line in text.splitlines()[:30]:
        match = _MD_HEADING_RE.match(line)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) <= 2:
                continue
            return candidate.split(" · ")[0].strip()

    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if not stripped or stripped.startswith("![") or stripped.startswith("|"):
            continue
        if _ALL_CAPS_LINE_RE.match(stripped) and not stripped.isdigit():
            if any(ch.isdigit() for ch in stripped):
                continue
            return stripped.replace("  ", " ").strip()
    return None


def _infer_brand_from_product_name(product_name: str | None) -> str | None:
    if not product_name:
        return None
    first = product_name.strip().split()[0]
    if first.isupper() and len(first) >= 3:
        return first
    return None


def _extract_sku_barcode_pairs(lines: list[str]) -> dict[str, str]:
    """
    Some pages list barcode as a raw digit line immediately after repeating the SKU.
    Example:
      COA90413A
      886284904131
    """
    out: dict[str, str] = {}
    for i in range(len(lines)):
        sku_line = lines[i].strip()
        if not _looks_like_sku(sku_line):
            continue
        # Look ahead a few lines, skipping empties, for a raw barcode line.
        for j in range(i + 1, min(len(lines), i + 5)):
            bc_line = lines[j].strip()
            if not bc_line:
                continue
            if not _BARCODE_RE.match(bc_line):
                break
            bc_norm = _normalize_barcode(bc_line)
            if bc_norm:
                out.setdefault(_normalize_sku(sku_line), bc_norm)
            break
    return out


def _parse_price_eur(value: str) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    raw = raw.replace("€", "").replace("â‚¬", "").strip()
    raw = raw.replace(".", "").replace(",", ".")
    try:
        return float(raw)
    except Exception:
        return None


def _extract_price_eur_from_text(value: str) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d{1,6}(?:[.,]\d{1,2})?)\s*€", value)
    if not match:
        return None
    return _parse_price_eur(match.group(1))


def _parse_int(value: str) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    match = re.search(r"\b(\d{1,6})\b", raw)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _parse_weight_g(value: str) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    raw = raw.replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|gr)\b", raw, flags=re.IGNORECASE)
    if not match:
        return None
    num = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "kg":
        num *= 1000.0
    return int(round(num))


def _parse_dimensions_cm(value: str) -> list[float] | None:
    """
    Parse dimensions like:
      "8,89 x 36,83 x 12,7 cm"
      "32x26x42cm"
      "15 - 21 x 2 x 1,2 cm"
    Returns a list of cm values (floats). If a range is present, uses the upper bound.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    raw = raw.replace("×", "x")
    raw = raw.replace(",", ".")
    if "cm" not in raw.lower():
        return None

    def parse_part(part: str) -> float | None:
        part = part.strip()
        # range like "15 - 21"
        rng = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", part)
        if rng:
            return float(rng.group(2))
        num = re.match(r"^(\d+(?:\.\d+)?)$", part)
        if num:
            return float(num.group(1))
        return None

    # Extract the first "a x b x c cm" style segment
    seg_match = re.search(r"([0-9.\s-]+(?:\s*x\s*[0-9.\s-]+){1,4})\s*cm\b", raw, flags=re.IGNORECASE)
    if not seg_match:
        return None
    seg = seg_match.group(1)
    parts = [p.strip() for p in re.split(r"\s*x\s*", seg) if p.strip()]
    dims: list[float] = []
    for p in parts:
        val = parse_part(p)
        if val is None:
            return None
        dims.append(val)
    return dims if dims else None


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("![") or stripped.startswith("[tbl-"):
        return True
    return False


def _should_update_product_name(line: str) -> bool:
    stripped = line.strip()
    if _is_noise_line(stripped):
        return False
    if stripped.startswith("|") or stripped.startswith("[tbl-"):
        return False
    if stripped.isdigit():
        return False
    if _BARCODE_RE.match(stripped):
        return False
    if _looks_like_sku(stripped):
        return False
    return True


def _build_flat_sentence(
    *,
    brand: str | None,
    category: str | None,
    product_name: str | None,
    sku: str,
    fields: dict[str, str],
    barcode_norm: str | None,
) -> str:
    parts: list[str] = []
    if category:
        parts.append(f"[Category: {category}]")
    if brand:
        parts.append(f"[Brand: {brand}]")
    if product_name:
        parts.append(f"[Product: {product_name}]")
    parts.append(f"SKU: {sku}")

    preferred = [
        "Item",
        "Modelo",
        "Model",
        "Talla",
        "Size",
        "Tamaño",
        "Cantidad",
        "Peso",
        "Weight",
        "PVPR/MSRP",
        "Precio",
        "Price",
    ]
    emitted: set[str] = set()
    for key in preferred:
        if key in fields and fields[key]:
            parts.append(f"{key}: {fields[key]}")
            emitted.add(key)

    for k in sorted(fields):
        if k in emitted:
            continue
        v = fields[k]
        if v:
            parts.append(f"{k}: {v}")

    if barcode_norm:
        parts.append(f"Barcode: {barcode_norm}")

    return ", ".join(parts).strip() + "."


def build_chunks(pages: list[dict[str, Any]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        source_file = page.get("source_file") or "unknown"
        physical_page_number, sub_page_number, ocr_page_number = _infer_physical_page_number(page)

        raw_text = page.get("text") or ""
        normalized_text = _normalize_text(raw_text)

        category = _infer_category(normalized_text)
        inferred_brand = _infer_brand_from_top(normalized_text)

        brand_override = None
        brand_override_reason = None
        if _GLORIA_SUBCATALOG_START <= physical_page_number <= _GLORIA_SUBCATALOG_END:
            brand_override = "GLORIA"
            brand_override_reason = "gloria_subcatalog_rule"

        lines = normalized_text.splitlines()
        sku_to_barcode = _extract_sku_barcode_pairs(lines)

        current_product_name: str | None = None
        current_brand = inferred_brand

        narrative_lines: list[str] = []
        chunk_index = 0
        sku_chunks_built = 0

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            heading = _MD_HEADING_RE.match(stripped)
            if heading:
                current_product_name = heading.group(1).strip()
                maybe_brand = _infer_brand_from_product_name(current_product_name)
                if maybe_brand:
                    current_brand = maybe_brand
                narrative_lines.append(stripped)
                i += 1
                continue

            if _should_update_product_name(stripped):
                current_product_name = stripped
                if _ALL_CAPS_LINE_RE.match(stripped) and not any(ch.isdigit() for ch in stripped) and len(stripped) <= 20:
                    current_brand = stripped
                narrative_lines.append(stripped)
                i += 1
                continue

            if stripped.startswith("|"):
                # OCR can break a table header across multiple lines before the separator row.
                sep_idx = None
                for lookahead in range(i + 1, min(len(lines), i + 8)):
                    if _is_md_table_separator(lines[lookahead]):
                        sep_idx = lookahead
                        break
                if sep_idx is None:
                    if not _is_noise_line(stripped):
                        narrative_lines.append(stripped)
                    i += 1
                    continue

                header_buf = stripped
                raw_table_lines = [lines[i]]
                for k in range(i + 1, sep_idx):
                    raw_table_lines.append(lines[k])
                    header_buf = f"{header_buf} {lines[k].strip()}"
                raw_table_lines.append(lines[sep_idx])

                if not header_buf.strip().startswith("|"):
                    header_buf = "| " + header_buf.strip()
                if not header_buf.strip().endswith("|"):
                    header_buf = header_buf.strip() + " |"

                header_cells = _parse_md_table_row(header_buf)
                data_rows: list[list[str]] = []

                j = sep_idx + 1
                while j < len(lines):
                    row_line = lines[j].rstrip("\n")
                    row_line_stripped = row_line.strip()
                    if not row_line_stripped.startswith("|"):
                        break
                    raw_table_lines.append(lines[j])

                    expected_pipes = len(header_cells) + 1 if header_cells else None
                    row_buf = row_line_stripped
                    # Some OCR outputs break table rows across multiple lines (barcode on the next line, etc).
                    while (
                        expected_pipes
                        and row_buf.count("|") < expected_pipes
                        and (j + 1) < len(lines)
                        and not lines[j + 1].strip().startswith("|")
                        and lines[j + 1].strip()
                    ):
                        j += 1
                        raw_table_lines.append(lines[j])
                        row_buf = f"{row_buf} {lines[j].strip()}"

                    row_cells = _parse_md_table_row(row_buf)
                    if row_cells:
                        data_rows.append(row_cells)
                    j += 1

                header_lc = [h.strip().lower() for h in header_cells]
                sku_col = None
                for idx_h, h in enumerate(header_lc):
                    if h in {"referencia", "sku", "ref", "ref.", "referencia/s", "referencia / sku"}:
                        sku_col = idx_h
                        break

                if sku_col is None:
                    # Try extracting SKU anchors from any cell (handles broken tables like COLOR -> SKU+EAN).
                    for row_idx, row in enumerate(data_rows):
                        for col_idx, cell in enumerate(row):
                            for token in re.split(r"\s+", cell.strip()):
                                if _looks_like_sku(token):
                                    sku = _normalize_sku(token)
                                    fields: dict[str, str] = {}
                                    for k_idx, header in enumerate(header_cells):
                                        if k_idx >= len(row):
                                            continue
                                        val = row[k_idx].strip()
                                        if val:
                                            fields[header.strip()] = val

                                    barcode_norm = sku_to_barcode.get(sku)
                                    if not barcode_norm:
                                        barcode_norm = _find_barcode_in_text(cell)

                                    brand = brand_override or current_brand or _infer_brand_from_product_name(current_product_name)
                                    flat = _build_flat_sentence(
                                        brand=brand,
                                        category=category,
                                        product_name=current_product_name,
                                        sku=sku,
                                        fields=fields,
                                        barcode_norm=barcode_norm,
                                    )

                                    chunk_id = f"{source_file}:p{physical_page_number}:sku:{sku}"
                                    if row_idx or col_idx:
                                        chunk_id = f"{chunk_id}:r{row_idx}c{col_idx}"

                                    meta: dict[str, Any] = {
                                        "source_file": source_file,
                                        "page_number": physical_page_number,
                                        "physical_page_number": physical_page_number,
                                        "sub_page_number": sub_page_number,
                                        "chunk_index": chunk_index,
                                        "chunk_id": chunk_id,
                                        "chunk_type": "product_sku_row",
                                        "sku": sku,
                                    }
                                    if category:
                                        meta["category"] = category
                                    if brand:
                                        meta["brand"] = brand
                                    if current_product_name:
                                        meta["product_name"] = current_product_name
                                    if barcode_norm:
                                        meta["barcode_norm"] = barcode_norm
                                    if brand_override_reason:
                                        meta["brand_override_reason"] = brand_override_reason
                                    if ocr_page_number is not None:
                                        meta["ocr_page_number"] = ocr_page_number
                                    price = _extract_price_eur_from_text(flat)
                                    if price is not None:
                                        meta["price_eur"] = price
                                    # Best-effort numeric filters from the flattened text.
                                    dims = _parse_dimensions_cm(flat)
                                    if dims:
                                        meta["dimensions_cm"] = dims
                                        meta["size_cm"] = max(dims)
                                    w = _parse_weight_g(flat)
                                    if w is not None:
                                        meta["weight_g"] = w

                                    chunks.append(Chunk(chunk_id=chunk_id, text=flat, meta=meta))
                                    sku_chunks_built += 1
                                    chunk_index += 1
                                    break
                            else:
                                continue
                            break

                    # Always keep the table itself as a low-priority context chunk (no metadata loss).
                    ctx_text = "\n".join(raw_table_lines).strip()
                    if ctx_text:
                        chunk_id = f"{source_file}:p{physical_page_number}:ctx:{chunk_index}"
                        meta = {
                            "source_file": source_file,
                            "page_number": physical_page_number,
                            "physical_page_number": physical_page_number,
                            "sub_page_number": sub_page_number,
                            "chunk_index": chunk_index,
                            "chunk_id": chunk_id,
                            "chunk_type": "table_context_no_sku",
                        }
                        if category:
                            meta["category"] = category
                        if current_product_name:
                            meta["product_name"] = current_product_name
                        if brand_override or current_brand:
                            meta["brand"] = brand_override or current_brand
                        if brand_override_reason:
                            meta["brand_override_reason"] = brand_override_reason
                        if ocr_page_number is not None:
                            meta["ocr_page_number"] = ocr_page_number
                        chunks.append(Chunk(chunk_id=chunk_id, text=ctx_text, meta=meta))
                        chunk_index += 1
                    i = j
                    continue

                for row_idx, row in enumerate(data_rows):
                    if sku_col >= len(row):
                        continue
                    sku = _normalize_sku(row[sku_col])
                    if not _looks_like_sku(sku):
                        continue

                    fields: dict[str, str] = {}
                    for col_idx, header in enumerate(header_cells):
                        if col_idx >= len(row) or col_idx == sku_col:
                            continue
                        val = row[col_idx].strip()
                        if val:
                            fields[header.strip()] = val

                    barcode_norm = sku_to_barcode.get(sku)
                    if not barcode_norm:
                        for v in fields.values():
                            bc = _normalize_barcode(v)
                            if bc:
                                barcode_norm = bc
                                break
                    if not barcode_norm:
                        for v in fields.values():
                            bc = _find_barcode_in_text(v)
                            if bc:
                                barcode_norm = bc
                                break

                    brand = brand_override or current_brand or _infer_brand_from_product_name(current_product_name)
                    flat = _build_flat_sentence(
                        brand=brand,
                        category=category,
                        product_name=current_product_name,
                        sku=sku,
                        fields=fields,
                        barcode_norm=barcode_norm,
                    )

                    chunk_id = f"{source_file}:p{physical_page_number}:sku:{sku}"
                    if row_idx:
                        chunk_id = f"{chunk_id}:r{row_idx}"

                    meta: dict[str, Any] = {
                        "source_file": source_file,
                        "page_number": physical_page_number,
                        "physical_page_number": physical_page_number,
                        "sub_page_number": sub_page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                        "chunk_type": "product_sku_row",
                        "sku": sku,
                    }
                    if category:
                        meta["category"] = category
                    if brand:
                        meta["brand"] = brand
                    if current_product_name:
                        meta["product_name"] = current_product_name
                    if barcode_norm:
                        meta["barcode_norm"] = barcode_norm
                    if brand_override_reason:
                        meta["brand_override_reason"] = brand_override_reason
                    if ocr_page_number is not None:
                        meta["ocr_page_number"] = ocr_page_number

                    for k in ("PVPR/MSRP", "PVPR", "PVP", "Precio", "Price"):
                        if k in fields:
                            price = _parse_price_eur(fields[k])
                            if price is not None:
                                meta["price_eur"] = price
                            break
                    if "price_eur" not in meta:
                        for v in fields.values():
                            price = _extract_price_eur_from_text(v)
                            if price is not None:
                                meta["price_eur"] = price
                                break
                    if "price_eur" not in meta:
                        price = _extract_price_eur_from_text(flat)
                        if price is not None:
                            meta["price_eur"] = price

                    # Numeric filters
                    for k in ("Ud. mín. de compra", "Ud. min. de compra", "Qté min.", "Qté min", "Min. order", "Min order"):
                        if k in fields:
                            v = _parse_int(fields[k])
                            if v is not None:
                                meta["min_order"] = v
                                break
                    for k in ("Peso", "Weight", "Poids"):
                        if k in fields:
                            v = _parse_weight_g(fields[k])
                            if v is not None:
                                meta["weight_g"] = v
                                break
                    for k in ("Tamaño", "Taille", "Size", "Tamaño Taille"):
                        if k in fields:
                            dims = _parse_dimensions_cm(fields[k])
                            if dims:
                                meta["dimensions_cm"] = dims
                                meta["size_cm"] = max(dims)
                                break

                    chunks.append(Chunk(chunk_id=chunk_id, text=flat, meta=meta))
                    sku_chunks_built += 1
                    chunk_index += 1

                i = j
                continue

            if not _is_noise_line(stripped):
                narrative_lines.append(stripped)
            i += 1

        narrative_text = "\n".join([ln for ln in narrative_lines if ln.strip()]).strip()
        if narrative_text:
            for idx, ch in enumerate(_simple_token_chunk(narrative_text, target_tokens=450, overlap=60)):
                chunk_id = f"{source_file}:p{physical_page_number}:n{idx}"
                meta: dict[str, Any] = {
                    "source_file": source_file,
                    "page_number": physical_page_number,
                    "physical_page_number": physical_page_number,
                    "sub_page_number": sub_page_number,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "chunk_type": "page_narrative",
                }
                if category:
                    meta["category"] = category
                if brand_override or inferred_brand:
                    meta["brand"] = brand_override or inferred_brand
                if brand_override_reason:
                    meta["brand_override_reason"] = brand_override_reason
                if ocr_page_number is not None:
                    meta["ocr_page_number"] = ocr_page_number
                chunks.append(Chunk(chunk_id=chunk_id, text=ch, meta=meta))
                chunk_index += 1

        if sku_chunks_built == 0 and not narrative_text and normalized_text.strip():
            for idx, ch in enumerate(_simple_token_chunk(normalized_text)):
                chunk_id = f"{source_file}:p{physical_page_number}:c{idx}"
                meta: dict[str, Any] = {
                    "source_file": source_file,
                    "page_number": physical_page_number,
                    "physical_page_number": physical_page_number,
                    "sub_page_number": sub_page_number,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "chunk_type": "page_fallback",
                }
                if ocr_page_number is not None:
                    meta["ocr_page_number"] = ocr_page_number
                chunks.append(Chunk(chunk_id=chunk_id, text=ch, meta=meta))
                chunk_index += 1

    return chunks


def _parse_page_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if ":" in raw:
        left, right = raw.split(":", 1)
    elif "-" in raw:
        left, right = raw.split("-", 1)
    else:
        raise ValueError("page range must look like START:END or START-END")
    start = int(left.strip())
    end = int(right.strip())
    if start > end:
        start, end = end, start
    return (start, end)


def _write_preview_jsonl(
    *,
    chunks: list[Chunk],
    out_path: Path,
    limit: int,
    skus: set[str] | None = None,
    pages: set[int] | None = None,
    page_range: tuple[int, int] | None = None,
    chunk_types: set[str] | None = None,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    normalized_skus = {_normalize_sku(s) for s in skus} if skus else None

    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            meta = c.meta or {}
            if chunk_types:
                if meta.get("chunk_type") not in chunk_types:
                    continue
            if normalized_skus:
                sku = meta.get("sku")
                if not sku or _normalize_sku(str(sku)) not in normalized_skus:
                    continue
            if pages or page_range:
                pn = meta.get("physical_page_number", meta.get("page_number"))
                try:
                    pn_int = int(pn)
                except Exception:
                    pn_int = None
                if pn_int is None:
                    continue
                if pages and pn_int not in pages:
                    continue
                if page_range and not (page_range[0] <= pn_int <= page_range[1]):
                    continue

            f.write(
                json.dumps(
                    {"chunk_id": c.chunk_id, "text": c.text, "metadata": meta},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if written >= limit:
                break

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Qdrant + BM25 indexes from ingested pages.")
    ap.add_argument("--pages-dir", default="data/ingested/pages", type=Path)
    ap.add_argument("--collection", default="catalog_es")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-api-key", default=None)
    ap.add_argument("--bm25-out", default="data/index/bm25.pkl", type=Path)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--recreate", action="store_true", help="Delete + recreate the Qdrant collection first.")
    ap.add_argument("--preview-out", type=Path, default=None, help="Write a JSONL preview of chunks, then exit.")
    ap.add_argument("--preview-limit", type=int, default=200, help="Max preview chunks to write.")
    ap.add_argument("--preview-sku", action="append", default=None, help="Filter preview to a specific SKU (repeatable).")
    ap.add_argument("--preview-page", action="append", type=int, default=None, help="Filter preview to a page (repeatable).")
    ap.add_argument("--preview-page-range", default=None, help="Filter preview to a page range START:END.")
    ap.add_argument(
        "--preview-chunk-type",
        action="append",
        default=None,
        help="Filter preview to chunk_type (repeatable), e.g. product_sku_row.",
    )
    args = ap.parse_args()

    pages = _load_pages(args.pages_dir)
    chunks = build_chunks(pages)
    if not chunks:
        raise SystemExit(f"No chunks built from: {args.pages_dir}")

    if args.preview_out:
        page_range = _parse_page_range(args.preview_page_range)
        written = _write_preview_jsonl(
            chunks=chunks,
            out_path=args.preview_out,
            limit=max(1, int(args.preview_limit)),
            skus=set(args.preview_sku) if args.preview_sku else None,
            pages=set(args.preview_page) if args.preview_page else None,
            page_range=page_range,
            chunk_types=set(args.preview_chunk_type) if args.preview_chunk_type else None,
        )
        print(f"Wrote {written} preview chunks to {args.preview_out}")
        return 0

    args.bm25_out.parent.mkdir(parents=True, exist_ok=True)
    if args.bm25_out.exists():
        backup = args.bm25_out.with_suffix(args.bm25_out.suffix + f".bak.{_utc_compact_timestamp()}")
        args.bm25_out.replace(backup)
        print(f"Backed up existing BM25 bundle to: {backup}")

    model = SentenceTransformer(args.model, device="cpu")

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    dim = model.get_sentence_embedding_dimension()
    if args.recreate:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    try:
        client.get_collection(args.collection)
    except Exception:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # Create payload indexes for filtered fields
        for field_name in ["chunk_type", "brand", "sku"]:
            client.create_payload_index(
                collection_name=args.collection,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    for i in tqdm(range(0, len(chunks), args.batch_size), desc="Embedding+upsert"):
        batch = chunks[i : i + args.batch_size]
        texts = [f"passage: {c.text}" for c in batch]
        embs = model.encode(texts, normalize_embeddings=True).tolist()

        client.upsert(
            collection_name=args.collection,
            points=[
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, batch[j].chunk_id)),
                    "vector": embs[j],
                    "payload": {"text": batch[j].text, **batch[j].meta},
                }
                for j in range(len(batch))
            ],
        )

    tokenized = [tokenize_es(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    product_dictionary = build_product_dictionary(pages)
    with args.bm25_out.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": [{"id": c.chunk_id, "text": c.text, "meta": c.meta} for c in chunks],
                "product_dictionary": product_dictionary,
            },
            f,
        )

    print(f"Upserted {len(chunks)} chunks to Qdrant collection {args.collection}")
    print(f"Wrote BM25 index to {args.bm25_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
