from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


_SECTION_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
_SKU_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9/*.-]{1,}\b")


def split_product_sections(text: str) -> list[tuple[str | None, str]]:
    lines = text.splitlines()
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = _SECTION_HEADING_RE.match(line.strip())
        if match:
            body = "\n".join(current_lines).strip()
            if body or current_heading is not None:
                sections.append((current_heading, body))
            current_heading = match.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body or current_heading is not None:
        sections.append((current_heading, body))

    return sections


def product_chunks(text: str) -> list[tuple[str, str | None]]:
    sections = split_product_sections(text)
    if len(sections) == 1 and sections[0][0] is None:
        return [(text.strip(), None)] if text.strip() else []

    out: list[tuple[str, str | None]] = []
    for heading, body in sections:
        chunk_text = f"# {heading}\n\n{body}".strip() if heading else body.strip()
        if chunk_text:
            out.append((chunk_text, heading))
    return out


def build_product_dictionary(pages: list[dict[str, Any]]) -> dict[str, str]:
    dictionary: dict[str, str] = {}
    for page in pages:
        text = page.get("text") or ""
        for heading, body in split_product_sections(text):
            if not heading or not body:
                continue
            for line in body.splitlines():
                if not line.strip().startswith("|"):
                    continue
                for sku in _extract_skus_from_markdown_row(line):
                    dictionary.setdefault(sku, heading)
    return dictionary


def enrich_text_with_product_names(text: str, product_dictionary: dict[str, str]) -> str:
    if not text or not product_dictionary:
        return text

    sorted_skus = sorted(product_dictionary, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(sku) for sku in sorted_skus) + r")\b")

    def replacer(match: re.Match[str]) -> str:
        sku = match.group(1)
        product_name = product_dictionary.get(sku)
        if not product_name:
            return sku
        return f"{sku} [Contexto: {product_name}]"

    return pattern.sub(replacer, text)


def enrich_query_with_product_names(query: str, product_dictionary: dict[str, str]) -> str:
    if not query or not product_dictionary:
        return query

    found: list[str] = []
    for sku in _find_query_skus(query):
        product_name = product_dictionary.get(sku)
        if product_name and product_name not in found:
            found.append(product_name)

    if not found:
        return query

    context_suffix = " ".join(found)
    return f"{query} {context_suffix}".strip()


def _extract_skus_from_markdown_row(row: str) -> list[str]:
    cells = [cell.strip() for cell in row.split("|")[1:-1]]
    if not cells:
        return []
    first_cell = cells[0]
    lowered = first_cell.lower()
    if lowered in {"ref.", "ref", "referencia"} or "---" in first_cell:
        return []
    return [candidate for candidate in _SKU_PATTERN.findall(first_cell) if _looks_like_sku(candidate)]


def _find_query_skus(query: str) -> Iterable[str]:
    for candidate in _SKU_PATTERN.findall(query.upper()):
        if _looks_like_sku(candidate):
            yield candidate


def _looks_like_sku(value: str) -> bool:
    return any(ch.isalpha() for ch in value) and any(ch.isdigit() for ch in value)
