from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageArtifact:
    source_file: str
    page_number: int
    text: str
    meta: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _product_to_text(p: dict[str, Any]) -> str:
    parts: list[str] = []
    ref = (p.get("ref") or "").strip()
    nombre = (p.get("nombre") or "").strip()
    if ref or nombre:
        parts.append(f"{ref} · {nombre}".strip(" ·"))
    desc = (p.get("descripcion") or "").strip()
    if desc:
        parts.append(desc)

    # Keep these as plain text lines for keyword search (codes, weights, prices).
    for k in ("tamano", "peso", "unidad_min", "pvpr"):
        v = p.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(f"{k}: {s}")
    return "\n".join(parts).strip()


def ingest_mistral_structured(
    *,
    input_dir: Path,
    out_dir: Path,
    source_name: str,
) -> list[Path]:
    """
    Ingest the Mistral OCR structured output we currently have as:
      data/raw/<catalog_name>/document-annotation.json

    Output:
      data/ingested/pages/page_0001.json (one per page_number from input)
    """
    annot_path = input_dir / "document-annotation.json"
    if not annot_path.exists():
        raise FileNotFoundError(str(annot_path))

    data = json.loads(annot_path.read_text(encoding="utf-8"))
    page_number = int(data.get("page_number") or 1)
    products: list[dict[str, Any]] = list(data.get("products") or [])

    page_lines: list[str] = []
    for p in products:
        t = _product_to_text(p)
        if t:
            page_lines.append(t)
            page_lines.append("")  # blank line between products
    page_text = "\n".join(page_lines).strip()

    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    artifact = PageArtifact(
        source_file=source_name,
        page_number=page_number,
        text=page_text,
        meta={
            "ingested_at": _utc_now_iso(),
            "method": "mistral_structured",
            "input_dir": str(input_dir),
            "product_count": len(products),
        },
    )

    out_path = pages_dir / f"page_{page_number:04d}.json"
    _safe_write_json(out_path, asdict(artifact))

    manifest = {
        "source_file": source_name,
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "created_at": _utc_now_iso(),
        "page_count": 1,
        "page_numbers": [page_number],
    }
    _safe_write_json(out_dir / "manifest_structured.json", manifest)

    return [out_path]


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest Mistral OCR structured outputs into per-page JSON.")
    ap.add_argument("--input-dir", required=True, type=Path, help="Folder containing document-annotation.json")
    ap.add_argument("--out", required=True, type=Path, help="Output directory (creates pages/)")
    ap.add_argument("--source-name", default=None, help="Logical source file name for metadata/citations")
    args = ap.parse_args()

    source_name = args.source_name or args.input_dir.name
    written = ingest_mistral_structured(input_dir=args.input_dir, out_dir=args.out, source_name=source_name)
    print(f"Wrote {len(written)} page JSON files to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

