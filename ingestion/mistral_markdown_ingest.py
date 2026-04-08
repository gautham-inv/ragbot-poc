from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TABLE_LINK_RE = re.compile(r"\[([^\]]+\.md)\]\([^)]+\)")


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


def _inline_page_tables(markdown: str, page_dir: Path) -> tuple[str, int]:
    count = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal count
        table_name = match.group(1)
        table_path = page_dir / table_name
        if not table_path.exists():
            return match.group(0)
        count += 1
        return table_path.read_text(encoding="utf-8").strip()

    return _TABLE_LINK_RE.sub(replacer, markdown), count


def _infer_page_number_start(input_dir: Path) -> int:
    candidates = [input_dir.name, input_dir.parent.name]
    for candidate in candidates:
        match = re.match(r"^(\d+)-(\d+)(?:\.pdf)?$", candidate)
        if match:
            return int(match.group(1))
    return 1


def ingest_mistral_markdown_folder(
    *,
    input_dir: Path,
    out_dir: Path,
    source_name: str,
    page_number_start: int | None = None,
) -> list[Path]:
    pages_root = input_dir / "pages"
    if not pages_root.exists():
        raise FileNotFoundError(str(pages_root))

    out_pages = out_dir / "pages"
    out_pages.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    page_numbers: list[int] = []
    resolved_start = page_number_start if page_number_start is not None else _infer_page_number_start(input_dir)

    for page_dir in sorted(pages_root.glob("page-*")):
        if not page_dir.is_dir():
            continue
        try:
            page_index = int(page_dir.name.split("-", 1)[1])
        except Exception:
            continue

        md_path = page_dir / "markdown.md"
        if not md_path.exists():
            continue

        text, tables_inlined = _inline_page_tables(md_path.read_text(encoding="utf-8"), page_dir)
        page_number = resolved_start + page_index - 1
        artifact = PageArtifact(
            source_file=source_name,
            page_number=page_number,
            text=text,
            meta={
                "ingested_at": _utc_now_iso(),
                "method": "mistral_markdown_pages",
                "input_dir": str(input_dir),
                "page_dir": str(page_dir),
                "tables_inlined": tables_inlined,
            },
        )
        out_path = out_pages / f"page_{page_number:04d}.json"
        _safe_write_json(out_path, asdict(artifact))
        written.append(out_path)
        page_numbers.append(page_number)

    manifest = {
        "source_file": source_name,
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "created_at": _utc_now_iso(),
        "method": "mistral_markdown_pages",
        "page_count": len(written),
        "tables_inlined": sum(
            json.loads(path.read_text(encoding="utf-8")).get("meta", {}).get("tables_inlined", 0)
            for path in written
        ),
        "page_numbers": sorted(page_numbers),
    }
    _safe_write_json(out_dir / "manifest_markdown.json", manifest)

    return written


def scan_and_ingest_raw_root(
    *,
    raw_root: Path,
    out_dir: Path,
    page_number_start: int | None = None,
) -> list[Path]:
    written: list[Path] = []
    for markdown_dir in sorted(path.parent for path in raw_root.glob("**/pages")):
        if not (markdown_dir / "pages").exists():
            continue
        source_name = markdown_dir.name
        written.extend(
            ingest_mistral_markdown_folder(
                input_dir=markdown_dir,
                out_dir=out_dir,
                source_name=source_name,
                page_number_start=page_number_start,
            )
        )
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest Mistral per-page markdown into per-page JSON.")
    ap.add_argument("--input-dir", type=Path, default=None, help="Folder containing pages/page-*/markdown.md")
    ap.add_argument("--scan-root", type=Path, default=None, help="Scan a raw root for OCR output folders")
    ap.add_argument("--out", required=True, type=Path, help="Output directory (creates pages/)")
    ap.add_argument("--source-name", default=None, help="Logical source file name for metadata/citations")
    ap.add_argument("--page-number-start", type=int, default=None, help="Override the first output page number")
    args = ap.parse_args()

    if not args.input_dir and not args.scan_root:
        raise SystemExit("Provide either --input-dir or --scan-root.")

    if args.input_dir:
        source_name = args.source_name or args.input_dir.name
        written = ingest_mistral_markdown_folder(
            input_dir=args.input_dir,
            out_dir=args.out,
            source_name=source_name,
            page_number_start=args.page_number_start,
        )
    else:
        written = scan_and_ingest_raw_root(
            raw_root=args.scan_root,
            out_dir=args.out,
            page_number_start=args.page_number_start,
        )

    print(f"Wrote {len(written)} page JSON files to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
