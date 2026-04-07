# RAG Bot POC (Spanish catalog)

Local-first RAG proof-of-concept over an **unstructured Spanish product catalog** (PDF pages with mixed text/images/tables).

## Quickstart (ingestion only)

1. Create a venv and install deps:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run docling ingestion (per-page JSON):

```bash
python -m ingestion.docling_ingest --pdf .\data\raw\catalog.pdf --out .\data\ingested
```

Outputs:
- `data/ingested/pages/page_0001.json` (one file per page)
- `data/ingested/assets/...` (exported page images / regions when available)

## Folder layout

- `data/raw/`: input PDFs
- `data/ingested/`: docling outputs (one JSON per page)
- `ingestion/`: offline ingestion code

