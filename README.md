# RAG Bot POC (Spanish catalog)

RAG proof-of-concept over an **unstructured Spanish product catalog** (PDF pages with mixed text/images/tables), using local embeddings and Qdrant Cloud for semantic retrieval.

## Quickstart

1. Create a venv and install deps:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create a local `.env` file from `.env.example` and fill in your keys.

3. Configure services:

```powershell
OPENROUTER_API_KEY=your-openrouter-api-key
EMBEDDING_MODEL=intfloat/multilingual-e5-small
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=catalog_es
```

4. Ingest OCR markdown into `data/ingested/pages`:

```bash
python -m ingestion.mistral_markdown_ingest --input-dir .\data\raw\mistra-ocr-output\output-1 --out .\data\ingested
```

Outputs:
- `data/ingested/pages/page_0001.json` (one file per page)

If your OCR folder page numbering should map to a different document range, use:

```bash
python -m ingestion.mistral_markdown_ingest --input-dir .\data\raw\ocr-playground-download-20260407T140841Z\200-396.pdf --out .\data\ingested --page-number-start 350
```

5. Rebuild BM25 + Qdrant:

```bash
python indexing\run_build.py
```

## Folder layout

- `data/raw/`: input PDFs
- `data/ingested/`: per-page JSON used for indexing and retrieval
- `ingestion/`: lightweight OCR-to-page-JSON ingestion code
- `retrieval/`: retrieval, fusion, and query helpers

