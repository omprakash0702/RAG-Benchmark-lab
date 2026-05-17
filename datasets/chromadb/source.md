# datasets/chromadb — ChromaDB Vector Store

Files in this folder are too large for the repository. Download from Google Drive:

> **Google Drive Link:** _[LINK TO BE ADDED]_

## Contents

| File | Description |
|---|---|
| `chroma.sqlite3` | ChromaDB metadata and embedding index — ~2M transaction embeddings (all-MiniLM-L6-v2, 384 dimensions) |

## After downloading

Place `chroma.sqlite3` here. The Basic RAG pipeline (`pipelines/basic_rag/retrieve.py`) reads from this directory automatically.

## Regenerate instead

```bash
python ingestion/chunking/chunk_narratives.py
python ingestion/embeddings/build_embeddings.py
# Runtime: several hours for 2M records
```
