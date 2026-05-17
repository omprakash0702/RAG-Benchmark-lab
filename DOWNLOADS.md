# Large File Downloads

All large data files are hosted on Google Drive. After cloning the repo, download the relevant files based on which pipelines you want to run.

> **Google Drive Link:** _[LINK TO BE ADDED]_

## What's on the Drive

| File / Folder | Size | Required for | Regenerate instead? |
|---|---|---|---|
| `datasets/raw/HI-Small_Trans.csv` | 454 MB | Everything | Source data — cannot regenerate |
| `datasets/raw/HI-Small_accounts.csv` | 33 MB | Everything | Source data — cannot regenerate |
| `datasets/processed/transaction_narratives.jsonl` | 2.7 GB | Basic RAG + GraphRAG ingestion | `python ingestion/narrative_generation/build_narratives.py` |
| `datasets/chunks/transaction_chunks.jsonl` | ~500 MB | ChromaDB ingestion | `python ingestion/chunking/chunk_narratives.py` |
| `datasets/graph/*.csv` (8 files) | 511 MB | TigerGraph load | `python pipelines/graph_rag/build_graph_data.py` |
| `datasets/chromadb/chroma.sqlite3` | ~1 GB | Basic RAG pipeline | `python ingestion/embeddings/build_embeddings.py` |
| `database/chroma/` (folder) | ~1 GB | Streamlit dashboard (Basic RAG) | `python ingestion/embeddings/build_embeddings.py` |
| `tigergraph-4.2.2-community-docker-image.tar.gz` | 1.5 GB | TigerGraph local setup | Download from [TigerGraph CE](https://dl.tigergraph.com) |

## Minimum Download for Quick Start

To run only the **Streamlit dashboard** with all three pipelines:

1. `datasets/raw/` — both CSVs (required for LLM-Only context)
2. `database/chroma/` — ChromaDB store (required for Basic RAG)
3. TigerGraph running with FinancialGraph loaded (required for GraphRAG)

## Setup After Downloading

See [README.md](README.md#setup--installation) for full setup instructions.
