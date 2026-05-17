# datasets/chunks — Chunked Narratives

File in this folder is too large for the repository. Download from Google Drive:

> **Google Drive Link:** _[LINK TO BE ADDED]_

## File

| File | Description |
|---|---|
| `transaction_chunks.jsonl` | Chunked version of `transaction_narratives.jsonl` — input to the ChromaDB embedding pipeline |

## Regenerate instead

```bash
python ingestion/chunking/chunk_narratives.py
# Requires: datasets/processed/transaction_narratives.jsonl
```
