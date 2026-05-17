# database — ChromaDB Persistent Storage

Files in this folder are too large for the repository. Download from Google Drive:

> **Google Drive Link:** _[LINK TO BE ADDED]_

## Contents

```
database/chroma/
├── chroma.sqlite3          # Main database
└── <collection-uuid>/
    ├── data_level0.bin     # HNSW index — embedding vectors
    ├── header.bin
    ├── index_metadata.pickle
    ├── length.bin
    └── link_lists.bin
```

## After downloading

Place the entire `chroma/` folder inside `database/`. The Streamlit dashboard and Basic RAG pipeline point to `database/chroma/` by default.

## Regenerate instead

```bash
python ingestion/embeddings/build_embeddings.py
```
