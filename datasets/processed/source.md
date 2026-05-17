# datasets/processed — Generated Narrative Data

File in this folder is too large for the repository (~2.7 GB). Download from Google Drive:

> **Google Drive Link:** _[LINK TO BE ADDED]_

## File

| File | Size | Description |
|---|---|---|
| `transaction_narratives.jsonl` | ~2.7 GB | One JSON record per transaction. Each contains a natural-language narrative ("Account X sent $A to Y via Wire on 2022-09-01") plus full metadata. Generated from `datasets/raw/` by `build_narratives.py`. |

## Regenerate instead

```bash
python ingestion/narrative_generation/build_narratives.py
# Runtime: ~2 hours for 2M records
```
