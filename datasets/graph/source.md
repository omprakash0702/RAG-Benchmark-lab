# datasets/graph — TigerGraph Import CSVs

Files in this folder are too large for the repository (~511 MB total). Download from Google Drive:

> **Google Drive Link:** _[LINK TO BE ADDED]_

## Files

| File | Size | Description |
|---|---|---|
| `accounts.csv` | 20 MB | Account vertices — account_id, entity_name, home_bank, entity_type |
| `transactions.csv` | 174 MB | Transaction vertices — tx_id, amount_paid, payment_currency, amount_received, receiving_currency, payment_format, timestamp, is_laundering, is_cross_bank, is_cross_currency, is_self_transfer |
| `banks.csv` | 263 KB | Bank vertices — bank_id, bank_name, country |
| `entities.csv` | 9.2 MB | Entity vertices — entity_id, entity_name, entity_type |
| `transferred_edges.csv` | 152 MB | TRANSFERRED_TO edges — direct account-to-account shortcut for BFS |
| `initiated_edges.csv` | 75 MB | INITIATED_TRANSACTION edges — account → transaction |
| `received_edges.csv` | 75 MB | RECEIVED_TRANSACTION edges — transaction → account |
| `bank_edges.csv` | 8.9 MB | BELONGS_TO_BANK edges — account → bank |

## After downloading

Load into TigerGraph:

```
# Schema: pipelines/graph_rag/tigergraph_schema.gsql
# Data:   pipelines/graph_rag/load_graph_data.gsql
```

## Regenerate instead

```bash
python pipelines/graph_rag/build_graph_data.py
# Requires: datasets/processed/transaction_narratives.jsonl
```
