# datasets/raw — Source Data

Files in this folder are too large for the repository. Download from Google Drive:

> **Google Drive Link:** _[LINK TO BE ADDED]_

## Files

| File | Size | Description |
|---|---|---|
| `HI-Small_Trans.csv` | 454 MB | IBM HI-Small synthetic transaction dataset — 2M rows (Sep 2022). Columns: Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering |
| `HI-Small_accounts.csv` | 33 MB | Account metadata — 518k rows. Columns: Bank Name, Bank ID, Account Number, Entity ID, Entity Name |

## After downloading

Place both files here, then run:

```bash
python ingestion/narrative_generation/build_narratives.py
```
