"""
build_narratives.py

Converts the banking relational dataset (HI-Small) into semantic transaction
narratives stored as JSONL. Designed to feed downstream phases:
  - chunking / embeddings
  - Basic RAG (ChromaDB)
  - GraphRAG (TigerGraph)
  - Retrieval benchmarking

Processes ~5M rows in memory-safe chunks and joins account metadata inline.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
TRANS_CSV = ROOT / "datasets" / "raw" / "HI-Small_Trans.csv"
ACCT_CSV = ROOT / "datasets" / "raw" / "HI-Small_accounts.csv"
OUTPUT_JSONL = ROOT / "datasets" / "processed" / "transaction_narratives.jsonl"

# Rows per pandas chunk — tuned for ~200 MB RAM headroom on a 434 MB file
CHUNK_SIZE = 100_000

# Rough token estimate: GPT-style avg 4 chars per token
CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def inspect_columns(path: Path, label: str) -> list[str]:
    """Read only the header row and print column names."""
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    log.info("Columns in %s (%s):", label, path.name)
    for i, c in enumerate(cols, 1):
        log.info("  %2d. %s", i, c)
    return cols


def load_accounts(path: Path) -> dict[str, dict]:
    """
    Load the accounts CSV into a dict keyed by Account Number for O(1) lookup.
    518 k rows × ~5 cols is ~50 MB — safe to hold in memory.
    Returns: { account_number: {bank_name, entity_name, entity_id} }
    """
    log.info("Loading accounts lookup table …")
    df = pd.read_csv(path, dtype=str).fillna("")
    # Normalise column names defensively in case schema drifts
    df.columns = df.columns.str.strip()
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        acct_num = _safe(row.get("Account Number"))
        if acct_num == "unknown":
            continue
        lookup[acct_num] = {
            "bank_name": _safe(row.get("Bank Name")),
            "entity_name": _safe(row.get("Entity Name")),
            "entity_id": _safe(row.get("Entity ID")),
        }
    log.info("Accounts lookup built: %d entries", len(lookup))
    return lookup


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def _safe(value, default: str = "unknown") -> str:
    """Return a clean string or a fallback for None / NaN / empty values."""
    if value is None:
        return default
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none", "") else default


def _format_amount(value) -> str:
    """Format a numeric amount with comma separators and 2 decimal places."""
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "unknown"


def build_narrative(row: pd.Series, acct_lookup: dict[str, dict]) -> str:
    """
    Construct a human-readable sentence from a single transaction row.

    Column mapping after duplicate-rename:
        Timestamp, From Bank, Sender Account, To Bank, Receiver Account,
        Amount Received, Receiving Currency, Amount Paid, Payment Currency,
        Payment Format, Is Laundering
    """
    sender_acct = _safe(row.get("Sender Account"))
    receiver_acct = _safe(row.get("Receiver Account"))
    from_bank_id = _safe(row.get("From Bank"))
    to_bank_id = _safe(row.get("To Bank"))
    amount_paid = _format_amount(row.get("Amount Paid"))
    pay_currency = _safe(row.get("Payment Currency"))
    amount_recv = _format_amount(row.get("Amount Received"))
    recv_currency = _safe(row.get("Receiving Currency"))
    pay_format = _safe(row.get("Payment Format"))
    timestamp = _safe(row.get("Timestamp"))

    # Enrich sender / receiver with entity names when available
    sender_info = acct_lookup.get(sender_acct, {})
    receiver_info = acct_lookup.get(receiver_acct, {})

    sender_label = (
        f"{sender_info['entity_name']} (Account {sender_acct})"
        if sender_info.get("entity_name")
        else f"Account {sender_acct}"
    )
    receiver_label = (
        f"{receiver_info['entity_name']} (Account {receiver_acct})"
        if receiver_info.get("entity_name")
        else f"Account {receiver_acct}"
    )

    # Resolve bank names from lookup; fall back to bank IDs from the transaction
    sender_bank = sender_info.get("bank_name") or f"Bank {from_bank_id}"
    receiver_bank = receiver_info.get("bank_name") or f"Bank {to_bank_id}"

    # Currency note — only add if paid and received currencies differ
    currency_note = ""
    if pay_currency != recv_currency and pay_currency != "unknown" and recv_currency != "unknown":
        currency_note = f", converted to {amount_recv} {recv_currency},"

    narrative = (
        f"{sender_label} at {sender_bank} transferred {amount_paid} {pay_currency}"
        f"{currency_note} to {receiver_label} at {receiver_bank} "
        f"via {pay_format} on {timestamp}."
    )
    return narrative


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def write_record(fh, tx_id: str, narrative: str, row: pd.Series) -> None:
    """Serialise one transaction to a JSONL line."""
    record = {
        "transaction_id": tx_id,
        "narrative": narrative,
        "metadata": {
            "sender": _safe(row.get("Sender Account")),
            "receiver": _safe(row.get("Receiver Account")),
            "from_bank": _safe(row.get("From Bank")),
            "to_bank": _safe(row.get("To Bank")),
            "amount_paid": _safe(row.get("Amount Paid")),
            "payment_currency": _safe(row.get("Payment Currency")),
            "amount_received": _safe(row.get("Amount Received")),
            "receiving_currency": _safe(row.get("Receiving Currency")),
            "payment_format": _safe(row.get("Payment Format")),
            "timestamp": _safe(row.get("Timestamp")),
            "is_laundering": _safe(row.get("Is Laundering")),
        },
    }
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    start = datetime.now()
    log.info("=== build_narratives.py started ===")

    # -- Validate inputs -----------------------------------------------------
    for p in (TRANS_CSV, ACCT_CSV):
        if not p.exists():
            log.error("Missing input file: %s", p)
            sys.exit(1)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    # -- Inspect schemas (always run so operator can verify column alignment) -
    log.info("--- Transaction CSV schema ---")
    inspect_columns(TRANS_CSV, "transactions")
    log.info("--- Accounts CSV schema ---")
    inspect_columns(ACCT_CSV, "accounts")

    # -- Build account lookup ------------------------------------------------
    acct_lookup = load_accounts(ACCT_CSV)

    # -- Count total rows for the progress bar (reads only row count) ---------
    log.info("Counting transaction rows (this may take a moment) …")
    total_rows = sum(1 for _ in open(TRANS_CSV, encoding="utf-8")) - 1  # subtract header
    log.info("Total transactions to process: %d", total_rows)

    # -- Stream-process the transaction CSV in chunks -------------------------
    # The transaction CSV has two columns both named "Account" (sender + receiver).
    # mangle_dupe_cols was removed in pandas 2.0, so we read the raw header first,
    # build deduplicated names ourselves, then pass them via names= + header=0.
    with open(TRANS_CSV, encoding="utf-8") as _fh:
        raw_header = _fh.readline().strip().split(",")

    seen: dict[str, int] = {}
    custom_cols: list[str] = []
    for col in raw_header:
        col = col.strip()
        seen[col] = seen.get(col, 0) + 1
        custom_cols.append(col if seen[col] == 1 else f"{col}_{seen[col]}")

    # Map the deduped names to semantic names used throughout this script
    col_renames = {c: c for c in custom_cols}   # identity by default
    if "Account" in custom_cols:
        col_renames["Account"] = "Sender Account"
    # Second occurrence of "Account" becomes "Account_2"
    if "Account_2" in custom_cols:
        col_renames["Account_2"] = "Receiver Account"

    total_narratives = 0
    total_chars = 0

    reader = pd.read_csv(
        TRANS_CSV,
        chunksize=CHUNK_SIZE,
        dtype=str,          # read everything as str to avoid mixed-type surprises
        names=custom_cols,  # supply deduplicated header
        header=0,           # skip the original header row
    )

    log.info("Writing narratives to: %s", OUTPUT_JSONL)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_fh, \
         tqdm(total=total_rows, unit="rows", desc="Narratives", ncols=90) as pbar:

        for chunk_idx, chunk in enumerate(reader):
            # Apply semantic renames (Account → Sender Account, Account_2 → Receiver Account)
            chunk.rename(columns=col_renames, inplace=True)

            # Strip whitespace from string columns to avoid dirty join keys
            chunk = chunk.apply(lambda col: col.str.strip() if col.dtype == object else col)

            for local_idx, row in chunk.iterrows():
                # Stable, reproducible transaction ID: chunk offset + local row index
                global_idx = chunk_idx * CHUNK_SIZE + (local_idx % CHUNK_SIZE)
                tx_id = f"TX{global_idx:08d}"

                try:
                    narrative = build_narrative(row, acct_lookup)
                    write_record(out_fh, tx_id, narrative, row)
                    total_chars += len(narrative)
                    total_narratives += 1
                except Exception as exc:
                    # Log bad rows but never abort the whole run
                    log.warning("Skipped row %s — %s", tx_id, exc)

            pbar.update(len(chunk))

    # -- Summary stats --------------------------------------------------------
    elapsed = (datetime.now() - start).total_seconds()
    estimated_tokens = total_chars // CHARS_PER_TOKEN
    avg_length = total_chars / total_narratives if total_narratives else 0

    log.info("=== Run complete ===")
    log.info("Total narratives generated : %d", total_narratives)
    log.info("Estimated token count      : %d", estimated_tokens)
    log.info("Average narrative length   : %.1f chars", avg_length)
    log.info("Output file                : %s", OUTPUT_JSONL)
    log.info("Elapsed time               : %.1f s", elapsed)

    # Final print for easy capture in CI / notebook output
    print("\n" + "=" * 60)
    print(f"  Narratives generated : {total_narratives:,}")
    print(f"  Estimated tokens     : {estimated_tokens:,}")
    print(f"  Avg narrative length : {avg_length:.1f} chars")
    print(f"  Output               : {OUTPUT_JSONL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
