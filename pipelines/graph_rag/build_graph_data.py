"""
build_graph_data.py — GraphRAG Graph Construction Pipeline

Transforms the transaction narratives JSONL into TigerGraph-ingestible node
and edge CSVs.  Each CSV maps directly to a vertex or edge type defined in the
TigerGraph schema for the HI-Small AML benchmark.

NODE TYPES
  Account     — unique financial account (sender or receiver)
  Bank        — financial institution identified by bank code
  Entity      — named business / person behind an account (from accounts CSV)
  Transaction — individual money movement event

EDGE TYPES
  TRANSFERRED_TO          Account → Account  (every transfer)
  BELONGS_TO_BANK         Account → Bank     (account's home institution)
  INITIATED_TRANSACTION   Account → Transaction (sender role)
  RECEIVED_TRANSACTION    Account → Transaction (receiver role)
  SAME_BANK_TRANSFER      Transaction attr   (from_bank == to_bank)
  CROSS_BANK_TRANSFER     Transaction attr   (from_bank != to_bank)

DESIGN
  - Streams the 5M-row JSONL in chunks; never loads the full file into memory.
  - Accounts and banks are deduplicated in-memory (two small dicts) and flushed
    at the end — cardinality is bounded (~300k accounts, <200 banks).
  - Transactions and edges are streamed directly to CSV writers, one row per
    narrative record, so memory use stays constant regardless of corpus size.
  - All output files are written to datasets/graph/.
  - If HI-Small_accounts.csv is present, entity names are joined in; otherwise
    Entity nodes are omitted and a warning is logged.
  - --dev flag processes only the first DEV_ROWS rows for fast iteration.

USAGE
  python pipelines/graph_rag/build_graph_data.py
  python pipelines/graph_rag/build_graph_data.py --dev
  python pipelines/graph_rag/build_graph_data.py --chunk_size 100000
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
NARRATIVES   = ROOT / "datasets" / "processed" / "transaction_narratives.jsonl"
ACCOUNTS_RAW = ROOT / "datasets" / "raw" / "HI-Small_accounts.csv"
GRAPH_DIR    = ROOT / "datasets" / "graph"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 50_000
DEV_ROWS   = 2_000_000   # rows processed in --dev mode

# Output filenames
OUT_ACCOUNTS    = GRAPH_DIR / "accounts.csv"
OUT_BANKS       = GRAPH_DIR / "banks.csv"
OUT_ENTITIES    = GRAPH_DIR / "entities.csv"
OUT_TXNS        = GRAPH_DIR / "transactions.csv"
OUT_TRANSFERRED = GRAPH_DIR / "transferred_edges.csv"
OUT_BANK_EDGES  = GRAPH_DIR / "bank_edges.csv"
OUT_INITIATED   = GRAPH_DIR / "initiated_edges.csv"
OUT_RECEIVED    = GRAPH_DIR / "received_edges.csv"

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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build TigerGraph-ingestible node/edge CSVs from narratives JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dev", action="store_true",
                   help=f"Process only first {DEV_ROWS:,} rows (fast iteration).")
    p.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                   help=f"Rows per JSONL read chunk (default: {CHUNK_SIZE:,}).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files (default: abort if present).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Guard: refuse to clobber existing outputs without --overwrite
# ---------------------------------------------------------------------------

def check_existing_outputs(overwrite: bool) -> None:
    existing = [f for f in [
        OUT_ACCOUNTS, OUT_BANKS, OUT_ENTITIES, OUT_TXNS,
        OUT_TRANSFERRED, OUT_BANK_EDGES, OUT_INITIATED, OUT_RECEIVED,
    ] if f.exists()]
    if existing and not overwrite:
        log.error(
            "Output files already exist. Use --overwrite to replace them:\n  %s",
            "\n  ".join(str(f) for f in existing),
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Optional: load entity names from raw accounts CSV
# ---------------------------------------------------------------------------

def load_entity_names() -> dict[str, str]:
    """Return {account_id -> entity_name} from HI-Small_accounts.csv if present."""
    if not ACCOUNTS_RAW.exists():
        log.warning(
            "HI-Small_accounts.csv not found at %s — Entity nodes will be omitted.",
            ACCOUNTS_RAW,
        )
        return {}

    log.info("Loading entity names from %s ...", ACCOUNTS_RAW.name)
    try:
        # Columns: Bank Name(0), Bank ID(1), Account Number(2), Entity ID(3), Entity Name(4)
        df = pd.read_csv(ACCOUNTS_RAW, usecols=[2, 4], header=0,
                         dtype=str, na_filter=False)
        df.columns = ["account_id", "entity_name"]
        result = dict(zip(df["account_id"], df["entity_name"]))
        log.info("  Loaded %d entity names.", len(result))
        return result
    except Exception as exc:
        log.warning("Could not load entity names: %s — continuing without.", exc)
        return {}


# ---------------------------------------------------------------------------
# CSV writers (opened once, kept open for the full streaming pass)
# ---------------------------------------------------------------------------

def open_streaming_writers(
    txn_f, transferred_f, initiated_f, received_f
) -> tuple:
    """Return four DictWriter objects for the streaming output files."""
    txn_writer = csv.DictWriter(txn_f, fieldnames=[
        "transaction_id",
        "amount_paid", "payment_currency",
        "amount_received", "receiving_currency",
        "payment_format", "timestamp",
        "is_laundering",
        "is_self_transfer", "is_cross_bank", "is_cross_currency",
        "transfer_type",
    ])
    txn_writer.writeheader()

    transferred_writer = csv.DictWriter(transferred_f, fieldnames=[
        "from_account", "to_account",
        "transaction_id",
        "amount_paid", "payment_currency",
        "timestamp", "transfer_type",
        "is_laundering",
    ])
    transferred_writer.writeheader()

    initiated_writer = csv.DictWriter(initiated_f, fieldnames=[
        "account_id", "transaction_id", "timestamp",
    ])
    initiated_writer.writeheader()

    received_writer = csv.DictWriter(received_f, fieldnames=[
        "account_id", "transaction_id", "timestamp",
    ])
    received_writer.writeheader()

    return txn_writer, transferred_writer, initiated_writer, received_writer


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------

def process_chunk(
    chunk: pd.DataFrame,
    accounts: dict,     # {account_id -> {"bank": bank_id}}  (mutated in-place)
    banks: set,         # set of bank_ids  (mutated in-place)
    txn_writer: csv.DictWriter,
    transferred_writer: csv.DictWriter,
    initiated_writer: csv.DictWriter,
    received_writer: csv.DictWriter,
) -> int:
    """Process one JSONL chunk; returns count of rows written."""
    meta_df = pd.json_normalize(chunk["metadata"])
    meta_df["transaction_id"] = chunk["transaction_id"].values

    rows_written = 0
    for row in meta_df.itertuples(index=False):
        sender   = str(row.sender).strip()
        receiver = str(row.receiver).strip()
        from_bank = str(row.from_bank).strip()
        to_bank   = str(row.to_bank).strip()
        amount_paid        = str(row.amount_paid).strip()
        payment_currency   = str(row.payment_currency).strip()
        amount_received    = str(row.amount_received).strip()
        receiving_currency = str(row.receiving_currency).strip()
        payment_format     = str(row.payment_format).strip()
        timestamp          = str(row.timestamp).strip()
        is_laundering      = str(row.is_laundering).strip()
        tid                = str(row.transaction_id).strip()

        # Derived flags
        is_self      = (sender == receiver)
        is_cross_bank = (from_bank != to_bank)
        is_cross_curr = (payment_currency != receiving_currency)

        if is_self:
            transfer_type = "SELF_TRANSFER"
        elif is_cross_bank:
            transfer_type = "CROSS_BANK"
        else:
            transfer_type = "SAME_BANK"

        # Accumulate account nodes
        if sender not in accounts:
            accounts[sender] = {"bank": from_bank}
        if receiver not in accounts:
            accounts[receiver] = {"bank": to_bank}

        # Accumulate bank nodes
        banks.add(from_bank)
        banks.add(to_bank)

        # Transaction node
        txn_writer.writerow({
            "transaction_id":     tid,
            "amount_paid":        amount_paid,
            "payment_currency":   payment_currency,
            "amount_received":    amount_received,
            "receiving_currency": receiving_currency,
            "payment_format":     payment_format,
            "timestamp":          timestamp,
            "is_laundering":      is_laundering,
            "is_self_transfer":   int(is_self),
            "is_cross_bank":      int(is_cross_bank),
            "is_cross_currency":  int(is_cross_curr),
            "transfer_type":      transfer_type,
        })

        # TRANSFERRED_TO edge: Account -> Account
        transferred_writer.writerow({
            "from_account":     sender,
            "to_account":       receiver,
            "transaction_id":   tid,
            "amount_paid":      amount_paid,
            "payment_currency": payment_currency,
            "timestamp":        timestamp,
            "transfer_type":    transfer_type,
            "is_laundering":    is_laundering,
        })

        # INITIATED_TRANSACTION edge: sender Account -> Transaction
        initiated_writer.writerow({
            "account_id":     sender,
            "transaction_id": tid,
            "timestamp":      timestamp,
        })

        # RECEIVED_TRANSACTION edge: receiver Account -> Transaction
        received_writer.writerow({
            "account_id":     receiver,
            "transaction_id": tid,
            "timestamp":      timestamp,
        })

        rows_written += 1

    return rows_written


# ---------------------------------------------------------------------------
# Final node CSV writers (flushed after streaming pass)
# ---------------------------------------------------------------------------

def write_account_nodes(accounts: dict, entity_names: dict) -> int:
    with open(OUT_ACCOUNTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["account_id", "home_bank", "entity_name"])
        w.writeheader()
        for acc_id, props in accounts.items():
            w.writerow({
                "account_id":  acc_id,
                "home_bank":   props["bank"],
                "entity_name": entity_names.get(acc_id, ""),
            })
    return len(accounts)


def write_bank_nodes(banks: set) -> int:
    with open(OUT_BANKS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bank_id"])
        w.writeheader()
        for bank_id in sorted(banks):
            w.writerow({"bank_id": bank_id})
    return len(banks)


def write_entity_nodes(accounts: dict, entity_names: dict) -> int:
    if not entity_names:
        log.info("Skipping entities.csv — no entity names available.")
        return 0

    # Entities: unique names that appear as account holders
    seen_entities: dict[str, str] = {}   # name -> first account_id seen
    for acc_id in accounts:
        name = entity_names.get(acc_id, "")
        if name and name not in seen_entities:
            seen_entities[name] = acc_id

    with open(OUT_ENTITIES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["entity_id", "entity_name", "representative_account"])
        w.writeheader()
        for name, acc_id in seen_entities.items():
            entity_id = "ENT_" + name.replace(" ", "_").upper()
            w.writerow({
                "entity_id":             entity_id,
                "entity_name":           name,
                "representative_account": acc_id,
            })
    return len(seen_entities)


def write_bank_edges(accounts: dict) -> int:
    """BELONGS_TO_BANK: Account -> Bank (one row per unique account)."""
    count = 0
    with open(OUT_BANK_EDGES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["account_id", "bank_id"])
        w.writeheader()
        for acc_id, props in accounts.items():
            w.writerow({"account_id": acc_id, "bank_id": props["bank"]})
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not NARRATIVES.exists():
        log.error("Narratives file not found: %s", NARRATIVES)
        sys.exit(1)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    check_existing_outputs(args.overwrite)

    entity_names = load_entity_names()

    # In-memory dedup structures (bounded cardinality)
    accounts: dict[str, dict] = {}
    banks: set[str] = set()

    log.info("=== Graph data build started ===")
    log.info("Input   : %s", NARRATIVES)
    log.info("Output  : %s", GRAPH_DIR)
    log.info("Dev mode: %s%s", args.dev, f" ({DEV_ROWS:,} rows)" if args.dev else "")
    log.info("Chunk   : %d rows", args.chunk_size)

    t0 = time.perf_counter()
    total_rows = 0
    chunk_count = 0

    with (
        open(OUT_TXNS,        "w", newline="", encoding="utf-8") as txn_f,
        open(OUT_TRANSFERRED, "w", newline="", encoding="utf-8") as transferred_f,
        open(OUT_INITIATED,   "w", newline="", encoding="utf-8") as initiated_f,
        open(OUT_RECEIVED,    "w", newline="", encoding="utf-8") as received_f,
    ):
        txn_w, transferred_w, initiated_w, received_w = open_streaming_writers(
            txn_f, transferred_f, initiated_f, received_f
        )

        reader = pd.read_json(NARRATIVES, lines=True, chunksize=args.chunk_size)

        for chunk in reader:
            if args.dev and total_rows >= DEV_ROWS:
                break

            if args.dev:
                remaining = DEV_ROWS - total_rows
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]

            written = process_chunk(
                chunk, accounts, banks,
                txn_w, transferred_w, initiated_w, received_w,
            )
            total_rows += written
            chunk_count += 1

            if chunk_count % 10 == 0:
                elapsed = time.perf_counter() - t0
                rate = total_rows / elapsed if elapsed > 0 else 0
                log.info(
                    "  Chunk %d | rows=%d | accounts=%d | banks=%d | %.0f rows/s",
                    chunk_count, total_rows, len(accounts), len(banks), rate,
                )

    # Flush dedup structures to CSVs
    log.info("Writing node CSVs ...")
    n_accounts = write_account_nodes(accounts, entity_names)
    n_banks    = write_bank_nodes(banks)
    n_entities = write_entity_nodes(accounts, entity_names)
    n_bank_edges = write_bank_edges(accounts)

    elapsed = time.perf_counter() - t0

    log.info("=== Graph data build complete ===")
    log.info("  Elapsed          : %.1f s", elapsed)
    log.info("  Transactions     : %d", total_rows)
    log.info("")
    log.info("  NODE COUNTS")
    log.info("    Account        : %d", n_accounts)
    log.info("    Bank           : %d", n_banks)
    log.info("    Entity         : %d", n_entities)
    log.info("    Transaction    : %d", total_rows)
    log.info("")
    log.info("  EDGE COUNTS (approx)")
    log.info("    TRANSFERRED_TO          : %d", total_rows)
    log.info("    BELONGS_TO_BANK         : %d", n_bank_edges)
    log.info("    INITIATED_TRANSACTION   : %d", total_rows)
    log.info("    RECEIVED_TRANSACTION    : %d", total_rows)
    log.info("")
    log.info("  OUTPUT FILES")
    for f in [OUT_ACCOUNTS, OUT_BANKS, OUT_ENTITIES, OUT_TXNS,
              OUT_TRANSFERRED, OUT_BANK_EDGES, OUT_INITIATED, OUT_RECEIVED]:
        size_mb = f.stat().st_size / 1_048_576 if f.exists() else 0
        log.info("    %-35s  %.1f MB", f.name, size_mb)


if __name__ == "__main__":
    main()
