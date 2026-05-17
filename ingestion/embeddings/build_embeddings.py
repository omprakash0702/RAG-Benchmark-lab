"""
build_embeddings.py

Generates local embeddings from chunked transaction narratives and stores them
in a persistent ChromaDB collection for retrieval benchmarking.

Input:  datasets/chunks/transaction_chunks.jsonl
Output: database/chroma/  (persistent ChromaDB storage)

--- Why phased embedding strategies matter ---
Embedding 5M+ vectors on CPU takes many hours. Iterating on retrieval quality,
chunk parameters, or prompt templates while waiting for a full run is impractical.
A phased approach lets you:
  1. Validate the full pipeline end-to-end on a small subset (DEV_MODE / --dev)
  2. Scale to a representative benchmark corpus (--max_chunks 500000)
  3. Run a complete production corpus only after the pipeline is validated (--full)

This is standard practice in large-scale RAG systems: index a representative
sample, tune retrieval parameters, then commit compute to the full corpus.
Full-corpus embedding is deferred until the pipeline, chunking strategy, and
embedding model choice are all confirmed — changing any of these invalidates the
entire vector store and requires a full re-embed.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Generator

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Execution mode defaults
# Change DEV_MODE / MAX_CHUNKS here, or override at runtime via argparse.
# ---------------------------------------------------------------------------

# Set True to process a tiny local subset — fast iteration during development.
DEV_MODE   = False

# Number of chunks to embed when not in dev/full mode.
# 2 000 000 ≈ 40 % of the full corpus — large enough for meaningful benchmarks,
# small enough to finish in reasonable wall-clock time on a CPU machine.
MAX_CHUNKS = 2_000_000

# Number of chunks in dev mode — just enough to exercise every code path.
DEV_SUBSET = 5_000

# ---------------------------------------------------------------------------
# Paths & model
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).resolve().parents[2]
INPUT_JSONL    = ROOT / "datasets" / "chunks" / "transaction_chunks.jsonl"
CHROMA_DIR     = ROOT / "database" / "chroma"
COLLECTION_NAME = "financial_transactions"
DEFAULT_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Batch size — single knob for both embedding and ChromaDB writes.
# Decoupling them adds complexity without meaningful benefit for this model/DB.
# 256 is the sweet spot for all-MiniLM-L6-v2 on CPU: enough parallelism in
# the tokenizer, small enough to not OOM on 8 GB RAM machines.
# ---------------------------------------------------------------------------
BATCH_SIZE = 256

# Storage estimate constants
BYTES_PER_VECTOR  = 384 * 4          # 384-dim float32
BYTES_PER_RECORD  = BYTES_PER_VECTOR + 600   # +metadata + document text + HNSW overhead

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
        description="Embed transaction chunks into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_embeddings.py                        # use DEV_MODE / MAX_CHUNKS defaults
  python build_embeddings.py --dev                  # fast dev subset (5 000 chunks)
  python build_embeddings.py --max_chunks 500000    # 500 k chunk benchmark run
  python build_embeddings.py --max_chunks 2000000   # 2 M chunk production run
  python build_embeddings.py --full                 # entire corpus (no limit)
        """,
    )
    p.add_argument(
        "--max_chunks", type=int, default=None,
        help="Maximum chunks to embed (overrides MAX_CHUNKS constant).",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Embed the complete corpus regardless of MAX_CHUNKS.",
    )
    p.add_argument(
        "--dev", action="store_true",
        help=f"Dev mode: process only {DEV_SUBSET} chunks for fast iteration.",
    )
    p.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Embedding + ChromaDB batch size (default: {BATCH_SIZE}).",
    )
    p.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def count_lines(path: Path) -> int:
    """Fast binary line count — avoids UTF-8 decode overhead on large files."""
    n = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            n += block.count(b"\n")
    return n


def load_chunks(path: Path, limit: int | None = None) -> Generator[dict, None, None]:
    """
    Stream chunk records from JSONL one line at a time, up to `limit` records.
    Streaming keeps RAM constant regardless of corpus size.
    """
    yielded = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            if limit is not None and yielded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                yielded += 1
            except json.JSONDecodeError as exc:
                log.warning("Malformed JSON at line %d — %s", line_num, exc)


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

def initialize_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and warm up a SentenceTransformer model."""
    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    model.encode(["warm-up"], show_progress_bar=False)  # trigger lazy init
    log.info("Model ready — embedding dimension: %d", model.get_embedding_dimension())
    return model


def calibrate_throughput(model: SentenceTransformer, samples: list[str]) -> float:
    """
    Run one real encode() call and return throughput in sentences/sec.
    Used to estimate wall-clock time before the main loop starts.
    """
    t0 = time.perf_counter()
    model.encode(samples, batch_size=len(samples), show_progress_bar=False,
                 normalize_embeddings=True)
    elapsed = time.perf_counter() - t0
    return len(samples) / elapsed if elapsed > 0 else float("inf")


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode texts into L2-normalised float32 vectors."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # unit vectors → cosine sim == dot product
        convert_to_numpy=True,
    )


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def initialize_chroma(
    db_path: Path,
    collection_name: str,
) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    """
    Open (or create) a persistent ChromaDB client and collection.
    embedding_function=None because all vectors arrive pre-computed —
    ChromaDB's built-in ONNX embedder is never invoked.
    """
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # matches normalised embeddings
        embedding_function=None,
    )
    log.info("ChromaDB path       : %s", db_path)
    log.info("Collection          : %s  (%d existing records)", collection_name, collection.count())
    return client, collection


def get_existing_ids(collection: chromadb.Collection) -> set[str]:
    """
    Fetch every chunk_id already stored in the collection.

    Paginated in batches of 50 000 to avoid a single massive response.
    Returns a set for O(1) membership checks during the main loop.

    This is the key to graceful resume: any chunk_id in this set is skipped,
    so interrupted runs continue exactly where they left off.
    """
    existing: set[str] = set()
    page_size = 50_000
    offset = 0
    while True:
        result = collection.get(
            limit=page_size,
            offset=offset,
            include=[],   # IDs only — no vectors, no documents, minimal I/O
        )
        batch_ids = result.get("ids", [])
        if not batch_ids:
            break
        existing.update(batch_ids)
        offset += len(batch_ids)
        if len(batch_ids) < page_size:
            break
    return existing


def _sanitize_metadata(chunk: dict) -> dict:
    """
    Coerce a chunk record's metadata into ChromaDB-safe types (str/int/float/bool).
    Promotes chunk_index, total_chunks, source_transaction_id to top-level so
    they are filterable at query time with where= clauses.
    """
    raw: dict = chunk.get("metadata", {})
    safe: dict = {}
    for key, val in raw.items():
        if isinstance(val, (str, int, float, bool)):
            safe[key] = val
        elif val is None:
            safe[key] = ""
        else:
            safe[key] = str(val)
    safe["chunk_index"]            = int(chunk.get("chunk_index", 0))
    safe["total_chunks"]           = int(chunk.get("total_chunks", 1))
    safe["source_transaction_id"]  = str(chunk.get("source_transaction_id", ""))
    return safe


def store_embeddings(
    collection: chromadb.Collection,
    ids: list[str],
    texts: list[str],
    embeddings: np.ndarray,
    metadatas: list[dict],
) -> None:
    """upsert() is idempotent on chunk_id — safe to call on duplicates."""
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )


# ---------------------------------------------------------------------------
# Pre-run summary
# ---------------------------------------------------------------------------

def print_prerun_summary(
    selected: int,
    already_done: int,
    to_embed: int,
    throughput_per_sec: float,
    batch_size: int,
) -> None:
    """Print estimated scope and resource impact before the main loop starts."""
    remaining_sec = to_embed / throughput_per_sec if throughput_per_sec > 0 else 0
    h, m, s = int(remaining_sec // 3600), int((remaining_sec % 3600) // 60), int(remaining_sec % 60)
    storage_mb = (to_embed * BYTES_PER_RECORD) / (1024 ** 2)

    print("\n" + "=" * 60)
    print("  PRE-RUN SUMMARY")
    print("=" * 60)
    print(f"  Chunks selected      : {selected:>12,}")
    print(f"  Already embedded     : {already_done:>12,}  (will be skipped)")
    print(f"  To embed this run    : {to_embed:>12,}")
    print(f"  Throughput (measured): {throughput_per_sec:>10,.0f}  chunks/sec")
    print(f"  Estimated runtime    : {h:02d}h {m:02d}m {s:02d}s")
    print(f"  Estimated new storage: {storage_mb:>10,.1f}  MB")
    print(f"  Batch size           : {batch_size:>12,}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(total_chars: int, chars_per_token: int = 4) -> int:
    return total_chars // chars_per_token


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve execution mode
    if args.full:
        limit = None
        mode_label = "FULL corpus"
    elif args.dev:
        limit = DEV_SUBSET
        mode_label = f"DEV ({DEV_SUBSET:,} chunks)"
    elif args.max_chunks is not None:
        limit = args.max_chunks
        mode_label = f"{limit:,} chunks"
    elif DEV_MODE:
        limit = DEV_SUBSET
        mode_label = f"DEV ({DEV_SUBSET:,} chunks)"
    else:
        limit = MAX_CHUNKS
        mode_label = f"{limit:,} chunks (MAX_CHUNKS)"

    t_start = time.perf_counter()
    log.info("=== build_embeddings.py started — mode: %s ===", mode_label)

    # -- Validate input ------------------------------------------------------
    if not INPUT_JSONL.exists():
        log.error("Input not found: %s", INPUT_JSONL)
        log.error("Run ingestion/chunking/chunk_narratives.py first.")
        sys.exit(1)

    # -- Count available chunks for display ----------------------------------
    log.info("Counting available chunks …")
    available = count_lines(INPUT_JSONL)
    selected  = available if limit is None else min(limit, available)
    log.info("Available chunks: %d  |  Selected: %d", available, selected)

    # -- Initialise model and DB --------------------------------------------
    model          = initialize_embedding_model(args.model)
    embedding_dim  = model.get_embedding_dimension()
    _, collection  = initialize_chroma(CHROMA_DIR, COLLECTION_NAME)

    # -- Load existing IDs for duplicate / resume protection -----------------
    log.info("Fetching existing IDs from ChromaDB (resume check) …")
    existing_ids = get_existing_ids(collection)
    already_done = len(existing_ids)
    log.info("Already embedded: %d", already_done)

    # -- Calibrate throughput on a small real sample -------------------------
    # Use up to 256 real chunks so the estimate reflects actual text lengths.
    calibration_texts: list[str] = []
    for rec in load_chunks(INPUT_JSONL, limit=BATCH_SIZE):
        t = rec.get("chunk_text", "").strip()
        if t:
            calibration_texts.append(t)

    throughput = calibrate_throughput(model, calibration_texts) if calibration_texts else 500.0
    to_embed   = max(0, selected - already_done)

    print_prerun_summary(selected, already_done, to_embed, throughput, args.batch_size)

    if to_embed == 0:
        log.info("Nothing to embed — all selected chunks are already in ChromaDB.")
        return

    # -- Main loop -----------------------------------------------------------
    pending_ids:   list[str]  = []
    pending_texts: list[str]  = []
    pending_meta:  list[dict] = []

    total_stored   = 0
    total_skipped  = 0
    total_chars    = 0
    loop_start     = time.perf_counter()
    last_log_time  = loop_start
    last_log_count = 0

    def _flush() -> None:
        """Embed and upsert the current pending batch in-place."""
        nonlocal total_stored
        vecs = generate_embeddings(model, pending_texts, args.batch_size)
        store_embeddings(collection, pending_ids, pending_texts, vecs, pending_meta)
        total_stored += len(pending_ids)
        pending_ids.clear()
        pending_texts.clear()
        pending_meta.clear()

    log.info("Starting embedding loop …")
    with tqdm(total=selected, unit="chunks", desc="Embedding", ncols=90, initial=already_done) as pbar:
        for chunk in load_chunks(INPUT_JSONL, limit=selected):
            chunk_id   = chunk.get("chunk_id", "")
            chunk_text = chunk.get("chunk_text", "").strip()

            if not chunk_id or not chunk_text:
                total_skipped += 1
                pbar.update(1)
                continue

            # Resume: skip chunks already persisted in ChromaDB
            if chunk_id in existing_ids:
                pbar.update(1)
                continue

            pending_ids.append(chunk_id)
            pending_texts.append(chunk_text)
            pending_meta.append(_sanitize_metadata(chunk))
            total_chars += len(chunk_text)

            if len(pending_ids) >= args.batch_size:
                try:
                    _flush()
                except Exception as exc:
                    log.error("Batch upsert failed — %s", exc)
                    total_skipped += len(pending_ids)
                    pending_ids.clear(); pending_texts.clear(); pending_meta.clear()

                # Detailed progress every 30 seconds
                now = time.perf_counter()
                if now - last_log_time >= 30:
                    elapsed       = now - loop_start
                    rate          = (total_stored) / elapsed if elapsed > 0 else 0
                    remaining     = (to_embed - total_stored) / rate if rate > 0 else 0
                    h, m, s = int(remaining // 3600), int((remaining % 3600) // 60), int(remaining % 60)
                    log.info(
                        "Progress: %d/%d stored | %.0f chunks/sec | ETA %02dh%02dm%02ds",
                        total_stored, to_embed, rate, h, m, s,
                    )
                    last_log_time  = now
                    last_log_count = total_stored

            pbar.update(1)

        # Flush final partial batch
        if pending_ids:
            try:
                _flush()
            except Exception as exc:
                log.error("Final batch upsert failed — %s", exc)
                total_skipped += len(pending_ids)

    # -- Final stats ---------------------------------------------------------
    persisted_count  = collection.count()
    elapsed          = time.perf_counter() - t_start
    avg_rate         = total_stored / elapsed if elapsed > 0 else 0
    estimated_tokens = estimate_tokens(total_chars)

    log.info("=== Run complete ===")
    log.info("Embeddings stored this run : %d", total_stored)
    log.info("Total in ChromaDB          : %d", persisted_count)
    log.info("Skipped / errors           : %d", total_skipped)
    log.info("Embedding dimension        : %d", embedding_dim)
    log.info("Collection name            : %s", COLLECTION_NAME)
    log.info("Estimated tokens embedded  : %d", estimated_tokens)
    log.info("Avg throughput             : %.0f chunks/sec", avg_rate)
    log.info("Total elapsed              : %.1f s", elapsed)

    print("\n" + "=" * 60)
    print(f"  Embeddings this run  : {total_stored:,}")
    print(f"  Total in ChromaDB    : {persisted_count:,}")
    print(f"  Embedding dimension  : {embedding_dim}")
    print(f"  Collection name      : {COLLECTION_NAME}")
    print(f"  Estimated tokens     : {estimated_tokens:,}")
    print(f"  Avg throughput       : {avg_rate:,.0f} chunks/sec")
    print(f"  Elapsed              : {elapsed:.1f} s")
    print(f"  ChromaDB path        : {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
