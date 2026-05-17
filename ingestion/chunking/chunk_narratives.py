"""
chunk_narratives.py

Chunks semantic transaction narratives into fixed-size text segments suitable
for downstream embedding, retrieval, and benchmarking experiments.

Input:  datasets/processed/transaction_narratives.jsonl
Output: datasets/chunks/transaction_chunks.jsonl

Designed for future experimentation with:
  - different chunk_size / chunk_overlap settings
  - ChromaDB and TigerGraph ingestion
  - retrieval quality and latency benchmarking
"""

import json
import logging
import sys
from pathlib import Path
from typing import Generator, Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = ROOT / "datasets" / "processed" / "transaction_narratives.jsonl"
OUTPUT_JSONL = ROOT / "datasets" / "chunks" / "transaction_chunks.jsonl"

# ---------------------------------------------------------------------------
# Splitter defaults — change here or override via CLI args in future phases
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 75

# Rough token estimate: GPT-style avg 4 chars per token
CHARS_PER_TOKEN = 4

# Write buffer: flush to disk every N chunks (reduces syscall overhead)
WRITE_BUFFER_SIZE = 5_000

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
# I/O helpers
# ---------------------------------------------------------------------------

def count_lines(path: Path) -> int:
    """Count lines without loading the file into memory."""
    total = 0
    with open(path, "rb") as fh:         # binary read is faster than text mode
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            total += chunk.count(b"\n")
    return total


def load_narratives(path: Path) -> Iterator[dict]:
    """
    Stream records from a JSONL file one at a time.
    Never loads the full file into memory — safe for multi-million-row files.

    Yields dicts with keys: transaction_id, narrative, metadata
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d — %s", line_num, exc)


def save_chunks(chunks: list[dict], fh) -> None:
    """Write a batch of chunk dicts to an already-open JSONL file handle."""
    for chunk in chunks:
        fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Construct a RecursiveCharacterTextSplitter with the given parameters.
    Separated so callers can swap sizes without touching the rest of the pipeline.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Sentence-aware separators — tries to split on full stops before spaces
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(
    records: Iterator[dict],
    splitter: RecursiveCharacterTextSplitter,
    total: int,
) -> Generator[dict, None, None]:
    """
    Yield chunk dicts for every record produced by the records iterator.

    Each yielded dict contains:
        chunk_id             — "{transaction_id}_c{index:03d}"
        source_transaction_id
        chunk_text
        chunk_index          — 0-based position within the source narrative
        total_chunks         — number of chunks this narrative produced
        metadata             — all original metadata fields, plus chunking params
    """
    chunk_size = splitter._chunk_size
    chunk_overlap = splitter._chunk_overlap

    with tqdm(total=total, unit="narratives", desc="Chunking", ncols=90) as pbar:
        for record in records:
            tx_id = record.get("transaction_id", "UNKNOWN")
            narrative = record.get("narrative", "")
            metadata = record.get("metadata", {})

            if not narrative:
                log.warning("Empty narrative for %s — skipping", tx_id)
                pbar.update(1)
                continue

            try:
                splits = splitter.split_text(narrative)
            except Exception as exc:
                log.warning("Splitter failed for %s — %s", tx_id, exc)
                pbar.update(1)
                continue

            n_splits = len(splits)
            for idx, text in enumerate(splits):
                yield {
                    "chunk_id": f"{tx_id}_c{idx:03d}",
                    "source_transaction_id": tx_id,
                    "chunk_text": text,
                    "chunk_index": idx,
                    "total_chunks": n_splits,
                    "metadata": {
                        **metadata,
                        # Embed chunking config so each chunk is self-describing
                        # — critical for benchmarking experiments with varying settings
                        "chunk_size_setting": chunk_size,
                        "chunk_overlap_setting": chunk_overlap,
                    },
                }

            pbar.update(1)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(total_chars: int) -> int:
    """Return a rough token count using the 4-chars-per-token heuristic."""
    return total_chars // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    log.info("=== chunk_narratives.py started ===")
    log.info("chunk_size=%d  chunk_overlap=%d", chunk_size, chunk_overlap)

    # -- Validate input ------------------------------------------------------
    if not INPUT_JSONL.exists():
        log.error("Input file not found: %s", INPUT_JSONL)
        log.error("Run ingestion/narrative_generation/build_narratives.py first.")
        sys.exit(1)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    # -- Count source records for the progress bar ---------------------------
    log.info("Counting source narratives …")
    total_narratives = count_lines(INPUT_JSONL)
    log.info("Source narratives: %d", total_narratives)

    # -- Set up splitter -----------------------------------------------------
    splitter = build_splitter(chunk_size, chunk_overlap)

    # -- Stream, chunk, write ------------------------------------------------
    records = load_narratives(INPUT_JSONL)
    chunks_gen = chunk_documents(records, splitter, total=total_narratives)

    total_chunks = 0
    total_chars = 0
    write_buffer: list[dict] = []

    log.info("Writing chunks to: %s", OUTPUT_JSONL)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_fh, \
         tqdm(unit="chunks", desc="Writing ", ncols=90, position=1, leave=True) as write_pbar:

        for chunk in chunks_gen:
            total_chars += len(chunk["chunk_text"])
            write_buffer.append(chunk)
            total_chunks += 1

            if len(write_buffer) >= WRITE_BUFFER_SIZE:
                save_chunks(write_buffer, out_fh)
                write_pbar.update(len(write_buffer))
                write_buffer.clear()

        # Flush remaining records
        if write_buffer:
            save_chunks(write_buffer, out_fh)
            write_pbar.update(len(write_buffer))
            write_buffer.clear()

    # -- Summary stats -------------------------------------------------------
    avg_chunk_size = total_chars / total_chunks if total_chunks else 0
    estimated_tokens = estimate_tokens(total_chars)
    expansion_ratio = total_chunks / total_narratives if total_narratives else 0

    log.info("=== Run complete ===")
    log.info("Total chunks generated     : %d", total_chunks)
    log.info("Average chunk size         : %.1f chars", avg_chunk_size)
    log.info("Estimated total tokens     : %d", estimated_tokens)
    log.info("Chunk expansion ratio      : %.2fx", expansion_ratio)
    log.info("Output file                : %s", OUTPUT_JSONL)

    print("\n" + "=" * 60)
    print(f"  Chunks generated     : {total_chunks:,}")
    print(f"  Avg chunk size       : {avg_chunk_size:.1f} chars")
    print(f"  Estimated tokens     : {estimated_tokens:,}")
    print(f"  Chunk/narrative ratio: {expansion_ratio:.2f}x")
    print(f"  Output               : {OUTPUT_JSONL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
