"""
retrieve.py — Basic RAG Semantic Retrieval Pipeline

Performs vector similarity search against the ChromaDB collection of
financial transaction embeddings.  Designed as the retrieval stage of a
Basic RAG benchmark that will be compared against:
  - LLM-only (no retrieval)
  - GraphRAG  (graph-augmented retrieval)

--- How semantic retrieval works ---
The query is encoded into a dense vector using the same embedding model
that was used at ingestion time (all-MiniLM-L6-v2).  ChromaDB then runs an
approximate nearest-neighbour (ANN) search over the HNSW index, returning
the k vectors whose cosine distance to the query vector is smallest.
Because both query and document vectors are L2-normalised, cosine distance
and dot-product distance are equivalent — smaller distance == higher similarity.

--- Why retrieval latency matters for benchmarking ---
In a production RAG system, retrieval latency is the dominant per-query cost
(LLM generation is slower in absolute terms but runs in parallel across
requests).  Measuring retrieval latency here lets us compare:
  - Basic RAG vector search  (this file)
  - GraphRAG graph traversal  (pipelines/graph_rag/)
...under identical query sets to isolate the retrieval overhead.

--- Token efficiency ---
Every retrieved chunk is prepended to the LLM prompt.  Returning more chunks
(higher top_k) improves recall but increases prompt token cost and latency.
Logging estimated token counts here feeds directly into the cost/quality
trade-off analysis in the benchmark evaluation phase.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parents[2]
CHROMA_DIR      = ROOT / "database" / "chroma"
COLLECTION_NAME = "financial_transactions"
DEFAULT_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5                   # default number of results to retrieve
CHARS_PER_TOKEN = 4         # GPT-style token estimate
PREVIEW_LEN     = 200       # characters shown in the chunk text preview

# ---------------------------------------------------------------------------
# Logging — INFO for benchmark runs, DEBUG for pipeline development
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
        description="Basic RAG — semantic retrieval from ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python retrieve.py --query "Suspicious money laundering transfers between banks"
  python retrieve.py --query "large wire transfers across currencies" --top_k 10
  python retrieve.py                          # launches interactive prompt
        """,
    )
    p.add_argument(
        "--query", type=str, default=None,
        help="Natural language query string.  Omit to enter interactively.",
    )
    p.add_argument(
        "--top_k", type=int, default=TOP_K,
        help=f"Number of results to retrieve (default: {TOP_K}).",
    )
    p.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="SentenceTransformer model for query encoding (must match ingestion model).",
    )
    p.add_argument(
        "--collection", type=str, default=COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {COLLECTION_NAME}).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model & database loaders
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load the embedding model used at ingestion time.
    Using the same model for queries is non-negotiable: a different model
    produces vectors in a different semantic space, making distance scores
    meaningless.
    """
    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    log.info("Model ready — dim: %d", model.get_embedding_dimension())
    return model


def load_chroma_collection(
    db_path: Path,
    collection_name: str,
) -> chromadb.Collection:
    """
    Open the persistent ChromaDB client and return the named collection.
    embedding_function=None because we supply pre-computed query vectors —
    ChromaDB's internal ONNX embedder is never invoked.
    """
    if not db_path.exists():
        log.error("ChromaDB directory not found: %s", db_path)
        log.error("Run ingestion/embeddings/build_embeddings.py first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except Exception:
        log.error(
            "Collection '%s' not found in %s.  Available: %s",
            collection_name,
            db_path,
            [c.name for c in client.list_collections()],
        )
        sys.exit(1)

    log.info(
        "Collection '%s' loaded — %d documents indexed",
        collection_name,
        collection.count(),
    )
    return collection


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_documents(
    query: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    top_k: int,
    where_filter: dict | None = None,
) -> dict:
    """
    Encode the query and perform ANN search against the ChromaDB HNSW index.

    Returns a dict with:
        results   — raw ChromaDB query response
        embed_ms  — query encoding time in milliseconds
        query_ms  — ChromaDB ANN search time in milliseconds
        total_ms  — total retrieval wall-clock time in milliseconds

    Timing is split into embed vs. search so benchmarks can attribute latency
    correctly (embed time is model-dependent; search time is index-dependent).
    """
    # -- Encode query --------------------------------------------------------
    t0 = time.perf_counter()
    query_vec: np.ndarray = model.encode(
        [query],
        normalize_embeddings=True,   # must match ingestion normalisation
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embed_ms = (time.perf_counter() - t0) * 1000

    # -- Vector similarity search -------------------------------------------
    t1 = time.perf_counter()
    query_kwargs: dict = dict(
        query_embeddings=query_vec.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where_filter:
        query_kwargs["where"] = where_filter
    results = collection.query(**query_kwargs)
    query_ms = (time.perf_counter() - t1) * 1000
    total_ms = embed_ms + query_ms

    return {
        "results":  results,
        "embed_ms": embed_ms,
        "query_ms": query_ms,
        "total_ms": total_ms,
        "where_filter": where_filter,
    }


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(texts: list[str]) -> int:
    """
    Rough token count for the retrieved context window.
    Used to estimate downstream LLM prompt cost before generation happens.
    4 chars ≈ 1 token is the standard GPT-family heuristic.
    """
    return sum(len(t) for t in texts) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(query: str, payload: dict, top_k: int) -> None:
    """
    Print a benchmark-style retrieval report.

    Layout is structured so that future benchmark scripts can parse it with
    simple string matching (rank lines, score lines, latency lines).
    """
    results  = payload["results"]
    ids      = results["ids"][0]
    docs     = results["documents"][0]
    metas    = results["metadatas"][0]
    dists    = results["distances"][0]

    # ChromaDB cosine space returns cosine distance ∈ [0, 2].
    # Similarity = 1 − distance makes intuitive sense to humans (1.0 = perfect).
    similarities = [round(1.0 - d, 4) for d in dists]

    retrieved_texts = docs
    token_estimate  = estimate_tokens(retrieved_texts)

    # -------------------------------------------------------------------------
    width = 70
    print("\n" + "=" * width)
    print("  BASIC RAG — RETRIEVAL RESULTS")
    print("=" * width)
    print(f"  Query      : {query}")
    print(f"  Top-K      : {top_k}")
    print(f"  Retrieved  : {len(ids)} chunks")
    print("-" * width)
    print(f"  Embed time : {payload['embed_ms']:>7.2f} ms")
    print(f"  Search time: {payload['query_ms']:>7.2f} ms")
    print(f"  Total time : {payload['total_ms']:>7.2f} ms")
    print(f"  Est. tokens: {token_estimate:>7,}  (retrieved context)")
    print("=" * width)

    for rank, (chunk_id, doc, meta, sim) in enumerate(
        zip(ids, docs, metas, similarities), start=1
    ):
        preview = doc[:PREVIEW_LEN].replace("\n", " ")
        if len(doc) > PREVIEW_LEN:
            preview += " …"

        print(f"\n  [{rank}] Similarity: {sim:.4f}  |  chunk_id: {chunk_id}")
        print(f"      Sender : {meta.get('sender', 'N/A')}"
              f"  ->  Receiver: {meta.get('receiver', 'N/A')}")
        print(f"      Amount : {meta.get('amount_paid', 'N/A')} {meta.get('payment_currency', '')}"
              f"  via {meta.get('payment_format', 'N/A')}")
        print(f"      Date   : {meta.get('timestamp', 'N/A')}"
              f"  |  Laundering flag: {meta.get('is_laundering', 'N/A')}")
        print(f"      Text   : {preview}")

    print("\n" + "=" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -- Resolve query -------------------------------------------------------
    query = args.query
    if not query:
        try:
            print("\nBasic RAG — Interactive Retrieval")
            print("Enter a natural language query (Ctrl-C to exit):\n")
            query = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not query:
        log.error("Empty query — nothing to retrieve.")
        sys.exit(1)

    # -- Load resources ------------------------------------------------------
    model      = load_embedding_model(args.model)
    collection = load_chroma_collection(CHROMA_DIR, args.collection)

    # -- Retrieve ------------------------------------------------------------
    log.info("Running retrieval for: %r", query)
    payload = retrieve_documents(query, model, collection, args.top_k)

    # -- Display -------------------------------------------------------------
    print_results(query, payload, args.top_k)

    # Structured log line — easy to grep in batch benchmark runs
    log.info(
        "RETRIEVAL_DONE | top_k=%d | embed_ms=%.2f | query_ms=%.2f | total_ms=%.2f | tokens=%d",
        args.top_k,
        payload["embed_ms"],
        payload["query_ms"],
        payload["total_ms"],
        estimate_tokens(payload["results"]["documents"][0]),
    )


if __name__ == "__main__":
    main()
