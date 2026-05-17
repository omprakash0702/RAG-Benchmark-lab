"""
generate_answer.py — Basic RAG Answer Generation Pipeline

Full RAG pipeline: semantic retrieval -> context construction -> LLM generation.
Designed as the Basic RAG leg of the benchmark:

    LLM-only  vs  Basic RAG (this file)  vs  GraphRAG

--- RAG architecture ---
Retrieval-Augmented Generation grounds the LLM in real data, reducing
hallucinations by forcing the model to reason only from retrieved evidence.
The pipeline has three stages:

  1. RETRIEVE   — encode the query, ANN-search ChromaDB, get top-k chunks
  2. AUGMENT    — format chunks into a structured context block
  3. GENERATE   — inject context + query into a prompt, call the LLM

--- Why retrieval reduces hallucinations ---
LLMs are trained to produce plausible text, not accurate text.  Without
grounding, a model asked about specific transactions will fabricate account
numbers, amounts, and patterns that sound correct but are not.  RAG pins the
model to a verifiable evidence set: any claim not supported by the context
is explicitly out of scope.

--- Token efficiency tradeoff ---
Every retrieved chunk costs prompt tokens.  More chunks = better recall but
higher cost and latency.  This file tracks estimated and actual token counts
so the benchmark phase can compute cost/quality curves across top_k values
and compare them against GraphRAG (which uses fewer but richer context nodes).

--- Latency tradeoff ---
Total latency = embed_ms + search_ms + llm_ms.
  - embed_ms / search_ms are retrieval costs, benchmarked in retrieve.py
  - llm_ms is network + generation cost, measured here
Separating these lets the evaluation phase attribute improvements to the
right stage when tuning the pipeline.

--- Benchmark reliability guarantee ---
The RAG_METRICS line is emitted inside a try/finally block so it prints
even when any stage of the pipeline fails.  Failed runs carry status=failure
and error_type so the benchmark harness records them as data.
"""

import logging
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import argparse

# Extracts entity name from the start of a transaction narrative, e.g.:
# "Sole Proprietorship #6323 (Account 80A4D6EB0) at ..." → "Sole Proprietorship #6323"
_ENTITY_RE = re.compile(r"^([\w\s]+#[\w\d]+)\s*\(Account", re.IGNORECASE)


def _extract_entity(narrative: str) -> str:
    m = _ENTITY_RE.match(narrative.strip())
    return m.group(1).strip() if m else ""

warnings.filterwarnings("ignore", category=FutureWarning, module="google")

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Reuse retrieval layer from retrieve.py (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from retrieve import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_MODEL,
    load_chroma_collection,
    load_embedding_model,
    retrieve_documents,
    estimate_tokens,
)

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

GEMINI_MODEL    = "gemini-2.5-flash"
OPENAI_MODEL    = "gpt-4o-mini"
OLLAMA_MODEL    = "llama3.2"
DEFAULT_BACKEND = "gemini"

TOP_K               = 5
MAX_CONTEXT_CHUNKS  = 5
MAX_CONTEXT_CHARS   = 12_000
CHARS_PER_TOKEN     = 4

MAX_RETRIES    = 3
BASE_BACKOFF_S = 2.0
_RETRYABLE_CODES = {429, 500, 503}

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
        description="Basic RAG — retrieval-augmented financial risk analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_answer.py --query "Possible laundering through repeated wire transfers"
  python generate_answer.py --query "Circular transactions between banks" --top_k 10
  python generate_answer.py --query "High-value cross-currency transfers" --model ollama
        """,
    )
    p.add_argument("--query", type=str, default=None)
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--max_context_chunks", type=int, default=MAX_CONTEXT_CHUNKS)
    p.add_argument("--model", type=str, default=DEFAULT_BACKEND,
                   choices=["gemini", "openai", "ollama"],
                   help="LLM backend. Gemini auto-falls back to OpenAI then Ollama on quota exhaustion.")
    p.add_argument("--gemini_model", type=str, default=GEMINI_MODEL)
    p.add_argument("--openai_model", type=str, default=OPENAI_MODEL)
    p.add_argument("--ollama_model", type=str, default=OLLAMA_MODEL)
    p.add_argument("--embedding_model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--collection", type=str, default=COLLECTION_NAME)
    p.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Context & prompt construction
# ---------------------------------------------------------------------------

def build_context(
    docs: list[str],
    metas: list[dict],
    dists: list[float],
    max_chunks: int,
    max_chars: int,
) -> tuple[str, int, bool]:
    blocks: list[str] = []
    total_chars = 0

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        if i >= max_chunks:
            break
        sim = round(1.0 - dist, 4)
        sender_entity = _extract_entity(doc)
        block = (
            f"[Transaction {i + 1}]  (similarity: {sim})\n"
            f"  Sender        : {meta.get('sender', 'N/A')}"
                + (f"  [{sender_entity}]" if sender_entity else "") + "\n"
            f"  Receiver      : {meta.get('receiver', 'N/A')}\n"
            f"  From bank     : {meta.get('from_bank', 'N/A')}\n"
            f"  To bank       : {meta.get('to_bank', 'N/A')}\n"
            f"  Amount paid   : {meta.get('amount_paid', 'N/A')} "
                f"{meta.get('payment_currency', '')}\n"
            f"  Amount rcvd   : {meta.get('amount_received', 'N/A')} "
                f"{meta.get('receiving_currency', '')}\n"
            f"  Payment type  : {meta.get('payment_format', 'N/A')}\n"
            f"  Timestamp     : {meta.get('timestamp', 'N/A')}\n"
            f"  Laundering    : {'YES' if meta.get('is_laundering') == '1' else 'no'}\n"
            f"  Narrative     : {doc}\n"
        )
        if total_chars + len(block) > max_chars:
            log.warning(
                "Context ceiling reached at chunk %d — truncating.", i + 1
            )
            return "\n".join(blocks), len(blocks), True
        blocks.append(block)
        total_chars += len(block)

    return "\n".join(blocks), len(blocks), False


DATASET_DATE_NOTE = (
    "DATASET NOTE: This transaction database covers the period 2022-09-01 to 2022-09-30. "
    "Temporal references such as 'last 3 days', 'recent', or 'today' should be "
    "interpreted relative to the dataset's end date (2022-09-30)."
)

# Maps lowercase query keywords to ChromaDB payment_format values
_FORMAT_KEYWORDS: dict[str, str] = {
    "cash":         "Cash",
    "wire":         "Wire",
    "ach":          "ACH",
    "cheque":       "Cheque",
    "check":        "Cheque",
    "credit card":  "Credit Card",
    "reinvestment": "Reinvestment",
}


def detect_payment_format(query: str) -> str | None:
    """Return a ChromaDB payment_format value if the query mentions a specific format."""
    q = query.lower()
    for keyword, fmt in _FORMAT_KEYWORDS.items():
        if keyword in q:
            return fmt
    return None


def build_prompt(query: str, context: str) -> str:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from query_intent import classify_query
    qtype = classify_query(query)

    if qtype == "factual":
        instructions = """INSTRUCTIONS:
- Answer the query DIRECTLY using only data explicitly shown in the evidence fields below.
- If asked for a ranking or top-N list, extract and rank only values that appear verbatim in the evidence.
- Present results as a numbered list or table.
- If the evidence does not contain the data needed to answer the query, open with:
  "EVIDENCE GAP: The retrieved records do not contain [specific missing info]. Showing closest matches only."
- Never fabricate values not present in the evidence."""
        response_header = "DIRECT ANSWER:"
    else:
        instructions = """ANALYSIS INSTRUCTIONS:
1. Describe what the retrieved transactions show: amounts, payment formats, sender/receiver accounts,
   laundering flags, cross-bank and cross-currency indicators. Use exact field values.
2. Identify any laundering indicators directly evidenced: flagged transactions, cross-currency,
   multi-bank routing, self-transfers, repeated sender/receiver pairs.
3. Do NOT label an account as a specific entity type unless the Narrative field explicitly states it.
4. State findings as "in the retrieved sample" — never generalise to the full dataset.
5. Assess financial risk: LOW / MEDIUM / HIGH — justify with at least 2 specific fields from evidence.
6. End with a SAMPLE LIMITATION note: state what aspect of the query the 5 records could not fully
   address, and what additional data would be needed. Keep this to 1–2 sentences."""
        response_header = "RISK ANALYSIS REPORT:"

    return f"""You are a financial analyst with expertise in AML and transaction data.
You have been given exactly 5 retrieved bank transaction records as evidence.

{DATASET_DATE_NOTE}

STRICT RULES — VIOLATIONS INVALIDATE THE ANALYSIS:
1. Only cite account IDs, amounts, dates, and banks that appear verbatim in the evidence below.
2. Never infer or assume entity type (sole proprietorship, corporation, partnership) unless
   the Narrative field explicitly names it for that account.
3. Never claim an account "matches" the query pattern unless the evidence fields directly support it.
4. These 5 records are a semantic sample, NOT a complete dataset scan. Never make dataset-wide claims.
5. ALWAYS analyze what IS present in the evidence first — describe the actual transactions, amounts,
   accounts, and flags you can see. If the evidence only partially covers the queried pattern,
   note the SPECIFIC GAP at the END of your analysis, not at the beginning.
6. Never refuse to analyze — even partial evidence produces a useful risk assessment.

{instructions}

--- RETRIEVED TRANSACTION EVIDENCE (5 records) ---
{context}
--- END OF EVIDENCE ---

ANALYST QUERY: {query}

{response_header}"""


# ---------------------------------------------------------------------------
# LLM generation with retry
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc)
    return any(f"{code}" in msg for code in _RETRYABLE_CODES)


def _is_quota_exhausted(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ("429", "resource_exhausted", "quota", "exhausted"))


def generate_with_gemini(
    prompt: str,
    api_key: str,
    model_name: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, float, int, int]:
    try:
        import google.genai as genai
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            latency_ms    = (time.perf_counter() - t0) * 1000
            answer        = response.text or ""
            usage         = getattr(response, "usage_metadata", None)
            input_tokens  = getattr(usage, "prompt_token_count",     0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            return answer, latency_ms, input_tokens, output_tokens

        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == max_retries:
                break
            delay = BASE_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(
                "Gemini transient error (attempt %d/%d) — retrying in %.0fs: %s",
                attempt, max_retries, delay, str(exc)[:120],
            )
            time.sleep(delay)

    raise RuntimeError(f"Gemini API failed after {max_retries} attempts: {last_exc}")


def generate_with_openai(
    prompt: str,
    api_key: str,
    model_name: str = OPENAI_MODEL,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, float, int, int]:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms    = (time.perf_counter() - t0) * 1000
            answer        = response.choices[0].message.content or ""
            input_tokens  = response.usage.prompt_tokens     if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            return answer, latency_ms, input_tokens, output_tokens
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == max_retries:
                break
            delay = BASE_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(
                "OpenAI transient error (attempt %d/%d) — retrying in %.0fs: %s",
                attempt, max_retries, delay, str(exc)[:120],
            )
            time.sleep(delay)
    raise RuntimeError(f"OpenAI API failed after {max_retries} attempts: {last_exc}")


def generate_with_ollama(
    prompt: str,
    model_name: str,
) -> tuple[str, float, int, int]:
    try:
        import ollama
    except ImportError:
        raise RuntimeError("ollama not installed. Run: pip install ollama")

    t0 = time.perf_counter()
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama call failed: {exc}. Ensure ollama serve is running."
        ) from exc

    latency_ms    = (time.perf_counter() - t0) * 1000
    answer        = response.message.content or ""
    input_tokens  = getattr(response, "prompt_eval_count", 0) or 0
    output_tokens = getattr(response, "eval_count",        0) or 0
    return answer, latency_ms, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def emit_metrics(
    backend: str,
    model_name: str,
    n_retrieved: int,
    n_context: int,
    embed_ms: float,
    search_ms: float,
    llm_ms: float,
    est_tokens: int,
    actual_in: int,
    actual_out: int,
    status: str,
    error_type: str,
) -> None:
    """
    Emit the machine-readable benchmark metrics line.
    Lives in its own function so the finally block in main() stays readable.
    Always followed by flush to guarantee delivery before process exit.
    """
    total_ms = embed_ms + search_ms + llm_ms
    log.info(
        "RAG_METRICS | backend=%s | model=%s | retrieved=%d | context=%d "
        "| embed_ms=%.1f | search_ms=%.1f | llm_ms=%.1f | total_ms=%.1f "
        "| est_tokens=%d | actual_in=%d | actual_out=%d "
        "| status=%s | error_type=%s",
        backend, model_name, n_retrieved, n_context,
        embed_ms, search_ms, llm_ms, total_ms,
        est_tokens, actual_in, actual_out,
        status, error_type or "none",
    )
    sys.stdout.flush()
    sys.stderr.flush()


def print_results(
    query: str,
    backend: str,
    model_name: str,
    n_retrieved: int,
    n_context: int,
    was_truncated: bool,
    est_context_tokens: int,
    actual_input_tokens: int,
    actual_output_tokens: int,
    embed_ms: float,
    search_ms: float,
    llm_ms: float,
    answer: str,
) -> None:
    total_ms = embed_ms + search_ms + llm_ms
    width = 70
    print("\n" + "=" * width)
    print("  BASIC RAG — GENERATION RESULTS")
    print("=" * width)
    print(f"  Query         : {query}")
    print(f"  Backend       : {backend} ({model_name})")
    print(f"  Retrieved     : {n_retrieved} chunks from ChromaDB")
    print(f"  Context used  : {n_context} chunks"
          + ("  [TRUNCATED]" if was_truncated else ""))
    print("-" * width)
    print("  LATENCY")
    print(f"    Embed       : {embed_ms:>8.2f} ms")
    print(f"    Search      : {search_ms:>8.2f} ms")
    print(f"    LLM         : {llm_ms:>8.2f} ms")
    print(f"    Total       : {total_ms:>8.2f} ms")
    print("-" * width)
    print("  TOKENS")
    print(f"    Est. context: {est_context_tokens:>8,}  (chars/4 heuristic)")
    if actual_input_tokens:
        print(f"    Actual input: {actual_input_tokens:>8,}  (from API)")
    if actual_output_tokens:
        print(f"    Actual output:{actual_output_tokens:>7,}  (from API)")
    print("=" * width)
    print("\n  GENERATED ANSWER\n")
    print(answer.strip())
    print("\n" + "=" * width)


# ---------------------------------------------------------------------------
# Main — with guaranteed metrics emission in finally
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Failure-default state — populated as each pipeline stage succeeds.
    # The finally block emits whatever state was reached when the failure occurred,
    # giving the benchmark harness accurate data about *where* the pipeline broke.
    _backend    = args.model
    _model_name = (args.gemini_model  if args.model == "gemini"
                   else args.openai_model if args.model == "openai"
                   else args.ollama_model)
    _retrieved  = 0
    _n_context  = 0
    _embed_ms   = 0.0
    _search_ms  = 0.0
    _llm_ms     = 0.0
    _est_tokens = 0
    _actual_in  = 0
    _actual_out = 0
    _status     = "failure"
    _error_type = ""

    try:
        load_dotenv(ENV_FILE)
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if args.model == "gemini" and not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        if args.model == "openai" and not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")

        query = args.query
        if not query:
            try:
                print("\nBasic RAG — Financial Risk Analysis")
                print("Enter a query (Ctrl-C to exit):\n")
                query = input("Query > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                return

        if not query:
            raise ValueError("Empty query — nothing to process")

        log.info("=== Basic RAG pipeline started ===")
        log.info("Query   : %s", query)
        log.info("Backend : %s", args.model)

        # -- Retrieval stage -------------------------------------------------
        embed_model = load_embedding_model(args.embedding_model)
        collection  = load_chroma_collection(CHROMA_DIR, args.collection)

        fmt_filter = detect_payment_format(query)
        where_filter = {"payment_format": fmt_filter} if fmt_filter else None
        if where_filter:
            log.info("Payment format filter detected: %s", fmt_filter)

        log.info("Retrieving top-%d chunks ...", args.top_k)
        payload = retrieve_documents(query, embed_model, collection, args.top_k,
                                     where_filter=where_filter)
        _embed_ms  = payload["embed_ms"]
        _search_ms = payload["query_ms"]

        docs  = payload["results"]["documents"][0]
        metas = payload["results"]["metadatas"][0]
        dists = payload["results"]["distances"][0]
        _retrieved = len(docs)

        # If format-filtered search returned nothing, fall back to unfiltered
        if not docs and where_filter:
            log.warning("Format filter '%s' returned 0 results — falling back to unfiltered search", fmt_filter)
            payload = retrieve_documents(query, embed_model, collection, args.top_k)
            _embed_ms  += payload["embed_ms"]
            _search_ms += payload["query_ms"]
            docs  = payload["results"]["documents"][0]
            metas = payload["results"]["metadatas"][0]
            dists = payload["results"]["distances"][0]
            _retrieved = len(docs)

        if not docs:
            raise RuntimeError("Retrieval returned no results — ChromaDB may be empty")

        # -- Context construction stage --------------------------------------
        context, _n_context, was_truncated = build_context(
            docs, metas, dists,
            max_chunks=args.max_context_chunks,
            max_chars=MAX_CONTEXT_CHARS,
        )
        _est_tokens = estimate_tokens([context])
        prompt = build_prompt(query, context)
        log.info("Prompt built — context: %d chunks | est. tokens: %d",
                 _n_context, _est_tokens)

        # -- Generation stage ------------------------------------------------
        log.info("Calling %s (%s) ...", _backend, _model_name)
        if args.model == "gemini":
            try:
                answer, _llm_ms, _actual_in, _actual_out = generate_with_gemini(
                    prompt, gemini_api_key, _model_name, args.max_retries,
                )
            except RuntimeError as _gem_exc:
                if not _is_quota_exhausted(_gem_exc):
                    raise
                log.warning("Gemini quota exhausted — falling back to OpenAI")
                if openai_api_key:
                    try:
                        answer, _llm_ms, _actual_in, _actual_out = generate_with_openai(
                            prompt, openai_api_key, args.openai_model, args.max_retries,
                        )
                        _backend    = "openai"
                        _model_name = args.openai_model
                    except RuntimeError as _oa_exc:
                        log.warning("OpenAI failed — falling back to Ollama: %s", _oa_exc)
                        answer, _llm_ms, _actual_in, _actual_out = generate_with_ollama(
                            prompt, args.ollama_model,
                        )
                        _backend    = "ollama"
                        _model_name = args.ollama_model
                else:
                    log.warning("OPENAI_API_KEY not set — falling back to Ollama")
                    answer, _llm_ms, _actual_in, _actual_out = generate_with_ollama(
                        prompt, args.ollama_model,
                    )
                    _backend    = "ollama"
                    _model_name = args.ollama_model
        elif args.model == "openai":
            answer, _llm_ms, _actual_in, _actual_out = generate_with_openai(
                prompt, openai_api_key, _model_name, args.max_retries,
            )
        else:  # ollama
            answer, _llm_ms, _actual_in, _actual_out = generate_with_ollama(
                prompt, _model_name,
            )

        if not answer:
            raise RuntimeError("LLM returned an empty response")

        _status = "success"
        print_results(
            query=query,
            backend=_backend,
            model_name=_model_name,
            n_retrieved=_retrieved,
            n_context=_n_context,
            was_truncated=was_truncated,
            est_context_tokens=_est_tokens,
            actual_input_tokens=_actual_in,
            actual_output_tokens=_actual_out,
            embed_ms=_embed_ms,
            search_ms=_search_ms,
            llm_ms=_llm_ms,
            answer=answer,
        )

    except Exception as exc:
        _error_type = type(exc).__name__
        log.error("Pipeline error [%s]: %s", _error_type, exc)
        log.debug(traceback.format_exc())

    finally:
        emit_metrics(
            _backend, _model_name,
            _retrieved, _n_context,
            _embed_ms, _search_ms, _llm_ms,
            _est_tokens, _actual_in, _actual_out,
            _status, _error_type,
        )


if __name__ == "__main__":
    main()
