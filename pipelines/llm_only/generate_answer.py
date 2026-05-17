"""
generate_answer.py — LLM-Only Baseline Pipeline

Pure language-model inference with NO retrieval augmentation.
Designed as the baseline leg of the benchmark:

    LLM-only (this file)  vs  Basic RAG  vs  GraphRAG

--- Why LLM-only is the baseline ---
Without retrieval, the model must answer entirely from parametric knowledge
baked in during pre-training.  For domain-specific datasets like the HI-Small
AML corpus, that knowledge is essentially zero: the model has never seen these
account numbers, entity names, or transaction records.  The baseline reveals
the answer quality floor — how well a frontier LLM can reason about financial
crime patterns from general knowledge alone, without access to any evidence.

--- Hallucination risk without retrieval ---
LLMs are trained to produce fluent, plausible text.  When asked about specific
transactions they have never seen, they will fabricate account IDs, amounts,
and relationship patterns with high confidence.  This is not a bug; it is the
expected behavior of a model optimized for coherence, not factual accuracy.
The benchmark quantifies this gap: a significant answer-quality improvement
from LLM-only -> Basic RAG -> GraphRAG demonstrates the value of retrieval.

--- Token "efficiency" without retrieval grounding ---
LLM-only prompts are shorter (no injected context), so input token cost and
latency are lower.  But the outputs are less accurate, so the per-useful-token
cost is actually much higher.  Tracking tokens here lets the evaluation phase
compute cost-per-correct-answer across all three pipeline modes.

--- Benchmark reliability guarantee ---
The LLM_ONLY_METRICS line is emitted inside a try/finally block so it prints
even when the pipeline fails.  Failed runs carry status=failure and error_type
so the benchmark harness records them as data rather than losing the run.
Failures are benchmark data: they reveal reliability characteristics of the
pipeline that a "hide all errors" approach would mask.
"""

import argparse
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="google")

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

DEFAULT_BACKEND     = "gemini"
GEMINI_MODEL        = "gemini-2.5-flash"
OPENAI_MODEL        = "gpt-4o-mini"
OLLAMA_MODEL        = "llama3.2"
DEFAULT_MAX_TOKENS  = 2048
DEFAULT_TEMPERATURE = 1.0
CHARS_PER_TOKEN     = 4

# Retry config for transient API errors (429 rate-limit, 503 unavailable)
MAX_RETRIES     = 3
BASE_BACKOFF_S  = 2.0   # doubles each attempt: 2s, 4s, 8s

# HTTP status codes that are retryable
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
        description="LLM-only baseline — financial risk analysis without retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_answer.py --query "Possible money laundering through repeated transfers"
  python generate_answer.py --query "Structuring across multiple bank accounts" --temperature 0.0
  python generate_answer.py --query "High-value cross-currency wire transfers" --max_tokens 512
        """,
    )
    p.add_argument("--query", type=str, default=None,
                   help="Financial risk query. Omit for interactive input.")
    p.add_argument("--model", type=str, default=DEFAULT_BACKEND,
                   choices=["gemini", "openai", "ollama"],
                   help=f"LLM backend (default: {DEFAULT_BACKEND}). Gemini auto-falls back to OpenAI then Ollama on quota exhaustion.")
    p.add_argument("--gemini_model", type=str, default=GEMINI_MODEL,
                   help=f"Gemini model name (default: {GEMINI_MODEL}).")
    p.add_argument("--openai_model", type=str, default=OPENAI_MODEL,
                   help=f"OpenAI model name (default: {OPENAI_MODEL}).")
    p.add_argument("--ollama_model", type=str, default=OLLAMA_MODEL,
                   help=f"Ollama model name (default: {OLLAMA_MODEL}).")
    p.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"Max output tokens for Gemini (default: {DEFAULT_MAX_TOKENS}).")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help=f"Sampling temperature 0.0-2.0 (default: {DEFAULT_TEMPERATURE}).")
    p.add_argument("--max_retries", type=int, default=MAX_RETRIES,
                   help=f"Max retries on transient API errors (default: {MAX_RETRIES}).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

DATASET_DATE_NOTE = (
    "DATASET CONTEXT: The transaction database covers 2022-09-01 to 2022-09-30. "
    "Interpret temporal references ('last 3 days', 'recent', 'today') "
    "relative to 2022-09-30 (the dataset end date)."
)


# ---------------------------------------------------------------------------
# Direct DB context — ChromaDB without embedding
# ---------------------------------------------------------------------------

CHROMA_DIR      = ROOT / "database" / "chroma"
COLLECTION_NAME = "financial_transactions"
DIRECT_SAMPLE   = 500    # records to pull for aggregation
DIRECT_SHOW     = 20     # individual transactions shown in prompt

_FORMAT_KEYWORDS: dict[str, str] = {
    "cash": "Cash", "wire": "Wire", "ach": "ACH",
    "cheque": "Cheque", "check": "Cheque",
    "credit card": "Credit Card", "reinvestment": "Reinvestment",
}


def _detect_format(query: str) -> str | None:
    q = query.lower()
    for kw, fmt in _FORMAT_KEYWORDS.items():
        if kw in q:
            return fmt
    return None


def fetch_direct_context(query: str) -> tuple[str, int]:
    """
    Pull real transaction records from ChromaDB without embedding.
    Returns (formatted_context, record_count).
    Falls back to empty string if ChromaDB is unavailable.
    """
    try:
        import chromadb
    except ImportError:
        return "", 0

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col    = client.get_collection(COLLECTION_NAME)
    except Exception:
        return "", 0

    fmt_filter = _detect_format(query)
    get_kwargs: dict = {"limit": DIRECT_SAMPLE, "include": ["documents", "metadatas"]}
    if fmt_filter:
        get_kwargs["where"] = {"payment_format": fmt_filter}

    try:
        result = col.get(**get_kwargs)
    except Exception:
        result = col.get(limit=DIRECT_SAMPLE, include=["documents", "metadatas"])
        fmt_filter = None

    docs  = result.get("documents") or []
    metas = result.get("metadatas") or []
    n     = len(docs)
    if n == 0:
        return "", 0

    # Python-side aggregation for ranking queries
    senders:   dict[str, int] = {}
    receivers: dict[str, int] = {}
    banks:     dict[str, int] = {}
    amounts:   list[float]    = []

    for m in metas:
        s = m.get("sender", "")
        r = m.get("receiver", "")
        b = m.get("from_bank", "")
        a = m.get("amount_paid", "")
        if s: senders[s]   = senders.get(s, 0)   + 1
        if r: receivers[r] = receivers.get(r, 0) + 1
        if b: banks[b]     = banks.get(b, 0)     + 1
        try: amounts.append(float(a))
        except (ValueError, TypeError): pass

    top_senders   = sorted(senders.items(),   key=lambda x: x[1], reverse=True)[:10]
    top_receivers = sorted(receivers.items(), key=lambda x: x[1], reverse=True)[:10]
    top_banks     = sorted(banks.items(),     key=lambda x: x[1], reverse=True)[:5]

    lines = [
        f"DATABASE SAMPLE: {n} records pulled directly from ChromaDB"
        + (f" (filter: payment_format={fmt_filter})" if fmt_filter else ""),
        f"Dataset period: 2022-09-01 to 2022-09-30",
        "",
        "AGGREGATED STATS (from this sample):",
        f"  Top accounts by SEND count:",
    ]
    for acct, cnt in top_senders:
        lines.append(f"    {acct}: {cnt} outgoing transactions")
    lines.append(f"  Top accounts by RECEIVE count:")
    for acct, cnt in top_receivers:
        lines.append(f"    {acct}: {cnt} incoming transactions")
    lines.append(f"  Top banks by transaction count:")
    for bk, cnt in top_banks:
        lines.append(f"    {bk}: {cnt} transactions")
    if amounts:
        lines.append(f"  Amount range: ${min(amounts):,.2f} — ${max(amounts):,.2f}")
        lines.append(f"  Average amount: ${sum(amounts)/len(amounts):,.2f}")

    lines.append("")
    lines.append(f"SAMPLE TRANSACTION RECORDS (first {min(DIRECT_SHOW, n)} of {n}):")
    for doc in docs[:DIRECT_SHOW]:
        lines.append(f"  {doc}")

    return "\n".join(lines), n


def build_prompt(query: str, db_context: str = "") -> str:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from query_intent import classify_query
    qtype = classify_query(query)

    has_data = bool(db_context.strip())

    if has_data:
        data_note = "You have been given REAL transaction records pulled directly from the database."
        constraints = """RULES:
- Answer ONLY from the data provided below — do not fabricate any values.
- If asked for a ranking, use the aggregated stats in the data.
- State that results are from a database sample (not the full 2M records) if relevant."""
        if qtype == "factual":
            instructions = """INSTRUCTIONS:
- Answer the query DIRECTLY using the aggregated stats and records below.
- For rankings/top-N: use the "Top accounts" stats in the data.
- Present results as a clear numbered list with real account IDs and counts."""
            response_header = "DIRECT ANSWER (from real database records):"
        else:
            instructions = """ANALYSIS INSTRUCTIONS:
1. Identify suspicious patterns in the actual records below.
2. Use real account IDs, amounts, and transaction details from the evidence.
3. Assess financial risk level: LOW / MEDIUM / HIGH — justify with specific records.
4. Recommend follow-up actions based on what the data shows."""
            response_header = "ANALYSIS REPORT (based on real database records):"
    else:
        data_note = "You have NO access to transaction data for this query."
        constraints = """RULES:
- Do NOT fabricate account IDs, amounts, or specific data.
- Clearly state you are reasoning from general knowledge only."""
        instructions = """INSTRUCTIONS:
1. Describe general AML patterns matching this query.
2. List key risk indicators to look for.
3. State that real data is needed for a specific answer."""
        response_header = "GENERAL GUIDANCE (no database access):"

    context_block = (
        f"\n--- REAL DATABASE RECORDS ---\n{db_context}\n--- END ---\n"
        if has_data else ""
    )

    return f"""You are a financial analyst specialising in AML and transaction data.
{DATASET_DATE_NOTE}

{data_note}

{constraints}

{instructions}
{context_block}
ANALYST QUERY: {query}

{response_header}"""


# ---------------------------------------------------------------------------
# Gemini call with exponential-backoff retry
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """Return True for transient HTTP errors worth retrying."""
    msg = str(exc)
    return any(f"{code}" in msg for code in _RETRYABLE_CODES)


def generate_with_gemini(
    prompt: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, float, int, int]:
    """
    Call Gemini with exponential-backoff retry on transient errors.
    Returns (answer, latency_ms, input_tokens, output_tokens).

    Retry covers:
      429 RESOURCE_EXHAUSTED — rate limit; back off and retry
      503 UNAVAILABLE        — transient server load; retry
      500 INTERNAL           — transient server error; retry once
    Does NOT retry 404 (wrong model) or 401/403 (auth failures).
    """
    try:
        import google.genai as genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            answer = response.text or ""
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


def _is_quota_exhausted(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ("429", "resource_exhausted", "quota", "exhausted"))


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
    model_name: str = OLLAMA_MODEL,
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
            f"Ollama call failed: {exc}. Ensure 'ollama serve' is running."
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
    llm_ms: float,
    input_tokens: int,
    output_tokens: int,
    status: str,
    error_type: str,
) -> None:
    """
    Emit the machine-readable benchmark metrics line.
    Called from the finally block in main() — guaranteed to execute.
    sys.stdout.flush() ensures the line reaches the benchmark harness
    even if the process exits immediately after.
    """
    log.info(
        "LLM_ONLY_METRICS | backend=%s | model=%s | retrieved=0 | context=0 "
        "| embed_ms=0.0 | search_ms=0.0 | llm_ms=%.1f | total_ms=%.1f "
        "| est_tokens=0 | actual_in=%d | actual_out=%d "
        "| status=%s | error_type=%s",
        backend, model_name, llm_ms, llm_ms, input_tokens, output_tokens,
        status, error_type or "none",
    )
    sys.stdout.flush()
    sys.stderr.flush()


def print_results(
    query: str,
    backend: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    llm_ms: float,
    answer: str,
) -> None:
    width = 70
    print("\n" + "=" * width)
    print("  LLM-ONLY — GENERATION RESULTS  (no retrieval)")
    print("=" * width)
    print(f"  Query         : {query}")
    print(f"  Backend       : {backend} ({model_name})")
    print(f"  Retrieval     : NONE  (pure parametric knowledge)")
    print("-" * width)
    print("  LATENCY")
    print(f"    Embed       :     0.00 ms  (no retrieval)")
    print(f"    Search      :     0.00 ms  (no retrieval)")
    print(f"    LLM         : {llm_ms:>8.2f} ms")
    print(f"    Total       : {llm_ms:>8.2f} ms")
    print("-" * width)
    print("  TOKENS")
    if input_tokens:
        print(f"    Input       : {input_tokens:>8,}  (from API)")
        print(f"    Output      : {output_tokens:>8,}  (from API)")
        print(f"    Total       : {input_tokens + output_tokens:>8,}")
    else:
        est = len(answer) // CHARS_PER_TOKEN
        print(f"    Est. output : {est:>8,}  (chars/4 heuristic)")
    print("=" * width)
    print("\n  GENERATED ANSWER\n")
    print(answer.strip())
    print("\n" + "=" * width)


# ---------------------------------------------------------------------------
# Main — with guaranteed metrics emission in finally
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    _backend    = args.model
    _model_name = (args.gemini_model  if args.model == "gemini"
                   else args.openai_model if args.model == "openai"
                   else args.ollama_model)
    _llm_ms     = 0.0
    _in_tok     = 0
    _out_tok    = 0
    _status     = "failure"
    _error_type = ""

    try:
        load_dotenv(ENV_FILE)
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if args.model == "gemini" and not gemini_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        if args.model == "openai" and not openai_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")

        query = args.query
        if not query:
            try:
                print("\nLLM-Only Baseline — Financial Risk Analysis (no retrieval)")
                print("Enter a query (Ctrl-C to exit):\n")
                query = input("Query > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                return

        if not query:
            raise ValueError("Empty query — nothing to process")

        log.info("=== LLM-only pipeline started ===")
        log.info("Query       : %s", query)
        log.info("Backend     : %s", _backend)
        log.info("Model       : %s", _model_name)
        log.info("Max tokens  : %d", args.max_tokens)
        log.info("Temperature : %.2f", args.temperature)

        # Fetch real data directly from ChromaDB (no embedding)
        t_db0 = time.perf_counter()
        db_context, db_records = fetch_direct_context(query)
        _db_ms = (time.perf_counter() - t_db0) * 1000
        if db_records:
            log.info("Direct DB   : %d records fetched in %.0f ms (no embedding)", db_records, _db_ms)
        else:
            log.info("Direct DB   : unavailable — proceeding without data")

        prompt = build_prompt(query, db_context)
        log.info("Calling %s (%s) ...", _backend, _model_name)

        if args.model == "gemini":
            try:
                answer, _llm_ms, _in_tok, _out_tok = generate_with_gemini(
                    prompt, gemini_key, _model_name,
                    args.max_tokens, args.temperature, args.max_retries,
                )
            except RuntimeError as _gem_exc:
                if not _is_quota_exhausted(_gem_exc):
                    raise
                log.warning("Gemini quota exhausted — falling back to OpenAI")
                if openai_key:
                    try:
                        answer, _llm_ms, _in_tok, _out_tok = generate_with_openai(
                            prompt, openai_key, args.openai_model, args.max_retries,
                        )
                        _backend    = "openai"
                        _model_name = args.openai_model
                    except RuntimeError as _oa_exc:
                        log.warning("OpenAI failed — falling back to Ollama: %s", _oa_exc)
                        answer, _llm_ms, _in_tok, _out_tok = generate_with_ollama(
                            prompt, args.ollama_model,
                        )
                        _backend    = "ollama"
                        _model_name = args.ollama_model
                else:
                    log.warning("OPENAI_API_KEY not set — falling back to Ollama")
                    answer, _llm_ms, _in_tok, _out_tok = generate_with_ollama(
                        prompt, args.ollama_model,
                    )
                    _backend    = "ollama"
                    _model_name = args.ollama_model
        elif args.model == "openai":
            answer, _llm_ms, _in_tok, _out_tok = generate_with_openai(
                prompt, openai_key, _model_name, args.max_retries,
            )
        else:  # ollama
            answer, _llm_ms, _in_tok, _out_tok = generate_with_ollama(
                prompt, _model_name,
            )

        if not answer:
            raise RuntimeError("LLM returned an empty response")

        _status = "success"
        print_results(query, _backend, _model_name, _in_tok, _out_tok, _llm_ms, answer)

    except Exception as exc:
        _error_type = type(exc).__name__
        log.error("Pipeline error [%s]: %s", _error_type, exc)
        log.debug(traceback.format_exc())

    finally:
        emit_metrics(_backend, _model_name, _llm_ms, _in_tok, _out_tok, _status, _error_type)


if __name__ == "__main__":
    main()
