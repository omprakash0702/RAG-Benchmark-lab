"""
run_benchmark.py — Automated Benchmark Runner

Executes a fixed query suite across pipeline modes and stores structured
results for cross-pipeline comparison.

Pipelines:
    llm_only  — pure LLM, no retrieval (parametric knowledge baseline)
    basic_rag — ChromaDB vector retrieval + LLM generation
    graph_rag — (future phase, wired in via PIPELINES list)

--- Why controlled benchmark suites matter ---
Ad-hoc evaluation ("this answer looks better") is not reproducible and is
biased by recency and question choice.  A fixed query suite with identical
inputs across every pipeline eliminates confounds: any difference in output
quality, latency, or token cost is attributable to the pipeline architecture,
not the query.

--- Why identical queries are critical ---
If LLM-only and Basic RAG are evaluated on different questions, the comparison
is meaningless.  Harder questions will make any pipeline look worse.  The same
query run through every pipeline under identical conditions is the only valid
way to measure retrieval augmentation benefit.

--- Why structured metrics are required for RAG evaluation ---
Free-text answers cannot be compared programmatically.  Structured numeric
logs (latency, token counts, retrieval counts) enable:
  - latency comparison (retrieval overhead vs. answer quality gain)
  - token cost analysis (prompt size vs. accuracy)
  - automated regression detection
  - scoring integration (BERTScore, Ollama judge, RAGAS in future phases)

--- Why failed runs must remain visible ---
A benchmark that silently drops failures inflates success metrics and hides
reliability differences between pipelines.  A pipeline that succeeds 80% of
the time is architecturally weaker than one that succeeds 100%, even if the
80% answers are slightly better.  Failed runs are recorded with status=failure,
error_type, and stderr so the cause can be diagnosed and fixed.
"""

import csv
import json
import logging
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT             = Path(__file__).resolve().parent.parent
QUERIES_FILE     = ROOT / "experiments" / "benchmark_queries.json"
RESULTS_DIR      = ROOT / "experiments" / "benchmark_results"
RESULTS_CSV      = RESULTS_DIR / "results.csv"
RESULTS_JSON     = RESULTS_DIR / "results.json"

LLM_ONLY_SCRIPT   = ROOT / "pipelines" / "llm_only"  / "generate_answer.py"
BASIC_RAG_SCRIPT  = ROOT / "pipelines" / "basic_rag" / "generate_answer.py"
GRAPH_RAG_SCRIPT  = ROOT / "pipelines" / "graph_rag" / "generate_answer.py"

# Seed account for graph_rag benchmark runs — a high-activity account with
# a known multi-hop subgraph (46 accounts, 28 banks, 161 transactions).
GRAPH_RAG_SEED_ACCOUNT = "8042DF7E0"

# Per-subprocess wall-clock timeout.  3 minutes covers model loading + API call.
SUBPROCESS_TIMEOUT_S = 180

# Retry config for the benchmark harness (separate from in-pipeline retries).
# A harness retry covers subprocess-level failures (timeout, OS crash, OOM).
MAX_HARNESS_RETRIES  = 1
HARNESS_RETRY_DELAY_S = 15   # give the API a rest before retrying

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
# Parsing patterns
#
# Both pipelines emit a pipe-delimited key=value log line.  The patterns are:
#
#   LLM_ONLY_METRICS | model=X | retrieved=0 | context=0 |
#     embed_ms=0.0 | search_ms=0.0 | llm_ms=F | total_ms=F |
#     est_tokens=0 | actual_in=N | actual_out=N | status=S | error_type=E
#
#   RAG_METRICS | backend=X | model=X | retrieved=N | context=N |
#     embed_ms=F | search_ms=F | llm_ms=F | total_ms=F |
#     est_tokens=N | actual_in=N | actual_out=N | status=S | error_type=E
#
# The regex captures any word=value pair (numeric or alphanumeric string).
# ---------------------------------------------------------------------------
_MARKER_RE = re.compile(r"(LLM_ONLY_METRICS|RAG_METRICS|GRAPH_RAG_METRICS)\s*\|(.+)$")
_KV_RE     = re.compile(r"(\w+)=([\w./-]+)")
_ANSWER_RE = re.compile(
    r"GENERATED ANSWER\s*\n\n(.*?)\n\n={60,}",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> list[dict]:
    if not path.exists():
        log.error("Query file not found: %s", path)
        sys.exit(1)
    with open(path, encoding="utf-8") as fh:
        queries = json.load(fh)
    log.info("Loaded %d benchmark queries from %s", len(queries), path.name)
    return queries


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def run_pipeline(
    script: Path,
    query: str,
    extra_args: list[str] | None = None,
) -> tuple[str, str, float, int]:
    """
    Launch a pipeline script and return (stdout, stderr, wall_ms, returncode).

    stdout and stderr are kept separate so the harness can store stderr
    (Python tracebacks, OS errors) in results.json for post-mortem debugging
    without it contaminating the metrics-line parser.

    Using subprocess keeps each pipeline's module-level state (loaded models,
    ChromaDB client) isolated per run, matching how the pipelines would be
    invoked in a real evaluation harness.
    """
    cmd = [sys.executable, str(script), "--query", query] + (extra_args or [])
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            cwd=str(ROOT),
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        return result.stdout, result.stderr, wall_ms, result.returncode

    except subprocess.TimeoutExpired as exc:
        wall_ms = (time.perf_counter() - t0) * 1000
        log.warning("Subprocess timed out after %ds: %s", SUBPROCESS_TIMEOUT_S, script.name)
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return stdout, f"TIMEOUT after {SUBPROCESS_TIMEOUT_S}s\n" + stderr, wall_ms, -1

    except Exception as exc:
        wall_ms = (time.perf_counter() - t0) * 1000
        log.error("Subprocess launch error: %s", exc)
        return "", str(exc), wall_ms, -2


def run_pipeline_with_retry(
    script: Path,
    query: str,
    extra_args: list[str] | None = None,
    max_retries: int = MAX_HARNESS_RETRIES,
) -> tuple[str, str, float, int]:
    """
    Wrap run_pipeline with harness-level retries for subprocess failures.

    Harness retries are distinct from in-pipeline Gemini API retries:
      - In-pipeline retries: handle transient API 429/503 inside the process
      - Harness retries: handle subprocess-level failures (timeout, OOM, crash)
        where the process never got a chance to emit a metrics line

    A successful metrics line (status=success OR status=failure) means the
    subprocess ran to completion — no harness retry needed even if the pipeline
    itself failed, because we have the data we need.
    """
    last_stdout, last_stderr, last_wall_ms, last_rc = "", "", 0.0, -1

    for attempt in range(1, max_retries + 2):   # +2: initial run + retries
        stdout, stderr, wall_ms, rc = run_pipeline(script, query, extra_args)
        last_stdout, last_stderr, last_wall_ms, last_rc = stdout, stderr, wall_ms, rc

        # If we got a metrics line the subprocess completed — no retry needed
        if _MARKER_RE.search(stdout):
            return stdout, stderr, wall_ms, rc

        if attempt <= max_retries:
            log.warning(
                "No metrics line found (attempt %d/%d, rc=%d) — retrying in %ds ...",
                attempt, max_retries + 1, rc, HARNESS_RETRY_DELAY_S,
            )
            time.sleep(HARNESS_RETRY_DELAY_S)

    return last_stdout, last_stderr, last_wall_ms, last_rc


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_metrics(stdout: str, pipeline: str) -> dict:
    """
    Extract numeric and string metrics from the pipeline's structured log line.

    Parses key=value pairs from the METRICS line rather than the pretty-print
    table — the log line is the machine-readable contract between pipelines and
    the benchmark harness.

    Falls back to zeroes + error classification on parse failure so a single
    bad run never aborts the benchmark loop.
    """
    defaults: dict = {
        "pipeline":    pipeline,
        "model":       "unknown",
        "backend":     pipeline,
        "retrieved":   0,
        "context":     0,
        "embed_ms":    0.0,
        "search_ms":   0.0,
        "llm_ms":      0.0,
        "total_ms":    0.0,
        "est_tokens":  0,
        "actual_in":   0,
        "actual_out":  0,
        "status":      "failure",
        "error_type":  "metrics_line_missing",
        "parse_ok":    False,
    }

    for line in stdout.splitlines():
        m = _MARKER_RE.search(line)
        if not m:
            continue
        marker = m.group(1)
        pairs  = dict(_KV_RE.findall(m.group(2)))
        if marker == "GRAPH_RAG_METRICS":
            # graph_rag uses different field names: nodes/edges, prompt_tokens/response_tokens
            defaults.update({
                "model":      pairs.get("model", pairs.get("backend", "unknown")),
                "backend":    pairs.get("backend", pipeline),
                "retrieved":  int(float(pairs.get("nodes", 0))),
                "context":    int(float(pairs.get("edges", 0))),
                "embed_ms":   float(pairs.get("retrieval_ms", 0.0)),
                "search_ms":  0.0,
                "llm_ms":     float(pairs.get("llm_ms",     0.0)),
                "total_ms":   float(pairs.get("total_ms",   0.0)),
                "est_tokens": int(float(pairs.get("prompt_tokens",   0))),
                "actual_in":  int(float(pairs.get("prompt_tokens",   0))),
                "actual_out": int(float(pairs.get("response_tokens", 0))),
                "status":     pairs.get("status",     "unknown"),
                "error_type": pairs.get("error_type", "none"),
                "parse_ok":   True,
            })
        else:
            defaults.update({
                "model":      pairs.get("model", pairs.get("backend", "unknown")),
                "backend":    pairs.get("backend", pipeline),
                "retrieved":  int(float(pairs.get("retrieved",  0))),
                "context":    int(float(pairs.get("context",    0))),
                "embed_ms":   float(pairs.get("embed_ms",   0.0)),
                "search_ms":  float(pairs.get("search_ms",  0.0)),
                "llm_ms":     float(pairs.get("llm_ms",     0.0)),
                "total_ms":   float(pairs.get("total_ms",   0.0)),
                "est_tokens": int(float(pairs.get("est_tokens",  0))),
                "actual_in":  int(float(pairs.get("actual_in",   0))),
                "actual_out": int(float(pairs.get("actual_out",  0))),
                "status":     pairs.get("status",     "unknown"),
                "error_type": pairs.get("error_type", "none"),
                "parse_ok":   True,
            })
        return defaults

    # Classify why the metrics line is absent
    if "TIMEOUT" in stdout or "TIMEOUT" in defaults.get("error_type", ""):
        defaults["error_type"] = "subprocess_timeout"
    elif not stdout.strip():
        defaults["error_type"] = "empty_subprocess_output"

    return defaults


def parse_answer(stdout: str) -> str:
    m = _ANSWER_RE.search(stdout)
    return m.group(1).strip() if m else ""


def classify_failure(metrics: dict, stderr: str) -> str:
    """
    Return a human-readable failure reason from parsed metrics + stderr.
    Used to populate the failure_reason column in results.
    """
    if metrics["status"] == "success":
        return ""
    et = metrics.get("error_type", "")
    if et in ("none", ""):
        et = metrics.get("error_type", "unknown")
    if "RuntimeError" in et or "RuntimeError" in stderr:
        # Dig out the Gemini error code from stderr if present
        for code in ("429", "503", "404", "401", "500"):
            if code in stderr:
                return f"gemini_{code}"
        return "runtime_error"
    if et == "subprocess_timeout":
        return "subprocess_timeout"
    if et == "metrics_line_missing":
        return "no_metrics_line"
    if et == "ValueError":
        return "bad_input"
    return et or "unknown"


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_query_header(qid: str, category: str, query: str, index: int, total: int) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  QUERY {index}/{total}  [{qid}]  category: {category}")
    print("=" * width)
    print(f"  {query}")
    print("=" * width)


def print_pipeline_result(label: str, metrics: dict, answer: str, wall_ms: float) -> None:
    status = metrics.get("status", "unknown")
    et     = metrics.get("error_type", "")
    badge  = "OK" if status == "success" else f"FAIL({et})"
    print(f"\n  -- {label} [{badge}] --")
    print(f"     LLM ms     : {metrics['llm_ms']:>8.1f}")
    print(f"     Total ms   : {metrics['total_ms']:>8.1f}   (wall: {wall_ms:.0f} ms)")
    print(f"     Input tok  : {metrics['actual_in']:>8,}")
    print(f"     Output tok : {metrics['actual_out']:>8,}")
    if metrics["retrieved"]:
        print(f"     Retrieved  : {metrics['retrieved']:>8}")
    if answer:
        preview = answer[:180].replace("\n", " ")
        print(f"     Answer     : {preview}{'...' if len(answer) > 180 else ''}")
    elif status != "success":
        print(f"     Answer     : [none — pipeline failed]")


def print_summary(df: pd.DataFrame) -> None:
    width = 70
    print("\n" + "=" * width)
    print("  BENCHMARK SUMMARY")
    print("=" * width)

    for pipeline in df["pipeline"].unique():
        sub     = df[df["pipeline"] == pipeline]
        ok      = sub[sub["status"] == "success"]
        failed  = sub[sub["status"] != "success"]
        success_rate = len(ok) / len(sub) * 100 if len(sub) else 0

        print(f"\n  Pipeline : {pipeline}")
        print(f"    Runs         : {len(sub)}  |  Success: {len(ok)}  |  Failed: {len(failed)}"
              f"  ({success_rate:.0f}% success rate)")

        if not ok.empty:
            print(f"    Avg LLM ms   : {ok['llm_ms'].mean():>8.1f}")
            print(f"    Avg total ms : {ok['total_ms'].mean():>8.1f}")
            print(f"    Avg input tok: {ok['actual_in'].mean():>8.1f}")
            print(f"    Avg out tok  : {ok['actual_out'].mean():>8.1f}")
            print(f"    Avg ans chars: {ok['answer_chars'].mean():>8.1f}")
        if not failed.empty:
            reasons = failed["failure_reason"].value_counts().to_dict()
            print(f"    Failure types: {reasons}")

    # Cross-pipeline comparison
    successful = {p: df[(df["pipeline"] == p) & (df["status"] == "success")]
                  for p in df["pipeline"].unique()}
    lo  = successful.get("llm_only",  pd.DataFrame())
    rag = successful.get("basic_rag", pd.DataFrame())
    grg = successful.get("graph_rag", pd.DataFrame())

    if not lo.empty and not rag.empty:
        tok_overhead = rag["actual_in"].mean() - lo["actual_in"].mean()
        print(f"\n  Basic RAG vs LLM-only")
        print(f"    LLM latency delta : {rag['llm_ms'].mean() - lo['llm_ms'].mean():+,.1f} ms avg")
        print(f"    Token overhead    : {tok_overhead:+,.1f} tokens avg (context injection)")
        print(f"    Avg answer length : {rag['answer_chars'].mean():,.0f} vs {lo['answer_chars'].mean():,.0f} chars")
    if not lo.empty and not grg.empty:
        tok_overhead = grg["actual_in"].mean() - lo["actual_in"].mean()
        print(f"\n  GraphRAG vs LLM-only")
        print(f"    LLM latency delta : {grg['llm_ms'].mean() - lo['llm_ms'].mean():+,.1f} ms avg")
        print(f"    Token overhead    : {tok_overhead:+,.1f} tokens avg (graph context)")
        print(f"    Avg nodes/edges   : {grg['retrieved'].mean():,.0f} nodes / {grg['context'].mean():,.0f} edges avg")
        print(f"    Avg answer length : {grg['answer_chars'].mean():,.0f} vs {lo['answer_chars'].mean():,.0f} chars")
    if not rag.empty and not grg.empty:
        print(f"\n  GraphRAG vs Basic RAG")
        print(f"    LLM latency delta : {grg['llm_ms'].mean() - rag['llm_ms'].mean():+,.1f} ms avg")
        print(f"    Token overhead    : {grg['actual_in'].mean() - rag['actual_in'].mean():+,.1f} tokens avg")
        print(f"    Avg answer length : {grg['answer_chars'].mean():,.0f} vs {rag['answer_chars'].mean():,.0f} chars")

    print("\n" + "=" * width)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main() -> None:
    run_id    = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat(timespec="seconds")

    log.info("=== run_benchmark.py started ===")
    log.info("Run ID    : %s", run_id)
    log.info("Timestamp : %s", timestamp)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_queries(QUERIES_FILE)

    # Pipeline registry. Each entry: (key, script_path, extra_cli_args)
    PIPELINES = [
        ("llm_only",  LLM_ONLY_SCRIPT,  []),
        ("basic_rag", BASIC_RAG_SCRIPT, ["--top_k", "5", "--max_context_chunks", "5"]),
        ("graph_rag", GRAPH_RAG_SCRIPT, [
            "--account", GRAPH_RAG_SEED_ACCOUNT,
            "--depth",   "3",
            "--model",   "openai",
        ]),
    ]

    all_rows: list[dict] = []

    for i, q in enumerate(queries, 1):
        qid      = q["id"]
        category = q["category"]
        query    = q["query"]

        print_query_header(qid, category, query, i, len(queries))

        for pipeline_key, script, extra_args in PIPELINES:
            log.info("Running %s for [%s] ...", pipeline_key, qid)

            stdout, stderr, wall_ms, rc = run_pipeline_with_retry(
                script, query, extra_args,
                max_retries=MAX_HARNESS_RETRIES,
            )

            metrics        = parse_metrics(stdout, pipeline_key)
            answer         = parse_answer(stdout)
            failure_reason = classify_failure(metrics, stderr)

            print_pipeline_result(
                pipeline_key.replace("_", " ").upper(),
                metrics, answer, wall_ms,
            )

            if metrics["status"] != "success":
                log.warning(
                    "[%s] %s — %s (rc=%d)",
                    qid, pipeline_key, failure_reason, rc,
                )

            row = {
                "run_id":         run_id,
                "timestamp":      timestamp,
                "query_id":       qid,
                "category":       category,
                "query":          query,
                "pipeline":       pipeline_key,
                "model":          metrics["model"],
                "status":         metrics["status"],
                "failure_reason": failure_reason,
                "retrieved":      metrics["retrieved"],
                "context":        metrics["context"],
                "embed_ms":       metrics["embed_ms"],
                "search_ms":      metrics["search_ms"],
                "llm_ms":         metrics["llm_ms"],
                "total_ms":       metrics["total_ms"],
                "wall_ms":        round(wall_ms, 1),
                "est_tokens":     metrics["est_tokens"],
                "actual_in":      metrics["actual_in"],
                "actual_out":     metrics["actual_out"],
                "answer_chars":   len(answer),
                "parse_ok":       metrics["parse_ok"],
                "stderr_snippet": stderr[:500] if stderr.strip() else "",
                "answer":         answer,
            }
            all_rows.append(row)

    # -- Persist results -------------------------------------------------------
    df = pd.DataFrame(all_rows)

    csv_cols = [c for c in df.columns if c not in ("answer", "stderr_snippet")]
    df[csv_cols].to_csv(RESULTS_CSV, index=False, quoting=csv.QUOTE_ALL)
    log.info("CSV results : %s", RESULTS_CSV)

    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_id":    run_id,
                "timestamp": timestamp,
                "queries":   len(queries),
                "pipelines": [p[0] for p in PIPELINES],
                "results":   all_rows,
            },
            fh, indent=2, ensure_ascii=False,
        )
    log.info("JSON results: %s", RESULTS_JSON)

    print_summary(df)
    log.info("=== Benchmark complete — run_id: %s ===", run_id)


if __name__ == "__main__":
    main()
