"""
evaluate_benchmark.py — Comprehensive RAG Benchmark Evaluator

Reads benchmark_results/results.json and scores each pipeline answer on:

  1. LATENCY     — retrieval_ms, llm_ms, total_ms (from benchmark run)
  2. COST        — estimated USD based on actual token counts + model pricing
  3. BERTSCORE   — semantic similarity vs expert reference answers (F1, roberta-base)
  4. LLM-JUDGE   — GPT-4o-mini scores each answer on 4 AML-specific dimensions (1-10):
                     relevance, evidence_quality, aml_insight, actionability

Outputs:
  - Console table (per-pipeline summary + per-query breakdown)
  - experiments/benchmark_results/evaluation_report.md
  - experiments/benchmark_results/scores.csv
"""

import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
ENV_FILE     = ROOT / ".env"
RESULTS_JSON = ROOT / "experiments" / "benchmark_results" / "results.json"
REFS_JSON    = ROOT / "experiments" / "reference_answers.json"
REPORT_MD    = ROOT / "experiments" / "benchmark_results" / "evaluation_report.md"
SCORES_CSV   = ROOT / "experiments" / "benchmark_results" / "scores.csv"

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
# Model pricing (USD per 1M tokens, as of mid-2025)
# ---------------------------------------------------------------------------
PRICING = {
    "gemini-2.5-flash": {"input": 0.075,  "output": 0.30},
    "gpt-4o-mini":      {"input": 0.150,  "output": 0.60},
    "default":          {"input": 0.150,  "output": 0.60},
}

JUDGE_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def compute_cost(model: str, actual_in: int, actual_out: int) -> float:
    p = PRICING.get(model, PRICING["default"])
    return (actual_in * p["input"] + actual_out * p["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def run_bertscore(predictions: list[str], references: list[str]) -> list[dict]:
    """Return per-example BERTScore P/R/F1 using roberta-base (CPU-friendly)."""
    log.info("Computing BERTScore for %d examples ...", len(predictions))
    try:
        from bert_score import score as bert_score_fn
        import warnings
        warnings.filterwarnings("ignore")
        P, R, F1 = bert_score_fn(
            predictions, references,
            model_type="distilbert-base-uncased",
            lang="en",
            verbose=False,
            device="cpu",
        )
        return [
            {"bertscore_p": round(float(p), 4),
             "bertscore_r": round(float(r), 4),
             "bertscore_f1": round(float(f), 4)}
            for p, r, f in zip(P, R, F1)
        ]
    except Exception as exc:
        log.warning("BERTScore failed: %s — filling with NaN", exc)
        return [{"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}] * len(predictions)


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert AML (Anti-Money Laundering) compliance analyst evaluating an AI system's response to an investigation query.

QUERY: {query}

AI ANSWER:
{answer}

Score this answer on each dimension from 1 to 10 (10 = best):

1. RELEVANCE (1-10): Does the answer directly address the specific AML pattern described in the query?
2. EVIDENCE_QUALITY (1-10): Does the answer cite specific, concrete evidence (account IDs, transaction counts, bank names, amounts, actual data)? Score 1-3 if only generic AML descriptions, 7-10 if cites specific real identifiers.
3. AML_INSIGHT (1-10): Quality and depth of AML pattern identification and reasoning. Does it correctly name the laundering technique and explain its mechanics?
4. ACTIONABILITY (1-10): Are the recommended follow-up investigative actions specific, realistic, and useful to an AML analyst?

Respond ONLY with valid JSON (no markdown, no explanation):
{{"relevance": <int>, "evidence_quality": <int>, "aml_insight": <int>, "actionability": <int>}}
"""


def llm_judge_answer(client, query: str, answer: str) -> dict:
    """Call GPT-4o-mini to score a single answer. Returns dict with 4 scores."""
    prompt = _JUDGE_PROMPT.format(query=query, answer=answer[:3000])
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            timeout=30,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        scores = json.loads(raw)
        return {
            "judge_relevance":       int(scores.get("relevance",       5)),
            "judge_evidence":        int(scores.get("evidence_quality", 5)),
            "judge_insight":         int(scores.get("aml_insight",     5)),
            "judge_actionability":   int(scores.get("actionability",   5)),
            "judge_overall":         round(
                (scores.get("relevance", 5) + scores.get("evidence_quality", 5) +
                 scores.get("aml_insight", 5) + scores.get("actionability", 5)) / 4, 2
            ),
        }
    except Exception as exc:
        log.warning("Judge failed for one answer: %s", exc)
        return {"judge_relevance": None, "judge_evidence": None,
                "judge_insight": None, "judge_actionability": None,
                "judge_overall": None}


def run_llm_judge(client, rows: list[dict]) -> list[dict]:
    """Score all rows. Returns list of score dicts aligned with rows."""
    scores = []
    for i, row in enumerate(rows, 1):
        log.info("  Judge %d/%d — %s / %s", i, len(rows), row["pipeline"], row["query_id"])
        s = llm_judge_answer(client, row["query"], row["answer"])
        scores.append(s)
        time.sleep(0.3)   # avoid rate-limiting
    return scores


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

PIPELINE_LABELS = {
    "llm_only":  "LLM-only",
    "basic_rag": "Basic RAG",
    "graph_rag": "GraphRAG",
}

PIPELINE_ORDER = ["llm_only", "basic_rag", "graph_rag"]


def _fmt(v, fmt=".2f", fallback="N/A"):
    if v is None:
        return fallback
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return fallback


def build_report(df: pd.DataFrame) -> str:
    lines = []
    w = 72

    lines.append("=" * w)
    lines.append("  RAG BENCHMARK EVALUATION REPORT")
    lines.append(f"  Run ID : {df['run_id'].iloc[0]}   |   Queries: {df['query_id'].nunique()}   |   Pipelines: {df['pipeline'].nunique()}")
    lines.append("=" * w)

    # ── Per-pipeline summary ──────────────────────────────────────────────
    lines.append("")
    lines.append("  PIPELINE SUMMARY")
    lines.append("-" * w)
    header = f"  {'Pipeline':<12} {'LLM ms':>8} {'Total ms':>9} {'Cost $':>9} {'BERTScore':>10} {'Judge':>7} {'Tokens-in':>10}"
    lines.append(header)
    lines.append("  " + "-" * (w - 2))

    for p in PIPELINE_ORDER:
        sub = df[df["pipeline"] == p]
        if sub.empty:
            continue
        label    = PIPELINE_LABELS.get(p, p)
        llm_ms   = sub["llm_ms"].mean()
        total_ms = sub["total_ms"].mean()
        cost     = sub["cost_usd"].sum()
        bscore   = sub["bertscore_f1"].mean()
        judge    = sub["judge_overall"].mean()
        tok_in   = sub["actual_in"].mean()
        lines.append(
            f"  {label:<12} {llm_ms:>8.0f} {total_ms:>9.0f} {cost:>9.5f}"
            f" {_fmt(bscore, '.4f'):>10} {_fmt(judge, '.2f'):>7} {tok_in:>10.0f}"
        )

    lines.append("")

    # ── Latency detail ────────────────────────────────────────────────────
    lines.append("  LATENCY BREAKDOWN  (avg ms across 8 queries)")
    lines.append("-" * w)
    for p in PIPELINE_ORDER:
        sub = df[df["pipeline"] == p]
        if sub.empty:
            continue
        label = PIPELINE_LABELS.get(p, p)
        retrieval = sub["embed_ms"].mean()   # retrieval / embed ms
        llm       = sub["llm_ms"].mean()
        total     = sub["total_ms"].mean()
        lines.append(f"  {label:<12}  retrieval={retrieval:>7.0f} ms  |  llm={llm:>7.0f} ms  |  total={total:>7.0f} ms")
    lines.append("")

    # ── Cost detail ───────────────────────────────────────────────────────
    lines.append("  COST BREAKDOWN  (8 queries, USD)")
    lines.append("-" * w)
    for p in PIPELINE_ORDER:
        sub = df[df["pipeline"] == p]
        if sub.empty:
            continue
        label     = PIPELINE_LABELS.get(p, p)
        cost      = sub["cost_usd"].sum()
        avg_in    = sub["actual_in"].mean()
        avg_out   = sub["actual_out"].mean()
        model     = sub["model"].iloc[0]
        lines.append(
            f"  {label:<12}  total=${cost:.5f}  avg_in={avg_in:.0f}  avg_out={avg_out:.0f}  model={model}"
        )
    lines.append("")

    # ── BERTScore detail ──────────────────────────────────────────────────
    lines.append("  BERTSCORE  (vs expert reference answers, distilbert-base-uncased)")
    lines.append("-" * w)
    bs_sub = df[["pipeline", "query_id", "bertscore_p", "bertscore_r", "bertscore_f1"]].copy()
    for p in PIPELINE_ORDER:
        sub = bs_sub[bs_sub["pipeline"] == p]
        if sub.empty:
            continue
        label = PIPELINE_LABELS.get(p, p)
        avg_p  = sub["bertscore_p"].mean()
        avg_r  = sub["bertscore_r"].mean()
        avg_f1 = sub["bertscore_f1"].mean()
        lines.append(f"  {label:<12}  P={_fmt(avg_p,'.4f')}  R={_fmt(avg_r,'.4f')}  F1={_fmt(avg_f1,'.4f')}")
    lines.append("")

    # ── LLM Judge detail ──────────────────────────────────────────────────
    lines.append("  LLM-AS-A-JUDGE  (GPT-4o-mini, 1-10 scale)")
    lines.append("-" * w)
    jcols = ["judge_relevance", "judge_evidence", "judge_insight", "judge_actionability", "judge_overall"]
    jlabels = ["Relevance", "Evidence", "Insight", "Action", "OVERALL"]
    header2 = f"  {'Pipeline':<12}" + "".join(f"  {l:>9}" for l in jlabels)
    lines.append(header2)
    lines.append("  " + "-" * (w - 2))
    for p in PIPELINE_ORDER:
        sub = df[df["pipeline"] == p]
        if sub.empty:
            continue
        label = PIPELINE_LABELS.get(p, p)
        vals  = [_fmt(sub[c].mean(), ".2f") for c in jcols]
        lines.append(f"  {label:<12}" + "".join(f"  {v:>9}" for v in vals))
    lines.append("")

    # ── Per-query judge scores ────────────────────────────────────────────
    lines.append("  PER-QUERY JUDGE SCORES  (overall, 1-10)")
    lines.append("-" * w)
    qids = sorted(df["query_id"].unique())
    col_w = 10
    hdr = f"  {'Query':<8}" + "".join(f" {PIPELINE_LABELS.get(p, p):>{col_w}}" for p in PIPELINE_ORDER)
    lines.append(hdr)
    lines.append("  " + "-" * (w - 2))
    for qid in qids:
        row_parts = [f"  {qid:<8}"]
        for p in PIPELINE_ORDER:
            cell = df[(df["pipeline"] == p) & (df["query_id"] == qid)]["judge_overall"]
            val  = _fmt(cell.iloc[0], ".1f") if not cell.empty else "N/A"
            row_parts.append(f" {val:>{col_w}}")
        lines.append("".join(row_parts))
    lines.append("")

    # ── Winner summary ────────────────────────────────────────────────────
    lines.append("  BENCHMARK VERDICT")
    lines.append("=" * w)

    pipeline_stats = {}
    for p in PIPELINE_ORDER:
        sub = df[df["pipeline"] == p]
        if sub.empty:
            continue
        pipeline_stats[p] = {
            "label":      PIPELINE_LABELS.get(p, p),
            "total_ms":   sub["total_ms"].mean(),
            "cost":       sub["cost_usd"].sum(),
            "bertscore":  sub["bertscore_f1"].mean(),
            "judge":      sub["judge_overall"].mean(),
        }

    if pipeline_stats:
        fastest   = min(pipeline_stats, key=lambda p: pipeline_stats[p]["total_ms"])
        cheapest  = min(pipeline_stats, key=lambda p: pipeline_stats[p]["cost"])
        bs_winner = max(pipeline_stats, key=lambda p: pipeline_stats[p]["bertscore"] or 0)
        jg_winner = max(pipeline_stats, key=lambda p: pipeline_stats[p]["judge"] or 0)

        lines.append(f"  Fastest pipeline  : {pipeline_stats[fastest]['label']}  ({pipeline_stats[fastest]['total_ms']:.0f} ms avg total)")
        lines.append(f"  Cheapest pipeline : {pipeline_stats[cheapest]['label']}  (${pipeline_stats[cheapest]['cost']:.5f} for 8 queries)")
        lines.append(f"  BERTScore winner  : {pipeline_stats[bs_winner]['label']}  (F1={_fmt(pipeline_stats[bs_winner]['bertscore'],'.4f')})")
        lines.append(f"  Judge score winner: {pipeline_stats[jg_winner]['label']}  (avg={_fmt(pipeline_stats[jg_winner]['judge'],'.2f')}/10)")

    lines.append("=" * w)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(ENV_FILE)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        log.error("OPENAI_API_KEY not set in .env — needed for LLM judge.")
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=openai_key)

    # Load results
    log.info("Loading results from %s ...", RESULTS_JSON)
    with open(RESULTS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data["results"]
    log.info("  %d rows loaded across %d pipelines", len(rows), len({r["pipeline"] for r in rows}))

    # Load reference answers
    with open(REFS_JSON, encoding="utf-8") as fh:
        refs_list = json.load(fh)
    refs = {r["id"]: r["reference"] for r in refs_list}

    # ── Cost ──────────────────────────────────────────────────────────────
    log.info("Computing costs ...")
    for row in rows:
        row["cost_usd"] = compute_cost(row["model"], row["actual_in"], row["actual_out"])

    # ── BERTScore ─────────────────────────────────────────────────────────
    predictions = [r["answer"] for r in rows]
    references  = [refs.get(r["query_id"], "") for r in rows]
    bs_scores   = run_bertscore(predictions, references)
    for row, bs in zip(rows, bs_scores):
        row.update(bs)

    # ── LLM Judge ─────────────────────────────────────────────────────────
    log.info("Running LLM-as-a-Judge on %d answers (GPT-4o-mini) ...", len(rows))
    judge_scores = run_llm_judge(client, rows)
    for row, js in zip(rows, judge_scores):
        row.update(js)

    # ── DataFrame ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Save scores CSV
    score_cols = [
        "run_id", "query_id", "category", "pipeline", "model",
        "actual_in", "actual_out", "cost_usd",
        "embed_ms", "llm_ms", "total_ms",
        "bertscore_p", "bertscore_r", "bertscore_f1",
        "judge_relevance", "judge_evidence", "judge_insight",
        "judge_actionability", "judge_overall",
    ]
    df[score_cols].to_csv(SCORES_CSV, index=False, quoting=csv.QUOTE_ALL)
    log.info("Scores CSV saved: %s", SCORES_CSV)

    # Build and print report
    report = build_report(df)
    print("\n" + report)

    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write("```\n" + report + "\n```\n")
    log.info("Report saved : %s", REPORT_MD)


if __name__ == "__main__":
    main()
