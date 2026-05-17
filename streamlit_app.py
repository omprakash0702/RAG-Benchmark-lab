"""
streamlit_app.py — RAG Benchmark Lab Dashboard

Full interactive UI for the AML GraphRAG benchmark:
  - Database overview (TigerGraph schema + ChromaDB stats)
  - 70 example queries with guided query interface
  - Live 3-pipeline comparison (LLM-only / Basic RAG / GraphRAG)
  - Per-pipeline API selection (Gemini / OpenAI)
  - LLM-as-a-Judge (Groq / OpenAI / Gemini)
  - BERTScore evaluation
  - Plotly visualisations (radar, bar, donut)
"""

import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT             = Path(__file__).resolve().parent
ENV_FILE         = ROOT / ".env"
QUERIES_FILE     = ROOT / "experiments" / "example_queries.json"
REFS_FILE        = ROOT / "experiments" / "reference_answers.json"
LLM_ONLY_SCRIPT  = ROOT / "pipelines" / "llm_only"  / "generate_answer.py"
BASIC_RAG_SCRIPT = ROOT / "pipelines" / "basic_rag" / "generate_answer.py"
GRAPH_RAG_SCRIPT = ROOT / "pipelines" / "graph_rag" / "generate_answer.py"

load_dotenv(ENV_FILE)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Benchmark Lab — AML Investigation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIPELINE_COLORS = {
    "LLM-only":  "#636EFA",
    "Basic RAG": "#EF553B",
    "GraphRAG":  "#00CC96",
}

GROQ_MODELS   = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-1.5-flash"]

KNOWN_ACCOUNTS = [
    ("8042DF7E0",  "Corporation #6579",          "46 connected accts, 161 txns, 28 banks, 4-hop"),
    ("81076F8E0",  "Unknown entity",              "Self-transfer account, 1 acct, 2 txns"),
    ("80A4D6EB0",  "Sole Proprietorship #6323",   "Connected to 8042DF7E0 network"),
    ("809C54730",  "Partnership #14283",          "Pass-through intermediary account"),
    ("8042E0040",  "Partnership #577",            "Multi-bank routing account"),
]

CURATED_QUERIES = [
    # Network Structure — GraphRAG best
    {"category": "Network Structure",     "best": "GraphRAG",
     "query": "Which accounts send money to the most unique recipients?"},
    {"category": "Network Structure",     "best": "GraphRAG",
     "query": "Find accounts receiving funds from many different senders (fan-in pattern)"},
    {"category": "Network Structure",     "best": "GraphRAG",
     "query": "Show circular money movement — accounts that send and receive back from the same counterparty"},
    {"category": "Network Structure",     "best": "GraphRAG",
     "query": "Which accounts act as pass-through nodes with high inbound and outbound volume?"},
    {"category": "Network Structure",     "best": "GraphRAG",
     "query": "Show top accounts sending Cash transactions to more than 5 unique recipients"},
    {"category": "Network Structure",     "best": "GraphRAG",
     "query": "Identify layering chains where funds move through 3 or more intermediary accounts"},
    {"category": "Network Structure",     "best": "GraphRAG",
     "query": "Which accounts are connected to both high-volume senders and receivers simultaneously?"},
    # Account Investigation — GraphRAG best
    {"category": "Account Investigation", "best": "GraphRAG",
     "query": "How many unique accounts does 8042DF7E0 send money to?"},
    {"category": "Account Investigation", "best": "GraphRAG",
     "query": "Map the full transaction network around account 8042DF7E0 up to 4 hops"},
    {"category": "Account Investigation", "best": "GraphRAG",
     "query": "Investigate suspicious cross-bank layering behavior involving account 8042DF7E0"},
    {"category": "Account Investigation", "best": "GraphRAG",
     "query": "Which accounts in the network of 809C54730 connect to other high-volume pass-through accounts?"},
    {"category": "Account Investigation", "best": "GraphRAG",
     "query": "Detect accounts that exclusively transact with other flagged or high-risk counterparties"},
    {"category": "Account Investigation", "best": "GraphRAG",
     "query": "Which accounts in the network have the highest ratio of outbound to inbound transactions?"},
    # Payment Format — Basic RAG best
    {"category": "Payment Format",        "best": "Basic RAG",
     "query": "Find sole proprietorship accounts with transaction volumes inconsistent with business size"},
    {"category": "Payment Format",        "best": "Basic RAG",
     "query": "Which accounts sent Wire transfers to multiple different banks?"},
    {"category": "Payment Format",        "best": "Basic RAG",
     "query": "Which accounts sent Cheque payments to multiple different banks?"},
    {"category": "Payment Format",        "best": "Basic RAG",
     "query": "Show large cross-currency transactions paid via ACH"},
    {"category": "Payment Format",        "best": "Basic RAG",
     "query": "Find Cash transactions above 500000 USD sent to more than one recipient"},
    # Cross-Bank Routing — Basic RAG best
    {"category": "Cross-Bank Routing",    "best": "Basic RAG",
     "query": "Large transactions routed through multiple banks across jurisdictions"},
    {"category": "Cross-Bank Routing",    "best": "Basic RAG",
     "query": "Funds moving across 5 or more banks before reaching ultimate beneficiary"},
    # AML Patterns — Any pipeline
    {"category": "AML Patterns",          "best": "Any",
     "query": "Multiple rapid transfers through intermediary accounts to obscure fund origin"},
    {"category": "AML Patterns",          "best": "Any",
     "query": "Account making 20 or more transactions per day all under the reporting threshold"},
    {"category": "AML Patterns",          "best": "Any",
     "query": "Round-trip transactions completing a full cycle within 48 hours"},
]

SUBPROCESS_TIMEOUT = 180

_METRICS_RE = re.compile(r"(LLM_ONLY_METRICS|RAG_METRICS|GRAPH_RAG_METRICS)\s*\|(.+)$")
_KV_RE      = re.compile(r"(\w+)=([\w./:@-]+)")
_ANSWER_RE  = re.compile(r"GENERATED ANSWER\s*\n\n(.*?)(?:\n\n={40,}|\Z)", re.DOTALL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_example_queries():
    with open(QUERIES_FILE, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_reference_answers():
    if not REFS_FILE.exists():
        return {}
    with open(REFS_FILE, encoding="utf-8") as f:
        refs = json.load(f)
    return {r["category"]: r["reference"] for r in refs}


def run_pipeline(script: Path, query: str, extra_args: list) -> tuple[str, str, float]:
    cmd = [sys.executable, str(script), "--query", query] + extra_args
    t0 = time.perf_counter()
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=SUBPROCESS_TIMEOUT, cwd=str(ROOT),
            encoding="utf-8", errors="replace",
        )
        wall = (time.perf_counter() - t0) * 1000
        return res.stdout, res.stderr, wall
    except subprocess.TimeoutExpired as exc:
        wall = (time.perf_counter() - t0) * 1000
        out  = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return out, f"TIMEOUT after {SUBPROCESS_TIMEOUT}s", wall
    except Exception as exc:
        return "", str(exc), (time.perf_counter() - t0) * 1000


def parse_metrics(stdout: str, pipeline: str) -> dict:
    defaults = dict(pipeline=pipeline, status="failure", error_type="no_metrics",
                    llm_ms=0.0, total_ms=0.0, actual_in=0, actual_out=0,
                    retrieved=0, context=0, embed_ms=0.0)
    for line in stdout.splitlines():
        m = _METRICS_RE.search(line)
        if not m:
            continue
        marker = m.group(1)
        pairs  = dict(_KV_RE.findall(m.group(2)))
        if marker == "GRAPH_RAG_METRICS":
            defaults.update(dict(
                status     = pairs.get("status",          "unknown"),
                error_type = pairs.get("error_type",      "none"),
                llm_ms     = float(pairs.get("llm_ms",    0)),
                total_ms   = float(pairs.get("total_ms",  0)),
                actual_in  = int(float(pairs.get("prompt_tokens",   0))),
                actual_out = int(float(pairs.get("response_tokens", 0))),
                retrieved  = int(float(pairs.get("nodes", 0))),
                context    = int(float(pairs.get("edges", 0))),
                embed_ms   = float(pairs.get("retrieval_ms", 0)),
                model      = pairs.get("model", pairs.get("backend", "?")),
            ))
        else:
            defaults.update(dict(
                status     = pairs.get("status",       "unknown"),
                error_type = pairs.get("error_type",   "none"),
                llm_ms     = float(pairs.get("llm_ms",     0)),
                total_ms   = float(pairs.get("total_ms",   0)),
                actual_in  = int(float(pairs.get("actual_in",  0))),
                actual_out = int(float(pairs.get("actual_out", 0))),
                retrieved  = int(float(pairs.get("retrieved",  0))),
                embed_ms   = float(pairs.get("embed_ms",   0)),
                model      = pairs.get("model", pairs.get("backend", "?")),
            ))
        return defaults
    return defaults


def parse_answer(stdout: str) -> str:
    m = _ANSWER_RE.search(stdout)
    if m:
        return m.group(1).strip()
    for line in reversed(stdout.splitlines()):
        if line.strip() and not line.startswith(("=", "-", " INFO", " WARN")):
            return line.strip()
    return ""


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = {
        "gpt-4o-mini":      (0.150, 0.600),
        "gpt-4o":           (2.500, 10.00),
        "gemini-2.5-flash": (0.075, 0.300),
        "gemini-1.5-flash": (0.075, 0.300),
    }
    p = pricing.get(model, (0.150, 0.600))
    return (tokens_in * p[0] + tokens_out * p[1]) / 1_000_000


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are a strict AML forensics evaluator scoring an AI pipeline's answer.

PIPELINE: {pipeline}
QUERY: {query}
ANSWER: {answer}

═══ PIPELINE DATA REALITY ═══
LLM-only  — ZERO database access. Every account ID, amount, or date it cites is invented by the model.
Basic RAG — retrieved 5 real transaction records. Cited specifics must trace to those 5 records only.
            5 records = a tiny sample. It cannot see the full network, circular paths, or multi-hop chains.
GraphRAG  — traversed a real account graph (BFS). Outputs like "Account X: in=4 out=3", confirmed
            circular pairs "A ↔ B", pass-through degree counts, and cross-bank totals ARE verified
            graph-computed facts. These count as strong evidence — score them 8+ for Evidence Grounding
            and Relationship Reasoning when present. GraphRAG sees the full network structure;
            Basic RAG sees only 5 isolated records.

═══ SCORE 6 DIMENSIONS (1–10 each) ═══

1. EVIDENCE_GROUNDING  [most important]
Does the answer use actual retrieved evidence rather than generic AML knowledge?
  8–10: cites specific account IDs / transaction IDs / graph edges / retrieved entities
  4–7 : describes patterns without tying them to specific retrieved data
  1–3 : generic AML lecture, vague suspicion wording, fabricated specifics
  → LLM-only MUST score ≤ 2 here. It has no evidence. No exceptions.

2. RELATIONSHIP_REASONING
Does the answer correctly identify network-level relationships?
  8–10: multi-hop chains, intermediary accounts, circular movement, cross-bank routing, graph topology
  4–7 : identifies direct pairs but misses multi-hop or structural patterns
  1–3 : treats each transaction in isolation, no topology awareness

3. INVESTIGATIVE_INSIGHT
Does the answer deliver meaningful AML interpretation beyond restating facts?
  8–10: names the laundering mechanic (layering / smurfing / round-trip), explains why it is suspicious
  4–7 : reasonable analysis but generic or surface-level
  1–3 : merely restates transactions, no interpretation

4. FACTUAL_CORRECTNESS
Are counts, account IDs, and relationships consistent with the provided context?
  8–10: no hallucinations, numerically consistent, every claim verifiable from evidence
  4–7 : mostly accurate with minor inconsistencies
  1–3 : fabricated entities, invented transactions, wrong counts
  → LLM-only MUST score ≤ 2 here. Its specifics are always invented.

5. ACTIONABILITY
Would this concretely help an AML investigator take the next step?
  8–10: prioritizes specific suspicious nodes, names accounts to investigate, clear follow-up
  4–7 : useful direction but too generic for direct action
  1–3 : generic compliance filler, nothing an investigator can act on

6. CONCISENESS
Does the answer avoid bloat and unnecessary repetition?
  8–10: dense, efficient — every sentence adds new information
  4–7 : some padding but mostly on-point
  1–3 : verbose context dumping, redundant summaries, repetitive wording

═══ HARD PENALTIES (subtract from final_score, cumulative) ═══
-5 : hallucinated account IDs not present in any evidence
-4 : invented transaction amounts or dates
-3 : unsupported accusations against specific accounts
-3 : contradictory reasoning (asserts X then implies not-X)

═══ COMPUTE final_score ═══
final_score = avg(6 dimensions) + sum(applicable penalties)
Clamp final_score to range [0.0, 10.0].
Round to 2 decimal places.

═══ OUTPUT — return ONLY this JSON, no markdown ═══
{{
  "evidence_grounding": <int 1-10>,
  "relationship_reasoning": <int 1-10>,
  "investigative_insight": <int 1-10>,
  "factual_correctness": <int 1-10>,
  "actionability": <int 1-10>,
  "conciseness": <int 1-10>,
  "final_score": <float 0.0-10.0>,
  "justification": "<one sentence: why this score, name any penalties applied>"
}}
"""


def judge_with_openai(query: str, answer: str, model: str, api_key: str, pipeline: str = "") -> dict:
    import openai
    client = openai.OpenAI(api_key=api_key)
    prompt = _JUDGE_PROMPT.format(query=query, answer=answer[:3000], pipeline=pipeline)
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=200, timeout=30,
    )
    raw = re.sub(r"```(?:json)?", "", resp.choices[0].message.content).strip().rstrip("`")
    return json.loads(raw)


def judge_with_groq(query: str, answer: str, model: str, api_key: str, pipeline: str = "") -> dict:
    from groq import Groq
    client = Groq(api_key=api_key)
    prompt = _JUDGE_PROMPT.format(query=query, answer=answer[:3000], pipeline=pipeline)
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=200,
    )
    raw = re.sub(r"```(?:json)?", "", resp.choices[0].message.content).strip().rstrip("`")
    return json.loads(raw)


def judge_with_gemini(query: str, answer: str, model: str, api_key: str, pipeline: str = "") -> dict:
    import google.genai as genai
    client = genai.Client(api_key=api_key)
    prompt = _JUDGE_PROMPT.format(query=query, answer=answer[:3000], pipeline=pipeline)
    resp = client.models.generate_content(model=model, contents=prompt)
    raw = re.sub(r"```(?:json)?", "", resp.text or "{}").strip().rstrip("`")
    return json.loads(raw)


def run_judge(query: str, answer: str, backend: str, model: str,
              openai_key: str, groq_key: str, gemini_key: str = "",
              pipeline: str = "") -> dict | None:
    try:
        if backend == "OpenAI":
            scores = judge_with_openai(query, answer, model, openai_key, pipeline=pipeline)
        elif backend == "Groq":
            scores = judge_with_groq(query, answer, model, groq_key, pipeline=pipeline)
        else:  # Gemini
            scores = judge_with_gemini(query, answer, model, gemini_key, pipeline=pipeline)
        # Use LLM-computed final_score; fall back to dim average if missing
        if "final_score" in scores:
            scores["overall"] = round(float(scores["final_score"]), 2)
        else:
            _dims = ["evidence_grounding", "relationship_reasoning", "investigative_insight",
                     "factual_correctness", "actionability", "conciseness"]
            scores["overall"] = round(max(0.0, sum(scores.get(d, 5) for d in _dims) / len(_dims)), 2)
        return scores
    except Exception as exc:
        st.warning(f"Judge failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(predictions: list[str], references: list[str]) -> tuple[list, list, list]:
    try:
        from bert_score import score as bscore
        P, R, F1 = bscore(predictions, references,
                          model_type="distilbert-base-uncased", lang="en",
                          verbose=False, device="cpu")
        return ([round(float(p), 4) for p in P],
                [round(float(r), 4) for r in R],
                [round(float(f), 4) for f in F1])
    except Exception:
        n = len(predictions)
        return [None] * n, [None] * n, [None] * n


# Signals that mark a query as graph-native (numerical/structural — BERTScore not meaningful)
_GRAPH_NATIVE_RE = re.compile(
    r"\bhow many\b|\bcount\b|\bnumber of\b|\btop \d|\bhighest\b|\blowest\b"
    r"|\bmost unique\b|\bunique recipients\b|\bunique accounts\b|\bratio of\b"
    r"|\bmap the full\b|\bup to \d hops?\b|\b[0-9A-F]{7,9}\b",
    re.IGNORECASE,
)
_INVESTIGATIVE_SIGNALS = [
    "investigate", "suspicious", "layering", "smurfing", "circular", "round-trip",
    "structuring", "cross-bank routing", "rapid transfers", "obscure fund",
    "intermediary", "pass-through", "multiple banks", "currency conversion",
    "self-transfer", "fan-out", "fan-in", "wire transfers", "repeated wire",
    "high frequency", "flagged", "high-risk counterpart", "reporting threshold",
    "large transactions routed", "funds moving", "routed through",
]

def _classify_for_bertscore(query: str) -> str:
    """Return 'graph-native' (skip) or 'investigative' (apply vs reference)."""
    if _GRAPH_NATIVE_RE.search(query):
        return "graph-native"
    q_low = query.lower()
    for sig in _INVESTIGATIVE_SIGNALS:
        if sig in q_low:
            return "investigative"
    return "investigative"  # default: treat as investigative


# Keyword → reference category mapping (ordered: first match wins)
_REF_KEYWORDS: list[tuple[list[str], str]] = [
    (["repeated wire", "wire transfers between foreign", "foreign banks"], "repeated_wire_transfers"),
    (["circular", "round-trip", "funds return", "return to the originating"], "circular_transactions"),
    (["rapid transfers through intermediary", "obscure fund origin", "intermediary accounts to obscure"], "layering_behavior"),
    (["high frequency small", "many recipients within hours", "small transfers from one account to many"], "high_frequency_transfers"),
    (["reporting threshold", "just below", "transactions just below", "structuring"], "structuring_patterns"),
    (["5 or more banks", "routed through multiple banks", "multiple banks across jurisdictions", "correspondent bank"], "multi_bank_routing"),
    (["currency conversion", "rapid conversion between currencies", "cross-currency"], "cross_currency_conversion"),
    (["self-transfer", "self-transfers", "unusual hours", "large self-transfer"], "self_transfers"),
    (["pass-through", "high inbound and outbound volume", "pass through nodes"], "pass_through_detection"),
    (["flagged or high-risk", "exclusively transact with other flagged", "flagged counterpart"], "flagged_counterparties"),
    (["cheque payment", "cheque payments to multiple", "cheque to multiple"], "cross_bank_cheque"),
    (["wire transfers to multiple different banks", "wire to multiple different banks"], "cross_bank_wire"),
    (["cash transactions", "cash to more than", "cash fan", "mass cash disbursement"], "cash_fan_out"),
    (["investigate suspicious cross-bank", "suspicious cross-bank laundering", "cross-bank laundering"], "suspicious_investigation"),
    (["layering", "funds move through 3", "funds move through 4", "multi-hop"], "layering_behavior"),
    (["large transactions routed", "funds moving across", "routed through"], "multi_bank_routing"),
]

def _find_reference_for_query(query: str, refs: dict) -> tuple[str | None, str | None]:
    """Return (category, reference_text) matched by keyword, or (None, None)."""
    q_low = query.lower()
    for keywords, category in _REF_KEYWORDS:
        if any(kw in q_low for kw in keywords):
            ref = refs.get(category)
            if ref:
                return category, ref
    return None, None


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def radar_chart(judge_results: dict) -> go.Figure:
    dims   = ["evidence_grounding", "relationship_reasoning", "investigative_insight",
              "factual_correctness", "actionability", "conciseness"]
    labels = ["Evidence", "Relationships", "Insight", "Factual", "Actionability", "Conciseness"]
    fig = go.Figure()
    for pipeline, scores in judge_results.items():
        if not scores:
            continue
        vals = [scores.get(d, 0) for d in dims]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels + [labels[0]],
            fill="toself", name=pipeline,
            line_color=PIPELINE_COLORS.get(pipeline),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True, title="LLM Judge — 6-Dimension Scores",
        height=420, margin=dict(t=50, b=20),
    )
    return fig


def latency_bar(metrics: dict) -> go.Figure:
    pipelines = list(metrics.keys())
    retrieval = [metrics[p].get("embed_ms", 0) for p in pipelines]
    llm_ms    = [metrics[p].get("llm_ms",   0) for p in pipelines]
    fig = go.Figure(data=[
        go.Bar(name="Retrieval", x=pipelines, y=retrieval,
               marker_color="#FFA15A"),
        go.Bar(name="LLM Generation", x=pipelines, y=llm_ms,
               marker_color=[PIPELINE_COLORS.get(p, "#636EFA") for p in pipelines]),
    ])
    fig.update_layout(barmode="stack", title="Latency Breakdown (ms)",
                      yaxis_title="ms", height=350, margin=dict(t=50, b=20))
    return fig


def token_bar(metrics: dict) -> go.Figure:
    pipelines = list(metrics.keys())
    tok_in  = [metrics[p].get("actual_in",  0) for p in pipelines]
    tok_out = [metrics[p].get("actual_out", 0) for p in pipelines]
    fig = go.Figure(data=[
        go.Bar(name="Input tokens",  x=pipelines, y=tok_in,
               marker_color=[PIPELINE_COLORS.get(p) for p in pipelines]),
        go.Bar(name="Output tokens", x=pipelines, y=tok_out,
               marker_color="#AB63FA"),
    ])
    fig.update_layout(barmode="group", title="Token Usage",
                      yaxis_title="tokens", height=350, margin=dict(t=50, b=20))
    return fig


def cost_donut(metrics: dict) -> go.Figure:
    pipelines, costs = [], []
    for p, m in metrics.items():
        model = m.get("model", "gpt-4o-mini")
        c = estimate_cost(model, m.get("actual_in", 0), m.get("actual_out", 0))
        pipelines.append(p)
        costs.append(round(c * 1000, 4))   # in milli-dollars
    fig = go.Figure(go.Pie(
        labels=pipelines, values=costs,
        hole=0.45,
        marker_colors=[PIPELINE_COLORS.get(p) for p in pipelines],
    ))
    fig.update_layout(title="Estimated Cost (milli-USD per query)",
                      height=350, margin=dict(t=50, b=20))
    return fig



# ---------------------------------------------------------------------------
# TAB 1 — Database Overview
# ---------------------------------------------------------------------------

def tab_overview():
    st.markdown("## Database Overview")

    # ── Plain-English summary ─────────────────────────────────────────────────
    st.markdown("### What is this database?")
    st.success(
        "This is a **synthetic financial transaction database** built for detecting money laundering. "
        "It contains **2 million bank transactions** made between **510,000+ accounts** across **30,000+ banks** "
        "during **September 2022**. Each transaction records who sent money, who received it, how much, "
        "in what currency, and via what payment method."
    )

    col_q, col_nq = st.columns(2)
    with col_q:
        st.markdown("#### ✅ What you CAN query")
        st.markdown("""
- **Suspicious money patterns** — layering, smurfing, structuring, circular transfers
- **Specific payment types** — Cash, Wire, ACH, Cheque, Credit Card
- **Account behavior** — accounts receiving from many sources, pass-through accounts, dormant-then-active
- **Cross-bank flows** — funds moving through multiple banks or jurisdictions
- **Network investigation** — who an account sends to, who connects to them (up to 4 hops)
- **High-value transactions** — amounts from \\$1 to \\$1.28 billion
- **Currency patterns** — USD, EUR, GBP, JPY, INR, RUB, BTC, ETH and more
        """)
    with col_nq:
        st.markdown("#### ❌ What you CANNOT query")
        st.markdown("""
- **Real-time data** — dataset is fixed at September 2022, no live updates
- **Dates outside Sep 2022** — queries like "last week" mean last week of Sep 2022
- **Named individuals** — accounts use anonymous hex IDs, no real names
- **Account balances** — only transaction flows are stored, not balances
- **Geographic locations** — no country/city data, only bank IDs
- **Non-financial data** — no KYC, emails, phone numbers, or identity documents
        """)

    st.markdown("#### How to get the best results")
    st.info(
        "**GraphRAG** needs an account ID (9-char hex like `8042DF7E0`) as its starting point — "
        "it then explores that account's full transaction network. "
        "**Basic RAG** works from a plain text description — just describe the suspicious pattern. "
        "**LLM-only** gives general AML guidance when you don't have a specific account."
    )

    st.markdown("---")
    st.markdown("### Technical Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### TigerGraph — Knowledge Graph")
        st.markdown("""
**Graph name:** `FinancialGraph`
**Dataset:** HI-Small AML Benchmark (2,000,000 transactions)

#### Vertex Types (Nodes)
| Type | Count | Description |
|------|-------|-------------|
| Account | ~510,511 | Financial accounts (hex IDs) |
| Transaction | 2,000,000 | Individual money movements |
| Bank | ~30,528 | Financial institutions |
| Entity | ~165,809 | Named businesses / persons |

#### Edge Types (Relationships)
| Type | Direction | Meaning |
|------|-----------|---------|
| TRANSFERRED_TO | Account → Account | Direct money transfer |
| INITIATED_TRANSACTION | Account → Transaction | Sender of transaction |
| RECEIVED_TRANSACTION | Account → Transaction | Receiver of transaction |
| BELONGS_TO_BANK | Account → Bank | Account's home institution |

#### Graph Traversal
- BFS from seed account up to **4 hops** deep
- Typical subgraph: **46 accounts**, **161 transactions**, **28 banks**
- Retrieval time: **~2–4 seconds**
        """)

    with col2:
        st.markdown("### ChromaDB — Vector Store")
        st.markdown("""
**Collection:** `financial_transactions`
**Embedding model:** `all-MiniLM-L6-v2` (384-dim)
**Total chunks:** 2,000,000

#### What's Embedded
Each chunk is a natural-language narrative of one transaction:
> *"Account 8042DF7E0 (Corporation #6579, bank 01674) sent $1,285,209 Rupee to account 80A4D6EB0 via Cheque on 2022-09-01. Cross-bank transfer. Not flagged."*

#### Retrieval
- Query embedded on-the-fly (300–600ms)
- Top-5 most similar chunks returned
- Feeds directly into Basic RAG pipeline

---

### Account ID Format
Accounts use **hex-string IDs** (9 chars): `8042DF7E0`

#### Queryable Range
| Field | Range |
|-------|-------|
| Transaction dates | 2022-09-01 to 2023-03-31 |
| Amount range | ~\$1 to \$1,285,209,696 |
| Currencies | USD, EUR, GBP, JPY, INR, RUB, BTC, ETH + more |
| Payment formats | Wire, Cheque, ACH, Credit Card, Cash |
| Banks | 30,528 unique institutions |
        """)

    st.markdown("---")
    st.markdown("### Account Network Examples")
    df_accts = pd.DataFrame(KNOWN_ACCOUNTS, columns=["Account ID", "Entity", "Network Summary"])
    st.dataframe(df_accts, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Pipeline Architecture")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.info("**LLM-only**\n\nQuery → LLM\n\n*No retrieval. Uses parametric knowledge only. Baseline.*")
    with col4:
        st.warning("**Basic RAG**\n\nQuery → ChromaDB embed → Top-5 chunks → LLM\n\n*Vector similarity retrieval. Fast but misses graph structure.*")
    with col5:
        st.success("**GraphRAG**\n\nAccount → TigerGraph BFS → Subgraph → Compressed context → LLM\n\n*Structural retrieval. Exposes multi-hop transfer chains.*")


# ---------------------------------------------------------------------------
# TAB 2 — Query Interface
# ---------------------------------------------------------------------------

def tab_query():
    st.markdown("## Run Investigation Query")

    _BEST_ICON = {"GraphRAG": "🟢", "Basic RAG": "🔵", "Any": "⚪"}

    # ── Query selector ────────────────────────────────────────────────────
    st.markdown("### Select a Query")

    qs_col, legend_col = st.columns([2, 2])
    with qs_col:
        cat_options = ["All"] + sorted(set(q["category"] for q in CURATED_QUERIES))
        cat_sel = st.selectbox("Category", cat_options, key="cat_sel")
    with legend_col:
        st.markdown(
            "<div style='padding-top:28px'>🟢 GraphRAG best &nbsp;&nbsp; 🔵 Basic RAG best &nbsp;&nbsp; ⚪ Any pipeline</div>",
            unsafe_allow_html=True,
        )

    filtered_q = CURATED_QUERIES if cat_sel == "All" else [
        q for q in CURATED_QUERIES if q["category"] == cat_sel
    ]
    radio_labels = [f"{_BEST_ICON.get(q['best'], '⚪')}  {q['query']}" for q in filtered_q]
    label_to_query = {label: q["query"] for label, q in zip(radio_labels, filtered_q)}

    selected_label = st.radio(
        "Query selection",
        radio_labels,
        key="query_radio",
        label_visibility="collapsed",
    )

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        if st.button("↓ Use Selected Query", key="use_sel_query", use_container_width=True):
            if selected_label and selected_label in label_to_query:
                st.session_state["query_text"] = label_to_query[selected_label]
                st.rerun()

    st.markdown("---")

    # ── Query input + Account ID ──────────────────────────────────────────
    col_q, col_a = st.columns([3, 1])

    with col_q:
        query = st.text_area(
            "Investigation Query",
            value=st.session_state.get("query_text", ""),
            height=100,
            placeholder="Select a query above or type your own...",
            key="query_input",
        )

    with col_a:
        st.markdown("**GraphRAG Seed Account**")
        acct_search = st.text_input(
            "Search accounts",
            placeholder="🔍 Filter by ID or name...",
            key="acct_search",
            label_visibility="collapsed",
        )

        s = acct_search.lower()
        filtered_accounts = [
            a for a in KNOWN_ACCOUNTS
            if not s or s in a[0].lower() or s in a[1].lower() or s in a[2].lower()
        ]

        acct_options = [f"{a[0]}  —  {a[1]}" for a in filtered_accounts] + ["✏️ Custom ID"]

        selected_acct = st.radio(
            "Account",
            acct_options,
            key="acct_radio",
            label_visibility="collapsed",
        )

        if selected_acct == "✏️ Custom ID":
            account_id = st.text_input(
                "Account ID",
                value="",
                placeholder="e.g. 8042DF7E0",
                key="custom_acct_id",
                label_visibility="collapsed",
            )
        else:
            account_id = selected_acct.split("  —  ")[0].strip()
            matched = next((a for a in KNOWN_ACCOUNTS if a[0] == account_id), None)
            if matched:
                st.caption(matched[2])

    # ── API Configuration ─────────────────────────────────────────────────
    st.markdown("### API Configuration")
    api_col1, api_col2, api_col3, api_col4 = st.columns(4)

    with api_col1:
        st.markdown("**LLM-only**")
        lo_backend = st.selectbox("Backend", ["Gemini", "OpenAI"], key="lo_backend")
        if lo_backend == "Gemini":
            lo_model = st.selectbox("Model", GEMINI_MODELS, key="lo_model")
        else:
            lo_model = st.selectbox("Model", OPENAI_MODELS, key="lo_model2")

    with api_col2:
        st.markdown("**Basic RAG**")
        br_backend = st.selectbox("Backend", ["Gemini", "OpenAI"], key="br_backend")
        if br_backend == "Gemini":
            br_model = st.selectbox("Model", GEMINI_MODELS, key="br_model")
        else:
            br_model = st.selectbox("Model", OPENAI_MODELS, key="br_model2")

    with api_col3:
        st.markdown("**GraphRAG**")
        gr_backend = st.selectbox("Backend", ["Gemini", "OpenAI"], key="gr_backend")
        if gr_backend == "Gemini":
            gr_model = st.selectbox("Model", GEMINI_MODELS, key="gr_model")
        else:
            gr_model = st.selectbox("Model", OPENAI_MODELS, key="gr_model2")
        gr_depth = st.slider("BFS Depth", 1, 4, 3, key="gr_depth")

    with api_col4:
        st.markdown("**LLM Judge**")
        judge_backend = st.selectbox("Judge Backend", ["Groq", "OpenAI", "Gemini"], key="judge_backend")
        if judge_backend == "Groq":
            judge_model = st.selectbox("Model", GROQ_MODELS, key="judge_model")
        elif judge_backend == "OpenAI":
            judge_model = st.selectbox("Model", OPENAI_MODELS, key="judge_model2")
        else:
            judge_model = st.selectbox("Model", GEMINI_MODELS, key="judge_model3")
        run_bertscore_flag = st.checkbox("Run BERTScore", value=True, key="run_bs")

    # ── Run button ────────────────────────────────────────────────────────
    st.markdown("---")
    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("🚀 Run All Pipelines", type="primary", use_container_width=True)
    with info_col:
        st.caption("Runs LLM-only (~15s) + Basic RAG (~120s) + GraphRAG (~20s) in sequence. Judge + BERTScore added after.")

    DEFAULT_ACCOUNT = "8042DF7E0"

    if run_clicked:
        if not query.strip():
            st.error("Please enter a query or select an example above.")
            return

        # Auto-fill empty account ID with default
        effective_account = account_id.strip() or DEFAULT_ACCOUNT
        if not account_id.strip():
            st.warning(f"GraphRAG Seed Account was empty — using default account **{DEFAULT_ACCOUNT}**.")

        # Warn if the query references a specific hex account ID that differs from the seed
        _hex_in_query = re.findall(r'\b[0-9A-Fa-f]{7,9}\b', query)
        if _hex_in_query:
            _mismatched = [h.upper() for h in _hex_in_query if h.upper() != effective_account.upper()]
            if _mismatched:
                st.warning(
                    f"Your query mentions account **{_mismatched[0]}** but the GraphRAG seed is "
                    f"**{effective_account}**. GraphRAG traverses from the seed — it will not have "
                    f"data about {_mismatched[0]}. Set the seed to **{_mismatched[0]}** for accurate results.",
                    icon="⚠️",
                )

        openai_key = os.getenv("OPENAI_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        groq_key   = os.getenv("GROQ_API_KEY",   "")

        # Build pipeline args
        backend_map = {"Gemini": "gemini", "OpenAI": "openai"}

        def _model_args(backend: str, model: str) -> list:
            if backend == "Gemini":
                return ["--gemini_model", model]
            else:
                return ["--openai_model", model]

        lo_args  = ["--model", backend_map[lo_backend]] + _model_args(lo_backend, lo_model)
        br_args  = ["--model", backend_map[br_backend], "--top_k", "5", "--max_context_chunks", "5"] + _model_args(br_backend, br_model)
        gr_args  = ["--model", backend_map[gr_backend], "--account", effective_account, "--depth", str(gr_depth)] + _model_args(gr_backend, gr_model)

        pipelines = [
            ("LLM-only",  LLM_ONLY_SCRIPT,  lo_args),
            ("Basic RAG", BASIC_RAG_SCRIPT, br_args),
            ("GraphRAG",  GRAPH_RAG_SCRIPT, gr_args),
        ]

        results = {}
        progress = st.progress(0, text="Starting pipelines...")

        for idx, (name, script, args) in enumerate(pipelines):
            progress.progress((idx) / len(pipelines), text=f"Running {name}...")
            with st.spinner(f"Running {name} ({backend_map.get(lo_backend if name=='LLM-only' else br_backend if name=='Basic RAG' else gr_backend, '?')})..."):
                stdout, stderr, wall_ms = run_pipeline(script, query, args)
            metrics = parse_metrics(stdout, name)
            answer  = parse_answer(stdout)
            results[name] = {
                "metrics": metrics,
                "answer":  answer,
                "wall_ms": wall_ms,
                "stderr":  stderr[:300] if stderr.strip() else "",
            }
            progress.progress((idx + 1) / len(pipelines), text=f"{name} done.")

        # Judge
        progress.progress(0.9, text="Running LLM Judge...")
        judge_results = {}
        for name, data in results.items():
            if data["answer"]:
                j = run_judge(query, data["answer"], judge_backend, judge_model,
                              openai_key, groq_key, gemini_key, pipeline=name)
                judge_results[name] = j
            else:
                judge_results[name] = None

        # BERTScore — reference-based evaluation for investigative queries.
        # Graph-native queries (counts, rankings, hex IDs) are skipped entirely.
        # For investigative queries, all 3 pipelines are scored vs the same
        # expert-written reference answer (P, R, F1).
        bs_scores    = {}   # {pipeline: {"P": float, "R": float, "F1": float}}
        bs_skipped   = {}   # {pipeline: reason string}
        bs_ref_category = ""
        bs_query_type   = "skip"
        if run_bertscore_flag:
            progress.progress(0.95, text="Computing BERTScore...")
            try:
                from bert_score import score as _bs_check  # noqa: F401
                bs_available = True
            except ImportError:
                bs_available = False
                bs_query_type = "not-installed"

            if bs_available:
                ref_answers   = load_reference_answers()
                bs_query_type = _classify_for_bertscore(query)
                if bs_query_type == "investigative":
                    ref_category, ref_text = _find_reference_for_query(query, ref_answers)
                    if ref_text:
                        bs_ref_category = ref_category
                        # LLM-only excluded: no retrieved evidence — vocab overlap is trivial.
                        # EVIDENCE GAP answers excluded: the disclaimer text itself uses
                        # AML vocabulary that inflates BERTScore without any real content.
                        for name in ["Basic RAG", "GraphRAG"]:
                            ans_text = results.get(name, {}).get("answer", "").strip()
                            if not ans_text:
                                continue
                            if "EVIDENCE GAP" in ans_text[:200].upper():
                                bs_skipped[name] = "EVIDENCE GAP declared — answer contains no real investigative content; score hidden to avoid misleading comparison"
                                continue  # do not compute or store score for this pipeline
                            Ps, Rs, F1s = compute_bertscore([ans_text], [ref_text])
                            if Ps[0] is not None:
                                bs_scores[name] = {"P": Ps[0], "R": Rs[0], "F1": F1s[0]}
                    else:
                        bs_query_type = "no-reference"

        progress.progress(1.0, text="Done!")

        # Detect Gemini -> OpenAI fallbacks per pipeline
        selected_backends = {
            "LLM-only":  lo_backend,
            "Basic RAG": br_backend,
            "GraphRAG":  gr_backend,
        }
        fallbacks = {}
        for name, data in results.items():
            actual = data["metrics"].get("model", "")
            chosen = selected_backends[name]
            if chosen == "Gemini" and actual and "gpt" in actual.lower():
                fallbacks[name] = actual

        st.session_state["run_results"]        = results
        st.session_state["judge_results"]      = judge_results
        st.session_state["bs_scores"]          = bs_scores
        st.session_state["bs_skipped"]         = bs_skipped
        st.session_state["bs_ref_category"]    = bs_ref_category
        st.session_state["bs_query_type"]      = bs_query_type
        st.session_state["last_query"]         = query
        st.session_state["selected_backends"]  = selected_backends
        st.session_state["fallbacks"]          = fallbacks
        st.session_state["judge_info"]         = f"{judge_backend} / {judge_model}"
        st.session_state["ran"]                = True
        st.success("All pipelines complete! See Results tab.")


# ---------------------------------------------------------------------------
# TAB 3 — Results & Evaluation
# ---------------------------------------------------------------------------

def tab_results():
    if not st.session_state.get("ran"):
        st.info("Run a query in the **Query** tab first.")
        return

    results           = st.session_state["run_results"]
    judge_results     = st.session_state["judge_results"]
    bs_scores         = st.session_state.get("bs_scores", {})
    bs_skipped        = st.session_state.get("bs_skipped", {})
    bs_ref_category   = st.session_state.get("bs_ref_category", "")
    bs_query_type     = st.session_state.get("bs_query_type", "skip")
    query             = st.session_state.get("last_query", "")
    fallbacks         = st.session_state.get("fallbacks", {})
    selected_backends = st.session_state.get("selected_backends", {})
    judge_info        = st.session_state.get("judge_info", "")

    st.markdown(f"## Results for Query")
    st.markdown(f"> *{query}*")
    if judge_info:
        st.caption(f"LLM Judge: {judge_info}")
    st.markdown("---")

    # ── Answer comparison table ───────────────────────────────────────────
    st.markdown("### Generated Answers")
    ans_cols = st.columns(3)
    for i, (name, data) in enumerate(results.items()):
        with ans_cols[i]:
            status = data["metrics"].get("status", "?")
            badge  = "✅" if status == "success" else "❌"
            actual_model = data["metrics"].get("model", "?")
            st.markdown(f"**{badge} {name}**")
            st.caption(f"Model: {actual_model} | Wall: {data['wall_ms']/1000:.1f}s")

            # Gemini fallback warning
            if name in fallbacks:
                st.warning(
                    f"**Gemini quota exhausted** — fell back to OpenAI ({fallbacks[name]}). "
                    "Select OpenAI directly to avoid this.",
                    icon="⚠️",
                )

            # Warn when LLM-only is shown — it has no real data
            if name == "LLM-only":
                st.warning(
                    "**No real transaction data.** This answer is based on general AML knowledge only. "
                    "Account IDs, amounts, and rankings (if any) are fabricated — do not use for investigation.",
                    icon="⚠️",
                )

            answer = data["answer"] or "*No answer generated*"
            st.text_area(f"answer_{name}", answer, height=280, label_visibility="collapsed")

    # ── Metrics table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Evaluation Metrics")

    rows = []
    for name, data in results.items():
        m   = data["metrics"]
        j   = judge_results.get(name) or {}
        bsp = bs_scores.get(name, {})
        model = m.get("model", "?")
        cost  = estimate_cost(model, m.get("actual_in", 0), m.get("actual_out", 0))
        rows.append({
            "Pipeline":        name,
            "Status":          m.get("status", "?"),
            "Model":           model,
            "Retrieval ms":    f"{m.get('embed_ms', 0):.0f}",
            "LLM ms":          f"{m.get('llm_ms', 0):.0f}",
            "Total ms":        f"{m.get('total_ms', 0):.0f}",
            "Wall s":          f"{data['wall_ms']/1000:.1f}",
            "Tokens in":       m.get("actual_in", 0),
            "Tokens out":      m.get("actual_out", 0),
            "Cost ($)":        f"{cost:.6f}",
            "Retrieved":       m.get("retrieved", 0),
            "BERT-P":    "N/A" if name == "LLM-only" else ("⚠️ GAP" if (name in bs_skipped and not bsp) else (f"{bsp['P']:.4f}"  if bsp else "—")),
            "BERT-R":    "N/A" if name == "LLM-only" else ("⚠️ GAP" if (name in bs_skipped and not bsp) else (f"{bsp['R']:.4f}"  if bsp else "—")),
            "BERT-F1":   "N/A" if name == "LLM-only" else ("⚠️ GAP" if (name in bs_skipped and not bsp) else (f"{bsp['F1']:.4f}" if bsp else "—")),
            "Evidence":        j.get("evidence_grounding",      "N/A"),
            "Relationships":   j.get("relationship_reasoning",  "N/A"),
            "Insight":         j.get("investigative_insight",   "N/A"),
            "Factual":         j.get("factual_correctness",     "N/A"),
            "Actionability":   j.get("actionability",           "N/A"),
            "Conciseness":     j.get("conciseness",             "N/A"),
            "Final Score":     j.get("overall",                 "N/A"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df.set_index("Pipeline"), use_container_width=True)

    # ── Judge justifications ──────────────────────────────────────────────
    just_rows = [
        {"Pipeline": n, "Score": j.get("overall", "?"), "Justification": j.get("justification", "")}
        for n, j in judge_results.items() if j and j.get("justification")
    ]
    if just_rows:
        st.caption("Judge reasoning:")
        st.dataframe(pd.DataFrame(just_rows).set_index("Pipeline"), use_container_width=True)

    # ── BERTScore vs Expert Reference ────────────────────────────────────
    st.markdown("---")
    st.markdown("### BERTScore — vs Expert Reference Answer")
    if bs_query_type == "not-installed":
        st.error(
            "**`bert_score` package not installed** — run `pip install bert-score` then restart the app.",
            icon="🚫",
        )
    elif bs_query_type == "graph-native":
        st.info(
            "**BERTScore not applied.** This query asks for numerical/structural output "
            "(counts, rankings, account IDs) — semantic similarity to a prose reference is not meaningful. "
            "Use LLM Judge scores above instead.",
            icon="ℹ️",
        )
    elif bs_query_type == "no-reference":
        st.warning(
            "No matching expert reference answer found for this query. "
            "BERTScore requires a human-written reference for the pattern type being queried.",
            icon="⚠️",
        )
    elif bs_scores:
        cat_display = bs_ref_category.replace("_", " ").title() if bs_ref_category else "matched"
        st.caption(
            f"Each pipeline's answer is compared against an expert-written AML reference "
            f"(category: **{cat_display}**). Precision = how much of the generated text overlaps the "
            "reference; Recall = how much of the reference is covered; F1 = harmonic mean."
        )
        bs_rows = []
        for pipe in ["LLM-only", "Basic RAG", "GraphRAG"]:
            bsp      = bs_scores.get(pipe, {})
            skip_msg = bs_skipped.get(pipe, "")
            if pipe == "LLM-only":
                bs_rows.append({
                    "Pipeline":       pipe,
                    "Precision":      "N/A",
                    "Recall":         "N/A",
                    "F1":             "N/A",
                    "Interpretation": "Excluded — no retrieved evidence; AML vocab overlap is trivial. LLM Judge penalises instead.",
                })
            else:
                if skip_msg:
                    interp = f"⚠️ Score hidden — {skip_msg}"
                elif bsp and bsp["F1"] >= 0.85:
                    interp = "Strong semantic overlap with expert reference"
                elif bsp and bsp["F1"] >= 0.75:
                    interp = "Moderate overlap"
                elif bsp:
                    interp = "Low overlap — answer diverges from reference"
                else:
                    interp = "Not computed"
                bs_rows.append({
                    "Pipeline":       pipe,
                    "Precision":      "—" if skip_msg else (f"{bsp['P']:.4f}" if bsp else "—"),
                    "Recall":         "—" if skip_msg else (f"{bsp['R']:.4f}" if bsp else "—"),
                    "F1":             "—" if skip_msg else (f"{bsp['F1']:.4f}" if bsp else "—"),
                    "Interpretation": interp,
                })
        st.dataframe(pd.DataFrame(bs_rows).set_index("Pipeline"), use_container_width=True)

        # Bar chart — show all computed scores; ⚠️ label marks unreliable ones
        valid_bs = {
            (p + " ⚠️" if p in bs_skipped else p): bs_scores[p]["F1"]
            for p in ["Basic RAG", "GraphRAG"] if p in bs_scores
        }
        if valid_bs:
            fig_bs = px.bar(
                x=list(valid_bs.keys()), y=list(valid_bs.values()),
                color=list(valid_bs.keys()),
                color_discrete_map=PIPELINE_COLORS,
                title="BERTScore F1 vs Expert Reference",
                labels={"x": "Pipeline", "y": "F1"},
                range_y=[0.0, 1.0],
            )
            fig_bs.update_layout(height=300, showlegend=False, margin=dict(t=50, b=20))
            st.plotly_chart(fig_bs, use_container_width=True)
    else:
        st.info("BERTScore not computed — check 'Run BERTScore' in the Query tab.", icon="ℹ️")

    # ── Visual Evaluations ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Visual Evaluation")

    v1, v2 = st.columns(2)
    with v1:
        metrics_dict = {n: d["metrics"] for n, d in results.items()}
        st.plotly_chart(latency_bar(metrics_dict), use_container_width=True)
    with v2:
        st.plotly_chart(token_bar(metrics_dict), use_container_width=True)

    v3, v4 = st.columns(2)
    with v3:
        valid_judge = {n: j for n, j in judge_results.items() if j}
        if valid_judge:
            st.plotly_chart(radar_chart(valid_judge), use_container_width=True)
        else:
            st.info("No judge scores available.")
    with v4:
        st.plotly_chart(cost_donut(metrics_dict), use_container_width=True)

    # ── Winner summary ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Benchmark Verdict")
    w1, w2, w3, w4 = st.columns(4)

    valid = {n: d for n, d in results.items() if d["metrics"].get("status") == "success"}
    if valid:
        fastest   = min(valid, key=lambda n: valid[n]["metrics"].get("total_ms", 9e9))
        cheapest  = min(valid, key=lambda n: estimate_cost(
            valid[n]["metrics"].get("model","gpt-4o-mini"),
            valid[n]["metrics"].get("actual_in",0),
            valid[n]["metrics"].get("actual_out",0)))
        least_tok = min(valid, key=lambda n: valid[n]["metrics"].get("actual_in", 9e9))

        with w1:
            st.metric("Fastest Pipeline", fastest,
                      f"{valid[fastest]['metrics'].get('total_ms',0):.0f} ms")
        with w2:
            st.metric("Fewest Input Tokens", least_tok,
                      f"{valid[least_tok]['metrics'].get('actual_in',0):,} tok")
        with w3:
            st.metric("Cheapest Pipeline", cheapest)

        judge_valid = {n: j for n, j in judge_results.items() if j and j.get("overall")}
        if judge_valid:
            best_judge = max(judge_valid, key=lambda n: judge_valid[n]["overall"])
            with w4:
                st.metric("Best Judge Score", best_judge,
                          f"{judge_valid[best_judge]['overall']}/10")

    # ── Error details ─────────────────────────────────────────────────────
    failed = {n: d for n, d in results.items() if d["metrics"].get("status") != "success"}
    if failed:
        st.markdown("---")
        st.markdown("### ❌ Pipeline Failures")
        for name, data in failed.items():
            with st.expander(f"**{name}** — {data['metrics'].get('error_type', 'unknown error')}", expanded=True):
                stderr = data.get("stderr", "")
                if stderr:
                    st.error(stderr[:600])
                if not data["answer"]:
                    st.warning("No answer was generated for this pipeline.")
                st.caption(f"Wall time: {data['wall_ms']/1000:.1f}s")
    elif any(d.get("stderr") for d in results.values()):
        with st.expander("⚠️ Pipeline Warnings"):
            for name, data in results.items():
                if data.get("stderr"):
                    st.markdown(f"**{name}:** `{data['stderr'][:400]}`")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Sidebar
    with st.sidebar:
        st.title("🔍 RAG Benchmark Lab")
        st.markdown("**AML Investigation Suite**")
        st.markdown("---")
        st.markdown("#### Quick Stats")
        st.metric("Transactions", "2,000,000")
        st.metric("Accounts", "510,511")
        st.metric("Banks", "30,528")
        st.metric("Embeddings", "2,000,000 chunks")
        st.markdown("---")
        st.markdown("#### Query Range Guide")
        st.info("Dataset covers **Sep 2022**. Temporal queries like 'last 3 days' are interpreted relative to 2022-09-30.")
        st.markdown("""
- **Dates:** 2022-09-01 to 2022-09-30
- **Amounts:** \\$1 to \\$1.28B
- **Currencies:** USD, EUR, GBP, JPY, INR, RUB, BTC, ETH
- **Payment types:** Wire, Cheque, ACH, Credit Card, Cash, Reinvestment
- **Account IDs:** 9-char hex (e.g. `8042DF7E0`)
        """)
        st.markdown("---")
        st.markdown("#### Pipeline Reliability")
        st.markdown("""
| Pipeline | Data source | Trust level |
|----------|-------------|-------------|
| **GraphRAG** | Real graph traversal | ✅ High — evidence-backed |
| **Basic RAG** | Real transactions | ✅ High — evidence-backed |
| **LLM-only** | General knowledge | ⚠️ Low — no real data |
        """)
        st.error("LLM-only answers contain **no real account IDs or amounts**. Use it for general AML guidance only, not for actual investigation.")
        st.markdown("---")
        st.markdown("#### AML Pattern Guide")
        st.markdown("""
| Pattern | Description |
|---------|-------------|
| Layering | Multi-hop transfers |
| Smurfing | Small transfers below threshold |
| Round-trip | Circular fund movement |
| Fan-out | 1 → many recipients |
| Fan-in | many → 1 recipient |
        """)
        st.markdown("---")
        st.caption("Powered by TigerGraph + ChromaDB + Gemini / GPT-4o-mini / Groq")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Database Overview", "🔎 Run Query", "📈 Results & Evaluation"])

    with tab1:
        tab_overview()
    with tab2:
        tab_query()
    with tab3:
        tab_results()


if __name__ == "__main__":
    if "ran" not in st.session_state:
        st.session_state["ran"] = False
    main()
