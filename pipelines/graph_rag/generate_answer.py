"""
generate_answer.py — GraphRAG Answer Generation Pipeline

Full GraphRAG pipeline: graph traversal ->subgraph context construction ->LLM reasoning.
Designed as the GraphRAG leg of the benchmark:

    LLM-only  vs  Basic RAG  vs  GraphRAG (this file)

--- GraphRAG architecture ---
Unlike vector RAG (which retrieves semantically similar text chunks), GraphRAG
traverses a structured knowledge graph to retrieve entity relationships and
transaction chains.  This produces richer, more connected context:

  1. RETRIEVE   — traverse TigerGraph from a seed account, collect entity subgraph
  2. AUGMENT    — convert graph nodes into a human-readable context narrative
  3. GENERATE   — inject graph narrative + query into an AML-aware prompt, call LLM

--- Why graph retrieval beats vector retrieval for AML ---
Money laundering exploits multi-hop transaction chains: A ->B ->C ->D.
Vector similarity search retrieves transactions that look like the query but
misses the structural pattern — that these accounts form a layering chain.
GraphRAG surfaces the chain explicitly:
  "Account A sent $X to B; B forwarded to C within 2 hours; C split into 3
   transfers to D, E, F."
The LLM can reason over this chain in a way flat text chunks cannot capture.

--- Token efficiency vs. Basic RAG ---
Graph context is structured and information-dense: a 5-hop subgraph with 20
nodes encodes relationships that would take hundreds of text chunks to cover in
a flat vector store.  This file handles the translation from graph nodes to
LLM-readable narrative.

--- Benchmark reliability guarantee ---
The GRAPH_RAG_METRICS line is emitted inside a try/finally block so it prints
even when any stage of the pipeline fails.  Failed runs carry status=failure
and error_type so the benchmark harness records them as data.
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
# Reuse retrieval layer from retrieve.py (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from retrieve import (
    connect_tigergraph,
    expand_account_subgraph,
    get_account_neighbors,
    get_high_risk_transactions,
    trace_transaction_paths,
    DEFAULT_DEPTH,
    DEFAULT_GRAPH,
    DEFAULT_HOST,
    DEFAULT_MAX_HOPS,
    DEFAULT_MIN_AMOUNT,
    DEFAULT_PASSWORD,
    DEFAULT_TOKEN,
    DEFAULT_USERNAME,
)

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

GEMINI_MODEL    = "gemini-2.5-flash"
OPENAI_MODEL    = "gpt-4o-mini"
OLLAMA_MODEL    = "llama3.2"
DEFAULT_BACKEND = "gemini"

MAX_CONTEXT_CHARS   = 5_500   # structural analysis adds fan-in/fan-out/pass-through sections
MAX_TX_DETAIL       = 6       # show top N transactions in full; rest summarised
MAX_ACCOUNTS_SHOWN  = 6       # inline account entries per hop group
MAX_CHAIN_SOURCES   = 5       # distinct source accounts in transfer-chain section
MAX_BANKS_INLINE    = 15      # bank IDs shown before truncation
CHARS_PER_TOKEN   = 4
MAX_RETRIES       = 3
BASE_BACKOFF_S    = 2.0
_RETRYABLE_CODES  = {429, 500, 503}

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
        description="GraphRAG — graph-traversal-augmented financial risk analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_answer.py --account ACC_12345 --query "Possible layering through multiple banks"
  python generate_answer.py --account ACC_12345 --method subgraph --depth 3
  python generate_answer.py --account ACC_12345 --method paths --max_hops 4 --min_amount 5000
  python generate_answer.py --account ACC_12345 --model ollama
        """,
    )
    p.add_argument("--query",   type=str, default=None,
                   help="AML investigation query. Omit for interactive input.")
    p.add_argument("--account", type=str, default=None,
                   help="Seed account ID to traverse from. Omit for interactive input.")
    p.add_argument("--method",  type=str, default="subgraph",
                   choices=["neighbors", "paths", "risk", "subgraph"],
                   help="Graph traversal method (default: subgraph).")
    p.add_argument("--depth",      type=int,   default=DEFAULT_DEPTH,
                   help=f"Subgraph traversal depth (default: {DEFAULT_DEPTH}).")
    p.add_argument("--max_hops",   type=int,   default=DEFAULT_MAX_HOPS,
                   help=f"Max hops for paths method (default: {DEFAULT_MAX_HOPS}).")
    p.add_argument("--min_amount", type=float, default=DEFAULT_MIN_AMOUNT,
                   help=f"Min transfer amount filter for paths (default: {DEFAULT_MIN_AMOUNT}).")
    p.add_argument("--model",      type=str,   default=DEFAULT_BACKEND,
                   choices=["gemini", "openai", "ollama"],
                   help=f"LLM backend (default: {DEFAULT_BACKEND}). Gemini auto-falls back to OpenAI then Ollama on quota exhaustion.")
    p.add_argument("--gemini_model", type=str, default=GEMINI_MODEL,
                   help=f"Gemini model name (default: {GEMINI_MODEL}).")
    p.add_argument("--openai_model", type=str, default=OPENAI_MODEL,
                   help=f"OpenAI model name (default: {OPENAI_MODEL}).")
    p.add_argument("--ollama_model", type=str, default=OLLAMA_MODEL,
                   help=f"Ollama model name (default: {OLLAMA_MODEL}).")
    p.add_argument("--max_retries",  type=int, default=MAX_RETRIES)
    p.add_argument("--host",     type=str, default=DEFAULT_HOST)
    p.add_argument("--username", type=str, default=DEFAULT_USERNAME)
    p.add_argument("--password", type=str, default=DEFAULT_PASSWORD)
    p.add_argument("--graph",    type=str, default=DEFAULT_GRAPH)
    p.add_argument("--token",    type=str, default=DEFAULT_TOKEN,
                   help="TigerGraph RESTPP API token (TigerGraph Cloud).")
    p.add_argument("--debug",    action="store_true",
                   help="Enable DEBUG logging (prints raw GSQL queries).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Graph context construction
# ---------------------------------------------------------------------------

def _fmt_account(node: dict) -> str:
    attr = node.get("attributes", {})
    v_id = node.get("v_id", "?")
    name = str(attr.get("entity_name", "")).strip() or "unnamed"
    bank = str(attr.get("home_bank",   "")).strip() or "unknown bank"
    return f"Account {v_id} ({name}, bank: {bank})"


def _fmt_transaction(node: dict) -> str:
    attr    = node.get("attributes", {})
    v_id    = node.get("v_id", "?")
    amt_p   = attr.get("amount_paid",          "?")
    curr_p  = attr.get("payment_currency",     "?")
    amt_r   = attr.get("amount_received",      "?")
    curr_r  = attr.get("receiving_currency",   "?")
    ptype   = attr.get("payment_type", attr.get("payment_format", "?"))
    ts      = attr.get("timestamp",            "?")
    flag    = attr.get("laundering_flag", attr.get("is_laundering", "0"))
    cb      = attr.get("is_cross_bank",        "0")
    cc      = attr.get("is_cross_currency",    "0")
    st      = attr.get("is_self_transfer",     "0")

    flag_str = "FLAGGED" if str(flag) in ("1", "True", "true") else "clean"
    tags = []
    if str(cb) in ("1", "True", "true"): tags.append("cross-bank")
    if str(cc) in ("1", "True", "true"): tags.append("cross-currency")
    if str(st) in ("1", "True", "true"): tags.append("self-transfer")
    tag_str = ", ".join(tags) if tags else "none"

    return (
        f"Transaction {v_id}: {amt_p} {curr_p} -> {amt_r} {curr_r}"
        f" via {ptype} at {ts} [{flag_str}; tags: {tag_str}]"
    )


def _fmt_bank(node: dict) -> str:
    attr    = node.get("attributes", {})
    v_id    = node.get("v_id", "?")
    name    = str(attr.get("bank_name", "")).strip() or v_id
    country = str(attr.get("country",   "")).strip()
    suffix  = f", {country}" if country else ""
    return f"Bank {v_id} ({name}{suffix})"


def _fmt_entity(node: dict) -> str:
    attr   = node.get("attributes", {})
    v_id   = node.get("v_id", "?")
    name   = str(attr.get("entity_name",  "")).strip() or v_id
    etype  = str(attr.get("entity_type",  "")).strip()
    suffix = f", type: {etype}" if etype else ""
    return f"Entity {v_id} ({name}{suffix})"


def _tx_priority_key(t: dict) -> tuple:
    """Sort key: flagged > cross-bank > cross-currency > amount desc."""
    attr   = t.get("attributes", {})
    flag   = str(attr.get("laundering_flag", attr.get("is_laundering", "0")))
    cb     = str(attr.get("is_cross_bank",    "0"))
    cc     = str(attr.get("is_cross_currency","0"))
    amount = float(attr.get("amount_paid", 0) or 0)
    return (
        0 if flag in ("1","True","true") else 1,
        0 if cb   in ("1","True","true") else 1,
        0 if cc   in ("1","True","true") else 1,
        -amount,
    )


_FORMAT_KEYWORDS: dict[str, str] = {
    "cash": "Cash", "wire": "Wire", "ach": "ACH",
    "cheque": "Cheque", "check": "Cheque",
    "credit card": "Credit Card", "reinvestment": "Reinvestment",
}


def _detect_query_format(query: str) -> str | None:
    q = query.lower()
    for kw, fmt in _FORMAT_KEYWORDS.items():
        if kw in q:
            return fmt
    return None


def build_graph_context(
    payload: dict,
    max_chars: int = MAX_CONTEXT_CHARS,
    query_format: str | None = None,
) -> tuple[str, int, int]:
    """
    Convert graph retrieval payload into a dense, token-efficient AML narrative.

    Design principle: structured summary > exhaustive lists.
    Every section is hard-capped so total context stays well under Basic RAG's
    token footprint while preserving the high-signal facts the LLM needs.
    """
    nodes   = payload.get("nodes", [])
    edges   = payload.get("edges", [])
    account = payload.get("query_account", "?")

    if not nodes:
        return (
            f"No graph data found for account {account}. "
            "Account may not exist or has no connected nodes.\n",
            0, 0,
        )

    accounts:     list[dict] = []
    transactions: list[dict] = []
    banks:        list[dict] = []

    for node in nodes:
        ntype = str(node.get("type") or node.get("v_type") or "").lower()
        if   "account"     in ntype: accounts.append(node)
        elif "transaction" in ntype: transactions.append(node)
        elif "bank"        in ntype: banks.append(node)

    transactions.sort(key=_tx_priority_key)

    def _is_flagged(t: dict) -> bool:
        attr = t.get("attributes", {})
        return str(attr.get("laundering_flag", attr.get("is_laundering", "0"))) in ("1","True","true")

    flagged_tx    = sum(1 for t in transactions if _is_flagged(t))
    cross_bank_tx = sum(1 for t in transactions if str(t.get("attributes",{}).get("is_cross_bank","0"))    in ("1","True","true"))
    cross_curr_tx = sum(1 for t in transactions if str(t.get("attributes",{}).get("is_cross_currency","0")) in ("1","True","true"))
    traversal_depth = payload.get("metrics", {}).get("traversal_depth", 0)

    seed_nodes  = [n for n in accounts if n.get("hop") == 0 or n.get("role") == "seed"]
    if not seed_nodes and accounts:
        seed_nodes = accounts[:1]
    seed = seed_nodes[0] if seed_nodes else {}
    seed_attr = seed.get("attributes", {})
    seed_id   = seed.get("v_id", account)
    seed_name = str(seed_attr.get("entity_name", "")).strip() or "unknown"
    seed_bank = str(seed_attr.get("home_bank",   "")).strip() or "unknown"

    transfer_edges = [e for e in edges if e.get("edge_type") == "TRANSFERRED_TO"]

    lines: list[str] = []

    # ── Header: everything the LLM needs at a glance ──────────────────────
    lines.append(
        f"SUBGRAPH[{seed_id}] {seed_name} | bank:{seed_bank} | {traversal_depth}-hop network"
    )
    lines.append(
        f"Network: {len(accounts)} accts | {len(transactions)} txns | {len(banks)} banks"
        f" | {flagged_tx} FLAGGED | {cross_bank_tx} cross-bank | {cross_curr_tx} cross-currency"
    )
    lines.append("")

    # ── Structural Pattern Analysis (computed from edges) ─────────────────
    if transfer_edges:
        # fan-in : receiver  -> set of unique senders
        # fan-out: sender    -> set of unique receivers
        fan_in_map:  dict[str, set[str]] = {}
        fan_out_map: dict[str, set[str]] = {}
        for e in transfer_edges:
            src, tgt = e["from_id"], e["to_id"]
            fan_in_map.setdefault(tgt, set()).add(src)
            fan_out_map.setdefault(src, set()).add(tgt)

        fan_in_ranked  = sorted(fan_in_map.items(),  key=lambda x: len(x[1]), reverse=True)
        fan_out_ranked = sorted(fan_out_map.items(), key=lambda x: len(x[1]), reverse=True)

        # pass-through: appears in BOTH fan-in and fan-out maps
        pass_through = [
            (a, len(fan_in_map[a]), len(fan_out_map[a]))
            for a in fan_in_map if a in fan_out_map
        ]
        pass_through.sort(key=lambda x: x[1] + x[2], reverse=True)

        # circular pairs: A->B and B->A both exist
        circular = [
            (a, b) for a, receivers in fan_out_map.items()
            for b in receivers
            if a in fan_in_map.get(b, set()) and a < b
        ]

        lines.append(f"STRUCTURAL PATTERNS ({len(transfer_edges)} edges):")

        # Fan-in: who receives from the most unique sources?
        lines.append(f"  Fan-in — top receivers by unique-sender count:")
        for acct_id, senders in fan_in_ranked[:5]:
            sample = ", ".join(list(senders)[:3]) + ("..." if len(senders) > 3 else "")
            lines.append(f"    {acct_id}: {len(senders)} senders [{sample}]")

        lines.append(f"  Fan-out — top senders by unique-target count:")
        for acct_id, receivers in fan_out_ranked[:5]:
            sample = ", ".join(list(receivers)[:3]) + ("..." if len(receivers) > 3 else "")
            lines.append(f"    {acct_id}: {len(receivers)} targets [{sample}]")

        if pass_through:
            lines.append(f"  Pass-through (both receive+send, top 3):")
            for acct_id, in_deg, out_deg in pass_through[:3]:
                lines.append(f"    {acct_id}: in={in_deg} | out={out_deg}")

        if circular:
            lines.append(f"  Circular pairs ({len(circular)} total, sample):")
            for a, b in circular[:3]:
                lines.append(f"    {a} <-> {b}")

        lines.append("")

    # ── Top transactions: compact single-line, priority sorted ────────────
    if transactions:
        lines.append(
            f"TOP TRANSACTIONS ({len(transactions)} total | showing top {min(MAX_TX_DETAIL, len(transactions))}"
            f" | {flagged_tx} flagged | {cross_bank_tx} cross-bank):"
        )
        for t in transactions[:MAX_TX_DETAIL]:
            attr  = t.get("attributes", {})
            amt   = attr.get("amount_paid",        "?")
            cp    = attr.get("payment_currency",   "?")
            cr    = attr.get("receiving_currency",  "?")
            ptype = attr.get("payment_format", attr.get("payment_type", attr.get("transfer_type", "?")))
            ts    = str(attr.get("timestamp", "?"))[:10]
            flag  = str(attr.get("laundering_flag", attr.get("is_laundering","0")))
            cb    = str(attr.get("is_cross_bank","0"))    in ("1","True","true")
            cc    = str(attr.get("is_cross_currency","0")) in ("1","True","true")
            tags  = ("FLAGGED " if flag in ("1","True","true") else "") + \
                    ("CB " if cb else "") + ("CC" if cc else "")
            curr_str = f"{cp}->{cr}" if cp != cr else cp
            lines.append(f"  ${amt} {curr_str} | {ptype} | {ts} | {tags.strip() or 'clean'}")

        remaining = len(transactions) - MAX_TX_DETAIL
        if remaining > 0:
            rem_flagged = sum(1 for t in transactions[MAX_TX_DETAIL:] if _is_flagged(t))
            lines.append(f"  +{remaining} more txns ({rem_flagged} flagged, highest-priority shown above)")
        lines.append("")

    # ── Payment-format-specific fan-out (when query targets a format) ────
    if query_format and transactions:
        fmt_txs = [
            t for t in transactions
            if str(t.get("attributes", {}).get(
                "payment_format",
                t.get("attributes", {}).get("payment_type", "")
            )).lower() == query_format.lower()
        ]
        if fmt_txs:
            fmt_tx_ids = {t.get("v_id") for t in fmt_txs}
            tx_sender:   dict[str, str] = {}
            tx_receiver: dict[str, str] = {}
            for e in edges:
                etype  = e.get("edge_type", "")
                fid, tid = e.get("from_id", ""), e.get("to_id", "")
                if etype == "INITIATED_TRANSACTION":
                    if tid in fmt_tx_ids:
                        tx_sender[tid] = fid       # account -> tx
                elif etype == "RECEIVED_TRANSACTION":
                    if fid in fmt_tx_ids:
                        tx_receiver[fid] = tid     # tx -> account
                    elif tid in fmt_tx_ids:
                        tx_receiver[tid] = fid     # account -> tx (reverse schema)

            fmt_fan_out: dict[str, set[str]] = {}
            for tx_id in fmt_tx_ids:
                sender   = tx_sender.get(tx_id)
                receiver = tx_receiver.get(tx_id)
                if sender and receiver and sender != receiver:
                    fmt_fan_out.setdefault(sender, set()).add(receiver)

            fmt_ranked = sorted(fmt_fan_out.items(), key=lambda x: len(x[1]), reverse=True)
            lines.append(f"{query_format.upper()} TRANSACTION ANALYSIS ({len(fmt_txs)} txns in subgraph):")
            if fmt_ranked:
                lines.append(f"  Accounts sending {query_format} to most unique recipients:")
                for acct, receivers in fmt_ranked[:5]:
                    lines.append(f"    {acct}: {len(receivers)} unique {query_format} recipients")
                above5 = [(a, r) for a, r in fmt_ranked if len(r) > 5]
                if above5:
                    lines.append(f"  Accounts with MORE THAN 5 unique {query_format} recipients:")
                    for acct, receivers in above5:
                        lines.append(f"    {acct}: {len(receivers)} recipients")
            else:
                lines.append(f"  No account-to-account {query_format} transfers found via transaction edges.")
                lines.append(f"  NOTE: {len(fmt_txs)} {query_format} transactions exist in this subgraph")
                lines.append(f"  but sender/receiver edge data is insufficient for fan-out ranking.")
            lines.append("")
        else:
            lines.append(f"NOTE: No {query_format} transactions found in the {seed_id} subgraph.")
            lines.append(f"  Try a different seed account or broader BFS depth.")
            lines.append("")

    # ── Banks: one compact line ───────────────────────────────────────────
    if banks:
        bank_ids = [b.get("v_id", "?") for b in banks]
        shown_bk = ", ".join(bank_ids[:MAX_BANKS_INLINE])
        extra_bk = f" +{len(bank_ids) - MAX_BANKS_INLINE} more" if len(bank_ids) > MAX_BANKS_INLINE else ""
        lines.append(f"BANKS ({len(banks)}): {shown_bk}{extra_bk}")
        lines.append("")

    context = "\n".join(lines)

    if len(context) > max_chars:
        context = context[:max_chars] + "\n[context truncated]\n"
        log.warning("Graph context truncated to %d characters.", max_chars)

    return context, len(nodes), len(edges)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

DATASET_DATE_NOTE = (
    "DATASET CONTEXT: The transaction database covers 2022-09-01 to 2022-09-30. "
    "Interpret temporal references ('last 3 days', 'recent', 'today') "
    "relative to 2022-09-30 (the dataset end date)."
)


def build_prompt(query: str, account: str, context: str) -> str:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from query_intent import classify_query
    qtype = classify_query(query)

    if qtype == "factual":
        instructions = f"""INSTRUCTIONS:
- Answer the query DIRECTLY from the graph evidence below — do NOT write a risk analysis report.
- If asked for a top-N ranking, extract the relevant metric (transfer count, receiver count, etc.)
  from the STRUCTURAL PATTERNS section and list the accounts ranked by that metric.
- Present results as a numbered list. Include account IDs and the exact metric values.
- If the requested criterion (e.g. high-volume pass-through accounts) is NOT found in the evidence,
  explicitly state that: "No accounts matching this criterion were found in the {account} network.
  The network contains only N accounts with the following connectivity..."
- If the query cannot be fully answered from this subgraph alone, provide the best available
  answer and note that results are scoped to the {account} network."""
        response_header = "DIRECT ANSWER:"
    else:
        instructions = """WRITE A FLOWING AML INVESTIGATION REPORT — not a structured template.
Use natural investigation prose. Every sentence must cite real data from the graph evidence.

COVER THESE IN ORDER (paragraph form, not headers/bullets):

1. NETWORK OVERVIEW: State total accounts, transactions, banks, flagged count, and
   cross-bank count from the evidence header in one opening sentence.

2. CONFIRMED PATTERNS: Name each laundering pattern found (layering / circular /
   pass-through / fan-out). For circular pairs, write: "Confirmed round-trip between
   Account A and Account B — funds returned to the originating cluster."
   For pass-through accounts, write: "Account X is a confirmed pass-through intermediary
   with N unique inbound senders and M unique outbound recipients."

3. LAUNDERING PHASE: State whether this is placement, layering, or integration phase,
   justified by the network structure.

4. RISK LEVEL: State HIGH / MEDIUM / LOW with 3 specific facts from the evidence
   (account IDs, exact counts, confirmed pattern names).

5. RECOMMENDED ACTIONS: 3–4 sentences naming specific account IDs to investigate,
   regulatory steps (SAR filing, enhanced due diligence, MLRO escalation), and
   what additional data would confirm the suspicion."""
        response_header = "AML INVESTIGATION REPORT:"

    return f"""You are a financial analyst with expertise in AML and transaction network analysis.
You have been given a GRAPH-STRUCTURED evidence package showing the transaction network
around account {account}, retrieved by traversing a financial knowledge graph.

{DATASET_DATE_NOTE}

STRICT RULES:
- Base your answer ONLY on the graph evidence provided below.
- Do NOT invent account numbers, amounts, or patterns not present in the evidence.
- STRUCTURAL PATTERNS (circular pairs, pass-through accounts, fan-in/fan-out rankings,
  cross-bank counts) are graph-computed facts — state them as CONFIRMED FINDINGS,
  not hypotheses. Write "Account X is a confirmed pass-through", not "may indicate".
- Reserve hedges ("insufficient data", "cannot confirm") ONLY for details that are
  genuinely absent from the evidence — e.g., who controls an account, or the
  purpose behind a transaction. Do NOT hedge on what the graph explicitly reports.
- Do NOT write "cannot be confidently traced" or "remains speculative" for any
  pattern the STRUCTURAL PATTERNS section explicitly lists.

{instructions}

--- RETRIEVED GRAPH EVIDENCE ---
{context}
--- END OF GRAPH EVIDENCE ---

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

    client   = genai.Client(api_key=api_key)
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
    method: str,
    account: str,
    nodes: int,
    edges: int,
    retrieval_ms: float,
    llm_ms: float,
    actual_in: int,
    actual_out: int,
    traversal_depth: int,
    status: str,
    error_type: str,
) -> None:
    """
    Emit the machine-readable benchmark metrics line.
    Called from the finally block — guaranteed to execute even on failure.
    """
    total_ms = retrieval_ms + llm_ms
    log.info(
        "GRAPH_RAG_METRICS | backend=%s | model=%s | method=%s | account=%s "
        "| nodes=%d | edges=%d "
        "| retrieval_ms=%.1f | llm_ms=%.1f | total_ms=%.1f "
        "| prompt_tokens=%d | response_tokens=%d "
        "| traversal_depth=%d | status=%s | error_type=%s",
        backend, model_name, method, account,
        nodes, edges,
        retrieval_ms, llm_ms, total_ms,
        actual_in, actual_out,
        traversal_depth,
        status, error_type or "none",
    )
    sys.stdout.flush()
    sys.stderr.flush()


def print_results(
    query: str,
    account: str,
    backend: str,
    model_name: str,
    method: str,
    nodes: int,
    edges: int,
    traversal_depth: int,
    retrieval_ms: float,
    llm_ms: float,
    actual_in: int,
    actual_out: int,
    answer: str,
) -> None:
    total_ms = retrieval_ms + llm_ms
    width = 70
    print("\n" + "=" * width)
    print("  GRAPH RAG — GENERATION RESULTS")
    print("=" * width)
    print(f"  Query         : {query}")
    print(f"  Seed account  : {account}")
    print(f"  Method        : {method}  (depth/hops: {traversal_depth})")
    print(f"  Backend       : {backend} ({model_name})")
    print(f"  Graph nodes   : {nodes}")
    print(f"  Graph edges   : {edges}")
    print("-" * width)
    print("  LATENCY")
    print(f"    Graph retrieval : {retrieval_ms:>8.2f} ms")
    print(f"    LLM generation  : {llm_ms:>8.2f} ms")
    print(f"    Total           : {total_ms:>8.2f} ms")
    print("-" * width)
    print("  TOKENS")
    if actual_in:
        print(f"    Actual input  : {actual_in:>8,}  (from API)")
    if actual_out:
        print(f"    Actual output : {actual_out:>8,}  (from API)")
    if not actual_in and not actual_out:
        print("    Token counts  : unavailable (Ollama or API error)")
    print("=" * width)
    print("\n  GENERATED ANSWER\n")
    try:
        print(answer.strip())
    except UnicodeEncodeError:
        print(answer.strip().encode(sys.stdout.encoding or "utf-8", errors="replace")
              .decode(sys.stdout.encoding or "utf-8"))
    print("\n" + "=" * width)


# ---------------------------------------------------------------------------
# Main — with guaranteed metrics emission in finally
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Failure-default state — populated as each pipeline stage succeeds.
    _backend      = args.model
    _model_name   = (args.gemini_model  if args.model == "gemini"
                     else args.openai_model if args.model == "openai"
                     else args.ollama_model)
    _method       = args.method
    _account      = args.account or ""
    _nodes        = 0
    _edges        = 0
    _retrieval_ms = 0.0
    _llm_ms       = 0.0
    _actual_in    = 0
    _actual_out   = 0
    _depth        = args.depth if args.method in ("neighbors", "subgraph") else args.max_hops
    _status       = "failure"
    _error_type   = ""

    try:
        load_dotenv(ENV_FILE)
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if args.model == "gemini" and not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        if args.model == "openai" and not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")

        # -- Interactive input -----------------------------------------------
        account = args.account
        if not account:
            try:
                print("\nGraphRAG — Financial Risk Analysis")
                print("Enter a seed account ID (Ctrl-C to exit):\n")
                account = input("Account ID > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                return

        if not account:
            raise ValueError("Empty account ID — nothing to traverse")
        _account = account

        query = args.query
        if not query:
            try:
                print("Enter an investigation query (Ctrl-C to exit):\n")
                query = input("Query > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                return

        if not query:
            raise ValueError("Empty query — nothing to process")

        log.info("=== GraphRAG pipeline started ===")
        log.info("Query   : %s", query)
        log.info("Account : %s", account)
        log.info("Method  : %s", args.method)
        log.info("Backend : %s", args.model)

        # -- Graph retrieval stage -------------------------------------------
        log.info("Connecting to TigerGraph at %s ...", args.host)
        conn = connect_tigergraph(args.host, args.username, args.password, args.graph, args.token)

        log.info("Running %s traversal from account %s ...", args.method, account)
        if args.method == "neighbors":
            payload = get_account_neighbors(
                conn, account, depth=args.depth, graph_name=args.graph,
            )
            _depth = 1
        elif args.method == "paths":
            payload = trace_transaction_paths(
                conn, account,
                max_hops=args.max_hops,
                min_amount=args.min_amount,
                graph_name=args.graph,
            )
            _depth = args.max_hops
        elif args.method == "risk":
            payload = get_high_risk_transactions(
                conn, account, graph_name=args.graph,
            )
            _depth = 1
        else:  # subgraph (default)
            payload = expand_account_subgraph(
                conn, account, depth=args.depth, graph_name=args.graph,
            )
            _depth = args.depth

        _retrieval_ms = payload["metrics"]["retrieval_ms"]
        _nodes        = payload["metrics"]["nodes_retrieved"]
        _edges        = payload["metrics"]["edges_retrieved"]
        _depth        = payload["metrics"]["traversal_depth"]

        log.info(
            "Retrieval done — nodes: %d | edges: %d | depth: %d | latency: %.1f ms",
            _nodes, _edges, _depth, _retrieval_ms,
        )

        # -- Context construction stage -------------------------------------
        log.info("Building graph context ...")
        query_format = _detect_query_format(query)
        if query_format:
            log.info("Payment format detected in query: %s", query_format)
        context, ctx_nodes, ctx_edges = build_graph_context(payload, query_format=query_format)
        prompt      = build_prompt(query, account, context)
        est_tokens  = len(prompt) // CHARS_PER_TOKEN
        log.info("Prompt built — est. tokens: %d", est_tokens)

        # -- Generation stage -----------------------------------------------
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
            account=account,
            backend=args.model,
            model_name=_model_name,
            method=args.method,
            nodes=_nodes,
            edges=_edges,
            traversal_depth=_depth,
            retrieval_ms=_retrieval_ms,
            llm_ms=_llm_ms,
            actual_in=_actual_in,
            actual_out=_actual_out,
            answer=answer,
        )

    except Exception as exc:
        _error_type = type(exc).__name__
        log.error("Pipeline error [%s]: %s", _error_type, exc)
        log.debug(traceback.format_exc())

    finally:
        emit_metrics(
            _backend, _model_name, _method, _account,
            _nodes, _edges,
            _retrieval_ms, _llm_ms,
            _actual_in, _actual_out,
            _depth, _status, _error_type,
        )


if __name__ == "__main__":
    main()
