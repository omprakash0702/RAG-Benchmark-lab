"""
retrieve.py — GraphRAG Graph Traversal Retrieval Pipeline

Performs multi-hop graph traversal against TigerGraph FinancialGraph to
retrieve structured subgraphs for AML investigation queries.  Designed as
the retrieval stage of the GraphRAG benchmark, alongside:
  - Basic RAG  (vector similarity search via ChromaDB)
  - LLM-only   (no retrieval)

--- Graph traversal vs vector retrieval ---
Vector retrieval (Basic RAG) encodes a query as a dense embedding and finds
the k nearest transaction chunks by cosine distance.  It excels at surface-
level semantic matching but cannot discover structural relationships that span
multiple hops: if Account A → B → C form a layering chain, a vector search
anchored on A will not surface C unless C appears verbatim near A in the raw
text.

Graph traversal starts from a seed vertex and walks edges hop-by-hop.  A 3-hop
BFS from Account A discovers:
  Hop 1: Accounts that directly sent/received money with A
  Hop 2: Their counterparties  (first ring of intermediaries)
  Hop 3: The next ring — where layering destinations often appear

This structural awareness is why GraphRAG can expose hidden AML networks that
vector search misses entirely.

--- Why GraphRAG can find hidden relationships ---
In AML typologies (layering, structuring, smurfing) funds move through chains
of intermediary accounts before reaching a destination.  Each intermediate
account may have zero textual similarity to the seed — but every hop is
reachable by following TRANSFERRED_TO edges.  Graph traversal makes the
structural link explicit without any semantic similarity requirement.

--- Latency tradeoffs of graph traversal ---
Vector ANN search is O(log N) in index size; dominant cost is embedding the
query.  Graph traversal is O(V + E) in the discovered subgraph: latency grows
with hop depth and the edge cardinality of hub accounts.  High-degree hub
accounts (thousands of counterparties) can make a 2-hop traversal slower than
a full vector scan of 5 M chunks.  Benchmarking both under identical workloads
isolates this overhead so the cost/quality trade-off can be evaluated.

--- Integration with generate_answer.py ---
Every retrieval method returns a dict with a uniform schema:
  query_account   : str          — seed account ID
  method          : str          — which retrieval strategy was used
  nodes           : list[dict]   — vertex data (id, type, attributes)
  edges           : list[dict]   — edge data   (from, to, type, attributes)
  metrics         : dict         — benchmark measurements (see below)

generate_answer.py will consume this dict to build a structured LLM prompt
where the graph subgraph replaces (or augments) vector-retrieved chunks as the
evidence window for AML reasoning.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import pyTigerGraph as tg
except ImportError:
    print("pyTigerGraph not installed.  Run: pip install pyTigerGraph")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Environment & paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_HOST     = os.getenv("TIGERGRAPH_HOST", "http://localhost")
DEFAULT_USERNAME = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
DEFAULT_PASSWORD = os.getenv("TIGERGRAPH_PASSWORD", "tigergraph")
DEFAULT_GRAPH    = os.getenv("GRAPH_NAME", "FinancialGraph")
DEFAULT_TOKEN    = os.getenv("TIGERGRAPH_TOKEN", "")

DEFAULT_DEPTH      = 2
DEFAULT_MAX_HOPS   = 3
DEFAULT_MIN_AMOUNT = 0.0
MAX_FRONTIER_SIZE  = 50
EDGE_BATCH_SIZE    = 5   # accounts per per-account GSQL batch (exact edge pairs)

# ---------------------------------------------------------------------------
# Logging — INFO for benchmark runs, DEBUG for full query inspection
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GSQL query builders
# ---------------------------------------------------------------------------
#
# Design rules (TigerGraph Community Edition 4.2.2 interpreted-query limits):
#
#   1. Header is always:  INTERPRET QUERY () FOR GRAPH <name> {
#      — empty parameter list; no VERTEX<T>, STRING, or INT params.
#      — VERTEX<T> typed params caused "Encountered ':' at line 1" because
#        pyTigerGraph strips the header before sending to the REST endpoint,
#        leaving the GSQL body without the param declarations.
#
#   2. All values (account_id, min_amount) are embedded directly as string or
#      numeric literals inside the query via Python f-string substitution.
#      — avoids undefined-variable errors when the param header is stripped.
#
#   3. Seed account selection uses:
#        WHERE a.account_id == "<literal>"
#      — account_id is the declared PRIMARY_ID attribute name, accessible as
#        a regular attribute.  primary_id (pseudo-attribute) triggered
#        "Unsupported condition rule" in CE 4.2.2 interpreted mode.
#
#   4. No WHILE loops, FOREACH, accumulators, UNION, or typed param blocks.
#
#   5. Every SELECT result set is printed with an explicit AS alias so the
#      Python extraction layer can locate it by key.
#
#   6. Transaction reverse-traversal (finding transactions received BY an
#      account) requires a full Transaction scan because RECEIVED_TRANSACTION
#      is directed FROM Transaction TO Account.  AllTx = {Transaction.*}
#      declares the scan set explicitly rather than relying on an implicit
#      type shorthand, which is not reliable in interpreted mode.
#
# NOTE: account IDs are embedded as string literals — _safe_id() escapes
# backslashes and double quotes.  This is acceptable for a benchmark tool
# running against a local TigerGraph instance under controlled conditions.


def _safe_id(account_id: str) -> str:
    """Escape an account ID for safe embedding in a GSQL string literal."""
    return account_id.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# Builder: 1-hop neighbors
# ---------------------------------------------------------------------------

def _build_neighbors_query(account_id: str, graph_name: str) -> str:
    sid = _safe_id(account_id)
    return "\n".join([
        f'INTERPRET QUERY () FOR GRAPH {graph_name} {{',
        '    AllAccounts = {Account.*};',
        f'    Seed     = SELECT a FROM AllAccounts:a WHERE a.account_id == "{sid}";',
        '    SentTo   = SELECT tgt FROM Seed:s -(TRANSFERRED_TO)-> Account:tgt;',
        f'    RecvFrom = SELECT src FROM AllAccounts:src -(TRANSFERRED_TO)-> Account:tgt WHERE tgt.account_id == "{sid}";',
        '    SeedBank = SELECT b   FROM Seed:s -(BELONGS_TO_BANK)-> Bank:b;',
        '    SeedEnt  = SELECT ent FROM Seed:s -(ASSOCIATED_WITH)-> Entity:ent;',
        '    PRINT Seed AS seed_account, SentTo AS sent_to, RecvFrom AS recv_from,',
        '          SeedBank AS banks, SeedEnt AS entities;',
        '}',
    ])


# ---------------------------------------------------------------------------
# Builder: multi-hop path traversal (chained SELECTs, no WHILE)
# ---------------------------------------------------------------------------

def _build_paths_query(
    account_id: str,
    max_hops: int,
    min_amount: float,
    graph_name: str,
) -> str:
    """
    Build a chained-SELECT traversal query for max_hops hops.

    Each hop is a separate SELECT expanding from the previous result set —
    no WHILE loop.  min_amount is embedded as a numeric literal so no GSQL
    parameter declaration is needed.  Each hop result is aliased hop0…hopN
    so _merge_hops() can determine actual traversal depth from non-empty sets.
    """
    sid = _safe_id(account_id)
    lines = [
        f'INTERPRET QUERY () FOR GRAPH {graph_name} {{',
        '    AllAccounts = {Account.*};',
        f'    Hop0 = SELECT a FROM AllAccounts:a WHERE a.account_id == "{sid}";',
    ]
    for i in range(1, max_hops + 1):
        lines.append(
            f'    Hop{i} = SELECT tgt FROM Hop{i - 1}:s'
            f' -(TRANSFERRED_TO:e)-> Account:tgt'
            f' WHERE e.amount_paid >= {min_amount};'
        )
    hop_prints = ", ".join(f"Hop{i} AS hop{i}" for i in range(max_hops + 1))
    lines.append(f'    PRINT {hop_prints};')
    lines.append('}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder: risk transactions
# ---------------------------------------------------------------------------

def _build_high_risk_query(account_id: str, graph_name: str) -> str:
    """
    Return all transactions initiated or received by account_id.

    laundering_only filtering is done in Python after the query returns —
    avoids the mixed `param == 0 OR vertex.attr == 1` WHERE condition that
    triggers "Unsupported condition rule" in CE 4.2.2.

    AllTx = {Transaction.*} declares the full scan set explicitly; using a
    bare type name in FROM is not reliable in CE 4.2.2 interpreted mode.
    """
    sid = _safe_id(account_id)
    return "\n".join([
        f'INTERPRET QUERY () FOR GRAPH {graph_name} {{',
        '    AllAccounts = {Account.*};',
        '    AllTx       = {Transaction.*};',
        f'    Seed   = SELECT a FROM AllAccounts:a WHERE a.account_id == "{sid}";',
        '    InitTx = SELECT t FROM Seed:s -(INITIATED_TRANSACTION)-> Transaction:t;',
        f'    RecvTx = SELECT t FROM AllTx:t -(RECEIVED_TRANSACTION)-> Account:tgt WHERE tgt.account_id == "{sid}";',
        '    PRINT Seed AS account, InitTx AS initiated_tx, RecvTx AS received_tx;',
        '}',
    ])


# ---------------------------------------------------------------------------
# BFS helpers: per-account GSQL traversal for exact edge pairs
# ---------------------------------------------------------------------------

def _build_frontier_hop_query(account_ids: list, graph_name: str) -> str:
    """
    One combined hop query for all frontier accounts — NO AllTx scan.

    Returns frontier vertex data, next-hop accounts, initiated transactions,
    and bank links in a single round-trip.  Received transactions are fetched
    separately in one post-BFS AllTx scan (_fetch_received_transactions).
    """
    id_filter = " OR ".join(
        f'a.account_id == "{_safe_id(aid)}"' for aid in account_ids
    )
    return "\n".join([
        f'INTERPRET QUERY () FOR GRAPH {graph_name} {{',
        '    AllAccounts = {Account.*};',
        f'    Frontier = SELECT a FROM AllAccounts:a WHERE {id_filter};',
        '    NextHop  = SELECT tgt FROM Frontier:s -(TRANSFERRED_TO)-> Account:tgt;',
        '    InitTx   = SELECT t FROM Frontier:s -(INITIATED_TRANSACTION)-> Transaction:t;',
        '    Banks    = SELECT b FROM Frontier:s -(BELONGS_TO_BANK)-> Bank:b;',
        '    PRINT Frontier AS frontier_accounts, NextHop AS next_hop, InitTx AS init_tx, Banks AS banks;',
        '}',
    ])


def _build_received_tx_query(account_ids: list, graph_name: str) -> str:
    """
    One AllTx scan for received transactions across all discovered accounts.
    Called ONCE after BFS completes, not per hop.
    """
    recv_filter = " OR ".join(
        f'tgt.account_id == "{_safe_id(aid)}"' for aid in account_ids
    )
    return "\n".join([
        f'INTERPRET QUERY () FOR GRAPH {graph_name} {{',
        '    AllTx = {Transaction.*};',
        f'    RecvTx = SELECT t FROM AllTx:t -(RECEIVED_TRANSACTION)-> Account:tgt WHERE {recv_filter};',
        '    PRINT RecvTx AS recv_tx;',
        '}',
    ])


def _fetch_received_transactions(
    conn: "tg.TigerGraphConnection",
    account_ids: list,
    graph_name: str,
) -> tuple[list[dict], list[dict]]:
    """
    Scan AllTx ONCE for all accounts that received transactions.
    Called once after BFS completes — not per hop.
    Returns (recv_tx_vertices, recv_edges).
    """
    if not account_ids:
        return [], []
    # Clamp to avoid oversized WHERE clause
    accounts_to_query = account_ids[:MAX_FRONTIER_SIZE]
    try:
        query = _build_received_tx_query(accounts_to_query, graph_name)
        raw, _ = _run_raw(conn, query)
        recv_txs  = _extract_vertices(raw, "recv_tx")
        recv_edges = [
            {
                "from_type": "Transaction", "from_id": tx.get("v_id", ""),
                "edge_type": "RECEIVED_TRANSACTION",
                "to_type":   "Account",     "to_id":   "",
                "attributes": {},
            }
            for tx in recv_txs
        ]
        return recv_txs, recv_edges
    except Exception as exc:
        log.debug("Post-BFS received-tx query failed: %s", exc)
        return [], []


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect_tigergraph(
    host: str = DEFAULT_HOST,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    graph_name: str = DEFAULT_GRAPH,
    token: str = DEFAULT_TOKEN,
) -> "tg.TigerGraphConnection":
    """
    Create and return an authenticated TigerGraph connection.

    TigerGraph Cloud (tgcloud.io): set TIGERGRAPH_HOST to your cluster URL
    and TIGERGRAPH_TOKEN to a pre-generated RESTPP token — the token path
    bypasses username/password entirely.

    Community Edition (local Docker): leave TIGERGRAPH_TOKEN blank; the
    code falls back to username/password and attempts createSecret() for CE.
    """
    log.info("Connecting to TigerGraph at %s (graph: %s)", host, graph_name)
    if token:
        conn = tg.TigerGraphConnection(
            host=host,
            graphname=graph_name,
            apiToken=token,
        )
        log.info("Connected with pre-set API token (Cloud mode)")
    else:
        conn = tg.TigerGraphConnection(
            host=host,
            graphname=graph_name,
            username=username,
            password=password,
        )
        try:
            conn.getToken(conn.createSecret())
            log.info("Token auth successful")
        except Exception:
            log.debug("Token auth unavailable — continuing with password auth")

    try:
        vertex_types = conn.getVertexTypes()
        log.info(
            "Connected — %d vertex types: %s",
            len(vertex_types),
            ", ".join(vertex_types),
        )
    except Exception as exc:
        log.warning("Could not verify graph schema: %s", exc)

    return conn


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _run_raw(
    conn: "tg.TigerGraphConnection",
    query: str,
) -> tuple[list[Any], float]:
    """
    Execute a fully-formed interpreted GSQL query string.

    No params are passed — all values are embedded as literals inside the
    query by the builder functions.  The full query string is logged at DEBUG
    level before execution so problems are diagnosable without changing code.
    """
    log.debug("Sending interpreted query:\n%s", query)
    t0 = time.perf_counter()
    try:
        results = conn.runInterpretedQuery(query)
    except Exception as exc:
        log.error("Interpreted query failed: %s", exc)
        log.error("Failed query was:\n%s", query)
        raise
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _extract_vertices(raw: list[Any], key: str) -> list[dict]:
    """Pull a named vertex list from interpreted query output."""
    for block in raw:
        if isinstance(block, dict) and key in block:
            return block[key]
    return []


def _extract_scalar(raw: list[Any], key: str, default: Any = 0) -> Any:
    for block in raw:
        if isinstance(block, dict) and key in block:
            return block[key]
    return default


def _merge_hops(raw: list[Any], max_hops: int) -> tuple[list[dict], int]:
    """
    Collect and deduplicate vertices from hop0 … hop{max_hops}.

    Returns (nodes, actual_depth) where actual_depth is the index of the last
    non-empty hop beyond the seed.  A disconnected account has hop0 only and
    actual_depth == 0.
    """
    seen: set[str] = set()
    nodes: list[dict] = []
    actual_depth = 0
    for i in range(max_hops + 1):
        hop = _extract_vertices(raw, f"hop{i}")
        for v in hop:
            vid = v.get("v_id", "")
            if vid not in seen:
                seen.add(vid)
                nodes.append({**v, "hop": i})
        if i > 0 and hop:
            actual_depth = i
    return nodes, actual_depth


def _log_metrics(metrics: dict, account_id: str, method: str) -> None:
    """Emit a structured GRAPH_RETRIEVAL_METRICS line for benchmark parsing."""
    log.info(
        "GRAPH_RETRIEVAL_METRICS | method=%s | account=%s | "
        "retrieval_ms=%.2f | nodes_retrieved=%d | edges_retrieved=%d | "
        "traversal_depth=%d | unique_accounts=%d | unique_transactions=%d | "
        "unique_banks=%d | unique_edges=%d",
        method,
        account_id,
        metrics["retrieval_ms"],
        metrics["nodes_retrieved"],
        metrics["edges_retrieved"],
        metrics["traversal_depth"],
        metrics.get("unique_accounts", 0),
        metrics.get("unique_transactions", 0),
        metrics.get("unique_banks", 0),
        metrics.get("unique_edges", 0),
    )


# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------

def get_account_neighbors(
    conn: "tg.TigerGraphConnection",
    account_id: str,
    depth: int = 1,
    graph_name: str = DEFAULT_GRAPH,
) -> dict:
    """
    Retrieve direct (1-hop) neighbors of account_id.

    Returns accounts that sent to or received from the seed, plus its bank
    and associated entity.  depth is accepted for API consistency with
    expand_account_subgraph; the underlying query always runs at 1 hop.

    Graceful handling:
    - Account not found → empty nodes list, warning logged.
    - No counterparties  → seed node returned alone (disconnected account).
    """
    log.info("get_account_neighbors: %s (depth=%d)", account_id, depth)

    query = _build_neighbors_query(account_id, graph_name)
    raw, elapsed_ms = _run_raw(conn, query)

    seed      = _extract_vertices(raw, "seed_account")
    sent_to   = _extract_vertices(raw, "sent_to")
    recv_from = _extract_vertices(raw, "recv_from")
    banks     = _extract_vertices(raw, "banks")
    entities  = _extract_vertices(raw, "entities")

    if not seed:
        log.warning("Account not found in graph: %s", account_id)
    if seed and not sent_to and not recv_from:
        log.info("Account %s has no transfer counterparties (disconnected)", account_id)

    nodes = (
          [{"type": "Account", "role": "seed",      **v} for v in seed]
        + [{"type": "Account", "role": "sent_to",   **v} for v in sent_to]
        + [{"type": "Account", "role": "recv_from", **v} for v in recv_from]
        + [{"type": "Bank",    "role": "bank",       **v} for v in banks]
        + [{"type": "Entity",  "role": "entity",     **v} for v in entities]
    )

    result = {
        "query_account": account_id,
        "method":        "get_account_neighbors",
        "nodes":         nodes,
        "edges":         [],
        "metrics": {
            "retrieval_ms":    round(elapsed_ms, 2),
            "nodes_retrieved": len(nodes),
            "edges_retrieved": 0,
            "traversal_depth": 1,
        },
    }
    _log_metrics(result["metrics"], account_id, "get_account_neighbors")
    return result


def trace_transaction_paths(
    conn: "tg.TigerGraphConnection",
    account_id: str,
    max_hops: int = DEFAULT_MAX_HOPS,
    min_amount: float = DEFAULT_MIN_AMOUNT,
    graph_name: str = DEFAULT_GRAPH,
) -> dict:
    """
    Multi-hop traversal following TRANSFERRED_TO edges from account_id.

    The query is built at call time as max_hops chained SELECT statements —
    one per hop — avoiding WHILE loops.  min_amount and account_id are both
    embedded as literals in the query string.

    Graceful handling:
    - Disconnected accounts (no outgoing edges) return the seed alone with
      traversal_depth == 0 — not an error.
    - Empty hop1 means the account sent no transfers above min_amount.
    """
    log.info(
        "trace_transaction_paths: %s (max_hops=%d, min_amount=%.2f)",
        account_id, max_hops, min_amount,
    )

    query = _build_paths_query(account_id, max_hops, min_amount, graph_name)
    raw, elapsed_ms = _run_raw(conn, query)

    nodes, actual_depth = _merge_hops(raw, max_hops)

    if not nodes:
        log.warning(
            "No traversal results for %s — account may be disconnected or not loaded",
            account_id,
        )

    result = {
        "query_account": account_id,
        "method":        "trace_transaction_paths",
        "nodes":         [{"type": "Account", **n} for n in nodes],
        "edges":         [],
        "metrics": {
            "retrieval_ms":    round(elapsed_ms, 2),
            "nodes_retrieved": len(nodes),
            "edges_retrieved": 0,
            "traversal_depth": actual_depth,
        },
    }
    _log_metrics(result["metrics"], account_id, "trace_transaction_paths")
    return result


def get_high_risk_transactions(
    conn: "tg.TigerGraphConnection",
    account_id: str,
    laundering_only: bool = False,
    graph_name: str = DEFAULT_GRAPH,
) -> dict:
    """
    Retrieve transactions initiated or received by account_id.

    The query returns all transactions unconditionally; laundering_only
    filtering is applied in Python.  This avoids the mixed
    `param == 0 OR vertex.attr == 1` WHERE condition that triggers
    "Unsupported condition rule" in CE 4.2.2.

    Graceful handling:
    - Account not found → warning logged, empty result.
    - Empty transactions for a known account → Phase 2 edge load may be
      incomplete; check load_graph_data.gsql RUN LOADING JOB status.
    """
    log.info(
        "get_high_risk_transactions: %s (laundering_only=%s)",
        account_id, laundering_only,
    )

    query = _build_high_risk_query(account_id, graph_name)
    raw, elapsed_ms = _run_raw(conn, query)

    account_data = _extract_vertices(raw, "account")
    init_tx      = _extract_vertices(raw, "initiated_tx")
    recv_tx      = _extract_vertices(raw, "received_tx")

    # Deduplicate transactions that appear in both initiated and received.
    seen_tx: set[str] = set()
    transactions: list[dict] = []
    for tx in init_tx + recv_tx:
        vid = tx.get("v_id", "")
        if vid not in seen_tx:
            seen_tx.add(vid)
            transactions.append(tx)

    # Apply laundering filter in Python — avoids unsupported WHERE condition.
    if laundering_only:
        transactions = [
            tx for tx in transactions
            if tx.get("attributes", {}).get("laundering_flag", 0) == 1
        ]

    if not account_data:
        log.warning("Account not found in graph: %s", account_id)
    if account_data and not transactions:
        log.info(
            "No transactions found for %s (laundering_only=%s) — "
            "edge load may be incomplete",
            account_id, laundering_only,
        )

    nodes = (
          [{"type": "Account",     **v} for v in account_data]
        + [{"type": "Transaction", **v} for v in transactions]
    )

    result = {
        "query_account": account_id,
        "method":        "get_high_risk_transactions",
        "nodes":         nodes,
        "edges":         [],
        "metrics": {
            "retrieval_ms":    round(elapsed_ms, 2),
            "nodes_retrieved": len(nodes),
            "edges_retrieved": 0,
            "traversal_depth": 1,
        },
    }
    _log_metrics(result["metrics"], account_id, "get_high_risk_transactions")
    return result


def expand_account_subgraph(
    conn: "tg.TigerGraphConnection",
    account_id: str,
    depth: int = DEFAULT_DEPTH,
    graph_name: str = DEFAULT_GRAPH,
) -> dict:
    """
    Frontier-based BFS expansion of the account subgraph up to `depth` hops.

    Each hop issues one GSQL round-trip for the current frontier accounts, then
    expands to newly discovered accounts for the next hop.  Accounts, transactions,
    banks, and edges are accumulated and deduplicated across all hops so the LLM
    receives the full connected suspicious structure — not just the seed's immediate
    neighbourhood.

    Graceful handling:
    - Missing account → warning logged, empty subgraph returned.
    - Isolated account (no outgoing edges) → seed + its own transactions/banks.
    - Large frontiers are clamped to MAX_FRONTIER_SIZE per hop to keep query size
      manageable on CE 4.2.2.
    """
    log.info("expand_account_subgraph (BFS): %s (depth=%d)", account_id, depth)

    t_start = time.perf_counter()

    # Accumulated results
    all_accounts:     list[dict] = []
    all_transactions: list[dict] = []
    all_banks:        list[dict] = []
    all_edges:        list[dict] = []

    # Dedup sets
    queued:          set[str]   = {_safe_id(account_id)}
    collected_accts: set[str]   = set()
    collected_tx:    set[str]   = set()
    collected_banks: set[str]   = set()
    collected_edges: set[tuple] = set()

    frontier: list[str] = [account_id]
    actual_depth = 0

    for hop in range(depth + 1):
        if not frontier:
            break

        frontier = frontier[:MAX_FRONTIER_SIZE]

        # One GSQL query per hop — no AllTx scan, fast regardless of tx volume
        query = _build_frontier_hop_query(frontier, graph_name)
        raw, _ = _run_raw(conn, query)

        # Collect current-frontier account vertex data
        for acct in _extract_vertices(raw, "frontier_accounts"):
            aid = acct.get("v_id", "")
            if aid and aid not in collected_accts:
                collected_accts.add(aid)
                all_accounts.append(acct)

        # Collect initiated transactions
        for tx in _extract_vertices(raw, "init_tx"):
            vid = tx.get("v_id", "")
            if vid and vid not in collected_tx:
                collected_tx.add(vid)
                all_transactions.append(tx)

        # Collect banks
        for bank in _extract_vertices(raw, "banks"):
            bid = bank.get("v_id", "")
            if bid and bid not in collected_banks:
                collected_banks.add(bid)
                all_banks.append(bank)

        # Build TRANSFERRED_TO edges: exact for single-account frontier,
        # bounded approximation for multi-account frontiers.
        # Cap at 150 pairs per hop to keep context manageable.
        next_hop_accounts = _extract_vertices(raw, "next_hop")
        MAX_EDGE_PAIRS_PER_HOP = 150
        edge_pairs = 0
        for src in frontier:
            if edge_pairs >= MAX_EDGE_PAIRS_PER_HOP:
                break
            for tgt in next_hop_accounts:
                if edge_pairs >= MAX_EDGE_PAIRS_PER_HOP:
                    break
                e = {"from_type": "Account", "from_id": src,
                     "edge_type": "TRANSFERRED_TO",
                     "to_type": "Account", "to_id": tgt.get("v_id", ""),
                     "attributes": {}}
                key = (e["from_id"], e["edge_type"], e["to_id"])
                if key not in collected_edges:
                    collected_edges.add(key)
                    all_edges.append(e)
                    edge_pairs += 1

        # Build INITIATED_TRANSACTION edges for single-account frontiers (exact).
        # For multi-account frontiers skip to avoid false attribution.
        if len(frontier) == 1:
            src = frontier[0]
            for tx in _extract_vertices(raw, "init_tx"):
                e = {"from_type": "Account", "from_id": src,
                     "edge_type": "INITIATED_TRANSACTION",
                     "to_type": "Transaction", "to_id": tx.get("v_id", ""),
                     "attributes": tx.get("attributes", {})}
                key = (e["from_id"], e["edge_type"], e["to_id"])
                if key not in collected_edges:
                    collected_edges.add(key)
                    all_edges.append(e)

        # Build next frontier from newly discovered TRANSFERRED_TO targets
        next_frontier: list[str] = []
        for acct in next_hop_accounts:
            aid = acct.get("v_id", "")
            safe_aid = _safe_id(aid)
            if safe_aid and safe_aid not in queued:
                queued.add(safe_aid)
                next_frontier.append(aid)
                if aid not in collected_accts:
                    collected_accts.add(aid)
                    all_accounts.append(acct)

        if next_frontier:
            actual_depth = hop + 1

        frontier = next_frontier

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    if not all_accounts:
        log.warning("Account not found or empty subgraph: %s", account_id)

    nodes = (
          [{"type": "Account",     **v} for v in all_accounts]
        + [{"type": "Transaction", **v} for v in all_transactions]
        + [{"type": "Bank",        **v} for v in all_banks]
    )

    result = {
        "query_account": account_id,
        "method":        "expand_account_subgraph",
        "nodes":         nodes,
        "edges":         all_edges,
        "metrics": {
            "retrieval_ms":       round(elapsed_ms, 2),
            "nodes_retrieved":    len(nodes),
            "edges_retrieved":    len(all_edges),
            "traversal_depth":    actual_depth,
            "unique_accounts":    len(all_accounts),
            "unique_transactions": len(all_transactions),
            "unique_banks":       len(all_banks),
            "unique_edges":       len(all_edges),
        },
    }
    _log_metrics(result["metrics"], account_id, "expand_account_subgraph")
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(result: dict) -> None:
    """Print a benchmark-style report, matching basic_rag/retrieve.py layout."""
    metrics = result["metrics"]
    nodes   = result["nodes"]
    edges   = result["edges"]

    node_types: dict[str, list] = {}
    for n in nodes:
        node_types.setdefault(n.get("type", "Unknown"), []).append(n)

    width = 70
    print("\n" + "=" * width)
    print("  GRAPH RAG — RETRIEVAL RESULTS")
    print("=" * width)
    print(f"  Account    : {result['query_account']}")
    print(f"  Method     : {result['method']}")
    print(f"  Depth      : {metrics['traversal_depth']} hop(s)")
    print("-" * width)
    print(f"  Total time : {metrics['retrieval_ms']:>7.2f} ms")
    print(f"  Nodes      : {metrics['nodes_retrieved']:>7,}")
    print(f"  Edges      : {metrics['edges_retrieved']:>7,}")
    print("=" * width)

    for ntype, nlist in node_types.items():
        print(f"\n  [{ntype}] — {len(nlist)} node(s)")
        for n in nlist[:5]:
            vid   = n.get("v_id", n.get("id", "?"))
            attrs = n.get("attributes", {})
            print(f"    id={vid}  |  {_format_attrs(attrs)}")
        if len(nlist) > 5:
            print(f"    … and {len(nlist) - 5} more")

    if edges:
        print(f"\n  [Edges] — {len(edges)} total")
        for e in edges[:5]:
            src   = e.get("from_id",  "?")
            tgt   = e.get("to_id",    "?")
            etype = e.get("e_type",   "?")
            attrs = e.get("attributes", {})
            amt   = attrs.get("amount_paid", "")
            flag  = attrs.get("is_laundering", "")
            print(f"    {src} --[{etype}]--> {tgt}  amount={amt}  flag={flag}")
        if len(edges) > 5:
            print(f"    … and {len(edges) - 5} more")

    print("\n" + "=" * width + "\n")


def _format_attrs(attrs: dict, max_keys: int = 4) -> str:
    parts = [f"{k}={v!r}" for k, v in list(attrs.items())[:max_keys]]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GraphRAG — multi-hop graph traversal retrieval from TigerGraph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python retrieve.py --account ACC_12345
  python retrieve.py --account ACC_12345 --method paths --max_hops 4
  python retrieve.py --account ACC_12345 --method subgraph --depth 3
  python retrieve.py --account ACC_12345 --method risk --laundering_only
  python retrieve.py --account ACC_12345 --debug
        """,
    )
    p.add_argument(
        "--account", type=str, required=True,
        help="Seed account ID to start traversal from.",
    )
    p.add_argument(
        "--method",
        choices=["neighbors", "paths", "risk", "subgraph"],
        default="subgraph",
        help="Retrieval method (default: subgraph).",
    )
    p.add_argument(
        "--depth", type=int, default=DEFAULT_DEPTH,
        help=f"Traversal depth for neighbors/subgraph (default: {DEFAULT_DEPTH}).",
    )
    p.add_argument(
        "--max_hops", type=int, default=DEFAULT_MAX_HOPS,
        help=f"Maximum hops for paths method (default: {DEFAULT_MAX_HOPS}).",
    )
    p.add_argument(
        "--min_amount", type=float, default=DEFAULT_MIN_AMOUNT,
        help=f"Minimum transfer amount filter for paths (default: {DEFAULT_MIN_AMOUNT}).",
    )
    p.add_argument(
        "--laundering_only", action="store_true",
        help="Return only laundering-flagged transactions (risk method only).",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging to print full GSQL query strings.",
    )
    p.add_argument(
        "--host",     type=str, default=DEFAULT_HOST,
        help="TigerGraph host URL (overrides .env).",
    )
    p.add_argument(
        "--username", type=str, default=DEFAULT_USERNAME,
        help="TigerGraph username (overrides .env).",
    )
    p.add_argument(
        "--password", type=str, default=DEFAULT_PASSWORD,
        help="TigerGraph password (overrides .env).",
    )
    p.add_argument(
        "--graph", type=str, default=DEFAULT_GRAPH,
        help="Graph name (overrides .env).",
    )
    p.add_argument(
        "--token", type=str, default=DEFAULT_TOKEN,
        help="TigerGraph RESTPP API token (TigerGraph Cloud). Overrides username/password.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    conn = connect_tigergraph(
        host=args.host,
        username=args.username,
        password=args.password,
        graph_name=args.graph,
        token=args.token,
    )

    dispatch = {
        "neighbors": lambda: get_account_neighbors(
            conn, args.account, depth=args.depth, graph_name=args.graph,
        ),
        "paths": lambda: trace_transaction_paths(
            conn, args.account, max_hops=args.max_hops,
            min_amount=args.min_amount, graph_name=args.graph,
        ),
        "risk": lambda: get_high_risk_transactions(
            conn, args.account, laundering_only=args.laundering_only,
            graph_name=args.graph,
        ),
        "subgraph": lambda: expand_account_subgraph(
            conn, args.account, depth=args.depth, graph_name=args.graph,
        ),
    }

    log.info("Running retrieval: method=%s  account=%s", args.method, args.account)
    result = dispatch[args.method]()
    print_results(result)

    m = result["metrics"]
    log.info(
        "RETRIEVAL_DONE | method=%s | account=%s | "
        "retrieval_ms=%.2f | nodes=%d | edges=%d | depth=%d",
        args.method, args.account,
        m["retrieval_ms"], m["nodes_retrieved"],
        m["edges_retrieved"], m["traversal_depth"],
    )


if __name__ == "__main__":
    main()
