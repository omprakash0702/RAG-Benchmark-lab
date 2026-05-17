# RAG Benchmark Lab — AML Investigation Pipeline Comparison

A benchmark system comparing three retrieval-augmented generation architectures for Anti-Money Laundering (AML) investigation on the IBM HI-Small synthetic transaction dataset. Each pipeline answers the same financial crime query and is evaluated on latency, cost, LLM-judge quality (6 dimensions), and BERTScore semantic similarity against expert reference answers.



## Project Overview

This project implements and benchmarks three distinct RAG pipelines applied to financial transaction analysis:

| Pipeline | Retrieval Method | Strength |
|---|---|---|
| **LLM-Only** | None (parametric knowledge) | Speed baseline; exposes hallucination risk |
| **Basic RAG** | Vector similarity (ChromaDB) | Semantic query matching; good for narrative queries |
| **GraphRAG** | Graph BFS (TigerGraph) | Structural patterns; fan-in/fan-out/circular detection |

All three pipelines run against the same query simultaneously. The Streamlit dashboard displays answers side-by-side with latency, token, cost, LLM-judge, and BERTScore metrics.

---

## Why GraphRAG for AML?

Money laundering exploits **multi-hop transaction chains** that flat-text retrieval cannot detect. GraphRAG traverses the actual financial network and surfaces structural evidence that neither LLM-Only nor Basic RAG can produce.

### Core Advantages

- **Structural pattern detection** — Identifies fan-in/fan-out, circular transfers (A→B→A), and confirmed pass-through intermediaries directly from graph topology. These are graph-computed facts from the edge set, not hallucinations.

- **Multi-hop chain reconstruction** — BFS traversal from a seed account reveals the full layering chain: who sent to whom, across how many hops, in what order. Basic RAG retrieves isolated transactions; GraphRAG retrieves the entire connected subgraph.

- **Pass-through intermediary identification** — Accounts appearing in both the fan-in map (many unique senders) and fan-out map (many unique recipients) are automatically confirmed as pass-through nodes — the structural hallmark of layering.

- **Circular pair detection** — Detects reciprocal edge pairs (A→B and B→A both present) as confirmed round-trip evidence of fund cycling, which is invisible to vector similarity search.

- **Real counts, not estimates** — When asked how many unique recipients account X sent to, GraphRAG returns the exact graph count. LLM-Only fabricates; Basic RAG returns at most 1 per retrieved document.

- **Token efficiency** — A 4-hop subgraph with 46 accounts and 161 transactions encodes relationship structure that would require hundreds of text chunks to cover. GraphRAG context is dense at ~1,100 tokens vs ~1,400 tokens for 5 Basic RAG chunks.

- **Speed** — TigerGraph BFS completes in 2–5 seconds. End-to-end GraphRAG wall time is ~8–20s vs ~90–120s for Basic RAG.

- **AML phase classification** — The graph structure directly reveals the laundering phase: placement (single large inflow), layering (multi-hop intermediaries with near-zero net balance), or integration (outflow to legitimate accounts).

- **Grounded, verifiable answers** — Every claim cites a graph-computed fact (e.g. "Account 8042DF7E0 is a confirmed pass-through with 4 inbound senders and 3 outbound recipients") that can be independently verified in TigerGraph.

- **No silent evidence gap** — Basic RAG fails silently when the queried account isn't in the retrieved chunks. GraphRAG's BFS is explicitly scoped to the seed account's network — it knows exactly what it found and what it did not.

### Where Basic RAG Is Better

| Query type | Better pipeline | Reason |
|---|---|---|
| Semantic narrative queries | Basic RAG | Embedding similarity retrieves contextually related transactions |
| Queries with no known seed account | Basic RAG | GraphRAG requires a seed account ID to start BFS |
| General AML pattern knowledge | Basic RAG | Retrieved chunks provide domain context the LLM can reason over |

---

## Dataset

### Source
IBM HI-Small synthetic AML dataset — designed to simulate realistic money laundering activity.

### Coverage
```
Period        : 2022-09-01 to 2022-09-30  (30 days)
Transactions  : ~2,000,000 records
Accounts      : ~510,000 unique
Banks         : ~30,000 unique
Laundering %  : ~0.2% flagged (Is_Laundering = 1)
```

### Raw CSV Schema (`datasets/raw/HI-Small_Trans.csv`, 454 MB)

| Column | Description |
|---|---|
| `Timestamp` | Transaction datetime (YYYY/MM/DD HH:MM) |
| `From Bank` | Sender's bank ID |
| `Account` | Sender account ID (9-char hex, e.g. `8042DF7E0`) |
| `To Bank` | Receiver's bank ID |
| `Account.1` | Receiver account ID |
| `Amount Received` | Amount credited to receiver |
| `Receiving Currency` | Currency of received amount |
| `Amount Paid` | Amount debited from sender |
| `Payment Currency` | Currency of paid amount |
| `Payment Format` | Wire / Cheque / ACH / Credit Card / Cash / Reinvestment |
| `Is Laundering` | Ground-truth label: 0 = clean, 1 = suspicious |

### Accounts Metadata (`datasets/raw/HI-Small_accounts.csv`, 33 MB)

| Column | Description |
|---|---|
| `Bank Name` | Human-readable bank name |
| `Bank ID` | Numeric bank identifier |
| `Account Number` | Account hex ID |
| `Entity ID` | Owning entity ID |
| `Entity Name` | Corporation / Sole Proprietorship / Partnership |

### Transaction Amounts & Currencies

```
Amount range  : $1 — $1,285,209,696
Currencies    : USD, EUR, GBP, JPY, INR, RUB, BTC, ETH, Rupee, Yuan, + others
Payment types : Wire, Cheque, ACH, Credit Card, Cash, Reinvestment
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG BENCHMARK LAB                               │
│                      Streamlit Dashboard UI                             │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │  user query + account ID
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐
   │  LLM-Only   │ │  Basic RAG   │ │    GraphRAG      │
   │  Baseline   │ │  (ChromaDB)  │ │  (TigerGraph)    │
   └──────┬──────┘ └──────┬───────┘ └────────┬─────────┘
          │               │                  │
          ▼               ▼                  ▼
   ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐
   │   Prompt    │ │  Embedding   │ │  BFS Subgraph    │
   │  (general   │ │  + HNSW ANN  │ │  Traversal       │
   │  knowledge) │ │  retrieval   │ │  (2-4 hops)      │
   └──────┬──────┘ └──────┬───────┘ └────────┬─────────┘
          │               │                  │
          └───────────────┼──────────────────┘
                          ▼
               ┌──────────────────┐
               │  LLM Generation  │
               │  Gemini / GPT /  │
               │  Ollama          │
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │  Evaluation      │
               │  LLM Judge +     │
               │  BERTScore       │
               └──────────────────┘
```

### Data Stores

```
┌──────────────────────┐    ┌───────────────────────┐    ┌─────────────────────┐
│   ChromaDB           │    │   TigerGraph CE 4.2.2 │    │   Raw CSV           │
│   (Vector Store)     │    │   (Graph Database)    │    │   (Source of Truth) │
│                      │    │                       │    │                     │
│  Collection:         │    │ Graph: FinancialGraph │    │  HI-Small_Trans.csv │
│  financial_          │    │                       │    │  454 MB / 2M rows   │
│  transactions        │    │  Vertices:            │    │                     │
│                      │    │   Account   ~510k     │    │  HI-Small_          │
│  Embedding model:    │    │   Bank      ~30k      │    │  accounts.csv       │
│  all-MiniLM-L6-v2    │    │   Entity    ~166k     │    │  33 MB / 518k rows  │
│  384 dimensions      │    │   Transaction 2M      │    │                     │
│                      │    │                       │    │                     │
│  Records: ~2M        │    │  Edges:               │    │                     │
│                      │    │   TRANSFERRED_TO  2M  │    │                     │
│                      │    │   INITIATED       2M  │    │                     │
│                      │    │   RECEIVED        2M  │    │                     │
│                      │    │   BELONGS_TO_BANK 510k│    │                     │
└──────────────────────┘    └───────────────────────┘    └─────────────────────┘
```

---

## Pipeline 1 — LLM-Only Baseline

The baseline: pure parametric knowledge, no retrieval. Reveals the hallucination floor when a model is asked about data it has never seen.

### Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────┐
│            Query Intent               │
│  classify_query() -> factual /        │
│                     investigation     │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│           Prompt Builder              │
│                                       │
│  ┌─────────────────────────────────┐  │
│  │  DATASET CONTEXT                │  │
│  │  (2022-09-01 to 2022-09-30)     │  │
│  ├─────────────────────────────────┤  │
│  │  CONSTRAINTS                    │  │
│  │  "Do NOT fabricate account IDs" │  │
│  ├─────────────────────────────────┤  │
│  │  FORMAT                         │  │
│  │  factual      -> numbered list  │  │
│  │  investigation -> risk report   │  │
│  └─────────────────────────────────┘  │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│           LLM Generation              │
│                                       │
│   Primary  : Gemini 2.5 Flash         │
│   Fallback : GPT-4o-mini              │
│   Fallback : Ollama (local)           │
│                                       │
│   Retry    : 3x, exponential backoff  │
│   (429 / 503 / 500 errors)            │
└───────────────────┬───────────────────┘
                    │
                    ▼
              Answer + Metrics
              LLM_ONLY_METRICS
```

### Key Files

| File | Lines | Purpose |
|---|---|---|
| `pipelines/llm_only/generate_answer.py` | 619 | Main pipeline driver |

### CLI

```bash
python pipelines/llm_only/generate_answer.py \
  --query "Top 5 accounts with maximum outgoing transfers" \
  --model gemini \
  --gemini_model gemini-2.5-flash \
  --max_tokens 2048 \
  --temperature 1.0
```

### Metrics Output

```
LLM_ONLY_METRICS | backend=gemini | model=gemini-2.5-flash | retrieved=0 | context=0
  | embed_ms=0.0 | search_ms=0.0 | llm_ms=3210.4 | total_ms=3210.4
  | est_tokens=0 | actual_in=183 | actual_out=57 | status=success | error_type=none
```

### Limitations

- No access to actual transaction records — answers based entirely on pre-training knowledge
- Account IDs, amounts, and rankings are fabricated with high confidence
- Streamlit dashboard shows an orange warning banner on all LLM-only answers

---

## Pipeline 2 — Basic RAG (Vector Semantic Retrieval)

Encodes the query with a sentence transformer and retrieves the most semantically similar transaction narratives from ChromaDB before prompting the LLM.

### Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────┐
│        Payment Format Detection       │
│  detect_payment_format(query)         │
│  "cash" -> filter: payment_format=Cash│
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│         Embedding Encoder             │
│                                       │
│  Model: all-MiniLM-L6-v2             │
│  Output: 384-dim dense vector         │
│  Latency: 300-600 ms (first call)     │
└───────────────────┬───────────────────┘
                    │  query vector
                    ▼
┌───────────────────────────────────────┐
│          ChromaDB HNSW Index          │
│                                       │
│  ANN search -> top-k nearest chunks   │
│  Optional where_filter: {             │
│    "payment_format": "Cash"           │
│  }                                    │
│  Fallback: unfiltered if 0 results    │
│  Latency: 100-200 ms                  │
└───────────────────┬───────────────────┘
                    │  top-5 chunks + metadata
                    ▼
┌───────────────────────────────────────┐
│         Context Builder               │
│                                       │
│  Max chunks : 5                       │
│  Max chars  : 12,000                  │
│  Per chunk  : narrative text          │
│              + similarity score       │
│              + metadata               │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│           Prompt Builder              │
│                                       │
│  Query intent -> factual /            │
│                  investigation        │
│                                       │
│  factual      -> "Answer DIRECTLY     │
│                  as numbered list"    │
│  investigation -> 8-step risk report  │
└───────────────────┬───────────────────┘
                    │
                    ▼
              LLM Generation
              (same fallback chain)
                    │
                    ▼
              Answer + Metrics
              RAG_METRICS
```

### Key Files

| File | Lines | Purpose |
|---|---|---|
| `pipelines/basic_rag/generate_answer.py` | 626 | Pipeline driver |
| `pipelines/basic_rag/retrieve.py` | 326 | ChromaDB retrieval logic |

### CLI

```bash
python pipelines/basic_rag/generate_answer.py \
  --query "Repeated Cash transactions in last 3 days" \
  --top_k 5 \
  --model gemini \
  --gemini_model gemini-2.5-flash
```

### Metrics Output

```
RAG_METRICS | backend=gemini | model=gemini-2.5-flash | retrieved=5 | context=5
  | embed_ms=412.3 | search_ms=187.6 | llm_ms=8923.1 | total_ms=9522.9
  | est_tokens=1174 | actual_in=1174 | actual_out=312 | status=success | error_type=none
```

### Limitations

- Retrieves semantically similar text, not structurally connected records
- Poor at structural queries (fan-in/fan-out) — retrieves single transactions, not network patterns
- ChromaDB stores ~2M records (subset of full dataset)

---

## Pipeline 3 — GraphRAG (Knowledge Graph Traversal)

Uses TigerGraph to perform Breadth-First Search (BFS) from a seed account, discovers the structural transaction network, then summarises it for the LLM.

### Architecture

```
User Query + Seed Account ID
    │
    ▼
┌───────────────────────────────────────┐
│        TigerGraph BFS Traversal       │
│                                       │
│  1-hop: direct neighbors              │
│  2-hop: neighbors-of-neighbors        │
│  (configurable up to 4 hops)          │
│                                       │
│  Queries via GSQL chained-SELECT      │
│  (TigerGraph CE 4.2.2 -- interpreted  │
│   GSQL, no accumulators)              │
│                                       │
│  Returns:                             │
│   Nodes: Account / Transaction / Bank │
│   Edges: TRANSFERRED_TO               │
└───────────────────┬───────────────────┘
                    │  raw subgraph
                    ▼
┌───────────────────────────────────────┐
│       Structural Pattern Analysis     │
│                                       │
│  Fan-In    : top 5 accounts by        │
│              unique sender count      │
│                                       │
│  Fan-Out   : top 5 accounts by        │
│              unique receiver count    │
│                                       │
│  Pass-Thru : accounts appearing in    │
│              both (intermediaries)    │
│                                       │
│  Circular  : A->B->A reciprocal pairs │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│         Graph Context Builder         │
│                                       │
│  Max chars : 5,500                    │
│                                       │
│  Sections:                            │
│   - Network header (node/edge counts) │
│   - Structural patterns               │
│   - Top transactions (prioritised:    │
│     flagged > cross-bank >            │
│     cross-currency > high-value)      │
│   - Account neighbourhood by hop      │
│   - Bank routing summary              │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│           Prompt Builder              │
│                                       │
│  Query intent -> factual /            │
│                  investigation        │
│                                       │
│  factual      -> "Extract ranking     │
│                  from STRUCTURAL      │
│                  PATTERNS"            │
│  investigation -> 8-step AML analysis │
└───────────────────┬───────────────────┘
                    │
                    ▼
              LLM Generation
              (same fallback chain)
                    │
                    ▼
              Answer + Metrics
              GRAPH_RAG_METRICS
```

### Graph Schema

```
         ┌──────────┐
         │   Bank   │
         └────▲─────┘
              │ BELONGS_TO_BANK
              │
┌─────────────┴──────────────────────────────┐
│                  Account                   │
│  account_id, home_bank, entity_name        │
└──────┬──────────────────────────┬──────────┘
       │ INITIATED_TRANSACTION    │ RECEIVED_TRANSACTION
       ▼                          ▼
┌────────────────────────────────────────────┐
│               Transaction                  │
│  amount_paid, payment_currency             │
│  payment_format, timestamp                 │
│  is_laundering, is_cross_bank              │
│  is_cross_currency, transfer_type          │
└────────────────────────────────────────────┘

  Account --[TRANSFERRED_TO]--> Account
  (direct shortcut edge for fast BFS fan-out/fan-in queries)
```

### Key Files

| File | Lines | Purpose |
|---|---|---|
| `pipelines/graph_rag/generate_answer.py` | 854 | Pipeline driver + context builder |
| `pipelines/graph_rag/retrieve.py` | 971 | TigerGraph traversal logic |
| `pipelines/graph_rag/build_graph_data.py` | 451 | CSV to TigerGraph transformation |

### CLI

```bash
python pipelines/graph_rag/generate_answer.py \
  --query "Investigate circular money movement patterns" \
  --account 8042DF7E0 \
  --depth 2 \
  --model gemini \
  --gemini_model gemini-2.5-flash
```

### Metrics Output

```
GRAPH_RAG_METRICS | backend=gemini | model=gemini-2.5-flash | nodes=46 | edges=200
  | retrieval_ms=4821.3 | llm_ms=5103.2 | total_ms=9924.5
  | prompt_tokens=1115 | response_tokens=57 | status=success | error_type=none
```

### Why GraphRAG Wins on Structural Queries

| Query | LLM-Only | Basic RAG | GraphRAG |
|---|---|---|---|
| "Top 5 accounts by recipient count" | Fabricates ~5 recipients | 1 recipient per retrieved doc | **32 unique recipients — real graph count** |
| "Which accounts are both senders and receivers?" | Generic description | Retrieves unrelated narratives | **Computes pass-through nodes exactly** |
| "Are there circular transfers A->B->A?" | Describes the concept | May retrieve one A->B transaction | **Detects reciprocal edge pairs** |
| "What accounts connect to 8042DF7E0?" | No knowledge of this account | Random semantic matches | **BFS returns 46 connected accounts** |

---

## Ingestion Pipeline

Three-stage pipeline transforms raw CSV into both ChromaDB vectors and TigerGraph nodes/edges.

```
HI-Small_Trans.csv (454 MB)          HI-Small_accounts.csv (33 MB)
        │                                        │
        └──────────────┬──────────────────────────┘
                       ▼
         ┌─────────────────────────────┐
         │   Stage 1: Narrative Gen    │
         │   build_narratives.py       │
         │                             │
         │   Per transaction:          │
         │   "Account X (Corp #N,      │
         │    bank B) sent $A to       │
         │    account Y via Format     │
         │    on Date."                │
         │                             │
         │   Output: 2.7 GB JSONL      │
         │   (2M narrative records)    │
         └──────────────┬──────────────┘
                        │
           ┌────────────┼──────────────────┐
           ▼            ▼                  ▼
  ┌──────────────┐ ┌──────────┐  ┌──────────────────┐
  │  Stage 2A:   │ │ Stage 2B:│  │   Stage 2C:      │
  │  Chunking    │ │  Graph   │  │   Graph Load     │
  │              │ │  Build   │  │                  │
  │  chunk_      │ │  build_  │  │  Upload CSVs to  │
  │  narratives  │ │  graph_  │  │  TigerGraph via  │
  │  .py         │ │  data.py │  │  GSQL DDL        │
  └──────┬───────┘ └────┬─────┘  └──────────────────┘
         │              │
         ▼              ▼
  ┌──────────────┐ ┌──────────────────────────────────┐
  │  Stage 3:    │ │  datasets/graph/ (511 MB total)   │
  │  Embedding   │ │  accounts.csv        20 MB        │
  │              │ │  banks.csv          263 KB        │
  │  build_      │ │  entities.csv         9.2 MB      │
  │  embeddings  │ │  transactions.csv   174 MB        │
  │  .py         │ │  transferred_edges  152 MB        │
  │              │ │  initiated_edges     75 MB        │
  │  all-MiniLM  │ │  received_edges      75 MB        │
  │  -L6-v2      │ │  bank_edges           8.9 MB      │
  └──────┬───────┘ └──────────────────────────────────┘
         │
         ▼
  ChromaDB (~2M vectors)
```

### Ingestion Scripts

| Script | Input | Output | Key Detail |
|---|---|---|---|
| `ingestion/narrative_generation/build_narratives.py` | Raw CSVs | `transaction_narratives.jsonl` (2.7 GB) | Streams in 100k-row chunks; enriches with entity names |
| `ingestion/chunking/chunk_narratives.py` | Narratives JSONL | Chunked JSONL | Fixed-size or sliding-window chunks |
| `ingestion/embeddings/build_embeddings.py` | Chunks | ChromaDB collection | all-MiniLM-L6-v2, 384-dim, stores full metadata |
| `pipelines/graph_rag/build_graph_data.py` | Narratives JSONL | 8 node/edge CSVs | Streams 50k rows/chunk; 4 vertex types + 4 edge types |

---

## Query Intent Classification

**File**: `pipelines/query_intent.py`

Routes each query to the appropriate response format before prompting the LLM.

```
Query text
    │
    ▼
┌──────────────────────────────────────────────────────┐
│               Keyword Scoring                        │
│                                                      │
│  FACTUAL keywords (+1 each):                         │
│   "top ", "most ", "highest", "how many",            │
│   "list ", "rank", "which account", "sorted by"      │
│                                                      │
│  INVESTIGATION keywords (+1 each):                   │
│   "suspicious", "risk", "laundering", "fraud",       │
│   "circular", "pattern", "structuring", "detect"     │
└──────────────────────┬───────────────────────────────┘
                       │
           ┌───────────┴───────────┐
    factual_score > 0         invest_score > factual
    AND >= invest_score?
           │                       │
           ▼                       ▼
       "factual"           "investigation"
           │                       │
           ▼                       ▼
    Numbered list format    8-step risk report format
```

**Impact per pipeline:**

| Pipeline | Factual response | Investigation response |
|---|---|---|
| LLM-only | Direct answer, no fabrication | General AML pattern description |
| Basic RAG | "Answer DIRECTLY as numbered list" | Full context-grounded risk report |
| GraphRAG | "Extract ranking from STRUCTURAL PATTERNS" | 8-step AML analysis with graph evidence |

---

## Streamlit Dashboard

**File**: `streamlit_app.py` (882 lines)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG Benchmark Lab                               │
├───────────────────┬─────────────────────────────────────────────────────┤
│   SIDEBAR         │  TAB 1: Database Overview                           │
│                   │  TAB 2: Run Query                                   │
│  Quick Stats:     │  TAB 3: Results & Evaluation                        │
│  2M transactions  ├─────────────────────────────────────────────────────┤
│  510k accounts    │                                                     │
│  30k banks        │  TAB 2 Layout:                                      │
│  Sep 2022         │                                                     │
│                   │  ┌─────────────────────────────────────────────┐    │
│  Query ranges:    │  │  Example Query Browser  (70 queries,        │    │
│  dates, amounts   │  │  7 categories)                              │    │
│  currencies       │  └─────────────────────────────────────────────┘    │
│  payment types    │                                                     │
│                   │  ┌─────────────────────────────────────────────┐    │
│  Pipeline         │  │  Query input text area                      │    │
│  reliability:     │  │  Account ID input (GraphRAG seed)           │    │
│  GraphRAG  ****   │  └─────────────────────────────────────────────┘    │
│  BasicRAG  ***    │                                                     │
│  LLM-Only  *      │  Pipeline Config (3 columns):                       │
│                   │  ┌────────────┐ ┌────────────┐ ┌────────────┐       │
│  AML patterns:    │  │ LLM-Only   │ │ Basic RAG  │ │ GraphRAG   │       │
│  Layering         │  │ backend +  │ │ backend +  │ │ backend +  │       │
│  Smurfing         │  │ model      │ │ model      │ │ model +    │       │
│  Circular         │  └────────────┘ └────────────┘ │ BFS depth  │       │
│  Cross-bank       │                                 └────────────┘      │
│  Currency conv.   │                                                     │
│                   │  [ Run All Pipelines ]                              │
└───────────────────┴─────────────────────────────────────────────────────┘

TAB 3 -- Results Layout:

  ┌──────────────┬──────────────┬──────────────┐
  │   LLM-Only   │  Basic RAG   │  GraphRAG    │
  │  WARNING:    │              │              │
  │  fabricated  │              │              │
  │              │              │              │
  │  Answer text │  Answer text │  Answer text │
  └──────────────┴──────────────┴──────────────┘

  Metrics Table:
  ┌─────────────────┬────────────┬───────────┬───────────┐
  │ Metric          │ LLM-Only   │ Basic RAG │ GraphRAG  │
  ├─────────────────┼────────────┼───────────┼───────────┤
  │ Wall time       │ 84.7s      │ 117.4s    │ 19.4s     │
  │ LLM latency     │ 3,210 ms   │ 8,923 ms  │ 5,103 ms  │
  │ Retrieval ms    │ 0          │ 600 ms    │ 4,821 ms  │
  │ Input tokens    │ 183        │ 1,174     │ 1,115     │
  │ Output tokens   │ 57         │ 312       │ 57        │
  │ Est. cost       │ $0.00004   │ $0.00038  │ $0.00034  │
  │ Retrieved docs  │ 0          │ 5         │ 46 nodes  │
  │ Judge score     │ 2.1/10     │ 5.4/10    │ 8.7/10    │
  │ BERTScore F1    │ 0.71       │ 0.79      │ 0.86      │
  └─────────────────┴────────────┴───────────┴───────────┘

  Visualisations:
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ Latency bar      │  │ Radar chart      │  │ Cost donut       │
  │ (stacked:        │  │ (LLM judge 4     │  │ (% share per     │
  │  retrieval+LLM)  │  │  dimensions)     │  │  pipeline)       │
  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Evaluation Suite

The benchmark uses two complementary evaluation methods that run after every query.

---

### LLM Judge — 6-Dimension Scoring (1–10 each)

An LLM (Gemini Flash, GPT-4o-mini, or Groq Llama) scores each pipeline answer on six dimensions. The `final_score` is a weighted aggregate returned as JSON.

| Dimension | What it measures |
|---|---|
| **Evidence Grounding** | Cites specific accounts, amounts, dates from retrieved evidence. Generic AML boilerplate = 1–3. Named accounts + real numbers = 7–10. GraphRAG graph-computed facts (in=4 out=3, circular pairs) score 8+. |
| **Relationship Reasoning** | Reasons about connections between entities. Multi-hop chains (A→B→C), confirmed circular pairs (A↔B), and pass-through topology score high. Isolated facts score low. |
| **Investigative Insight** | Identifies the laundering phase (placement/layering/integration), classifies risk level, and explains WHY a pattern is suspicious — not just what it is. |
| **Factual Correctness** | All stated facts are consistent with retrieved evidence. Hallucinated account IDs or amounts score 0. LLM-Only is heavily penalised here. |
| **Actionability** | Concrete next steps naming specific account IDs. "File SAR on 8042DF7E0 and freeze 80A4D6EB0 pending MLRO escalation" scores higher than "file a SAR". |
| **Conciseness** | Free of boilerplate, repetition, and disclaimer padding. Dense, precise prose scores 8–10. Bloated hedge-heavy text scores 3–5. |

**What the judge knows about each pipeline:**

| Pipeline | Judge context |
|---|---|
| LLM-Only | No retrieval — any specific account IDs are fabricated. Evidence Grounding capped at 1–2. |
| Basic RAG | Retrieved 5 text chunks via semantic similarity. Facts must match retrieved text. |
| GraphRAG | Traversed a real graph via BFS. Structural facts (in=N, out=M, circular pairs, pass-through counts) are graph-computed verified facts — score Evidence Grounding 8+ for citing these. |

---

### BERTScore — Semantic Similarity vs Expert Reference

Measures semantic overlap between each pipeline's answer and a human-written expert reference answer using `distilbert-base-uncased`. Reports Precision, Recall, and F1.

**Query classification (before scoring):**

| Query type | BERTScore | Reason |
|---|---|---|
| Graph-native: "how many", "count", "top N", contains hex account ID | Skipped | Numerical/structural output doesn't map to a prose reference — use Judge scores |
| Investigative: "suspicious", "circular", "pass-through", "laundering" | Applied | Prose answer can be fairly compared to expert reference |

**Pipeline handling:**

| Pipeline | Treatment |
|---|---|
| LLM-Only | Always N/A — AML vocabulary trivially overlaps with references even when hallucinating. Judge Factual Correctness penalises instead. |
| Basic RAG | Scored if answer has investigative content; hidden (⚠️ GAP) if answer leads with an Evidence Gap disclaimer |
| GraphRAG | Scored normally |

**14 expert reference answers** in `experiments/reference_answers.json` cover: repeated wire transfers, circular transactions, layering behavior, high-frequency transfers, structuring patterns, multi-bank routing, cross-currency conversion, self-transfers, pass-through detection, flagged counterparties, cross-bank cheque/wire, cash fan-out, and suspicious investigation.

**Score interpretation:**

| F1 | Meaning |
|---|---|
| ≥ 0.85 | Strong semantic overlap with expert reference |
| 0.75–0.84 | Moderate overlap |
| < 0.75 | Low overlap — answer diverges from reference |
| ⚠️ GAP | Score hidden — answer is a disclaimer, not analysis |
| N/A | LLM-Only excluded by design |
| — | No expert reference for this query category |

---

### Automated Benchmark

```bash
python experiments/run_benchmark.py    # runs 8 canonical queries × 3 pipelines
python experiments/evaluate_benchmark.py  # BERTScore + judge scoring → reports
```

Outputs go to `experiments/benchmark_results/` (results.csv, results.json, scores.csv, evaluation_report.md).

---

## Performance Comparison

Typical results on AML investigative queries:

| Metric | LLM-Only | Basic RAG | GraphRAG |
|---|---|---|---|
| Wall time | ~70–90s | ~90–120s | **~8–20s** |
| LLM latency | ~3–4s | ~7–10s | ~3–6s |
| Retrieval latency | 0 | ~400–600ms | ~2–5s (BFS) |
| Input tokens | ~200 | ~1,300–1,400 | ~700–1,200 |
| LLM Judge score | 1–3 / 10 | 4–6 / 10 | **7–9 / 10** |
| BERTScore F1 | N/A | 0.62–0.72 | **0.72–0.85** |
| Factual accuracy | Fabricated | Partially grounded | **Graph-verified** |

GraphRAG is **4–6x faster end-to-end** and the only pipeline that produces graph-verified, structurally grounded AML investigation reports.

### Query Type Suitability

| Query Category | Best Pipeline | Reason |
|---|---|---|
| Top-N rankings (fan-out, fan-in) | **GraphRAG** | Counts unique graph neighbours |
| Circular transfers | **GraphRAG** | Detects reciprocal edge pairs |
| Pass-through accounts | **GraphRAG** | Finds accounts with high in- and out-degree |
| Account network analysis | **GraphRAG** | BFS discovers full connected subgraph |
| Narrative / semantic queries | **Basic RAG** | Semantic similarity finds related text |
| General AML pattern knowledge | **Basic RAG** | Retrieved context adds real evidence |
| Speed-critical / simple lookup | **LLM-Only** | Lowest latency when data accuracy not required |

---

## Technology Stack

| Layer | Library / Service | Notes |
|---|---|---|
| **LLM APIs** | google-genai | gemini-2.5-flash default |
| | openai | gpt-4o-mini fallback |
| | ollama | llama3.2 local fallback |
| **Vector DB** | chromadb | Persistent HNSW index |
| **Graph DB** | TigerGraph CE 4.2.2 | Interpreted GSQL; no accumulators |
| | pyTigerGraph | Python REST client |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2, 384-dim |
| **Data** | pandas | CSV streaming + aggregation |
| **UI** | streamlit | Interactive dashboard |
| **Visualisation** | plotly | Radar, bar, donut charts |
| **Evaluation** | bert-score | BERTScore F1 |
| **Utilities** | python-dotenv, tqdm, requests | |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- TigerGraph CE 4.2.2 running on `localhost:14240` (GraphRAG pipeline only)
- API keys: Gemini and/or OpenAI

### Install

```bash
git clone <repo>
cd rag-benchmark-lab
pip install -r requirements.txt
```

### Environment

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
TIGERGRAPH_HOST=http://localhost
TIGERGRAPH_USERNAME=tigergraph
TIGERGRAPH_PASSWORD=tigergraph
TIGERGRAPH_TOKEN=
GRAPH_NAME=FinancialGraph
```

### Data Ingestion (first-time setup)

```bash
# 1. Generate narrative text from raw transactions
python ingestion/narrative_generation/build_narratives.py

# 2. Chunk narratives for vector ingestion
python ingestion/chunking/chunk_narratives.py

# 3. Build ChromaDB vector store
python ingestion/embeddings/build_embeddings.py

# 4. Build TigerGraph CSV files
python pipelines/graph_rag/build_graph_data.py

# 5. Load CSVs into TigerGraph (via TigerGraph Studio or GSQL)
```

---

## Usage

### Streamlit Dashboard (recommended)

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

### Run Individual Pipelines

```bash
# LLM-Only
python pipelines/llm_only/generate_answer.py \
  --query "Top 5 accounts with maximum outgoing transfers"

# Basic RAG
python pipelines/basic_rag/generate_answer.py \
  --query "Repeated Cash transactions in last 3 days"

# GraphRAG
python pipelines/graph_rag/generate_answer.py \
  --query "Investigate circular money movement" \
  --account 8042DF7E0 \
  --depth 2
```

### Run Full Benchmark

```bash
python experiments/run_benchmark.py
python experiments/evaluate_benchmark.py
```

---

## Known Queryable Accounts

These accounts have well-characterised networks in TigerGraph:

| Account ID | Entity | Network Profile |
|---|---|---|
| `8042DF7E0` | Corporation #6579 | 46 connected accounts, 161 txns, 28 banks, 4-hop depth |
| `80A4D6EB0` | Sole Proprietorship #6323 | Connected to 8042DF7E0 network |
| `809C54730` | Partnership #14283 | Pass-through intermediary |
| `8042E0040` | Partnership #577 | Multi-bank routing account |
| `81076F8E0` | Unknown entity | Self-transfer account (1 acct, 2 txns) |

---

## Benchmark Results Summary

Headline numbers across all three pipelines on AML investigative queries (seed account `8042E0040`, 5 canonical queries):

### Token Usage & Cost

| Pipeline | Avg Input Tokens | Avg Output Tokens | Avg Cost / Query |
|---|---|---|---|
| LLM-Only | ~200 | ~70 | ~$0.000040 |
| Basic RAG | ~1,370 | ~300 | ~$0.000380 |
| **GraphRAG** | **~950** | **~100** | **~$0.000180** |

- **GraphRAG uses ~31% fewer input tokens than Basic RAG** (dense structured graph context vs 5 verbose text chunks)
- **GraphRAG costs ~53% less per query than Basic RAG** on average
- LLM-Only is cheapest but produces fabricated results — not usable for investigations

### Latency

| Pipeline | Avg Wall Time | Retrieval | LLM Generation |
|---|---|---|---|
| LLM-Only | ~75s | 0ms | ~3,500ms |
| Basic RAG | ~100s | ~500ms | ~8,500ms |
| **GraphRAG** | **~12s** | ~3,500ms (BFS) | ~2,800ms |

- **GraphRAG is 5–8x faster end-to-end than Basic RAG** — primarily because it sends fewer tokens to the LLM
- Basic RAG's high wall time is driven by subprocess startup + embedding + large LLM context
- TigerGraph BFS (3–5s) is slower than ChromaDB ANN search (400–600ms) but the LLM step is much faster

### Accuracy: LLM Judge + BERTScore

| Pipeline | Avg Judge Score (/ 10) | Judge Pass Rate (≥ 7) | Avg BERTScore F1 |
|---|---|---|---|
| LLM-Only | 1.5–2.5 | 0% | N/A (excluded) |
| Basic RAG | 4.5–6.0 | 0–10% | 0.62–0.72 |
| **GraphRAG** | **7.5–8.5** | **80–100%** | **0.72–0.85** |

- GraphRAG achieves **7.5–8.5 / 10** on the 6-dimension LLM judge by citing confirmed graph facts (circular pairs, pass-through degree, cross-bank counts)
- Basic RAG averages 4.5–6.0 — penalised for generic AML boilerplate and Evidence Gap responses when the queried account isn't in the retrieved chunks
- LLM-Only averages 1.5–2.5 — heavily penalised for fabricated account IDs and zero Evidence Grounding

### Summary Verdict

| Category | Winner |
|---|---|
| Lowest latency | GraphRAG (5–8x faster than Basic RAG) |
| Fewest tokens | GraphRAG (~31% fewer than Basic RAG) |
| Lowest cost | GraphRAG (~53% cheaper than Basic RAG) |
| Best judge score | GraphRAG (7.5–8.5 vs 4.5–6.0 for Basic RAG) |
| Best BERTScore F1 | GraphRAG (0.72–0.85 vs 0.62–0.72 for Basic RAG) |
| Best for semantic queries | Basic RAG |

---

## Sample Queries

### Best Queries for GraphRAG (BERTScore + Judge both score high)

Use seed account `8042E0040` for all of these:

```
Circular money movement where funds return to the originating account
Account acting as a pass-through receiving and immediately re-sending funds
Multiple rapid transfers through intermediary accounts to obscure fund origin
Investigate suspicious cross-bank laundering behavior involving intermediary accounts
Large transactions routed through multiple banks across jurisdictions
```

### AML Investigation Queries (all pipelines produce answers)

```
Suspicious repeated wire transfers between foreign banks in different countries
High frequency small transfers from one account to many recipients within hours
Transactions just below reporting thresholds from the same account repeated daily
Rapid conversion between currencies across multiple banks with no clear business purpose
Large self-transfers within the same account at unusual hours
```

### Factual / Ranking Queries (GraphRAG only — requires seed account)

```
Top 5 accounts with maximum outgoing transfers
Which accounts receive funds from the most unique senders?
Are there any accounts that transfer money back and forth between each other?
What accounts are connected to account 8042DF7E0?
```

> For factual queries mentioning a specific hex account ID, set the GraphRAG seed to that same account. The dashboard warns you if they differ.

---

## Project Structure

```
rag-benchmark-lab/
├── .env                              # API keys
├── requirements.txt                  # Python dependencies
├── streamlit_app.py                  # Interactive dashboard (882 lines)
│
├── pipelines/
│   ├── query_intent.py               # Factual vs investigation classifier
│   ├── llm_only/
│   │   └── generate_answer.py        # LLM-only baseline (619 lines)
│   ├── basic_rag/
│   │   ├── generate_answer.py        # Basic RAG pipeline (626 lines)
│   │   └── retrieve.py               # ChromaDB retrieval (326 lines)
│   └── graph_rag/
│       ├── generate_answer.py        # GraphRAG pipeline (854 lines)
│       ├── retrieve.py               # TigerGraph traversal (971 lines)
│       └── build_graph_data.py       # CSV to graph transform (451 lines)
│
├── ingestion/
│   ├── narrative_generation/
│   │   └── build_narratives.py       # CSV to JSONL narratives (296 lines)
│   ├── chunking/
│   │   └── chunk_narratives.py       # Narrative chunking (254 lines)
│   └── embeddings/
│       └── build_embeddings.py       # ChromaDB ingestion (499 lines)
│
├── datasets/
│   ├── raw/                          # Source CSVs (487 MB)
│   ├── processed/                    # Narrative JSONL (2.7 GB)
│   ├── graph/                        # TigerGraph CSVs (511 MB)
│   └── chromadb/                     # ChromaDB collection
│
├── experiments/
│   ├── benchmark_queries.json        # 8 canonical AML queries
│   ├── example_queries.json          # 70 example queries (7 categories)
│   ├── reference_answers.json        # Ground-truth answers for BERTScore
│   ├── run_benchmark.py              # Automated benchmark runner (527 lines)
│   ├── evaluate_benchmark.py         # Scoring & evaluation (405 lines)
│   └── benchmark_results/            # CSV / JSON / MD outputs
│
└── database/
    └── chroma/                       # ChromaDB persistent storage
```
