"""
query_intent.py — Query type detection for adaptive prompt selection.

Detects whether a query is asking for a direct factual answer (ranking, count,
list) or an AML investigation/risk analysis, so each pipeline can choose the
appropriate response format instead of forcing every query into a risk report.
"""

_FACTUAL_KEYWORDS = [
    "top ", "top-", "top5", "top10",
    "most ", "maximum", "highest", "largest", "biggest",
    "minimum", "lowest", "least ", "smallest",
    "how many", "count ", "number of",
    "which account", "which accounts",
    "list ", "show me", "show all", "give me", "find accounts",
    "rank ", "ranking", "sorted by", "order by",
    "average", "total ", "sum ",
    "who sends", "who receives", "who transferred",
    "what is the", "what are the",
]

_INVESTIGATION_KEYWORDS = [
    "investigate", "suspicious", "detect", "identify", "flag",
    "analyse", "analyze", "risk", "laundering", "fraud",
    "layering", "structuring", "smurfing", "circular",
    "cross-bank", "cross bank", "intermediary", "pass-through",
    "pattern", "behavior", "behaviour", "unusual",
    "high-volume", "high volume", "network of", "connect to",
    "money flow", "fund flow", "suspicious activity",
]


def classify_query(query: str) -> str:
    """
    Return 'factual' or 'investigation' based on query keywords.

    factual      → user wants a list / ranking / count / direct lookup
    investigation → user wants AML risk analysis / pattern detection
    """
    q = query.lower()
    factual_score = sum(1 for kw in _FACTUAL_KEYWORDS if kw in q)
    invest_score  = sum(1 for kw in _INVESTIGATION_KEYWORDS if kw in q)

    # Factual only when clearly dominant — investigation wins on ties
    if factual_score > 0 and factual_score > invest_score:
        return "factual"
    return "investigation"
