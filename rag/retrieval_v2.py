from __future__ import annotations

"""Multi-channel SQLite retrieval with bounded candidate generation and diagnostics."""
import re
import time
from collections import defaultdict
from typing import Any

try:
    from .core import japanese_terms
except ImportError:
    from core import japanese_terms


_SYNONYMS = {
    "検索": ("全文検索", "索引", "インデックス"),
    "インデックス": ("索引", "検索", "チャンク化", "FTS5", "登録"),
    "構築": ("登録", "作成", "索引化"),
    "メモリ": ("RAM", "容量", "ピークメモリ"),
    "共存": ("同時実行", "同一環境", "メモリ競合"),
    "モデル": ("推論", "モデルロード"),
    "DB": ("データベース", "SQLite", "検索DB"),
    "登録": ("格納", "保存", "追加"),
    "チャンク": ("分割", "分割単位"),
}
_RRF_K = 60


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().lower()


def expand_query(query: str) -> list[str]:
    normalized = normalize_query(query)
    terms = list(dict.fromkeys(japanese_terms(normalized)))
    expanded = list(terms)
    for concept, synonyms in _SYNONYMS.items():
        if concept in normalized:
            for synonym in synonyms:
                expanded.extend([synonym.lower(), *japanese_terms(synonym)])
    return list(dict.fromkeys(expanded))


def _exact_terms(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(
        r"[A-Za-z_][A-Za-z0-9_.-]*|\d+(?:\.\d+)?\s*(?:GB|MB|GiB|MiB|%)?",
        query, re.I)))


def _match_expression(terms: list[str]) -> str:
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _rrf(rankings: dict[str, list[dict]]) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for rows in rankings.values():
        for rank, row in enumerate(rows, 1):
            scores[int(row["id"])] += 1.0 / (_RRF_K + rank)
    return scores


def search_with_diagnostics(store: Any, query: str, limit: int = 5) -> tuple[list[dict], dict]:
    """Fuse expanded FTS, original-query bigrams and literal exact matches.

    Candidate generation stays inside SQLite FTS5. It never materializes the
    entire chunks table into Python, unlike the first prototype implementation.
    """
    started = time.perf_counter()
    normalized = normalize_query(query)
    terms = list(dict.fromkeys(japanese_terms(normalized)))
    expanded = expand_query(normalized)
    exact = _exact_terms(query)
    limit = max(1, min(int(limit), 100))
    diagnostics: dict[str, Any] = {
        "original_query": query, "normalized_query": normalized,
        "query_terms": terms, "expanded_terms": expanded,
        "ngrams": [t for t in terms if len(t) == 2],
        "channels": {}, "candidate_ranks": [], "rejected_candidates": [],
        "failure_classification": [],
    }
    if not terms and not exact:
        diagnostics["failure_classification"] = ["QUERY_TOKENIZATION_FAILURE"]
        diagnostics["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        diagnostics["result_count"] = 0
        return [], diagnostics

    db = store.db
    cap = max(40, limit * 8)
    rankings: dict[str, list[dict]] = {}

    def fts_channel(name: str, channel_terms: list[str]) -> list[dict]:
        if not channel_terms:
            return []
        try:
            rows = db.execute("""SELECT c.id,c.content,d.title,d.path,bm25(chunks_fts) AS rank
                FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
                JOIN documents d ON d.id=c.document_id
                WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
                (_match_expression(channel_terms), cap)).fetchall()
            return [{"id": int(r["id"]), "text": r["content"], "source": r["title"],
                     "path": r["path"], "raw_score": float(r["rank"])} for r in rows]
        except Exception as exc:
            diagnostics["channels"][name + "_error"] = type(exc).__name__
            return []

    # Expanded recall channel and a separate original-query channel.
    rankings["fts5_bm25"] = fts_channel("fts5_bm25", list(dict.fromkeys(terms + expanded)))
    rankings["bigram"] = fts_channel("bigram", terms)

    exact_rows = []
    if exact:
        clauses = " OR ".join("instr(lower(c.content), lower(?)) > 0" for _ in exact)
        exact_rows = db.execute(f"""SELECT c.id,c.content,d.title,d.path
            FROM chunks c JOIN documents d ON d.id=c.document_id
            WHERE {clauses} LIMIT ?""", (*exact, cap)).fetchall()
    rankings["exact_match"] = [
        {"id": int(r["id"]), "text": r["content"], "source": r["title"],
         "path": r["path"], "raw_score": 1.0} for r in exact_rows]

    fused = _rrf(rankings)
    max_rrf = max(fused.values(), default=0.0)
    by_id: dict[int, dict] = {}
    for channel, rows in rankings.items():
        diagnostics["channels"].setdefault(channel, {
            "candidate_count": len(rows), "top_ids": [row["id"] for row in rows[:10]],
        })
        for rank, row in enumerate(rows, 1):
            item = by_id.setdefault(row["id"], dict(row))
            item.setdefault("channel_ranks", {})[channel] = rank

    query_set = set(terms)
    for row in by_id.values():
        doc_terms = set(japanese_terms(row["text"]))
        lexical = len(query_set & doc_terms) / max(1, len(query_set))
        consensus = fused[row["id"]] / max_rrf if max_rrf else 0.0
        # Preserve the answering gate's established score semantics:
        # lexical coverage is primary; normalized channel consensus is a tie-break.
        row["lexical_overlap"] = lexical
        row["rrf_score"] = round(fused[row["id"]], 6)
        row["channel_count"] = len(row.get("channel_ranks", {}))
        row["score"] = round(0.75 * lexical + 0.25 * consensus, 4)

    ordered = sorted(by_id.values(), key=lambda row: (
        row["score"], row["channel_count"], row["rrf_score"]), reverse=True)
    selected = ordered[:limit]
    selected_ids = {row["id"] for row in selected}
    diagnostics["candidate_ranks"] = [
        {"id": row["id"], "source": row["source"], "rank": i + 1,
         "score": row["score"], "lexical_overlap": round(row["lexical_overlap"], 4),
         "channels": row["channel_ranks"], "rrf_score": row["rrf_score"]}
        for i, row in enumerate(ordered[:40])]
    diagnostics["rejected_candidates"] = [
        {"id": row["id"], "source": row["source"], "reason": "candidate_cutoff"}
        for row in ordered if row["id"] not in selected_ids][:20]
    ]
    if not selected:
        classes = []
        if not terms and not exact:
            classes.append("QUERY_TOKENIZATION_FAILURE")
        elif not expanded:
            classes.append("QUERY_EXPANSION_FAILURE")
        elif not any(diagnostics["channels"].get(k, {}).get("candidate_count", 0)
                     for k in ("fts5_bm25", "bigram", "exact_match")):
            classes.extend(["LEXICAL_MISMATCH", "RETRIEVAL_CONFIDENCE_LOW"])
        diagnostics["failure_classification"] = classes
    diagnostics["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    diagnostics["result_count"] = len(selected)
    diagnostics["candidate_count"] = len(ordered)
    return [{k: v for k, v in row.items() if k != "id"} for row in selected], diagnostics
