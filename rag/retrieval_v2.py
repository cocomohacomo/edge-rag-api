from __future__ import annotations

"""Lightweight multi-channel retrieval with explainable diagnostics.

This module deliberately keeps SQLite/FTS5 as the only storage dependency.
Query expansion affects retrieval only; it is never treated as answer evidence.
"""
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


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().lower()


def expand_query(query: str) -> list[str]:
    """Return unique original and conservative Japanese synonym terms."""
    terms = list(dict.fromkeys(japanese_terms(normalize_query(query))))
    expanded = list(terms)
    for term in terms:
        for synonym in _SYNONYMS.get(term, ()):
            expanded.extend(japanese_terms(synonym))
    return list(dict.fromkeys(expanded))


def _exact_terms(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(
        r"[A-Za-z_][A-Za-z0-9_.-]*|\d+(?:\.\d+)?\s*(?:GB|MB|GiB|MiB|%)?",
        query, re.I)))


def _rrf(rankings: dict[str, list[dict]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for rows in rankings.values():
        for rank, row in enumerate(rows, 1):
            scores[int(row["id"])] += 1.0 / (k + rank)
    return scores


def search_with_diagnostics(store: Any, query: str, limit: int = 5) -> tuple[list[dict], dict]:
    """Fuse FTS5, exact identifier/numeric and bigram channels using RRF."""
    started = time.perf_counter()
    normalized = normalize_query(query)
    terms = japanese_terms(normalized)
    expanded = expand_query(normalized)
    exact = _exact_terms(query)
    diagnostics: dict[str, Any] = {
        "original_query": query, "normalized_query": normalized,
        "query_terms": terms, "expanded_terms": expanded,
        "ngrams": [t for t in terms if len(t) == 2],
        "channels": {}, "rejected_candidates": [],
        "failure_classification": [],
    }
    if not terms and not exact:
        diagnostics["failure_classification"] = ["QUERY_TOKENIZATION_FAILURE"]
        diagnostics["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return [], diagnostics

    rankings: dict[str, list[dict]] = {}
    db = store.db
    # Existing FTS5/BM25 channel, preserving the current index and ranking.
    fts_terms = list(dict.fromkeys(terms + expanded))
    if fts_terms:
        expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in fts_terms)
        try:
            rows = db.execute("""SELECT c.id,c.content,d.title,d.path,bm25(chunks_fts) AS rank
                FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
                JOIN documents d ON d.id=c.document_id
                WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
                (expr, max(40, limit * 8))).fetchall()
            rankings["fts5_bm25"] = [
                {"id": int(r["id"]), "text": r["content"], "source": r["title"],
                 "path": r["path"], "raw_score": float(r["rank"])} for r in rows]
        except Exception as exc:
            diagnostics["channels"]["fts5_bm25_error"] = type(exc).__name__
            rankings["fts5_bm25"] = []

    # Exact identifier/numeric channel. Literal matching avoids tokenizer loss
    # for names such as i5-10310U, FTS syntax, paths and numeric constraints.
    exact_rows = []
    if exact:
        clauses = " OR ".join("instr(lower(c.content), lower(?)) > 0" for _ in exact)
        exact_rows = db.execute(f"""SELECT c.id,c.content,d.title,d.path
            FROM chunks c JOIN documents d ON d.id=c.document_id
            WHERE {clauses} LIMIT ?""", (*exact, max(40, limit * 8))).fetchall()
    rankings["exact_match"] = [
        {"id": int(r["id"]), "text": r["content"], "source": r["title"],
         "path": r["path"], "raw_score": 1.0} for r in exact_rows]

    # Bigram channel independent of FTS query parser behavior.
    bigram_set = set(terms)
    bigram_rows = db.execute("""SELECT c.id,c.content,d.title,d.path,c.terms
        FROM chunks c JOIN documents d ON d.id=c.document_id""").fetchall()
    bigram_scored = []
    for row in bigram_rows:
        doc_terms = set(str(row["terms"]).split())
        overlap = len(bigram_set & doc_terms) / max(1, len(bigram_set))
        if overlap > 0:
            bigram_scored.append((overlap, row))
    bigram_scored.sort(key=lambda x: x[0], reverse=True)
    rankings["bigram"] = [
        {"id": int(r["id"]), "text": r["content"], "source": r["title"],
         "path": r["path"], "raw_score": score}
        for score, r in bigram_scored[:max(40, limit * 8)]]

    fused = _rrf(rankings)
    by_id: dict[int, dict] = {}
    for channel, rows in rankings.items():
        diagnostics["channels"][channel] = {
            "candidate_count": len(rows),
            "top_ids": [row["id"] for row in rows[:10]],
        }
        for row in rows:
            item = by_id.setdefault(row["id"], dict(row))
            item.setdefault("channel_ranks", {})[channel] = rows.index(row) + 1
    ordered = sorted(by_id.values(), key=lambda row: fused.get(row["id"], 0), reverse=True)
    for row in ordered:
        row["rrf_score"] = round(fused[row["id"]], 6)
        row["channel_count"] = len(row.get("channel_ranks", {}))
        row["score"] = round(min(1.0, row["rrf_score"] * 20), 4)
    selected = ordered[:max(1, limit)]
    selected_ids = {row["id"] for row in selected}
    diagnostics["candidate_ranks"] = [
        {"id": row["id"], "source": row["source"], "rank": i + 1,
         "channels": row["channel_ranks"], "rrf_score": row["rrf_score"]}
        for i, row in enumerate(ordered[:40])]
    diagnostics["rejected_candidates"] = [
        {"id": row["id"], "source": row["source"], "reason": "candidate_cutoff"}
        for row in ordered if row["id"] not in selected_ids][:20]
    if not selected:
        classes = []
        if not terms and not exact:
            classes.append("QUERY_TOKENIZATION_FAILURE")
        elif not expanded:
            classes.append("QUERY_EXPANSION_FAILURE")
        elif not bigram_rows:
            classes.append("DOCUMENT_INDEX_FAILURE")
        else:
            classes.extend(["LEXICAL_MISMATCH", "RETRIEVAL_CONFIDENCE_LOW"])
        diagnostics["failure_classification"] = classes
    elif len(selected) < min(limit, len(ordered)):
        diagnostics["failure_classification"] = ["CANDIDATE_CUTOFF_FAILURE"]
    diagnostics["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    diagnostics["result_count"] = len(selected)
    return [{k: v for k, v in row.items() if k != "id"} for row in selected], diagnostics
