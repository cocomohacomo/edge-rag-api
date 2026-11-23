from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable
import json
import urllib.request

try:
    from .core import japanese_terms
except ImportError:  # direct execution via rag/cli.py
    from core import japanese_terms


ABSTAIN = "関連する根拠が十分ではないため、回答できません。"


@dataclass(frozen=True)
class RetrievalDecision:
    accepted: tuple[dict, ...]
    rejected: tuple[dict, ...]
    confidence: float
    reason: str


def _overlap(question: str, text: str) -> float:
    q = set(japanese_terms(question))
    if not q:
        return 0.0
    d = set(japanese_terms(text))
    return len(q & d) / len(q)


def retrieval_gate(
    question: str,
    hits: Iterable[dict],
    *,
    min_score: float = 0.08,
    min_overlap: float = 0.10,
    limit: int = 3,
) -> RetrievalDecision:
    """Filter lexical false positives before generation; no model or embedding required."""
    ranked = []
    rejected = []
    for hit in hits:
        overlap = _overlap(question, hit.get("text", ""))
        score = float(hit.get("score", 0.0))
        item = {**hit, "query_overlap": round(overlap, 4)}
        # Either a strong combined rank or a minimum lexical coverage is required.
        if score >= min_score and overlap >= min_overlap:
            ranked.append(item)
        else:
            rejected.append(item)
    ranked.sort(key=lambda x: (x["query_overlap"], float(x.get("score", 0))), reverse=True)
    accepted = tuple(ranked[:max(1, limit)])
    confidence = max((x["query_overlap"] for x in accepted), default=0.0)
    reason = "relevant_evidence" if accepted else "no_relevant_evidence"
    return RetrievalDecision(accepted, tuple(rejected), confidence, reason)


def build_evidence_prompt(question: str, hits: Iterable[dict], max_chars: int = 1400) -> str:
    """Compact evidence-first prompt with explicit abstention and answer-shape rules."""
    blocks = []
    used = 0
    for i, hit in enumerate(hits, 1):
        text = re.sub(r"\s+", " ", str(hit.get("text", ""))).strip()
        source = str(hit.get("source", "unknown"))
        block = f"[根拠{i}] {source}: {text}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        blocks.append(block[:remaining])
        used += min(len(block), remaining)
    evidence = "\n".join(blocks) if blocks else "（根拠なし）"
    return (
        "日本語で回答。根拠文書に明記された事実だけを使い、質問に直接1文で答える。"
        "推測・一般論・前置きは禁止。根拠がなければ「文書に記載がありません」と答える。\n"
        f"Q:{question}\n根拠:{evidence}\nA:"
    )


def validate_answer(answer: str, question: str, hits: Iterable[dict]) -> tuple[bool, str]:
    """Conservative deterministic guard against empty, off-topic and unsupported numeric output."""
    cleaned = re.sub(r"^回答[:：]?\s*", "", (answer or "").strip())
    cleaned = re.sub(r"\n?\[\d+\]\s*$", "", cleaned).strip()
    if not cleaned:
        return False, "empty"
    if re.search(r"</?(?:system_thinking|think|answer|analysis)[^>]*>", cleaned, re.I):
        return False, "internal_tag"
    if re.search(r"(?:質問|根拠文書|Q|A)[:：]\s*$", cleaned):
        return False, "incomplete_tail"
    if cleaned in {"文書に記載がありません", ABSTAIN}:
        return True, "abstention"
    sources = list(hits)
    context = " ".join(str(h.get("text", "")) for h in sources)
    answer_terms = set(japanese_terms(cleaned))
    context_terms = set(japanese_terms(context))
    if not answer_terms or len(answer_terms & context_terms) / max(1, len(answer_terms)) < 0.45:
        return False, "insufficient_evidence_overlap"
    # Numeric claims are high-risk in meeting minutes and specifications.
    numbers = re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?:%|GB|MB|日|月|年|時|分)?", cleaned, re.I)
    context_numbers = set(re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?:%|GB|MB|日|月|年|時|分)?", context, re.I))
    if any(n not in context_numbers for n in numbers):
        return False, "unsupported_number"
    q_terms = set(japanese_terms(question))
    if q_terms and len(answer_terms & q_terms) == 0 and len(answer_terms & context_terms) < 3:
        return False, "question_mismatch"
    return True, "accepted"


def extractive_fallback(question: str, hits: Iterable[dict], max_sentences: int = 3) -> str:
    """Return the best source sentences, preserving provenance and avoiding free-form claims."""
    candidates = []
    q_terms = set(japanese_terms(question))
    for hit in hits:
        text = str(hit.get("text", "")).strip()
        for sentence in re.split(r"(?<=[。！？!?])\s*|\n+", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = set(japanese_terms(sentence))
            coverage = len(q_terms & terms) / max(1, len(q_terms))
            if coverage > 0:
                candidates.append((coverage, len(sentence), sentence, hit.get("source", "unknown")))
    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    selected = []
    seen = set()
    for _, _, sentence, source in candidates:
        if sentence in seen:
            continue
        seen.add(sentence)
        selected.append(f"{sentence}（出典: {source}）")
        if len(selected) >= max_sentences:
            break
    return "\n".join(selected) if selected else ABSTAIN


def answer_with_pipeline(
    question: str,
    hits: Iterable[dict],
    generator: Callable[[str], str] | None = None,
    *,
    retry: bool = False,
) -> dict:
    """Gate -> evidence prompt -> validate -> one constrained retry -> extractive fallback."""
    decision = retrieval_gate(question, hits)
    if not decision.accepted:
        return {"answer": ABSTAIN, "status": "abstained", "reason": decision.reason,
                "retrieval": decision}
    selected = list(decision.accepted)
    if generator is not None:
        prompt = build_evidence_prompt(question, selected)
        try:
            candidate = generator(prompt)
        except Exception:
            candidate = ""
        valid, reason = validate_answer(candidate, question, selected)
        if valid and reason != "abstention":
            return {"answer": candidate.strip(), "status": "generated", "reason": "accepted",
                    "retrieval": decision}
        if retry:
            retry_prompt = prompt + "\n直前の回答は検証に失敗しました。根拠中の回答可能な事実だけを1文で回答してください。"
            try:
                candidate = generator(retry_prompt)
            except Exception:
                candidate = ""
            valid, reason = validate_answer(candidate, question, selected)
            if valid and reason != "abstention":
                return {"answer": candidate.strip(), "status": "regenerated", "reason": "accepted",
                        "retrieval": decision}
    return {"answer": extractive_fallback(question, selected), "status": "extractive_fallback",
            "reason": "generation_unavailable_or_invalid", "retrieval": decision}


def local_llama_generator(endpoint: str = "http://127.0.0.1:8080", max_tokens: int = 96,
                          timeout: int = 300) -> Callable[[str], str]:
    """Create an offline-only adapter for a locally running llama.cpp server."""
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("LLM endpoint must be a local HTTP loopback address")
    url = endpoint.rstrip("/") + "/completion"

    def generate(prompt: str) -> str:
        payload = json.dumps({
            "prompt": prompt, "n_predict": max_tokens, "temperature": 0.0,
            "top_k": 1, "top_p": 1.0, "seed": 42, "cache_prompt": True,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body.get("content") or "")
    return generate
