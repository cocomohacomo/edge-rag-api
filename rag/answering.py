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


def _query_terms(question: str) -> set[str]:
    terms = set(japanese_terms(question))
    # Generic interrogative/function bigrams add noise to short Japanese queries.
    stop = {"何を", "何で", "何が", "です", "ます", "こと", "方法", "教え", "くだ", "さい"}
    return terms - stop or terms


def _overlap(question: str, text: str) -> float:
    q = _query_terms(question)
    if not q:
        return 0.0
    d = set(japanese_terms(text))
    return len(q & d) / len(q)


def generation_budget(question: str, base_tokens: int = 48) -> int:
    """Keep the existing 48-token ceiling; allocate fewer tokens to simple fact lookups."""
    q = question.strip()
    multi_attribute = bool(re.search(
        r"(担当者.*期限|期限.*担当者|方式.*要件|要件.*方式|端末.*メモリ|"
        r"メモリ.*端末|実行方式.*CPU|CPU.*実行方式|\sと\s|及び|ならびに|"
        r"それぞれ|複数|一覧|比較)", q
    ))
    return min(base_tokens, 48) if multi_attribute or len(q) >= 28 else min(base_tokens, 32)


def _generation_payload(value: object) -> tuple[str, dict]:
    if isinstance(value, dict):
        return str(value.get("content") or ""), value
    return str(value or ""), {}


def retrieval_gate(
    question: str,
    hits: Iterable[dict],
    *,
    min_score: float = 0.08,
    min_overlap: float = 0.06,
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
        # A strong lexical match may rescue a borderline BM25 score. This is
        # intentionally conservative: weak overlap never passes on score alone.
        if (score >= min_score and overlap >= min_overlap) or (
            score >= min_score * 0.65 and overlap >= max(0.18, min_overlap * 2.5)
        ):
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
        "日本語で回答。根拠だけを使い、質問が求める属性を漏れなく簡潔に回答する。"
        "複数項目は「、」で列挙し、質問文や見出しを繰り返さない。推測・一般論は禁止。"
        "根拠がなければ「文書に記載がありません」と答える。\n"
        f"Q:{question}\n根拠:{evidence}\nA:"
    )


def validate_answer(answer: str, question: str, hits: Iterable[dict], generation: dict | None = None) -> tuple[bool, str]:
    """Conservative deterministic guard against empty, off-topic and unsupported numeric output."""
    cleaned = re.sub(r"^回答[:：]?\s*", "", (answer or "").strip())
    cleaned = re.sub(r"\n?\[\d+\]\s*$", "", cleaned).strip()
    if not cleaned:
        return False, "empty"
    if generation:
        # llama.cpp reports stop=true for EOS/stop-word termination. A false
        # value with all allocated tokens consumed is a hard truncation signal.
        budget = int(generation.get("n_predict") or generation.get("budget") or 0)
        predicted = int(generation.get("tokens_predicted") or generation.get("generated_tokens") or 0)
        stopped = generation.get("stop")
        if stopped is False or (budget and predicted >= budget and stopped is not True):
            return False, "token_limit"
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
    q_terms = _query_terms(question)
    if q_terms and len(answer_terms & q_terms) == 0 and len(answer_terms & context_terms) < 3:
        return False, "question_mismatch"
    return True, "accepted"


def extractive_fallback(question: str, hits: Iterable[dict], max_sentences: int = 3) -> str:
    """Return the best source sentences, preserving provenance and avoiding free-form claims."""
    candidates = []
    q_terms = _query_terms(question)
    attribute_patterns = {
        "担当者": r"担当者|氏名|責任者", "期限": r"期限|締切|期日|まで",
        "数値": r"\\d+(?:月|日|年|GB|MB|%)?", "方式": r"方式|方法|構成|手段",
        "要件": r"要件|条件|必要|対象", "場所": r"場所|地域|住所",
    }
    requested = [name for name, pattern in attribute_patterns.items()
                 if re.search(pattern, question)]
    for hit in hits:
        text = str(hit.get("text", "")).strip()
        for sentence in re.split(r"(?<=[。！？!?])\s*|\n+", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = set(japanese_terms(sentence))
            coverage = len(q_terms & terms) / max(1, len(q_terms))
            attribute_hits = sum(
                1 for name in requested
                if (name == "数値" and re.search(r"\\d", sentence)) or
                   (name != "数値" and re.search(attribute_patterns[name], sentence))
            )
            score = coverage + 0.20 * attribute_hits
            if coverage > 0 or attribute_hits:
                candidates.append((score, len(sentence), sentence, hit.get("source", "unknown")))
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
            generated = generator(prompt)
            candidate, metadata = _generation_payload(generated)
        except Exception:
            candidate, metadata = "", {}
        valid, reason = validate_answer(candidate, question, selected, metadata)
        if valid and reason != "abstention":
            return {"answer": candidate.strip(), "status": "generated", "reason": "accepted",
                    "retrieval": decision}
        if retry:
            retry_prompt = prompt + "\n直前の回答は検証に失敗しました。根拠中の回答可能な事実だけを1文で回答してください。"
            try:
                generated = generator(retry_prompt)
                candidate, metadata = _generation_payload(generated)
            except Exception:
                candidate, metadata = "", {}
            valid, reason = validate_answer(candidate, question, selected, metadata)
            if valid and reason != "abstention":
                return {"answer": candidate.strip(), "status": "regenerated", "reason": "accepted",
                        "retrieval": decision}
    return {"answer": extractive_fallback(question, selected), "status": "extractive_fallback",
            "reason": "generation_unavailable_or_invalid", "retrieval": decision}


def local_llama_generator(endpoint: str = "http://127.0.0.1:8080", max_tokens: int = 48,
                          timeout: int = 300) -> Callable[[str], dict]:
    """Create an offline-only adapter for a locally running llama.cpp server."""
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("LLM endpoint must be a local HTTP loopback address")
    url = endpoint.rstrip("/") + "/completion"

    def generate(prompt: str) -> str:
        question = prompt.split("\\nQ:", 1)[-1].split("\\n根拠:", 1)[0]
        budget = generation_budget(question, max_tokens)
        payload = json.dumps({
            "prompt": prompt, "n_predict": budget, "temperature": 0.0,
            "top_k": 1, "top_p": 1.0, "seed": 42, "cache_prompt": True,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return {"content": str(body.get("content") or ""), "stop": body.get("stop"),
                "tokens_predicted": body.get("tokens_predicted"), "n_predict": budget}
    return generate
