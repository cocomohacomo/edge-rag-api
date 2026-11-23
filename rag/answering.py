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



END_MARKER = "<END_ANSWER>"
CONTINUE_MARKER = "<CONTINUE>"


def _strip_control_markers(text: str) -> tuple[str, str | None]:
    state = "continue" if CONTINUE_MARKER in text else ("end" if END_MARKER in text else None)
    cleaned = text.replace(END_MARKER, "").replace(CONTINUE_MARKER, "").strip()
    return cleaned, state


def _required_slots(question: str) -> dict[str, re.Pattern]:
    q = question
    slots = {}
    if re.search(r"担当者|責任者|氏名|誰|だれ", q):
        slots["owner"] = re.compile(r"担当者|責任者|氏名|[一-龥]{2,4}")
    if re.search(r"期限|締切|期日|いつまで", q):
        slots["deadline"] = re.compile(r"\d{1,4}年|\d{1,2}月|\d{1,2}日|期限|締切|期日")
    if re.search(r"端末|CPU|プロセッサ", q, re.I):
        slots["device"] = re.compile(r"i[3579]-?\d{3,5}|Windows|CPU|プロセッサ", re.I)
    if re.search(r"メモリ|RAM|容量", q, re.I):
        slots["memory"] = re.compile(r"\d+(?:\.\d+)?\s*(?:GB|MB|GiB|MiB|RAM)", re.I)
    if re.search(r"方式|方法|構成|どうやって|何を使", q):
        slots["method"] = re.compile(r"方式|方法|構成|使用|利用|実行|FTS5|2-gram|CPU|ローカル|オフライン")
    if re.search(r"何件|何人|何個|何回|何％|割合|数値|件数|人数", q):
        slots["quantity"] = re.compile(r"\d+(?:[.,]\d+)?")
    if re.search(r"場所|地域|住所|どこ", q):
        slots["location"] = re.compile(r"場所|地域|住所|市|町|村|県")
    return slots


def _slot_values(slot: str, evidence: str) -> list[str]:
    patterns = {
        "owner": r"(?:担当者|責任者|氏名)\s*(?:は|:|：)?\s*([一-龥ぁ-んァ-ヶA-Za-z]{2,8}?)(?:です|さん|氏|が|に|へ|を|まで|[。,.、]|$)",
        "deadline": r"(?:\d{1,4}年)?\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日",
        "device": r"i[3579]-?\d{3,5}|Windows\s*\d+",
        "memory": r"\d+(?:\.\d+)?\s*(?:GB|MB|GiB|MiB|RAM)",
        "quantity": r"\d+(?:[.,]\d+)?",
        "method": r"SQLite FTS5|FTS5|2-gram|BM25|n-gram|ベクトル検索|ローカル実行|CPUのみ",
    }
    pattern = patterns.get(slot)
    return list(dict.fromkeys(re.findall(pattern, evidence, re.I))) if pattern else []


def _missing_slots(question: str, answer: str, hits: Iterable[dict]) -> list[str]:
    evidence = " ".join(str(h.get("text", "")) for h in hits)
    missing = []
    for slot, pattern in _required_slots(question).items():
        values = _slot_values(slot, evidence)
        if values:
            if not any(value.lower() in answer.lower() for value in values):
                missing.append(slot)
        elif not pattern.search(answer):
            missing.append(slot)
    return missing


def _is_truncated(metadata: dict) -> bool:
    if not metadata:
        return False
    if metadata.get("stopped_limit") is True or metadata.get("truncated") is True:
        return True
    if metadata.get("stop_type") == "limit":
        return True
    if metadata.get("stop_type") in {"eos", "word"}:
        return False
    budget = int(metadata.get("n_predict") or metadata.get("budget") or 0)
    predicted = int(metadata.get("tokens_predicted") or metadata.get("generated_tokens") or 0)
    if budget and predicted >= budget and not (
        metadata.get("stopped_eos") is True or metadata.get("stopped_word") is True
    ):
        return True
    return False


def _append_without_overlap(previous: str, addition: str) -> str:
    addition = addition.strip()
    if not previous:
        return addition
    if not addition or addition in previous:
        return previous
    max_overlap = min(len(previous), len(addition), 160)
    for size in range(max_overlap, 7, -1):
        if previous[-size:] == addition[:size]:
            return previous + addition[size:]
    # Avoid a repeated first sentence when the model ignores the continuation instruction.
    first = re.split(r"(?<=[。！？!?])", addition, maxsplit=1)[0].strip()
    if first and first in previous:
        addition = addition[len(first):].lstrip(" 、，。\n")
    return previous + ("" if not addition else ("\n" if previous.endswith(("。", "！", "？", "\n")) else "") + addition)


def _continuation_prompt(base_prompt: str, accumulated: str, missing: list[str]) -> str:
    return (
        base_prompt + accumulated +
        "\n\n[CONTINUATION]\n前回の回答に続けてください。既出内容を繰り返さず、"
        "不足情報だけを根拠から補ってください。"
        f"未充足項目: {', '.join(missing) if missing else '文章の完結'}。"
        f"意味的に完結したら {END_MARKER}、続きが必要なら {CONTINUE_MARKER} を末尾に付けてください。\n"
    )


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
        # A strong lexical match may rescue a borderline BM25 score. This is
        # intentionally conservative: weak overlap never passes on score alone.
        if (score >= min_score and overlap >= min_overlap) or (
            score >= min_score * 0.65 and overlap >= max(0.25, min_overlap * 2.5)
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
    evidence = "\\n".join(blocks) if blocks else "（根拠なし）"
    requirements = list(_required_slots(question))
    requirement_text = "、".join(requirements) if requirements else "質問の主語・条件・結論"
    return (
        "日本語で回答。根拠だけを使い、最初に質問を要求属性へ分解し、根拠文から各属性の値を対応付けて回答する。"
        "要求属性は漏れなく満たすこと。複数項目は短い箇条書きまたは「、」で列挙する。"
        "根拠に明記された数値・固有名詞・URL・パスは文字列を改変せず正確に転記する。"
        "根拠にない推測、一般論、補足説明は禁止。質問文や見出しを繰り返さない。"
        "回答が完結したら末尾に <END_ANSWER>、続きが必要なら <CONTINUE> を付ける。"
        "根拠がなければ「文書に記載がありません」と答える。\\n"
        f"要求属性:{requirement_text}\\nQ:{question}\\n根拠:{evidence}\\nA:"
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
        "数値": r"\d+(?:月|日|年|GB|MB|%)?", "方式": r"方式|方法|構成|手段",
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
                if (name == "数値" and re.search(r"\d", sentence)) or
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
    continuation: bool = True,
    max_continuations: int = 2,
) -> dict:
    """Evidence gate -> bounded semantic continuation -> validation -> extractive fallback."""
    decision = retrieval_gate(question, hits)
    if not decision.accepted:
        return {"answer": ABSTAIN, "status": "abstained", "reason": decision.reason,
                "retrieval": decision, "generation_rounds": 0}
    selected = list(decision.accepted)
    if generator is None:
        return {"answer": extractive_fallback(question, selected), "status": "extractive_fallback",
                "reason": "generation_unavailable", "retrieval": decision, "generation_rounds": 0}

    base_prompt = build_evidence_prompt(question, selected)
    accumulated = ""
    rounds = []
    last_reason = "empty"
    max_rounds = 1 + (max(0, min(2, max_continuations)) if continuation else 0)
    for round_index in range(max_rounds):
        prompt = base_prompt if round_index == 0 else _continuation_prompt(
            base_prompt, accumulated, _missing_slots(question, accumulated, selected)
        )
        try:
            raw = generator(prompt)
            candidate, metadata = _generation_payload(raw)
        except Exception:
            candidate, metadata = "", {}
        candidate, marker = _strip_control_markers(candidate)
        rounds.append({
            "round": round_index + 1,
            "tokens": int(metadata.get("tokens_predicted") or metadata.get("generated_tokens") or 0),
            "budget": int(metadata.get("n_predict") or metadata.get("budget") or 0),
            "truncated": _is_truncated(metadata),
            "marker": marker,
        })
        previous = accumulated
        if candidate:
            accumulated = _append_without_overlap(accumulated, candidate)
        valid, last_reason = validate_answer(accumulated, question, selected)
        missing = _missing_slots(question, accumulated, selected)
        text_complete = bool(re.search(r"[。！？!?）」』】]$", accumulated.strip()))
        # Completion is determined by evidence and requested-slot coverage, not
        # by the model's marker or token-limit flag. A complete answer must not
        # trigger another expensive inference merely because n_predict was hit.
        semantic_complete = valid and not missing and text_complete
        if semantic_complete:
            return {"answer": accumulated.strip(), "status": "generated",
                    "reason": "semantic_completion", "retrieval": decision,
                    "generation_rounds": rounds, "missing_slots": []}
        if not continuation or round_index + 1 >= max_rounds:
            break
        if not accumulated:
            last_reason = "empty"
            break
        # Continue only when a requested slot is absent from the answer but
        # its concrete value is present in retrieved evidence. A merely unfinished
        # sentence or a model CONTINUE marker is not enough: prior E2E runs showed
        # those cases often consumed two more rounds and still fell back.
        evidence_text = " ".join(str(hit.get("text", "")) for hit in selected)
        continuable_missing = [
            slot for slot in missing if _slot_values(slot, evidence_text)
        ]
        if not valid or not continuable_missing:
            break
        # Repeated output cannot add information; stop rather than burn a round.
        if accumulated == previous:
            break

    if retry and not accumulated:
        try:
            raw = generator(base_prompt + "\\n根拠中の回答可能な事実だけを簡潔に回答してください。")
            candidate, _ = _generation_payload(raw)
            candidate, _ = _strip_control_markers(candidate)
            accumulated = candidate
        except Exception:
            pass
    valid, last_reason = validate_answer(accumulated, question, selected)
    missing = _missing_slots(question, accumulated, selected)
    if valid and not missing and re.search(r"[。！？!?）」』】]$", accumulated.strip()):
        return {"answer": accumulated.strip(), "status": "generated",
                "reason": "semantic_completion", "retrieval": decision,
                "generation_rounds": rounds, "missing_slots": []}
    fallback = extractive_fallback(question, selected, max_sentences=4)
    return {"answer": fallback, "status": "extractive_fallback",
            "reason": last_reason if last_reason != "accepted" else "incomplete_answer",
            "retrieval": decision, "generation_rounds": rounds, "missing_slots": missing}


def local_llama_generator(endpoint: str = "http://127.0.0.1:8080", max_tokens: int = 48,
                          timeout: int = 300) -> Callable[[str], dict]:
    """Create an offline-only adapter for a locally running llama.cpp server."""
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("LLM endpoint must be a local HTTP loopback address")
    url = endpoint.rstrip("/") + "/completion"

    def generate(prompt: str) -> dict:
        question = prompt.split("\nQ:", 1)[-1].split("\n根拠:", 1)[0]
        budget = min(24, max_tokens) if "[CONTINUATION]" in prompt else generation_budget(question, max_tokens)
        payload = json.dumps({
            "prompt": prompt, "n_predict": budget, "temperature": 0.0,
            "top_k": 1, "top_p": 1.0, "seed": 42, "cache_prompt": True,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return {"content": str(body.get("content") or ""), "stop": body.get("stop"),
                "stopped_eos": body.get("stopped_eos"), "stopped_word": body.get("stopped_word"),
                "stopped_limit": body.get("stopped_limit"), "stop": body.get("stop"),
                "tokens_predicted": body.get("tokens_predicted"), "n_predict": budget}
    return generate
