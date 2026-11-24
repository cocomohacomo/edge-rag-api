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
    legacy_continue = "<CONTINUATION>"
    state = ("continue" if CONTINUE_MARKER in text or legacy_continue in text
             else ("end" if END_MARKER in text else None))
    cleaned = (text.replace(END_MARKER, "").replace(CONTINUE_MARKER, "")
               .replace(legacy_continue, "").strip())
    return cleaned, state


def _required_slots(question: str) -> dict[str, re.Pattern]:
    q = question
    slots = {}
    if re.search(r"担当者|責任者|氏名|誰|だれ", q):
        slots["owner"] = re.compile(r"担当者|責任者|氏名|[一-龥]{2,4}")
    if re.search(r"期限|締切|期日|いつまで", q):
        slots["deadline"] = re.compile(r"\d{1,4}年|\d{1,2}月|\d{1,2}日|期限|締切|期日")
    if re.search(r"端末|CPU|プロセッサ", q, re.I):
        slots["device"] = re.compile(r"i[3579]-?\d{3,5}[A-Za-z]?|Windows|CPU|CPU|プロセッサ", re.I)
    if re.search(r"メモリ|RAM|容量", q, re.I):
        slots["memory"] = re.compile(r"\d+(?:\.\d+)?\s*(?:GB|MB|GiB|MiB|RAM)", re.I)
    if re.search(r"方式|方法|構成|どうやって|何を使", q):
        slots["method"] = re.compile(r"方式|方法|構成|使用|利用|実行|FTS5|2-gram|CPU|ローカル|オフライン")
    if re.search(r"何件|何人|何個|何回|何％|割合|数値|件数|人数", q):
        slots["quantity"] = re.compile(r"\d+(?:[.,]\d+)?")
    if re.search(r"場所|地域|住所|どこ", q):
        slots["location"] = re.compile(r"場所|地域|住所|市|町|村|県")
    return slots


def _requirement_names(question: str) -> list[str]:
    names = list(_required_slots(question))
    if re.search(r"URL|リンク|ウェブサイト|アドレス", question, re.I):
        names.append("url")
    if re.search(r"パス|ファイル名|保存先|ディレクトリ", question):
        names.append("path")
    return list(dict.fromkeys(names))


def _relation_specs(question: str) -> list[dict]:
    """Deterministic relation requirements for questions whose answer needs linked facts."""
    specs = []
    if re.search(r"共存|同時に|メモリ制約|メモリ.*モデル|モデル.*メモリ", question):
        specs.append({"id": "model_search_db_coexistence", "label": "モデルと検索DBの共存",
                      "pattern": r"モデル.{0,30}(?:検索DB|検索データベース).{0,20}共存|(?:検索DB|検索データベース).{0,30}モデル.{0,20}共存"})
    if re.search(r"オフライン|ネットワーク要件|接続", question):
        specs.append({"id": "offline_runtime_requirement", "label": "初回導入後の実行時ネットワーク要件",
                      "pattern": r"初回導入後.{0,50}オフライン|実行時.{0,30}ネットワーク接続.{0,20}必要としない"})
    if re.search(r"表土|農地.*現状変更|現状変更.*農地|営農可能性", question):
        specs.append({"id": "topsoil_and_cultivation", "label": "表土・土壌と施工後の営農可能性",
                      "pattern": r"表土.{0,100}(?:土壌|営農可能性)|(?:土壌|営農可能性).{0,100}表土"})
    return specs


def _relation_evidence(spec: dict, hits: Iterable[dict]) -> list[dict]:
    found = []
    pattern = re.compile(spec["pattern"])
    for hit in hits:
        text = str(hit.get("text", ""))
        matched = []
        for sentence in re.split(r"(?<=[。！？!?])\s*|\n+", text):
            sentence = sentence.strip()
            if sentence and pattern.search(sentence):
                matched.append(sentence)
        # Some relations span adjacent sentences; retain the source passage as
        # one provenance unit rather than incorrectly dropping the relation.
        if not matched and pattern.search(text):
            matched.append(text.strip())
        for sentence in matched:
            found.append({"source": str(hit.get("source", "unknown")),
                          "text": sentence, "evidence": sentence})
    return found


def _missing_relations(question: str, answer: str, hits: Iterable[dict]) -> list[str]:
    missing = []
    answer_terms = set(japanese_terms(answer))
    for spec in _relation_specs(question):
        entries = _relation_evidence(spec, hits)
        if not entries:
            missing.append("relation:" + spec["id"])
            continue
        # Require the answer to carry the relation-bearing evidence, not merely one
        # isolated attribute such as "8GB".
        supported = False
        anchors = {
            "model_search_db_coexistence": {"モデル", "検索DB", "共存"},
            "offline_runtime_requirement": {"オフライン", "ネットワーク", "実行時"},
            "topsoil_and_cultivation": {"表土", "土壌", "営農可能性"},
        }.get(spec["id"], set())
        for entry in entries:
            relation_terms = set(japanese_terms(entry["text"]))
            overlap = len(answer_terms & relation_terms) / max(1, len(relation_terms))
            anchor_hits = sum(1 for term in anchors if term in answer)
            if overlap >= 0.30 or (anchors and anchor_hits >= 2):
                supported = True
                break
        if not supported:
            missing.append("relation:" + spec["id"])
    return missing


def _numeric_claim_diagnostics(answer: str, hits: Iterable[dict]) -> dict:
    """Expose exact unsupported numeric tokens without weakening strict validation."""
    context = " ".join(str(hit.get("text", "")) for hit in hits)
    pattern = r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?:%|GB|MB|GiB|MiB|日|月|年|時|分)?"
    answer_numbers = list(dict.fromkeys(re.findall(pattern, answer or "", re.I)))
    evidence_numbers = list(dict.fromkeys(re.findall(pattern, context, re.I)))
    return {
        "answer_numeric_claims": answer_numbers,
        "evidence_numeric_claims": evidence_numbers,
        "unsupported_numeric_claims": [value for value in answer_numbers if value not in evidence_numbers],
    }


def build_answer_plan(question: str, hits: Iterable[dict]) -> dict:
    """Build a deterministic, evidence-linked answer plan without another model call."""
    sources = list(hits)
    requirements = _requirement_names(question)
    q_terms = _query_terms(question)
    mapping = {}
    for slot in requirements:
        candidates = []
        for hit in sources:
            text = str(hit.get("text", ""))
            if slot == "url":
                matched = re.findall(r"""https?://[^\s<>"'、。]+""", text)
            elif slot == "path":
                matched = re.findall(r"(?<![A-Za-z0-9./:])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", text)
            else:
                matched = _slot_values(slot, text)
            if matched:
                candidates.append({
                    "source": str(hit.get("source", "unknown")),
                    "values": list(dict.fromkeys(matched))[:4],
                    "evidence": text[:360],
                })
        mapping[slot] = candidates
    missing = [slot for slot in requirements if not mapping.get(slot)]
    relation_requirements = _relation_specs(question)
    relation_mapping = {spec["id"]: _relation_evidence(spec, sources) for spec in relation_requirements}
    missing_relations = [key for key, entries in relation_mapping.items() if not entries]
    # Keep evidence ordering deterministic: requested-attribute coverage first,
    # then lexical question overlap, then the retrieval score.
    ranked = []
    for index, hit in enumerate(sources):
        text = str(hit.get("text", ""))
        covered = sum(1 for slot in requirements if any(
            item["source"] == str(hit.get("source", "unknown"))
            for item in mapping.get(slot, [])
        ))
        lexical = len(q_terms & set(japanese_terms(text))) / max(1, len(q_terms))
        ranked.append((covered, lexical, float(hit.get("score", 0.0)), -index, hit))
    ranked.sort(key=lambda item: item[:4], reverse=True)
    return {
        "question": question,
        "requirements": requirements,
        "evidence_mapping": mapping,
        "missing_requirements": missing,
        "relation_requirements": relation_requirements,
        "relation_mapping": relation_mapping,
        "missing_relations": missing_relations,
        "answer_order": requirements,
        "selected_sources": [str(item[4].get("source", "unknown")) for item in ranked],
        "validation": {
            "evidence_supported": bool(sources),
            "requirements_complete": not missing and not missing_relations,
            "relations_complete": not missing_relations,
            "literal_integrity": "pending",
        },
    }


def assemble_evidence_answer(plan: dict) -> str | None:
    """Assemble a concise answer only from values explicitly extracted from evidence."""
    labels = {
        "owner": "担当者", "deadline": "期限", "device": "端末",
        "memory": "メモリ", "method": "方式", "quantity": "数量",
        "location": "場所", "url": "URL", "path": "パス",
    }
    requirements = list(plan.get("requirements", []))
    if (not requirements and not plan.get("relation_requirements")) or plan.get("missing_requirements") or plan.get("missing_relations"):
        return None
    lines = []
    for slot in requirements:
        entries = plan.get("evidence_mapping", {}).get(slot, [])
        values = []
        sources = []
        for entry in entries:
            for value in entry.get("values", []):
                if value not in values:
                    values.append(value)
            source = str(entry.get("source", "")).strip()
            if source and source not in sources:
                sources.append(source)
        if not values:
            return None
        rendered = "、".join(values[:3])
        provenance = "、".join(sources[:2])
        suffix = f"（出典: {provenance}）" if provenance else ""
        lines.append(f"{labels.get(slot, slot)}: {rendered}{suffix}")
    for spec in plan.get("relation_requirements", []):
        entries = plan.get("relation_mapping", {}).get(spec["id"], [])
        if not entries:
            return None
        unique = list(dict.fromkeys(item["text"] for item in entries))
        lines.append(f"{spec['label']}: " + " / ".join(unique[:2]))
    return "\n".join(lines) + "。"


def build_progress_summary(plan: dict, *, validated: bool = False) -> dict:
    """UI-neutral status summary; expose only completed checks, never hidden reasoning."""
    return {
        "title": "回答を整理しました",
        "checks": [
            {"id": "requirements", "label": "質問の要求項目を抽出", "done": True},
            {"id": "evidence", "label": "根拠資料から該当情報を選択", "done": bool(plan.get("evidence_mapping"))},
            {"id": "coverage", "label": "回答項目の不足を確認", "done": True},
            {"id": "integrity", "label": "数値・URL・パスの整合性を検証", "done": validated},
        ],
        "missing_requirements": list(plan.get("missing_requirements", [])),
        "missing_relations": list(plan.get("missing_relations", [])),
    }


def _slot_values(slot: str, evidence: str) -> list[str]:
    patterns = {
        "owner": r"(?:担当者|責任者|氏名)\s*(?:は|:|：)?\s*([一-龥ぁ-んァ-ヶA-Za-z]{2,8}?)(?:です|さん|氏|が|に|へ|を|まで|[。,.、]|$)",
        "deadline": r"(?:\d{1,4}年)?\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日",
        "device": r"i[3579]-?\d{3,5}[A-Za-z]?|Windows\s*\d+|CPU(?:のみ)?",
        "memory": r"\d+(?:\.\d+)?\s*(?:GB|MB|GiB|MiB|RAM)",
        "quantity": r"\d+(?:[.,]\d+)?",
        "method": r"SQLite FTS5|FTS5|2-gram|BM25|n-gram|ベクトル検索|ローカル実行|CPUのみ",
        "url": r"""https?://[^\s<>"'、。]+""",
        "path": r"(?<![A-Za-z0-9./:])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+",
    }
    pattern = patterns.get(slot)
    return list(dict.fromkeys(re.findall(pattern, evidence, re.I))) if pattern else []


def _missing_slots(question: str, answer: str, hits: Iterable[dict]) -> list[str]:
    evidence = " ".join(str(h.get("text", "")) for h in hits)
    missing = []
    for slot in _requirement_names(question):
        pattern = _required_slots(question).get(slot, re.compile(r"https?://" if slot == "url" else r"/"))

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
    if "stop" in metadata and metadata.get("stop") is False:
        return True
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


def _clean_generation_delta(candidate: str, question: str) -> str:
    """Remove echoed prompt/control lines before answer text is accumulated."""
    text, _ = _strip_control_markers(candidate or "")
    cleaned = []
    for line in text.splitlines():
        value = line.strip()
        if not value or value == "[CONTINUATION]" or re.match(r"^\\[根拠\\d+\\]", value):
            continue
        if re.match(r"^(?:Q|質問|根拠資料?|回答計画|要求属性)[:：]", value, re.I):
            continue
        value = re.sub(r"^(?:A|回答)[:：]\s*", "", value, flags=re.I)
        if question and question in value:
            value = value.replace(question, "").strip()
        if value:
            cleaned.append(value)
    result = "\n".join(cleaned).strip()
    # A model may echo the exact question without a Q: prefix.
    if question:
        result = result.replace(question, "").strip()
    return result


def _continuation_prompt(question: str, hits: Iterable[dict], accumulated: str,
                         missing: list[str]) -> str:
    """Build a fresh delta-only prompt; never concatenate onto the initial prompt."""
    missing_text = "、".join(missing) if missing else "回答の自然な完結"
    evidence = []
    for index, hit in enumerate(hits, 1):
        source = str(hit.get("source", "unknown"))
        passage = re.sub(r"\s+", " ", str(hit.get("text", ""))).strip()
        if passage:
            evidence.append(f"[根拠{index}] {source}: {passage[:700]}")
    return (
        "あなたは既存回答を補完する役割です。次の情報だけを使ってください。\n"
        f"質問: {question}\n"
        "根拠資料:\n" + "\n".join(evidence) + "\n"
        f"既存回答（変更せず保持）:\n<EXISTING_ANSWER>\n{accumulated}\n</EXISTING_ANSWER>\n"
        f"不足項目: {missing_text}\n"
        "出力規則: 不足項目に対応する追加文だけを出力する。既存回答を再掲しない。"
        "質問、根拠、見出し、Q:、A:を出力しない。"
        "不足項目がない場合は既存回答を繰り返さず、完結を示す終了マーカーだけを出力する。"
        f"追加文の末尾には {END_MARKER}、追加がさらに必要な場合のみ {CONTINUE_MARKER} を付ける。\n"
        "追加文:"
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
    hits = list(hits)
    plan = build_answer_plan(question, hits)
    source_order = {source: index for index, source in enumerate(plan["selected_sources"])}
    hits = sorted(hits, key=lambda hit: source_order.get(str(hit.get("source", "unknown")), len(source_order)))
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
    requirements = list(plan["requirements"])
    requirement_text = "、".join(requirements) if requirements else "質問の主語・条件・結論"
    plan_lines = []
    for slot in requirements:
        values = []
        for item in plan["evidence_mapping"].get(slot, []):
            values.extend(item["values"])
        plan_lines.append(f"{slot}: {', '.join(dict.fromkeys(values)) if values else '根拠なし'}")
    for spec in plan.get("relation_requirements", []):
        entries = plan.get("relation_mapping", {}).get(spec["id"], [])
        evidence_lines = " / ".join(item["text"] for item in entries[:2]) or "根拠なし"
        plan_lines.append(f"関係性[{spec['id']}]: {evidence_lines}")
    plan_text = "\n".join(plan_lines)
    return (
        "日本語で回答。根拠だけを使い、最初に質問を要求属性へ分解し、根拠文から各属性の値を対応付けて回答する。"
        "要求属性は漏れなく満たすこと。複数項目は短い箇条書きまたは「、」で列挙する。"
        "根拠に明記された数値・固有名詞・URL・パスは文字列を改変せず正確に転記する。"
        "根拠にない推測、一般論、補足説明は禁止。質問文や見出しを繰り返さない。"
        "回答が完結したら末尾に <END_ANSWER>、続きが必要なら <CONTINUE> を付ける。"
        "根拠がなければ「文書に記載がありません」と答える。\n"
        f"要求属性:{requirement_text}\n回答計画:{plan_text}\nQ:{question}\n根拠:{evidence}\nA:"
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
    # URLs and relative file paths are exact identifiers and must exist in evidence.
    answer_urls = re.findall(r"""https?://[^\s<>"'、。]+""", cleaned)
    context_urls = set(re.findall(r"""https?://[^\s<>"'、。]+""", context))
    if any(value not in context_urls for value in answer_urls):
        return False, "unsupported_url"
    answer_paths = re.findall(r"(?<![A-Za-z0-9./:])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", cleaned)
    context_paths = set(re.findall(r"(?<![A-Za-z0-9./:])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", context))
    if any(value not in context_paths for value in answer_paths):
        return False, "unsupported_path"
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
    prefer_evidence_assembly: bool = True,
    force_generation: bool = False,
) -> dict:
    """Evidence gate -> optional deterministic assembly -> bounded LLM loop -> validated fallback.

    force_generation bypasses the pre-generation evidence assembler. When enabled,
    a missing generator is reported explicitly instead of silently returning an
    assembled answer, which makes generation-required callers and benchmarks auditable.
    """
    decision = retrieval_gate(question, hits)
    plan = build_answer_plan(question, decision.accepted)
    if not decision.accepted:
        return {"answer": ABSTAIN, "status": "abstained", "reason": decision.reason,
                "retrieval": decision, "generation_rounds": 0,
                "answer_plan": plan, "progress": build_progress_summary(plan)}
    selected = list(decision.accepted)
    if prefer_evidence_assembly and not force_generation:
        assembled = assemble_evidence_answer(plan)
        assembled_valid = (bool(assembled) and validate_answer(assembled, question, selected)[0]
                           and not _missing_relations(question, assembled, selected))
        if assembled_valid:
            return {"answer": assembled, "status": "evidence_assembled",
                    "reason": "pre_generation_all_requirements_have_direct_evidence",
                    "retrieval": decision, "generation_rounds": 0, "missing_slots": [],
                    "answer_plan": plan, "progress": build_progress_summary(plan, validated=True)}
    if generator is None:
        if force_generation:
            return {"answer": ABSTAIN, "status": "generation_unavailable",
                    "reason": "forced_generation_requires_generator", "retrieval": decision,
                    "generation_rounds": 0, "answer_plan": plan,
                    "progress": build_progress_summary(plan)}
        assembled = assemble_evidence_answer(plan)
        assembled_valid = (bool(assembled) and validate_answer(assembled, question, selected)[0]
                       and not _missing_relations(question, assembled, selected))
        if assembled_valid:
            return {"answer": assembled, "status": "evidence_assembled",
                    "reason": "all_requirements_have_direct_evidence", "retrieval": decision,
                    "generation_rounds": 0, "missing_slots": [],
                    "answer_plan": plan, "progress": build_progress_summary(plan, validated=True)}
        return {"answer": extractive_fallback(question, selected), "status": "extractive_fallback",
                "reason": "generation_unavailable", "retrieval": decision, "generation_rounds": 0,
                "answer_plan": plan, "progress": build_progress_summary(plan)}

    base_prompt = build_evidence_prompt(question, selected)
    accumulated = ""
    rounds = []
    last_reason = "empty"
    max_rounds = 1 + (max(0, min(2, max_continuations)) if continuation else 0)

    # Explicit bounded generation state machine. A round may continue only when
    # the partial answer is grounded and retrieved evidence can fill a missing slot.
    state = "initial"
    for round_index in range(max_rounds):
        if round_index == 0:
            prompt = base_prompt
            state = "generating_initial"
        else:
            missing_before = _missing_slots(question, accumulated, selected) + _missing_relations(question, accumulated, selected)
            prompt = _continuation_prompt(question, selected, accumulated, missing_before)
            state = "generating_continuation"
        try:
            raw = generator(prompt)
            candidate, metadata = _generation_payload(raw)
        except Exception as exc:
            rounds.append({
                "round": round_index + 1, "tokens": 0, "budget": 0,
                "truncated": False, "marker": None, "state": "generation_error",
                "error": type(exc).__name__,
            })
            last_reason = "generation_error"
            state = "failed"
            break

        candidate, marker = _strip_control_markers(candidate)
        candidate = _clean_generation_delta(candidate, question)
        truncated = _is_truncated(metadata)
        rounds.append({
            "round": round_index + 1,
            "tokens": int(metadata.get("tokens_predicted") or metadata.get("generated_tokens") or 0),
            "budget": int(metadata.get("n_predict") or metadata.get("budget") or 0),
            "truncated": truncated,
            "marker": marker,
            "state": "received",
            "stop": metadata.get("stop"),
            "stop_type": metadata.get("stop_type"),
            "completion_marker": marker,
        })

        previous = accumulated
        if candidate:
            accumulated = _append_without_overlap(accumulated, candidate)
        valid, last_reason = validate_answer(accumulated, question, selected)
        missing = _missing_slots(question, accumulated, selected) + _missing_relations(question, accumulated, selected)
        text_complete = marker == "end" or bool(re.search(r"[。！？!?）」』】]$", accumulated.strip()))
        rounds[-1]["validation_reason"] = last_reason
        rounds[-1]["numeric_diagnostics"] = _numeric_claim_diagnostics(accumulated, selected)
        rounds[-1]["completion_signal"] = marker or ("punctuation" if text_complete else None)
        if valid and not missing and text_complete:
            rounds[-1]["state"] = "completed"
            state = "completed"
            return {"answer": accumulated.strip(), "status": "generated",
                    "reason": "semantic_completion", "retrieval": decision,
                    "generation_rounds": rounds, "missing_slots": [],
                    "answer_plan": plan, "progress": build_progress_summary(plan, validated=True)}

        if not continuation or round_index + 1 >= max_rounds:
            rounds[-1]["state"] = "round_limit" if round_index + 1 >= max_rounds else "single_pass"
            state = "exhausted"
            break
        if not accumulated or accumulated == previous:
            rounds[-1]["state"] = "no_progress"
            state = "stalled"
            break

        # Do not ask the model to invent absent facts. Every missing requirement
        # must have at least one concrete value in retrieved evidence.
        evidence_text = " ".join(str(hit.get("text", "")) for hit in selected)
        relation_ids = {"relation:" + spec["id"] for spec in _relation_specs(question)
                        if _relation_evidence(spec, selected)}
        continuable_missing = [slot for slot in missing
                               if (slot in relation_ids) or _slot_values(slot, evidence_text)]
        if missing and len(continuable_missing) != len(missing):
            rounds[-1]["state"] = "missing_evidence"
            state = "blocked"
            break

        # A continuation is useful only when the accumulated text itself is
        # evidence-aligned. A marker or truncation flag alone never authorizes it.
        answer_terms = set(japanese_terms(accumulated))
        evidence_terms = set(japanese_terms(evidence_text))
        overlap = len(answer_terms & evidence_terms) / max(1, len(answer_terms))
        unsupported_numbers = re.findall(
            r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?:%|GB|MB|日|月|年|時|分)?",
            accumulated, re.I
        )
        evidence_numbers = set(re.findall(
            r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?:%|GB|MB|日|月|年|時|分)?",
            evidence_text, re.I
        ))
        if overlap < 0.45 or any(value not in evidence_numbers for value in unsupported_numbers):
            rounds[-1]["state"] = "ungrounded_partial"
            state = "blocked"
            break

        # Missing evidence-backed slots take priority. Otherwise continue only
        # when the model explicitly requested continuation or hit its token cap.
        # A hard token-limit stop may truncate an otherwise grounded sentence
        # before the model emits its control marker. Continue only while unfinished.
        # A grounded but unfinished answer also deserves one bounded completion
        # attempt when the model stopped at EOS without emitting a control marker.
        # This avoids treating punctuation/marker omission as a reason to discard
        # otherwise useful generated content.
        should_continue = bool(continuable_missing) or (
            not text_complete and (marker == "continue" or truncated or valid)
        )
        if not should_continue:
            rounds[-1]["state"] = "model_finished"
            state = "finished_incomplete"
            break
        rounds[-1]["state"] = "continue"
        state = "continuing"

    if retry and not accumulated and state != "failed":
        try:
            raw = generator(base_prompt + "\n根拠中の回答可能な事実だけを簡潔に回答してください。")
            candidate, _ = _generation_payload(raw)
            candidate, _ = _strip_control_markers(candidate)
            accumulated = candidate
        except Exception:
            pass
    valid, validation_reason = validate_answer(accumulated, question, selected)
    if state != "failed":
        last_reason = validation_reason
    missing = _missing_slots(question, accumulated, selected) + _missing_relations(question, accumulated, selected)
    if valid and not missing and re.search(r"[。！？!?）」』】]$", accumulated.strip()):
        return {"answer": accumulated.strip(), "status": "generated",
                "reason": "semantic_completion", "retrieval": decision,
                "generation_rounds": rounds, "missing_slots": [],
                "answer_plan": plan, "progress": build_progress_summary(plan, validated=True)}
    # Generation-required callers must be able to distinguish an incomplete
    # generation from a deterministic answer. Never hide a failed LLM attempt
    # by silently replacing it with evidence assembly.
    if force_generation:
        draft_valid, draft_reason = validate_answer(accumulated, question, selected)
        if state == "failed":
            forced_status = "generation_error"
            forced_reason = last_reason
            visible_answer = accumulated.strip() if draft_valid else ABSTAIN
        else:
            forced_status = "generation_incomplete"
            forced_reason = (
                "requirements_incomplete" if missing else
                ("answer_not_grounded" if not draft_valid else "completion_not_confirmed")
            )
            visible_answer = accumulated.strip() if draft_valid else ABSTAIN
        return {"answer": visible_answer, "status": forced_status,
                "reason": forced_reason, "validation_reason": draft_reason,
                "retrieval": decision, "generation_rounds": rounds,
                "missing_slots": missing, "answer_plan": plan,
                "progress": build_progress_summary(plan, validated=False),
                "numeric_diagnostics": _numeric_claim_diagnostics(accumulated, selected)}

    assembled = assemble_evidence_answer(plan)
    assembled_valid = (bool(assembled) and validate_answer(assembled, question, selected)[0]
                       and not _missing_relations(question, assembled, selected))
    if assembled_valid:
        return {"answer": assembled, "status": "evidence_assembled",
                "reason": "generation_incomplete_but_requirements_have_direct_evidence",
                "retrieval": decision, "generation_rounds": rounds, "missing_slots": [],
                "answer_plan": plan, "progress": build_progress_summary(plan, validated=True)}
    fallback = extractive_fallback(question, selected, max_sentences=4)
    fallback_missing_slots = _missing_slots(question, fallback, selected)
    fallback_missing_relations = _missing_relations(question, fallback, selected)
    fallback_complete = not fallback_missing_slots and not fallback_missing_relations
    fallback_reason = (last_reason if last_reason != "accepted" else "incomplete_answer")
    return {"answer": fallback, "status": "extractive_fallback",
            "reason": fallback_reason,
            "retrieval": decision, "generation_rounds": rounds, "missing_slots": missing,
            "fallback_validation": {"complete": fallback_complete,
                "missing_slots": fallback_missing_slots,
                "missing_relations": fallback_missing_relations},
            "numeric_diagnostics": _numeric_claim_diagnostics(accumulated, selected),
            "answer_plan": plan, "progress": build_progress_summary(plan, validated=valid and not missing)}


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
                "stopped_limit": body.get("stopped_limit"), "stop_type": body.get("stop_type"),
                "truncated": body.get("truncated"),
                "tokens_predicted": body.get("tokens_predicted"), "n_predict": budget}
    return generate