"""Attribute-level evidence binding, targeted retry prompts, and strict final checks.

Evidence IDs are assigned by retrieval order. Every binding points to one exact
retrieved passage and stores offsets into that passage's original text.
"""
from __future__ import annotations

import json
import re
from typing import Iterable


def build_attribute_ledger(plan: dict, hits: Iterable[dict]) -> dict:
    evidence = []
    evidence_by_id = {}
    for index, hit in enumerate(hits, 1):
        evidence_id = f"E{index}"
        source = str(hit.get("source", "unknown"))
        text = str(hit.get("text", ""))
        entry = {"evidence_id": evidence_id, "source": source, "text": text}
        evidence.append(entry)
        evidence_by_id[evidence_id] = entry

    attributes = []
    for name in plan.get("requirements", []):
        bindings = []
        values_seen = []
        for candidate in plan.get("evidence_mapping", {}).get(name, []):
            evidence_id = candidate.get("evidence_id")
            targets = ([evidence_by_id[evidence_id]]
                       if evidence_id in evidence_by_id else [])
            # Compatibility for hand-built legacy plans: only bind by source if
            # that source resolves to exactly one passage.
            if not targets:
                matches = [entry for entry in evidence if
                           entry["source"] == str(candidate.get("source", "unknown"))]
                if len(matches) == 1:
                    targets = matches
            for target in targets:
                text = target["text"]
                for value in candidate.get("values", []):
                    value = str(value)
                    start = text.find(value)
                    if start < 0:
                        continue
                    binding = {
                        "evidence_id": target["evidence_id"],
                        "source": target["source"],
                        "start": start,
                        "end": start + len(value),
                        "value": value,
                        "quote": text[start:start + len(value)],
                    }
                    if binding not in bindings:
                        bindings.append(binding)
                    if value not in values_seen:
                        values_seen.append(value)

        value_evidence = {}
        for binding in bindings:
            value_evidence.setdefault(binding["value"], set()).add(binding["evidence_id"])
        independent_evidence_ids = {
            evidence_id for ids in value_evidence.values() for evidence_id in ids
        }
        conflict = len(value_evidence) > 1 and len(independent_evidence_ids) > 1
        status = "insufficient" if not bindings else ("conflict" if conflict else "supported")
        attributes.append({
            "attribute": name,
            "status": status,
            "values": values_seen,
            "bindings": bindings,
        })
    return {
        "question": plan.get("question", ""),
        "evidence": evidence,
        "attributes": attributes,
    }


def validate_attribute_ledger(ledger: dict) -> dict:
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in ledger.get("evidence", [])
        if item.get("evidence_id")
    }
    results = []
    for item in ledger.get("attributes", []):
        status = item.get("status")
        bindings = item.get("bindings", [])
        valid_bindings = []
        for binding in bindings:
            evidence = evidence_by_id.get(str(binding.get("evidence_id")))
            if evidence is None:
                continue
            try:
                start, end = int(binding.get("start", -1)), int(binding.get("end", -1))
            except (TypeError, ValueError):
                continue
            quote = str(binding.get("quote", ""))
            value = str(binding.get("value", ""))
            if (
                evidence.get("source") == binding.get("source")
                and 0 <= start < end <= len(str(evidence.get("text", "")))
                and str(evidence.get("text", ""))[start:end] == quote == value
            ):
                valid_bindings.append(binding)

        values = list(dict.fromkeys(
            str(binding.get("value", "")) for binding in valid_bindings
            if binding.get("value") is not None
        ))
        if not valid_bindings:
            result = "insufficient"
        elif status == "conflict":
            result = "conflict"
        elif status == "supported" and len(values) == 1:
            result = "supported"
        else:
            result = "insufficient"
        results.append({"attribute": item.get("attribute"), "status": result})
    return {
        "attributes": results,
        "has_conflict": any(item["status"] == "conflict" for item in results),
        "supported": [item["attribute"] for item in results if item["status"] == "supported"],
        "insufficient": [item["attribute"] for item in results if item["status"] == "insufficient"],
    }


def build_composer_prompt(base_prompt: str, ledger: dict, validation: dict) -> str:
    """Give the composer only uniquely supported attribute values."""
    allowed = []
    by_name = {item["attribute"]: item for item in ledger.get("attributes", [])}
    for name in validation.get("supported", []):
        item = by_name.get(name, {})
        allowed.append({
            "attribute": name,
            "value": item.get("values", [None])[0],
            "evidence": item.get("bindings", []),
        })
    template_policy = (
        "\n回答文の作成手順:\n"
        "1. 質問に適した短い回答文を作成する。\n"
        "2. 属性値はATTRIBUTE_BINDINGS_JSONの検証済み値だけを使い、"
        "言い換え・補完・推測・単位変更をしない。\n"
        "3. 検証済み属性をすべて一度ずつ含める。"
        "プレースホルダーを最終回答に残さない。\n"
        "4. 根拠にない説明や一般論を追加しない。回答本文だけを出力する。\n"
    )
    return (base_prompt + "\nATTRIBUTE_BINDINGS_JSON (verified data only):\n"
            + json.dumps(allowed, ensure_ascii=False, separators=(",", ":"))
            + template_policy)


def build_attribute_retry_prompt(
    question: str,
    accumulated_answer: str,
    ledger: dict,
    validation: dict,
    missing_attributes: Iterable[str],
) -> str:
    """Request only a missing, already evidence-supported attribute delta."""
    selected = set(missing_attributes)
    by_name = {item["attribute"]: item for item in ledger.get("attributes", [])}
    targets = []
    for name in missing_attributes:
        item = by_name.get(name)
        if not item or name not in validation.get("supported", []):
            continue
        targets.append({
            "attribute": name,
            "value": item.get("values", [None])[0],
            "bindings": item.get("bindings", []),
        })
    if not targets:
        raise ValueError("attribute retry requires supported missing attributes")
    return (
        "日本語で、未回答の属性だけを補完してください。\n"
        "質問本文\n" + str(question) + "\n"
        "現在までに確定した回答\n" + str(accumulated_answer) + "\n"
        "今回の再生成対象属性JSON (verified data only):\n"
        + json.dumps(targets, ensure_ascii=False, separators=(",", ":")) + "\n"
        "制約:\n"
        "- 今回の対象属性だけを追加し、確定済み回答を繰り返さない。\n"
        "- 各値はbindingsのquoteと完全一致させ、evidence_idの根拠だけを使う。\n"
        "- 対象外の属性、推測、一般論、根拠にない説明を追加しない。\n"
        "- 根拠にある値を簡潔な日本語で出力する。見出し・手順・JSONは出力しない。\n"
        "追加回答本文\n"
    )


def validate_composed_answer(answer: str, ledger: dict, validation: dict) -> tuple[bool, str]:
    text = (answer or "").strip()
    if not text:
        return False, "empty"
    if validation.get("has_conflict"):
        return False, "attribute_conflict"
    if re.search(r"\{\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\}", text):
        return False, "unfilled_template_placeholder"
    by_name = {item["attribute"]: item for item in ledger.get("attributes", [])}
    evidence_by_id = {item.get("evidence_id"): item for item in ledger.get("evidence", [])}
    for name in validation.get("supported", []):
        item = by_name.get(name, {})
        values = item.get("values", [])
        bindings = item.get("bindings", [])
        if len(values) != 1 or values[0] not in text:
            return False, "verified_attribute_missing:" + str(name)
        if not any(binding.get("evidence_id") in evidence_by_id
                   and binding.get("quote") == values[0]
                   for binding in bindings):
            return False, "verified_attribute_without_evidence:" + str(name)
    for name in validation.get("insufficient", []):
        values = by_name.get(name, {}).get("values", [])
        if any(value and value in text for value in values):
            return False, "unverified_attribute:" + str(name)
    if re.search(r"</?(?:system|assistant|developer|instruction|think|analysis)[^>]*>", text, re.I):
        return False, "control_or_role_tag"
    return True, "accepted"
