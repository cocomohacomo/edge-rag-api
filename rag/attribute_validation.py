"""Attribute-level evidence binding, validation, composition, and final checks.

All offsets are zero-based Python string offsets into the exact retrieved text.
No model output is treated as evidence.
"""
from __future__ import annotations

import re
from typing import Iterable


def build_attribute_ledger(plan: dict, hits: Iterable[dict]) -> dict:
    sources = {}
    for index, hit in enumerate(hits, 1):
        source = str(hit.get("source", "unknown"))
        text = str(hit.get("text", ""))
        evidence_id = f"E{index}"
        sources.setdefault(source, []).append({"evidence_id": evidence_id, "text": text})
    attributes = []
    for name in plan.get("requirements", []):
        bindings = []
        distinct = []
        for entry in plan.get("evidence_mapping", {}).get(name, []):
            for source_entry in sources.get(str(entry.get("source", "unknown")), []):
                text = source_entry["text"]
                for value in entry.get("values", []):
                    value = str(value)
                    start = text.find(value)
                    if start < 0:
                        continue
                    binding = {"evidence_id": source_entry["evidence_id"],
                               "source": entry.get("source", "unknown"),
                               "start": start, "end": start + len(value),
                               "value": value, "quote": text[start:start + len(value)]}
                    if binding not in bindings:
                        bindings.append(binding)
                    if value not in distinct:
                        distinct.append(value)
        # Multiple candidate strings extracted from one passage are not, by themselves,\n        # contradictory evidence (e.g. a method sentence may mention CPU-only and\n        # local execution). A conflict requires distinct values independently\n        # asserted by separate retrieved evidence records.\n        value_evidence = {}\n        for binding in bindings:\n            value_evidence.setdefault(binding["value"], set()).add(binding["evidence_id"])\n        independently_supported = [value for value, ids in value_evidence.items() if ids]\n        conflict = len(independently_supported) > 1 and len({\n            evidence_id for ids in value_evidence.values() for evidence_id in ids\n        }) > 1\n        status = "insufficient" if not bindings else ("conflict" if conflict else "supported")
        attributes.append({"attribute": name, "status": status,
                           "values": distinct, "bindings": bindings})
    return {"question": plan.get("question", ""), "attributes": attributes}


def validate_attribute_ledger(ledger: dict) -> dict:
    results = []
    for item in ledger.get("attributes", []):
        status = item.get("status")
        bindings = item.get("bindings", [])
        valid = bool(bindings) and all(
            int(b.get("start", -1)) >= 0
            and int(b.get("end", -1)) > int(b.get("start", -1))
            and b.get("quote") == b.get("value")
            and len(b.get("quote", "")) == int(b["end"]) - int(b["start"])
            for b in bindings
        )
        if not valid:
            result = "insufficient"
        elif status == "conflict":
            result = "conflict"
        elif status == "supported" and len(item.get("values", [])) == 1:
            result = "supported"
        else:
            result = "insufficient"
        results.append({"attribute": item.get("attribute"), "status": result})
    return {"attributes": results,
            "has_conflict": any(x["status"] == "conflict" for x in results),
            "supported": [x["attribute"] for x in results if x["status"] == "supported"],
            "insufficient": [x["attribute"] for x in results if x["status"] == "insufficient"]}


def build_composer_prompt(base_prompt: str, ledger: dict, validation: dict) -> str:
    import json
    allowed = []
    by_name = {x["attribute"]: x for x in ledger.get("attributes", [])}
    for name in validation.get("supported", []):
        item = by_name.get(name, {})
        allowed.append({"attribute": name, "value": item.get("values", [None])[0],
                        "evidence": item.get("bindings", [])})
    return (base_prompt + "\nATTRIBUTE_BINDINGS_JSON (verified data only):\n"
            + json.dumps(allowed, ensure_ascii=False, separators=(",", ":"))
            + "\n文章化できるのは上記の検証済み属性だけです。未充足属性は未記載とし、"
              "推測・補完・別の根拠からの追加は禁止。属性値は原文どおり保持してください。\n")


def validate_composed_answer(answer: str, ledger: dict, validation: dict) -> tuple[bool, str]:
    text = (answer or "").strip()
    if not text:
        return False, "empty"
    if validation.get("has_conflict"):
        return False, "attribute_conflict"
    by_name = {x["attribute"]: x for x in ledger.get("attributes", [])}
    for name in validation.get("supported", []):
        values = by_name.get(name, {}).get("values", [])
        if len(values) != 1 or values[0] not in text:
            return False, "verified_attribute_missing:" + str(name)
    # Unsupported attribute values are never allowed to leak into the final answer.
    for name in validation.get("insufficient", []):
        values = by_name.get(name, {}).get("values", [])
        if any(value and value in text for value in values):
            return False, "unverified_attribute:" + str(name)
    if re.search(r"</?(?:system|assistant|developer|instruction|think|analysis)[^>]*>", text, re.I):
        return False, "control_or_role_tag"
    return True, "accepted"
