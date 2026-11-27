"""Deterministic completeness metrics and independent evidence-unit checker."""
from __future__ import annotations

import re


_NUMERIC_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?:万円|億円|千円|%|GB|MB|GiB|MiB|日|月|年|時|分)?",
    re.I,
)


def _claim_tokens(text: str) -> list[str]:
    """Tokenize Japanese runs into the same character bigrams used by retrieval."""
    tokens = []
    for token in re.findall(r"[ぁ-んァ-ヶ一-龠ー]+|[a-z0-9_]+", (text or "").casefold()):
        if re.fullmatch(r"[ぁ-んァ-ヶ一-龠ー]+", token):
            tokens.extend([token] if len(token) == 1 else
                          [token[index:index + 2] for index in range(len(token) - 1)])
        else:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def missing_supported_attributes(ledger: dict, validation: dict, answer: str) -> list[str]:
    """Return only supported attributes whose exact bound value is absent."""
    text = answer or ""
    by_name = {str(item.get("attribute", "")): item for item in ledger.get("attributes", [])}
    missing = []
    for name in validation.get("supported", []):
        values = [str(value) for value in by_name.get(name, {}).get("values", [])
                  if value is not None and str(value)]
        if len(values) != 1 or values[0] not in text:
            missing.append(str(name))
    return missing


def evaluate_attribute_completeness(ledger: dict, validation: dict, answer: str) -> dict:
    attributes = list(ledger.get("attributes", []))
    supported = list(dict.fromkeys(validation.get("supported", [])))
    missing = missing_supported_attributes(ledger, validation, answer)
    complete = len(supported) - len(missing)
    unsupported = [
        str(item.get("attribute", "")) for item in attributes
        if item.get("attribute") and item.get("attribute") not in supported
    ]
    return {
        "attribute_count": len(attributes),
        "supported_count": len(supported),
        "complete_count": complete,
        "completeness_rate": round(complete / len(supported), 4) if supported else 1.0,
        "missing_supported_attributes": missing,
        "unsupported_attributes": unsupported,
        "conflict": bool(validation.get("has_conflict")),
    }


def validate_attribute_claims(answer: str, ledger: dict, validation: dict) -> dict:
    text = answer or ""
    by_name = {str(item.get("attribute", "")): item for item in ledger.get("attributes", [])}
    missing = missing_supported_attributes(ledger, validation, text)
    unsupported_values = []
    for name in dict.fromkeys(validation.get("insufficient", [])):
        values = [str(value) for value in by_name.get(name, {}).get("values", [])
                  if value is not None and str(value)]
        unsupported_values.extend(value for value in values if value in text)
    unsupported_values = list(dict.fromkeys(unsupported_values))
    conflict = bool(validation.get("has_conflict"))
    return {
        "valid": not missing and not unsupported_values and not conflict,
        "missing_attributes": missing,
        "unsupported_values": unsupported_values,
        "conflict": conflict,
        "limitation": "Only explicit evidence-bound values are checked; semantic entailment is handled by the independent claim checker.",
    }


def validate_answer_claims(answer: str, ledger: dict, validation: dict,
                           *, min_support_ratio: float = 0.40) -> dict:
    """Independent sentence checker; each claim must map to one exact evidence unit."""
    text = (answer or "").strip()
    evidence = [
        item for item in ledger.get("evidence", [])
        if item.get("evidence_id") and str(item.get("text", "")).strip()
    ]
    evidence_by_id = {str(item["evidence_id"]): item for item in evidence}
    claims = []
    unsupported_claims = []
    for raw_claim in re.split(r"(?<=[。！？!?])\s*|\n+", text):
        claim = re.sub(r"[（(]出典\s*[:：][^）)]*[）)]", "", raw_claim).strip()
        if not claim:
            continue
        tokens = _claim_tokens(claim)
        if not tokens:
            continue
        claim_numbers = list(dict.fromkeys(_NUMERIC_RE.findall(claim)))
        claim_urls = re.findall(r"""https?://[^\s<>"'、。]+""", claim)
        claim_paths = re.findall(
            r"(?<![A-Za-z0-9./:])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", claim
        )
        identifiers = list(dict.fromkeys(claim_numbers + claim_urls + claim_paths))
        best_id, best_ratio, best_missing = None, 0.0, identifiers
        for item in evidence:
            passage = str(item.get("text", ""))
            matched = sum(1 for token in tokens if token in _claim_tokens(passage))
            ratio = matched / len(tokens)
            missing_here = [
                value for value in identifiers
                if value.casefold() not in passage.casefold()
            ]
            if (ratio, -len(missing_here)) > (best_ratio, -len(best_missing)):
                best_id = str(item["evidence_id"])
                best_ratio = ratio
                best_missing = missing_here
        unsupported_identifiers = best_missing if identifiers else []
        supported = bool(best_id and best_ratio >= min_support_ratio) and not unsupported_identifiers
        entry = {
            "claim": claim,
            "evidence_ids": ([best_id] if supported else []),
            "support_ratio": round(best_ratio, 4),
            "unsupported_identifiers": list(dict.fromkeys(unsupported_identifiers)),
            "supported": supported,
        }
        claims.append(entry)
        if not supported:
            unsupported_claims.append(entry)

    attribute_bindings = {}
    by_name = {str(item.get("attribute", "")): item for item in ledger.get("attributes", [])}
    for name in validation.get("supported", []):
        item = by_name.get(str(name), {})
        values = [str(value) for value in item.get("values", []) if value is not None]
        bindings = item.get("bindings", [])
        present = bool(values and values[0] in text)
        source_ids = []
        if values:
            for binding in bindings:
                evidence_id = str(binding.get("evidence_id"))
                evidence_item = evidence_by_id.get(evidence_id)
                if evidence_item is None or binding.get("quote") != values[0]:
                    continue
                try:
                    start, end = int(binding.get("start", -1)), int(binding.get("end", -1))
                except (TypeError, ValueError):
                    continue
                source_text = str(evidence_item.get("text", ""))
                if (
                    evidence_item.get("source") == binding.get("source")
                    and 0 <= start < end <= len(source_text)
                    and source_text[start:end] == values[0]
                ):
                    source_ids.append(evidence_id)
        source_ids = sorted(set(source_ids))
        attribute_bindings[str(name)] = {
            "present": present,
            "evidence_ids": source_ids if present else [],
            "supported": present and bool(source_ids),
        }

    unsupported_identifiers = sorted({
        value for claim in claims for value in claim["unsupported_identifiers"]
    })
    attributes_supported = all(item["supported"] for item in attribute_bindings.values())
    valid = bool(claims) and not unsupported_claims and not unsupported_identifiers and attributes_supported
    return {
        "valid": valid,
        "claims": claims,
        "unsupported_claims": [item["claim"] for item in unsupported_claims],
        "unsupported_identifiers": unsupported_identifiers,
        "attribute_bindings": attribute_bindings,
        "evidence_unit_count": len(evidence),
        "limitation": "Deterministic lexical and exact-identifier checks; not a full semantic entailment proof.",
    }
