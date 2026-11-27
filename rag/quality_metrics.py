"""Deterministic completeness metrics and independent evidence-unit checker."""
from __future__ import annotations

import re


_CLAIM_TOKEN_RE = re.compile(
    r"[一-龥ぁ-んァ-ヶー]{2,}|[A-Za-z][A-Za-z0-9_.:/-]*|"
    r"\d+(?:[.,]\d+)?(?:万円|億円|千円|%|GB|MB|GiB|MiB|日|月|年|時|分)?",
    re.I,
)
_NUMERIC_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?:%|GB|MB|GiB|MiB|日|月|年|時|分)?",
    re.I,
)


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
    by_name = {str(item.get("attribute", "")): item for item in attributes}
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
        tokens = list(dict.fromkeys(_CLAIM_TOKEN_RE.findall(claim)))
        if not tokens:
            continue
        best_id, best_ratio = None, 0.0
        for item in evidence:
            passage = str(item.get("text", ""))
            matched = sum(1 for token in tokens if token.casefold() in passage.casefold())
            ratio = matched / len(tokens)
            if ratio > best_ratio:
                best_id, best_ratio = str(item["evidence_id"]), ratio

        claim_numbers = list(dict.fromkeys(_NUMERIC_RE.findall(claim)))
        claim_urls = re.findall(r"""https?://[^\s<>"'、。]+""", claim)
        claim_paths = re.findall(
            r"(?<![A-Za-z0-9./:])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", claim
        )
        exact_support_ids = []
        for item in evidence:
            passage = str(item.get("text", ""))
            if all(value.casefold() in passage.casefold() for value in claim_numbers + claim_urls + claim_paths):
                if claim_numbers or claim_urls or claim_paths:
                    exact_support_ids.append(str(item["evidence_id"]))
        unsupported_identifiers = []
        for value in claim_numbers + claim_urls + claim_paths:
            if not any(value.casefold() in str(item.get("text", "")).casefold() for item in evidence):
                unsupported_identifiers.append(value)
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
        source_ids = sorted({
            str(binding.get("evidence_id")) for binding in bindings
            if values
            and binding.get("evidence_id") in evidence_by_id
            and binding.get("quote") == values[0]
        })
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
