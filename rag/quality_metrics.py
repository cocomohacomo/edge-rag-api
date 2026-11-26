"""Deterministic attribute completeness and evidence-bound claim diagnostics."""

def evaluate_attribute_completeness(ledger: dict, validation: dict, answer: str) -> dict:
    attributes = list(ledger.get("attributes", []))
    supported = set(validation.get("supported", []))
    total = len(attributes)
    missing, unsupported, complete = [], [], 0
    for item in attributes:
        name = str(item.get("attribute", ""))
        values = [str(v) for v in item.get("values", []) if v is not None]
        if name in supported and values and any(v in (answer or "") for v in values):
            complete += 1
        elif name in supported:
            missing.append(name)
        else:
            unsupported.append(name)
    return {"attribute_count": total, "supported_count": len(supported),
            "complete_count": complete, "completeness_rate": round(complete / total, 4) if total else 0.0,
            "missing_supported_attributes": missing, "unsupported_attributes": unsupported,
            "conflict": bool(validation.get("has_conflict"))}

def validate_attribute_claims(answer: str, ledger: dict, validation: dict) -> dict:
    """Require supported values and reject values from insufficient attributes."""
    text = answer or ""
    by_name = {item.get("attribute"): item for item in ledger.get("attributes", [])}
    missing, unsupported_values = [], []
    for name in validation.get("supported", []):
        values = [str(v) for v in by_name.get(name, {}).get("values", []) if v is not None]
        if not values or not any(v in text for v in values):
            missing.append(name)
    for name in validation.get("insufficient", []):
        values = [str(v) for v in by_name.get(name, {}).get("values", []) if v is not None]
        unsupported_values.extend(v for v in values if v in text)
    conflict = bool(validation.get("has_conflict"))
    return {"valid": not missing and not unsupported_values and not conflict,
            "missing_attributes": missing, "unsupported_values": list(dict.fromkeys(unsupported_values)),
            "conflict": conflict}
