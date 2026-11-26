"""Deterministic, evidence-ledger-based attribute quality metrics."""


def evaluate_attribute_completeness(ledger: dict, validation: dict, answer: str) -> dict:
    attributes = list(ledger.get("attributes", []))
    supported = list(dict.fromkeys(validation.get("supported", [])))
    by_name = {str(item.get("attribute", "")): item for item in attributes}
    missing, unsupported, complete = [], [], 0
    text = answer or ""
    for name in supported:
        values = [str(value) for value in by_name.get(name, {}).get("values", []) if value is not None and str(value)]
        if values and any(value in text for value in values):
            complete += 1
        else:
            missing.append(name)
    supported_count = len(supported)
    for item in attributes:
        name = str(item.get("attribute", ""))
        if name and name not in supported:
            unsupported.append(name)
    return {
        "attribute_count": len(attributes),
        "supported_count": supported_count,
        "complete_count": complete,
        "completeness_rate": round(complete / supported_count, 4) if supported_count else 1.0,
        "missing_supported_attributes": missing,
        "unsupported_attributes": unsupported,
        "conflict": bool(validation.get("has_conflict")),
    }


def validate_attribute_claims(answer: str, ledger: dict, validation: dict) -> dict:
    """Check required supported values and explicitly known insufficient values."""
    text = answer or ""
    by_name = {str(item.get("attribute", "")): item for item in ledger.get("attributes", [])}
    missing, unsupported_values = [], []
    for name in dict.fromkeys(validation.get("supported", [])):
        values = [str(value) for value in by_name.get(name, {}).get("values", []) if value is not None and str(value)]
        if not values or not any(value in text for value in values):
            missing.append(name)
    for name in dict.fromkeys(validation.get("insufficient", [])):
        values = [str(value) for value in by_name.get(name, {}).get("values", []) if value is not None and str(value)]
        unsupported_values.extend(value for value in values if value in text)
    conflict = bool(validation.get("has_conflict"))
    unsupported_values = list(dict.fromkeys(unsupported_values))
    return {
        "valid": not missing and not unsupported_values and not conflict,
        "missing_attributes": missing,
        "unsupported_values": unsupported_values,
        "conflict": conflict,
        "limitation": "Only explicitly recorded insufficient values can be matched; novel unsupported claims require claim-to-evidence extraction.",
    }
