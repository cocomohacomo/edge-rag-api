"""Evidence-first ledger used to ground mandatory LLM generation.

Evidence is treated as data, never as instructions. This module extracts a compact,
provenance-preserving ledger for the answer prompt and does not call an LLM.
"""
from __future__ import annotations

import re
from typing import Iterable


def build_evidence_ledger(question: str, hits: Iterable[dict], plan: dict | None = None) -> dict:
    entries = []
    for hit_index, hit in enumerate(hits, 1):
        source = str(hit.get("source", "unknown"))
        text = str(hit.get("text", "")).strip()
        if not text:
            continue
        # Keep retrieval-order identity stable with Attribute Ledger E{index}.
        # Duplicate passages are retained because different retrieved chunks must
        # never silently collapse into a different evidence ID.
        # Preserve exact source text; extraction is deliberately lexical and conservative.
        facts = []
        for pattern in (r"https?://[^\s<>、。]+", r"\d+(?:[.,]\d+)?\s*(?:GB|MB|GiB|MiB|%|日|月|年|時|分)",
                       r"i[3579]-?\d{3,5}[A-Za-z]?", r"\d{1,4}年\d{1,2}月\d{1,2}日", r"\d{1,2}月\d{1,2}日"):
            facts.extend(re.findall(pattern, text, re.I))
        entries.append({"evidence_id": f"E{hit_index}", "source": source,
                       "text": text[:900], "literal_values": list(dict.fromkeys(facts))[:24]})
    requirements = (plan or {}).get("requirements", [])
    missing = (plan or {}).get("missing_requirements", [])
    return {"question": question, "required_attributes": list(requirements),
            "attributes_without_evidence": list(missing), "evidence": entries}


def render_ledger(ledger: dict) -> str:
    import json
    return json.dumps(ledger, ensure_ascii=False, separators=(",", ":"))
