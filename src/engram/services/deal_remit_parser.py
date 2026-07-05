"""Small remittance/trustee trigger-state parser."""

from __future__ import annotations


def parse_trigger_states(text: str, *, deal_id: str, source_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].lower() in {"oc", "ic"}:
            rows.append({"deal_id": deal_id, "trigger_type": parts[0].upper(), "period": parts[1], "passed": parts[2].lower() == "pass", "source_id": source_id})
    return rows
