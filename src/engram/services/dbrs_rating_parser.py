"""DBRS rating-transition parser for scoreable deal targets."""

from __future__ import annotations


def parse_rating_transitions(text: str, *, deal_id: str, source_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4 and parts[0].lower() != "tranche_id":
            rows.append({"deal_id": deal_id, "tranche_id": parts[0], "rating_from": parts[1], "rating_to": parts[2], "action_at": parts[3], "source_id": source_id})
    return rows
