"""Evidence assembly for branch forecasting."""

from __future__ import annotations

from typing import Any

from engram.forecasting.splits import _parse_record_time, record_time_filter


def assemble_evidence(
    loan_id: str,
    as_of: str,
    budget: int,
    motif_library: list[str],
    *,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select prior, record-time-valid evidence without inducing motifs at prediction time."""
    selected: list[dict[str, Any]] = []
    excluded_future = 0
    for row in rows:
        row_as_of = str(row.get("as_of", ""))
        if row.get("loan_id") != loan_id or _parse_record_time(row_as_of) > _parse_record_time(as_of):
            continue
        filtered = record_time_filter(row, as_of)
        if filtered != row:
            excluded_future += 1
        text = " ".join(str(value) for value in filtered.get("features", {}).values()).lower()
        if motif_library and not any(motif.lower() in text for motif in motif_library):
            continue
        selected.append(filtered)
    selected = sorted(selected, key=lambda row: str(row.get("as_of", "")), reverse=True)[:budget]
    return {
        "loan_id": loan_id,
        "as_of": as_of,
        "message_ids": [str(row.get("message_id")) for row in selected],
        "items": selected,
        "excluded_counts": {"future_record_time": excluded_future},
    }
