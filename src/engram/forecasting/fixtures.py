"""Checked-in forecasting fixtures used by the Phase 0 harness."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

_FIXTURE_FILES = {
    "track_b_synthetic": "track_b_synthetic_rows.jsonl",
}


def load_forecast_fixture_rows(name: str) -> list[dict[str, Any]]:
    """Load canonical forecast rows from a checked-in fixture.

    Fixtures are intentionally small, synthetic, and tracked in git so the
    leakage harness can be exercised without ignored local data files.
    """
    try:
        file_name = _FIXTURE_FILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_FIXTURE_FILES))
        raise ValueError(f"unknown forecast fixture {name!r}; known fixtures: {known}") from exc

    fixture = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "fixtures"
        / "track_b"
        / file_name
    )

    rows = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
    return deepcopy(rows)
