"""Distillation helpers for forecast heads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engram.forecasting.protocol import Forecaster


class DistilledForecaster:
    """Small deterministic student that stores teacher soft labels by feature signature."""

    name = "distilled_lookup"


    def __init__(self, default_distribution: dict[str, float]) -> None:
        self.default_distribution = default_distribution
        self._table: dict[str, dict[str, float]] = {}

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        del train_rows

    def add(self, features: dict[str, Any], probabilities: dict[str, float]) -> None:
        self._table[_signature(features)] = probabilities

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        return self._table.get(_signature(features), self.default_distribution)


def distill(
    teacher: Forecaster, student: DistilledForecaster | None, rows: list[dict[str, Any]], *, temperature: float = 1.0
) -> DistilledForecaster:
    """Fit a deterministic student on teacher probabilities."""
    del temperature
    teacher.fit(rows)
    first = teacher.predict_proba(rows[0]["features"]) if rows else {}
    distilled = student or DistilledForecaster(first)
    for row in rows:
        distilled.add(row["features"], teacher.predict_proba(row["features"]))
    return distilled


def _signature(features: dict[str, Any]) -> str:
    return repr(sorted(features.items()))
