"""Graph-native forecast heads with checkpoint-free deterministic adapters."""

from __future__ import annotations

from collections import Counter
from typing import Any

from engram.forecasting.tfm import _synthetic_credit_prior
from engram.models.track_b import DelinquencyBucket

_CLASSES = [bucket.value for bucket in DelinquencyBucket]


class UltraForecaster:
    name = "ultra_deterministic_adapter"

    def __init__(self, *, edges: list[dict[str, str]] | None = None) -> None:
        self.edges = edges or []
        self._counts: Counter[str] = Counter()

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        self._counts = Counter(str(row.get("label", {}).get("next_bucket")) for row in train_rows)

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        heuristic = _synthetic_credit_prior(features)
        if heuristic is not None:
            sharpened = {bucket: value * value for bucket, value in heuristic.items()}
            total = sum(sharpened.values())
            return {bucket: value / total for bucket, value in sharpened.items()}
        loan_id = str(features.get("loan_id", ""))
        graph_votes = Counter(edge["tail"] for edge in self.edges if edge.get("head") == loan_id)
        raw = {
            bucket: self._counts.get(bucket, 0) + graph_votes.get(bucket, 0) + 1
            for bucket in _CLASSES
        }
        total = sum(raw.values())
        return {bucket: value / total for bucket, value in raw.items()}

    def load_optional_backend(self) -> object:
        import torch  # type: ignore[import-not-found]

        return torch


class TemporalGNNForecaster(UltraForecaster):
    name = "temporal_gnn_deterministic_adapter"
