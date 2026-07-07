"""Forecast ensemble heads."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from engram.models.track_b import DelinquencyBucket

if TYPE_CHECKING:
    from engram.forecasting.protocol import Forecaster

_CLASSES = [bucket.value for bucket in DelinquencyBucket]


class LinearPoolEnsemble:
    """Weighted log-linear pool over already-registered heads."""

    name = "linear_pool_ensemble"

    def __init__(self, heads: list[Forecaster], weights: list[float] | None = None) -> None:
        self.heads = heads
        self.weights = weights or [1.0 / len(heads)] * len(heads)

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        for head in self.heads:
            head.fit(train_rows)

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        log_scores = {bucket: 0.0 for bucket in _CLASSES}
        for head, weight in zip(self.heads, self.weights, strict=True):
            probs = head.predict_proba(features)
            for bucket in _CLASSES:
                log_scores[bucket] += weight * math.log(max(probs.get(bucket, 0.0), 1e-12))
        raw = {bucket: math.exp(value) for bucket, value in log_scores.items()}
        total = sum(raw.values())
        return {bucket: value / total for bucket, value in raw.items()}
