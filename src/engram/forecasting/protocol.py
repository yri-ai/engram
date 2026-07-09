"""Common forecaster protocol for prediction-upgrade model heads."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from engram.models.track_b import DelinquencyBucket
from engram.services.track_b_forecasting import BaselineForecaster

_ALL_BUCKETS = [bucket.value for bucket in DelinquencyBucket]


@runtime_checkable
class Forecaster(Protocol):
    """Minimal protocol every Track B forecast head must satisfy."""

    name: str

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        """Fit model state from canonical training rows."""
        ...

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        """Return bucket -> probability for the next delinquency bucket."""
        ...


class BaselineForecasterAdapter:
    """Adapter exposing the legacy transition-matrix baseline via Forecaster."""

    name = "baseline_transition_matrix"

    def __init__(self, model: BaselineForecaster | None = None) -> None:
        self._model = model or BaselineForecaster()

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        """Fit the wrapped baseline forecaster."""
        self._model.fit(train_rows)

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        """Predict all DelinquencyBucket probabilities in enum order.

        The legacy BaselineForecaster learns only labels observed in training.
        The Phase 0 harness needs a stable full-class distribution, so this
        adapter pads missing buckets with 0 and renormalizes defensive copies.
        """
        raw = self._model.predict(features)["probabilities"]
        probabilities = {bucket: float(raw.get(bucket, 0.0)) for bucket in _ALL_BUCKETS}
        total = sum(probabilities.values())
        if total <= 0.0:
            return {bucket: 1.0 / len(_ALL_BUCKETS) for bucket in _ALL_BUCKETS}
        return {bucket: probability / total for bucket, probability in probabilities.items()}
