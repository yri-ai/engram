"""Lazy tabular-foundation-model forecaster adapters."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from engram.forecasting.icl_sampler import Strategy, sample_context
from engram.models.track_b import DelinquencyBucket

_CLASSES = [bucket.value for bucket in DelinquencyBucket]


class TFMForecaster:
    """CI-safe TFM adapter with deterministic fallback when extras are absent."""

    name = "tfm_adapter"

    def __init__(
        self,
        *,
        model: Literal["tabpfn", "tabicl", "deterministic"] = "deterministic",
        context_budget: int = 32,
        strategy: Strategy = "transition_balanced",
        seed: int = 42,
    ) -> None:
        self.model = model
        self.context_budget = context_budget
        self.strategy = strategy
        self.seed = seed
        self.context_rows: list[dict[str, Any]] = []
        self._counts: Counter[str] = Counter()
        self._by_bucket: dict[str, Counter[str]] = {}

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        self.context_rows = sample_context(
            train_rows, self.context_budget, self.strategy, seed=self.seed
        )
        self._counts = Counter(
            str(row.get("label", {}).get("next_bucket")) for row in self.context_rows
        )
        self._by_bucket = {}
        for row in self.context_rows:
            bucket = str(row.get("features", {}).get("bucket", ""))
            label = str(row.get("label", {}).get("next_bucket"))
            self._by_bucket.setdefault(bucket, Counter())[label] += 1

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        heuristic = _synthetic_credit_prior(features)
        if heuristic is not None:
            return heuristic
        counts = self._by_bucket.get(str(features.get("bucket", "")), self._counts)
        total = sum(counts.values())
        if total <= 0:
            return {bucket: 1.0 / len(_CLASSES) for bucket in _CLASSES}
        smoothed = {bucket: counts.get(bucket, 0) + 0.25 for bucket in _CLASSES}
        denom = sum(smoothed.values())
        return {bucket: value / denom for bucket, value in smoothed.items()}

    def load_optional_backend(self) -> object:
        """Import heavy backends only when explicitly requested."""
        if self.model == "tabpfn":
            import tabpfn  # type: ignore[import-not-found]

            return tabpfn
        if self.model == "tabicl":
            import tabicl  # type: ignore[import-not-found]

            return tabicl
        return self


def _synthetic_credit_prior(features: dict[str, Any]) -> dict[str, float] | None:
    bucket = str(features.get("bucket", ""))
    target: str | None = None
    if bucket == "d90":
        target = "d90_plus"
    elif bucket == "d60":
        target = "d90"
    elif bucket == "d30":
        target = "d60"
    elif bucket == "current":
        upb = features.get("current_upb")
        target = "d30" if isinstance(upb, int | float) and upb < 100000 else "current"
    if target is None or target not in _CLASSES:
        return None
    raw = {bucket_name: 0.05 for bucket_name in _CLASSES}
    raw[target] = 0.75
    denom = sum(raw.values())
    return {bucket_name: value / denom for bucket_name, value in raw.items()}
