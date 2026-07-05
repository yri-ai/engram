"""Leakage canary for the Phase 0 forecast harness."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Protocol

from engram.forecasting.metrics import multiclass_brier_score
from engram.forecasting.splits import record_time_filter
from engram.models.track_b import DelinquencyBucket

if TYPE_CHECKING:
    from collections.abc import Callable

_BUCKETS = [bucket.value for bucket in DelinquencyBucket]
_CANARY_FEATURE = "shuffled_future_bucket"
_FUTURE_RECORDED_FROM = "9999-12-31"


class LeakageCanaryFailure(AssertionError):  # noqa: N818
    """Raised when a future-recorded feature can improve harness scores."""


class _CanaryForecaster(Protocol):
    name: str

    def fit(self, train_rows: list[dict[str, Any]]) -> None: ...

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]: ...


def run_leakage_canary(
    forecaster_factory: Callable[[], _CanaryForecaster],
    rows: list[dict[str, Any]],
    *,
    allowed_delta: float = 0.05,
    detection_delta: float = 0.25,
) -> dict[str, float | bool | str]:
    """Assert future-recorded features cannot improve a forecaster's score.

    The canary first proves the harness is red-capable with a built-in leaky
    forecaster, then runs the supplied forecaster on rows poisoned with a label
    feature whose provenance is after ``as_of``. A leakage-clean harness filters
    that feature before scoring, so raw poisoned rows must not beat filtered rows.
    """
    poisoned_rows = _inject_future_label_feature(rows)
    red_team_delta = _score_delta(lambda: _FutureFeatureForecaster(), poisoned_rows)
    red_team_detected = red_team_delta > detection_delta
    if not red_team_detected:
        raise LeakageCanaryFailure(
            f"leakage canary was not red-capable: delta={red_team_delta:.6f}"
        )

    future_feature_delta = _score_delta(forecaster_factory, poisoned_rows)
    if future_feature_delta > allowed_delta:
        raise LeakageCanaryFailure(
            "future-recorded feature improved forecast score "
            f"by {future_feature_delta:.6f}, above allowed delta {allowed_delta:.6f}"
        )

    return {
        "status": "passed",
        "red_team_detected": red_team_detected,
        "future_feature_delta": future_feature_delta,
        "allowed_delta": allowed_delta,
    }


def _score_delta(
    forecaster_factory: Callable[[], _CanaryForecaster], rows: list[dict[str, Any]]
) -> float:
    raw_score = _score_rows(forecaster_factory, rows)
    filtered_rows = [record_time_filter(row, row["as_of"]) for row in rows]
    filtered_score = _score_rows(forecaster_factory, filtered_rows)
    return filtered_score - raw_score


def _score_rows(
    forecaster_factory: Callable[[], _CanaryForecaster], rows: list[dict[str, Any]]
) -> float:
    train_rows = [row for row in rows if row.get("split") == "train"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    if not train_rows or not eval_rows:
        raise ValueError("leakage canary requires train and eval rows")

    forecaster = forecaster_factory()
    forecaster.fit(train_rows)
    predictions = [forecaster.predict_proba(row["features"]) for row in eval_rows]
    labels = [row["label"]["next_bucket"] for row in eval_rows]
    return multiclass_brier_score(predictions, labels, _BUCKETS)


def _inject_future_label_feature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    poisoned = deepcopy(rows)
    for row in poisoned:
        row.setdefault("features", {})[_CANARY_FEATURE] = row["label"]["next_bucket"]
        row.setdefault("feature_provenance", {})[_CANARY_FEATURE] = [
            {"source_id": "leakage-canary", "recorded_from": _FUTURE_RECORDED_FROM}
        ]
    return poisoned


class _FutureFeatureForecaster:
    name = "future_feature_red_team"

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        del train_rows

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        leaked = features.get(_CANARY_FEATURE)
        if isinstance(leaked, str) and leaked in _BUCKETS:
            return {bucket: 1.0 if bucket == leaked else 0.0 for bucket in _BUCKETS}
        return {bucket: 1.0 / len(_BUCKETS) for bucket in _BUCKETS}
