"""Task 0.5 leakage canary tests."""

from __future__ import annotations

from typing import Any

import pytest

from engram.forecasting.canary import LeakageCanaryFailure, run_leakage_canary
from engram.forecasting.fixtures import load_forecast_fixture_rows
from engram.forecasting.protocol import BaselineForecasterAdapter
from engram.models.track_b import DelinquencyBucket


class DeliberatelyLeakyForecaster:
    name = "deliberately_leaky"

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        del train_rows

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        leaked = features.get("shuffled_future_bucket")
        buckets = [bucket.value for bucket in DelinquencyBucket]
        if isinstance(leaked, str) and leaked in buckets:
            return {bucket: 1.0 if bucket == leaked else 0.0 for bucket in buckets}
        return {bucket: 1.0 / len(buckets) for bucket in buckets}


@pytest.mark.slow
@pytest.mark.forecast
def test_leakage_canary_flags_deliberately_leaky_forecaster():
    rows = load_forecast_fixture_rows("track_b_synthetic")

    with pytest.raises(LeakageCanaryFailure, match="future-recorded feature improved"):
        run_leakage_canary(lambda: DeliberatelyLeakyForecaster(), rows)


@pytest.mark.slow
@pytest.mark.forecast
def test_leakage_canary_passes_honest_baseline():
    rows = load_forecast_fixture_rows("track_b_synthetic")

    result = run_leakage_canary(lambda: BaselineForecasterAdapter(), rows)

    assert result["status"] == "passed"
    assert result["red_team_detected"] is True
    assert result["future_feature_delta"] <= result["allowed_delta"]
