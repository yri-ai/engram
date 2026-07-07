"""Task 0.2 forecaster protocol and baseline adapter tests."""

from __future__ import annotations

from typing import Any

from engram.forecasting.protocol import BaselineForecasterAdapter, Forecaster
from engram.models.track_b import DelinquencyBucket


def _training_rows_without_rare_buckets() -> list[dict[str, Any]]:
    return [
        {"features": {"bucket": "current"}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "d30"}},
        {"features": {"bucket": "d30"}, "label": {"next_bucket": "current"}},
    ]


def test_baseline_adapter_conforms_to_forecaster_protocol():
    forecaster = BaselineForecasterAdapter()

    assert isinstance(forecaster, Forecaster)
    assert forecaster.name == "baseline_transition_matrix"


def test_baseline_adapter_predict_proba_sums_to_one():
    forecaster = BaselineForecasterAdapter()
    forecaster.fit(_training_rows_without_rare_buckets())

    probabilities = forecaster.predict_proba({"bucket": "current"})

    assert sum(probabilities.values()) == 1.0


def test_baseline_adapter_pads_probabilities_to_all_delinquency_buckets():
    forecaster = BaselineForecasterAdapter()
    forecaster.fit(_training_rows_without_rare_buckets())

    probabilities = forecaster.predict_proba({"bucket": "current"})

    assert list(probabilities) == [bucket.value for bucket in DelinquencyBucket]
    assert probabilities[DelinquencyBucket.D60.value] == 0.0
    assert probabilities[DelinquencyBucket.D90.value] == 0.0
    assert probabilities[DelinquencyBucket.D90_PLUS.value] == 0.0
    assert probabilities[DelinquencyBucket.REO.value] == 0.0
