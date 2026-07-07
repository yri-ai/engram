"""Phase 5 conformal, output contract, and monitoring coverage."""

from __future__ import annotations

from engram.forecasting.conformal import (
    conformal_threshold,
    prediction_set,
    timing_band,
    weighted_conformal_threshold,
)
from engram.forecasting.monitoring import rolling_calibration_alarm
from engram.forecasting.output import PredictionReport


def test_conformal_prediction_sets_and_timing_bands():
    calibration = [({"current": 0.9, "d30": 0.1}, "current"), ({"current": 0.2, "d30": 0.8}, "d30")]
    threshold = conformal_threshold(calibration, alpha=0.1)
    weighted = weighted_conformal_threshold(
        [(calibration[0][0], calibration[0][1], 2.0)], alpha=0.1
    )
    assert threshold <= 0.2
    assert weighted <= 0.2
    assert prediction_set({"current": 0.95, "d30": 0.05}, threshold) == ["current"]
    assert timing_band([1, 2, 3, 4], alpha=0.25) == (1, 4)


def test_prediction_report_contract_and_calibration_alarm():
    report = PredictionReport(
        prediction_id="p1",
        as_of="2025-01-01",
        calibrated_probabilities={"current": 0.8, "d30": 0.2},
        conformal_set=["current"],
        evidence_chain=["message-1"],
        flip_conditions=["new delinquency evidence"],
        model_attribution={"tfm": 0.6, "graph": 0.4},
        cost={"usd": 0.01},
    )
    assert report.model_dump()["prediction_id"] == "p1"
    alarm = rolling_calibration_alarm([{"current": 0.9, "d30": 0.1}], ["d30"], threshold=0.05)
    assert alarm["alarm"] is True
