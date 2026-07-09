"""Drift and calibration monitoring helpers."""

from __future__ import annotations

from engram.forecasting.metrics import expected_calibration_error


def rolling_calibration_alarm(
    predictions: list[dict[str, float]], labels: list[str], *, threshold: float = 0.15
) -> dict[str, object]:
    ece = expected_calibration_error(predictions, labels, n_bins=10)
    return {"ece": ece, "alarm": ece > threshold}
