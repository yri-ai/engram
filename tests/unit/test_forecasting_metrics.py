"""Task 0.3 forecast metric known-answer tests."""

from __future__ import annotations

import pytest

from engram.forecasting.metrics import (
    calibration_bins,
    expected_calibration_error,
    log_loss,
    loss_weighted_error,
    multiclass_brier_score,
    one_vs_rest_auc,
    top1_accuracy,
)

CLASSES = ["current", "d30", "d60"]
PREDICTIONS = [
    {"current": 0.8, "d30": 0.2, "d60": 0.0},
    {"current": 0.1, "d30": 0.7, "d60": 0.2},
    {"current": 0.2, "d30": 0.3, "d60": 0.5},
]
LABELS = ["current", "d60", "d60"]


def test_multiclass_brier_log_loss_and_top1_known_answers():
    assert multiclass_brier_score(PREDICTIONS, LABELS, CLASSES) == pytest.approx(0.5333333333)
    assert log_loss(PREDICTIONS, LABELS, CLASSES) == pytest.approx(0.8419095484)
    assert top1_accuracy(PREDICTIONS, LABELS) == pytest.approx(2 / 3)


def test_ece_and_reliability_curve_known_answers():
    bins = calibration_bins(PREDICTIONS, LABELS, n_bins=2)

    assert bins == [
        {"bin_lower": 0.0, "bin_upper": 0.5, "count": 1, "accuracy": 1.0, "confidence": 0.5},
        {"bin_lower": 0.5, "bin_upper": 1.0, "count": 2, "accuracy": 0.5, "confidence": 0.75},
    ]
    assert expected_calibration_error(PREDICTIONS, LABELS, n_bins=2) == pytest.approx(1 / 3)


def test_one_vs_rest_auc_per_transition_known_answer():
    auc = one_vs_rest_auc(PREDICTIONS, LABELS, CLASSES)

    assert auc["current"] == pytest.approx(1.0)
    assert auc["d30"] is None
    assert auc["d60"] == pytest.approx(1.0)


def test_loss_weighted_error_default_cost_matrix_uses_bucket_distance():
    predictions = [
        {"current": 0.1, "d30": 0.8, "d60": 0.1},
        {"current": 0.1, "d30": 0.1, "d60": 0.8},
    ]
    labels = ["current", "current"]

    assert loss_weighted_error(predictions, labels, CLASSES) == pytest.approx(1.5)


def test_metric_degenerate_cases():
    assert multiclass_brier_score([], [], CLASSES) == 0.0
    assert log_loss([], [], CLASSES) == 0.0
    assert top1_accuracy([], []) == 0.0
    assert expected_calibration_error([], [], n_bins=10) == 0.0
    assert calibration_bins([], [], n_bins=2) == [
        {"bin_lower": 0.0, "bin_upper": 0.5, "count": 0, "accuracy": 0.0, "confidence": 0.0},
        {"bin_lower": 0.5, "bin_upper": 1.0, "count": 0, "accuracy": 0.0, "confidence": 0.0},
    ]
    assert one_vs_rest_auc([{"current": 1.0}], ["current"], ["current"]) == {"current": None}
