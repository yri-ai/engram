"""Prediction and evaluation harness for Engram forecasting."""

from engram.forecasting.fixtures import load_forecast_fixture_rows
from engram.forecasting.metrics import (
    assign_calibration_bucket,
    binary_brier_score,
    calibration_bins,
    expected_calibration_error,
    log_loss,
    log_score,
    loss_weighted_error,
    multiclass_brier_score,
    one_vs_rest_auc,
    probability_assigned_to_resolved_branch,
    top1_accuracy,
    top_1_accuracy,
    top_k_accuracy,
)
from engram.forecasting.protocol import BaselineForecasterAdapter, Forecaster

__all__ = [
    "BaselineForecasterAdapter",
    "Forecaster",
    "assign_calibration_bucket",
    "binary_brier_score",
    "calibration_bins",
    "expected_calibration_error",
    "load_forecast_fixture_rows",
    "log_loss",
    "log_score",
    "loss_weighted_error",
    "multiclass_brier_score",
    "one_vs_rest_auc",
    "probability_assigned_to_resolved_branch",
    "top1_accuracy",
    "top_1_accuracy",
    "top_k_accuracy",
]
