"""Forecast metric helpers and lifecycle scoring seams.

The single-run lifecycle scoring helpers delegate to
``engram.services.forecast_scoring`` so there is one canonical implementation,
while the Phase 0 evaluation harness also gets batch metrics over prediction
rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast, overload

import numpy as np

from engram.services.forecast_scoring import (
    assign_calibration_bucket,
    binary_brier_score,
    log_score,
    probability_assigned_to_resolved_branch,
    top_1_accuracy,
    top_k_accuracy,
)
from engram.services.forecast_scoring import (
    multiclass_brier_score as _single_multiclass_brier_score,
)

ProbabilityRow = Mapping[str, float]


@overload
def multiclass_brier_score(
    arg1: Sequence[ProbabilityRow], arg2: Sequence[str], classes: Sequence[str]
) -> float: ...


@overload
def multiclass_brier_score(arg1: Mapping[str, float], arg2: str, classes: None = None) -> float: ...


def multiclass_brier_score(
    arg1: object,
    arg2: object,
    classes: Sequence[str] | None = None,
) -> float:
    """Return multiclass Brier score for either batch rows or one forecast run.

    ``multiclass_brier_score(probabilities, resolved_branch)`` is the lifecycle
    scoring seam. ``multiclass_brier_score(predictions, labels, classes)`` is
    the Phase 0 harness batch metric.
    """
    if classes is None:
        if not isinstance(arg2, str):
            raise TypeError("resolved_branch must be a string")
        if not isinstance(arg1, Mapping):
            raise TypeError("probabilities must be a mapping when classes is omitted")
        probabilities = {str(key): float(value) for key, value in arg1.items()}
        return _single_multiclass_brier_score(probabilities, arg2)

    if isinstance(arg2, str):
        raise TypeError("labels must be a sequence when classes is provided")
    predictions = cast("Sequence[ProbabilityRow]", arg1)
    labels = cast("Sequence[str]", arg2)
    if not predictions:
        return 0.0
    total = 0.0
    for prediction, label in zip(predictions, labels, strict=True):
        for class_name in classes:
            predicted = float(prediction.get(class_name, 0.0))
            observed = 1.0 if class_name == label else 0.0
            total += (predicted - observed) ** 2
    return total / len(predictions)


def log_loss(
    predictions: Sequence[ProbabilityRow], labels: Sequence[str], classes: Sequence[str]
) -> float:
    """Return clipped multiclass negative log likelihood."""
    if not predictions:
        return 0.0
    del classes  # included for API symmetry and future class validation
    epsilon = 1e-15
    total = 0.0
    for prediction, label in zip(predictions, labels, strict=True):
        probability = min(max(float(prediction.get(label, 0.0)), epsilon), 1.0 - epsilon)
        total -= math.log(probability)
    return total / len(predictions)


def top1_accuracy(predictions: Sequence[ProbabilityRow], labels: Sequence[str]) -> float:
    """Return the fraction of rows whose highest-probability class is correct."""
    if not predictions:
        return 0.0
    correct = 0
    for prediction, label in zip(predictions, labels, strict=True):
        if not prediction:
            continue
        top_class = max(prediction, key=lambda class_name: prediction[class_name])
        correct += int(top_class == label)
    return correct / len(predictions)


def calibration_bins(
    predictions: Sequence[ProbabilityRow], labels: Sequence[str], n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Return reliability-curve bins for top-label confidence calibration."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bins: list[dict[str, float | int]] = [
        {
            "bin_lower": index / n_bins,
            "bin_upper": (index + 1) / n_bins,
            "count": 0,
            "accuracy": 0.0,
            "confidence": 0.0,
        }
        for index in range(n_bins)
    ]
    if not predictions:
        return bins

    correct_totals = [0.0 for _ in range(n_bins)]
    confidence_totals = [0.0 for _ in range(n_bins)]
    counts = [0 for _ in range(n_bins)]
    for prediction, label in zip(predictions, labels, strict=True):
        if prediction:
            top_class = max(prediction, key=lambda class_name: prediction[class_name])
            confidence = float(prediction[top_class])
        else:
            top_class = ""
            confidence = 0.0
        index = max(0, min(math.ceil(confidence * n_bins) - 1, n_bins - 1))
        counts[index] += 1
        confidence_totals[index] += confidence
        correct_totals[index] += float(top_class == label)

    for index, item in enumerate(bins):
        count = counts[index]
        item["count"] = count
        if count:
            item["accuracy"] = correct_totals[index] / count
            item["confidence"] = confidence_totals[index] / count
    return bins


def expected_calibration_error(
    predictions: Sequence[ProbabilityRow], labels: Sequence[str], n_bins: int = 10
) -> float:
    """Return top-label expected calibration error."""
    if not predictions:
        return 0.0
    bins = calibration_bins(predictions, labels, n_bins=n_bins)
    total = len(predictions)
    ece = 0.0
    for item in bins:
        count = int(item["count"])
        ece += (count / total) * abs(float(item["accuracy"]) - float(item["confidence"]))
    return ece


def one_vs_rest_auc(
    predictions: Sequence[ProbabilityRow], labels: Sequence[str], classes: Sequence[str]
) -> dict[str, float | None]:
    """Return one-vs-rest AUC for each class, or None when undefined."""
    return {class_name: _binary_auc(predictions, labels, class_name) for class_name in classes}


def _binary_auc(
    predictions: Sequence[ProbabilityRow], labels: Sequence[str], positive_class: str
) -> float | None:
    positive_scores = [
        float(prediction.get(positive_class, 0.0))
        for prediction, label in zip(predictions, labels, strict=True)
        if label == positive_class
    ]
    negative_scores = [
        float(prediction.get(positive_class, 0.0))
        for prediction, label in zip(predictions, labels, strict=True)
        if label != positive_class
    ]
    if not positive_scores or not negative_scores:
        return None

    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def loss_weighted_error(
    predictions: Sequence[ProbabilityRow],
    labels: Sequence[str],
    classes: Sequence[str],
    cost_matrix: Mapping[tuple[str, str], float] | None = None,
) -> float:
    """Return mean top-label error weighted by transition cost."""
    if not predictions:
        return 0.0
    costs = cost_matrix or _distance_cost_matrix(classes)
    total = 0.0
    for prediction, label in zip(predictions, labels, strict=True):
        if not prediction:
            predicted = classes[0]
        else:
            predicted = max(prediction, key=lambda class_name: prediction[class_name])
        total += float(costs.get((label, predicted), 0.0 if label == predicted else 1.0))
    return total / len(predictions)


def _distance_cost_matrix(classes: Sequence[str]) -> dict[tuple[str, str], float]:
    indexes = {class_name: index for index, class_name in enumerate(classes)}
    distances = np.abs(
        np.subtract.outer(
            np.arange(len(classes), dtype=float), np.arange(len(classes), dtype=float)
        )
    )
    return {
        (actual, predicted): float(distances[indexes[actual], indexes[predicted]])
        for actual in classes
        for predicted in classes
    }


__all__ = [
    "assign_calibration_bucket",
    "binary_brier_score",
    "calibration_bins",
    "expected_calibration_error",
    "log_loss",
    "log_score",
    "loss_weighted_error",
    "multiclass_brier_score",
    "one_vs_rest_auc",
    "ProbabilityRow",
    "probability_assigned_to_resolved_branch",
    "top1_accuracy",
    "top_1_accuracy",
    "top_k_accuracy",
]
