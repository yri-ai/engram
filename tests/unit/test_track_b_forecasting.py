"""Tests for Track B baseline forecaster."""

from engram.services.track_b_forecasting import BaselineForecaster


def test_baseline_forecaster_predicts():
    """Baseline model produces predictions with probabilities."""
    model = BaselineForecaster()

    # Train with some feature rows
    train_rows = [
        {"features": {"bucket": "current"}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "d30"}},
        {"features": {"bucket": "d30"}, "label": {"next_bucket": "d60"}},
        {"features": {"bucket": "d30"}, "label": {"next_bucket": "current"}},
    ]
    model.fit(train_rows)

    # Predict
    pred = model.predict({"bucket": "current"})
    assert "probabilities" in pred
    assert "top_bucket" in pred
    assert sum(pred["probabilities"].values()) > 0.99  # sums to ~1.0
    # Most common transition from current is current
    assert pred["top_bucket"] == "current"


def test_baseline_forecaster_handles_unseen_bucket():
    model = BaselineForecaster()
    train_rows = [
        {"features": {"bucket": "current"}, "label": {"next_bucket": "current"}},
    ]
    model.fit(train_rows)

    # Predict for a bucket not in training data
    pred = model.predict({"bucket": "d90"})
    assert "top_bucket" in pred
    # Falls back to global distribution
    assert pred["top_bucket"] == "current"


def test_baseline_forecaster_backtest():
    model = BaselineForecaster()
    train_rows = [
        {"features": {"bucket": "current"}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "d30"}},
    ]
    model.fit(train_rows)

    eval_rows = [
        {"features": {"bucket": "current"}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "d30"}},
    ]
    results = model.backtest(eval_rows)
    assert "top1_accuracy" in results
    assert "brier_score" in results
    assert "sample_count" in results
    assert results["sample_count"] == 2
    assert 0.0 <= results["top1_accuracy"] <= 1.0
    assert results["brier_score"] >= 0.0
