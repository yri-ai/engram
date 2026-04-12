"""Tests for H3 predictive primitives."""

from engram.services.h3_primitives import (
    TransitionMatrixPrimitive,
    BranchRankingPrimitive,
    LatentTransitionPrimitive,
    compute_ece,
)


def _train_rows():
    return [
        {"features": {"bucket": "current"}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "d30"}},
        {"features": {"bucket": "d30"}, "label": {"next_bucket": "d60"}},
        {"features": {"bucket": "d30"}, "label": {"next_bucket": "current"}},
    ]


def test_transition_matrix_fit_predict():
    model = TransitionMatrixPrimitive("test")
    model.fit(_train_rows())
    pred = model.predict({"bucket": "current"})
    assert pred["top_bucket"] == "current"
    assert sum(pred["probabilities"].values()) > 0.99


def test_transition_matrix_backtest():
    model = TransitionMatrixPrimitive("test")
    model.fit(_train_rows())
    results = model.backtest(_train_rows())
    assert 0.0 <= results["top1_accuracy"] <= 1.0
    assert results["brier_score"] >= 0.0


def test_branch_ranking_primitive():
    rows = [
        {"features": {"bucket": "current"}, "label": {"next_bucket": "stable"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "stable"}},
        {"features": {"bucket": "current"}, "label": {"next_bucket": "deteriorate"}},
    ]
    model = BranchRankingPrimitive()
    model.fit(rows)
    pred = model.predict({"bucket": "current"})
    assert pred["top_bucket"] == "stable"


def test_latent_transition_uses_prev_bucket():
    rows = [
        {"features": {"bucket": "current", "prev_bucket": None}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current", "prev_bucket": None}, "label": {"next_bucket": "current"}},
        {"features": {"bucket": "current", "prev_bucket": "d30"}, "label": {"next_bucket": "d30"}},
        {"features": {"bucket": "current", "prev_bucket": "d30"}, "label": {"next_bucket": "d30"}},
    ]
    model = LatentTransitionPrimitive()
    model.fit(rows)
    # Without momentum: predict current
    pred_no_change = model.predict({"bucket": "current", "prev_bucket": None})
    assert pred_no_change["top_bucket"] == "current"
    # With momentum (recently changed): predict d30
    pred_change = model.predict({"bucket": "current", "prev_bucket": "d30"})
    assert pred_change["top_bucket"] == "d30"


def test_compute_ece():
    model = TransitionMatrixPrimitive("test")
    model.fit(_train_rows())
    ece = compute_ece(_train_rows(), model)
    assert 0.0 <= ece <= 1.0
