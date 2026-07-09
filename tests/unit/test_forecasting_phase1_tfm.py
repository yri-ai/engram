"""Phase 1 TFM, feature, sampler, and distillation coverage."""

from __future__ import annotations

import numpy as np

from engram.forecasting.distill import distill
from engram.forecasting.features import FeatureConfig, rows_to_matrix
from engram.forecasting.fixtures import load_forecast_fixture_rows
from engram.forecasting.icl_sampler import sample_context
from engram.forecasting.tfm import TFMForecaster


def test_rows_to_matrix_round_trip_unknown_category_and_nan(tmp_path):
    rows = load_forecast_fixture_rows("track_b_synthetic")
    x_train, labels, names, config = rows_to_matrix(rows[:3])
    assert x_train.shape[0] == 3
    assert labels
    assert any(name.endswith("=<UNK>") for name in names)

    path = tmp_path / "features.json"
    config.save(path)
    loaded = FeatureConfig.load(path)
    new_row = {
        "features": {"bucket": "brand_new", "credit_score": None},
        "label": {"next_bucket": "current"},
    }
    x_eval, _, eval_names, _ = rows_to_matrix([new_row], loaded)
    assert eval_names == names
    assert np.isnan(x_eval[0][loaded.numeric_features.index("credit_score")])


def test_sample_context_strategies_are_budgeted_and_deterministic():
    rows = load_forecast_fixture_rows("track_b_synthetic")
    first = sample_context(rows, 5, "transition_balanced", seed=7)
    second = sample_context(rows, 5, "transition_balanced", seed=7)
    assert len(first) == 5
    assert [row["loan_id"] for row in first] == [row["loan_id"] for row in second]
    assert len(sample_context(rows, 3, "recency_weighted")) == 3


def test_tfm_forecaster_conforms_and_distills_teacher_predictions():
    rows = load_forecast_fixture_rows("track_b_synthetic")
    forecaster = TFMForecaster(context_budget=8)
    forecaster.fit(rows[:8])
    probs = forecaster.predict_proba(rows[-1]["features"])
    assert abs(sum(probs.values()) - 1.0) < 1e-9

    student = distill(forecaster, None, rows[:8])
    assert student.predict_proba(rows[0]["features"]) == forecaster.predict_proba(
        rows[0]["features"]
    )
