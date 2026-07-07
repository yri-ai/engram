"""Task 0.6 baseline ladder tests."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")

from engram.forecasting.baselines import GBMForecaster, HazardForecaster
from engram.forecasting.fixtures import load_forecast_fixture_rows
from engram.forecasting.metrics import multiclass_brier_score
from engram.forecasting.protocol import Forecaster
from engram.models.track_b import DelinquencyBucket

CLASSES = [bucket.value for bucket in DelinquencyBucket]
UNIFORM_BRIER = 1.0 - (1.0 / len(CLASSES))


def _fixture_train_eval() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = load_forecast_fixture_rows("track_b_synthetic")
    return (
        [row for row in rows if row["split"] == "train"],
        [row for row in rows if row["split"] == "eval"],
    )


@pytest.mark.forecast
def test_hazard_forecaster_conforms_and_beats_uniform_on_synthetic_fixture():
    train_rows, eval_rows = _fixture_train_eval()
    forecaster = HazardForecaster(random_state=7)

    forecaster.fit(train_rows)  # type: ignore[arg-type]
    predictions = [forecaster.predict_proba(row["features"]) for row in eval_rows]  # type: ignore[index]
    labels = [row["label"]["next_bucket"] for row in eval_rows]  # type: ignore[index]

    assert isinstance(forecaster, Forecaster)
    assert forecaster.name == "hazard_logistic"
    assert all(list(prediction) == CLASSES for prediction in predictions)
    assert all(sum(prediction.values()) == pytest.approx(1.0) for prediction in predictions)
    assert multiclass_brier_score(predictions, labels, CLASSES) < UNIFORM_BRIER


@pytest.mark.forecast
def test_gbm_forecaster_conforms_beats_uniform_and_is_deterministic():
    train_rows, eval_rows = _fixture_train_eval()
    first = GBMForecaster(random_state=11, n_estimators=12)
    second = GBMForecaster(random_state=11, n_estimators=12)

    first.fit(train_rows)  # type: ignore[arg-type]
    second.fit(train_rows)  # type: ignore[arg-type]
    first_predictions = [first.predict_proba(row["features"]) for row in eval_rows]  # type: ignore[index]
    second_predictions = [second.predict_proba(row["features"]) for row in eval_rows]  # type: ignore[index]
    labels = [row["label"]["next_bucket"] for row in eval_rows]  # type: ignore[index]

    assert isinstance(first, Forecaster)
    assert first.name == "gbm_lightgbm"
    assert first_predictions == second_predictions
    assert all(list(prediction) == CLASSES for prediction in first_predictions)
    assert multiclass_brier_score(first_predictions, labels, CLASSES) < UNIFORM_BRIER
