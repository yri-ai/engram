"""Task 0.1 prerequisites for the prediction-upgrade harness."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from scripts.data_collection.run_forecast_harness import resolve_input_rows

from engram.forecasting.fixtures import load_forecast_fixture_rows
from engram.models.track_b import DelinquencyBucket
from engram.services.track_b_forecasting import BaselineForecaster

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "track_b"


def test_forecast_optional_dependency_groups_and_markers_are_registered():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    optional = pyproject["project"]["optional-dependencies"]
    assert optional["forecast"] == [
        "scikit-learn>=1.5",
        "lightgbm>=4.5",
        "pandas>=2.2",
    ]
    assert optional["forecast-tfm"] == ["tabpfn>=3.0", "tabicl>=2.0", "torch>=2.4"]
    assert optional["forecast-graph"] == ["torch>=2.4", "torch-geometric>=2.6"]
    assert optional["forecast-conformal"] == ["crepes>=0.7"]

    marker_names = {
        marker.split(":", maxsplit=1)[0]
        for marker in pyproject["tool"]["pytest"]["ini_options"]["markers"]
    }
    assert {
        "integration",
        "slow",
        "forecast",
        "forecast_tfm",
        "forecast_graph",
        "forecast_conformal",
    } <= marker_names


@pytest.mark.forecast
def test_track_b_synthetic_fixture_loader_produces_canonical_rows():
    rows = load_forecast_fixture_rows("track_b_synthetic")

    assert len(rows) >= 12
    assert {row["split"] for row in rows} == {"train", "eval", "holdout"}
    assert {row["label"]["next_bucket"] for row in rows} >= {
        DelinquencyBucket.D90_PLUS.value,
        DelinquencyBucket.REO.value,
    }

    first = rows[0]
    assert set(first) >= {"event_id", "message_id", "loan_id", "as_of", "split", "features", "label"}
    assert first["label"]["horizon_months"] == 1
    assert first["features"]["bucket"] in {bucket.value for bucket in DelinquencyBucket}

    poisoned = [row for row in rows if "future_recorded_canary" in row["features"]]
    assert poisoned, "fixture must include a future-recorded feature canary for Task 0.4"
    for row in poisoned:
        provenance = row["feature_provenance"]["future_recorded_canary"]
        assert all(item["recorded_from"] > row["as_of"] for item in provenance)


@pytest.mark.forecast
def test_baseline_fixture_artifact_is_reproducible_to_one_basis_point():
    rows = load_forecast_fixture_rows("track_b_synthetic")
    expected = json.loads((FIXTURE_DIR / "track_b_synthetic_baseline.json").read_text())

    model = BaselineForecaster()
    model.fit([row for row in rows if row["split"] == "train"])
    actual = model.backtest([row for row in rows if row["split"] == "eval"])

    assert actual["sample_count"] == expected["sample_count"]
    assert actual["top1_accuracy"] == pytest.approx(expected["top1_accuracy"], abs=0.001)
    assert actual["brier_score"] == pytest.approx(expected["brier_score"], abs=0.001)
    assert expected["classes"] == [bucket.value for bucket in DelinquencyBucket]


def test_harness_resolves_checked_in_fixture_and_refuses_missing_local_events(tmp_path: Path):
    fixture_rows = resolve_input_rows(events=None, fixture="track_b_synthetic")
    assert fixture_rows == load_forecast_fixture_rows("track_b_synthetic")

    missing = tmp_path / "missing-events.ndjson"
    with pytest.raises(SystemExit) as exc:
        resolve_input_rows(events=missing, fixture=None)
    assert exc.value.code == 2
