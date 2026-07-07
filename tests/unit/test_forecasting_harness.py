"""Task 0.7 harness runner and Gate 6 artifact tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from scripts.data_collection.run_forecast_harness import main, run_scoreboard

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")

from engram.forecasting.fixtures import load_forecast_fixture_rows

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.forecast
def test_scoreboard_contains_registered_models_metrics_and_canary_status(tmp_path: Path):
    rows = load_forecast_fixture_rows("track_b_synthetic")

    scoreboard = run_scoreboard(
        rows,
        model_names=["baseline", "hazard", "gbm"],
        output_dir=tmp_path,
        fixture_name="track_b_synthetic",
        decision_path=tmp_path / "gate-6.md",
    )

    assert scoreboard["fixture"] == "track_b_synthetic"
    assert scoreboard["sample_count"] == 4
    assert scoreboard["leakage_canary"]["status"] == "passed"
    assert scoreboard["leakage_canary"]["harness_filter_path_detected"] is True
    assert scoreboard["gate_6"]["leakage_canary_passed"] is True
    assert {model["model"] for model in scoreboard["models"]} == {
        "baseline",
        "hazard",
        "gbm",
    }
    for model in scoreboard["models"]:
        assert set(model["metrics"]) >= {
            "brier_score",
            "top1_accuracy",
            "log_loss",
            "ece",
            "loss_weighted_error",
        }
        assert model["windows"][0]["window_id"] == "fixture_eval"
        assert model["windows"][0]["sample_count"] == 4
        assert model["windows"][0]["calibration_bins"]


@pytest.mark.forecast
def test_harness_cli_writes_scoreboard_summary_and_gate_decision(tmp_path: Path):
    gate_path = tmp_path / "track-b-gate-6-decision.md"
    exit_code = main(
        [
            "--fixture",
            "track_b_synthetic",
            "--models",
            "baseline,hazard,gbm",
            "--output-dir",
            str(tmp_path),
            "--decision-path",
            str(gate_path),
        ]
    )

    assert exit_code == 0
    scoreboard_path = tmp_path / "forecast_scoreboard_v1.json"
    summary_path = tmp_path / "forecast_scoreboard_v1.md"

    scoreboard = json.loads(scoreboard_path.read_text())
    assert scoreboard["gate_6"]["status"] == "PASS"
    assert scoreboard["gate_6"]["baseline_brier_matches_expected"] is True
    assert summary_path.read_text().startswith("# Forecast Scoreboard v1")
    assert "Gate 6 status: PASS" in gate_path.read_text()
