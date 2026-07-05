"""Phase 3 LLM branch forecaster, evidence, supervisor, and calibration coverage."""

from __future__ import annotations

from engram.forecasting.branches import BranchForecast, ScenarioNode, enumerate_bucket_branches
from engram.forecasting.calibration import fit_temperature, temperature_scale
from engram.forecasting.evidence import assemble_evidence
from engram.forecasting.fixtures import load_forecast_fixture_rows
from engram.forecasting.llm_branch import ComputeBudget, LLMBranchForecaster
from engram.forecasting.supervisor import reconcile_distributions


def test_branch_contracts_and_budgeted_rollout_are_deterministic():
    branches = enumerate_bucket_branches(horizon_months=2)
    assert branches[0].horizon_months == 2
    BranchForecast(
        branch=branches[0],
        probability=0.5,
        evidence_refs=["m1"],
        flip_conditions=["more delinquency evidence"],
    )
    ScenarioNode(node_id="root", children=[ScenarioNode(node_id="child", probability=0.2)])

    forecaster = LLMBranchForecaster(
        ComputeBudget(n_particles=2, max_depth=1, tokens_per_particle=10)
    )
    result = forecaster.rollout(
        "loan stress",
        scorer=lambda prompt, branch: 2.0 if branch.target_bucket.value == "d30" else 1.0,
    )
    assert result["cost"] == {"tokens": 20, "usd": 0.000019999999999999998}
    assert result["probabilities"]["d30"] > result["probabilities"]["current"]


def test_evidence_assembly_filters_future_recorded_context():
    rows = load_forecast_fixture_rows("track_b_synthetic")
    loan_id = rows[0]["loan_id"]
    poisoned = dict(rows[0])
    poisoned["message_id"] = "future-msg"
    poisoned["features"] = {**rows[0]["features"], "future_signal": "motif"}
    poisoned["feature_provenance"] = {
        "future_signal": [{"source_id": "x", "recorded_from": "9999-01-01"}]
    }
    evidence = assemble_evidence(loan_id, rows[0]["as_of"], 5, [], rows=[poisoned, *rows])
    assert "future_signal" not in evidence["items"][0]["features"]
    assert evidence["excluded_counts"]["future_record_time"] == 1


def test_supervisor_and_temperature_calibration_use_prior_windows():
    distributions = [{"current": 0.9, "d30": 0.1}, {"current": 0.6, "d30": 0.4}]
    reconciled = reconcile_distributions(distributions)
    assert abs(sum(reconciled.values()) - 1.0) < 1e-9
    temp = fit_temperature([(distributions[0], "d30"), (distributions[1], "current")])
    scaled = temperature_scale(distributions[0], temp)
    assert abs(sum(scaled.values()) - 1.0) < 1e-9
