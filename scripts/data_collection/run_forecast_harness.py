"""Forecast harness runner for Phase 0 prediction-upgrade evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from engram.forecasting.canary import run_leakage_canary
from engram.forecasting.fixtures import load_forecast_fixture_rows
from engram.forecasting.metrics import (
    calibration_bins,
    expected_calibration_error,
    log_loss,
    loss_weighted_error,
    multiclass_brier_score,
    one_vs_rest_auc,
    top1_accuracy,
)
from engram.forecasting.protocol import BaselineForecasterAdapter, Forecaster
from engram.forecasting.splits import record_time_filter
from engram.models.track_b import DelinquencyBucket

_MISSING_EVENTS_MESSAGE = """Forecast harness input is missing.
Provide --fixture track_b_synthetic for the checked-in synthetic fixture, or
create the local Track B events file first (for example outputs/track_b/events.ndjson)
and pass it with --events PATH. The harness will not silently weaken Gate 6.
"""
_CLASSES = [bucket.value for bucket in DelinquencyBucket]
_SCOREBOARD_VERSION = 1
_DEFAULT_OUTPUT_DIR = Path("outputs/results")
_EXPECTED_BASELINE = Path("tests/fixtures/track_b/track_b_synthetic_baseline.json")
_GATE_6_DECISION = Path("docs/plans/decisions/track-b-gate-6-decision.md")
_CANARY_FEATURE = "harness_future_bucket"
_FUTURE_RECORDED_FROM = "9999-12-31"


def _load_events_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(2)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_input_rows(events: Path | None, fixture: str | None) -> list[dict[str, Any]]:
    """Resolve harness input rows from either a local events path or checked-in fixture."""
    if events is not None and fixture is not None:
        print("Pass either --events or --fixture, not both.", file=sys.stderr)
        raise SystemExit(2)
    if fixture is not None:
        return load_forecast_fixture_rows(fixture)
    if events is not None:
        if not events.exists():
            print(_MISSING_EVENTS_MESSAGE, file=sys.stderr)
            raise SystemExit(2)
        return _load_events_rows(events)

    default_events = Path("outputs/track_b/events.ndjson")
    if default_events.exists():
        return _load_events_rows(default_events)
    print(_MISSING_EVENTS_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


def build_forecaster(name: str) -> Forecaster:
    """Create a registered Phase 0 forecaster by CLI name."""
    if name == "baseline":
        return BaselineForecasterAdapter()
    if name == "hazard":
        from engram.forecasting.baselines import HazardForecaster

        return HazardForecaster()
    if name == "gbm":
        from engram.forecasting.baselines import GBMForecaster

        return GBMForecaster()
    if name == "tfm":
        from engram.forecasting.tfm import TFMForecaster

        return TFMForecaster()
    if name == "ultra":
        from engram.forecasting.graph_head import UltraForecaster

        return UltraForecaster()
    if name == "temporal_gnn":
        from engram.forecasting.graph_head import TemporalGNNForecaster

        return TemporalGNNForecaster()
    if name == "ensemble":
        from engram.forecasting.baselines import HazardForecaster
        from engram.forecasting.ensemble import LinearPoolEnsemble
        from engram.forecasting.tfm import TFMForecaster

        return LinearPoolEnsemble([TFMForecaster(), HazardForecaster()])
    raise ValueError(
        f"unknown forecast model {name!r}; expected baseline,hazard,gbm,tfm,ultra,temporal_gnn,ensemble"
    )


def run_scoreboard(
    rows: list[dict[str, Any]],
    *,
    model_names: list[str],
    output_dir: Path,
    fixture_name: str | None = None,
    decision_path: Path = _GATE_6_DECISION,
) -> dict[str, Any]:
    """Run registered models over the fixture eval split and persist scoreboard artifacts."""
    generated_at = datetime.now(UTC).isoformat()
    train_rows, eval_rows = _prepare_train_eval(rows)

    model_results = [_score_model(name, train_rows, eval_rows) for name in model_names]
    canary = run_leakage_canary(lambda: BaselineForecasterAdapter(), rows)
    canary.update(_run_harness_filter_canary(rows))
    gate_6 = _gate_6_status(model_results, fixture_name, canary)

    scoreboard: dict[str, Any] = {
        "schema_version": _SCOREBOARD_VERSION,
        "generated_at": generated_at,
        "fixture": fixture_name,
        "sample_count": len(eval_rows),
        "classes": _CLASSES,
        "models": model_results,
        "leakage_canary": canary,
        "gate_6": gate_6,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    scoreboard_path = output_dir / f"forecast_scoreboard_v{_SCOREBOARD_VERSION}.json"
    summary_path = output_dir / f"forecast_scoreboard_v{_SCOREBOARD_VERSION}.md"
    scoreboard_path.write_text(json.dumps(scoreboard, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(_render_summary(scoreboard))
    _write_gate_6_decision(scoreboard, scoreboard_path, summary_path, decision_path)
    return scoreboard


def _prepare_train_eval(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [record_time_filter(row, row["as_of"]) for row in rows if row.get("split") == "train"]
    eval_rows = [record_time_filter(row, row["as_of"]) for row in rows if row.get("split") == "eval"]
    if not train_rows or not eval_rows:
        raise ValueError("forecast harness requires non-empty train and eval rows")
    return train_rows, eval_rows


def _score_model(
    name: str, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    forecaster = build_forecaster(name)
    return _score_forecaster(name, forecaster, train_rows, eval_rows)


def _score_forecaster(
    name: str,
    forecaster: Forecaster,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    forecaster.fit(train_rows)
    predictions = [forecaster.predict_proba(row["features"]) for row in eval_rows]
    labels = [row["label"]["next_bucket"] for row in eval_rows]
    metrics = {
        "brier_score": multiclass_brier_score(predictions, labels, _CLASSES),
        "top1_accuracy": top1_accuracy(predictions, labels),
        "log_loss": log_loss(predictions, labels, _CLASSES),
        "ece": expected_calibration_error(predictions, labels, n_bins=10),
        "loss_weighted_error": loss_weighted_error(predictions, labels, _CLASSES),
    }
    return {
        "model": name,
        "forecaster_name": forecaster.name,
        "metrics": metrics,
        "windows": [
            {
                "window_id": "fixture_eval",
                "train_count": len(train_rows),
                "sample_count": len(eval_rows),
                "metrics": metrics,
                "calibration_bins": calibration_bins(predictions, labels, n_bins=10),
                "one_vs_rest_auc": one_vs_rest_auc(predictions, labels, _CLASSES),
            }
        ],
    }


def _run_harness_filter_canary(rows: list[dict[str, Any]]) -> dict[str, float | bool]:
    poisoned = _inject_harness_future_label_feature(rows)
    raw_train = [row for row in poisoned if row.get("split") == "train"]
    raw_eval = [row for row in poisoned if row.get("split") == "eval"]
    filtered_train, filtered_eval = _prepare_train_eval(poisoned)

    raw_score = _score_forecaster(
        "harness_filter_probe", _FutureFeatureProbeForecaster(), raw_train, raw_eval
    )["metrics"]["brier_score"]
    filtered_score = _score_forecaster(
        "harness_filter_probe", _FutureFeatureProbeForecaster(), filtered_train, filtered_eval
    )["metrics"]["brier_score"]
    delta = float(filtered_score) - float(raw_score)
    if delta <= 0.25:
        raise AssertionError(f"harness filter canary was not red-capable: delta={delta:.6f}")
    return {
        "harness_filter_path_detected": True,
        "harness_filter_delta": delta,
    }


def _inject_harness_future_label_feature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    poisoned = cast("list[dict[str, Any]]", json.loads(json.dumps(rows)))
    for row in poisoned:
        row.setdefault("features", {})[_CANARY_FEATURE] = row["label"]["next_bucket"]
        row.setdefault("feature_provenance", {})[_CANARY_FEATURE] = [
            {"source_id": "harness-filter-canary", "recorded_from": _FUTURE_RECORDED_FROM}
        ]
    return poisoned


class _FutureFeatureProbeForecaster:
    name = "harness_future_feature_probe"

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        del train_rows

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        leaked = features.get(_CANARY_FEATURE)
        if isinstance(leaked, str) and leaked in _CLASSES:
            return {bucket: 1.0 if bucket == leaked else 0.0 for bucket in _CLASSES}
        return {bucket: 1.0 / len(_CLASSES) for bucket in _CLASSES}


def _gate_6_status(
    model_results: list[dict[str, Any]], fixture_name: str | None, canary: dict[str, Any]
) -> dict[str, Any]:
    baseline_result = next((result for result in model_results if result["model"] == "baseline"), None)
    if baseline_result is None:
        return {
            "status": "FAIL",
            "missing_required_model": "baseline",
            "baseline_brier": None,
            "expected_baseline_brier": None,
            "baseline_brier_matches_expected": False,
            "leakage_canary_passed": False,
            "hazard_on_board": any(result["model"] == "hazard" for result in model_results),
            "gbm_on_board": any(result["model"] == "gbm" for result in model_results),
            "local_ginnie_events_present": Path("outputs/track_b/events.ndjson").exists(),
        }
    baseline = baseline_result
    baseline_brier = float(baseline["metrics"]["brier_score"])
    expected_brier: float | None = None
    matches_expected = fixture_name != "track_b_synthetic"
    if fixture_name == "track_b_synthetic":
        expected = json.loads(_EXPECTED_BASELINE.read_text())
        expected_brier = float(expected["brier_score"])
        matches_expected = abs(baseline_brier - expected_brier) <= 0.001
    hazard_present = any(result["model"] == "hazard" for result in model_results)
    gbm_present = any(result["model"] == "gbm" for result in model_results)
    canary_passed = canary.get("status") == "passed" and canary.get("harness_filter_path_detected") is True
    status = "PASS" if matches_expected and canary_passed and hazard_present and gbm_present else "FAIL"
    return {
        "status": status,
        "baseline_brier": baseline_brier,
        "expected_baseline_brier": expected_brier,
        "baseline_brier_matches_expected": matches_expected,
        "leakage_canary_passed": canary_passed,
        "hazard_on_board": hazard_present,
        "gbm_on_board": gbm_present,
        "local_ginnie_events_present": Path("outputs/track_b/events.ndjson").exists(),
    }


def _render_summary(scoreboard: dict[str, Any]) -> str:
    lines = [
        "# Forecast Scoreboard v1",
        "",
        f"Generated at: `{scoreboard['generated_at']}`",
        f"Fixture: `{scoreboard.get('fixture')}`",
        f"Sample count: `{scoreboard['sample_count']}`",
        f"Gate 6 status: {scoreboard['gate_6']['status']}",
        f"Leakage canary: {scoreboard['leakage_canary']['status']}",
        "",
        "| Model | Brier | Top-1 | Log loss | ECE | Loss-weighted error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in scoreboard["models"]:
        metrics = model["metrics"]
        lines.append(
            f"| {model['model']} | {metrics['brier_score']:.6f} | "
            f"{metrics['top1_accuracy']:.6f} | {metrics['log_loss']:.6f} | "
            f"{metrics['ece']:.6f} | {metrics['loss_weighted_error']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_gate_6_decision(
    scoreboard: dict[str, Any], scoreboard_path: Path, summary_path: Path, decision_path: Path
) -> None:
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    gate = scoreboard["gate_6"]
    baseline_brier = gate["baseline_brier"]
    baseline_brier_text = "None" if baseline_brier is None else f"{baseline_brier:.6f}"
    lines = [
        "# Track B Gate 6 Decision — Phase 0 Forecast Harness",
        "",
        f"Gate 6 status: {gate['status']}",
        "",
        "## Thresholds",
        "- Checked-in synthetic fixture baseline Brier reproduces expected artifact within ±0.001.",
        "- Leakage canary is green.",
        "- Baseline, hazard, and GBM models are present on the scoreboard.",
        "",
        "## Observed",
        f"- Baseline Brier: `{baseline_brier_text}`",
        f"- Expected baseline Brier: `{gate['expected_baseline_brier']}`",
        f"- Baseline Brier matches expected: `{gate['baseline_brier_matches_expected']}`",
        f"- Leakage canary: `{scoreboard['leakage_canary']['status']}`",
        f"- Harness filter canary path detected: `{scoreboard['leakage_canary']['harness_filter_path_detected']}`",
        f"- Gate 6 leakage canary passed: `{gate['leakage_canary_passed']}`",
        f"- Hazard on board: `{gate['hazard_on_board']}`",
        f"- GBM on board: `{gate['gbm_on_board']}`",
        f"- Local Ginnie events present: `{gate['local_ginnie_events_present']}`",
        "",
        "## Artifacts",
        f"- Scoreboard JSON: `{scoreboard_path}`",
        f"- Scoreboard summary: `{summary_path}`",
        "",
        "## Decision",
        "Proceed to Phase 1 only if this document says PASS. Do not add forecast heads without scoreboard entries.",
        "",
    ]
    decision_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 0 forecast harness.")
    parser.add_argument("--events", type=Path, default=None, help="Canonical Track B rows/events JSONL")
    parser.add_argument("--fixture", default=None, help="Checked-in fixture name, e.g. track_b_synthetic")
    parser.add_argument(
        "--models",
        default="baseline,hazard,gbm",
        help="Comma-separated registered models: baseline,hazard,gbm,tfm,ultra,temporal_gnn,ensemble",
    )
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decision-path", type=Path, default=_GATE_6_DECISION)
    args = parser.parse_args(argv)

    rows = resolve_input_rows(events=args.events, fixture=args.fixture)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    scoreboard = run_scoreboard(
        rows,
        model_names=model_names,
        output_dir=args.output_dir,
        fixture_name=args.fixture,
        decision_path=args.decision_path,
    )
    print(json.dumps({"gate_6": scoreboard["gate_6"], "input_rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
