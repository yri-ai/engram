from __future__ import annotations

import json
from typing import TYPE_CHECKING

from engram.services.research_data import (
    build_normalized_scaffold,
    build_research_fixtures,
    build_snapshot_manifest,
    build_time_split_manifest,
)
from engram.services.research_experiments import (
    build_calibration_report,
    run_thin_slice_experiment,
)

if TYPE_CHECKING:
    from pathlib import Path


def run_research_pipeline(
    data_dir: Path,
    output_root: Path,
    fixture_per_split: int = 250,
    sweep_budgets: list[int] | None = None,
) -> dict[str, object]:
    budgets = sweep_budgets if sweep_budgets is not None else [100, 250, 500]

    manifests_dir = output_root / "manifests"
    normalized_dir = output_root / "normalized"
    fixtures_dir = output_root / "fixtures"
    results_dir = output_root / "results"

    snapshot_manifest_path = manifests_dir / "research_snapshot.json"
    normalized_path = normalized_dir / "research_scaffold.ndjson"
    split_manifest_path = manifests_dir / "research_splits.json"

    snapshot_manifest = build_snapshot_manifest(data_dir, snapshot_manifest_path)
    build_normalized_scaffold(data_dir, normalized_path)
    split_manifest = build_time_split_manifest(normalized_path, split_manifest_path)

    fixture_paths = build_research_fixtures(
        normalized_path,
        split_manifest_path,
        fixtures_dir,
        per_split=fixture_per_split,
    )

    thin_slice_path = results_dir / "thin_slice_v0.json"
    thin_slice = run_thin_slice_experiment(
        baseline_path=fixture_paths["baseline"],
        reduced_path=fixture_paths["reduced_context"],
        distractor_path=fixture_paths["distractor"],
        output_path=thin_slice_path,
        branches=["stability", "distress", "refi"],
    )

    calibration_path = results_dir / "thin_slice_v0_calibration.json"
    build_calibration_report(thin_slice_path, calibration_path, bins=10)

    sweep_rows: list[dict[str, object]] = []
    for budget in budgets:
        budget_dir = fixtures_dir / f"budget_{budget}"
        budget_fixture_paths = build_research_fixtures(
            normalized_path,
            split_manifest_path,
            budget_dir,
            per_split=budget,
        )
        run_out = results_dir / f"thin_slice_v0_budget_{budget}.json"
        budget_result = run_thin_slice_experiment(
            baseline_path=budget_fixture_paths["baseline"],
            reduced_path=budget_fixture_paths["reduced_context"],
            distractor_path=budget_fixture_paths["distractor"],
            output_path=run_out,
            branches=["stability", "distress", "refi"],
        )
        sweep_rows.append(
            {
                "budget_per_split": budget,
                "profile_metrics": budget_result["profiles"],
                "stability": budget_result["stability"],
            }
        )

    sweep_path = results_dir / "context_budget_sweep.json"
    sweep_payload = {
        "generated_from": str(normalized_path),
        "budgets": sweep_rows,
    }
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")

    summary: dict[str, object] = {
        "snapshot_totals": snapshot_manifest["totals"],
        "split_counts": split_manifest["counts"],
        "thin_slice_profiles": thin_slice["profiles"],
        "artifacts": {
            "snapshot_manifest": str(snapshot_manifest_path),
            "normalized_scaffold": str(normalized_path),
            "split_manifest": str(split_manifest_path),
            "baseline_fixture": str(fixture_paths["baseline"]),
            "reduced_fixture": str(fixture_paths["reduced_context"]),
            "distractor_fixture": str(fixture_paths["distractor"]),
            "thin_slice_result": str(thin_slice_path),
            "calibration_report": str(calibration_path),
            "sweep_report": str(sweep_path),
        },
    }
    return summary
