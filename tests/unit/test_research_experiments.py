from __future__ import annotations

import json
from typing import TYPE_CHECKING

from engram.services.research_experiments import build_calibration_report, run_thin_slice_experiment

if TYPE_CHECKING:
    from pathlib import Path


def _write_ndjson(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_run_thin_slice_experiment_outputs_metrics(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.ndjson"
    reduced = tmp_path / "reduced_context.ndjson"
    distractor = tmp_path / "distractor.ndjson"

    records = [
        {"record_id": "r1", "source": "edgar", "event_date": "2024-03-01", "metadata": {"a": 1}},
        {"record_id": "r2", "source": "fannie", "event_date": "2025-02-01", "metadata": {"b": 2}},
    ]
    _write_ndjson(baseline, [{**r, "profile": "baseline"} for r in records])
    _write_ndjson(reduced, [{**r, "profile": "reduced_context"} for r in records])
    _write_ndjson(distractor, [{**r, "profile": "distractor", "distractor": True} for r in records])

    out_path = tmp_path / "results" / "thin_slice_v0.json"
    result = run_thin_slice_experiment(
        baseline,
        reduced,
        distractor,
        out_path,
        branches=["stability", "distress", "refi"],
    )

    assert out_path.exists()
    assert set(result["profiles"].keys()) == {"baseline", "reduced_context", "distractor"}

    for profile in ("baseline", "reduced_context", "distractor"):
        metrics = result["profiles"][profile]
        assert 0.0 <= metrics["top1_accuracy"] <= 1.0
        assert 0.0 <= metrics["avg_confidence"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 2.0
        assert metrics["record_count"] == 2

    stability = result["stability"]
    assert 0.0 <= stability["baseline_vs_reduced_top1_agreement"] <= 1.0
    assert 0.0 <= stability["baseline_vs_distractor_top1_agreement"] <= 1.0


def test_run_thin_slice_experiment_writes_ranking_samples(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.ndjson"
    reduced = tmp_path / "reduced_context.ndjson"
    distractor = tmp_path / "distractor.ndjson"
    rec = {"record_id": "x1", "source": "ginnie", "event_date": "2026-01-01", "metadata": {"p": 1}}

    _write_ndjson(baseline, [{**rec, "profile": "baseline"}])
    _write_ndjson(reduced, [{**rec, "profile": "reduced_context"}])
    _write_ndjson(distractor, [{**rec, "profile": "distractor", "distractor": True}])

    out_path = tmp_path / "results" / "thin_slice_v0.json"
    result = run_thin_slice_experiment(
        baseline,
        reduced,
        distractor,
        out_path,
        branches=["stability", "distress", "refi"],
    )

    samples = result["samples"]
    assert len(samples) == 3
    assert {s["profile"] for s in samples} == {"baseline", "reduced_context", "distractor"}
    for sample in samples:
        assert sample["top_branch"] in {"stability", "distress", "refi"}
        assert set(sample["scores"].keys()) == {"stability", "distress", "refi"}


def test_build_calibration_report_from_experiment_samples(tmp_path: Path) -> None:
    result_path = tmp_path / "thin_slice_v0.json"
    result_payload = {
        "samples": [
            {
                "profile": "baseline",
                "truth_branch": "stability",
                "top_branch": "stability",
                "scores": {"stability": 0.90, "distress": 0.05, "refi": 0.05},
            },
            {
                "profile": "baseline",
                "truth_branch": "distress",
                "top_branch": "stability",
                "scores": {"stability": 0.80, "distress": 0.10, "refi": 0.10},
            },
            {
                "profile": "distractor",
                "truth_branch": "refi",
                "top_branch": "refi",
                "scores": {"stability": 0.20, "distress": 0.20, "refi": 0.60},
            },
        ]
    }
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")

    out_path = tmp_path / "calibration_report.json"
    report = build_calibration_report(result_path, out_path, bins=5)

    assert out_path.exists()
    assert report["total_samples"] == 3
    assert 0.0 <= report["overall_accuracy"] <= 1.0
    assert len(report["bins"]) == 5
    assert any(b["count"] > 0 for b in report["bins"])
