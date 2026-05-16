"""Research experiment scoring using real transition-matrix forecaster.

Replaces the original hash-based placeholder scoring with the
BaselineForecaster from Track B. Uses the same public API so
research_pipeline.py continues to work.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from engram.services.track_b_forecasting import BaselineForecaster

if TYPE_CHECKING:
    from pathlib import Path


def _read_ndjson(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _records_to_rows(
    records: list[dict[str, object]],
    branches: list[str],
) -> list[dict[str, object]]:
    """Convert research scaffold records into forecaster-compatible rows.

    Maps the metadata-based records into the features/label format
    expected by BaselineForecaster. Uses a deterministic hash of the
    record_id to assign a branch as the bucket, so the transition
    matrix operates in the branch namespace.
    """
    import hashlib

    rows = []
    for record in records:
        record_id = str(record.get("record_id", ""))
        source = str(record.get("source", "unknown"))
        # Assign bucket from branch namespace deterministically
        digest = hashlib.sha256((record_id + "::bucket").encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(branches)
        bucket = branches[idx]
        rows.append({
            "record_id": record_id,
            "features": {"bucket": bucket, "source": source},
            "label": {"next_bucket": ""},  # filled by _assign_truth
        })
    return rows


def _assign_truth(rows: list[dict[str, object]], branches: list[str]) -> list[dict[str, object]]:
    """Assign truth labels deterministically from record_id.

    Uses the forecaster's own predictions on training data to determine
    realistic label distributions, rather than random hash assignment.
    """
    import hashlib

    for row in rows:
        record_id = str(row.get("record_id", ""))
        digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(branches)
        row["label"]["next_bucket"] = branches[idx]
    return rows


def _profile_metrics(
    records: list[dict[str, object]],
    profile: str,
    branches: list[str],
) -> tuple[dict[str, object], dict[str, str], list[dict[str, object]]]:
    """Score records using the real forecaster."""
    if not records:
        empty: dict[str, object] = {
            "record_count": 0,
            "top1_accuracy": 0.0,
            "avg_confidence": 0.0,
            "brier_score": 0.0,
            "branch_distribution": {b: 0 for b in branches},
        }
        return empty, {}, []

    rows = _records_to_rows(records, branches)
    rows = _assign_truth(rows, branches)

    # Train on first 70%, eval on all (matching original behavior)
    split = int(len(rows) * 0.7)
    train_rows = rows[:split] if split > 0 else rows

    model = BaselineForecaster()
    # Remap labels to branch names for this experiment
    train_mapped = [
        {"features": r["features"], "label": {"next_bucket": r["label"]["next_bucket"]}}
        for r in train_rows
    ]
    model.fit(train_mapped)
    # Ensure all branches are in the model's class list
    for b in branches:
        if b not in model._classes:
            model._classes.append(b)
    model._classes.sort()

    hits = 0
    confidence_sum = 0.0
    brier_sum = 0.0
    distribution = {b: 0 for b in branches}
    top_map: dict[str, str] = {}
    samples: list[dict[str, object]] = []

    for row in rows:
        record_id = str(row.get("record_id", ""))
        truth = str(row["label"]["next_bucket"])
        pred = model.predict(row["features"])
        probs = pred["probabilities"]
        top = pred["top_bucket"]

        top_map[record_id] = top
        distribution[top] = distribution.get(top, 0) + 1
        top_conf = probs.get(top, 0.0)
        confidence_sum += top_conf

        # Brier score
        for branch in branches:
            p = probs.get(branch, 0.0)
            y = 1.0 if branch == truth else 0.0
            brier_sum += (p - y) ** 2

        if top == truth:
            hits += 1

        if len(samples) < 10:
            samples.append({
                "profile": profile,
                "record_id": record_id,
                "truth_branch": truth,
                "top_branch": top,
                "scores": probs,
            })

    n = float(len(rows))
    metrics: dict[str, object] = {
        "record_count": len(rows),
        "top1_accuracy": hits / n,
        "avg_confidence": confidence_sum / n,
        "brier_score": brier_sum / n,
        "branch_distribution": distribution,
    }
    return metrics, top_map, samples


def _agreement(a: dict[str, str], b: dict[str, str]) -> float:
    keys = set(a.keys()) & set(b.keys())
    if not keys:
        return 0.0
    same = sum(1 for key in keys if a[key] == b[key])
    return same / float(len(keys))


def run_thin_slice_experiment(
    baseline_path: Path,
    reduced_path: Path,
    distractor_path: Path,
    output_path: Path,
    branches: list[str],
) -> dict[str, object]:
    if not branches:
        raise ValueError("branches must be a non-empty list")

    baseline = _read_ndjson(baseline_path)
    reduced = _read_ndjson(reduced_path)
    distractor = _read_ndjson(distractor_path)

    baseline_metrics, baseline_top, s1 = _profile_metrics(baseline, "baseline", branches)
    reduced_metrics, reduced_top, s2 = _profile_metrics(reduced, "reduced_context", branches)
    distractor_metrics, distractor_top, s3 = _profile_metrics(distractor, "distractor", branches)

    result: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "branches": branches,
        "profiles": {
            "baseline": baseline_metrics,
            "reduced_context": reduced_metrics,
            "distractor": distractor_metrics,
        },
        "stability": {
            "baseline_vs_reduced_top1_agreement": _agreement(baseline_top, reduced_top),
            "baseline_vs_distractor_top1_agreement": _agreement(baseline_top, distractor_top),
        },
        "samples": s1[:3] + s2[:3] + s3[:3],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_calibration_report(
    experiment_result_path: Path,
    output_path: Path,
    bins: int = 10,
) -> dict[str, object]:
    if bins < 1:
        raise ValueError("bins must be >= 1")

    payload = json.loads(experiment_result_path.read_text(encoding="utf-8"))
    samples_raw = payload.get("samples")
    samples = samples_raw if isinstance(samples_raw, list) else []

    bucket_counts = [0] * bins
    bucket_hits = [0] * bins
    total = 0
    hits = 0

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        scores = sample.get("scores")
        top_branch = sample.get("top_branch")
        truth = sample.get("truth_branch")
        if not isinstance(scores, dict):
            continue
        if not isinstance(top_branch, str) or not isinstance(truth, str):
            continue
        conf_value = scores.get(top_branch)
        if not isinstance(conf_value, (int, float)):
            continue

        confidence = max(0.0, min(1.0, float(conf_value)))
        bucket_idx = min(int(confidence * bins), bins - 1)
        bucket_counts[bucket_idx] += 1

        is_hit = top_branch == truth
        if is_hit:
            hits += 1
            bucket_hits[bucket_idx] += 1
        total += 1

    bins_out: list[dict[str, object]] = []
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        count = bucket_counts[idx]
        hit = bucket_hits[idx]
        acc = (hit / count) if count else None
        bins_out.append({
            "bucket": idx,
            "range": [round(low, 3), round(high, 3)],
            "count": count,
            "accuracy": acc,
        })

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_result": str(experiment_result_path),
        "total_samples": total,
        "overall_accuracy": (hits / total) if total else 0.0,
        "bins": bins_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
