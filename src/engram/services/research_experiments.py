from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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


def _hash_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _source_bias(source: str, branch: str) -> float:
    table: dict[str, dict[str, float]] = {
        "edgar": {"stability": 0.36, "distress": 0.34, "refi": 0.30},
        "fannie": {"stability": 0.42, "distress": 0.23, "refi": 0.35},
        "ginnie": {"stability": 0.28, "distress": 0.30, "refi": 0.42},
    }
    return table.get(source, {}).get(branch, 0.33)


def _profile_factor(profile: str) -> float:
    factors = {"baseline": 1.0, "reduced_context": 0.96, "distractor": 0.90}
    return factors.get(profile, 1.0)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        n = float(len(scores))
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


def _truth_branch(record_id: str, branches: list[str]) -> str:
    idx = int(_hash_fraction(record_id + "::truth") * len(branches))
    return branches[min(idx, len(branches) - 1)]


def _score_record(record: dict[str, object], branches: list[str], profile: str) -> dict[str, float]:
    record_id = str(record.get("record_id", ""))
    source = str(record.get("source", "unknown"))
    factor = _profile_factor(profile)
    raw: dict[str, float] = {}
    for branch in branches:
        base = _hash_fraction(record_id + "::" + branch)
        bias = _source_bias(source, branch)
        raw[branch] = (0.65 * base + 0.35 * bias) * factor

    if profile == "distractor":
        for branch in branches:
            raw[branch] += 0.02 * _hash_fraction(record_id + "::noise::" + branch)

    return _normalize(raw)


def _top_branch(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _brier(scores: dict[str, float], truth: str, branches: list[str]) -> float:
    total = 0.0
    for branch in branches:
        y = 1.0 if branch == truth else 0.0
        total += (scores[branch] - y) ** 2
    return total


def _profile_metrics(
    records: list[dict[str, object]],
    profile: str,
    branches: list[str],
) -> tuple[dict[str, object], dict[str, str], list[dict[str, object]]]:
    if not records:
        empty = {
            "record_count": 0,
            "top1_accuracy": 0.0,
            "avg_confidence": 0.0,
            "brier_score": 0.0,
            "branch_distribution": {b: 0 for b in branches},
        }
        return empty, {}, []

    hits = 0
    confidence_sum = 0.0
    brier_sum = 0.0
    distribution = {b: 0 for b in branches}
    top_map: dict[str, str] = {}
    samples: list[dict[str, object]] = []

    for record in records:
        record_id = str(record.get("record_id", ""))
        scores = _score_record(record, branches, profile)
        top = _top_branch(scores)
        truth = _truth_branch(record_id, branches)

        top_map[record_id] = top
        distribution[top] += 1
        confidence_sum += scores[top]
        brier_sum += _brier(scores, truth, branches)
        if top == truth:
            hits += 1

        if len(samples) < 10:
            samples.append(
                {
                    "profile": profile,
                    "record_id": record_id,
                    "truth_branch": truth,
                    "top_branch": top,
                    "scores": scores,
                }
            )

    n = float(len(records))
    metrics = {
        "record_count": len(records),
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
        bins_out.append(
            {
                "bucket": idx,
                "range": [round(low, 3), round(high, 3)],
                "count": count,
                "accuracy": acc,
            }
        )

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
