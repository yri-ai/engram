#!/usr/bin/env python3
"""Run H2 context ablation experiment.

Usage:
    uv run python scripts/data_collection/run_h2_context_ablation.py \
        --events outputs/track_b/events.ndjson \
        --output outputs/results/h2_context_ablation_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from engram.models.track_b import DelinquencyBucket, TrackBEvent
from engram.services.h2_experiments import run_h2_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H2 context ablation")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--eval-end", default="2025-12-31")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logger.info("Loading events from %s", args.events)
    events = []
    with open(args.events) as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            row = json.loads(line)
            features = row["features"]
            events.append(TrackBEvent(
                loan_id=row["loan_id"],
                as_of=date.fromisoformat(row["as_of"]),
                bucket=DelinquencyBucket(features["bucket"]),
                current_upb=features["current_upb"],
                interest_rate=features.get("interest_rate"),
                original_upb=features.get("original_upb"),
                credit_score=features.get("credit_score"),
                state=features.get("state"),
            ))
    logger.info("Loaded %d events", len(events))

    artifact = run_h2_experiment(events, train_end=args.train_end, eval_end=args.eval_end)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact.to_dict(), indent=2))

    for name, result in artifact.profiles.items():
        logger.info(
            "  %s: accuracy=%.4f brier=%.4f distractor_drop=%.4f",
            name, result.top1_accuracy_mean, result.brier_score_mean, result.distractor_drop,
        )
    logger.info("Evidence gap coverage: %.4f", artifact.evidence_gap_coverage)
    logger.info("Competing cause: %s", artifact.competing_cause_discrimination)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
