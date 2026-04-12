#!/usr/bin/env python3
"""Run H3 predictive primitive comparison.

Usage:
    uv run python scripts/data_collection/run_h3_comparison.py \
        --events outputs/track_b/events.ndjson \
        --output outputs/results/h3_primitive_comparison_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from engram.models.track_b import DelinquencyBucket, TrackBEvent
from engram.services.h3_experiments import run_h3_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H3 predictive primitive comparison")
    parser.add_argument("--events", type=Path, required=True, help="Path to events.ndjson")
    parser.add_argument("--output", type=Path, required=True, help="Output artifact JSON")
    parser.add_argument("--train-end", default="2025-06-30", help="Train split cutoff")
    parser.add_argument("--eval-end", default="2025-12-31", help="Eval split cutoff")
    parser.add_argument("--limit", type=int, default=None, help="Max event rows to load")
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

    artifact = run_h3_experiment(
        events,
        train_end=args.train_end,
        eval_end=args.eval_end,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact.to_dict(), indent=2))

    logger.info("Winner: %s", artifact.winner)
    for name, result in artifact.primitives.items():
        logger.info(
            "  %s: accuracy=%.4f±%.4f brier=%.4f±%.4f",
            name, result.top1_accuracy_mean, result.top1_accuracy_std,
            result.brier_score_mean, result.brier_score_std,
        )
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
