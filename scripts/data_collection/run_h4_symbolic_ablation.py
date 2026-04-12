#!/usr/bin/env python3
"""Run H4 symbolic pruning ablation.

Usage:
    uv run python scripts/data_collection/run_h4_symbolic_ablation.py \
        --events outputs/track_b/events.ndjson \
        --output outputs/results/h4_symbolic_ablation_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from engram.models.track_b import DelinquencyBucket, TrackBEvent
from engram.services.h4_symbolic import run_h4_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H4 symbolic pruning ablation")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--eval-end", default="2025-12-31")
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
            ))
    logger.info("Loaded %d events", len(events))

    result = run_h4_experiment(events, train_end=args.train_end, eval_end=args.eval_end)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    wo = result["without_symbolic"]
    logger.info("Without symbolic: contradiction=%.4f recall=%.4f", wo["contradiction_rate"], wo["recall"])
    for level in ["loose", "medium", "hard"]:
        t = result["tightness"][level]
        logger.info("  %s: contradiction=%.4f recall=%.4f novelty_prune=%.4f",
                     level, t["contradiction_rate"], t["recall"], t["novelty_prune_rate"])
    logger.info("Selected: %s", result["tightness"]["selected_tightness"])
    logger.info("Contradiction reduction: %.4f", result["contradiction_reduction"])
    logger.info("Recall loss: %.4f", result["recall_loss"])
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
