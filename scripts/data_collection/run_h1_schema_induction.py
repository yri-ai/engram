#!/usr/bin/env python3
"""Run H1 schema induction experiment.

Usage:
    uv run python scripts/data_collection/run_h1_schema_induction.py \
        --events outputs/track_b/events.ndjson \
        --output outputs/results/h1_schema_library_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from engram.models.track_b import DelinquencyBucket, TrackBEvent
from engram.services.h1_transfer import run_h1_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1 schema induction")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=3)
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

    result = run_h1_experiment(events, window=args.window, min_support=args.min_support)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    logger.info("Induced %d motifs (top 50 stored)", result["total_motifs_induced"])
    logger.info("Average schema size: %.1f nodes", result["average_schema_size"])
    logger.info("Transfer score: %.4f", result["transfer_score"])
    te = result["transfer_eval"]
    logger.info("  baseline accuracy: %.4f", te["in_family_baseline_accuracy"])
    logger.info("  schema-guided accuracy: %.4f", te["schema_guided_accuracy"])
    logger.info("  schema coverage: %.4f", te["schema_coverage"])
    gs = result["granularity_sweep"]
    logger.info("Selected granularity: %s", gs["selected_granularity"])
    for g in ["event_only", "event_plus_state", "event_state_gate"]:
        logger.info("  %s: accuracy=%.4f nodes=%.1f motifs=%d",
                     g, gs[g]["accuracy"], gs[g]["avg_nodes"], gs[g]["motif_count"])
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
