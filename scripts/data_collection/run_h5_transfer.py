#!/usr/bin/env python3
"""Run H5 cross-structure transfer experiment.

Usage:
    uv run python scripts/data_collection/run_h5_transfer.py \
        --events outputs/track_b/events.ndjson \
        --output outputs/results/h5_transfer_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from engram.models.track_b import DelinquencyBucket, TrackBEvent
from engram.services.h5_transfer import run_h5_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H5 cross-structure transfer")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
                state=features.get("state"),
            ))
    logger.info("Loaded %d events", len(events))

    result = run_h5_experiment(events)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    logger.info("Families: %s", result["families"])
    logger.info("Family sizes: %s", result.get("family_sizes", {}))
    logger.info("Shared-core accuracy: %.4f", result["shared_core"]["accuracy"])
    logger.info("Family-specific accuracy: %.4f", result["family_specific"]["accuracy"])
    logger.info("Cross-family accuracy: %.4f", result.get("cross_family", {}).get("accuracy", 0.0))
    logger.info("Family drift: %.4f", result["family_drift"])
    logger.info("Transferable motif score: %.4f", result["transferable_motif_score"])
    logger.info("Lift vs shared-core: %.4f", result["lift_vs_shared_core"])
    logger.info("Structural drivers: %s", result["structural_drivers"])
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
