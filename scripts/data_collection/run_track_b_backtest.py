#!/usr/bin/env python3
"""Run Track B baseline backtest.

Reads events.ndjson (with splits and labels), trains a transition matrix
model on train split, and evaluates on eval split.

Usage:
    uv run python scripts/data_collection/run_track_b_backtest.py \
        --events outputs/track_b/events.ndjson \
        --output outputs/results/track_b_forecast_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from engram.services.track_b_forecasting import BaselineForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track B baseline backtest")
    parser.add_argument("--events", type=Path, required=True, help="Path to events.ndjson")
    parser.add_argument("--output", type=Path, required=True, help="Output forecast JSON")
    args = parser.parse_args()

    rows = []
    with open(args.events) as f:
        for line in f:
            rows.append(json.loads(line))

    train = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]
    holdout = [r for r in rows if r["split"] == "holdout"]

    logger.info("Splits: train=%d, eval=%d, holdout=%d", len(train), len(eval_rows), len(holdout))

    model = BaselineForecaster()
    model.fit(train)

    results = model.backtest(eval_rows)

    # Build sample predictions
    sample_predictions = []
    for row in eval_rows[:200]:
        pred = model.predict(row["features"])
        sample_predictions.append({
            "event_id": row["event_id"],
            "message_id": row["message_id"],
            "truth_bucket": row["label"]["next_bucket"],
            "predicted_bucket": pred["top_bucket"],
            "probabilities": pred["probabilities"],
        })

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "train_count": len(train),
            "eval_count": len(eval_rows),
            "holdout_count": len(holdout),
        },
        "sample_count": results["sample_count"],
        "top1_accuracy": results["top1_accuracy"],
        "brier_score": results["brier_score"],
        "sample_predictions": sample_predictions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    logger.info(
        "Results: accuracy=%.4f brier=%.4f samples=%d",
        results["top1_accuracy"],
        results["brier_score"],
        results["sample_count"],
    )
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
