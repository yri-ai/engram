#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from engram.services.research_experiments import run_thin_slice_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thin-slice branch forecasting experiment v0")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "fixtures" / "baseline.ndjson",
    )
    parser.add_argument(
        "--reduced",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "fixtures"
        / "reduced_context.ndjson",
    )
    parser.add_argument(
        "--distractor",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "fixtures" / "distractor.ndjson",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "results" / "thin_slice_v0.json",
    )
    args = parser.parse_args()

    branches = ["stability", "distress", "refi"]
    result = run_thin_slice_experiment(
        baseline_path=args.baseline,
        reduced_path=args.reduced,
        distractor_path=args.distractor,
        output_path=args.output,
        branches=branches,
    )

    logger.info("Wrote experiment results to %s", args.output)
    logger.info("Profile metrics: %s", result["profiles"])
    logger.info("Stability: %s", result["stability"])


if __name__ == "__main__":
    main()
