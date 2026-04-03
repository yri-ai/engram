#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from engram.services.research_pipeline import run_research_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full research data pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Data root directory (default: ./data)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Output root directory (default: ./data)",
    )
    parser.add_argument(
        "--fixture-per-split",
        type=int,
        default=250,
        help="Number of fixture records per split profile",
    )
    parser.add_argument(
        "--sweep-budgets",
        type=int,
        nargs="*",
        default=[100, 250, 500],
        help="Context budget values to sweep",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "results" / "pipeline_summary.json",
        help="Path to write pipeline summary JSON",
    )
    args = parser.parse_args()

    summary = run_research_pipeline(
        data_dir=args.data_dir,
        output_root=args.output_root,
        fixture_per_split=args.fixture_per_split,
        sweep_budgets=args.sweep_budgets,
    )

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote pipeline summary to %s", args.summary_output)
    logger.info("Split counts: %s", summary["split_counts"])


if __name__ == "__main__":
    main()
