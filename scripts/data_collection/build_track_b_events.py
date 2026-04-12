#!/usr/bin/env python3
"""Build Track B canonical events from Ginnie Mae LoanPerf data.

Usage:
    uv run python scripts/data_collection/build_track_b_events.py \
        --input data/ginnie/202512/LoanPerf_202512.zip \
        --output outputs/track_b/events.ndjson \
        --limit 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path

from engram.services.track_b_ginnie_parser import parse_loanperf_records
from engram.services.track_b_dataset import build_labeled_rows, assign_splits, validate_no_leakage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track B events from Ginnie Mae LoanPerf")
    parser.add_argument("--input", type=Path, required=True, help="Path to LoanPerf zip file")
    parser.add_argument("--output", type=Path, required=True, help="Output NDJSON path")
    parser.add_argument("--limit", type=int, default=None, help="Max records to parse")
    parser.add_argument("--train-end", default="2025-12-31", help="Train split cutoff (ISO date)")
    parser.add_argument("--eval-end", default="2026-03-31", help="Eval split cutoff (ISO date)")
    args = parser.parse_args()

    logger.info("Parsing %s (limit=%s)", args.input, args.limit)
    with zipfile.ZipFile(args.input) as zf:
        txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
        if not txt_files:
            logger.error("No .txt files found in %s", args.input)
            return
        logger.info("Reading %s", txt_files[0])
        with zf.open(txt_files[0]) as f:
            events = list(parse_loanperf_records(f, limit=args.limit))

    logger.info("Parsed %d events", len(events))

    rows = build_labeled_rows(events)
    logger.info("Built %d labeled rows", len(rows))

    rows = assign_splits(rows, train_end=args.train_end, eval_end=args.eval_end)
    validate_no_leakage(rows)

    splits = {"train": 0, "eval": 0, "holdout": 0}
    for r in rows:
        splits[r["split"]] += 1
    logger.info("Splits: %s", splits)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
