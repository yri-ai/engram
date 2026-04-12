#!/usr/bin/env python3
"""Build Track B canonical events from Ginnie Mae data.

Supports two input modes:
  --input LoanPerf_XXXXXX.zip     (pipe-delimited, 2 months per loan)
  --payhist llpaymhist_XXXXXX.zip (48-month delinquency history per loan)
  --loanperf LoanPerf_XXXXXX.zip  (optional: join features from LoanPerf)

Usage:
    # From payment history (recommended — 48 months of transitions):
    uv run python scripts/data_collection/build_track_b_events.py \
        --payhist data/ginnie/202602/llpaymhist_202602.zip \
        --loanperf data/ginnie/202512/LoanPerf_202512.zip \
        --output outputs/track_b/events.ndjson \
        --limit 50000

    # From LoanPerf only (2 months, limited transitions):
    uv run python scripts/data_collection/build_track_b_events.py \
        --input data/ginnie/202512/LoanPerf_202512.zip \
        --output outputs/track_b/events.ndjson \
        --limit 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path

from engram.models.track_b import TrackBEvent
from engram.services.track_b_ginnie_parser import parse_loanperf_records
from engram.services.track_b_payhist_parser import parse_payhist_records, expand_history_to_events
from engram.services.track_b_dataset import build_labeled_rows, assign_splits, validate_no_leakage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_loanperf_features(path: Path, limit: int | None) -> dict[str, TrackBEvent]:
    """Load LoanPerf records into a loan_id → event lookup for feature enrichment."""
    with zipfile.ZipFile(path) as zf:
        txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
        if not txt_files:
            return {}
        with zf.open(txt_files[0]) as f:
            events = list(parse_loanperf_records(f, limit=limit))
    # Keep the most recent record per loan_id
    by_loan: dict[str, TrackBEvent] = {}
    for e in events:
        if e.loan_id not in by_loan or e.as_of > by_loan[e.loan_id].as_of:
            by_loan[e.loan_id] = e
    logger.info("Loaded %d LoanPerf features (%d unique loans)", len(events), len(by_loan))
    return by_loan


def _enrich_event(event: TrackBEvent, features: dict[str, TrackBEvent]) -> TrackBEvent:
    """Copy static features (rate, UPB, credit score, state) from LoanPerf onto a payhist event."""
    ref = features.get(event.loan_id)
    if ref is None:
        return event
    return TrackBEvent(
        loan_id=event.loan_id,
        as_of=event.as_of,
        bucket=event.bucket,
        current_upb=ref.current_upb if event.current_upb == 0.0 else event.current_upb,
        interest_rate=ref.interest_rate,
        original_upb=ref.original_upb,
        credit_score=ref.credit_score,
        state=ref.state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track B events from Ginnie Mae data")
    parser.add_argument("--input", type=Path, default=None, help="LoanPerf zip (2-month mode)")
    parser.add_argument("--payhist", type=Path, default=None, help="Payment history zip (48-month mode)")
    parser.add_argument("--loanperf", type=Path, default=None, help="LoanPerf zip for feature enrichment")
    parser.add_argument("--output", type=Path, required=True, help="Output NDJSON path")
    parser.add_argument("--limit", type=int, default=None, help="Max loan records to parse")
    parser.add_argument("--train-end", default="2025-06-30", help="Train split cutoff (ISO date)")
    parser.add_argument("--eval-end", default="2025-12-31", help="Eval split cutoff (ISO date)")
    args = parser.parse_args()

    if not args.input and not args.payhist:
        parser.error("Must provide either --input (LoanPerf) or --payhist (payment history)")

    events: list[TrackBEvent] = []

    if args.payhist:
        logger.info("Parsing payment history: %s (limit=%s)", args.payhist, args.limit)
        with zipfile.ZipFile(args.payhist) as zf:
            txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
            with zf.open(txt_files[0]) as f:
                records = list(parse_payhist_records(f, limit=args.limit))
        logger.info("Parsed %d loan histories", len(records))

        for rec in records:
            events.extend(expand_history_to_events(rec))
        logger.info("Expanded to %d monthly events", len(events))

        # Enrich with LoanPerf features if available
        if args.loanperf:
            features = _load_loanperf_features(args.loanperf, limit=None)
            events = [_enrich_event(e, features) for e in events]
            enriched = sum(1 for e in events if e.interest_rate is not None)
            logger.info("Enriched %d/%d events with LoanPerf features", enriched, len(events))

    elif args.input:
        logger.info("Parsing LoanPerf: %s (limit=%s)", args.input, args.limit)
        with zipfile.ZipFile(args.input) as zf:
            txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
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
