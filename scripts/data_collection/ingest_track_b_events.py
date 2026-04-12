#!/usr/bin/env python3
"""Ingest Track B events into Engram via POST /messages.

Reads events.ndjson (from build_track_b_events.py), renders each as text,
and sends to the running Engram API with deterministic message_ids.

Usage:
    uv run python scripts/data_collection/ingest_track_b_events.py \
        --api-url http://localhost:8000 \
        --events outputs/track_b/events.ndjson \
        --limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import httpx

from engram.models.track_b import DelinquencyBucket, TrackBEvent
from engram.services.track_b_event_text import build_ingest_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _row_to_event(row: dict) -> TrackBEvent:
    features = row["features"]
    return TrackBEvent(
        loan_id=row["loan_id"],
        as_of=date.fromisoformat(row["as_of"]),
        bucket=DelinquencyBucket(features["bucket"]),
        current_upb=features["current_upb"],
        interest_rate=features.get("interest_rate"),
        original_upb=features.get("original_upb"),
        credit_score=features.get("credit_score"),
        state=features.get("state"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Track B events into Engram")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Engram API URL")
    parser.add_argument("--events", type=Path, required=True, help="Path to events.ndjson")
    parser.add_argument("--limit", type=int, default=None, help="Max events to ingest")
    parser.add_argument("--conversation-id", default="track-b", help="Conversation ID")
    parser.add_argument("--group-id", default="ginnie-loans", help="Group ID")
    args = parser.parse_args()

    rows = []
    with open(args.events) as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            rows.append(json.loads(line))

    logger.info("Loaded %d event rows from %s", len(rows), args.events)

    processed = 0
    skipped = 0
    errors = 0

    with httpx.Client(base_url=args.api_url, timeout=60.0) as client:
        for row in rows:
            event = _row_to_event(row)
            payload = build_ingest_payload(
                event,
                conversation_id=args.conversation_id,
                group_id=args.group_id,
            )
            try:
                resp = client.post("/messages", json=payload)
                resp.raise_for_status()
                result = resp.json()
                if result.get("entities_extracted", 0) > 0:
                    processed += 1
                else:
                    skipped += 1  # dedup
            except httpx.HTTPError as e:
                errors += 1
                if errors <= 3:
                    logger.warning("Error ingesting %s: %s", event.message_id, e)

    logger.info("Done: %d processed, %d skipped (dedup), %d errors", processed, skipped, errors)


if __name__ == "__main__":
    main()
