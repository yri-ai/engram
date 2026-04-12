"""Tests for Track B event text renderer and ingestion payload builder."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.track_b_event_text import render_event_text, build_ingest_payload


def _sample_event() -> TrackBEvent:
    return TrackBEvent(
        loan_id="1023374917",
        as_of=date(2025, 10, 1),
        bucket=DelinquencyBucket.CURRENT,
        current_upb=301.28,
        interest_rate=7.875,
        original_upb=73000.0,
        credit_score=730,
        state="NC",
    )


def test_render_event_text_includes_key_fields():
    event = _sample_event()
    text = render_event_text(event)
    assert "1023374917" in text
    assert "current" in text.lower()
    assert "301.28" in text or "301" in text
    assert "NC" in text


def test_render_event_text_handles_delinquency():
    event = TrackBEvent(
        loan_id="LN999",
        as_of=date(2026, 1, 1),
        bucket=DelinquencyBucket.D60,
        current_upb=150000.0,
    )
    text = render_event_text(event)
    assert "60" in text.lower() or "d60" in text.lower()
    assert "LN999" in text


def test_render_event_text_handles_missing_optional_fields():
    event = TrackBEvent(
        loan_id="LN001",
        as_of=date(2026, 2, 1),
        bucket=DelinquencyBucket.REO,
        current_upb=0.0,
    )
    text = render_event_text(event)
    assert "LN001" in text
    assert "reo" in text.lower()


def test_build_ingest_payload_shape():
    event = _sample_event()
    payload = build_ingest_payload(
        event,
        conversation_id="track-b",
        group_id="ginnie-loans",
        tenant_id="default",
    )
    assert payload["text"] == render_event_text(event)
    assert payload["speaker"] == "dataset"
    assert payload["message_id"] == "track-b-1023374917-202510"
    assert payload["conversation_id"] == "track-b"
    assert payload["group_id"] == "ginnie-loans"
    assert payload["tenant_id"] == "default"
    assert "timestamp" in payload


def test_build_ingest_payload_deterministic():
    event = _sample_event()
    p1 = build_ingest_payload(event)
    p2 = build_ingest_payload(event)
    assert p1["message_id"] == p2["message_id"]
    assert p1["text"] == p2["text"]
