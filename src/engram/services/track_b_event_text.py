"""Render TrackBEvent into natural language text for LLM extraction pipeline."""

from __future__ import annotations

from typing import Any

from engram.models.track_b import DelinquencyBucket, TrackBEvent


_BUCKET_DESCRIPTIONS = {
    DelinquencyBucket.CURRENT: "current (performing)",
    DelinquencyBucket.D30: "30 days delinquent",
    DelinquencyBucket.D60: "60 days delinquent",
    DelinquencyBucket.D90: "90 days delinquent",
    DelinquencyBucket.D90_PLUS: "90+ days delinquent (seriously delinquent)",
    DelinquencyBucket.REO: "REO (real estate owned / post-foreclosure)",
}


def render_event_text(event: TrackBEvent) -> str:
    """Render a loan-month observation as natural language.

    This text is fed to the LLM extraction pipeline via POST /messages,
    so it should contain enough detail for the LLM to extract entities,
    relationships, and facts.
    """
    bucket_desc = _BUCKET_DESCRIPTIONS.get(event.bucket, event.bucket.value)
    parts = [
        f"Loan {event.loan_id} as of {event.as_of.isoformat()}:",
        f"delinquency status is {bucket_desc}",
        f"with unpaid principal balance of ${event.current_upb:,.2f}.",
    ]

    details = []
    if event.interest_rate is not None:
        details.append(f"interest rate {event.interest_rate}%")
    if event.original_upb is not None:
        details.append(f"original balance ${event.original_upb:,.0f}")
    if event.credit_score is not None:
        details.append(f"borrower credit score {event.credit_score}")
    if event.state is not None:
        details.append(f"property in {event.state}")

    if details:
        parts.append("Details: " + ", ".join(details) + ".")

    return " ".join(parts)


def build_ingest_payload(
    event: TrackBEvent,
    conversation_id: str = "track-b",
    group_id: str = "ginnie-loans",
    tenant_id: str = "default",
) -> dict[str, Any]:
    """Build a POST /messages payload from a TrackBEvent.

    Uses deterministic message_id for idempotent ingestion.
    """
    return {
        "text": render_event_text(event),
        "speaker": "dataset",
        "timestamp": f"{event.as_of.isoformat()}T00:00:00Z",
        "message_id": event.message_id,
        "conversation_id": conversation_id,
        "group_id": group_id,
        "tenant_id": tenant_id,
    }
