"""Tests for the Engram CLI commands."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from engram.cli import main as cli


def test_http_client_ingest_messages_sets_defaults() -> None:
    """EngramHTTPClient.ingest_messages should fill defaults before POSTing."""

    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        return httpx.Response(
            200,
            json={
                "message_id": payload["message_id"],
                "entities_extracted": 2,
                "relationships_inferred": 1,
                "conflicts_resolved": 0,
                "processing_time_ms": 123.4,
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="http://testserver")
    client = cli.EngramHTTPClient(api_url="http://testserver", client=http_client)

    messages = [
        {
            "text": "Kendra switched to Adidas",
            "speaker": "Coach",
        }
    ]

    results = client.ingest_messages(
        messages=messages,
        conversation_id="conv-1",
        tenant_id="tenant-1",
        group_id=None,
    )

    assert len(results) == 1
    assert captured[0]["conversation_id"] == "conv-1"
    # group_id should fall back to conversation when not provided
    assert captured[0]["group_id"] == "conv-1"

    client.close()
    http_client.close()


class _StubClient:
    def __init__(self) -> None:
        self.ingest_calls: list[tuple] = []
        self.search_calls: list[tuple] = []
        self.closed = False

    def ingest_messages(self, **kwargs):  # type: ignore[no-untyped-def]
        self.ingest_calls.append(kwargs)
        return [
            {
                "message_id": "m-1",
                "entities_extracted": 2,
                "relationships_inferred": 1,
                "conflicts_resolved": 0,
                "processing_time_ms": 123,
            }
        ]

    def search_entities(self, query: str, tenant_id: str):  # type: ignore[override]
        self.search_calls.append((query, tenant_id))
        return [
            {"id": "entity-1", "conversation_id": "default"},
        ]

    def get_entity(self, entity_id: str):  # type: ignore[override]
        assert entity_id == "entity-1"
        return {"id": entity_id, "canonical_name": "Kendra"}

    def get_active_relationships(self, **_kwargs):  # type: ignore[no-untyped-def]
        return [
            cli.RelationshipRow(
                target_id="Adidas",
                rel_type="prefers",
                confidence=0.9,
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                evidence="Switched to Adidas for arch support",
            )
        ]

    def point_in_time(self, **_kwargs):  # type: ignore[no-untyped-def]
        return []

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_ingest_command(monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path) -> None:
    stub = _StubClient()
    monkeypatch.setattr(cli, "_build_client", lambda *_, **__: stub)

    payload = {"messages": [{"text": "Hello", "speaker": "Coach"}]}
    file_path = tmp_path / "messages.json"
    file_path.write_text(json.dumps(payload))

    result = runner.invoke(cli.app, ["ingest", str(file_path)])

    assert result.exit_code == 0
    assert stub.ingest_calls, "ingest_messages should be called"
    assert stub.closed is True


def test_cli_query_command(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    stub = _StubClient()
    monkeypatch.setattr(cli, "_build_client", lambda *_, **__: stub)

    result = runner.invoke(cli.app, ["query", "Kendra"])

    assert result.exit_code == 0
    assert "Active relationships" in result.stdout
    assert stub.search_calls == [("Kendra", "default")]
    assert stub.closed is True
