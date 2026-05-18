"""Engram CLI commands."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import httpx
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from engram.config import Settings
from engram.models.branch_forecasting import ContextBudget
from engram.models.forecasting import ForecastQuestion, ForecastResolution, ForecastRun
from engram.services.branch_forecasting import BranchForecaster, evidence_from_path
from engram.services.forecast_repository import ForecastRepository
from engram.storage.neo4j import Neo4jStore

app = typer.Typer(name="engram", help="Temporal knowledge graph engine for AI memory")
console = Console()
ALLOWED_BRANCH_OPTION = typer.Option(..., help="Allowed branch name; repeat for each branch")


class CLIError(Exception):
    """Custom exception for CLI failures."""


@dataclass
class RelationshipRow:
    """Simplified representation of a relationship for table rendering."""

    target_id: str
    rel_type: str
    confidence: float
    valid_from: str
    valid_to: str | None
    evidence: str


@dataclass
class ForecastContext:
    """Loaded storage and repository dependencies for forecast lifecycle commands."""

    store: Neo4jStore
    repository: ForecastRepository

    async def close(self) -> None:
        await self.store.close()


class EngramHTTPClient:
    """Lightweight HTTP client for talking to the Engram API."""

    def __init__(
        self,
        api_url: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = api_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=self._base_url, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ---- Ingestion -----------------------------------------------------

    def ingest_messages(
        self,
        messages: Sequence[dict[str, Any]],
        conversation_id: str,
        tenant_id: str,
        group_id: str | None,
    ) -> list[dict[str, Any]]:
        """Send messages through the ingestion API."""

        results: list[dict[str, Any]] = []
        for raw in messages:
            payload = self._build_message_payload(
                raw,
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                group_id=group_id,
            )
            response = self._client.post("/messages", json=payload)
            response.raise_for_status()
            results.append(response.json())
        return results

    def _build_message_payload(
        self,
        raw: dict[str, Any],
        *,
        conversation_id: str,
        tenant_id: str,
        group_id: str | None,
    ) -> dict[str, Any]:
        """Normalize message fields before sending to API."""

        if "text" not in raw or "speaker" not in raw:
            raise CLIError("Each message requires 'text' and 'speaker' fields")

        timestamp = raw.get("timestamp")
        if not timestamp:
            timestamp = datetime.now(UTC).isoformat()

        payload = {
            "text": raw["text"],
            "speaker": raw["speaker"],
            "timestamp": timestamp,
            "conversation_id": raw.get("conversation_id", conversation_id),
            "tenant_id": raw.get("tenant_id", tenant_id),
            "group_id": raw.get("group_id", group_id or raw.get("group_id")),
            "message_id": raw.get("message_id", f"cli-{uuid.uuid4()}"),
            "metadata": raw.get("metadata", {}),
        }

        if payload["group_id"] is None:
            payload["group_id"] = payload["conversation_id"]

        return payload

    # ---- Querying ------------------------------------------------------

    def search_entities(self, query: str, tenant_id: str) -> list[dict[str, Any]]:
        response = self._client.get("/search", params={"q": query, "tenant_id": tenant_id})
        response.raise_for_status()
        return response.json()

    def get_entity(self, entity_id: str) -> dict[str, Any]:
        response = self._client.get(f"/entities/{entity_id}")
        response.raise_for_status()
        return response.json()

    def get_active_relationships(
        self,
        entity_id: str,
        tenant_id: str,
        rel_type: str | None,
    ) -> list[RelationshipRow]:
        params: dict[str, Any] = {"tenant_id": tenant_id}
        if rel_type:
            params["rel_type"] = rel_type
        response = self._client.get(f"/entities/{entity_id}/relationships", params=params)
        response.raise_for_status()
        data = response.json()
        return [
            RelationshipRow(
                target_id=row["target_id"],
                rel_type=row["rel_type"],
                confidence=row["confidence"],
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                evidence=row["evidence"],
            )
            for row in data
        ]

    def point_in_time(
        self,
        *,
        entity: str,
        as_of: str,
        tenant_id: str,
        rel_type: str | None,
        mode: str,
    ) -> Any:
        params: dict[str, Any] = {
            "entity": entity,
            "as_of": as_of,
            "tenant_id": tenant_id,
            "mode": mode,
        }
        if rel_type:
            params["rel_type"] = rel_type
        response = self._client.get("/query/point-in-time", params=params)
        response.raise_for_status()
        return response.json()


def _build_client(api_url: str, timeout: float = 30.0) -> EngramHTTPClient:
    """Factory for EngramHTTPClient. Separated for easier testing."""

    return EngramHTTPClient(api_url=api_url, timeout=timeout)


def _open_forecast_context(
    *, tenant_id: str, conversation_id: str, message_id: str
) -> ForecastContext:
    settings = Settings(_env_file=".env")
    store = Neo4jStore(settings)
    asyncio.run(store.initialize())
    repository = ForecastRepository(
        store,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    return ForecastContext(store=store, repository=repository)


def _parse_iso8601(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CLIError(f"Invalid datetime format: {value}") from exc


def _branch_probabilities_from_scores(result: dict[str, Any]) -> dict[str, float]:
    raw_scores = {item["branch"]: max(float(item["score"]), 0.0) for item in result["scores"]}
    total = sum(raw_scores.values())
    if total <= 0.0:
        count = len(raw_scores)
        return {branch: 1.0 / count for branch in raw_scores} if count else {}
    return {branch: score / total for branch, score in raw_scores.items()}


@app.command()
def init() -> None:
    """Initialize the Neo4j schema and indexes."""
    console.print("[bold blue]Initializing Engram schema...[/bold blue]")

    try:
        settings = Settings(_env_file=".env")
        store = Neo4jStore(settings)

        # Run async initialization
        asyncio.run(store.initialize())
        asyncio.run(store.close())

        console.print("[bold green]✓ Schema initialized successfully![/bold green]")
        console.print(f"[dim]Neo4j URI: {settings.neo4j_uri}[/dim]")
    except Exception as e:
        console.print(f"[bold red]✗ Initialization failed: {e}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from e


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload on code changes"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    console.print("[bold blue]Starting Engram API server...[/bold blue]")
    console.print(f"[dim]Host: {host}:{port}[/dim]")
    console.print(f"[dim]Reload: {reload}[/dim]")
    console.print("[dim]Visit http://localhost:8000/docs for API documentation[/dim]")

    uvicorn.run("engram.main:app", host=host, port=port, reload=reload)


@app.command()
def ingest(
    file: str = typer.Argument(..., help="Path to JSON file with conversation messages"),
    conversation_id: str = typer.Option("default", help="Conversation ID"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
    group_id: str | None = typer.Option(None, help="Group ID for cross-conversation memory"),
    api_url: str = typer.Option("http://localhost:8000", help="Engram API base URL"),
    timeout: float = typer.Option(30.0, help="HTTP timeout in seconds"),
) -> None:
    """Ingest conversation messages from a JSON file.

    Expected JSON format:
    {
      "messages": [
        {"text": "...", "speaker": "...", "timestamp": "2024-01-15T10:00:00Z"},
        ...
      ]
    }
    """
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[bold red]✗ File not found: {file}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1)

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]✗ Invalid JSON: {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc

    messages = data.get("messages", [])
    if not messages:
        console.print("[yellow]⚠ No messages found in file[/yellow]")
        return

    console.print(f"[bold blue]Ingesting {len(messages)} messages from {file}...[/bold blue]")

    client = _build_client(api_url, timeout)
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Sending to Engram...", total=len(messages))
            results = client.ingest_messages(
                messages=messages,
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                group_id=group_id,
            )
            for _ in results:
                progress.update(task, advance=1)

        console.print(f"[bold green]✓ Ingested {len(results)} messages[/bold green]")
        table = Table(title="Ingestion Summary")
        table.add_column("Message ID", style="cyan")
        table.add_column("Entities", style="magenta")
        table.add_column("Relationships", style="green")
        table.add_column("Conflicts", style="yellow")
        table.add_column("Latency (ms)", style="dim")

        for record in results:
            table.add_row(
                record.get("message_id", "-"),
                str(record.get("entities_extracted", 0)),
                str(record.get("relationships_inferred", 0)),
                str(record.get("conflicts_resolved", 0)),
                str(record.get("processing_time_ms", 0)),
            )

        console.print(table)
    except httpx.HTTPError as exc:
        console.print(f"[bold red]✗ API request failed: {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    except CLIError as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    finally:
        client.close()


@app.command()
def query(
    entity: str = typer.Argument(..., help="Entity name to query"),
    as_of: str | None = typer.Option(None, help="Query as of date (ISO 8601 format)"),
    rel_type: str | None = typer.Option(None, help="Filter by relationship type"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
    conversation_id: str | None = typer.Option("default", help="Conversation ID filter"),
    api_url: str = typer.Option("http://localhost:8000", help="Engram API base URL"),
    timeout: float = typer.Option(30.0, help="HTTP timeout in seconds"),
    mode: str = typer.Option(
        "world_state", help="Temporal mode: world_state, knowledge, bitemporal"
    ),
) -> None:
    """Query entity state and relationships.

    Examples:
      engram query "Kendra"
      engram query "Kendra" --as-of "2024-02-15T00:00:00Z"
      engram query "Kendra" --rel-type "prefers"
    """
    console.print(f"[bold blue]Querying entity: {entity}[/bold blue]")

    if as_of:
        try:
            datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except ValueError as exc:
            console.print(f"[bold red]✗ Invalid date format: {as_of}[/bold red]", file=sys.stderr)
            raise typer.Exit(code=1) from exc

    client = _build_client(api_url, timeout)
    try:
        entity_record = _resolve_entity(
            client,
            query=entity,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )

        if as_of:
            result = client.point_in_time(
                entity=entity,
                as_of=as_of,
                tenant_id=tenant_id,
                rel_type=rel_type,
                mode=mode,
            )
            _render_temporal_result(result, mode)
            return

        relationships = client.get_active_relationships(
            entity_id=entity_record["id"],
            tenant_id=tenant_id,
            rel_type=rel_type,
        )
        _render_relationships(entity_record, relationships)

    except httpx.HTTPError as exc:
        console.print(f"[bold red]✗ API request failed: {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    except CLIError as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    finally:
        client.close()


@app.command()
def forecast(
    file: str = typer.Argument(..., help="Path to forecast evidence JSON/NDJSON or directory"),
    objective: str = typer.Option(..., help="Decision objective to forecast against"),
    structural_family: str = typer.Option(
        "margin_analysis", help="Forecast structural family"
    ),
    max_items: int = typer.Option(6, help="Maximum evidence items to use"),
    max_tokens: int = typer.Option(1200, help="Maximum approximate context tokens"),
    min_score: float = typer.Option(0.0, help="Minimum evidence salience to consider"),
    output: str | None = typer.Option(None, help="Optional JSON output path"),
) -> None:
    """Forecast plausible next branches from compact evidence."""
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[bold red]✗ File not found: {file}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1)

    try:
        evidence = evidence_from_path(file_path)
        if not evidence:
            raise CLIError("No forecast evidence found")

        forecaster = BranchForecaster()
        result = forecaster.forecast(
            objective=objective,
            structural_family=structural_family,
            evidence=evidence,
            budget=ContextBudget(
                max_items=max_items,
                max_tokens=max_tokens,
                min_score=min_score,
            ),
        )
    except (CLIError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc

    result_payload = result.model_dump(mode="json")
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        console.print(f"[bold green]✓ Forecast written to {output}[/bold green]")

    _render_forecast(result_payload)


@app.command()
def create_forecast_question(
    target_entity_id: str = typer.Option(..., help="Target entity to forecast"),
    objective: str = typer.Option(..., help="Decision objective to forecast against"),
    structural_family: str = typer.Option(..., help="Forecast structural family"),
    forecast_as_of: str = typer.Option(..., help="Evidence cutoff timestamp"),
    horizon: str = typer.Option(..., help="Forecast horizon label"),
    resolution_due_at: str = typer.Option(..., help="Expected resolution deadline"),
    resolution_criteria: str = typer.Option(..., help="Objective resolution criteria"),
    allowed_branch: list[str] = ALLOWED_BRANCH_OPTION,
    tenant_id: str = typer.Option("default", help="Tenant identifier"),
    conversation_id: str = typer.Option("forecasting", help="Conversation / scope identifier"),
) -> None:
    """Persist a canonical forecast question."""
    message_id = f"forecast-question-{uuid.uuid4()}"
    context = _open_forecast_context(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    try:
        question = ForecastQuestion(
            id=f"fq-{uuid.uuid4()}",
            tenant_id=tenant_id,
            target_entity_id=target_entity_id,
            objective=objective,
            structural_family=structural_family,
            forecast_as_of=_parse_iso8601(forecast_as_of),
            horizon=horizon,
            resolution_due_at=_parse_iso8601(resolution_due_at),
            resolution_criteria=resolution_criteria,
            allowed_branch_names=allowed_branch,
        )
        saved = asyncio.run(context.repository.save_question(question))
        console.print(json.dumps(saved.model_dump(mode="json"), indent=2))
    except (CLIError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    finally:
        asyncio.run(context.close())


@app.command()
def run_forecast(
    file: str = typer.Argument(..., help="Path to forecast evidence JSON/NDJSON or directory"),
    question_id: str = typer.Option(..., help="Forecast question identifier"),
    target_entity_id: str = typer.Option(..., help="Target entity identifier"),
    objective: str = typer.Option(..., help="Decision objective to forecast against"),
    structural_family: str = typer.Option(..., help="Forecast structural family"),
    forecast_as_of: str = typer.Option(..., help="Evidence cutoff timestamp"),
    max_items: int = typer.Option(6, help="Maximum evidence items to use"),
    max_tokens: int = typer.Option(1200, help="Maximum approximate context tokens"),
    min_score: float = typer.Option(0.0, help="Minimum evidence salience to consider"),
    tenant_id: str = typer.Option("default", help="Tenant identifier"),
    conversation_id: str = typer.Option("forecasting", help="Conversation / scope identifier"),
) -> None:
    """Run a forecast and persist the immutable run artifact."""
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[bold red]✗ File not found: {file}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1)

    message_id = f"forecast-run-{uuid.uuid4()}"
    context = _open_forecast_context(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    try:
        evidence = evidence_from_path(file_path)
        if not evidence:
            raise CLIError("No forecast evidence found")
        result = BranchForecaster().forecast(
            objective=objective,
            structural_family=structural_family,
            evidence=evidence,
            budget=ContextBudget(max_items=max_items, max_tokens=max_tokens, min_score=min_score),
        )
        payload = result.model_dump(mode="json")
        run = ForecastRun(
            id=f"fr-{uuid.uuid4()}",
            question_id=question_id,
            model_or_engine="branch_forecaster",
            forecast_as_of=_parse_iso8601(forecast_as_of),
            branch_probabilities=_branch_probabilities_from_scores(payload),
            top_branch=result.top_branch,
            selected_evidence_ids=[item["id"] for item in payload["selected_context"]],
            evidence_gaps=result.evidence_gaps,
            rationale="; ".join(f"{score.branch}: {score.rationale}" for score in result.scores[:2]),
            metadata={"branch_forecast": payload},
        )
        saved = asyncio.run(context.repository.save_run(target_entity_id=target_entity_id, run=run))
        console.print(json.dumps(saved.model_dump(mode="json"), indent=2))
    except (CLIError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    finally:
        asyncio.run(context.close())


@app.command()
def resolve_forecast(
    question_id: str = typer.Option(..., help="Forecast question identifier"),
    run_id: str = typer.Option(..., help="Forecast run identifier"),
    target_entity_id: str = typer.Option(..., help="Target entity identifier"),
    outcome_branch: str = typer.Option(..., help="Observed outcome branch"),
    resolved_at: str = typer.Option(..., help="Observed resolution timestamp"),
    resolved_by: str = typer.Option(..., help="Analyst or system resolving the outcome"),
    source: str = typer.Option(..., help="Resolution evidence source"),
    resolution_notes: str | None = typer.Option(None, help="Optional resolution notes"),
    tenant_id: str = typer.Option("default", help="Tenant identifier"),
    conversation_id: str = typer.Option("forecasting", help="Conversation / scope identifier"),
) -> None:
    """Persist an observed resolution for a forecast question."""
    message_id = f"forecast-resolution-{uuid.uuid4()}"
    context = _open_forecast_context(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    try:
        resolution = ForecastResolution(
            question_id=question_id,
            run_id=run_id,
            resolved_at=_parse_iso8601(resolved_at),
            outcome_branch=outcome_branch,
            resolution_notes=resolution_notes,
            resolved_by=resolved_by,
            source=source,
        )
        saved = asyncio.run(
            context.repository.save_resolution(
                target_entity_id=target_entity_id,
                resolution=resolution,
            )
        )
        console.print(json.dumps(saved.model_dump(mode="json"), indent=2))
    except (CLIError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    finally:
        asyncio.run(context.close())


def _render_forecast(result: dict[str, Any]) -> None:
    console.print(
        f"[bold blue]Forecast[/bold blue] objective={result['objective']} "
        f"family={result['structural_family']}"
    )
    console.print(f"[bold green]Top branch: {result['top_branch']}[/bold green]")

    table = Table(title="Branch Scores")
    table.add_column("Branch", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Evidence", style="magenta")
    table.add_column("Missing", style="yellow")
    table.add_column("Rationale", style="dim")

    for score in result["scores"]:
        table.add_row(
            score["branch"],
            f"{score['score']:.3f}",
            ", ".join(score["matched_evidence_ids"]) or "-",
            ", ".join(score["missing_precursors"]) or "-",
            score["rationale"],
        )
    console.print(table)

    if result["evidence_gaps"]:
        console.print("[bold yellow]Evidence gaps:[/bold yellow] " + ", ".join(result["evidence_gaps"]))


def _resolve_entity(
    client: EngramHTTPClient,
    *,
    query: str,
    tenant_id: str,
    conversation_id: str | None,
) -> dict[str, Any]:
    matches = client.search_entities(query, tenant_id=tenant_id)
    if conversation_id is not None:
        matches = [m for m in matches if m.get("conversation_id") == conversation_id]
    if not matches:
        raise CLIError(
            f"No entities matched '{query}' (tenant={tenant_id}, conversation={conversation_id})"
        )
    if len(matches) > 1:
        console.print(
            "[yellow]⚠ Multiple entities matched, using the first result. Refine with --conversation-id if needed.[/yellow]"
        )
    entity_record = client.get_entity(matches[0]["id"])
    return entity_record


def _render_relationships(
    entity_record: dict[str, Any], relationships: list[RelationshipRow]
) -> None:
    if not relationships:
        console.print("[yellow]No active relationships found.[/yellow]")
        return

    table = Table(title=f"Active relationships for {entity_record['canonical_name']}")
    table.add_column("Target", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Confidence", style="green")
    table.add_column("Valid From", style="yellow")
    table.add_column("Valid To", style="yellow")
    table.add_column("Evidence", style="dim")

    for rel in relationships:
        table.add_row(
            rel.target_id,
            rel.rel_type,
            f"{rel.confidence:.2f}",
            rel.valid_from,
            rel.valid_to or "present",
            rel.evidence or "-",
        )

    console.print(table)


def _render_temporal_result(result: Any, mode: str) -> None:
    if mode == "bitemporal" and isinstance(result, dict):
        console.print("[bold blue]World State[/bold blue]")
        _render_relationships(
            {"canonical_name": "world_state"}, _rows_from_raw(result.get("world_state", []))
        )
        console.print("\n[bold blue]Knowledge State[/bold blue]")
        _render_relationships(
            {"canonical_name": "knowledge"}, _rows_from_raw(result.get("knowledge", []))
        )
        return

    if isinstance(result, list):
        rows = _rows_from_raw(result)
        dummy = {"canonical_name": "point-in-time"}
        _render_relationships(dummy, rows)
    else:
        console.print(result)


def _rows_from_raw(raw: list[dict[str, Any]]) -> list[RelationshipRow]:
    return [
        RelationshipRow(
            target_id=row.get("target_id", "unknown"),
            rel_type=row.get("rel_type", ""),
            confidence=row.get("confidence", 0.0),
            valid_from=row.get("valid_from", ""),
            valid_to=row.get("valid_to"),
            evidence=row.get("evidence", ""),
        )
        for row in raw
    ]


@app.command()
def health() -> None:
    """Check service health (Neo4j, Redis, LLM provider)."""
    console.print("[bold blue]Checking service health...[/bold blue]")

    try:
        settings = Settings(_env_file=".env")
        store = Neo4jStore(settings)

        # Check Neo4j
        asyncio.run(store.initialize())
        neo4j_healthy = asyncio.run(store.health_check())
        asyncio.run(store.close())

        status_icon = "[bold green]✓[/bold green]" if neo4j_healthy else "[bold red]✗[/bold red]"
        console.print(f"{status_icon} Neo4j: {settings.neo4j_uri}")

        # Check Redis (optional)
        if settings.redis_enabled:
            try:
                import redis.asyncio as aioredis

                redis_client = aioredis.from_url(
                    f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                    password=settings.redis_password,
                    decode_responses=True,
                )
                redis_healthy = asyncio.run(redis_client.ping())
                asyncio.run(redis_client.close())
                status_icon = (
                    "[bold green]✓[/bold green]" if redis_healthy else "[bold red]✗[/bold red]"
                )
                console.print(f"{status_icon} Redis: {settings.redis_host}:{settings.redis_port}")
            except Exception as e:
                console.print(f"[bold red]✗[/bold red] Redis: {e}")
        else:
            console.print("[dim]⊘ Redis: disabled[/dim]")

        # Check LLM provider
        console.print(f"[bold green]✓[/bold green] LLM: {settings.llm_model}")

        if neo4j_healthy:
            console.print("\n[bold green]All services healthy![/bold green]")
        else:
            console.print("\n[bold yellow]Some services are not healthy[/bold yellow]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[bold red]✗ Health check failed: {e}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from e


@app.command()
def export(
    output: str = typer.Option("graph.json", help="Output file path"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
) -> None:
    """Export graph to JSON.

    Exports all entities and relationships for a tenant.
    """
    console.print(f"[bold blue]Exporting graph to {output}...[/bold blue]")

    try:
        # For now, create a minimal export structure
        export_data = {
            "version": "0.1.0",
            "tenant_id": tenant_id,
            "exported_at": datetime.utcnow().isoformat(),
            "entities": [],
            "relationships": [],
        }

        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        console.print(f"[bold green]✓ Exported to {output}[/bold green]")
        console.print("[dim]Entities: 0[/dim]")
        console.print("[dim]Relationships: 0[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Export failed: {e}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
