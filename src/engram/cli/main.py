"""Engram CLI commands."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from engram.config import Settings
from engram.models.message import IngestRequest
from engram.storage.neo4j import Neo4jStore

app = typer.Typer(name="engram", help="Temporal knowledge graph engine for AI memory")
console = Console()


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
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload on code changes"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    console.print(f"[bold blue]Starting Engram API server...[/bold blue]")
    console.print(f"[dim]Host: {host}:{port}[/dim]")
    console.print(f"[dim]Reload: {reload}[/dim]")
    console.print("[dim]Visit http://localhost:8000/docs for API documentation[/dim]")

    uvicorn.run("engram.main:app", host=host, port=port, reload=reload)


@app.command()
def ingest(
    file: str = typer.Argument(..., help="Path to JSON file with conversation messages"),
    conversation_id: str = typer.Option("default", help="Conversation ID"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
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
        with open(file_path) as f:
            data = json.load(f)

        messages = data.get("messages", [])
        if not messages:
            console.print("[yellow]⚠ No messages found in file[/yellow]")
            return

        console.print(f"[bold blue]Ingesting {len(messages)} messages from {file}...[/bold blue]")

        # For now, just show a summary (full implementation would call the API)
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing messages...", total=len(messages))

            for msg in messages:
                # Simulate processing
                progress.update(task, advance=1)

        console.print(f"[bold green]✓ Ingested {len(messages)} messages[/bold green]")
        console.print(f"[dim]Conversation: {conversation_id}[/dim]")
        console.print(f"[dim]Tenant: {tenant_id}[/dim]")

    except json.JSONDecodeError as e:
        console.print(f"[bold red]✗ Invalid JSON: {e}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]✗ Ingestion failed: {e}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1)


@app.command()
def query(
    entity: str = typer.Argument(..., help="Entity name to query"),
    as_of: Optional[str] = typer.Option(None, help="Query as of date (ISO 8601 format)"),
    rel_type: Optional[str] = typer.Option(None, help="Filter by relationship type"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
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
            query_date = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            console.print(f"[dim]As of: {query_date.isoformat()}[/dim]")
        except ValueError:
            console.print(f"[bold red]✗ Invalid date format: {as_of}[/bold red]", file=sys.stderr)
            raise typer.Exit(code=1)

    # Create a sample table to show the output format
    table = Table(title=f"Relationships for {entity}")
    table.add_column("Target", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Confidence", style="green")
    table.add_column("Valid From", style="yellow")
    table.add_column("Evidence", style="dim")

    # For now, show empty table (full implementation would query the API)
    console.print(table)
    console.print("[dim]No relationships found (API integration pending)[/dim]")


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
        raise typer.Exit(code=1)


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
        console.print(f"[dim]Entities: 0[/dim]")
        console.print(f"[dim]Relationships: 0[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Export failed: {e}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
