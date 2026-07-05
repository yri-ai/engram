"""Record-time-correct graph snapshot export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class GraphEdge:
    head: str
    rel: str
    tail: str
    valid_from: str
    recorded_from: str


def filter_edges_as_of(edges: list[GraphEdge], as_of: str) -> list[GraphEdge]:
    """Keep only edges recorded by ``as_of``."""
    as_of_dt = _parse_record_time(as_of)
    return [
        edge
        for edge in edges
        if _parse_record_time(edge.recorded_from) <= as_of_dt
        and _parse_record_time(edge.valid_from) <= as_of_dt
    ]


def export_graph_snapshot(edges: list[GraphEdge], as_of: str, output_dir: Path) -> dict[str, Path]:
    """Write NDJSON edge snapshot plus entity/relation vocab files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = filter_edges_as_of(edges, as_of)
    edge_path = output_dir / f"edges_{as_of}.ndjson"
    entity_vocab = sorted({edge.head for edge in filtered} | {edge.tail for edge in filtered})
    relation_vocab = sorted({edge.rel for edge in filtered})
    edge_path.write_text(
        "".join(json.dumps(asdict(edge), sort_keys=True) + "\n" for edge in filtered)
    )
    entity_path = output_dir / "entity_vocab.json"
    relation_path = output_dir / "relation_vocab.json"
    entity_path.write_text(json.dumps(entity_vocab, indent=2) + "\n")
    relation_path.write_text(json.dumps(relation_vocab, indent=2) + "\n")
    return {"edges": edge_path, "entities": entity_path, "relations": relation_path}


def _parse_record_time(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    if "T" not in normalized:
        normalized = f"{normalized}T23:59:59+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
