"""Phase 2 graph export, graph heads, and ensemble coverage."""

from __future__ import annotations

import json

from engram.forecasting.baselines import HazardForecaster
from engram.forecasting.ensemble import LinearPoolEnsemble
from engram.forecasting.fixtures import load_forecast_fixture_rows
from engram.forecasting.graph_export import GraphEdge, export_graph_snapshot, filter_edges_as_of
from engram.forecasting.graph_head import TemporalGNNForecaster, UltraForecaster
from engram.forecasting.tfm import TFMForecaster
from engram.services.track_b_graph_features import graph_features


def test_graph_export_filters_future_recorded_edges_and_writes_vocab(tmp_path):
    edges = [
        GraphEdge("loan:1", "transitions_to", "current", "2025-01-01", "2025-03-01T10:00:00+00:00"),
        GraphEdge("loan:1", "transitions_to", "d90", "2025-02-01", "2025-12-31"),
    ]
    assert [edge.tail for edge in filter_edges_as_of(edges, "2025-03-01")] == ["current"]

    paths = export_graph_snapshot(edges, "2025-03-01", tmp_path)
    exported = [json.loads(line) for line in paths["edges"].read_text().splitlines()]
    assert len(exported) == 1
    assert json.loads(paths["relations"].read_text()) == ["transitions_to"]


def test_graph_heads_and_ensemble_are_protocol_compatible():
    rows = load_forecast_fixture_rows("track_b_synthetic")
    graph = UltraForecaster(edges=[{"head": rows[-1]["loan_id"], "tail": "d30"}])
    graph.fit(rows[:8])
    probs = graph.predict_proba(rows[-1]["features"])
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert TemporalGNNForecaster().name == "temporal_gnn_deterministic_adapter"

    ensemble = LinearPoolEnsemble([TFMForecaster(context_budget=4), HazardForecaster()])
    ensemble.fit(rows[:8])
    ensemble_probs = ensemble.predict_proba(rows[-1]["features"])
    assert abs(sum(ensemble_probs.values()) - 1.0) < 1e-9


def test_track_b_graph_features_are_record_time_safe():
    features = graph_features("loan-1", "2025-02-01", driver=None)
    assert features["graph_entity_degree"] >= 0
    assert "graph_supersession_count" in features
