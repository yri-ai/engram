"""Canonical tabular feature builder for forecast heads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class FeatureConfig:
    """Stable feature configuration with persisted categorical vocabularies."""

    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    category_maps: dict[str, list[str]] = field(default_factory=dict)
    include_event_history: bool = True

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> FeatureConfig:
        payload = json.loads(path.read_text())
        return cls(**payload)


def rows_to_matrix(
    rows: list[dict[str, Any]], feature_config: FeatureConfig | None = None
) -> tuple[np.ndarray, list[str], list[str], FeatureConfig]:
    """Convert canonical rows to ``X, y, feature_names, config`` deterministically."""
    config = feature_config or _infer_config(rows)
    if not config.category_maps:
        config.category_maps = _fit_categories(rows, config.categorical_features)

    feature_names = [*config.numeric_features]
    for name in config.categorical_features:
        feature_names.extend(f"{name}={value}" for value in config.category_maps.get(name, []))
        feature_names.append(f"{name}=<UNK>")

    matrix = [_encode_row(row, config) for row in rows]
    labels = [str(row.get("label", {}).get("next_bucket", "")) for row in rows]
    return np.asarray(matrix, dtype=float), labels, feature_names, config


def _infer_config(rows: list[dict[str, Any]]) -> FeatureConfig:
    numeric: set[str] = set()
    categorical: set[str] = set()
    for row in rows:
        for key, value in row.get("features", {}).items():
            if isinstance(value, (bool, int | float)) or value is None:
                numeric.add(key)
            else:
                categorical.add(key)
    return FeatureConfig(numeric_features=sorted(numeric), categorical_features=sorted(categorical))


def _fit_categories(rows: list[dict[str, Any]], names: list[str]) -> dict[str, list[str]]:
    maps: dict[str, list[str]] = {}
    for name in names:
        values = {
            str(row.get("features", {}).get(name))
            for row in rows
            if row.get("features", {}).get(name) is not None
        }
        maps[name] = sorted(values)
    return maps


def _encode_row(row: dict[str, Any], config: FeatureConfig) -> list[float]:
    features = row.get("features", {})
    values: list[float] = []
    for name in config.numeric_features:
        raw = features.get(name)
        values.append(float(raw) if isinstance(raw, int | float) else np.nan)
    for name in config.categorical_features:
        raw = features.get(name)
        encoded = str(raw) if raw is not None else "<MISSING>"
        known = config.category_maps.get(name, [])
        values.extend(1.0 if encoded == value else 0.0 for value in known)
        values.append(0.0 if encoded in known else 1.0)
    return values
