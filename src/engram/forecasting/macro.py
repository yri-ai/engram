"""Lazy macro TSFM adapter."""

from __future__ import annotations


class ChronosMacroAdapter:
    def predict(self, history: list[float], horizon: int) -> list[float]:
        last = history[-1] if history else 0.0
        return [last for _ in range(horizon)]

    def load_optional_backend(self) -> object:
        import torch  # type: ignore[import-not-found]

        return torch
