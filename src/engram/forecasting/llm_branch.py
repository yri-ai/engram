"""Mockable LLM branch rollout engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from engram.forecasting.branches import Branch, enumerate_bucket_branches

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class ComputeBudget:
    n_particles: int = 3
    max_depth: int = 1
    tokens_per_particle: int = 128


class LLMBranchForecaster:
    name = "llm_branch_mockable"

    def __init__(self, compute_budget: ComputeBudget | None = None) -> None:
        self.compute_budget = compute_budget or ComputeBudget()

    def rollout(
        self, prompt: str, scorer: Callable[[str, Branch], float] | None = None
    ) -> dict[str, object]:
        branches = enumerate_bucket_branches(self.compute_budget.max_depth)
        weights: dict[str, float] = {branch.target_bucket.value: 0.0 for branch in branches}
        for _ in range(self.compute_budget.n_particles):
            for branch in branches:
                score = scorer(prompt, branch) if scorer else 1.0
                weights[branch.target_bucket.value] += max(score, 0.0)
        total = sum(weights.values()) or 1.0
        probabilities = {key: value / total for key, value in weights.items()}
        tokens = self.compute_budget.n_particles * self.compute_budget.tokens_per_particle
        return {"probabilities": probabilities, "cost": {"tokens": tokens, "usd": tokens * 0.000001}}
