"""Deal-level prediction engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engram.forecasting.scenarios import sample_collateral_paths
from engram.forecasting.waterfall import simulate

if TYPE_CHECKING:
    from engram.models.deal import DealSpec


def predict_deal(deal_spec: DealSpec, as_of: str, n_paths: int, compute_budget: int) -> dict[str, object]:
    del as_of, compute_budget
    paths = sample_collateral_paths([100.0, 90.0, 80.0], n_paths)
    losses = 0
    breaches = 0
    for path in paths:
        outcome = simulate(deal_spec, path)
        breaches += int(any(outcome.trigger_states.values()))
        principal_balance = sum(tranche.balance for tranche in deal_spec.tranches)
        collateral_available = sum(path)
        severe_shortfall = collateral_available < (principal_balance * 0.25)
        losses += int(severe_shortfall)
    denom = max(n_paths, 1)
    return {"p_trigger_breach": breaches / denom, "p_tranche_loss": losses / denom, "expected_waterfall_path": paths[0] if paths else []}
