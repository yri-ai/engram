"""Pure deterministic deal waterfall simulator."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.models.deal import DealSpec


@dataclass(frozen=True, slots=True)
class DealOutcome:
    tranche_cashflows: dict[str, list[float]]
    trigger_states: dict[str, bool]
    retained_cash: list[float]
    ending_balances: dict[str, float]


def simulate(deal_spec: DealSpec, collateral_cashflows: list[float]) -> DealOutcome:
    deal_spec.require_verified()
    balances = {tranche.tranche_id: tranche.balance for tranche in deal_spec.tranches}
    tranche_cashflows: dict[str, list[float]] = {
        tranche.tranche_id: [] for tranche in deal_spec.tranches
    }
    retained: list[float] = []
    ordered = sorted(deal_spec.tranches, key=lambda tranche: tranche.seniority)
    for inflow in collateral_cashflows:
        remaining = inflow
        period_payments = {tranche.tranche_id: 0.0 for tranche in deal_spec.tranches}
        steps = sorted(deal_spec.waterfall, key=lambda step: step.priority)
        if steps:
            for step in steps:
                tranche_id = step.tranche_id
                if tranche_id not in balances:
                    continue
                if step.rule == "residual":
                    pay = remaining
                elif step.rule == "interest":
                    pay = min(
                        remaining, balances[tranche_id] * _coupon_for(deal_spec, tranche_id) / 12.0
                    )
                else:
                    pay = min(remaining, balances[tranche_id])
                    balances[tranche_id] -= pay
                remaining -= pay
                period_payments[tranche_id] += pay
        else:
            for tranche in ordered:
                pay = min(remaining, balances[tranche.tranche_id])
                balances[tranche.tranche_id] -= pay
                remaining -= pay
                period_payments[tranche.tranche_id] += pay
        for tranche in ordered:
            tranche_cashflows[tranche.tranche_id].append(period_payments[tranche.tranche_id])
        retained.append(remaining)
    trigger_states = {
        trigger.trigger_id: _evaluate_trigger_breach(
            trigger.formula, trigger.threshold, collateral_cashflows, deal_spec
        )
        for trigger in deal_spec.triggers
    }
    return DealOutcome(
        tranche_cashflows=tranche_cashflows,
        trigger_states=trigger_states,
        retained_cash=retained,
        ending_balances=dict(balances),
    )


def _evaluate_trigger_breach(
    formula: str, threshold: float, collateral_cashflows: list[float], deal_spec: DealSpec
) -> bool:
    variables = {
        "assets": float(sum(collateral_cashflows)),
        "liabilities": float(sum(tranche.balance for tranche in deal_spec.tranches)) or 1.0,
        "cashflow": float(sum(collateral_cashflows)),
    }
    value = _safe_eval_formula(formula, variables)
    return value < threshold


def _safe_eval_formula(formula: str, variables: dict[str, float]) -> float:
    tree = ast.parse(formula, mode="eval")
    return float(_eval_node(tree.body, variables))


def _eval_node(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"unknown formula variable {node.id!r}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, variables)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported formula expression")


def _coupon_for(deal_spec: DealSpec, tranche_id: str) -> float:
    for tranche in deal_spec.tranches:
        if tranche.tranche_id == tranche_id:
            return tranche.coupon
    return 0.0
