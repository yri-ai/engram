"""Schema-guided branch forecasting primitives.

This module implements the thin-slice forecasting path described in
``docs/plans/2026-04-01-branch-forecasting-v0.md``: deterministic branch
ranking first, small context budgets, and a Bayesian feedback shell instead of
an RL policy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

from engram.models.branch_forecasting import (
    BranchDefinition,
    BranchFeedback,
    BranchForecast,
    BranchScore,
    ContextBudget,
    EvidenceItem,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


MARGIN_ANALYSIS_BRANCHES = [
    BranchDefinition(
        name="margin_expansion",
        description="Margins improve as revenue leverage or efficiency programs outweigh costs.",
        precursor_events=["pricing_power", "cost_reduction", "operating_leverage"],
        blocked_by_events=["input_cost_pressure", "demand_weakness"],
        prior=0.35,
    ),
    BranchDefinition(
        name="margin_compression",
        description="Margins deteriorate as costs, mix, or demand weakness overwhelm offsets.",
        precursor_events=["input_cost_pressure", "demand_weakness", "negative_mix_shift"],
        blocked_by_events=["pricing_power", "cost_reduction"],
        prior=0.4,
    ),
    BranchDefinition(
        name="margin_stability",
        description="Margins stay range-bound because opposing forces offset each other.",
        precursor_events=["stable_demand", "offsetting_cost_actions"],
        blocked_by_events=[],
        prior=0.25,
    ),
]


DEFAULT_BRANCH_FAMILIES: dict[str, list[BranchDefinition]] = {
    "margin_analysis": MARGIN_ANALYSIS_BRANCHES,
    "real_estate_acquisition": [
        BranchDefinition(
            name="advance_diligence",
            description="The deal has enough operating, rent roll, and underwriting support to keep underwriting.",
            precursor_events=[
                "offering_memorandum",
                "rent_roll",
                "operating_statement",
                "underwriting_model",
                "tax_data",
            ],
            blocked_by_events=[
                "environmental_risk",
                "site_constraint",
                "legal_document_risk",
                "tax_increase",
            ],
            prior=0.35,
        ),
        BranchDefinition(
            name="reprice_or_restructure",
            description="The deal likely needs price, debt, reserve, or structure changes before proceeding.",
            precursor_events=[
                "debt_pressure",
                "tax_increase",
                "operating_statement",
                "rent_roll",
            ],
            blocked_by_events=["rent_growth_support", "tax_savings"],
            prior=0.4,
        ),
        BranchDefinition(
            name="diligence_blocked",
            description="Third-party, legal, environmental, or site diligence may block or materially delay the deal.",
            precursor_events=[
                "environmental_risk",
                "site_constraint",
                "legal_document_risk",
                "survey_title",
            ],
            blocked_by_events=["tax_savings", "rent_growth_support"],
            prior=0.25,
        ),
    ],
}

EVENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("input_cost_pressure", ("cost pressure", "freight", "commodity", "inflation")),
    ("demand_weakness", ("demand weakness", "soft demand", "lower volume", "slowing demand")),
    ("negative_mix_shift", ("negative mix", "mix shift", "lower margin mix")),
    ("pricing_power", ("pricing power", "price increase", "raised prices", "pricing")),
    ("cost_reduction", ("cost reduction", "efficiency", "restructuring", "savings")),
    ("operating_leverage", ("operating leverage", "fixed cost leverage", "scale benefits")),
    ("stable_demand", ("stable demand", "steady demand", "stable volume")),
    ("offsetting_cost_actions", ("offsetting cost", "cost offset", "productivity offset")),
    ("offering_memorandum", ("offering memorandum", " om", "_om", " offering ")),
    ("rent_roll", ("rent roll", "rent-roll", "rr -", "_rr", "lease charges")),
    (
        "operating_statement",
        ("t12", "trailing", "profit and loss", "operating statement", "financial workbook"),
    ),
    ("underwriting_model", ("underwriting", "acq summary", "acquisition summary")),
    ("debt_pressure", ("debt matrix", "refinance", "loan", "interest rate")),
    ("tax_increase", ("trim notice", "tax increase", "reassessment")),
    ("tax_savings", ("tax savings",)),
    ("tax_data", ("tax bill", "tax bills", "property tax")),
    ("rent_growth_support", ("rent growth", "rent regression", "comparables", "comps")),
    ("environmental_risk", ("environmental", "phase 1", "loma", "flood")),
    ("site_constraint", ("site plan", "geotech", "geo report", "building areas", "unit matrix")),
    ("survey_title", ("survey", "boundary", "topo", "title")),
    ("legal_document_risk", ("purchase and sale", "psa", "lease", "loi", "condominium documents")),
)


def _event_match_score(evidence: EvidenceItem, event_type: str) -> float:
    """Return a soft deterministic score for evidence-event alignment."""
    normalized_event = event_type.casefold().replace("_", " ")
    normalized_type = evidence.event_type.casefold().replace("_", " ")
    normalized_text = evidence.text.casefold().replace("_", " ")

    if normalized_type == normalized_event:
        return 1.0
    if normalized_event in normalized_text:
        return 0.8
    tokens = set(normalized_event.split())
    if tokens and tokens.issubset(set(normalized_text.split())):
        return 0.6
    return 0.0


def evidence_from_records(records: Iterable[dict[str, object]]) -> list[EvidenceItem]:
    """Convert loose JSON-like records into forecast evidence.

    The preferred input is explicit ``event_type`` records. For early CLI use,
    plain text records are also accepted and mapped through a conservative
    keyword table for the first structural family.
    """
    items: list[EvidenceItem] = []
    for idx, record in enumerate(records):
        text_raw = record.get("text") or record.get("evidence") or record.get("claim") or ""
        text = str(text_raw).strip()
        if not text:
            continue

        metadata_raw = record.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
        event_type = str(record.get("event_type") or metadata.get("event_type") or "")
        if not event_type:
            event_type = infer_event_type(text)
        if not event_type:
            event_type = "unknown"

        salience_raw = record.get("salience", record.get("confidence", 1.0))
        salience = float(salience_raw) if isinstance(salience_raw, int | float) else 1.0
        tokens_raw = record.get("tokens")
        tokens = (
            int(tokens_raw)
            if isinstance(tokens_raw, int) and tokens_raw > 0
            else _estimate_tokens(text)
        )

        items.append(
            EvidenceItem(
                id=str(record.get("id") or record.get("record_id") or f"e{idx + 1}"),
                text=text,
                event_type=event_type,
                source=str(record["source"]) if "source" in record else None,
                timestamp=str(record["timestamp"]) if "timestamp" in record else None,
                salience=max(0.0, salience),
                tokens=tokens,
                metadata={str(key): value for key, value in metadata.items()},
            )
        )
    return items


def evidence_from_path(path: Path, *, max_files: int = 250) -> list[EvidenceItem]:
    """Load forecast evidence from JSON/NDJSON files or a structured data directory."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return evidence_from_directory(path, max_files=max_files)
    if path.suffix.casefold() == ".json":
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return evidence_from_records(_records_from_payload(payload))
    if path.suffix.casefold() == ".ndjson":
        import json

        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return evidence_from_records(records)
    return evidence_from_records([record_from_file(path, path.parent)])


def evidence_from_directory(directory: Path, *, max_files: int = 250) -> list[EvidenceItem]:
    """Create evidence from a deal-room style directory tree."""
    records: list[dict[str, object]] = []
    for file_path in sorted(directory.rglob("*")):
        if len(records) >= max_files:
            break
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        if file_path.suffix.casefold() not in {
            ".json",
            ".ndjson",
            ".pdf",
            ".xlsx",
            ".xls",
            ".docx",
            ".doc",
        }:
            continue
        records.append(record_from_file(file_path, directory))
    return evidence_from_records(records)


def record_from_file(file_path: Path, root: Path) -> dict[str, object]:
    """Map a structured sample file into a loose forecast evidence record."""
    try:
        relative = file_path.relative_to(root)
    except ValueError:
        relative = file_path
    text = str(relative)
    event_type = infer_event_type(text)
    return {
        "id": _safe_evidence_id(relative),
        "text": f"Structured diligence file available: {text}",
        "event_type": event_type or "unknown",
        "source": str(file_path),
        "salience": _salience_for_file(file_path, event_type),
        "metadata": {
            "path": str(file_path),
            "relative_path": str(relative),
            "suffix": file_path.suffix.casefold(),
        },
    }


def _records_from_payload(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("evidence", "records", "messages"):
        records = payload.get(key)
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    return []


def infer_event_type(text: str) -> str:
    """Infer the first known forecast event type from text."""
    normalized = text.casefold()
    for event_type, keywords in EVENT_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return event_type
    return ""


def _estimate_tokens(text: str) -> int:
    # A stable approximation is enough for context-budget pruning.
    return max(1, math.ceil(len(text.split()) * 1.3))


def _filter_evidence_as_of(
    evidence: Iterable[EvidenceItem], forecast_as_of: datetime | None
) -> list[EvidenceItem]:
    if forecast_as_of is None:
        return list(evidence)

    filtered: list[EvidenceItem] = []
    for item in evidence:
        if item.timestamp is None:
            filtered.append(item)
            continue
        try:
            item_time = datetime.fromisoformat(item.timestamp.replace("Z", "+00:00"))
        except ValueError:
            filtered.append(item)
            continue
        if item_time <= forecast_as_of:
            filtered.append(item)
    return filtered


def _safe_evidence_id(path: Path) -> str:
    stem = str(path.with_suffix(""))
    safe = "".join(char.casefold() if char.isalnum() else "-" for char in stem)
    return "-".join(part for part in safe.split("-") if part)[:120]


def _salience_for_file(file_path: Path, event_type: str) -> float:
    suffix = file_path.suffix.casefold()
    if event_type == "unknown":
        return 0.2
    if suffix in {".xlsx", ".xls"}:
        return 0.9
    if suffix in {".docx", ".doc"}:
        return 0.85
    if suffix == ".pdf":
        return 0.75
    return 0.6


class ContextBudgetSelector:
    """Select compact, discriminative evidence under explicit budgets."""

    def select(
        self,
        evidence: Iterable[EvidenceItem],
        branches: Iterable[BranchDefinition],
        budget: ContextBudget,
    ) -> list[EvidenceItem]:
        """Return the highest information-value evidence within budget."""
        branch_list = list(branches)
        scored = [
            (self._information_value(item, branch_list), item)
            for item in evidence
            if item.salience >= budget.min_score
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].tokens, pair[1].id))

        selected: list[EvidenceItem] = []
        token_total = 0
        for score, item in scored:
            if score <= 0.0:
                continue
            if len(selected) >= budget.max_items:
                break
            if token_total + item.tokens > budget.max_tokens:
                continue
            selected.append(item)
            token_total += item.tokens
        return selected

    def _information_value(
        self,
        evidence: EvidenceItem,
        branches: list[BranchDefinition],
    ) -> float:
        matches = 0
        blockers = 0
        for branch in branches:
            matches += sum(
                1
                for event_type in branch.precursor_events
                if _event_match_score(evidence, event_type)
            )
            blockers += sum(
                1
                for event_type in branch.blocked_by_events
                if _event_match_score(evidence, event_type)
            )

        # Reward discriminative evidence that either supports or rules out a branch.
        # Salience keeps upstream extraction confidence in the loop without making
        # the selector a flat top-K salience picker.
        return evidence.salience * float(matches + blockers)


class BayesianUpdateShell:
    """Small online belief store using Beta priors per objective and branch."""

    def __init__(self) -> None:
        self._beliefs: dict[tuple[str, str], list[float]] = defaultdict(lambda: [1.0, 1.0])

    def update(self, feedback: BranchFeedback) -> None:
        alpha_beta = self._beliefs[(feedback.objective, feedback.branch)]
        if feedback.useful:
            alpha_beta[0] += feedback.weight
        else:
            alpha_beta[1] += feedback.weight

    def expected_relevance(self, objective: str, branch: str) -> float:
        alpha, beta = self._beliefs[(objective, branch)]
        return alpha / (alpha + beta)

    def uncertainty(self, objective: str, branch: str) -> float:
        alpha, beta = self._beliefs[(objective, branch)]
        total = alpha + beta
        return (alpha * beta) / ((total * total) * (total + 1.0))


class BranchRankingScaffold:
    """Deterministic branch scorer using precursor and blocker constraints."""

    def __init__(self, bayes: BayesianUpdateShell | None = None) -> None:
        self._bayes = bayes or BayesianUpdateShell()

    def rank(
        self,
        objective: str,
        branches: Iterable[BranchDefinition],
        evidence: Iterable[EvidenceItem],
    ) -> list[BranchScore]:
        branch_list = list(branches)
        evidence_list = list(evidence)
        raw_scores = [
            self._score_branch(objective, branch, evidence_list) for branch in branch_list
        ]
        max_score = max((score for score, _ in raw_scores), default=1.0)
        normalizer = max(max_score, 1.0)

        results = []
        for raw_score, branch_score in raw_scores:
            score = max(0.0, min(1.0, raw_score / normalizer))
            results.append(branch_score.model_copy(update={"score": score}))
        results.sort(key=lambda item: (-item.score, item.branch))
        return results

    def _score_branch(
        self,
        objective: str,
        branch: BranchDefinition,
        evidence: list[EvidenceItem],
    ) -> tuple[float, BranchScore]:
        matched: list[str] = []
        missing: list[str] = []
        blocked: list[str] = []
        support = 0.0
        penalty = 0.0

        for precursor in branch.precursor_events:
            precursor_matches = [
                item for item in evidence if _event_match_score(item, precursor) > 0.0
            ]
            if precursor_matches:
                best = max(
                    precursor_matches,
                    key=lambda item: _event_match_score(item, precursor) * item.salience,
                )
                matched.append(best.id)
                support += _event_match_score(best, precursor) * best.salience
            else:
                missing.append(precursor)

        for blocker in branch.blocked_by_events:
            blocker_matches = [item for item in evidence if _event_match_score(item, blocker) > 0.0]
            for item in blocker_matches:
                blocked.append(item.id)
                penalty += _event_match_score(item, blocker) * item.salience

        prior = math.sqrt(branch.prior)
        belief = self._bayes.expected_relevance(objective, branch.name)
        uncertainty_bonus = self._bayes.uncertainty(objective, branch.name)
        raw_score = max(0.0, prior + support + belief + uncertainty_bonus - penalty)

        if blocked and matched:
            rationale = "supported with offsetting evidence"
        elif blocked:
            rationale = "blocked by contradictory evidence"
        elif missing:
            rationale = "supported but missing precursor evidence"
        else:
            rationale = "all precursor constraints satisfied"

        return raw_score, BranchScore(
            branch=branch.name,
            score=0.0,
            matched_evidence_ids=sorted(set(matched)),
            missing_precursors=missing,
            blocked_by_evidence_ids=sorted(set(blocked)),
            rationale=rationale,
        )


class BranchForecaster:
    """Minimal branch forecaster for transition-first prediction."""

    def __init__(
        self,
        branch_families: dict[str, list[BranchDefinition]] | None = None,
        selector: ContextBudgetSelector | None = None,
        bayes: BayesianUpdateShell | None = None,
    ) -> None:
        self._branch_families = branch_families or DEFAULT_BRANCH_FAMILIES
        self._selector = selector or ContextBudgetSelector()
        self._bayes = bayes or BayesianUpdateShell()
        self._ranker = BranchRankingScaffold(self._bayes)

    def forecast(
        self,
        objective: str,
        structural_family: str,
        evidence: Iterable[EvidenceItem],
        budget: ContextBudget | None = None,
        forecast_as_of: datetime | None = None,
    ) -> BranchForecast:
        branches = self._branch_families.get(structural_family)
        if not branches:
            raise ValueError(f"unknown structural family: {structural_family}")

        context_budget = budget or ContextBudget()
        filtered_evidence = _filter_evidence_as_of(evidence, forecast_as_of)
        selected = self._selector.select(filtered_evidence, branches, context_budget)
        scores = self._ranker.rank(objective, branches, selected)
        top_branch = scores[0].branch if scores else ""
        evidence_gaps = sorted({gap for score in scores[:2] for gap in score.missing_precursors})

        return BranchForecast(
            objective=objective,
            structural_family=structural_family,
            top_branch=top_branch,
            scores=scores,
            selected_context=selected,
            evidence_gaps=evidence_gaps,
        )

    def update(self, feedback: BranchFeedback) -> None:
        self._bayes.update(feedback)
