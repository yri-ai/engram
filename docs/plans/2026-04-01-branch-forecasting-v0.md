# Branch Forecasting v0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a pragmatic branch-forecasting strategy that identifies decision-relevant historical context using a thin-slice-first approach, starting with one structural family and 2-3 branches.

**Architecture:** A "Context-Aware Branch Selection" framework that uses a context budget selector and a deterministic branch ranking scaffold. It incorporates a Bayesian update shell to refine relevance over time, prioritizing information value for specific decision objectives.

**Tech Stack:** Python 3.11+ | Neo4j | LiteLLM | Pydantic v2 | pytest

---

## Options Considered

| Option | Approach | Trade-offs |
| :--- | :--- | :--- |
| **1. RL Policy** | Optimize for reward (user clicks/feedback) | **Cons:** Sparse rewards, delayed feedback, high data requirement, "black box" policy. |
| **2. Vector RAG** | Top-K similarity search | **Cons:** Lacks temporal awareness, redundant results, no causal understanding. |
| **3. Causal Bayesian (Rec)** | Causal discovery + Info-Gain selection | **Pros:** Interpretable, sample-efficient, handles delayed feedback, auditable evidence. |

## Recommended Approach: Thin-Slice-First

We will implement a **Thin-Slice-First** version of the Causal Bayesian framework. This approach starts with a single structural family (e.g., "margin analysis") and 2-3 branches to ensure immediate testability. We prioritize a deterministic ranking scaffold and a context budget selector over a full causal-discovery buildout, ensuring an environment-light and tests-first start.

---

## Two-Week Execution Sequence

### Week 1: Environment & Contracts

1. **Task 1: Environment & Lock**
   - Ensure `uv.lock` is up-to-date and environment is stable.
   - Define core Pydantic contracts for `BranchForecast` and `ContextBudget`.

2. **Task 2: Unit Tests & Ranking Scaffold**
   - Write unit tests for deterministic branch ranking.
   - Implement the `BranchRankingScaffold` with hardcoded causal priors for the first structural family.

3. **Task 3: Context Budget Selector**
   - Implement `ContextBudgetSelector` to limit the amount of historical context fed to the LLM.
   - Add tests for budget enforcement and pruning logic.

4. **Task 4: Minimal Implementation & Bayesian Shell**
   - Implement a minimal `BranchForecaster` service.
   - Add a `BayesianUpdateShell` that captures feedback but uses simple heuristics for now.
   - Focused integration test with a single mock conversation.

### Week 2: Stress Testing & Calibration

5. **Task 5: Distractor Stress Tests**
   - Add "noise" claims to the test fixtures to verify ranking robustness.
   - Ensure the forecaster correctly ignores irrelevant branches.

6. **Task 6: Context-Budget Sweeps**
   - Run sweeps across different budget levels to find the optimal balance between context and latency.
   - Log performance metrics and token usage.

7. **Task 7: Calibration Logging**
   - Implement detailed logging for branch selection decisions ("Why this branch?").
   - Add a calibration dashboard/report for manual review of ranking quality.

8. **Task 8: End-to-End Pass**
   - Run a full e2e pass using the `coaching-demo` with branch forecasting enabled.
   - Verify that the selected context improves extraction quality in a demonstrable way.

---

## Test Strategy

- **Unit Layer:** Focus on `BranchRankingScaffold` and `ContextBudgetSelector` with deterministic inputs.
- **Integration Layer:** Verify the `BranchForecaster` interacts correctly with the existing extraction pipeline.
- **Eval-Fixture Layer:** Use a subset of `feature-fixtures` to validate ranking against known "ground truth" causal chains.

---

## Immediate Start (Today)

1. `uv sync`
2. `uv run pytest tests/unit -q`
3. run focused new tests
4. `uv run pytest -q`

---

## Risks & Mitigations

- **Risk: Fixture Readiness.** The `feature-fixtures` pipeline might not be fully ready for complex branch testing.
  - **Mitigation:** Start with manual, hardcoded fixtures for the first structural family in Week 1.
- **Risk: Dependency Drift.** Rapid changes in the repo might break the new forecasting contracts.
  - **Mitigation:** Strict adherence to the defined Pydantic models and frequent `uv sync` checks.
- **Risk: Overfitting to One Family.** The ranking logic might become too specific to "margin analysis."
  - **Mitigation:** Ensure the `BranchRankingScaffold` is designed to accept different "structural family" configurations.
