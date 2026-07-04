# Prediction Upgrade — Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Companion to:** `docs/plans/2026-07-04-prediction-upgrade-roadmap.md` (the "why" and research grounding)
**This document:** the "how" — task-by-task, grounded in the current codebase.

**Tech Stack:** Python 3.11+ | uv | pytest | Pydantic v2 | Neo4j | LiteLLM | numpy (+ new optional deps per phase)

---

## Codebase Ground Truth (what we build on)

| Existing asset | Location | Role in this plan |
|---|---|---|
| `BaselineForecaster` (fit/predict/backtest) | `src/engram/services/track_b_forecasting.py` | Defines the de-facto forecaster protocol; stays as yardstick |
| Dataset builder + `LeakageError` + split validation | `src/engram/services/track_b_dataset.py` | Extended, not replaced — walk-forward and record-time replay build on `assign_splits`/`validate_no_leakage` |
| Event-history features | `src/engram/services/track_b_graph_features.py::extract_features_from_event_history` | Feature source for TFM head; extended with true graph features |
| Backtest runner | `scripts/data_collection/run_track_b_backtest.py` | Superseded by harness runner (kept working during transition) |
| H1 motifs | `src/engram/services/h1_schema.py::induce_motifs` | Evidence selection for LLM head |
| H2 context profiles | `src/engram/services/h2_context.py::profile_schema_guided`, `compute_evidence_gaps`, `compute_competing_cause_discrimination` | Context budgeting + flip-condition output for LLM head |
| H4 pruning | `src/engram/services/h4_symbolic.py::prune_predictions` | Consistency check on LLM head outputs (Gate 5 said: re-test when LLM models arrive) |
| H5 transfer harness | `scripts/data_collection/run_h5_transfer.py` | Reused for cross-asset-class transfer tests |
| Data + fetchers | `data/{ginnie,fannie,edgar,dbrs}`, `scripts/data_collection/fetch_*.py` | Phase 4 extends with ABS-EE + remittance ingestion |
| Test conventions | `tests/unit/test_track_b_*.py`, pytest via uv, ruff/mypy | Every task ships tests in the same pattern |

**Structural decision:** new subpackage `src/engram/forecasting/` for the prediction subsystem (the `services/` flat namespace is already 25+ modules; forecasting is becoming a system, not a service). `services/track_b_forecasting.py` re-exports for backward compatibility until Phase 2 completes.

**Dependency policy:** heavy ML deps go in new optional groups so the core stays light:

```toml
[project.optional-dependencies]
forecast = ["scikit-learn>=1.5", "lightgbm>=4.5", "pandas>=2.2"]
forecast-tfm = ["tabpfn>=3.0", "tabicl>=2.0", "torch>=2.4"]
forecast-graph = ["torch>=2.4", "torch-geometric>=2.6"]
forecast-conformal = ["crepes>=0.7"]  # or MAPIE; decide in Task 5.1
```

Install per phase: `uv sync --extra forecast` etc. CI runs core tests always, extras-gated tests behind markers (`@pytest.mark.forecast_tfm`, registered in `pyproject.toml`).

---

## Phase 0 — Evaluation Harness & Scoreboard (Tasks 0.1–0.6, ~2 weeks)

### Task 0.1: Forecaster protocol + package scaffold
- Create `src/engram/forecasting/__init__.py`, `src/engram/forecasting/protocol.py`:

```python
class Forecaster(Protocol):
    name: str
    def fit(self, train_rows: list[dict[str, Any]]) -> None: ...
    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]: ...  # bucket -> prob
```

- Adapter wrapping existing `BaselineForecaster` (its `predict()["probabilities"]` already matches).
- **Tests:** `tests/unit/test_forecasting_protocol.py` — adapter conforms; probabilities sum to 1; all `DelinquencyBucket` values covered.

### Task 0.2: Metrics module
- `src/engram/forecasting/metrics.py`: multiclass Brier (move logic out of `BaselineForecaster.backtest`), log-loss, top-1 accuracy, ECE (10-bin), reliability-curve data, one-vs-rest AUC per transition, and `loss_weighted_error` with a cost matrix (config-driven; default: cost ∝ bucket distance, `current→d90_plus` most expensive).
- Pure numpy; no new deps.
- **Tests:** `test_forecasting_metrics.py` — known-answer tests (hand-computed Brier/ECE on 3-row fixtures), degenerate cases (single class, empty).

### Task 0.3: Walk-forward splitter + record-time replay
- `src/engram/forecasting/splits.py`:
  - `walk_forward_windows(rows, n_windows, step_months)` → list of (train, eval) row sets by `as_of`; delegates leakage checks to existing `validate_no_leakage`.
  - `record_time_filter(rows, as_of)` — for graph-sourced rows, drop any feature derived from events with `recorded_from > as_of`. For current NDJSON rows this is `as_of`-based; the graph path lands in Task 2.1. Interface defined now so every head builds against it.
  - Origination-cohort splitter (by first-seen year) for vintage-shift evaluation.
- **Tests:** `test_forecasting_splits.py` — window boundaries exact; a row never appears in its own training window; cohort splits partition cleanly.

### Task 0.4: Leakage canary
- `src/engram/forecasting/canary.py`: `run_leakage_canary(forecaster_factory, rows)` — clones the dataset, injects the *label* into a feature column, retrains; asserts score improves massively (canary detects that the harness *would* catch leakage), then runs the real dataset with shuffled-future features and asserts score does **not** improve.
- Wire into CI as `@pytest.mark.slow`.
- **Tests:** canary flags a deliberately leaky forecaster fixture; passes the honest baseline.

### Task 0.5: Baseline ladder
- `src/engram/forecasting/baselines.py`:
  - `HazardForecaster` — discrete-time multinomial logistic hazard (sklearn `LogisticRegression`, features from `extract_features_from_event_history`).
  - `GBMForecaster` — LightGBM multiclass with modest fixed hyperparameters + one tuned config (document the search).
- Requires `--extra forecast`.
- **Tests:** `test_forecasting_baselines.py` — both conform to protocol; beat uniform on synthetic data with known transition structure; deterministic under fixed seed.

### Task 0.6: Harness runner + scoreboard artifact
- `scripts/data_collection/run_forecast_harness.py`: runs N registered forecasters over walk-forward windows, emits `outputs/results/forecast_scoreboard_v{n}.json` (per-model, per-window, per-metric + calibration curves) and a markdown summary table.
- Registry pattern: `--models baseline,hazard,gbm`.
- **Gate 6 artifact:** scoreboard on frozen Ginnie fixture reproduces the existing `track_b_forecast_v1.json` Brier for `baseline` within ±0.001; canary green. Write `docs/plans/decisions/track-b-gate-6-decision.md`.

**Exit criteria Phase 0:** `uv run pytest tests/unit -q` green; scoreboard runs end-to-end on existing `events.ndjson`; GBM and hazard numbers on the board.

---

## Phase 1 — TFM Head (Tasks 1.1–1.5, ~3–4 weeks)

### Task 1.1: Feature table builder
- `src/engram/forecasting/features.py`: `rows_to_matrix(rows, feature_config)` — canonical row dicts → (X, y, feature_names) with categorical encoding map persisted (JSON) for reproducibility. Includes `extract_features_from_event_history` outputs when event history is supplied.
- **Tests:** round-trip stability; unseen-category handling; missing-value passthrough (TFMs handle NaN natively — do not impute).

### Task 1.2: Stratified in-context sampler
- `src/engram/forecasting/icl_sampler.py`: `sample_context(rows, budget, strategy)` where strategy ∈ {`random`, `class_balanced`, `transition_balanced`, `recency_weighted`}. Transition-balanced over-samples rare moves (current→d60+, cure events) — the whole point of TFMs here is exploiting balanced small contexts.
- **Tests:** budget respected; per-class minimums met; deterministic under seed.

### Task 1.3: `TFMForecaster`
- `src/engram/forecasting/tfm.py`: wraps TabPFN-3 and TabICL v2 behind the protocol (`model=` constructor arg). Batched prediction; context re-sampled per walk-forward window (never across window boundary).
- Requires `--extra forecast-tfm`; tests behind `forecast_tfm` marker with a tiny synthetic fixture (CI-safe, CPU, <30s).
- **Tests:** protocol conformance; context never contains eval rows (assert via message_id intersection).

### Task 1.4: Distillation
- `src/engram/forecasting/distill.py`: `distill(teacher, student, rows)` — fit student (GBM/MLP) on teacher's predicted probabilities (soft labels, temperature param). Output: production-cost scorer.
- **Tests:** student Brier within ε of teacher on held-out synthetic data.

### Task 1.5: Graph-feature ablation + Gate 7
- Extend `track_b_graph_features.py` with the docstring's promised graph-derived features (entity degree, relationship-type counts, supersession counts for the loan's neighborhood) behind `graph_features(loan_id, as_of, driver)` — record-time filtered via Task 0.3 interface.
- Harness run: `{tfm, tfm+graph_features, gbm, gbm+graph_features}` × walk-forward.
- **Gate 7 artifact + decision doc:** TFM beats tuned GBM on walk-forward Brier; graph features ≥3% relative Brier improvement. Kill/rescope rules as in roadmap §Phase 1.

---

## Phase 2 — Graph-Native Head (Tasks 2.1–2.4, ~4–6 weeks)

### Task 2.1: Record-time-correct graph export
- `src/engram/forecasting/graph_export.py`: Cypher → edge-list snapshots `(head, rel, tail, valid_from, recorded_from)` with `recorded_from <= as_of` filter (this is the leakage-clean TKG claim — enforce in the query, test it hard).
- Export format: torch-geometric-ready NDJSON + entity/relation vocab files under `outputs/graph_exports/`.
- **Tests:** `test_graph_export.py` — synthetic graph fixture: a fact recorded after `as_of` never appears; supersession chains export correctly.

### Task 2.2: Track B events into the graph
- `scripts/data_collection/ingest_track_b_events.py` already exists — extend to write loan→pool→state-transition edges natively (currently events are text-ingested). Deterministic IDs per existing `{tenant}:{group}:{type}:{name}` scheme.
- **Tests:** idempotent re-ingest (existing Redis dedup path); edge counts match event counts on fixture.

### Task 2.3: ULTRA head
- `src/engram/forecasting/graph_head.py`: `UltraForecaster` — load pretrained ULTRA checkpoint (via `forecast-graph` extra + vendored inference wrapper), zero-shot link prediction scores for `(loan, transitions_to, bucket_state)` candidate edges → normalized to protocol probabilities. Fine-tune variant flag.
- Evaluate against a temporal-GNN alternative only if zero-shot shows signal (cheap-first sequencing).
- **Tests:** protocol conformance on exported fixture graph; scores vary with graph structure (ablate edges → scores move).

### Task 2.4: Ensemble v1 + Gate 8
- `src/engram/forecasting/ensemble.py`: `LinearPoolEnsemble` (weighted log-linear pool, weights fit on validation window) over registered heads.
- Harness run: `{tfm+gbm}` vs `{tfm+gbm+graph}`.
- **Gate 8 artifact + decision doc:** positive ensemble Brier delta from the graph head, per roadmap. Kill → park until Phase 4 deal graphs, documented.

---

## Phase 3 — Agentic LLM Branch Forecaster (Tasks 3.1–3.6, ~5–6 weeks)

Builds on `docs/plans/2026-04-01-branch-forecasting-v0.md` scaffolding and existing H1/H2/H4 modules.

### Task 3.1: Branch space + contracts
- `src/engram/forecasting/branches.py`: Pydantic models `Branch` (target bucket or 2-step chain per H3 primitives), `BranchForecast` (branch, prob, evidence refs, flip_conditions), `ScenarioNode` for tree rollouts.
- **Tests:** schema validation; branch enumeration matches `DelinquencyBucket` × horizon.

### Task 3.2: Evidence assembly (H1+H2 reuse)
- `src/engram/forecasting/evidence.py`: `assemble_evidence(loan_id, as_of, budget)` — `induce_motifs` output selects motif-relevant events; `profile_schema_guided` + budget enforcement shapes the context; returns structured evidence with graph provenance (message_ids).
- **Tests:** budget respected; distractor fixture (from H2 test patterns) excluded by schema-guided profile.

### Task 3.3: LLM rollout engine with test-time compute budget
- `src/engram/forecasting/llm_branch.py`: `LLMBranchForecaster(compute_budget)` — particle-style parallel rollouts via LiteLLM (existing provider config in `config/`), each rollout scores branches; aggregation across particles → branch distribution. Budget knob = n_particles × max_depth; log tokens/cost per prediction into the scoreboard artifact.
- Prompts in `config/prompts/` (existing convention), versioned.
- **Tests:** mocked-LLM determinism test (existing `test_llm_provider.py` mock pattern); budget accounting exact; aggregation math property-tested (particle weights sum to 1).

### Task 3.4: H4 re-ablation on LLM head
- Run `prune_predictions` over LLM branch outputs; re-run the Gate 5 tightness sweep via `run_h4_symbolic_ablation.py` pointed at LLM predictions.
- **Artifact:** `outputs/results/h4_llm_ablation_v1.json` — measures the contradiction rate Gate 5 predicted would appear with LLM models.

### Task 3.5: Supervisor + post-hoc calibration
- `src/engram/forecasting/supervisor.py`: reconciles all heads' distributions per prediction (disagreement features → weighting), then `calibration.py`: isotonic + temperature scaling fit **only on prior walk-forward windows**, plus extremization parameter (fit, not hardcoded).
- **Tests:** calibration never fit on the window being scored (assert by construction); ECE improves on synthetic overconfident inputs.

### Task 3.6: Gate 9
- Full harness run, all heads + supervisor. Cost/latency table per head in the artifact.
- **Gate 9 decision doc:** per roadmap thresholds; demotion path (LLM → explanation duty) is an explicit decision option, not a failure.

---

## Phase 4 — Deal Track (Tasks 4.1–4.7, ~6–10 weeks; 4.1 starts right after Phase 0)

### Task 4.1: ABS-EE + remittance fetchers
- `scripts/data_collection/fetch_edgar_absee.py` (pattern-match existing `fetch_edgar_cmbs.py`): Reg AB II loan-level EX-102 exhibits → `data/edgar/absee/`; manifest entries in `data/manifests/` (existing convention).
- Remittance/trustee report parser for trigger states → `src/engram/services/deal_remit_parser.py` (pattern: `track_b_payhist_parser.py`).
- **Tests:** parser golden-file tests on 2–3 real filings checked into `tests/fixtures/`.

### Task 4.2: Deal spec schema
- `src/engram/models/deal.py`: Pydantic — `DealSpec`, `Tranche` (balance, coupon, seniority), `Trigger` (OC/IC, formula AST, threshold), `Covenant`, `WaterfallStep` (ordered payment rules as a restricted expression language, NOT free Python).
- Bitemporal storage: deal specs are versioned graph objects; amendments supersede via the existing fact-supersession machinery.
- **Tests:** `test_deal_models.py` — round-trip serialization; formula AST rejects unsafe expressions.

### Task 4.3: Waterfall simulator
- `src/engram/forecasting/waterfall.py`: `simulate(deal_spec, collateral_cashflows) -> DealOutcome` (tranche cashflows, trigger states, covenant pass/fail per period). Pure, deterministic, no I/O.
- **Validation:** unit-test against hand-computed toy deals AND against real trustee reports (Task 4.1 data): given actual collateral, simulator must reproduce reported tranche distributions within tolerance. This is the credibility test — budget real time for it.
- **Tests:** `test_waterfall.py` — sequential-pay, pro-rata, trigger-flip scenarios; conservation of cash (inflow = outflow + retained, exact).

### Task 4.4: Deal-doc extraction pipeline
- `src/engram/forecasting/deal_extract.py`: LLM extraction (LiteLLM, prompts in `config/prompts/deal/`) from offering docs → `DealSpec` candidates with per-field confidence + source spans. Human verification: emit a review markdown per deal (`outputs/deal_review/{deal_id}.md`) — approved specs get `verified: true` before the simulator will accept them (hard check).
- **Metric:** field-level extraction accuracy on a 10-deal hand-labeled set. Gate on ≥95% for waterfall-critical fields (tranche sizes, trigger thresholds).
- **Tests:** mocked-LLM parse tests; simulator refuses unverified specs.

### Task 4.5: Collateral scenario generator
- `src/engram/forecasting/scenarios.py`: sample collateral paths from Phase 1–3 heads' loan-level distributions (correlated via systematic macro factor); macro paths from a TSFM adapter `src/engram/forecasting/macro.py` (Chronos-2 zero-shot on rates/HPI; `forecast-tfm` extra covers torch).
- **Tests:** path statistics match head marginals; seedable.

### Task 4.6: Deal outcome engine
- `src/engram/forecasting/deal_engine.py`: `predict_deal(deal_spec, as_of, n_paths, compute_budget)` → P(trigger breach by horizon), P(tranche loss), expected waterfall path — Monte Carlo over Task 4.5 scenarios through Task 4.3 simulator.
- Output conforms to a deal-level `Forecaster`-analog protocol so the harness scores it.
- **Tests:** degenerate deals (single tranche, no triggers) analytic-match; more paths → converging estimates.

### Task 4.7: Deal eval + Gate 10
- Eval targets: DBRS rating transitions (`data/dbrs/cmbs`), trigger breaches from remittance data.
- Comparators: rating-agency transition matrix baseline; flat-LLM-direct-prediction baseline (same LLM, no simulator).
- **Gate 10 decision doc** per roadmap.

---

## Phase 5 — Conformal & Trust (Tasks 5.1–5.3, ~3 weeks, overlaps 3–4)

### Task 5.1: Conformal wrapper
- `src/engram/forecasting/conformal.py`: split-conformal prediction sets over bucket probabilities (per walk-forward window); weighted conformal (likelihood-ratio weights on vintage covariates) for shift; time-to-event conformal bands for D90+/breach timing. Library decision (crepes vs MAPIE vs hand-rolled ~200 lines) recorded in the task PR — hand-rolled is acceptable, the math is small.
- **Tests:** empirical coverage on synthetic data within [target − 2%, target + 2%] over 1000 trials; weighted variant maintains coverage under injected covariate shift.

### Task 5.2: Prediction output contract
- `src/engram/forecasting/output.py`: `PredictionReport` Pydantic model — calibrated probs, conformal set, evidence chain (graph message_ids), flip conditions (`compute_evidence_gaps` reuse), model attribution (which heads, what weights), cost. This is the single public output type; API endpoint `POST /forecast` in the FastAPI app returns it.
- **Tests:** contract completeness; API integration test in `tests/integration/`.

### Task 5.3: Drift & calibration monitoring
- Extend harness with rolling-window ECE/coverage tracking; `scripts/data_collection/run_calibration_audit.py` emits quarterly gate-style report to `docs/plans/decisions/` (auto-drafted, human-approved).
- **Tests:** alarm fires on synthetic drift fixture; silent on stationary fixture.

---

## Standing Rules (all phases)

1. **TDD in repo style:** every task lands `tests/unit/test_*.py` first, `uv run pytest tests/unit -q` green before merge; `ruff` + `mypy` clean (existing configs).
2. **No head without a scoreboard entry.** A model that isn't registered in the harness doesn't exist.
3. **Record-time discipline:** any feature or context assembled for a prediction at `as_of` must pass through `record_time_filter` (Task 0.3). Code review checklist item.
4. **Gate decision docs** in `docs/plans/decisions/track-b-gate-{n}-decision.md`, same format as gates 1–5 (thresholds, observed, kill-condition check, direction change).
5. **Artifacts** to `outputs/results/*.json`, versioned suffix, `generated_at` stamped (existing convention).
6. **Cost tracking:** every LLM-touching component logs tokens + $ into its artifact. Gate 9/10 decisions require the cost column.

## Immediate Start (today)

```bash
uv sync
uv run pytest tests/unit -q                      # confirm green baseline
mkdir -p src/engram/forecasting
# Task 0.1: protocol + adapter + tests
uv run pytest tests/unit/test_forecasting_protocol.py -q
```

## Dependency / Risk Notes

- **TabPFN-3/TabICL licensing + API surface**: verify current license terms permit our MIT-core + commercial-layer model before Task 1.3; fallback is TabICL (research-friendly) or distilled-only deployment.
- **ULTRA checkpoint compat**: pin torch/PyG versions in `forecast-graph` extra; vendored wrapper isolates us from upstream churn.
- **CI weight**: extras-gated markers keep core CI fast; nightly job runs `forecast*` markers.
- **Neo4j Community vector/index limits at deal scale**: revisit `ARCHITECTURE.md §How does it scale` when deal graphs land; not a Phase 0–3 concern.
