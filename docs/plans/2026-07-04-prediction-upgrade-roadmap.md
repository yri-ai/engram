# Prediction Upgrade Roadmap: From Transition Matrix to Deal-Outcome Engine

**Date:** July 4, 2026
**Status:** Proposed
**Goal:** Make Engram's prediction components state-of-the-art, with the end target of predicting structured finance and complex deal outcomes (RMBS/CMBS/CLO performance, trigger breaches, covenant defaults, sponsor stress, workout paths).

---

## 1. Where We Are (honest baseline)

| Component | Current state | Verdict |
|---|---|---|
| `BaselineForecaster` (`track_b_forecasting.py`) | First-order transition matrix P(next_bucket \| current_bucket), ignores all features | Yardstick only. Uses none of Engram's graph, none of the loan covariates |
| `track_b_graph_features.py` | Event-history features (lookback windows) exist; true graph-derived features (entity/relationship structure) are a stated TODO in the module docstring | Underexploited — the graph is Engram's moat and isn't feeding predictions yet |
| H1 schema induction | Validated: compact, transferable precursor schemas | Keep as evidence-selection layer |
| H2 minimal-context | Validated: schema-guided small context beats big context | Keep — directly matches 2025-26 findings on LLM forecasters |
| H3 transition-first | Validated: next-state > terminal-event prediction | Keep as core primitive |
| H4 symbolic pruning | NO-GO as primary (Gate 5): base model already consistent | Re-activates in Phase 3+ where LLMs *do* produce contradictions, and in Phase 4 where deal rules are hard constraints |
| Data | Ginnie Mae loan-months, Fannie, EDGAR CMBS, DBRS fetchers | Good spine; needs deal-level layers |

The core gap: **the current forecaster is a counting model.** Everything below is about replacing counting with (a) foundation-model priors, (b) graph-native reasoning, and (c) agentic LLM forecasting with disciplined uncertainty — while keeping the repo's gate/kill-criteria culture.

---

## 2. What the 2025–2026 frontier says (research summary)

1. **Tabular foundation models (TFMs) now beat tuned GBMs on exactly our problem class.** TabPFN v2 was the first TFM to outperform tuned gradient-boosted trees ([Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)); TabPFN-3 ([technical report](https://priorlabs.ai/technical-reports/tabpfn-3), [arXiv:2605.13986](https://arxiv.org/abs/2605.13986)) and TabICL v2 (Feb 2026) extend scale. A 2026 credit-risk benchmark found TabICL ranks first for PD classification and TabPFNv2 first for LGD regression, ahead of tuned GBMs — with the advantage strongest on **small/low-default portfolios and specialized segments** ([Foundation Models for Credit Risk Prediction](https://arxiv.org/pdf/2605.18147), [Mission Lane benchmark](https://medium.com/mission-lane-tech-blog/tabicl-under-the-microscope-benchmarking-tabular-foundation-models-for-enterprise-credit-risk-ad8315f9bec4)). Crédit Agricole's Creditplus is already deploying **distilled TabPFN** for regulated credit decisioning. Structured finance deals are exactly the small-N, low-default regime where TFMs shine.

2. **Temporal knowledge graph forecasting has converged on hybrid structure+LLM.** SOTA combines GNN structural modeling with LLM semantic reasoning: entity-state tuning ([arXiv:2602.12389](https://arxiv.org/pdf/2602.12389)), LLM re-ranking over GNN candidates (SKER), mixture-of-graph-experts fast/slow thinking (FMOGE), rule-confidence learning ([CountTRuCoLa](https://arxiv.org/pdf/2509.09474)), and structured-reasoning TKG completion ([RECIPE-TKG](https://arxiv.org/pdf/2505.17794)). Separately, **ULTRA-class KG foundation models** do zero-shot link prediction on *any* relational graph, beating per-graph trained models ([arXiv:2310.04562](https://arxiv.org/abs/2310.04562)) — meaning Engram's bitemporal graph can get learned relational priors without training from scratch.

3. **LLM forecasting agents reached superforecaster parity in late 2025.** The AIA Forecaster hit Brier 0.0753 vs 0.0740 human SOTA on ForecastBench-Market via three ingredients: **agentic evidence search, a supervisor agent reconciling an ensemble of forecasts, and statistical post-hoc calibration/extremization** ([AIA technical report](https://arxiv.org/html/2511.07678v1), [ForecastBench tracking](https://forecastingresearch.substack.com/p/ai-llm-forecasting-model-forecastbench-benchmark)). Caveats that matter for us: economic/financial questions are the weakest topic area, and raw LLM probabilities are overconfident without calibration ([Prediction Arena real-money results](https://arxiv.org/pdf/2605.00420)). Sequential Bayesian updating of linguistic beliefs is emerging for streaming-evidence settings ([arXiv:2604.18576](https://arxiv.org/html/2604.18576)) — a natural fit for Engram's bitemporal "what did we know when" model.

4. **Test-time compute scaling works for structured prediction.** Particle-based Monte Carlo inference gets 4–16x better scaling than deterministic search ([arXiv:2502.01618](https://arxiv.org/html/2502.01618v3)); adaptive parallel MCTS ([arXiv:2604.00510](https://arxiv.org/pdf/2604.00510)) and budget-controlled reasoning ([survey](https://arxiv.org/html/2507.02076v1)) make scenario-tree exploration affordable. This is the natural engine for the branch-forecasting v0 design already in `docs/plans/2026-04-01-branch-forecasting-v0.md`.

5. **Uncertainty quantification is now deployable, not academic.** Censoring-aware conformal survival methods give distribution-free coverage guarantees for exactly our right-censored loan/deal data: weighted conformal under covariate shift ([arXiv:2512.03738](https://arxiv.org/pdf/2512.03738)), conformal survival bands for risk screening ([arXiv:2505.04568](https://arxiv.org/pdf/2505.04568)), doubly-robust conformalized survival (ICML 2025).

6. **Multi-state modeling has gone neural.** Semi-structured multi-state delinquency models combine hazard structure with neural components ([arXiv:2603.26309](https://arxiv.org/html/2603.26309)); MENSA does multi-event survival with trajectory likelihoods ([arXiv:2409.06525](https://arxiv.org/pdf/2409.06525)); survNODE parametrizes Kolmogorov forward equations with neural ODEs. These give the *probabilistically correct* scaffold that a transition matrix approximates crudely.

7. **Deal-document intelligence is production-grade.** LLM agents now extract covenants, EBITDA definitions, add-back baskets, and trigger levels from 90-page credit agreements into structured form (V7, Harvey, Octus/Covenant Review). For deal-outcome prediction, the deal's *contractual mechanics* (waterfall, triggers, covenants) are deterministic once extracted — a symbolic simulator, not a learned model. This is where H4's symbolic layer stops being a safety net and becomes load-bearing.

8. **Time-series foundation models cover the macro leg.** Chronos-2 and TimesFM-class models give SOTA zero-shot probabilistic forecasts for rate/HPI/unemployment covariates ([2026 TSFM benchmarks](https://www.mdpi.com/2813-0324/11/1/32)); multimodal joint language+time-series models (Chronicle, [arXiv:2605.20268](https://arxiv.org/pdf/2605.20268)) are the forward-looking bet.

---

## 3. Target architecture

```
                        ┌────────────────────────────────────────────┐
                        │  LAYER 4: DECISION HEAD                    │
                        │  calibrated probabilities + conformal      │
                        │  intervals + "what would flip this" gaps   │
                        └───────────────▲────────────────────────────┘
                                        │ supervisor agent (reconcile + extremize + calibrate)
        ┌───────────────┬───────────────┼────────────────┬───────────────────┐
        │               │               │                │                   │
┌───────┴──────┐ ┌──────┴───────┐ ┌─────┴──────┐ ┌───────┴────────┐ ┌────────┴────────┐
│ LAYER 3a     │ │ LAYER 3b     │ │ LAYER 3c   │ │ LAYER 3d       │ │ LAYER 3e        │
│ TFM head     │ │ Graph head   │ │ LLM branch │ │ Hazard head    │ │ Deal simulator  │
│ TabPFN-3 /   │ │ ULTRA-class  │ │ forecaster │ │ semi-struct.   │ │ waterfall +     │
│ TabICL ICL   │ │ + temporal   │ │ (agentic,  │ │ multi-state,   │ │ trigger logic   │
│ on loan/deal │ │ GNN over     │ │ test-time  │ │ neural ODE     │ │ (symbolic,      │
│ features     │ │ bitemporal   │ │ MC scenario│ │ competing      │ │ deterministic)  │
│              │ │ graph        │ │ trees)     │ │ risks          │ │                 │
└───────▲──────┘ └──────▲───────┘ └─────▲──────┘ └───────▲────────┘ └────────▲────────┘
        │               │               │                │                   │
        └───────────────┴───────┬───────┴────────────────┴───────────────────┘
                        ┌───────┴────────────────────────────────────┐
                        │  LAYER 2: EVIDENCE & FEATURES              │
                        │  H1 precursor schemas → H2 minimal context │
                        │  graph features · macro TSFM forecasts ·   │
                        │  extracted deal terms (covenants/triggers) │
                        └───────────────▲────────────────────────────┘
                        ┌───────────────┴────────────────────────────┐
                        │  LAYER 1: ENGRAM BITEMPORAL GRAPH          │
                        │  loans · deals · tranches · sponsors ·     │
                        │  servicers · triggers · covenants · docs   │
                        └────────────────────────────────────────────┘
```

Key design decisions:

- **Ensemble of heterogeneous heads, one supervisor.** This is the AIA-Forecaster recipe generalized: independent predictors with different inductive biases (in-context tabular prior, relational structure, agentic reasoning, hazard structure, contractual mechanics), reconciled by a supervisor and then *statistically calibrated*. No single model is the bet.
- **The graph is the substrate, not a head.** Engram's bitemporal graph feeds every head: point-in-time correct features (no lookahead leakage — `recorded_from` is the leakage guard, a capability none of the TKG baselines have), precursor-schema evidence selection for the LLM head, and relational structure for the graph head.
- **Symbolic where the world is symbolic, learned where it isn't.** Deal waterfalls, OC/IC triggers, and covenant tests are deterministic given inputs. Simulate them exactly (Layer 3e); spend learned capacity only on the uncertain inputs (defaults, prepayments, recoveries, sponsor behavior).

---

## 4. Phased plan

### Phase 0 — Scoreboard & leakage-proof evaluation harness (2 weeks)

Nothing below is trustworthy without this. Build once, reuse for every gate.

- Extend `run_track_b_backtest.py` into an evaluation harness: temporal walk-forward splits (train ≤ t, predict t+h), origination-cohort splits, and **record-time replays** ("predict using only what was `recorded_from` ≤ t" — Engram's bitemporal model makes this trivially auditable; it is the differentiator).
- Metrics: multiclass Brier (already there), log-loss, calibration curves + ECE, AUC per transition, and **economic metrics** (loss-weighted error: missing a current→D90 jump costs more than D30→current noise).
- Baseline ladder as yardsticks only: existing transition matrix → covariate-conditioned discrete-time hazard → tuned LightGBM. These exist to be beaten and to catch leakage (if a fancy model beats GBM by 30 points, suspect leakage first).
- **Gate 6 entry criterion:** harness reproduces current Brier on frozen fixtures; leakage canary test (shuffle future into features → score must *not* improve) passes.

### Phase 1 — Tabular foundation model head (3–4 weeks)

The fastest large win. Replace counting with in-context priors.

- Add `TFMForecaster` alongside `BaselineForecaster` behind a common protocol (`fit/predict/backtest` signature already exists).
- TabPFN-3 and TabICL v2 for next-bucket classification on loan-month features + `track_b_graph_features` output. Context-window strategy for >50K rows: stratified in-context sampling per prediction batch (over-sample rare transitions — TFMs' small-data advantage means we can feed *balanced* contexts), plus a **distilled student** (TabPFN → GBM/MLP distillation, the Creditplus pattern) for full-portfolio scoring at production cost.
- Ablation: features-only vs features+graph-features. This is the first quantitative test of whether Engram's graph adds predictive lift over flat features.
- **Gate 7:** TFM head beats tuned LightGBM on walk-forward Brier AND graph features contribute ≥ measurable lift (target: ≥3% relative Brier improvement from graph features). Kill: if graph features add nothing here, the graph-head phase (2) gets re-scoped before build.

### Phase 2 — Graph-native head on the bitemporal graph (4–6 weeks)

- Represent Track B (and later deals) natively in the graph: loan→pool→deal→tranche edges, servicer/sponsor entities, state transitions as temporal edges with `valid_from/recorded_from` already in schema.
- Two candidates, evaluated against each other:
  - **ULTRA-class KG foundation model** zero-shot/fine-tuned on the Engram graph for transition-event link prediction (no per-graph training; matches our multi-tenant, many-small-graphs reality).
  - **Temporal GNN with entity-state tuning** (the 2026 TKG SOTA pattern): learn evolving entity states over the event stream, condition transition predictions on them.
- Use record-time replay from Phase 0 to guarantee the GNN never sees future edges — this makes Engram one of the only leakage-clean TKG forecasting setups in existence, worth publishing on its own.
- **Gate 8:** graph head adds ensemble lift over Phase 1 alone (positive Brier delta in ensemble, not just solo performance). Kill: if relational structure adds nothing on loan-level data, park the graph head until deal-level data (Phase 4) where relational structure is richer (sponsor↔deal↔tranche webs), and say so explicitly.

### Phase 3 — Agentic LLM branch forecaster with test-time compute (5–6 weeks)

This upgrades `branch-forecasting-v0` from deterministic scaffold to the AIA recipe.

- **Evidence layer:** H1 precursor schemas select the branch-relevant subgraph; H2 context budgets keep it minimal. (Both already validated — they were built for exactly this.)
- **Branch enumeration:** transition-first primitives (H3) define the branch space (next state / 2-step chains), not open-ended narrative.
- **Scenario exploration:** particle-based Monte Carlo / adaptive MCTS over branch trees with an explicit compute budget knob; each rollout produces a probability-annotated path. Sequential Bayesian updating of branch beliefs as new events arrive (fits the ingest-stream architecture; the Bayesian update shell in branch-forecasting v0 becomes real).
- **Supervisor + calibration:** reconcile LLM branch probabilities with Phase 1/2 heads; post-hoc extremization + isotonic/Platt calibration fit on walk-forward history. Raw LLM probabilities are never surfaced.
- **H4 symbolic layer re-activation:** LLM heads *will* produce temporally inconsistent branches (unlike the transition matrix). The Gate 5 decision explicitly predicted this: "the 0.15 contradiction threshold should be re-evaluated when LLM-based models replace the transition matrix." Re-run the H4 ablation against the LLM head.
- Output contract per prediction: probability distribution + top evidence + **"what evidence would flip this"** (the competing-cause discipline from Workstream 2).
- **Gate 9:** ensemble with LLM head beats Phase 1+2 ensemble on Brier; H4 re-ablation measured; cost/latency budget documented. Kill: if LLM head only matches TFM head at 100x cost, demote it to explanation-generation duty (evidence narratives on top of TFM/graph predictions) rather than probability duty.

### Phase 4 — Structured finance deal track (6–10 weeks, parallel-start after Phase 1)

The destination. Two sub-tracks:

**4a. Deal data layer**
- Extend existing fetchers: EDGAR **ABS-EE loan-level exhibits** (Reg AB II — machine-readable loan tapes for RMBS/CMBS/auto), CMBS via existing `fetch_edgar_cmbs.py`, trustee/remittance report ingestion, DBRS/rating-action history (fetcher exists).
- **Deal-document extraction pipeline:** LLM agents extract the symbolic deal spec from offering docs/indentures — tranche structure, waterfall rules, OC/IC trigger definitions and levels, covenant tests, EBITDA definitions and carve-outs. Store as versioned symbolic objects in the graph (bitemporal: amendments supersede, exactly like facts do today). Human-in-the-loop verification UI for extracted terms; extraction accuracy is a gated metric, not an assumption.

**4b. Deal outcome engine**
- **Deterministic waterfall/trigger simulator** (pure Python, unit-tested against trustee reports): given collateral cashflows → tranche cashflows, trigger states, covenant pass/fail. This is H4 symbolic reborn as the core engine where the world really is symbolic.
- Collateral scenario inputs come from the Phase 1–3 heads (default/prepay/severity distributions) + **TSFM macro conditioning** (Chronos-2/TimesFM zero-shot rate/HPI paths as covariate scenarios).
- Deal outcome = distribution over simulator outputs across sampled collateral paths: P(OC trigger breach by Q4), P(tranche writedown), expected workout path. Predictions are *mechanically consistent* by construction — no learned model can produce an impossible waterfall.
- Deal-level targets for eval: rating transitions (DBRS data), trigger breach events from remittance data, realized tranche losses.
- **Gate 10:** on a held-out deal cohort, engine beats (i) rating-agency transition matrices and (ii) a flat LLM asked to predict deal outcomes directly. Kill criteria per component, not whole-track.

### Phase 5 — Uncertainty, trust, and decision-grade output (3 weeks, overlaps 3–4)

- **Censoring-aware conformal prediction** on all heads: conformal survival bands for time-to-event (delinquency/trigger breach), weighted conformal under covariate shift (essential — origination vintages shift constantly). Guaranteed-coverage prediction sets, not just point probabilities.
- Continuous calibration monitoring in the harness: rolling ECE, coverage audits, drift alarms wired into the existing gate-decision culture (auto-generate a gate-style report per quarter).
- Every prediction ships with: calibrated probability, conformal set, evidence chain (graph paths), and flip conditions. This is the difference between "a model score" and something a credit committee can act on.

### Phase 6 — Future tech bets (ongoing, 10–20% time)

- **RL fine-tuning on forecasting rewards:** fine-tune an open reasoning model with proper-scoring-rule rewards (Brier as reward signal) on Engram's replayed history — the logical endpoint of test-time-compute forecasting, and record-time replay gives us contamination-free training data (a real, rare asset; cf. data-contamination concerns in TKG evaluation, [arXiv:2601.13658](https://arxiv.org/pdf/2601.13658)).
- **Multimodal language+time-series models** (Chronicle-class) once open checkpoints mature: joint reasoning over remittance narratives and performance curves.
- **Causal world-model induction** over deal graphs: learn intervention-aware structure ("if servicer advances stop, what breaks first") — upgrade path for the branch forecaster.
- Watch: TabICL v3+, ULTRA successors, ForecastBench leaderboard (LLM-superforecaster parity projected late 2026 — we should ride that curve, not rebuild it).

---

## 5. Sequencing & dependencies

```
Weeks:   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
Phase 0  ██ ██
Phase 1        ██ ██ ██ ██
Phase 2                    ██ ██ ██ ██ ██
Phase 3                    ██ ██ ██ ██ ██ ██
Phase 4a          ██ ██ ██ ██ ██ ██ ██
Phase 4b                            ██ ██ ██ ██ ██ ██ ██
Phase 5                                   ██ ██ ██
Phase 6  ─ ─ ─ ─ ─ ─ ─ ─ ─ (background) ─ ─ ─ ─ ─ ─ ─ ─ ─
```

Critical path: Phase 0 → Phase 1 → (2 ∥ 3 ∥ 4a) → 4b → 5. Phase 4a (data + document extraction) starts early because data acquisition is always the long pole.

## 6. Risks

| Risk | Mitigation |
|---|---|
| TFM context limits vs portfolio scale | Stratified ICL sampling + distillation (proven pattern in regulated credit) |
| LLM head cost blowup | Compute-budget knob from day 1; demotion path defined at Gate 9 |
| Deal-doc extraction errors poison the simulator | Extraction is gated + human-verified; simulator unit-tested against trustee reports |
| Financial topics are LLMs' weakest forecasting area | LLM is one head of five, never solo; supervisor + calibration always on |
| Leakage via future knowledge | Record-time replay is mandatory in the harness; leakage canary in CI |
| Overfitting to Ginnie/agency data idiosyncrasies | H5 transfer harness already exists — reuse for cross-asset-class transfer tests |

## 7. What makes this defensible

Every competitor can call TabPFN or an LLM. The compounding advantages here are: (1) **bitemporal record-time replay** — provably leakage-free training and evaluation, which almost nobody else can offer; (2) **precursor-schema evidence selection** — validated H1/H2 machinery that keeps LLM forecasting cheap and robust to distractors; (3) **symbolic deal mechanics** — predictions that respect waterfall/trigger logic by construction; (4) the **gate/kill-criteria culture** — every head earns its place on the scoreboard or gets demoted.

---

## Appendix: Source list

- TabPFN v2: [Nature](https://www.nature.com/articles/s41586-024-08328-6) · TabPFN-3: [report](https://priorlabs.ai/technical-reports/tabpfn-3), [arXiv:2605.13986](https://arxiv.org/abs/2605.13986)
- TFMs for credit risk: [arXiv:2605.18147](https://arxiv.org/pdf/2605.18147) · [Mission Lane TabICL benchmark](https://medium.com/mission-lane-tech-blog/tabicl-under-the-microscope-benchmarking-tabular-foundation-models-for-enterprise-credit-risk-ad8315f9bec4)
- AIA Forecaster: [arXiv:2511.07678](https://arxiv.org/html/2511.07678v1) · ForecastBench status: [FRI substack](https://forecastingresearch.substack.com/p/ai-llm-forecasting-model-forecastbench-benchmark) · real-money caveat: [Foresight/Prediction Arena](https://arxiv.org/pdf/2605.00420)
- Sequential Bayesian linguistic beliefs: [arXiv:2604.18576](https://arxiv.org/html/2604.18576)
- TKG forecasting SOTA: [entity-state tuning, arXiv:2602.12389](https://arxiv.org/pdf/2602.12389) · [RECIPE-TKG](https://arxiv.org/pdf/2505.17794) · [CountTRuCoLa](https://arxiv.org/pdf/2509.09474) · contamination: [arXiv:2601.13658](https://arxiv.org/pdf/2601.13658)
- KG foundation models: [ULTRA, arXiv:2310.04562](https://arxiv.org/abs/2310.04562)
- Test-time compute: [particle MC, arXiv:2502.01618](https://arxiv.org/html/2502.01618v3) · [adaptive MCTS, arXiv:2604.00510](https://arxiv.org/pdf/2604.00510) · [budget survey, arXiv:2507.02076](https://arxiv.org/html/2507.02076v1)
- Conformal survival: [weighted under shift, arXiv:2512.03738](https://arxiv.org/pdf/2512.03738) · [survival bands, arXiv:2505.04568](https://arxiv.org/pdf/2505.04568)
- Multi-state neural: [semi-structured delinquency, arXiv:2603.26309](https://arxiv.org/html/2603.26309) · [MENSA, arXiv:2409.06525](https://arxiv.org/pdf/2409.06525)
- TSFMs: [2026 zero/few/full-shot benchmark](https://www.mdpi.com/2813-0324/11/1/32) · [Chronicle multimodal, arXiv:2605.20268](https://arxiv.org/pdf/2605.20268)
- Schema induction lineage: [complex event schema induction, arXiv:2104.06344](https://arxiv.org/pdf/2104.06344)
