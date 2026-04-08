[Research Lab Memo Engram](Research%20Lab%20Memo%20Engram.md)

**Best-path thesis:**  
Prediction quality will improve if the system reasons over **small induced precursor schemas and branch constraints** rather than large undifferentiated context windows. This is consistent with the schema-induction paper’s staged graph construction, the AER benchmark’s focus on noisy multi-document direct-cause identification, and NeSTR’s symbolic temporal consistency layer.

---

## Workstream 1: Schema induction as the core object

### Hypothesis 1

There exist reusable, structure-specific **precursor schemas** that are short enough to induce automatically and useful enough to outperform flat retrieval-based evidence pools for forecasting near-future transitions. The ACL paper is the best starting point because it explicitly induces hierarchical and temporal event schemas via skeleton construction, event expansion, and relation verification.

### Experiments

1. **Schema discovery by domain family**
    - Choose 3–5 heterogeneous domains, not just public securities.
    - Good set:
        - structured product unwind / trigger risk
        - private credit deterioration
        - sponsor / financing stress
        - litigation escalation
        - operational narrative breakdown
    - For each, induce schemas from mixed corpora and compare whether the schemas are:
        - compact
        - reusable across cases
        - interpretable by humans
2. **Schema granularity experiment**
    - Compare:
        - event-only schemas
        - event + latent state schemas
        - event + state + gate schemas
    - Goal: find the minimum abstraction level that still captures unfolding structure.
3. **Schema transfer experiment**
    - Induce on one subset of cases, test on new cases in the same structural family.
    - Measure whether the induced schema still explains unfolding chains without heavy manual patching.

### Success criteria

- Humans can recognize recurring precursor motifs across cases.
- Schemas remain compact rather than exploding into giant story graphs.
- Induced schemas transfer within a structural family with only minor edits.

### Kill criteria

Kill or downgrade schema induction as the primary path if:

- schemas become case-specific storytelling rather than reusable blueprints
- human review says they are descriptive but not decision-relevant
- transfer across same-family cases is poor
- schema size balloons with little gain

---

## Workstream 2: Minimal-context abductive reasoning

### Hypothesis 2

For this class of problems, **more context hurts** unless the system is forced to isolate direct causal evidence and discriminate among a small number of competing explanations. The AER benchmark was built around exactly this failure mode: multi-document noise, indirect background factors, and semantically similar distractors.

### Experiments

1. **Context budget sweep**
    - Run the same tasks under:
        - full corpus context
        - retrieved top-K passages
        - schema-guided evidence only
        - minimal discriminative evidence only
    - Measure whether performance peaks at a smaller evidence frontier.
2. **Distractor stress test**
    - Add semantically similar but non-causal distractors.
    - Test whether systems with schema-guided or abductive filtering degrade less than plain retrieval or plain long-context prompting.
3. **Competing-cause experiment**
    - Present two or three plausible direct-cause chains.
    - Ask the system not only to choose one, but to identify:
        - why the chosen branch dominates
        - what missing evidence would flip the choice

### Success criteria

- Smaller context windows beat or match larger ones.
- Schema-guided evidence selection is more robust to distractors.
- The system can surface decisive evidence gaps, not just overconfident answers.

### Kill criteria

Kill or downgrade the minimal-context thesis if:

- performance consistently improves with more context
- distractor robustness does not improve with schema-guided filtering
- the model cannot distinguish direct causes from related background factors

---

## Workstream 3: Transition-first prediction

### Hypothesis 3

The best primitive is not terminal event prediction but **next-state or short-chain prediction**. That aligns with your critique and is more compatible with induced schemas than endpoint labels.

### Experiments

1. **Predictive primitive comparison**  
    Compare four targets on the same corpora:
    - terminal event
    - next state transition
    - next two-step chain
    - branch ranking among plausible futures
2. **Chain-length sensitivity**
    - Test 1-step, 2-step, 3-step, 4-step predictions.
    - Look for the horizon where useful structure exists before uncertainty collapses.
3. **Observed vs latent transition experiment**
    - Compare models that only use explicit events vs models allowed to infer latent intermediate states.

### Success criteria

- Transition or short-chain prediction outperforms direct endpoint prediction on calibration, robustness, and interpretability.
- There is a stable useful horizon, likely 1–3 steps.

### Kill criteria

Kill or downgrade transition-first as the center if:

- short-chain prediction is no better than endpoint prediction
- latent state inference adds only narrative complexity, not measurable benefit
- chain length beyond one step collapses into noise across domains

---

## Workstream 4: Neuro-symbolic pruning, but only as discipline

### Hypothesis 4

A symbolic temporal consistency layer helps by **pruning impossible futures and resolving contradictions**, but should not be the main engine. NeSTR is relevant here because it integrates symbolic temporal representations with iterative abductive correction, but it is fundamentally a temporal reasoning framework, not a full domain-forecasting solution.

### Experiments

1. **With-vs-without symbolic constraints**
    - Same abductive or transition tasks
    - Compare:
        - pure LLM reasoning
        - LLM + symbolic temporal consistency checks
    - Measure contradiction rate and branch-pruning quality.
2. **Constraint tightness experiment**
    - Loose symbolic constraints
    - medium constraints
    - hard pruning
    - Goal: find whether symbolic layers improve discipline without over-brittleness.
3. **Failure analysis**
    - Track whether symbolic rules prune genuine novelty or only bad reasoning.

### Success criteria

- Fewer logically inconsistent chains
- Better branch elimination
- Better reliability under ambiguous temporal evidence

### Kill criteria

Kill or strictly narrow this path if:

- symbolic pruning mainly removes valid edge cases
- gains are small compared with simpler evidence-filtering methods
- maintenance burden becomes too high for open-world domains

---

## Workstream 5: Structure-specific grammars

### Hypothesis 5

Different structures and instruments have different **transition grammars**. A single unified schema language may be too coarse. This is the core extension beyond public-securities thinking.

### Experiments

1. **Cross-structure comparison**
    - Compare schemas and transition motifs across:
        - public corporate narratives
        - private credit
        - structured products
        - real-estate capital stacks
    - Ask whether the same abstract grammar works or whether each needs its own motif family.
2. **Shared-core vs family-specific schema**
    - Test:
        - one universal schema language
        - one core shared layer + family-specific motifs
    - My bet is the second wins.
3. **Amplification/gating study**
    - For each family, identify whether outcomes are driven more by:
        - threshold gates
        - waterfall logic
        - optionality collapse
        - counterpart behavior
        - covenant or contract triggers

### Success criteria

- Clear family-specific motif clusters emerge.
- A shared-core + family-specific extension outperforms one universal frame.

### Kill criteria

Kill the “one grand theory” path if:

- families diverge too much for a shared ontology to remain meaningful
- mapping everything into one schema degrades explanatory power

---

## Suggested phase order

### Phase 1: Find the predictive object

Do Workstreams 1, 2, and 3 first.

This phase answers:

- Are schemas real and reusable?
- Does minimal context beat large context?
- Is transition/chain prediction better than endpoint prediction?

If these three do not work, stop there.

### Phase 2: Add discipline

Only then run Workstream 4.

This phase answers:

- Do symbolic constraints help or just make the system brittle?

### Phase 3: Generalize across structures

Then run Workstream 5.

This phase answers:

- Can this become a cross-domain science rather than a niche trick?

---

## Hard kill thresholds

Here are the non-negotiable stop signs.

Kill this research direction, or radically narrow it, if after Phase 1:

1. **Schemas are not reusable.**
    - If induced schemas mostly redescribe single cases, stop calling them schemas.
2. **Minimal context does not help.**
    - If full context consistently wins, your anti-context thesis is wrong or incomplete.
3. **Transition prediction is not more stable than endpoint prediction.**
    - If the transition-first frame gives no gain, the core conceptual shift is not pulling its weight.
4. **Cross-domain structure collapses.**
    - If every structural family needs a totally separate logic with no shared patterns, then you do not have one research program; you have multiple narrower ones.
5. **Symbolic pruning adds maintenance pain without reliability gains.**
    - Then it is academic garnish.

---

## Deliverables from the agenda

At the end of this program, you should have:

- a **taxonomy of predictive primitives**
- a **library of precursor schemas / motifs**
- evidence on whether **minimal discriminative context** is real
- evidence on whether **transition-first** beats endpoint-first
- a decision on whether neuro-symbolic pruning is useful or a distraction

---

## My blunt recommendation

The first three experiments I would run are:

1. **Schema induction across 3 structural families**
2. **Context budget sweep with heavy distractors**
3. **Event vs transition vs short-chain prediction comparison**

If those do not produce sharp signal, stop romanticizing this path.

If they do, then you have something genuinely newer than 2001 logic systems and newer than generic long-context LLM mush. The papers you highlighted support that ordering: schema induction gives the object, AER gives the stress environment, and NeSTR gives a possible discipline layer later