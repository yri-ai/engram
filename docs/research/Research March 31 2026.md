# Research: Causal Bayesian Information Selection for Temporal Graphs

**Date:** March 31, 2026
**Purpose:** Explore alternatives to RL for branch forecasting — taxonomy of temporal KG reasoning approaches

---

## 1. The Contrastive Expansion: Representation Learning

If we drop the "reward" signal, we replace it with **Information Maximization**. Instead of an agent looking for a payout, we use a system that learns to recognize the "fingerprint" of a causal event chain.

- **Temporal Contrastive Learning (TCL):** We create "positive pairs" of events (a Q1 claim and the subsequent Q3 outcome) and "negative pairs" (the same Q1 claim and a random, unrelated outcome). The model is trained to minimize the distance between positive pairs in a high-dimensional space.
    
- **The Benefit:** This bypasses the "sparse reward" problem. You don't need a specific outcome to learn; the model simply learns that certain event structures are _temporally consistent_ with specific results.
    

---

## 2. The Causal Discovery Approach: SCMs

Instead of predicting relevance, we **discover the graph.** Rather than treating the graph as a fixed input for a GNN, we use **Causal Discovery (CD)** algorithms to find the "Directed Acyclic Graph" (DAG) that explains the data.

- **Constraint-Based Discovery:** Using tests like the **PC Algorithm** or **FCI (Fast Causal Inference)** to prune the graph. If Event A and Outcome C are independent given Event B, we know the "information" flows through B.
    
- **The Formula:** We move from a simple prediction $P(Y | X)$ to an interventionist framework:
    
    $$P(Y | do(X))$$
    
    This allows the system to answer counterfactuals: _"If this Q1 claim hadn't happened, would the Q3 margin miss still occur?"_
    

---

## 3. Neural Temporal Point Processes (TPP)

If the "bitemporal" aspect is the priority, TPPs are the superior alternative to RL. They model the "intensity" of events over time.

- **How it works:** Instead of discrete nodes, every event has an intensity function $\lambda(t)$. A "strategic commitment" in Q1 increases the intensity (probability) of a "narrative shift" in Q2.
    
- **Conditioned Relevance:** You can use **Marked Point Processes** to attach the "objective" (e.g., Margin Target) as a mark, allowing the system to learn which historical intensities are most predictive of that specific mark.
    

---

## Comparison: RL vs. The Expanded Approach

|**Feature**|**Reinforcement Learning (RL)**|**Causal Discovery / SSL Expansion**|
|---|---|---|
|**Feedback Loop**|Requires explicit reward/outcome.|Uses intrinsic data structure.|
|**Data Efficiency**|Needs millions of "episodes."|Highly efficient with sparse data.|
|**Interpretability**|"Black box" policy.|Transparent Causal DAGs.|
|**Delay Handling**|Struggling with credit assignment.|Explicitly models temporal lags.|

---

## 4. The "Stage 2" Realignment

Without RL, your Stage 2 "Contextual Bandit" becomes a **Differentiable Ranking Engine**.

Instead of an agent picking an action, you use a **Cross-Encoder** that scores the "Alignment Score" between the current context graph and the candidate event chain. This score is trained using a **Triplet Loss** function:

> **Score(Context, Causal Event) >> Score(Context, Noise Event)**

This creates a system that is far more stable than an RL agent because the "gradient" is always clear, even if the actual financial outcome hasn't arrived yet.

---



## The Problem With RL Here

RL optimizes for **reward**. But what reward? In recommendation systems, reward is clear — the user clicked, bought, or engaged. In decision support, the "reward" is: _did the user make a better decision because we surfaced this claim?_

That reward is:

- Delayed by weeks or months (deal closes, earnings come in)
- Confounded by everything else the user knew
- Sparse (most claims are neither clearly useful nor clearly noise)
- Subjective (two analysts disagree on what was useful)

You'd spend more time engineering the reward function than building the predictor. And a poorly specified reward function produces a system that games the metric — surfacing claims users _like_ rather than claims they _need_.

That's the core distinction: **recommendation optimizes for preference, decision support optimizes for information value.** Those are fundamentally different objectives, and RL is built for the first one.

## The Expanded Framework: Causal Bayesian Information Selection

Instead of one monolithic RL policy, build a composable system where each subproblem gets the right mechanism:

```
┌─────────────────────────────────────────────────────────┐
│           CAUSAL BAYESIAN INFORMATION SELECTION          │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │   CAUSAL     │  │  BAYESIAN    │  │ INFORMATION   │ │
│  │   DISCOVERY  │  │  BELIEF      │  │ THEORETIC     │ │
│  │              │  │  NETWORK     │  │ SELECTION     │ │
│  │ Learn which  │  │ Maintain     │  │ Select claims │ │
│  │ events cause │  │ beliefs      │  │ that maximize │ │
│  │ which        │  │ about claim  │  │ expected      │ │
│  │ outcomes     │  │ relevance    │  │ information   │ │
│  │              │  │ per context  │  │ gain for the  │ │
│  │ (structure)  │  │ (beliefs)    │  │ decision      │ │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘ │
│         │                 │                   │         │
│         ▼                 ▼                   ▼         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           COUNTERFACTUAL REASONING              │   │
│  │                                                 │   │
│  │  "What would the decision-maker do differently  │   │
│  │   if they knew this claim?"                     │   │
│  │                                                 │   │
│  │  (LLM-powered, but structured by the above)    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Let me break down each component and why it's better than RL for this problem.

## Component 1: Causal Discovery

Instead of learning a policy through trial and error, **discover the causal structure directly from the data.**

Your claim store already contains temporal relationships. When CEO says "we're investing $500M in AI" in Q1, and Q3 shows "AI revenue up 40%," that's a causal link. You don't need RL to discover it — you need Granger causality or temporal causal discovery on the claim graph.

```
Claim A (Q1): "Investing $500M in AI infrastructure"
    ↓ causal link (discovered from data)
Claim B (Q3): "AI-related revenue increased 40% YoY"
    ↓ causal link
Claim C (Q4): "Operating margins expanded 200bps"
```

Once you have this causal graph, predicting relevance becomes **graph traversal**, not policy optimization. When an analyst asks about Q4 margins, you trace backwards through the causal graph to find upstream claims. The childhood-fear-predicts-board-presentation-anxiety problem becomes: trace the causal chain through the graph, not learn a reward function.

**Methods:** Neural Granger Causality, PCMCI for temporal causal discovery, or the FinDKG approach from the papers we found.

**Why better than RL:** Causal discovery gives you an _interpretable structure_. You can show the analyst WHY a claim is relevant — "this Q1 capex commitment causally influenced the Q3 margin compression you're analyzing." RL gives you a score. The causal graph gives you an explanation.

## Component 2: Bayesian Belief Network

Instead of learning a static policy, **maintain and update beliefs about each claim's relevance as new evidence arrives.**

```python
class ClaimBelief:
    """Bayesian belief about a claim's relevance to different decision types."""
    
    def __init__(self, claim):
        self.claim = claim
        # Prior: based on claim type, age, confidence
        self.relevance_prior = {
            "margin_analysis": Beta(1, 1),      # uninformed prior
            "growth_assessment": Beta(1, 1),
            "risk_evaluation": Beta(1, 1),
            "acquisition_due_diligence": Beta(1, 1),
        }
    
    def update(self, decision_type, was_useful: bool):
        """Update belief based on feedback."""
        if was_useful:
            self.relevance_prior[decision_type].alpha += 1
        else:
            self.relevance_prior[decision_type].beta += 1
    
    def expected_relevance(self, decision_type) -> float:
        """Expected relevance = mean of Beta distribution."""
        prior = self.relevance_prior[decision_type]
        return prior.alpha / (prior.alpha + prior.beta)
    
    def uncertainty(self, decision_type) -> float:
        """How uncertain are we? High uncertainty = explore."""
        prior = self.relevance_prior[decision_type]
        return prior.variance()
```

This handles several things RL struggles with:

**Cold start:** With a new claim, the Beta(1,1) prior gives 0.5 relevance — uncertain, worth exploring. After 3 positive signals, it's Beta(4,1) = 0.8. After 3 negative, Beta(1,4) = 0.2. No training loop needed.

**Objective-dependent:** The same claim has different belief distributions for different decision types. A capex claim might be Beta(8,2) for margin analysis but Beta(2,6) for growth assessment.

**Exploration via uncertainty:** Thompson Sampling falls out naturally. Sample from each claim's Beta distribution. High-uncertainty claims occasionally sample high, getting surfaced and generating new evidence. This IS the bandit mechanism, but derived from principled Bayesian inference rather than engineered reward.

**Why better than RL:** Bayesian updating is more sample-efficient. RL needs hundreds of episodes to learn a good policy. Bayesian updating shifts beliefs meaningfully after 5-10 observations. With your small initial user base, you won't have RL-scale data for months. Bayesian beliefs work from day one.

## Component 3: Information-Theoretic Selection

This is the key conceptual departure from RL. Instead of asking "which claims maximize reward?" ask: **"which claims maximize the expected information gain for this decision?"**

```
Information Gain = H(Decision | Situation) - H(Decision | Situation, Claim)
```

In words: How much does knowing this claim REDUCE the uncertainty about the optimal decision?

A claim that confirms what you already know has zero information gain — even if it's "relevant." A claim that forces you to revise your analysis has high information gain — even if it's surprising or unwelcome.

This naturally handles several problems:

**Redundancy:** If you've already surfaced Claim A (Tesla cut prices 6%), then Claim B (Tesla reduced ASP across markets) has near-zero information gain despite being highly relevant. RL would surface both because both get positive reward. Information gain surfaces A and skips B because B adds no new information.

**Diversity:** Information gain automatically diversifies the claim set. The top 5 by information gain will naturally span different aspects of the decision because each subsequent claim is evaluated against the information already provided.

**Surprise value:** The "topically distant but causally critical" claims have the HIGHEST information gain precisely because they contain information the analyst wouldn't have accessed through normal retrieval. The childhood fear claim is maximally informative for the coaching decision because it completely changes the decision landscape.

**Practical implementation:**

```python
def select_claims(situation, candidates, objective, k=5):
    """Select k claims that maximize total information gain."""
    selected = []
    remaining = list(candidates)
    
    # Infer the decision implied by the situation
    decision_space = infer_decision(situation, objective)  # LLM call
    
    for _ in range(k):
        best_claim = None
        best_gain = -1
        
        for claim in remaining:
            # How much does this claim reduce decision uncertainty?
            # Given what we've already selected?
            gain = estimate_information_gain(
                claim, 
                decision_space, 
                already_selected=selected,
                causal_graph=get_causal_ancestors(claim),
                belief=claim.relevance_belief(objective),
            )
            
            if gain > best_gain:
                best_gain = gain
                best_claim = claim
        
        selected.append(best_claim)
        remaining.remove(best_claim)
        
        # Update decision space given the new claim
        decision_space = update_decision_space(decision_space, best_claim)
    
    return selected
```

**How to estimate information gain without a trained model:**

At Stage 1 (LLM-based), ask Gemini 3.1 Pro:

```
Given this situation and decision context:
{situation}
{objective}

And these claims already provided to the analyst:
{already_selected}

How much would knowing this NEW claim change the analysis?

Claim: {candidate}

Rate the INFORMATION GAIN (not relevance):
- HIGH: Forces a revision of the current analysis
- MEDIUM: Adds a meaningful new dimension not yet covered  
- LOW: Confirms or slightly refines what's already known
- NONE: Redundant or irrelevant

Explain in one sentence what new information this claim provides
that the analyst doesn't already have from the selected claims.
```

This prompt is fundamentally different from the relevance-scoring prompt. It asks "what's NEW here?" not "is this related?" That distinction is the entire value of the information-theoretic frame.

## Component 4: Counterfactual Reasoning

The LLM's role shifts from "score this claim" (judge) to "what would change?" (counterfactual reasoner). This is more natural for LLMs and produces better results because it's asking for reasoning, not classification.

```
Two analysts are evaluating this situation:
{situation}

Analyst A has access to this claim: {claim}
Analyst B does not.

1. What conclusion would Analyst A reach?
2. What conclusion would Analyst B reach?
3. How different are their conclusions?
4. If the difference is significant, what specific aspect of
   the decision does this claim change?
```

The counterfactual frame produces three things the relevance frame doesn't: a causal explanation (what changes), a magnitude estimate (how much it changes), and a mechanism (through which pathway it changes). All three are valuable for the decision surface.

## How It All Composes

```
Input: situation + candidate_claims + user_objective

Step 1: CAUSAL GRAPH LOOKUP
  → Find causal ancestors/descendants of entities in the situation
  → Expand candidate set with claims connected via causal paths
  → This surfaces topically distant but causally connected claims

Step 2: BAYESIAN PRIOR SCORING  
  → Score each candidate using belief distributions
  → Conditioned on the objective type
  → High uncertainty claims get flagged for exploration

Step 3: INFORMATION-THEORETIC SELECTION
  → Greedy selection maximizing marginal information gain
  → Each new claim evaluated against claims already selected
  → LLM estimates information gain via counterfactual reasoning
  → Automatically handles redundancy and diversity

Step 4: BAYESIAN UPDATE
  → User feedback updates belief distributions
  → "Useful" → alpha++, "Noise" → beta++
  → Updates are per-claim, per-objective-type
  → No retraining needed — beliefs update online

Step 5: CAUSAL GRAPH UPDATE
  → If user confirms a causal link ("yes, the Q1 capex 
    DID explain Q3 margins"), strengthen that edge
  → If user rejects ("no, that was coincidental"), weaken it
  → Graph structure evolves with evidence
```

## Why This Is Better Than RL

|Dimension|RL Approach|Expanded Approach|
|---|---|---|
|**Cold start**|Needs hundreds of episodes|Works from day one (priors + LLM)|
|**Interpretability**|"Score: 0.87"|"This Q1 claim causally upstream of Q3 outcome via capex→margin pathway"|
|**Sample efficiency**|Needs thousands of labeled tuples|Bayesian updating shifts after 5-10 signals|
|**Objective conditioning**|Needs separate policy per objective|Same graph + beliefs, different conditioning|
|**Redundancy handling**|Doesn't address (top 5 might all say the same thing)|Information gain naturally diversifies|
|**Reward engineering**|Biggest risk — wrong reward = wrong behavior|No reward function. Information gain is principled.|
|**Exploration**|Needs explicit ε-greedy or Thompson Sampling|Falls out naturally from Bayesian uncertainty|
|**Delayed feedback**|Core RL challenge|Bayesian updating handles any feedback timing|
|**Composability**|Monolithic policy|Each component independently testable|

## What You Keep From the RL Research

The papers we found are still valuable, but you use them differently:

- **PURPLE** → Use their Plackett-Luce ranking model, but train it with information gain rather than reward
- **FinDKG** → Use their extraction pipeline and temporal graph structure directly
- **CluSTeR / TITer** → Use their temporal graph traversal for causal discovery, not for RL policy
- **RAG-RL curriculum learning** → Apply to the counterfactual reasoning prompt (easy cases first)
- **Temporal Causal RL** → Use their temporal logic formalism for encoding causal constraints in the graph

## The Build Sequence Changes

|Phase|RL Version|Expanded Version|
|---|---|---|
|3|LLM judge scores relevance|LLM estimates information gain via counterfactual|
|3.5|Contextual bandit|Bayesian belief network + Thompson sampling|
|4|Forgetting = confidence decay|Forgetting = Bayesian belief decay + causal graph pruning|
|5|Decision surface from top-K by score|Decision surface from max-info-gain set with causal explanations|
|6|Schema inference|Causal graph discovery (same output, stronger method)|
|Future|Full RL policy training|Causal graph + Bayesian beliefs + information gain — all evolving online, no training loop|

The biggest practical win: **you never need a training pipeline.** No GPU for policy training. No reward engineering. No episodes. The system improves through Bayesian updating (fast, cheap, online) and causal graph refinement (structural, interpretable, persistent). The LLM handles the hard reasoning (counterfactuals), and the Bayesian framework handles the learning.

## The real problem stack

Your idea is not one problem. It is **five stacked problems**:

1. **Represent the world as time-indexed events on a graph**
    
2. **Select the historically relevant subgraph for the current objective**
    
3. **Separate causal history from merely correlated history**
    
4. **Learn from outcomes that arrive later**
    
5. **Produce auditable chains of evidence for a decision**
    

That decomposition matches the research landscape much better than “RL on context graphs.” The mature center of the field is temporal KG reasoning; the thinner, more novel edge is causal subhistory plus delayed-outcome supervision plus bi-temporal event handling. ([arXiv](https://arxiv.org/abs/2403.04782?utm_source=chatgpt.com "A Survey on Temporal Knowledge Graph: Representation Learning and Applications"))

---

## Taxonomy by problem class

### A. Temporal graph forecasting

**Question:** What future edge, event, or state is likely next?

This is the base layer. Temporal knowledge graph work is mostly about extrapolating future events from timestamped historical facts. The survey literature treats this as a mature family with established datasets, tasks, and model categories. ([arXiv](https://arxiv.org/abs/2403.04782?utm_source=chatgpt.com "A Survey on Temporal Knowledge Graph: Representation Learning and Applications"))

**Use for you:** mandatory foundation  
**Not enough by itself:** it predicts, but usually does not tell you which historical events were _decision-relevant_.

---

### B. Historically relevant event selection

**Question:** Which parts of history matter for _this_ prediction?

This is much closer to your “context graph” intuition. xERTE frames forecasting as reasoning over a **query-relevant subgraph**, and HisRES explicitly models **historically relevant events** by combining recent evolution with globally relevant history. ([OpenReview](https://openreview.net/forum?id=pGIHq1m7PU&utm_source=chatgpt.com "Explainable Subgraph Reasoning for Forecasting on ..."))

**Use for you:** core intellectual home  
**Closest fit today:** “objective-conditioned relevance learning”

---

### C. Causal subhistory identification

**Question:** Which earlier events are causally upstream, rather than just correlated?

This is where your novelty starts getting sharper. The IJCAI 2024 CSI paper explicitly introduces **causal subhistory identification** for temporal KG extrapolation and claims to be the first to bring causality directly into this TKG extrapolation setting. ([IJCAI](https://www.ijcai.org/proceedings/2024/0365.pdf?utm_source=chatgpt.com "Temporal Knowledge Graph Extrapolation via Causal ..."))

**Use for you:** differentiator  
**Closest fit today:** “causal relevance ranking over event history”

---

### D. Delayed-outcome supervision

**Question:** How do we learn when labels only arrive after time passes?

This is a separate axis from the model family. _Future-as-Label_ is highly relevant because it treats resolved future outcomes as supervision and formalizes learning under a temporal gap between prediction time and outcome time. ([arXiv](https://arxiv.org/abs/2601.06336?utm_source=chatgpt.com "Future-as-Label: Scalable Supervision from Real-World Outcomes"))

**Use for you:** absolutely central  
**Why it matters:** your real signal often arrives weeks or months later.

---

### E. Bi-temporal event modeling

**Question:** What happened when, and when did we learn it?

This is the valid-time vs ingestion-time distinction. Zep is one of the clearest adjacent systems using a temporally aware KG architecture for agent memory and is relevant because it treats time as a first-class structural concern for dynamic knowledge. ([arXiv](https://arxiv.org/abs/2501.13956?utm_source=chatgpt.com "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"))

**Use for you:** architecture differentiator  
**State of field:** important, but underdeveloped as a predictive learning literature

---

### F. Domain-specific financial event reasoning

**Question:** Can event graphs support investment or corporate-outcome prediction?

This is the application layer. THGNN shows heterogeneous temporal graphs are useful in finance, though mostly for movement prediction rather than causal evidence ranking. TRACE is much closer to your direction because it uses a temporal financial KG with rule-guided chain-of-evidence reasoning and interpretable stock prediction. ([arXiv](https://arxiv.org/abs/2305.08740?utm_source=chatgpt.com "Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction"))

**Use for you:** nearest market-adjacent evidence  
**Important distinction:** finance work exists, but your Deal Intelligence framing is still more decision-support oriented than most market-direction papers.

---

## Taxonomy by model family

### 1. Embedding / representation-learning models

These compress temporal graph structure into vector spaces for prediction. This is the backbone of a lot of TKG work and the center of the survey literature. ([arXiv](https://arxiv.org/abs/2403.04782?utm_source=chatgpt.com "A Survey on Temporal Knowledge Graph: Representation Learning and Applications"))

**Best for:** strong baselines, scalable retrieval, pretraining  
**Weakness:** limited auditability by default

---

### 2. Explainable subgraph reasoning models

These choose or construct a local evidence graph around a query. xERTE is the canonical reference here. ([OpenReview](https://openreview.net/forum?id=pGIHq1m7PU&utm_source=chatgpt.com "Explainable Subgraph Reasoning for Forecasting on ..."))

**Best for:** “why did the model think this?”  
**Best fit for you:** yes

---

### 3. Historical relevance structuring models

These explicitly separate recent snapshots from globally important historical events. HisRES is the key citation. ([arXiv](https://arxiv.org/abs/2405.10621?utm_source=chatgpt.com "Historically Relevant Event Structuring for Temporal Knowledge Graph Reasoning"))

**Best for:** your context-graph intuition  
**Best fit for you:** very high

---

### 4. Causal filtering / causal subhistory models

These aim to remove non-causal historical clutter before reasoning. CSI is the key reference. ([IJCAI](https://www.ijcai.org/proceedings/2024/0365.pdf?utm_source=chatgpt.com "Temporal Knowledge Graph Extrapolation via Causal ..."))

**Best for:** “not all history deserves equal weight”  
**Best fit for you:** very high

---

### 5. Contrastive / self-supervised temporal learning

CENET is a notable example using historical contrastive learning for event forecasting in TKGs. ([Atailab](https://www.atailab.cn/ir2023fall/pdf/2023_AAAI%20Temporal%20Knowledge%20Graph%20Reasoning%20with%20Historical%20Contrastive%20Learning.pdf?utm_source=chatgpt.com "Temporal Knowledge Graph Reasoning with Historical ..."))

**Best for:** pretraining relevance without expensive labels  
**Best fit for you:** high, especially in early stages

---

### 6. Heterogeneous temporal GNNs

These model multiple node/edge types and evolving interactions; THGNN is the finance example. ([arXiv](https://arxiv.org/abs/2305.08740?utm_source=chatgpt.com "Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction"))

**Best for:** mixed event/entity types  
**Best fit for you:** good as infrastructure, weaker as a full explanation system

---

### 7. Rule-guided symbolic-neural hybrids

TRACE is the most recent and closest example: rule-constrained exploration plus text-grounded evidence plus human-readable chains. ([arXiv](https://arxiv.org/abs/2603.12500?utm_source=chatgpt.com "TRACE: Temporal Rule-Anchored Chain-of-Evidence on Knowledge Graphs for Interpretable Stock Movement Prediction"))

**Best for:** auditability and decision support  
**Best fit for you:** extremely high

---

## Taxonomy by supervision regime

### A. Immediate supervised labels

Standard forecasting setup.  
**For you:** too weak alone.

### B. Delayed supervised labels

Outcome arrives later; train retrospectively.  
**For you:** core regime. _Future-as-Label_ is the nearest formal framing. ([arXiv](https://arxiv.org/abs/2601.06336?utm_source=chatgpt.com "Future-as-Label: Scalable Supervision from Real-World Outcomes"))

### C. Weak supervision

Rules, heuristics, analyst annotations, proxy outcomes.  
**For you:** useful for bootstrapping.

### D. Self-supervision

Masked event prediction, temporal order prediction, contrastive path learning. CENET is relevant here. ([Atailab](https://www.atailab.cn/ir2023fall/pdf/2023_AAAI%20Temporal%20Knowledge%20Graph%20Reasoning%20with%20Historical%20Contrastive%20Learning.pdf?utm_source=chatgpt.com "Temporal Knowledge Graph Reasoning with Historical ..."))  
**For you:** likely the right Stage 1.

### E. RL / bandits

Possible later for active evidence gathering or adaptive querying, but not the right center of gravity. _Future-as-Label_ uses RL-style reward semantics, but the key contribution is still outcome-based supervision from resolved futures. ([arXiv](https://arxiv.org/abs/2601.06336?utm_source=chatgpt.com "Future-as-Label: Scalable Supervision from Real-World Outcomes"))

---

## Taxonomy by time semantics

### 1. Snapshot-time models

Reason over discrete graph slices.

### 2. Event-stream models

Reason over continuous streams of events.

### 3. Bi-temporal models

Track both event time and ingestion/knowledge time. Zep is the clearest adjacent signal here. ([arXiv](https://arxiv.org/abs/2501.13956?utm_source=chatgpt.com "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"))

For your work, **bi-temporal event streams** should stay in scope. That is one of your strongest structural differentiators.

---

## What to read vs ignore

### Read first

- **TKG survey** for the landscape and vocabulary. ([arXiv](https://arxiv.org/abs/2403.04782?utm_source=chatgpt.com "A Survey on Temporal Knowledge Graph: Representation Learning and Applications"))
    
- **xERTE** for explainable subgraph reasoning. ([OpenReview](https://openreview.net/forum?id=pGIHq1m7PU&utm_source=chatgpt.com "Explainable Subgraph Reasoning for Forecasting on ..."))
    
- **HisRES** for historically relevant event structuring. ([arXiv](https://arxiv.org/abs/2405.10621?utm_source=chatgpt.com "Historically Relevant Event Structuring for Temporal Knowledge Graph Reasoning"))
    
- **CSI** for causal subhistory identification. ([IJCAI](https://www.ijcai.org/proceedings/2024/0365.pdf?utm_source=chatgpt.com "Temporal Knowledge Graph Extrapolation via Causal ..."))
    
- **Future-as-Label** for delayed-outcome supervision. ([arXiv](https://arxiv.org/abs/2601.06336?utm_source=chatgpt.com "Future-as-Label: Scalable Supervision from Real-World Outcomes"))
    
- **TRACE** for finance-adjacent interpretable temporal KG reasoning. ([arXiv](https://arxiv.org/abs/2603.12500?utm_source=chatgpt.com "TRACE: Temporal Rule-Anchored Chain-of-Evidence on Knowledge Graphs for Interpretable Stock Movement Prediction"))
    

### Read second

- **CENET** for contrastive temporal event learning. ([Atailab](https://www.atailab.cn/ir2023fall/pdf/2023_AAAI%20Temporal%20Knowledge%20Graph%20Reasoning%20with%20Historical%20Contrastive%20Learning.pdf?utm_source=chatgpt.com "Temporal Knowledge Graph Reasoning with Historical ..."))
    
- **THGNN** for heterogeneous temporal graph modeling in finance. ([arXiv](https://arxiv.org/abs/2305.08740?utm_source=chatgpt.com "Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction"))
    
- **Zep** for bi-temporal architecture instincts. ([arXiv](https://arxiv.org/abs/2501.13956?utm_source=chatgpt.com "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"))
    

### Mostly ignore for now

- Pure static KG completion papers
    
- Generic forecasting transformers with no graph/event explanation
    
- RL papers that focus on control or exploration rather than delayed outcome learning on temporal event structure
    
- Papers that say “context graph” but do not actually formalize temporal event relevance
    

---

## Where your idea sits

Your strongest framing is here:

**Temporal decision-support systems over bi-temporal heterogeneous event graphs, using objective-conditioned relevance learning to identify causal subhistory under delayed outcome supervision.**

That puts you at the intersection of:

- temporal KG reasoning,
    
- explainable subgraph selection,
    
- causal subhistory identification,
    
- delayed real-world supervision,
    
- and interpretable financial event reasoning. ([arXiv](https://arxiv.org/abs/2405.10621?utm_source=chatgpt.com "Historically Relevant Event Structuring for Temporal Knowledge Graph Reasoning"))
    

That is a better novelty claim than “RL on context graphs.”

---

## The practical MVP taxonomy

If I were building this, I would organize the stack like this:

**Layer 1: Graph substrate**  
Bi-temporal heterogeneous event graph

**Layer 2: Candidate history retrieval**  
Recent window + historically relevant global retrieval

**Layer 3: Relevance scorer**  
Objective-conditioned event / path ranking

**Layer 4: Causal filter**  
Subhistory pruning to reduce spurious correlations

**Layer 5: Outcome learner**  
Train from delayed realized outcomes

**Layer 6: Evidence product**  
Human-readable chain of evidence for decision support

That stack is directly supported by adjacent literature even though your exact combination is still sparse. ([arXiv](https://arxiv.org/abs/2405.10621?utm_source=chatgpt.com "Historically Relevant Event Structuring for Temporal Knowledge Graph Reasoning"))

## The one-sentence taxonomy

If you want the shortest possible version:

**Foundation:** temporal KG forecasting  
**Core mechanism:** explainable relevance subgraph selection  
**Differentiator:** causal subhistory identification  
**Training signal:** delayed outcome supervision  
**Structural edge:** bi-temporal event streams  
**Application:** interpretable financial decision support

The taxonomy above informed the following research thesis: [Research Thesis Engram](Research%20Thesis%20Engram.md)