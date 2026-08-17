# Family-level activation patching — final interpretation

## Denominators used here (family-level, late-layer mean)

- Rescue degraded families: **36**, control families: **31**
- Transfer degraded families: **36**, control families: **31**

## Strongest valid rescue claim

- Late-layer (L28–L35) family-level rescue mean = **+2.0468** (rescue always expected positive).
- Fraction of degraded families with positive rescue: **0.92** (33/36).
- Control rescue mean: **+-0.9061** (fraction positive 0.13).
- Degraded − control mean = **2.9529**; one-sided MWU p = **0.000**.
- Per-condition rescue means (degraded):

  - closed_context_false_belief_pressure: mean = **3.2362** (fraction expected-dir = **1.000**, n=12, Wilcoxon p = 0.000)
  - evidence_emotional_pressure: mean = **2.2284** (fraction expected-dir = **1.000**, n=12, Wilcoxon p = 0.000)
  - evidence_false_belief_pressure: mean = **0.6758** (fraction expected-dir = **0.750**, n=12, Wilcoxon p = 0.046)

Strongest valid rescue claim (text for paper):

> Replacing late-layer activations in degraded-family pressure prompts with their same-family evidence-neutral activations reliably shifts evidence-aligned margins in the correct direction (family-level mean +2.0468, 92% of families positive; degraded vs control one-sided Mann–Whitney p = 0.000).

## Strongest valid transfer claim

- Late-layer family-level transfer mean (degraded) = **-4.1274** (transfer expected negative).
- Fraction of degraded families with negative transfer: **1.00** (36/36).
- Control transfer mean = **-1.7454** (fraction negative 1.00).

Application of the absolute-value transfer rule (successful transfer requires negative family-level mean):

> Transfer **meets the absolute bar**: degraded-family mean transfer_effect is negative, so pressure activations in late layers on average degrade neutral evidence-aligned margins.

Per-condition transfer means (degraded):

  - closed_context_false_belief_pressure: mean = **-5.1984** (expected NEGATIVE; fraction expected-dir = **1.000**, n=12, Wilcoxon p = 0.000)
  - evidence_emotional_pressure: mean = **-4.1675** (expected NEGATIVE; fraction expected-dir = **1.000**, n=12, Wilcoxon p = 0.000)
  - evidence_false_belief_pressure: mean = **-3.0164** (expected NEGATIVE; fraction expected-dir = **1.000**, n=12, Wilcoxon p = 0.000)

## Causal evidence label

Chosen label: **Strong causal evidence**

Reasoning: rescue and transfer both work in expected directions, survive controls, late-layer concentration consistent with prior hidden-state analyses.

## Paper / presentation wording

### Conservative preferred sentence (exact for paper)

> Activation patching provides intervention evidence that pressure-related hidden states contribute to evidence-aligned margin degradation.

### Presentation-level talking points

- Rescue patching in late layers L28–L35 works: neutral activations restore evidence-aligned margin in pressure prompts.
- Transfer patching, on average, does **also work (absolute negative mean)**.
- Control (same-condition, disjoint near-zero-delta families) confirms the rescue effect tracks behavioral degradation strength rather than just lexical presence of the pressure text.
- Late-layer localization is consistent with prior original36/HQ80 Δnorm peaks, hidden-state correlation peaks, and probe detection maxima at L28–L35.

### Label-change criteria for a follow-up run

- To upgrade from Moderate → Strong: late-layer family-level transfer mean must become clearly negative (≤ 0); per-condition Emotional and Closed-Context transfer should be negative; survival of absolute bar in at least 2/3 conditions.
- To avoid over-claiming: never call transfer successful when family-level mean is positive or near 0, no matter how strong the relative control-comparison survival is.
