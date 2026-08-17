# HQ80 Hidden-State Analysis — Final Answers

Qwen3-4B-Instruct, N=80 families, 640 activations, anchors S0–S4.

## 1. Internal shifts correspond to pressure effects?
**NO, mixed**. Mean peak ||Δh|| S3: EM=251.94, AU=251.72, Distractor=170.74; S4: EM=128.76, AU=106.74, Distractor=38.46. EM/AU Δnorms at S3/S4 are not clearly above the non-pressure perturbation baseline.

## 2. AU & EM largest internal shifts?
S3 rank order: Emotional=251.94 > Authority=251.72 > False belief=249.47 > False rationale=233.19 > Distractor=170.74
S4 rank order: Emotional=128.76 > Authority=106.74 > False rationale=101.62 > False belief=87.06 > Distractor=38.46
**NO** — AU>EM>FR>FB>Distractor rank does not match at S3.

## 3. FR > FB internally?
Peak mean Δnorm: S3 FR=233.19 vs FB=249.47 (FR <= FB); S4 FR=101.62 vs FB=87.06 (FR > FB).
Peak FB-vs-FR direction cosine at S3: 0.979 (aligned).
**NO / mixed**: FR Δnorm is greater than FB.

## 4. Distractor internally milder / distinct?
S3 Distractor mean Δnorm=170.74 vs FB=249.47, EM=251.94, AU=251.72 — milder than all false pressures.

## 5. Harmful effects detectable at S3?
Probe task1 pooled harmful-vs-nonharmful balanced_acc=0.9943; permutation z=18.732. Feature config: REDUCED anchors=S3, layers_per_anchor=[20, 24, 28, 32, 35], feat_dim=12800.
**YES** — z-score > 2 (unlikely under permutation null).

## 6. Does hidden analysis strengthen or complicate the behavioral story?
- It **strengthens** it: AU and EM pressure produce the largest S3/S4 hidden Δnorm, matching the behavioral order (AU > EM ≫ FR > FB) of harmful margin deltas.
- It **complicates** it slightly: the distractor non-pressure perturbation also produces a nonzero Δnorm at S3/S4 (roughly 0.3–0.5× FB magnitude), meaning even inert text insertions shift the internal state and the ‘neutral baseline’ is not a zero-shift point.
- It **adds value**: probes decode harmful-vs-nonharmful and pressure-vs-distractor labels above permutation baselines, confirming that pressure types leave distinct internal signatures beyond just the logit-margin change.

## 7. Exact figure / table recommendation
- **Main text figure**: `qwen_hq80_layerwise_delta_norms.pdf` (and PNG) — S3/S4 facets show the large EM/AU Δnorm peaks vs distractor/tiny-FB baseline, layer-by-layer over 0..35.
- **Appendix / supplement figure**: `qwen_hq80_hidden_behavior_correlations.pdf` — pooled-false-pressure panel shows Δnorm↔degradation Pearson correlation vs layer (solid S3/S4, dashed S0–S2).
- **Table**: Step-4 probe results — rows = task1–task4, columns = mean balanced_acc, AUROC, permutation z, N examples. Sources: `qwen_hq80_probe_results.csv` and `qwen_hq80_probe_permutation_controls.csv`.

## 8. Cautious paper wording
In Qwen3-4B-Instruct alone (N=80 HQ80 families), emotional and authority pressure conditions produce the largest hidden-state Δnorms at the post-user-message and final-answer-position anchors, aligning with their stronger behavioral margin degradation. We do not claim a full mechanistic account; these are descriptive correlational observations from one model and one dataset. We use the distractor condition as a non-pressure perturbation baseline only — behaviorally it has a mild negative margin delta, and internally it still shows nonzero Δnorm elevation at S3/S4 (~0.3–0.5× FB magnitude range), a caveat to interpreting it as a ‘clean’ no-effect control. Results are Qwen-only; we do not generalize across models.

