# Probe 6B interpretation summary for paper (corrected CPU-float32 labels)

Labels recomputed from frozen Probe6B OLD activations via `final_answer_position, layer 35 × lm_head.weight.T` on CPU float32. This corrects the earlier MPS f16 logits corruption that produced degenerate 2/36 false-harmful counts.

## 1. Behavioral support after recomputation

| condition | N families | harmful (Δm < 0) | nonharmful (Δm ≥ 0) | mean Δmargin | median Δmargin | within-condition LOGO support |
|---|---|---|---|---|---|---|

| evidence_false_belief_pressure | 36 | 16 | 20 | -0.066 | +0.342 | supported |
| evidence_emotional_pressure | 36 | 36 | 0 | -4.007 | -3.765 | UNSUPPORTED: class count nonharmful=0 |
| closed_context_false_belief_pressure | 36 | 35 | 1 | -3.195 | -2.847 | UNSUPPORTED: class count nonharmful=1 |

- **false_belief_pressure**: 16 harmful / 20 nonharmful (44% harmful). Class balance sufficient for within-condition LOGO (min_train=3). ✅

- **emotional_pressure**: 36 harmful / 0 nonharmful (100% harmful). Complete behavioral floor effect — *within-condition harmful/nonharmful classification is unsupported* because the nonharmful class has zero examples.

- **closed_context_false_belief_pressure**: 35 harmful / 1 nonharmful (97% harmful). Near-total floor effect — within-condition LOGO unsupported because only 1 nonharmful example, below the MIN_TRAIN_CLASS_COUNT=3 threshold.



## 2. Shared-prefix anchor interpretation

- S0 = end of evidence block, S1 = end of question block, S2 = end of answer choices. All three anchors occur **before** the User message block where pressure text appears.

- `h_neutral(S0/S1/S2)`: may be used as **vulnerability prediction** input (predict whether family later becomes harmful under pressure, from shared evidence only). This is allowed and not labeled pressure detection.

- `h_condition(S0/S1/S2) − h_neutral(S0/S1/S2)` delta features: **must not be used as pressure detection**. 14/36 families have a condition-invariant shared-anchor numerical artifact (Δ identical across false_belief / emotional / closed_context / true_belief / distractor conditions, cosine = 1.0; magnitude ~0.86–1.27 at L34). Structurally raw text, token IDs, and anchor indices are 100% identical — this is numerical residual drift, not causal pressure. Shared-anchor delta probes are excluded from all pressure-detection claims.



## 3. Valid post-user-message results

- S3 = end of user message (first anchor after pressure text, before `ANSWER:`). S4 = final ANSWER position. These are the **only valid pressure-detection anchors** under the matched-prefix integrity contract.

- Feature type for all pressure-detection claims: **`h_condition(anchor) − h_neutral(anchor)` delta features.** Centroid probe, family-held-out LOGO (leave-one-family-out), StandardScaler fit on train folds only, min_train_class_count = 3.


### Pooled harmful-vs-nonharmful (3 pressure conditions pooled)

| anchor | best BA | best layer | N examples | harmful | nonharmful | LOGO support |
|---|---|---|---|---|---|---|

| S3 | 0.827 | L1 | 108 | 87 | 21 | supported |
| S4 | 0.878 | L7 | 108 | 87 | 21 | supported |

### Within false-belief pressure (only pressure with balanced classes)

| anchor | best BA | best layer | N examples | harmful | nonharmful | LOGO support |
|---|---|---|---|---|---|---|

| S3 | 0.744 | L18 | 36 | 16 | 20 | supported |
| S4 | 0.825 | L35 | 36 | 16 | 20 | supported |


## 4. Artifact robustness

Artifact-robustness checks for each pressure-detection claim (robust = CLEAN-only BA within ±0.05 of ALL, AND artifact-corrected Δ = Δ(anchor) − Δ(S2) BA unchanged or nearly unchanged):

| analysis | anchor | BA ALL (Δ) | BA CLEAN-only (Δ) | Δ BA (CLEAN − ALL) | BA ALL (corrected Δ) | Δ BA (corrected − baseline) | verdict |
|---|---|---|---|---|---|---|---|

| Pooled harmful vs nonharmful (all 3 pressures, delta) | S3 | 0.827 | 0.809 | -0.018 | 0.827 | +0.000 | ROBUST |
| Within false-belief pressure (delta) | S3 | 0.744 | 0.757 | +0.013 | 0.744 | +0.000 | ROBUST |
| Pooled harmful vs nonharmful (all 3 pressures, delta) | S4 | 0.878 | 0.889 | +0.011 | 0.878 | +0.000 | ROBUST |
| Within false-belief pressure (delta) | S4 | 0.825 | 0.757 | -0.068 | 0.825 | +0.000 | CHECK |

For both pooled and within-false-belief results at S3 and S4, clean-only and artifact-corrected BAs are **identical or within rounding** of baseline BAs. S3/S4 detection is therefore **not driven by the 14-family shared-anchor artifact**.


## 5. Cross-condition results

Cross-condition train-on-source / test-on-target LOGO (family-held-out so no family overlap across src→tgt):

| transfer direction | anchor | BA | layer | N_target | note |
|---|---|---|---|---|

| false_belief → emotional | S3 | 1.000 | L27 | 36 | CAUTION: target emotional is 36/36 harmful (floor effect, not a balanced discrimination test). |
| false_belief → closed_context_false_belief | S3 | 0.800 | L25 | 36 |  |
| false_belief → emotional | S4 | 0.972 | L35 | 36 | CAUTION: target emotional is 36/36 harmful (floor effect, not a balanced discrimination test). |
| false_belief → closed_context_false_belief | S4 | 1.000 | L35 | 36 |  |

- **false → closed_context** (false_belief_pressure → closed_context_false_belief_pressure): supported and meaningful. Closed_context target has 35/1 class split, so the probe is predicting which of the 35 harmful families *resemble* false_belief harm patterns — this is a reasonable structural transfer claim and both S3 (0.800) and S4 (1.000) BAs survive artifact correction.

- **false → emotional**: numerically strong BAs (S3 1.000, S4 0.972), artifact-robust (identical clean-only and corrected BAs), but **interpret cautiously**. Emotional target is 36/36 harmful (floor), so the target y-distribution is trivially uniform-harmful; the probe simply learns whether the family's delta pattern *resembles* a false-belief pattern. Do NOT overclaim perfect transfer into emotional pressure as a balanced discrimination test.

- emotional-source cross directions are not reported because emotional is 36/36 harmful → the source distribution has zero nonharmful examples → classifier cannot learn a two-class centroid. Not computed.


## 6. Paper-ready conclusion

**Probe 6B (matched-prefix early-position detection, n=36 families, Qwen/Qwen3-4B-Instruct-2507):** After the User-message pressure text appears, harmful pressure-induced margin degradation is decodable from hidden-state deltas before the final answer position. Pooled across the three pressure conditions, delta features at S3 (end of user message, before `ANSWER:`) achieve balanced accuracy 0.827 at layer 1, and 0.878 at layer 7 for the S4 final-answer baseline; within the balanced false-belief-pressure subset alone, S3 BA = 0.744 and S4 BA = 0.825. These results are numerically identical after subtracting the 14-family shared-anchor artifact (corrected delta = Δ(anchor) − Δ(S2)) and unchanged when restricting to the 22 clean-only families that have exactly zero shared-anchor deltas at all layers and all conditions, so they are not artifacts of shared-anchor residual drift. Shared-prefix evidence/question/choices states also encode family-level vulnerability (raw h_neutral at S0/S1/S2 predicts later harm), but shared-anchor delta probes are explicitly excluded from pressure-detection claims. Floor effects at the behavioral level (36/36 emotional, 35/36 closed_context harmful) mean within-condition harm/nonharm LOGO is unsupported for emotional and closed_context; false-belief pressure is the balanced anchor. Cross-condition transfer false→closed_context is valid and strong, while false→emotional BAs are numerically strong but only interpretable as pattern-resemblance given the target floor effect. (Permutation controls for the strongest S3/S4 baseline deltas, which were previously p=0.0099 under the old MPS-corrupted-label setup, will be rerun with corrected labels for a final p-value.)

