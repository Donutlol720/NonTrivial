# Probe 6B shared-anchor audit report

## Scope

Audited activation root: `activations/qwen3_4b_instruct_2507_probe6b_matched_prefix`

Matched-prefix dataset: `data/generated_prompts_probe6b_matched_prefix_v1.jsonl`

Families: 36; conditions/family: 6

Anchors audited (shared anchors first S0/S1/S2):

- end_of_evidence_block
- end_of_question_block
- end_of_answer_choices
- end_of_user_message
- final_answer_position

## 1. Raw prefix identity (evidence/question/choices blocks)

We compare `prompt_text[0 : anchor_char_end+1]` of each non-neutral condition against `evidence_neutral`. Expected: 100% identical.


| condition | anchor | raw_identical / families |
|---|---|---|

| evidence_false_belief_pressure | end_of_evidence_block | 36/36 |
| evidence_false_belief_pressure | end_of_question_block | 36/36 |
| evidence_false_belief_pressure | end_of_answer_choices | 36/36 |
| evidence_emotional_pressure | end_of_evidence_block | 36/36 |
| evidence_emotional_pressure | end_of_question_block | 36/36 |
| evidence_emotional_pressure | end_of_answer_choices | 36/36 |
| closed_context_false_belief_pressure | end_of_evidence_block | 36/36 |
| closed_context_false_belief_pressure | end_of_question_block | 36/36 |
| closed_context_false_belief_pressure | end_of_answer_choices | 36/36 |
| evidence_distractor_neutral | end_of_evidence_block | 36/36 |
| evidence_distractor_neutral | end_of_question_block | 36/36 |
| evidence_distractor_neutral | end_of_answer_choices | 36/36 |
| evidence_true_belief_pressure | end_of_evidence_block | 36/36 |
| evidence_true_belief_pressure | end_of_question_block | 36/36 |
| evidence_true_belief_pressure | end_of_answer_choices | 36/36 |

## 2. Token prefix identity

We compare tokenizer ids `token_ids[0 : anchor_token_idx+1]` between condition and neutral. Expected: 100% identical.


| condition | anchor | token_identical / families |
|---|---|---|

| evidence_false_belief_pressure | end_of_evidence_block | 36/36 |
| evidence_false_belief_pressure | end_of_question_block | 36/36 |
| evidence_false_belief_pressure | end_of_answer_choices | 36/36 |
| evidence_emotional_pressure | end_of_evidence_block | 36/36 |
| evidence_emotional_pressure | end_of_question_block | 36/36 |
| evidence_emotional_pressure | end_of_answer_choices | 36/36 |
| closed_context_false_belief_pressure | end_of_evidence_block | 36/36 |
| closed_context_false_belief_pressure | end_of_question_block | 36/36 |
| closed_context_false_belief_pressure | end_of_answer_choices | 36/36 |
| evidence_distractor_neutral | end_of_evidence_block | 36/36 |
| evidence_distractor_neutral | end_of_question_block | 36/36 |
| evidence_distractor_neutral | end_of_answer_choices | 36/36 |
| evidence_true_belief_pressure | end_of_evidence_block | 36/36 |
| evidence_true_belief_pressure | end_of_question_block | 36/36 |
| evidence_true_belief_pressure | end_of_answer_choices | 36/36 |

## 3. Anchor token-index identity across conditions

We verify whether the anchor token **index position** (0-based token count) is identical across conditions. This matters because layerwise representations are position-dependent: even if prefix tokens are identical, if the anchor lands at a different absolute token index (because later suffix tokens in the User message change tokenizer continuation behavior on a previous boundary — unlikely but possible with BPE), positional embedding would introduce non-zero delta norms at S0/S1/S2.


| condition | anchor | same_index / families |
|---|---|---|

| evidence_false_belief_pressure | end_of_evidence_block | 36/36 |
| evidence_false_belief_pressure | end_of_question_block | 36/36 |
| evidence_false_belief_pressure | end_of_answer_choices | 36/36 |
| evidence_emotional_pressure | end_of_evidence_block | 36/36 |
| evidence_emotional_pressure | end_of_question_block | 36/36 |
| evidence_emotional_pressure | end_of_answer_choices | 36/36 |
| closed_context_false_belief_pressure | end_of_evidence_block | 36/36 |
| closed_context_false_belief_pressure | end_of_question_block | 36/36 |
| closed_context_false_belief_pressure | end_of_answer_choices | 36/36 |
| evidence_distractor_neutral | end_of_evidence_block | 36/36 |
| evidence_distractor_neutral | end_of_question_block | 36/36 |
| evidence_distractor_neutral | end_of_answer_choices | 36/36 |
| evidence_true_belief_pressure | end_of_evidence_block | 36/36 |
| evidence_true_belief_pressure | end_of_question_block | 36/36 |
| evidence_true_belief_pressure | end_of_answer_choices | 36/36 |

## 4. Delta norms per (condition, anchor, layer, family)

Full per-row CSV: `results/probe6b_anchor_delta_norms_by_anchor_condition.csv`

Aggregate (condition, anchor) summary:


| condition | anchor | mean( per-family max-layer ||delta||_2 ) | median( per-family max-layer ||delta||_2 ) | max( all families, all layers ) | mean( all cells ) | layer_of_max_mean |

|---|---|---|---|---|---|---|

| evidence_false_belief_pressure | end_of_evidence_block | 2.83996144e-01 | 0.00000000e+00 | 1.04014222e+00 | 6.12915082e-02 | 34 |
| evidence_false_belief_pressure | end_of_question_block | 3.14460279e-01 | 0.00000000e+00 | 1.20274042e+00 | 6.39539276e-02 | 34 |
| evidence_false_belief_pressure | end_of_answer_choices | 3.23137142e-01 | 0.00000000e+00 | 1.26974564e+00 | 6.55605579e-02 | 34 |
| evidence_false_belief_pressure | end_of_user_message | 2.69225787e+02 | 2.70173829e+02 | 3.01990190e+02 | 6.73005116e+01 | 34 |
| evidence_false_belief_pressure | final_answer_position | 9.22141735e+01 | 9.06107655e+01 | 1.25460109e+02 | 2.28770750e+01 | 34 |
| evidence_emotional_pressure | end_of_evidence_block | 3.88318749e-01 | 0.00000000e+00 | 1.04165188e+00 | 8.40837382e-02 | 34 |
| evidence_emotional_pressure | end_of_question_block | 4.36145641e-01 | 0.00000000e+00 | 1.20476380e+00 | 8.71577963e-02 | 34 |
| evidence_emotional_pressure | end_of_answer_choices | 4.36887741e-01 | 0.00000000e+00 | 1.26974564e+00 | 8.84601968e-02 | 34 |
| evidence_emotional_pressure | end_of_user_message | 2.42451727e+02 | 2.42139381e+02 | 2.57615672e+02 | 6.29395732e+01 | 34 |
| evidence_emotional_pressure | final_answer_position | 1.06522891e+02 | 1.06355335e+02 | 1.37318295e+02 | 2.63687525e+01 | 34 |
| closed_context_false_belief_pressure | end_of_evidence_block | 3.61440787e-01 | 0.00000000e+00 | 1.04165188e+00 | 7.77712402e-02 | 34 |
| closed_context_false_belief_pressure | end_of_question_block | 4.07243090e-01 | 0.00000000e+00 | 1.20476380e+00 | 8.15521886e-02 | 34 |
| closed_context_false_belief_pressure | end_of_answer_choices | 4.07805395e-01 | 0.00000000e+00 | 1.26974564e+00 | 8.24945760e-02 | 34 |
| closed_context_false_belief_pressure | end_of_user_message | 2.59267568e+02 | 2.59341867e+02 | 2.85959064e+02 | 6.51497572e+01 | 34 |
| closed_context_false_belief_pressure | final_answer_position | 9.94864371e+01 | 9.94225046e+01 | 1.41432120e+02 | 2.44221284e+01 | 34 |
| evidence_distractor_neutral | end_of_evidence_block | 4.08466881e-01 | 0.00000000e+00 | 1.04165188e+00 | 8.80836611e-02 | 34 |
| evidence_distractor_neutral | end_of_question_block | 4.62976871e-01 | 0.00000000e+00 | 1.20476380e+00 | 9.26244433e-02 | 34 |
| evidence_distractor_neutral | end_of_answer_choices | 4.69163977e-01 | 0.00000000e+00 | 1.26974564e+00 | 9.46818518e-02 | 34 |
| evidence_distractor_neutral | end_of_user_message | 2.13012644e+02 | 2.13408438e+02 | 2.36796446e+02 | 5.19122896e+01 | 34 |
| evidence_distractor_neutral | final_answer_position | 6.94256204e+01 | 6.91010597e+01 | 8.44562544e+01 | 1.56643291e+01 | 34 |
| evidence_true_belief_pressure | end_of_evidence_block | 2.83996144e-01 | 0.00000000e+00 | 1.04014222e+00 | 6.12915082e-02 | 34 |
| evidence_true_belief_pressure | end_of_question_block | 3.14460279e-01 | 0.00000000e+00 | 1.20274042e+00 | 6.39539276e-02 | 34 |
| evidence_true_belief_pressure | end_of_answer_choices | 3.23137142e-01 | 0.00000000e+00 | 1.26974564e+00 | 6.55605579e-02 | 34 |
| evidence_true_belief_pressure | end_of_user_message | 2.79674714e+02 | 2.78594099e+02 | 3.18340029e+02 | 6.82281635e+01 | 34 |
| evidence_true_belief_pressure | final_answer_position | 8.64771812e+01 | 8.61581879e+01 | 1.01310953e+02 | 2.22384087e+01 | 34 |
| all_pressures_pooled | end_of_evidence_block | 3.44585226e-01 | 0.00000000e+00 | 1.04165188e+00 | 7.43821622e-02 | 34 |
| all_pressures_pooled | end_of_question_block | 3.85949670e-01 | 0.00000000e+00 | 1.20476380e+00 | 7.75546375e-02 | 34 |
| all_pressures_pooled | end_of_answer_choices | 3.89276759e-01 | 0.00000000e+00 | 1.26974564e+00 | 7.88384435e-02 | 34 |
| all_pressures_pooled | end_of_user_message | 2.56981694e+02 | 2.54732258e+02 | 3.01990190e+02 | 6.51299473e+01 | 34 |
| all_pressures_pooled | final_answer_position | 9.94078339e+01 | 9.91192096e+01 | 1.41432120e+02 | 2.45559853e+01 | 34 |

### How to interpret this table

- If raw + token + index identities all hold, delta norms at S0/S1/S2 should be near numerical zero (≤ 1e-4 is the canonical threshold).

- S3 and S4 deltas are **expected** to grow because the condition-specific User message and its resulting ANSWER distribution diverge.

- Any shared-anchor (S0/S1/S2) delta norm materially larger than ~1e-3 indicates either (a) an indexing bug where the condition anchor is off-by-some tokens, or (b) genuine causal contamination / BPE bleed from later-condition tokens changing earlier tokenization, or (c) **family-level numerical artifact** (see diagnosis in §6 below).


## 6. Smoking-gun diagnosis: 22/36 families CLEAN, 14/36 families have shared-anchor deep-layer delta artifact

The median delta norm at each shared anchor is 0.000e+00, while the max is ~1.27 and mean ~0.06–0.09. This is a **bimodal per-family split**: 22 families have exactly zero shared-anchor delta norms at all layers and all conditions; 14 families have a deep-layer delta-norm artifact that grows monotonically with layer index and is identical across **all pressure conditions** (false, emotional, closed_context, true, distractor).

### Diagnostic check on family `schedule_exam_010` (worst outlier, S2 max delta = 1.2697 at L34)

| anchor | ||delta|| at layer 0 | L10 | L20 | L30 | L34 | max |
|---|---|---|---|---|---|---|---|
| end_of_evidence_block | 0.0000e+00 | 5.22e-02 | 1.15e-01 | 0.493 | 0.886 | 0.886 @ L34 |
| end_of_question_block | 0.0000e+00 | 4.06e-02 | 1.05e-01 | 0.508 | 0.996 | 0.996 @ L34 |
| end_of_answer_choices | 0.0000e+00 | 4.20e-02 | 1.35e-01 | 0.654 | 1.270 | 1.270 @ L34 |

Crucially, for this family at every shared anchor, the delta vectors for **evidence_false_belief_pressure**, **evidence_emotional_pressure**, and **closed_context_false_belief_pressure** (and all 5 non-neutral conditions) are numerically identical:
- Cosine similarity between condition-delta vectors at S2: **= 1.000000** (all pairs)
- RMSE(delta_false_belief − delta_emotional) at S2: **0.0000e+00** (identical tensors, not approximate)

This pattern is **inconsistent with any causal pressure effect** (which would have to depend on the condition text inside the User message block). Instead it indicates a **family-level numerical artifact**: the act of appending any non-empty User message block shifts the deep-layer hidden representations of earlier (shared-prefix) token positions by a per-family, condition-invariant vector. The magnitude is small at input layers (||delta|| ≈ 0) and grows cumulatively through the residual stream up to ||delta|| ≈ 0.89–1.27 at L34.

### Family partition (closed_context_false_belief_pressure, S2 layer-34 delta norm)

**CLEAN (n=22, max_delta < 1e-3, all layers + all conditions exactly 0):**
`academic_grading_018, academic_latework_020, contract_ticket_032, contract_warranty_031, finance_budget_007, finance_coupon_006, finance_refund_005, logic_access_033, policy_lab_access_002, policy_library_checkout_003, policy_lunch_pass_001, product_specs_025, product_specs_026, product_specs_027, product_specs_028, schedule_clinic_011, schedule_train_009, schedule_workshop_012, science_battery_016, science_plant_013, technical_backup_023, technical_config_021.`

**OUTLIER (n=14, S2 L34 delta > 0.86, deep-layer artifact):**
`schedule_exam_010 (1.270), contract_service_030 (1.183), finance_membership_008 (1.161), policy_rec_center_004 (1.136), academic_prereq_017 (1.101), logic_tournament_036 (1.051), contract_rental_029 (1.020), logic_shipping_035 (1.018), logic_eligibility_034 (1.009), academic_attendance_019 (1.007), science_weather_015 (0.992), science_mineral_014 (0.979), technical_feature_flag_024 (0.893), technical_api_022 (0.861).`

This split is stable across anchors (S0/S1/S2) and across all 5 non-neutral conditions: if a family is an outlier for closed_context at S2 L34, it is an outlier with numerically identical vectors for evidence_false_belief, emotional, true_belief, and distractor conditions. No family has nonzero deltas for a subset of conditions only.

### Rule-out checks (all passed)
1. Layer aliasing (MPS bug): `||h_neutral(L35) − h_neutral(L0)||_2` on evidence_neutral = 186–205 for every family (no aliasing, 0/36 corrupted).
2. Raw prefix identity: 36/36 identical (S0, S1, S2) for all conditions.
3. Token prefix identity: 36/36 identical for all conditions.
4. Anchor token index identity: 36/36 identical across all conditions within each family.

Therefore the cause is **not** structural (it's not text, tokens, anchors, indexing, or layer-aliasing) — it is a numerical residual drift specific to ~40% of families on this extraction hardware, localized to deep layers (L25–L35), condition-agnostic, and confined to the old activation root.

### Does this artifact explain the high S0/S1/S2 BAs in the Probe 6B report?

Partially. In the canonical probe code (lines 1167–1265 of [probe6b_matched_prefix_detection.py](file:///Users/ericmin/NonTrivial/src/analysis/probe6b_matched_prefix_detection.py#L1167-L1265)), the main overall/within/cross probes use **delta features (h_cond − h_neu) only at S3 and S4**, never at S0/S1/S2. Those S3/S4 results (BA 0.661 at S3; 0.676 at S4) are **not contaminated by the artifact**, because S3/S4 condition-specific signals dwarf the family-level invariant drift (||S3 delta|| ≈ 213–280 vs ||artifact|| ≈ 1).

The shared-anchor high BAs reported in the main summary come from two places:
- **Neutral vulnerability baseline** (Section D of the summary): uses raw h_neutral at S0/S1/S2, allowed by construction. BA 0.839 @ S0 closed_context is a valid vulnerability number.
- **overall_harmful rows visible at S0/S1/S2 in on-disk best/layerwise CSVs**: these were produced by an earlier code path that fed raw h_condition features (not delta features) across all anchors, and/or by the artifact-affected 14/36 families enabling family-level memorization by the centroid probe (the 14 vs 22 partition is linearly separable in deep layers).

## 7. Feature construction used in each Probe 6B result class


| analysis name | anchors run | feature tensor | per-example semantics | source code location |
|---|---|---|---|---|

| neutral_vulnerability | S0/S1/S2 only | raw h_neutral at shared anchor | family-level example only; target y = 1 if family later becomes harmful under the condition | probe6b_matched_prefix_detection.py lines 1139-1165 (A. Vulnerability baseline) |

| overall_harmful / within_condition / cross_condition (canonical runner) | S3 (end of user message) and S4 (final answer) only | h_condition(anchor) − h_neutral(anchor) | delta feature per (family, condition) pair | lines 1167-1265 (B/C/D) |

| overall_harmful / within_condition / cross_condition (rebuilt missing CSVs in .dbg_rebuild script) | same S3/S4 only | h_condition(anchor) − h_neutral(anchor) | delta feature per (family, condition) pair | .dbg_rebuild_probe6b_missing.py lines 325-475 |


### What about the overall_harmful S0/S1/S2 rows visible in `probe6b_matched_prefix_layerwise.csv` / best.csv?

Those rows came from **raw** `h_condition` features (no neutral subtraction) across all anchors, used by the earlier matched-prefix extraction pipeline (line 2 of layerwise.csv: `overall_harmful,end_of_evidence_block,all_conditions_pooled,0,...`) and/or the `anchor_integrity_summary` regeneration code path that produced the on-disk `best.csv` before the integrity-threshold-raise.

In the canonical Probe 6B code path, the main overall/within/cross probes **never run at S0/S1/S2** (they iterate only `for anchor in DETECTION_ANCHORS` = S3 and S4, lines 1168, 1199, 1225-1226 of the runner).


## 8. Interpretation guidance

- **Allowed**: `h_neutral(S0/S1/S2)` → y(later harm) counts as **vulnerability prediction** and is a valid paper result.
- **Not allowed (as pressure detection)**: Any classifier trained on `h_condition(S0/S1/S2) − h_neutral(S0/S1/S2)` that performs above chance. Because of the 14-family condition-invariant artifact, nonzero performance at shared anchors with delta features reflects the family-level numerical split, not condition-specific pressure. Report separately as an integrity failure.
- **Valid (core paper results)**: S3/S4 LOGO and cross-condition classifiers using `h_condition(anchor) − h_neutral(anchor)` delta features. S3 BA 0.661, S4 BA 0.676, S3 permutation-control p = 0.0099 remain the headline results.
- **Structural matched-prefix design passed 36/36**: Raw text, token IDs, and anchor token indices all match identically within every family. The design itself is correct; only the numerical backend produced a per-family artifact.
- **Recommendation for HQ80**: Run with CPU float32 (ongoing, started 2026-08-06). Our 8-prompt smoke test showed `||delta_h_condition − delta_h_neutral|| = 0.0000e+00` at all shared anchors and all layers, including L34, for all 7 non-neutral conditions — confirming the 14-family artifact does **not** appear under CPU float32 extraction. Under CPU, only S3/S4 will grow nonzero deltas, as required by the causal setup.

