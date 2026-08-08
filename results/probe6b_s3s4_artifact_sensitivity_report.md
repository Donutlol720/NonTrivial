# Probe 6B S3/S4 artifact-sensitivity audit (behavior margins recomputed correctly)

## Behavior recomputation

The serialized `logits_last_token` in OLD activations was corrupted by MPS f16 overflow. We recomputed per-prompt margins by applying the lm_head weight on CPU float32 to the saved final-layer (L35) hidden state at the final-answer-position anchor (`final_answer_position L35 @ lm_head`), which is exactly the workaround used in the HQ80 runner.

Token choice strategy: `lead_space: A, B` (A_id=362, B_id=425).

## Family splits

- CLEAN families (n=22): shared-anchor delta norms exactly 0.
  `academic_grading_018`, `academic_latework_020`, `contract_ticket_032`, `contract_warranty_031`, `finance_budget_007`, `finance_coupon_006`, `finance_refund_005`, `logic_access_033`, `policy_lab_access_002`, `policy_library_checkout_003`, `policy_lunch_pass_001`, `product_specs_025`, `product_specs_026`, `product_specs_027`, `product_specs_028`, `schedule_clinic_011`, `schedule_train_009`, `schedule_workshop_012`, `science_battery_016`, `science_plant_013`, `technical_backup_023`, `technical_config_021`

- OUTLIER families (n=14): condition-invariant S2 L34 delta = 0.86–1.27.
  `academic_attendance_019`, `academic_prereq_017`, `contract_rental_029`, `contract_service_030`, `finance_membership_008`, `logic_eligibility_034`, `logic_shipping_035`, `logic_tournament_036`, `policy_rec_center_004`, `schedule_exam_010`, `science_mineral_014`, `science_weather_015`, `technical_api_022`, `technical_feature_flag_024`

## Behavior by split (N / n_harmful / n_nonharmful per pressure condition)

| split | condition | N | n_harmful | n_nonharmful | mean(Δmargin) | median(Δmargin) |
|---|---|---|---|---|---|---|

| ALL | evidence_false_belief_pressure | 36 | 16 | 20 | -0.066 | +0.342 |
| ALL | evidence_emotional_pressure | 36 | 36 | 0 | -4.007 | -3.765 |
| ALL | closed_context_false_belief_pressure | 36 | 35 | 1 | -3.195 | -2.847 |
| CLEAN | evidence_false_belief_pressure | 22 | 7 | 15 | +0.327 | +0.507 |
| CLEAN | evidence_emotional_pressure | 22 | 22 | 0 | -3.665 | -3.658 |
| CLEAN | closed_context_false_belief_pressure | 22 | 21 | 1 | -2.612 | -2.413 |
| OUTLIER | evidence_false_belief_pressure | 14 | 9 | 5 | -0.684 | -0.194 |
| OUTLIER | evidence_emotional_pressure | 14 | 14 | 0 | -4.544 | -4.374 |
| OUTLIER | closed_context_false_belief_pressure | 14 | 14 | 0 | -4.112 | -3.673 |

## Feature modes compared

- `delta` (baseline): `h_condition(anchor) − h_neutral(anchor)`.

- `corrected_delta`: `h_condition(anchor) − h_neutral(anchor) − (h_condition(S2) − h_neutral(S2))` (subtracts the per-family condition-invariant S2 artifact).

- Splits compared: ALL 36 fams, CLEAN 22 fams, OUTLIER 14 fams.


## Per-analysis best-layer balanced accuracy (3-way comparison)

| analysis | anchor | orig_BA_all | orig_layer | orig_support | clean_BA | clean_layer | clean_support | outlier_BA | outlier_layer | outlier_support | corrected_BA_all | corrected_layer | corrected_support |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_evidence_emotional_pressure_to_closed_context_false_belief_pressure | S3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| cross_evidence_emotional_pressure_to_closed_context_false_belief_pressure | S4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| cross_evidence_false_belief_pressure_to_closed_context_false_belief_pressure | S3 | 0.800 | L25 | supported | 0.786 | L20 | supported | 0.857 | L14 | supported | 0.800 | L25 | supported |
| cross_evidence_false_belief_pressure_to_closed_context_false_belief_pressure | S4 | 1.000 | L35 | supported | 1.000 | L35 | supported | 1.000 | L35 | supported | 1.000 | L35 | supported |
| cross_evidence_false_belief_pressure_to_evidence_emotional_pressure | S3 | 1.000 | L27 | supported | 1.000 | L27 | supported | 1.000 | L21 | supported | 1.000 | L27 | supported |
| cross_evidence_false_belief_pressure_to_evidence_emotional_pressure | S4 | 0.972 | L35 | supported | 1.000 | L35 | supported | 1.000 | L14 | supported | 0.972 | L35 | supported |
| overall_harmful_pooled | S3 | 0.827 | L1 | supported | 0.809 | L1 | supported | 0.851 | L1 | supported | 0.827 | L1 | supported |
| overall_harmful_pooled | S4 | 0.878 | L7 | supported | 0.889 | L6 | supported | 0.878 | L8 | supported | 0.878 | L7 | supported |
| within_closed_context_false_belief_pressure | S3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| within_closed_context_false_belief_pressure | S4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| within_evidence_emotional_pressure | S3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| within_evidence_emotional_pressure | S4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| within_evidence_false_belief_pressure | S3 | 0.744 | L18 | supported | 0.757 | L22 | supported | 0.889 | L28 | supported | 0.744 | L18 | supported |
| within_evidence_false_belief_pressure | S4 | 0.825 | L35 | supported | 0.757 | L34 | supported | 0.833 | L35 | supported | 0.825 | L35 | supported |

## Supporting CSVs

- `probe6b_s3s4_clean_vs_outlier_results.csv`: full 36-layer rows for CLEAN-vs-OUTLIER-vs-ALL sensitivity test (baseline delta feature mode only).

- `probe6b_s3s4_artifact_corrected_results.csv`: full 36-layer rows on ALL split, comparing `delta` vs `corrected_delta`.


## Interpretation

- **Survives artifact-correction (primary criterion)**: if corrected_delta BA is ≥ 0.60, and close to baseline BA, the S3/S4 pressure signal is not driven by the 14-family S2 artifact.

- **Survives clean-only split (secondary criterion)**: if CLEAN-only BA matches ALL-only BA within ±0.04, the result is robust to removing the 14 outlier families entirely.

- **OUTLIER families alone**: if BA in outlier families only is much higher than CLEAN-only BA, the pooled result may be outlier-driven and should be reported split-by-split.

- **Permutation control (unchanged)**: the strongest ALL-split S3 result (baseline delta) should be compared against the canonical p=0.0099 100-permutation result reported earlier; artifact-corrected variant retains the same test if BA is similar.

