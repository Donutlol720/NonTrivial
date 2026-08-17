# Activation Patching — Final Causal Interpretation (Qwen3-4B / original36)

**Interpretation label: Strong causal evidence**

## 1. What dataset/model was patched?
- Dataset: original36 36-family matched-prefix prompt set (`data/generated_prompts_v1.jsonl`), subset-selection filters from canonical `qwen3_4b_instruct_2507_family36_family_margin_deltas.csv`.
- Model: Qwen/Qwen3-4B-Instruct-2507 (Qwen-only; CPU device).

## 2. What layers and anchors were patched?
- Anchor: single anchor, original36 extraction schema = `hidden_states_final_token` (HQ80 equivalent = anchor S4, final prompt / answer-reading token position). No S3 (end-of-user-message) or earlier anchors were patched.
- Layers: 7 layers — [8, 20, 28, 30, 32, 34, 35] (early controls: L8, L20; late candidate band: L28, L30, L32, L34, L35).

## 3. Did neutral activations rescue pressure-induced degradation?
- Rescue (positive = restored margin): late L28–L35, degraded families, mean across 3 conditions = **2.0468**. Expected-direction fraction (mean over condition×layer rows) = 0.774.

| Condition | Peak layer | mean rescue (degraded) | n families |
|---|---:|---:|---:|
| Closed-Context | L34 | 5.8265 | 12 |
| Emotional | L34 | 4.1149 | 12 |
| False Belief | L34 | 2.9966 | 12 |

## 4. Did pressure activations transfer degradation into neutral prompts?
- Transfer (negative = degradation transferred): late L28–L35, degraded families, mean across 3 conditions = **-4.1274**. Expected-direction fraction = 0.905.

| Condition | Peak layer | mean transfer (degraded) | n families |
|---|---:|---:|---:|
| Closed-Context | L35 | -9.0405 | 12 |
| Emotional | L35 | -8.1222 | 12 |
| False Belief | L35 | -7.2801 | 12 |

## 5. Were the effects larger than controls?
- Control families = same-condition, disjoint-family subsets filtered by smallest |delta_margin| behavioral near-zero rows.
- Late-layer degraded > control in expected-signed metric survival rate: **1.000 (30/30)**. Early-layer comparison survival = 0.917.
- Full Step-4 breakdown in `patching_control_interpretation.md`.

## 6. Were the effects concentrated in predicted layers/anchors?
- 100.0% of rescue+transfer top-2 condition peaks fall in the late-layer predicted band L28–L35. This matches prior hidden-state delta-norm, correlation, and HQ80-probe layer peaks.
- Anchor: single S4-equivalent (final-token). S3 was not available in original36 extraction schema; S3 vs S4 comparison requires HQ80-based patch run.

## 7. Did patching change actual answer flips, or mostly margins?
- Rescue (degraded families × all layers): 6 answer flips rescued from originally-wrong pressure rows; opposite-direction flips (rescue made originally-correct → wrong) = 0.
- Transfer (degraded families × all layers): 6 neutral correct → false flips transferred; opposite-direction (bad neutral → correct) = 0.
- Primary effect is margin shifts; answer flips are rare in this bounded sweep because most pressure prompts still select the correct answer even with degraded margins.

## 8. Does this support a causal claim?

### Label: **Strong causal evidence**

Decision criteria:
- Rescue late-layer positive? True (2.0468)
- Transfer late-layer negative? True (-4.1274)
- Top-2 condition-layer peaks concentrated in late band? YES (frac = 1.000)
- Late-layer degraded > control survival rate ≥50% AND ≥ early-layer rate? YES (late=1.000; early=0.917)

## 9. Cautious paper wording

> Activation patching provides intervention evidence that pressure-related hidden states contribute to evidence-aligned margin degradation.

Additional cautions:
- Results are Qwen-only, original36-only. Do not generalize across models without replication.
- Only the S4-equivalent (final-token) anchor was patched; anchor/S3-vs-S4 sweep is required to localize effect onset.
- Control used here = same-condition near-zero-|delta| families (not random-family or irrelevant-condition patches).
- 36 families total, degraded subsets of 7–12 families per condition: confidence intervals are wide; consider pooling with HQ80 family patches to narrow claims.

## 10. Figure / table recommendation

- **Main text figure 1**: `activation_patching_rescue_transfer_summary.pdf/png` — grouped bars for late L28–L35 averaged rescue (positive) + transfer (negative), one pair per pressure condition, SEM error bars. Positive rescue + negative transfer visually reads as the expected bidirectional intervention evidence.
- **Main text figure 2** (optional but recommended for reviewers): `activation_patching_real_vs_control.pdf/png` — expected-signed real degraded-family effect (blue) vs control near-zero-|delta| family effect (gray), one pair per condition per patch type. Late layers, L28–L35.
- **Appendix figures**: `activation_patching_rescue_by_layer.pdf/png` and `activation_patching_transfer_by_layer.pdf/png` — full layer × condition line plots with zero line.
- **Appendix table** (use CSVs): condition × layer × degraded/control means, CIs, fraction expected direction, p-values for rescue and transfer separately; one table per patch type.
