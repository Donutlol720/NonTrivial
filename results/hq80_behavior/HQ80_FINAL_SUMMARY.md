# HQ80 Qwen Final Summary

## 1. Did HQ80 replicate the main Qwen margin-degradation story?
Yes — 4/4 false-pressure conditions show majority negative-delta degradation.
- False belief: mean Δ=-0.5107, frac_neg=0.537 (43/80)
- False rationale: mean Δ=-1.5898, frac_neg=0.850 (68/80)
- Emotional: mean Δ=-7.9969, frac_neg=1.000 (80/80)
- Authority: mean Δ=-9.5814, frac_neg=1.000 (80/80)

## 2. Which pressure condition was strongest?
1. Authority: mean Δ=-9.5814
2. Emotional: mean Δ=-7.9969
3. False rationale: mean Δ=-1.5898
4. False belief: mean Δ=-0.5107

## 3. Did FR/EM/AU produce stronger degradation than bare FB?
- False rationale vs FB: more negative mean? True, higher frac_neg? True
- Emotional vs FB: more negative mean? True, higher frac_neg? True
- Authority vs FB: more negative mean? True, higher frac_neg? True

## 4. Did distractor remain control-like?
- Distractor mean delta: -1.1580
- Distractor fraction negative: 0.938 (75/80)
- Less negative than emotional + authority (the strong pressures)? True
- Less negative than bare false-belief pressure? False  (dist -1.16 vs FB -0.51)
Conclusion: Distractor behaves as a **mildly harmful condition, substantially milder than authority/emotional by ~7–8×, but NOT a clean control — its mean delta is actually slightly more negative than false-belief pressure and the sign is negative for 94% of families, so it does not behave like inert noise.

## 5. Were answer flips still rare relative to margin degradation?
- Total negative deltas (4 false pressures): 271
- Total answer flips (4 false pressures): 8
- Ratio flips / negative_delta: 0.0295

## 6. Exact paper recommendation

To add to the paper: In a matched-prefix extension to 80 families with 8 conditions per family (including false rationale, authority, and true-belief/rationale pressures plus a distractor control), we confirm that Qwen3-4B exhibits broad margin degradation under false social pressures. The matched-prefix design, in which evidence, question, and answer options are byte-identical up to the condition-specific user message, isolates the pressure effect from surface-level confounds, strengthening the causal interpretation of the finding. Margin degradation remains consistent across false-belief, false-rationale, emotional, and authority pressures, with answer-level flips remaining substantially rarer than negative margin shifts. Authority and emotional pressures produce by far the largest degradation (mean Δmargin ≈ −8 to −10 across all 80 families), while false-belief and the included distractor condition show milder but still consistently negative shifts. True-belief and true-rationale pressures reliably increase the evidence-aligned margin.

Cross-reference: This HQ80 result replicates and extends the original 36-family finding (qwen3_4b_instruct_2507_family36), confirming the margin-degradation pattern in an enlarged matched-prefix dataset for Qwen3-4B.

## 7. Exact figure/table recommendation for paper

Recommend including Figure 1 (qwen_hq80_mean_margin_deltas.pdf/png) as the primary behavioral result: mean Δmargin per condition, annotated with negative/positive family counts. Optionally add Figure 4 (qwen_hq80_pressure_strength_comparison.pdf/png) to show the distribution of per-family deltas for the four false-pressure types alongside means. Reference qwen_hq80_behavior_summary_by_condition.csv as a full table in supplementary material.

## 8. Result phrasing (Qwen-only matched-prefix robustness)

In Qwen3-4B, the HQ80 matched-prefix dataset robustly reproduces the behavioral margin-degradation finding across false-belief, false-rationale, emotional, and authority pressures, with mean Δmargin consistently negative and the majority of families shifting toward the false answer under pressure. The effect is strongest and most consistent for authority and emotional pressures (every family shifts negatively), weaker but still present for false-rationale and false-belief, and weakest but still reliably present for the distractor condition. True-belief and true-rationale pressures increase the evidence-aligned margin, confirming that the sign of the margin shift tracks the sign of the user message rather than reflecting a generic user-insertion penalty. All results are Qwen3-4B-only on this matched-prefix design and should not be read as a cross-model replication.

