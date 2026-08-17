# HQ80 Qwen Hidden-State Technical Summary

Feature configuration: REDUCED anchors=S3, layers_per_anchor=[20, 24, 28, 32, 35], feat_dim=12800
N_families: 36

## Step 1 — Mean peak Δnorm by condition at S3 / S4

| Condition | Anchor | Peak layer | Mean Δnorm | Median Δnorm |
|---|---|---:|---:|---:|
| False belief | S3 | 34 | 249.472 | 248.134 |
| False belief | S4 | 34 | 87.059 | 86.661 |
| False rationale | S3 | 34 | 233.188 | 230.862 |
| False rationale | S4 | 34 | 101.619 | 97.379 |
| Emotional | S3 | 34 | 251.941 | 250.248 |
| Emotional | S4 | 34 | 128.764 | 126.225 |
| Authority | S3 | 34 | 251.722 | 247.501 |
| Authority | S4 | 34 | 106.736 | 104.700 |
| True belief | S3 | 34 | 258.581 | 257.511 |
| True belief | S4 | 34 | 88.090 | 87.621 |
| True rationale | S3 | 34 | 254.237 | 251.866 |
| True rationale | S4 | 34 | 93.928 | 91.545 |
| Distractor | S3 | 34 | 170.739 | 167.596 |
| Distractor | S4 | 34 | 38.457 | 38.179 |

## Step 3 — Strongest pooled false-pressure Δnorm↔degradation correlation

- Strongest: group=pooled_false_pressure, anchor=S3, layer=0, pearson=-0.755556, spearman=-0.648932, n=320

## Step 4 — Probe results

- task1_pooled_harmful_vs_nonharmful: balanced_acc mean=0.9943 (best fold=1.0000)
- task2a_FB_degraded: balanced_acc mean=0.7411 (best fold=0.8125)
- task2b_FR_degraded: balanced_acc mean=0.6678 (best fold=0.8214)
- task2e_distr_degraded: balanced_acc mean=0.7333 (best fold=0.8667)
- task3_ordinal_false_pressure_strength: balanced_acc mean=0.9969 (best fold=1.0000)
- task4_pressure_perturb_vs_distractor: balanced_acc mean=1.0000 (best fold=1.0000)

### Permutation controls
- task1_pooled_harmful_vs_nonharmful: true_bal_acc=0.9943, perm_z=18.732, p_est<=0.0323 (N_perm=30)
- task3_ordinal_false_pressure_strength: true_bal_acc=0.9969, perm_z=20.747, p_est<=0.0323 (N_perm=30)
- task4_pressure_perturb_vs_distractor: true_bal_acc=1.0000, perm_z=15.789, p_est<=0.0323 (N_perm=30)

