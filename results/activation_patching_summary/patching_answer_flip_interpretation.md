# Activation Patching — Answer Flip Report (Step 5)

Answer-choice change: predicted = correct iff margin > 0.
- `original_pressure_pred_correct`: original pressure prompt predicted A/B = correct?
- `patched_pred_correct`: patched run predicted = correct?
- Rescue 'flipped rescued': original_wrong (≤0) → patched_correct (>0)
- Transfer 'flipped degraded': original_neutral_correct (>0) → patched_wrong (≤0)

## Rescue flip summary (degraded families)
| Condition | Layer | n | originally wrong (pressure margin ≤ 0) | rescued flips → correct | opposite flips → wrong |
|---|---:|---:|---:|---:|---:|
| Closed-Context | 8 | 12 | 1 | 0 | 0 |
| Closed-Context | 20 | 12 | 1 | 1 | 0 |
| Closed-Context | 28 | 12 | 1 | 1 | 0 |
| Closed-Context | 30 | 12 | 1 | 1 | 0 |
| Closed-Context | 32 | 12 | 1 | 1 | 0 |
| Closed-Context | 34 | 12 | 1 | 1 | 0 |
| Closed-Context | 35 | 12 | 1 | 1 | 0 |
| Emotional | 8 | 12 | 0 | 0 | 0 |
| Emotional | 20 | 12 | 0 | 0 | 0 |
| Emotional | 28 | 12 | 0 | 0 | 0 |
| Emotional | 30 | 12 | 0 | 0 | 0 |
| Emotional | 32 | 12 | 0 | 0 | 0 |
| Emotional | 34 | 12 | 0 | 0 | 0 |
| Emotional | 35 | 12 | 0 | 0 | 0 |
| False Belief | 8 | 12 | 0 | 0 | 0 |
| False Belief | 20 | 12 | 0 | 0 | 0 |
| False Belief | 28 | 12 | 0 | 0 | 0 |
| False Belief | 30 | 12 | 0 | 0 | 0 |
| False Belief | 32 | 12 | 0 | 0 | 0 |
| False Belief | 34 | 12 | 0 | 0 | 0 |
| False Belief | 35 | 12 | 0 | 0 | 0 |

## Transfer flip summary (degraded families)
| Condition | Layer | n | originally correct neutral (margin > 0) | transfer flips → false answer | opposite flips → correct (against expectation) |
|---|---:|---:|---:|---:|---:|
| Closed-Context | 8 | 12 | 12 | 0 | 0 |
| Closed-Context | 20 | 12 | 12 | 0 | 0 |
| Closed-Context | 28 | 12 | 12 | 1 | 0 |
| Closed-Context | 30 | 12 | 12 | 1 | 0 |
| Closed-Context | 32 | 12 | 12 | 1 | 0 |
| Closed-Context | 34 | 12 | 12 | 1 | 0 |
| Closed-Context | 35 | 12 | 12 | 1 | 0 |
| Emotional | 8 | 12 | 12 | 0 | 0 |
| Emotional | 20 | 12 | 12 | 0 | 0 |
| Emotional | 28 | 12 | 12 | 0 | 0 |
| Emotional | 30 | 12 | 12 | 0 | 0 |
| Emotional | 32 | 12 | 12 | 0 | 0 |
| Emotional | 34 | 12 | 12 | 0 | 0 |
| Emotional | 35 | 12 | 12 | 1 | 0 |
| False Belief | 8 | 12 | 12 | 0 | 0 |
| False Belief | 20 | 12 | 12 | 0 | 0 |
| False Belief | 28 | 12 | 12 | 0 | 0 |
| False Belief | 30 | 12 | 12 | 0 | 0 |
| False Belief | 32 | 12 | 12 | 0 | 0 |
| False Belief | 34 | 12 | 12 | 0 | 0 |
| False Belief | 35 | 12 | 12 | 0 | 0 |

Totals across degraded families (3 cond × 7 layers × families per condition, n families differ):
- Rescue: original-wrong pressure rows total = 7; rescued-flip to correct total = 6; anti-flip to wrong = 0
- Transfer: original-correct neutral rows total = 252; transfer-flip to false total = 6; anti-flip = 0

Interpretation: most of the rescue/transfer effect will usually be *margins*, not full answer flips, because behavioral margins in original36 are often still positive (just reduced) under pressure. A substantial flip count is not required for a causal claim; consistent margin shifts are the primary evidence.
