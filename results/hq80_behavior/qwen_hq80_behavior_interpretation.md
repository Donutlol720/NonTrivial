# HQ80 Qwen Behavioral Interpretation

## Q1: Does HQ80 replicate main Qwen margin-degradation story?

**Answer: Yes** (4/4 false-pressure conditions have fraction_negative > 0.5 AND mean delta < 0).

| Condition | Mean delta | Fraction negative |
|---|---|---|
| False belief | -0.5107 | 0.537 (43/80) |
| False rationale | -1.5898 | 0.850 (68/80) |
| Emotional | -7.9969 | 1.000 (80/80) |
| Authority | -9.5814 | 1.000 (80/80) |

## Q2: Stronger pressure types vs bare false-belief?

Ranking from strongest (most negative mean delta) to weakest:

1. **Authority**: mean=-9.5814, median=-8.7120
2. **Emotional**: mean=-7.9969, median=-7.2849
3. **False rationale**: mean=-1.5898, median=-1.4963
4. **False belief**: mean=-0.5107, median=-0.0722

Strongest pressure: **Authority**

## Q3: FB vs FR vs EM vs AU — detail table

| Pressure | Mean delta | Fraction negative | N negative |
|---|---|---|---|
| False belief | -0.5107 | 0.537 | 43 |
| False rationale | -1.5898 | 0.850 | 68 |
| Emotional | -7.9969 | 1.000 | 80 |
| Authority | -9.5814 | 1.000 | 80 |

## Q4: Does distractor remain control-like?

- Distractor mean delta: -1.1580
- Distractor fraction negative: 0.938 (75/80)
- False-belief mean delta (mildest false pressure): -0.5107
- Worst (most negative) false-pressure mean (authority): -9.5814

**Answer: Partially — distractor is substantially less harmful than the strong false pressures (emotional and authority), but it does NOT behave like a clean neutral control.**
- Distractor mean Δ = −1.16 is milder than emotional (−8.0) or authority (−9.6) by ~7–8×, but the sign is consistently negative and affects 94% of families (75/80), which is not behavior consistent with a true inert control. False-belief pressure (mean Δ = −0.51) is actually milder than distractor; distractor is therefore not a floor effect.
- Mean near 0? False (|-1.1580| < 1.0)
- Fraction near 0.5? False (0.938)
- Less negative than emotional + authority only? True
- Less negative than ALL 4 false pressures (incl. false-belief & false-rationale)? False

## Q5: Are answer flips still rare?

- Total negative_delta across 4 false pressures: 271
- Total answer_flips across 4 false pressures: 8
- Flips per 100 negatives: 2.95

**Answer: Yes** — flips per 100 negatives = 2.95

## Q6: Do true-belief / true-rationale pressure INCREASE the margin?

- True belief: mean_delta=3.0882, fraction_positive=0.925 (74/80)
- True rationale: mean_delta=3.0116, fraction_positive=0.875 (70/80)

**True belief**: INCREASES margin (mean positive, >50% positive)
**True rationale**: INCREASES margin (mean positive, >50% positive)

## Q7: Matched-prefix design — compare to original36

| Condition | Original36 mean Δ | HQ80 mean Δ | Direction consistent? | Magnitude ratio (HQ80/Orig36) |
|---|---|---|---|---|
| False belief | 0.1871 | -0.5107 | No | -2.730 |
| Emotional | -0.9731 | -7.9969 | Yes | 8.218 |

**Answer: Matched-prefix design strengthens the original36 conclusion**
- FB: |HQ80|=0.5107 vs |Orig36|=0.1871
- EM: |HQ80|=7.9969 vs |Orig36|=0.9731

