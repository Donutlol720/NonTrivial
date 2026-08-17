# Activation Patching — Control Comparisons (Step 4)

Control in this bounded sweep = same-condition, different-family subset selected by the smallest-|delta_margin| filter (behaviorally near-zero- or slight-positive-delta families; disjoint from degraded families per condition).

Interpretation: causal claim is stronger when degraded-family patching effect (in expected direction) >> control patching effect at the same layer/condition.

Legend:
- real_minus_control_expected_signed = (degraded effect in expected dir) − (control effect in expected dir). Positive → degraded survives control.
- p(Mann-Whitney one-sided) tests degraded ≥ control in expected-signed metric.

| Condition | Patch type | Layer | mean degraded | mean control | degraded − control (expected-signed) | p(MWU ≥) | survives? |
|---|---|---:|---:|---:|---:|---:|---|
| Closed-Context | rescue | 8 | 0.2811 | 0.3043 | -0.0232 | 0.48357 | no |
| Closed-Context | rescue | 20 | 1.1427 | 0.1156 | 1.0271 | 0.03417 | **YES** |
| Closed-Context | rescue | 28 | 3.0617 | -0.9243 | 3.9861 | 0.00004 | **YES** |
| Closed-Context | rescue | 30 | 3.7273 | -0.4001 | 4.1274 | 0.00002 | **YES** |
| Closed-Context | rescue | 32 | 4.8711 | 0.8653 | 4.0057 | 0.00002 | **YES** |
| Closed-Context | rescue | 34 | 5.8265 | 1.2703 | 4.5562 | 0.00002 | **YES** |
| Closed-Context | rescue | 35 | -1.3054 | -6.8053 | 5.4999 | 0.00002 | **YES** |
| Closed-Context | transfer | 8 | -0.1674 | -0.1061 | 0.0613 | 0.05013 | **YES** |
| Closed-Context | transfer | 20 | -1.6313 | -0.6859 | 0.9454 | 0.00038 | **YES** |
| Closed-Context | transfer | 28 | -3.2802 | -0.0276 | 3.2526 | 0.00002 | **YES** |
| Closed-Context | transfer | 30 | -3.7518 | -0.2851 | 3.4667 | 0.00002 | **YES** |
| Closed-Context | transfer | 32 | -4.6187 | -1.0526 | 3.5660 | 0.00002 | **YES** |
| Closed-Context | transfer | 34 | -5.3007 | -1.5922 | 3.7086 | 0.00002 | **YES** |
| Closed-Context | transfer | 35 | -9.0405 | -5.5111 | 3.5294 | 0.00002 | **YES** |
| Emotional | rescue | 8 | -0.0999 | -0.1051 | 0.0052 | 0.60249 | **YES** |
| Emotional | rescue | 20 | 0.9821 | 0.4000 | 0.5821 | 0.01311 | **YES** |
| Emotional | rescue | 28 | 3.0964 | 0.8390 | 2.2574 | 0.00002 | **YES** |
| Emotional | rescue | 30 | 3.3376 | 1.0871 | 2.2504 | 0.00002 | **YES** |
| Emotional | rescue | 32 | 3.6386 | 1.4384 | 2.2002 | 0.00002 | **YES** |
| Emotional | rescue | 34 | 4.1149 | 1.5756 | 2.5393 | 0.00002 | **YES** |
| Emotional | rescue | 35 | -3.0453 | -6.7316 | 3.6864 | 0.00002 | **YES** |
| Emotional | transfer | 8 | 0.1169 | 0.1233 | 0.0064 | 0.29168 | **YES** |
| Emotional | transfer | 20 | -1.4231 | -0.6995 | 0.7236 | 0.00305 | **YES** |
| Emotional | transfer | 28 | -2.7540 | -0.8285 | 1.9255 | 0.00002 | **YES** |
| Emotional | transfer | 30 | -2.9121 | -0.9322 | 1.9799 | 0.00002 | **YES** |
| Emotional | transfer | 32 | -3.2542 | -1.2354 | 2.0188 | 0.00002 | **YES** |
| Emotional | transfer | 34 | -3.7948 | -1.7746 | 2.0202 | 0.00002 | **YES** |
| Emotional | transfer | 35 | -8.1222 | -6.2723 | 1.8499 | 0.00002 | **YES** |
| False Belief | rescue | 8 | 0.0444 | 0.0057 | 0.0387 | 0.07861 | **YES** |
| False Belief | rescue | 20 | 0.4780 | -0.1628 | 0.6408 | 0.02020 | **YES** |
| False Belief | rescue | 28 | 0.9436 | -0.8031 | 1.7467 | 0.00008 | **YES** |
| False Belief | rescue | 30 | 1.2702 | -0.5827 | 1.8528 | 0.00003 | **YES** |
| False Belief | rescue | 32 | 2.2508 | 0.3981 | 1.8527 | 0.00002 | **YES** |
| False Belief | rescue | 34 | 2.9966 | 0.9542 | 2.0424 | 0.00002 | **YES** |
| False Belief | rescue | 35 | -4.0824 | -6.3819 | 2.2995 | 0.00015 | **YES** |
| False Belief | transfer | 8 | 0.0813 | 0.1100 | 0.0287 | 0.03913 | **YES** |
| False Belief | transfer | 20 | -0.8390 | -0.3318 | 0.5072 | 0.00431 | **YES** |
| False Belief | transfer | 28 | -1.2565 | 0.4214 | 1.6779 | 0.00005 | **YES** |
| False Belief | transfer | 30 | -1.4597 | 0.3325 | 1.7923 | 0.00002 | **YES** |
| False Belief | transfer | 32 | -2.1980 | -0.4185 | 1.7795 | 0.00002 | **YES** |
| False Belief | transfer | 34 | -2.8877 | -1.1296 | 1.7581 | 0.00003 | **YES** |
| False Belief | transfer | 35 | -7.2801 | -5.7670 | 1.5131 | 0.00019 | **YES** |

## Summary counts
- total condition×layer comparisons (all 3 conditions × 2 patch types × 7 layers = 42; minus missing controls): 42
- comparisons where degraded > control in expected dir: 41/42 (0.976)
- rescue: 20/21 survive
- transfer: 21/21 survive
- Closed-Context: 13/14 survive
- Emotional: 14/14 survive
- False Belief: 14/14 survive

## Late-layer (L28/L30/L32/L34/L35) vs early-control (L8/L20)

- rescue: late L[28, 30, 32, 34, 35] avg (degraded−control) expected-signed = 2.9935; early L[8, 20] = 0.3785
- transfer: late L[28, 30, 32, 34, 35] avg (degraded−control) expected-signed = 2.3892; early L[8, 20] = 0.3788
