# Activation Patching vs Prior Hidden-State Analyses (Step 6)

## Expected-signed patching effect pooled across conditions (degraded families)

| Patch type | Layer | pooled expected-signed mean | n conditions |
|---|---:|---:|---:|
| rescue | 8 | 0.0752 | 3 |
| rescue | 20 | 0.8676 | 3 |
| rescue | 28 | 2.3673 | 3 |
| rescue | 30 | 2.7783 | 3 |
| rescue | 32 | 3.5868 | 3 |
| rescue | 34 | 4.3127 | 3 |
| rescue | 35 | -2.8110 | 3 |
| transfer | 8 | -0.0103 | 3 |
| transfer | 20 | 1.2978 | 3 |
| transfer | 28 | 2.4302 | 3 |
| transfer | 30 | 2.7078 | 3 |
| transfer | 32 | 3.3570 | 3 |
| transfer | 34 | 3.9944 | 3 |
| transfer | 35 | 8.1476 | 3 |

## Prior layer-peak evidence (for comparison)

1. Original36 / HQ80 delta-norm peaks are in **late layers L28–L34** (often bimodal at L28 and L32/L34) at anchors S3/S4. S3 = end of user message (post-pressure text insertion); S4 = final-answer position (the single anchor actually patched here).
2. HQ80 Step-3 Δnorm↔degradation Pearson/Spearman correlations peak in **late layers**, S3/S4 usually above S0–S2.
3. HQ80 probe (Step 4) max detection with S3/S4 features: we used S3-only L20/L24/L28/L32/L35 subset; strongest effects consistently at **late layers** (hidden state readouts) rather than early.
4. Probe 6B S3/S4 detection: prior S3/S4 harmfulness-vs-nonharmful and pressure-strength decoders were strongest near final layers.

## 6 questions

**Q1. Strongest causal layers vs hidden-state layers?**

Rescue top-2 layers per condition (expected signed):
  - Closed-Context: L34 mean_signed_expected=5.8265
  - Closed-Context: L32 mean_signed_expected=4.8711
  - Emotional: L34 mean_signed_expected=4.1149
  - Emotional: L32 mean_signed_expected=3.6386
  - False Belief: L34 mean_signed_expected=2.9966
  - False Belief: L32 mean_signed_expected=2.2508
Transfer top-2 layers per condition:
  - Closed-Context: L35 mean_signed_expected=9.0405
  - Closed-Context: L34 mean_signed_expected=5.3007
  - Emotional: L35 mean_signed_expected=8.1222
  - Emotional: L34 mean_signed_expected=3.7948
  - False Belief: L35 mean_signed_expected=7.2801
  - False Belief: L34 mean_signed_expected=2.8877
Count of top-2 peaks per condition falling in late-layer band L28–L35: rescue 6/6; transfer 6/6.

**Q2. Behavioral strength vs patching strength?**

| Condition | mean |delta_margin| (degraded subset, larger = more behavioral degradation) | rescue peak expected-signed | transfer peak expected-signed |
|---|---:|---:|---:|
| Closed-Context | 4.6068 | 5.8265 | 9.0405 |
| Emotional | 2.6536 | 4.1149 | 8.1222 |
| False Belief | 1.8490 | 2.9966 | 7.2801 |

**Q3. Does patching support the claim?**

Average expected-signed effect (rescue pos, transfer neg → both good) across late L28–L35 × degraded × 3 conditions = 3.0871. Controls comparison see Step 4.

**Q4. Contradiction with prior interpretations?**

- No contradictions with delta-norm peaks (late layers) detected so far.
- Closed-context condition control family count was only 7 (fewer candidates passing near-zero-delta filter); rescue/transfer in that condition may be noisier than the evidence-based pressures.
- Control subset (near-zero |delta|) is NOT a random / distractor condition: it is *same-pressure-text* on families where the behavioral margin-delta happened by chance to be small. So this control tests whether the patching effect scales with the *behavioral* pairing (strong expected if degraded > small-expected-delta control), NOT a lexical-distractor or random-family control. See caveats in final interpretation.
