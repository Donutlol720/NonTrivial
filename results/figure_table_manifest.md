# Figure and table manifest (Task 4)

Every entry below lists the absolute path, source inputs, verified numerical anchors, and the paper claim it supports. No new model inference was performed. Only original-36 frozen outputs and corrected Probe-6B Task-2 outputs were used. HQ80 partial outputs are not used.

---

## Figures (all in `figures/`)

### F1. `figures/table1_behavioral_deltas.pdf`

- **Source CSVs**:
  - [qwen3_4b_instruct_2507_family36_behavior_summary.txt](file:///Users/ericmin/NonTrivial/results/qwen3_4b_instruct_2507_family36_behavior_summary.txt) — canonical original-36 Table 1 means (false Δ=+0.187, emotional Δ=−0.973, distractor Δ=+0.515, closed vs false Δ=−2.136).
  - [qwen3_4b_instruct_2507_family36_family_margin_deltas.csv](file:///Users/ericmin/NonTrivial/results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv) (for true-belief pressure, which the summary.txt omits).
- **Verified values on plot**:
  - False belief vs neutral: `mean Δ = +0.187` (12 Δ<0 / 24 Δ≥0).
  - True belief vs neutral: `mean Δ = +2.683` (1 Δ<0 / 35 Δ≥0).
  - Distractor vs neutral: `mean Δ = +0.515` (9 Δ<0 / 27 Δ≥0).
  - Emotional vs neutral: `mean Δ = −0.973` (26 Δ<0 / 10 Δ≥0).
  - Closed context vs false belief: `mean Δ = −1.949` (31 Δ<0 / 5 Δ≥0; 1 observed answer flip).
- **Claim supported**: Emotional pressure produces mean margin degradation ~1 logit; false-belief pressure does *not* produce mean degradation on this 36-family set (actually strengthens by +0.19, but 12/36 families individually degrade). Closed-context over false-belief doubles the emotional magnitude to −1.95. Closed-context is the only condition with observed answer flips (1 of 36 families).

### F2. `figures/layerwise_delta_norms.pdf`

- **Source CSV**: [qwen3_4b_instruct_2507_family36_layerwise_delta_norms.csv](file:///Users/ericmin/NonTrivial/results/qwen3_4b_instruct_2507_family36_layerwise_delta_norms.csv).
- **Lines plotted**: 5 delta_types = `false_pressure_delta`, `emotional_pressure_delta`, `closed_context_delta`, `true_pressure_delta`, `distractor_delta` (one line each, 36 layers).
- **Verified values**:
  - All 5 curves start near ~0.09–0.15 L0 (input embedding) and grow monotonically, peaking at **L34** with magnitude 48–93.
  - Peak order consistent with behavioral order: emotional (92.9) > closed (85.6) > true (74.2) > false (73.2) > distractor (48.7).
  - L35: distractor noticeably smallest (23.7) vs pressures 32–43.
- **Claim supported**: Pressure/condition representations diverge from neutral monotonically across Qwen3-4B layers; distractor-neutral delta is the smallest post-L10. Large L34 norms (~50–100) confirm deep-layer directional shifts rather than noise.

### F3. `figures/layerwise_cosines.pdf`

- **Source CSV**: [qwen3_4b_instruct_2507_family36_layerwise_delta_cosines.csv](file:///Users/ericmin/NonTrivial/results/qwen3_4b_instruct_2507_family36_layerwise_delta_cosines.csv).
- **Pairs plotted** (exactly as specified): False vs True, False vs Emotional, False vs Closed, False vs Distractor, Emotional vs Closed (5 lines, 36 layers each).
- **Verified values**:
  - False↔True peaks 0.956 @ L11, drops to 0.575 by L35.
  - False↔Emotional flat mean 0.725 across all layers.
  - False↔Closed steadily rises from 0.657 L0 → 0.779 L35.
  - False↔Distractor lowest of all pairs (mean 0.304) — always distinct.
  - Emotional↔Closed mid-range 0.487–0.728.
- **Claim supported**: Pressure delta directions share early-layer cosine but diverge by layer; distractor direction is structurally unlike all four pressure directions (≥0.6 dissimilarity at every layer). False and closed share an increasingly aligned direction in the deep layers.

### F4. `figures/hidden_behavior_correlations.pdf`

- **Source CSV**: [qwen3_4b_instruct_2507_family36_hidden_behavior_correlations.csv](file:///Users/ericmin/NonTrivial/results/qwen3_4b_instruct_2507_family36_hidden_behavior_correlations.csv).
- **Lines plotted**: Pearson r vs layer for the 4 main delta types vs their matched degradation g = −Δm (false, emotional, closed ctx, distractor).
- **Verified best Pearson r**:
  - Closed context `r=0.844 @ L32`; false `r=0.728 @ L32`; emotional `r=0.344 @ L30`; distractor `r=0.466 @ L35`.
- **Claim supported**: Hidden-layer norm magnitude strongly aligns with behavioral degradation *in held-out families* for false and especially closed-context pressures (r ≥ 0.73 late layers). Emotional alignment is considerably weaker (r ≤ 0.34), consistent with emotional's more uniform floor effect (36/36 harmful in Probe6B recomputation).

### F5. `figures/probe_harmful_decodability_by_layer.pdf`

- **Source CSV**: [qwen3_4b_instruct_2507_family36_direction_projection_probe.csv](file:///Users/ericmin/NonTrivial/results/qwen3_4b_instruct_2507_family36_direction_projection_probe.csv).
- **Lines plotted**: Held-out direction-projection Pearson r (centroid-probe surrogate) vs layer for false, emotional, closed-context (3 lines, 36 layers), each with star marker at best layer.
- **Verified best layers match the summary.txt exactly**:
  - False: `r=0.358 @ L32` matches canonical direction-probe summary (`best Pearson layer 32, r=0.3576`).
  - Emotional: `r=0.442 @ L20` matches `best Pearson layer 20, r=0.4422`.
  - Closed: `r=0.499 @ L28` matches `best Pearson layer 28, r=0.4988`.
- **Claim supported**: Cross-family pressure direction generalizes even without explicit LOGO centroid two-class training. Closed-context direction probe is the strongest single original-36 decodability result (r=0.499). Emotional peaks earliest at middle layers (L20), while false/closed peak at L28–L32.

### F6. `figures/probe6b_s3s4_summary.pdf`

- **Source CSVs (corrected-label recomputation)**:
  - [probe6b_s3s4_clean_vs_outlier_results.csv](file:///Users/ericmin/NonTrivial/results/probe6b_s3s4_clean_vs_outlier_results.csv) — 36-layer BA sweeps for the corrected labels.
  - [probe6b_valid_anchor_results.csv](file:///Users/ericmin/NonTrivial/results/probe6b_valid_anchor_results.csv) — best BA/layer reference.
- **Panel configuration**: 1×2 subplots: (left) pooled harmful-vs-nonharmful; (right) within false-belief pressure. S3 and S4 curves only; S0/S1/S2 **not plotted as pressure-detection results**. Figure footnote explicitly describes the shared-anchor artifact exclusion.
- **Verified values match spec exactly**:
  - Left pooled: S3 `BA=0.827 @ L1`; S4 `BA=0.878 @ L7`.
  - Right within false: S3 `BA=0.744 @ L18`; S4 `BA=0.825 @ L35`.
- **Claim supported**: Matched-prefix Probe 6B, with CPU-float32 recomputed labels (16 false/20 nonharmful balanced) and artifact-corrected sensitivity, shows pressure-induced margin degradation is decodable from delta features at **S3 (end of user message) before the final ANSWER**, and rises 5–8 BA points by S4 (ANSWER). Both panels are artifact-robust (identical BA on CLEAN families and with Δ−Δ(S2) correction).

---

## LaTeX tables (all in `results/`)

### T1. `results/table1_behavior_summary_for_latex.tex`

- **Source CSVs/TXT**:
  - [qwen3_4b_instruct_2507_family36_behavior_summary.txt](file:///Users/ericmin/NonTrivial/results/qwen3_4b_instruct_2507_family36_behavior_summary.txt) — 12/26/9/35 negative-count ground truth for false/emotional/distractor/closed-context comparisons (canonical Table-1 source).
  - [qwen3_4b_instruct_2507_family36_family_margin_deltas.csv](file:///Users/ericmin/NonTrivial/results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv) — per-family delta columns (true-belief row only).
- **Verified rows**: false vs neutral (Δ=+0.187, 12<0 / 24≥0, 0 flips), true vs neutral (+2.683, 1<0 / 35≥0, 0 flips), distractor vs neutral (+0.515, 9<0 / 27≥0, 0 flips), emotional vs neutral (−0.973, 26<0 / 10≥0, 0 flips), closed ctx vs false (−1.949, 31<0 / 5≥0, **1 answer flip**).
- **Claim supported**: Original 36-family behavioral Table 1, ready to drop into LaTeX.

### T2. `results/probe6b_summary_for_latex.tex`

- **Source CSV**: [probe6b_valid_anchor_results.csv](file:///Users/ericmin/NonTrivial/results/probe6b_valid_anchor_results.csv) (4 valid-anchor rows × S3/S4, plus 2 cross-condition rows with dagger/ddagger footnotes).
- **Verified rows**: Pooled S3 (0.827/L1, supported, Robust), Pooled S4 (0.878/L7, Robust), Within false S3 (0.744/L18, Robust), Within false S4 (0.825/L35, Robust/Marginal on clean-only Δ0.068); Cross false→closed†; Cross false→emotional‡ with explicit class-imbalance caution.
- **Claim supported**: Corrected-labels Probe 6B valid-anchor summary table with explicit cross-condition fragility footnotes.

### T3. `results/probe_support_summary_for_latex.tex`

- **Source CSV**: [probe6b_support_table.csv](file:///Users/ericmin/NonTrivial/results/probe6b_support_table.csv) (3 behavioral + 4 valid-probe rows).
- **Verified rows**:
  - Behavior false pressure: 16 harm / 20 nonharm → **Supported** (balanced).
  - Behavior emotional pressure: 36 harm / 0 nonharm → **Unsupported** (floor, nonharmful empty).
  - Behavior closed ctx: 35 harm / 1 nonharm → **Unsupported** (near-floor, 1 nonharmful).
  - Probe pooled and probe within-false: Supported; LOGO contract preserved; artifact-corrected BA identical.
- **Claim supported**: One table explaining *why* within emotional and within closed ctx are not reported as supported classification analyses in the paper, paired with the two analyses that are supported.

---

## Interpretation restrictions reflected in every figure/table

1. S0/S1/S2 vulnerability encoding is described separately (e.g., Probe 6B Fig6 footnote explicitly excludes S0/S1/S2 from pressure-detection claims because of the 14-family shared-anchor numerical artifact and general family-level vulnerability confound).
2. Only S3/S4 are labeled "valid post-user-message" pressure-detection anchors in Fig6/T2.
3. Emotional within-condition is never described as a supported harmful/nonharmful classification task (36/36 harmful, empty class).
4. Closed-context within-condition described as unsupported or effectively unsupported (35/36 harmful → MIN_TRAIN_CLASS_COUNT violation).
5. Cross false→emotional explicitly marked ‡ not a balanced discrimination test (target 0 nonharmful = pattern-resemblance interpretation only).
6. Cross false→closed explicitly marked † fragile (target only 1 nonharmful).
7. No permutation p-values reported anywhere in the 3 tables / 6 figures (the original 100-run Probe6B permutation p=0.0099 was under MPS-corrupted labels; corrected-label perms are not yet computed per your directive).
