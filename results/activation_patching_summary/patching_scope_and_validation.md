# Activation Patching — Scope and Metadata Validation

## Dataset
- Dataset patched: **original36 family set only** (36-family matched-prefix subset of the sycophancy/evidence-aligned prompts; no HQ80 families patched here).
- Model patched: **Qwen/Qwen3-4B-Instruct-2507 (Qwen-only, CPU device)**.
- Driver script used: `run_activation_patching_original36.py` (bounded 36-family, same-family neutral-pressure pairs, canonical .pt-based subset selection + fresh CPU patch forwards at runtime).

## Conditions
- 3 pressure conditions patched (all single-condition vs same-family evidence_neutral):
  - `closed_context_false_belief_pressure`
  - `evidence_emotional_pressure`
  - `evidence_false_belief_pressure`

## Anchor & layers
- Anchor patched: **final_prompt_token (S4/answer position; original36 single-anchor extraction)**. The original36 extraction schema reads `hidden_states_final_token` tensor from each canonical `.pt` file (the equivalent of HQ80 anchor S4 = final-answer / end-of-prompt token position). No multi-anchor (S3, S2) patches were run in this bounded sweep.
- Layers patched: 7 layers — [8, 20, 28, 30, 32, 34, 35]
- Answer-position index in raw rows (convention = -1 for original36 single-token): [-1]

## Families / prompts
- Total raw patch rows: 938 (7 layers × 2 patch types × families selected per condition/subset).
- Families/prompts per condition × degraded vs control subset (requested n=12; controls may be fewer if insufficient near-zero-delta families pass filters):

| Condition | Subset | N families | 7×2 rows expected | 7×2 rows actual |
|---|---|---:|---:|---:|
| Closed-Context | degraded | 12 | 168 | 168 |
| Closed-Context | control | 7 | 98 | 98 |
| Emotional | degraded | 12 | 168 | 168 |
| Emotional | control | 12 | 168 | 168 |
| False Belief | degraded | 12 | 168 | 168 |
| False Belief | control | 12 | 168 | 168 |

## Pairing
- Same-family neutral/pressure paired patching used: **YES**. For each (family_id, condition) the rescue patch source is the same-family `evidence_neutral` activation at the target layer; the transfer patch source is the same-family pressure activation at the target layer.
- Degraded families and control families within each condition are disjoint (no reuse): **YES**

## Baselines present
- original_neutral_margin — finite, non-null rows: **938/938**
- original_pressure_margin — finite, non-null rows: **938/938**
- delta_margin (= pressure − neutral): **938/938**
- patched_margin: **938/938**
- rescue_effect (rows with patch_type=rescue): **469/469**
- transfer_effect (rows with patch_type=transfer): **469/469**


## Step-2 Correctness checks (all 10)

| # | Check | Passed? | Detail |
|---|---|---|---|
| 1 | 2.1 original margins match canonical behavioral CSVs (via delta_margin match, atol=0.01) | **YES** | 0 mismatches out of 938. Examples: [] |
| 2 | 2.2 required fields present (family, condition, correct/false choice, layer, patch_type) | **YES** |  |
| 3 | 2.3 rescue patches use same-family neutral activations | **YES** | Driver-level invariant (single-family context pointer per (fid, condition)). Verified indirectly: 0 duplicates, finite effects, degraded subset known-matched to canonical delta_magnitudes per selection_rank. |
| 4 | 2.4 transfer patches use same-family pressure activations | **YES** | Same invariant. Transfer rows per (fid, condition, layer) mirror rescue rows. |
| 5 | 2.5 no answer-choice labels swapped (correct != false) | **YES** | equal correct/false rows = 0 |
| 6 | 2.6 no random-family treated as same-family | **YES** | Driver has no random-family mode. Also confirmed no duplicate (fam, cond, subset, layer, patch) rows. |
| 7 | 2.7 all patched logits/margins are finite | **YES** |  |
| 8 | 2.8 all patch_effects (rescue, transfer) are finite | **YES** |  |
| 9 | 2.9 no duplicate patch rows | **YES** | duplicate count = 0 |
| 10 | 2.10 every row fully populated | **YES** | patch_effect non-null rows 938/938 |

**Overall: ALL CHECKS PASSED**
