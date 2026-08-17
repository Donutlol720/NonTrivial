"""
Bounded activation-patching experiment: original 36-family Qwen3-4B.

Patch types:
  - rescue  (neutral -> pressure  at a single transformer layer output)
  - transfer (pressure -> neutral  at a single transformer layer output)

Conditions tested:
  - evidence_false_belief_pressure
  - evidence_emotional_pressure
  - closed_context_false_belief_pressure
(reference: evidence_neutral)

Outputs go into results/activation_patching_qwen3_4b_original36/
Canonical original36 .pt baselines are READ for subset selection + cached
activations (no rerunning 2 baseline passes per family-condition).
Only patched forward passes actually execute the model.
"""

import argparse, csv, json, sys, time, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.load_model import load_local_model, pick_device
from src.extraction.extract_multi_family_states_and_logits import (
    choose_answer_token_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_transformer_block_list(model):
    for top in ("model", "transformer", "backbone"):
        if hasattr(model, top):
            sub = getattr(model, top)
            for sub_attr in ("layers", "h", "blocks", "decoder_layers"):
                if hasattr(sub, sub_attr):
                    obj = getattr(sub, sub_attr)
                    if isinstance(obj, (list, torch.nn.ModuleList)):
                        return f"{top}.{sub_attr}", obj
    raise RuntimeError("Could not find transformer block list.")


def get_all_layer_caches_from_pt(pt_path: Path, layers_list, answer_pos_override=None):
    """
    Read a saved canonical .pt file and return:
        dict: layer_idx -> (answer position hidden state vector, 1D float32 cpu)
    Also return meta dict with seq_len, answer_pos_index, correct/false choice,
    prompt text.

    Supports two schemas:
      (A) original36 single-anchor extraction schema:
            d["hidden_states_final_token"] : Tensor (n_layers, d_model)
            d["extraction_position"]       : "final_prompt_token"
            d["answer_logit_prompt"]       : prompt text used in forward pass
      (B) HQ80 multi-anchor schema (for completeness, not required by this run):
            d["hidden_states_by_anchor"] | d["hidden_states_by_anchor_names"]
            d["anchor_positions"]

    NOTE on saved logits in .pt files: the ORIGINAL36 pipeline was run under the
    old MPS backend with a known hidden-state aliasing bug, so d["logit_A"],
    d["logit_B"], d["logit_margin"] from disk are often 0.0 or garbage. We
    therefore never use those saved logits as margin references. We instead:
    (1) read the margin reference values from the canonical
    qwen3_4b_instruct_2507_family36_family_margin_deltas.csv (the paper values,
    recomputed CPU float32 from the hidden states), and (2) recompute unpatched
    baselines with a fresh deterministic forward pass now.
    """
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    # ---- Schema detection ----
    hs_tensor = None
    answer_pos_index = -1
    # (A) original36 schema
    if "hidden_states_final_token" in d:
        hs_tensor = d["hidden_states_final_token"]
        answer_pos_index = -1
    else:
        # (B) HQ80 multi-anchor schema (not expected here, but kept for completeness)
        hs_key = None
        for cand in ("hidden_states_by_anchor", "hidden_states_by_anchor_names"):
            if cand in d:
                hs_key = cand
                break
        if hs_key is None:
            raise RuntimeError(f"no hidden_states key in {pt_path.name}, keys={sorted(d.keys())}")
        hs_container = d[hs_key]
        if not isinstance(hs_container, torch.Tensor):
            raise RuntimeError(f"unexpected hs_container type: {type(hs_container)}")
        hs_tensor = hs_container
        anchor_pos = d.get("anchor_positions") or {}
        if isinstance(anchor_pos, str):
            try:
                anchor_pos = json.loads(anchor_pos)
            except Exception:
                anchor_pos = {}
        final_answer_from_anchors = None
        if isinstance(anchor_pos, dict):
            final_answer_from_anchors = anchor_pos.get("final_answer_position")
        token_seq_len = int(d.get("token_seq_len") or 0)
        if final_answer_from_anchors is not None:
            seq_len = max(token_seq_len, int(final_answer_from_anchors) + 1)
            answer_pos_index = int(final_answer_from_anchors)
        else:
            seq_len = token_seq_len
            answer_pos_index = seq_len - 1

    if answer_pos_override is not None:
        answer_pos_index = answer_pos_override

    if not isinstance(hs_tensor, torch.Tensor):
        raise RuntimeError(f"hidden_states tensor not found in {pt_path.name}")
    n_layers = hs_tensor.shape[0]
    cached = {}
    for L in layers_list:
        if L < n_layers:
            cached[L] = hs_tensor[L].detach().float().cpu().contiguous()
        else:
            raise RuntimeError(f"layer {L} out of range, n_layers={n_layers} in {pt_path}")

    prompt_text = (
        d.get("answer_logit_prompt")
        or d.get("prompt_text")
        or d.get("prompt")
        or ""
    )
    meta = {
        "correct_choice": str(d.get("correct_choice")),
        "false_choice":   str(d.get("false_choice")),
        "token_strategy": d.get("token_strategy"),
        "answer_position_index": answer_pos_index,
        "prompt_text": prompt_text,
        "_schema": "original36_final_token" if "hidden_states_final_token" in d else "hq80_multi_anchor",
        "_n_layers_tensor": n_layers,
        "_pt_logit_A_or_zero": float(d.get("logit_A") or 0.0),
        "_pt_logit_B_or_zero": float(d.get("logit_B") or 0.0),
        "_pt_logit_margin_or_zero": float(d.get("logit_margin") or 0.0),
    }
    return cached, meta


def patchable_forward(model, tokenizer, prompt_text, device, block_list_attr_name,
                      block_list, layer_idx, replacement_vector, idA, idB):
    """Forward pass with a single-layer output hook; returns (logit_A, logit_B, margin)."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]
    pos_index = -1  # always patch final position

    replacement = replacement_vector.to(device=device, dtype=block_list[0].__class__.__bases__[0].__bases__[0] if False else next(block_list[0].parameters()).dtype)

    patched = False
    def make_hook():
        def hook(module, args, output):
            nonlocal patched
            # module output: tuple (hidden_state, ...) or Tensor
            if isinstance(output, tuple):
                hs = output[0]  # shape (B, T, D)
                rest = output[1:]
            else:
                hs = output
                rest = ()
            new_hs = hs.clone()
            # replace at pos_index -1
            new_hs[0, pos_index, :] = replacement.to(dtype=new_hs.dtype, device=new_hs.device)
            patched = True
            if isinstance(output, tuple):
                return (new_hs,) + rest
            return new_hs
        return hook

    handle = block_list[layer_idx].register_forward_hook(make_hook())
    try:
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
        logits_last = out.logits[0, -1, :].float().cpu()
    finally:
        handle.remove()
    if not patched:
        raise RuntimeError(f"hook did not fire on layer {layer_idx} ({block_list_attr_name}[{layer_idx}])")
    lA = float(logits_last[idA].item())
    lB = float(logits_last[idB].item())
    return lA, lB


def baseline_forward(model, tokenizer, prompt_text, device, idA, idB):
    """Unpatched baseline forward pass."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
    logits_last = out.logits[0, -1, :].float().cpu()
    return float(logits_last[idA].item()), float(logits_last[idB].item())


def baseline_forward_with_live_cache(model, tokenizer, prompt_text, device, layers_to_cache, idA, idB):
    """
    Baseline forward WITH output_hidden_states=True.  Returns:
        lA, lB                   : logit_A, logit_B at final position
        cache_dict            : dict L -> 1D CPU float32 vector for each L in layers_to_cache
                              (transformer block outputs, not input_embeds excluded)
        token_seq_len         : sequence length of prompt tokenization (for meta)
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    token_seq_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    logits_last = out.logits[0, -1, :].float().cpu()
    lA = float(logits_last[idA].item())
    lB = float(logits_last[idB].item())
    hidden = out.hidden_states  # tuple length = input_embeds + n_transformer_blocks
    cache_dict = {}
    for L in layers_to_cache:
        # hidden[L+1] = output of transformer block L
        vec = hidden[L + 1][0, -1, :].detach().float().cpu().contiguous()
        cache_dict[L] = vec
    return lA, lB, cache_dict, token_seq_len


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------
def select_subsets(delta_margin_csv, conditions_map, n_target=12):
    """
    conditions_map: {"evidence_false_belief_pressure": "delta_false_pressure",
                     "evidence_emotional_pressure": "delta_emotional_pressure",
                     "closed_context_false_belief_pressure": "delta_closed_context"}
    Returns dict[condition] -> dict:
        degraded  -> list[(family_id, delta_margin, original_neutral_margin, original_pressure_margin, rank)]
        control   -> same structure
    """
    df = pd.read_csv(delta_margin_csv)
    selected = {}
    for cond, delta_col in conditions_map.items():
        neutral_col = {
            "evidence_false_belief_pressure": "logit_margin_evidence_neutral",
            "evidence_emotional_pressure":   "logit_margin_evidence_neutral",
            "closed_context_false_belief_pressure": "logit_margin_evidence_neutral",
        }[cond]
        pressure_col = {
            "evidence_false_belief_pressure": "logit_margin_evidence_false_belief_pressure",
            "evidence_emotional_pressure":   "logit_margin_evidence_emotional_pressure",
            "closed_context_false_belief_pressure": "logit_margin_closed_context_false_belief_pressure",
        }[cond]

        rows = []
        for _, r in df.iterrows():
            delta = float(r[delta_col])
            neutral_m = float(r[neutral_col])
            pressure_m = float(r[pressure_col])
            rows.append((str(r["family_id"]), delta, neutral_m, pressure_m))

        # degraded = most negative delta
        sorted_neg = sorted(rows, key=lambda x: x[1])
        degraded_candidates = [row for row in sorted_neg if row[1] < 0]
        degraded = degraded_candidates[:n_target]
        # controls = smallest |delta| positive or near-zero
        # take rows with delta >= -0.5 (near-zero or slight positive) sorted by |delta| asc
        control_candidates = sorted([row for row in rows if row[1] >= -0.25], key=lambda x: abs(x[1]))
        # exclude degraded families from controls to avoid overlap
        degraded_ids = {d[0] for d in degraded}
        control = [c for c in control_candidates if c[0] not in degraded_ids][:n_target]

        def with_rank(rows_list, label):
            out = []
            for rank, (fid, delta, nm, pm) in enumerate(rows_list, start=1):
                reason = f"{label} rank={rank} (delta={delta:+.4f})"
                out.append({"family_id": fid, "delta_margin": delta,
                            "original_neutral_margin": nm, "original_pressure_margin": pm,
                            "selection_rank_or_reason": reason})
            return out
        selected[cond] = {
            "degraded": with_rank(degraded, "degraded"),
            "control":  with_rank(control,  "control"),
        }
    return selected


# ---------------------------------------------------------------------------
# Plots (optional, only if matplotlib importable)
# ---------------------------------------------------------------------------
def try_make_plots(out_dir, summary_csv_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib not available: {e}; skip pdf plots")
        return
    try:
        df = pd.read_csv(summary_csv_path)
    except Exception as e:
        print(f"[plots] could not read summary csv: {e}")
        return
    conditions = list(df["condition"].unique())
    subset_types = list(df["subset_type"].unique())
    layers = sorted(df["layer"].unique())
    layer_idx_axis = list(range(len(layers)))

    for patch_type, ylabel, fname in [
        ("rescue",   "mean rescue_effect (patched pressure margin - original)", "rescue_effect_by_layer.pdf"),
        ("transfer", "mean transfer_effect (patched neutral margin - original)",  "transfer_effect_by_layer.pdf"),
    ]:
        fig, axes = plt.subplots(1, len(subset_types), figsize=(6.0 * len(subset_types), 5.2), sharey=True)
        if len(subset_types) == 1:
            axes = [axes]
        for ax, st in zip(axes, subset_types):
            for cond in conditions:
                sub = df[(df.patch_type==patch_type) & (df.subset_type==st) & (df.condition==cond)].sort_values("layer")
                if sub.empty:
                    continue
                y = [float(sub.loc[sub.layer==L, "mean_effect"].iloc[0]) if (sub.layer==L).any() else np.nan for L in layers]
                ax.plot(layer_idx_axis, y, marker="o", linewidth=1.8, label=cond)
            ax.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
            ax.set_xticks(layer_idx_axis)
            ax.set_xticklabels([str(L) for L in layers])
            ax.set_xlabel("Layer")
            ax.set_title(f"{patch_type} — {st}")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"Activation patching: {patch_type} effect by layer", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"[plots] wrote {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--layers", type=str, default="8,20,28,30,32,34,35")
    ap.add_argument("--n-per-subset", type=int, default=12)
    ap.add_argument("--conditions", type=str,
                    default="evidence_false_belief_pressure,evidence_emotional_pressure,closed_context_false_belief_pressure")
    ap.add_argument("--activation-root", type=Path,
                    default=REPO / "activations" / "qwen3_4b_instruct_2507")
    ap.add_argument("--dataset", type=Path, default=REPO / "data" / "generated_prompts_v1.jsonl")
    ap.add_argument("--delta-margin-csv", type=Path,
                    default=REPO / "results" / "qwen3_4b_instruct_2507_family36_family_margin_deltas.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "results" / "activation_patching_qwen3_4b_original36")
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "model_cache",
                    help="HuggingFace cache dir for model/tokenizer weights")
    ap.add_argument("--verify-baseline-n", type=int, default=4,
                    help="recompute this many fresh baseline margins per condition to verify match vs .pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    layers = [int(x) for x in args.layers.split(",")]
    conditions_requested = args.conditions.split(",")

    COND_DELTA_MAP = {
        "evidence_false_belief_pressure": "delta_false_pressure",
        "evidence_emotional_pressure": "delta_emotional_pressure",
        "closed_context_false_belief_pressure": "delta_closed_context",
    }
    conditions_map = {c: COND_DELTA_MAP[c] for c in conditions_requested}

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "activation_patching_raw_results.csv"
    summary_csv = out_dir / "activation_patching_summary_by_condition_layer.csv"
    sel_csv = out_dir / "activation_patching_selected_examples.csv"
    summary_txt = out_dir / "activation_patching_summary.txt"

    print(f"[init] device={device} model={args.model} layers={layers} n_per_subset={args.n_per_subset}")
    print(f"[init] activation_root = {args.activation_root}")
    print(f"[init] out_dir = {out_dir}")

    # --- Load prompts --------------------------------------------------------
    rows_prompts = [json.loads(l) for l in args.dataset.read_text().splitlines() if l.strip()]
    prompts_by_key = {(r["family_id"], r.get("condition") or r.get("prompt_type")): r for r in rows_prompts}
    print(f"[init] dataset prompts: {len(rows_prompts)} rows")

    # --- Load model ----------------------------------------------------------
    print("[model] loading ...")
    t0 = time.time()
    model, tokenizer = load_local_model(
        args.model,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
        cache_dir=str(args.cache_dir) if args.cache_dir else "",
    )
    model.eval()
    block_list_name, block_list = find_transformer_block_list(model)
    total_layers = len(block_list)
    t1 = time.time()
    print(f"[model] loaded in {t1-t0:.1f} s; block_list = {block_list_name} len={total_layers}")

    idA, idB, token_strategy = choose_answer_token_ids(tokenizer)
    print(f"[tokens] strategy={token_strategy}, idA={idA} (decode={tokenizer.decode([idA])!r}), "
          f"idB={idB} (decode={tokenizer.decode([idB])!r})")
    for L in layers:
        assert 0 <= L < total_layers, f"layer {L} out of range (max index {total_layers-1})"

    # --- Select subsets ------------------------------------------------------
    selected = select_subsets(args.delta_margin_csv, conditions_map, n_target=args.n_per_subset)
    sel_rows = []
    for cond, d in selected.items():
        for subset_type in ("degraded", "control"):
            for r in d[subset_type]:
                sel_rows.append({
                    "family_id": r["family_id"],
                    "condition": cond,
                    "subset_type": subset_type,
                    "delta_margin": f"{r['delta_margin']:+.6f}",
                    "original_neutral_margin": f"{r['original_neutral_margin']:+.6f}",
                    "original_pressure_margin": f"{r['original_pressure_margin']:+.6f}",
                    "selection_rank_or_reason": r["selection_rank_or_reason"],
                })
    with open(sel_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel_rows[0].keys()))
        w.writeheader(); w.writerows(sel_rows)
    print(f"[select] wrote {sel_csv} ({len(sel_rows)} rows)")
    for cond, d in selected.items():
        print(f"  {cond}: degraded={len(d['degraded'])}, control={len(d['control'])}")

    # --- Build metadata and prompts for all needed families/conditions ------
    # NOTE: The on-disk original36 .pt files were extracted under the old MPS
    # backend, which had a hidden-state aliasing bug. Therefore, we:
    #   * read .pt files ONLY for metadata (correct_choice, false_choice,
    #     prompt_text via answer_logit_prompt, token_strategy), never for
    #     cached hidden states.
    #   * cache ALL layer vectors LIVE on CPU float32 via baseline forwards
    #     with output_hidden_states=True, which we just verified reproduces
    #     the canonical CSV margins within float-epsilon agreement.
    meta_cache = {}
    dataset_prompt_key = {}
    for prow in rows_prompts:
        pfid = str(prow.get("family_id"))
        pcond = str(prow.get("condition") or prow.get("prompt_type") or prow.get("intended_condition") or "")
        a_text = prow.get("answer_logit_prompt")
        if not a_text:
            pieces = []
            for k in ("evidence","retrieved_evidence","preamble","question","scenario_question"):
                if prow.get(k):
                    pieces.append(str(prow[k]).rstrip())
            if prow.get("choice_a") and prow.get("choice_b"):
                pieces.append(f"Choices:\nA. {prow['choice_a']}\nB. {prow['choice_b']}")
            user_msg = prow.get("user_message") or prow.get("sycophancy_message") or ""
            if user_msg:
                pieces.append(f"User message:\n{user_msg}")
            pieces.append("Answer with only A or B.\n\nANSWER:")
            a_text = "\n\n".join(pieces)
        dataset_prompt_key[(pfid, pcond)] = a_text
    def prompt_for(fid, cond, fallback_meta):
        text = (fallback_meta.get("prompt_text") or "").strip()
        if len(text) > 100:
            return text
        t = dataset_prompt_key.get((str(fid), str(cond)))
        if t:
            return t
        for (f, c), tt in dataset_prompt_key.items():
            if f == str(fid):
                return tt
        return ""

    families_conditions_needed = []  # ordered deduped list
    seen = set()
    for cond, d in selected.items():
        for st in ("degraded", "control"):
            for r in d[st]:
                for fc in [(r["family_id"], "evidence_neutral"), (r["family_id"], cond)]:
                    if fc not in seen:
                        seen.add(fc)
                        families_conditions_needed.append(fc)
    missing_meta = []
    for (fid, c) in families_conditions_needed:
        fname = f"{fid}_{c}.pt"
        ptpath = args.activation_root / fid / fname
        # Try to read .pt metadata; if missing, fall back to prompts dataset
        if ptpath.exists():
            try:
                dmeta = torch.load(ptpath, map_location="cpu", weights_only=False)
                meta_cache[(fid, c)] = {
                    "correct_choice": str(dmeta.get("correct_choice", "")),
                    "false_choice":   str(dmeta.get("false_choice", "")),
                    "token_strategy": dmeta.get("token_strategy"),
                    "answer_position_index": -1,  # live forward = final position
                    "_schema": dmeta.get("extraction_position", "pt_fallback"),
                }
                text_fallback = (dmeta.get("answer_logit_prompt")
                                 or dmeta.get("prompt_text")
                                 or dmeta.get("prompt") or "")
                meta_cache[(fid, c)]["prompt_text"] = text_fallback
            except Exception as e:
                missing_meta.append((fid, c, f"pt read failed: {e}"))
                meta_cache[(fid, c)] = {"correct_choice": "A", "false_choice": "B",
                                         "answer_position_index": -1, "prompt_text": "",
                                         "_schema": "dataset_only"}
        else:
            missing_meta.append((fid, c, "no .pt file (dataset only)"))
            meta_cache[(fid, c)] = {"correct_choice": "A", "false_choice": "B",
                                     "answer_position_index": -1, "prompt_text": "",
                                     "_schema": "dataset_only"}

        # If dataset JSONL has correct/false choice, prefer that
        for prow in rows_prompts:
            if str(prow.get("family_id")) == str(fid) and \
               (str(prow.get("condition") or prow.get("prompt_type")) == str(c)):
                if prow.get("correct_choice"):
                    meta_cache[(fid, c)]["correct_choice"] = str(prow["correct_choice"])
                if prow.get("false_choice"):
                    meta_cache[(fid, c)]["false_choice"] = str(prow["false_choice"])
                break

    if missing_meta:
        print(f"[warn] missing/partial metadata for {len(missing_meta)} pairs (using dataset fallbacks):")
        for row in missing_meta[:5]:
            print(f"    {row}")

    # --- Live-cache baseline forwards for every needed (family, condition) ---
    # For each (fid, cond) we run ONE deterministic forward pass with
    # output_hidden_states=True. This gives us:
    #   * live unpatched logits (matches CSV margin up to float roundtrip)
    #   * layer vectors for ALL target layers for patching
    # This is ~144 total baseline live forwards (= 72 pairs × 2) plus
    # 72×7×2 = 1008 patch forwards.
    print(f"[live-cache] {len(families_conditions_needed)} unique (family, cond) to cache live")
    live_cache = {}  # (fid, cond) -> {"lA":, "lB":, "caches": {L->vec}, "token_seq_len":}
    baseline_checks = []
    verify_done_per_condition = defaultdict(int)
    start_live = time.time()
    for i, (fid, c) in enumerate(families_conditions_needed, start=1):
        meta = meta_cache[(fid, c)]
        text = prompt_for(fid, c, meta)
        if not text:
            print(f"[live-cache {i}/{len(families_conditions_needed)}] SKIP no prompt: {fid} {c}")
            continue
        try:
            lA, lB, cachelayers, seqlen = baseline_forward_with_live_cache(
                model, tokenizer, text, device, layers, idA, idB
            )
        except Exception as e:
            print(f"[live-cache {i}/{len(families_conditions_needed)}] FAIL {fid} {c}: {e}")
            live_cache[(fid, c)] = None
            continue
        # sanity: vectors differ by layer (should for clean CPU extraction)
        norms = [(L, float(v.norm())) for L, v in cachelayers.items()]
        normset = {round(n, 3) for _, n in norms}
        if len(normset) < 2:
            print(f"[warn] (fid={fid}, c={c}) all 7 layer norms identical = {norms[0][1]}; aliasing?")
        live_cache[(fid, c)] = {"lA": lA, "lB": lB, "caches": cachelayers,
                                "token_seq_len": seqlen, "prompt_text": text}

        # Baseline reproduction check (for paper CSV margin reference)
        parent_cond = None  # is this neutral or a pressure condition?
        if c == "evidence_neutral":
            # find its parent pressure and csv_ref
            for cond, d in selected.items():
                for st in ("degraded", "control"):
                    for r in d[st]:
                        if r["family_id"] == fid:
                            parent_cond = cond
                            csv_ref = float(r["original_neutral_margin"])
                            correct_choice = meta.get("correct_choice")
                            margin = (lA - lB) if correct_choice == "A" else (lB - lA)
                            diff = margin - csv_ref
                            baseline_checks.append((f"{fid} neutral", correct_choice,
                                                    f"csv_ref={csv_ref:+.4f}",
                                                    f"fresh={margin:+.4f}", f"diff={diff:+.4f}"))
                            break
        else:
            for cond, d in selected.items():
                if cond != c:
                    continue
                for st in ("degraded", "control"):
                    for r in d[st]:
                        if r["family_id"] == fid and verify_done_per_condition[cond] < args.verify_baseline_n + 5:
                            csv_ref = float(r["original_pressure_margin"])
                            correct_choice = meta.get("correct_choice")
                            margin = (lA - lB) if correct_choice == "A" else (lB - lA)
                            diff = margin - csv_ref
                            baseline_checks.append((f"{fid} pressure", correct_choice,
                                                    f"csv_ref={csv_ref:+.4f}",
                                                    f"fresh={margin:+.4f}", f"diff={diff:+.4f}"))
                            verify_done_per_condition[cond] += 1
                            break
        print(f"[live-cache {i:3d}/{len(families_conditions_needed)}] "
              f"{fid:40s} {c:45s} lA={lA:+.2f} lB={lB:+.2f} seq={seqlen}")

    elapsed_live = time.time() - start_live
    print(f"[live-cache] done in {elapsed_live:.1f} s ({elapsed_live/max(1,len(families_conditions_needed)):.1f} s/pass)")
    print(f"[baseline] reproduction checks ({len(baseline_checks)}):")
    for b in baseline_checks:
        print(f"  {b}")

    # --- Patching main -------------------------------------------------------
    raw_rows = []
    failures = []
    examples_counted = 0
    hook_effect_sanity_printed = False
    start_t = time.time()
    for cond, d in selected.items():
        for subset_type in ("degraded", "control"):
            for r in d[subset_type]:
                fid = r["family_id"]
                keyN = (fid, "evidence_neutral"); keyP = (fid, cond)
                if keyN not in live_cache or keyP not in live_cache \
                        or live_cache[keyN] is None or live_cache[keyP] is None:
                    failures.append((fid, cond, subset_type, "live_cache missing"))
                    continue
                if keyN not in meta_cache or keyP not in meta_cache:
                    continue
                metaN = meta_cache[keyN]; metaP = meta_cache[keyP]
                liveN = live_cache[keyN]; liveP = live_cache[keyP]
                cacheN = liveN["caches"]; cacheP = liveP["caches"]
                correct_choice = metaP["correct_choice"]
                false_choice   = metaP["false_choice"]
                prompt_neutral  = live_cache[keyN]["prompt_text"]
                prompt_pressure = live_cache[keyP]["prompt_text"]
                if not prompt_neutral or not prompt_pressure:
                    failures.append((fid, cond, subset_type,
                                     f"missing prompt text (len N={len(prompt_neutral)}, P={len(prompt_pressure)})"))
                    continue
                answer_position_index = -1  # live final token
                original_neutral_margin_ref  = float(r["original_neutral_margin"])
                original_pressure_margin_ref = float(r["original_pressure_margin"])
                delta_margin = float(r["delta_margin"])
                # Unpatched margins from live baselines (these are ground-truth
                # for the current run since they come from identical CPU f32
                # forward that also produced the layer cache vectors).
                lA_unp_N, lB_unp_N = liveN["lA"], liveN["lB"]
                lA_unp_P, lB_unp_P = liveP["lA"], liveP["lB"]
                current_unpatched_neutral_margin  = (lA_unp_N - lB_unp_N) if correct_choice == "A" else (lB_unp_N - lA_unp_N)
                current_unpatched_pressure_margin = (lA_unp_P - lB_unp_P) if correct_choice == "A" else (lB_unp_P - lA_unp_P)
                examples_counted += 1
                first_diff_sanity_for_this_example = None
                for L in layers:
                    # ----- rescue: neutral[L] -> pressure prompt
                    try:
                        lAr, lBr = patchable_forward(model, tokenizer, prompt_pressure, device,
                                                    block_list_name, block_list, L, cacheN[L], idA, idB)
                    except Exception as e:
                        failures.append((fid, cond, subset_type, L, "rescue", str(e)))
                        continue
                    rescue_margin = (lAr - lBr) if correct_choice == "A" else (lBr - lAr)
                    rescue_effect  = rescue_margin - original_pressure_margin_ref
                    raw_rows.append({
                        "family_id": fid,
                        "condition": cond,
                        "subset_type": subset_type,
                        "layer": L,
                        "patch_type": "rescue",
                        "correct_choice": correct_choice,
                        "false_choice": false_choice,
                        "original_neutral_margin":  f"{original_neutral_margin_ref:.6f}",
                        "original_pressure_margin": f"{original_pressure_margin_ref:.6f}",
                        "delta_margin": f"{delta_margin:.6f}",
                        "patched_margin": f"{rescue_margin:.6f}",
                        "rescue_effect": f"{rescue_effect:+.6f}",
                        "transfer_effect": "",
                        "answer_token_id_A": idA,
                        "answer_token_id_B": idB,
                        "answer_position_index": answer_position_index,
                    })
                    if not hook_effect_sanity_printed:
                        diff_sum = abs(lAr - lA_unp_P) + abs(lBr - lB_unp_P)
                        first_diff_sanity_for_this_example = (L, diff_sum, rescue_effect)
                    # ----- transfer: pressure[L] -> neutral prompt
                    try:
                        lAt, lBt = patchable_forward(model, tokenizer, prompt_neutral, device,
                                                    block_list_name, block_list, L, cacheP[L], idA, idB)
                    except Exception as e:
                        failures.append((fid, cond, subset_type, L, "transfer", str(e)))
                        continue
                    transfer_margin = (lAt - lBt) if correct_choice == "A" else (lBt - lAt)
                    transfer_effect = transfer_margin - original_neutral_margin_ref
                    raw_rows.append({
                        "family_id": fid,
                        "condition": cond,
                        "subset_type": subset_type,
                        "layer": L,
                        "patch_type": "transfer",
                        "correct_choice": correct_choice,
                        "false_choice": false_choice,
                        "original_neutral_margin":  f"{original_neutral_margin_ref:.6f}",
                        "original_pressure_margin": f"{original_pressure_margin_ref:.6f}",
                        "delta_margin": f"{delta_margin:.6f}",
                        "patched_margin": f"{transfer_margin:.6f}",
                        "rescue_effect": "",
                        "transfer_effect": f"{transfer_effect:+.6f}",
                        "answer_token_id_A": idA,
                        "answer_token_id_B": idB,
                        "answer_position_index": answer_position_index,
                    })
                    if not hook_effect_sanity_printed:
                        diff_sum = abs(lAt - lA_unp_N) + abs(lBt - lB_unp_N)
                        if first_diff_sanity_for_this_example is not None:
                            L0, ds, re = first_diff_sanity_for_this_example
                            print(f"[sanity hook-effect] FIRST ENCOUNTERED EXAMPLE: {fid} cond={cond} subset={subset_type}")
                            print(f"  rescue L={L0}: orig P margin={current_unpatched_pressure_margin:+.4f}, "
                                  f"patched P margin={rescue_margin:+.4f}, diff={rescue_margin-current_unpatched_pressure_margin:+.4f}, "
                                  f"sum|Δlogits|={ds:.4f}")
                            print(f"  transfer L={L}: orig N margin={current_unpatched_neutral_margin:+.4f}, "
                                  f"patched N margin={transfer_margin:+.4f}, diff={transfer_margin-current_unpatched_neutral_margin:+.4f}, "
                                  f"sum|Δlogits|={diff_sum:.4f}")
                            hook_effect_sanity_printed = True
                print(f"[patch {examples_counted:03d}] {fid:40s} {cond:45s} {subset_type:8s} "
                      f"saved delta={delta_margin:+.3f}  (live this run: N={current_unpatched_neutral_margin:+.2f}, "
                      f"P={current_unpatched_pressure_margin:+.2f})")

    print(f"\n[patch done] raw rows = {len(raw_rows)}, failures = {len(failures)}")
    elapsed = time.time() - start_t
    print(f"[timing] patching wall = {elapsed:.1f} s")
    if failures:
        for f in failures[:15]:
            print("  FAIL:", f)

    # Write raw rows
    fieldnames = ["family_id","condition","subset_type","layer","patch_type","correct_choice","false_choice",
                  "original_neutral_margin","original_pressure_margin","delta_margin","patched_margin",
                  "rescue_effect","transfer_effect","answer_token_id_A","answer_token_id_B","answer_position_index"]
    with open(raw_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)
    print(f"[output] wrote {raw_csv}")

    # --- Aggregate summary by condition/layer/patch_type/subset --------------
    df_raw = pd.DataFrame(raw_rows)
    def _num_effect(r):
        if r.patch_type == "rescue":
            return float(r["rescue_effect"]) if r["rescue_effect"] != "" else np.nan
        else:
            return float(r["transfer_effect"]) if r["transfer_effect"] != "" else np.nan
    df_raw["_eff"] = df_raw.apply(_num_effect, axis=1)
    summary_rows = []
    for (cond, st, L, ptype), grp in df_raw.groupby(["condition","subset_type","layer","patch_type"]):
        effs = grp["_eff"].dropna().values
        n = len(effs)
        if n == 0:
            continue
        if ptype == "rescue":
            pred_dir_count = int(np.sum(effs > 0))
        else:  # transfer
            pred_dir_count = int(np.sum(effs < 0))
        summary_rows.append({
            "condition": cond, "subset_type": st, "layer": L, "patch_type": ptype,
            "n": n,
            "mean_effect": f"{np.mean(effs):+.6f}",
            "median_effect": f"{np.median(effs):+.6f}",
            "std_effect": f"{np.std(effs):.6f}",
            "num_effects_in_predicted_direction": pred_dir_count,
            "fraction_effects_in_predicted_direction": f"{pred_dir_count/n:.4f}",
        })
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"[output] wrote {summary_csv} ({len(summary_rows)} rows)")

    # --- Optional plots ------------------------------------------------------
    try_make_plots(out_dir, summary_csv)

    # --- Summary TXT ---------------------------------------------------------
    lines = []
    lines.append("Activation patching experiment — Qwen3-4B original 36 families")
    lines.append("=" * 80)
    lines.append(f"model checkpoint     = {args.model}")
    lines.append(f"dataset path         = {args.dataset}")
    lines.append(f"activation root (.pt)= {args.activation_root}")
    lines.append(f"delta margin CSV     = {args.delta_margin_csv}")
    lines.append(f"device               = {device}")
    lines.append(f"conditions tested    = {list(conditions_map.keys())}")
    lines.append(f"layers patched       = {layers}  (block list = {block_list_name}, total layers = {total_layers})")
    lines.append(f"activation position  = answer final token (index -1 per pass; answer_position_index saved per prompt)")
    lines.append(f"token convention     = {token_strategy}")
    lines.append(f"  A token id = {idA}  decode repr = {tokenizer.decode([idA])!r}")
    lines.append(f"  B token id = {idB}  decode repr = {tokenizer.decode([idB])!r}")
    lines.append("")
    lines.append("Examples per condition/subset (from selected_examples.csv):")
    for cond, d in selected.items():
        lines.append(f"  {cond:45s}  degraded={len(d['degraded']):2d}  control={len(d['control']):2d}")
    lines.append(f"Total (family, condition) pairs actually patched = {examples_counted}")
    lines.append(f"Total patched passes (family×cond×2 patch types×{len(layers)} layers"
                 f" = {examples_counted*2*len(layers)})")
    lines.append("")
    lines.append(f"Baseline reproduction checks (first {args.verify_baseline_n} per condition, neutral+pressure):")
    for b in baseline_checks:
        lines.append(f"  {b}")
    lines.append("")
    lines.append("Main rescue findings (degraded examples, predicted effect > 0):")
    df_summ = pd.DataFrame(summary_rows)
    for cond in conditions_map.keys():
        lines.append(f"\n  Condition: {cond} — degraded subset rescue")
        sub = df_summ[(df_summ.condition == cond) & (df_summ.subset_type == "degraded") & (df_summ.patch_type == "rescue")].sort_values("layer")
        for L in layers:
            row = sub[sub.layer == L]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(f"    L={L:2d}  mean={r['mean_effect']:>10s}  median={r['median_effect']:>10s}  "
                         f"n={r['n']}  frac_pred_dir={r['fraction_effects_in_predicted_direction']}")
    lines.append("\nMain transfer findings (degraded examples, predicted effect < 0):")
    for cond in conditions_map.keys():
        lines.append(f"\n  Condition: {cond} — degraded subset transfer")
        sub = df_summ[(df_summ.condition == cond) & (df_summ.subset_type == "degraded") & (df_summ.patch_type == "transfer")].sort_values("layer")
        for L in layers:
            row = sub[sub.layer == L]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(f"    L={L:2d}  mean={r['mean_effect']:>10s}  median={r['median_effect']:>10s}  "
                         f"n={r['n']}  frac_pred_dir={r['fraction_effects_in_predicted_direction']}")
    # Late vs control comparison
    def _avg(df, conds, layers_sub, st, ptype):
        vals = []
        for c in conds:
            sub = df[(df.condition == c) & (df.subset_type == st) & (df.patch_type == ptype) & (df.layer.isin(layers_sub))]
            for _, r in sub.iterrows():
                vals.append(float(r["mean_effect"]))
        return (np.mean(vals) if vals else float("nan"), len(vals))
    ctrl_layers = [L for L in (8, 20) if L in layers]
    late_layers = [L for L in (28, 30, 32, 34) if L in layers]
    late = _avg(df_summ, list(conditions_map.keys()), late_layers, "degraded", "rescue")
    ctrl = _avg(df_summ, list(conditions_map.keys()), ctrl_layers, "degraded", "rescue")
    lines.append("\nLate vs control (degraded, rescue mean over all 3 conditions):")
    lines.append(f"  control layers {ctrl_layers} avg mean rescue = {ctrl[0]:+.4f} ({ctrl[1]} points)")
    lines.append(f"  late layers    {late_layers} avg mean rescue = {late[0]:+.4f} ({late[1]} points)")
    late_t = _avg(df_summ, list(conditions_map.keys()), late_layers, "degraded", "transfer")
    ctrl_t = _avg(df_summ, list(conditions_map.keys()), ctrl_layers, "degraded", "transfer")
    lines.append(f"\nLate vs control (degraded, transfer mean over all 3 conditions):")
    lines.append(f"  control layers {ctrl_layers} avg mean transfer = {ctrl_t[0]:+.4f} ({ctrl_t[1]} points)")
    lines.append(f"  late layers    {late_layers} avg mean transfer = {late_t[0]:+.4f} ({late_t[1]} points)")
    lines.append("\nControl-subset comparison (rescue predicted direction fraction usually weaker than degraded):")
    for ptype, pred_sign in (("rescue", ">0"), ("transfer", "<0")):
        lines.append(f"\n  {ptype} (predicted {pred_sign}) degraded vs control, by layer:")
        for L in layers:
            sub_deg = df_summ[(df_summ.patch_type==ptype)&(df_summ.subset_type=="degraded")&(df_summ.layer==L)]
            sub_ctl = df_summ[(df_summ.patch_type==ptype)&(df_summ.subset_type=="control")&(df_summ.layer==L)]
            fd = "; ".join([f"{r.condition} {r.fraction_effects_in_predicted_direction}" for _,r in sub_deg.iterrows()]) if not sub_deg.empty else "-"
            fc = "; ".join([f"{r.condition} {r.fraction_effects_in_predicted_direction}" for _,r in sub_ctl.iterrows()]) if not sub_ctl.empty else "-"
            lines.append(f"    L={L:2d}  degraded: {fd}   |   control: {fc}")
    if failures:
        lines.append(f"\nFailures / warnings ({len(failures)}):")
        for f in failures[:20]:
            lines.append(f"  {f}")
    lines.append(f"\nTotal failures: {len(failures)}")
    lines.append(f"Elapsed patching time: {elapsed:.1f} s")

    summary_txt.write_text("\n".join(lines) + "\n")
    print(f"[output] wrote {summary_txt}")

    # Quick recommendation block printed to stdout
    print("\n" + "="*80)
    print("Late vs control degraded rescue sign (positive = predicted direction):")
    print(f"  ctrl {ctrl_layers}: {ctrl[0]:+.4f}")
    print(f"  late {late_layers}: {late[0]:+.4f}")
    print("Late vs control degraded transfer sign (negative = predicted direction):")
    print(f"  ctrl {ctrl_layers}: {ctrl_t[0]:+.4f}")
    print(f"  late {late_layers}: {late_t[0]:+.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
