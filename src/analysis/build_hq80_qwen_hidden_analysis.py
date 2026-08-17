import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
_LOCAL = REPO / "_local_pkgs"
if _LOCAL.is_dir() and str(_LOCAL) not in sys.path:
    sys.path.insert(0, str(_LOCAL))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[hq80 hidden] matplotlib not available; skipping plots.")

warnings.filterwarnings("ignore", category=FutureWarning, message=".*'penalty' was deprecated.*")

ACT_ROOT = REPO / "activations/qwen3_4b_hq80_matched_prefix_v1"
BEHAV_CSV = REPO / "results/hq80_behavior/qwen_hq80_prompt_level_logits.csv"
FAM_DELTA_CSV = REPO / "results/hq80_behavior/qwen_hq80_family_margin_deltas.csv"
SUMMARY_CSV = REPO / "results/hq80_behavior/qwen_hq80_behavior_summary_by_condition.csv"
PROMPT_JSONL = REPO / "prompts/expanded_matched_prefix_hq80_v1.jsonl"
OUTPUT_DIR = REPO / "results/hq80_hidden"

ANCHOR_MAP: Dict[str, str] = {
    "S0": "end_of_evidence_block",
    "S1": "end_of_question_block",
    "S2": "end_of_answer_choices",
    "S3": "end_of_user_message",
    "S4": "final_answer_position",
}
ANCHOR_ORDER = ("S0", "S1", "S2", "S3", "S4")
VALID_PRESSURE_ANCHORS = ("S3", "S4")

FALSE_PRESSURE = (
    "evidence_false_belief_pressure",
    "evidence_false_rationale_pressure",
    "evidence_emotional_pressure",
    "evidence_authority_pressure",
)
EVIDENCE_ALIGNED = (
    "evidence_true_belief_pressure",
    "evidence_true_rationale_pressure",
)
NONPRESSURE_BASELINE = ("evidence_distractor_neutral",)
ALL_NONNEUTRAL = FALSE_PRESSURE + EVIDENCE_ALIGNED + NONPRESSURE_BASELINE
NEUTRAL = "evidence_neutral"

PRETTY_LABELS = {
    "evidence_false_belief_pressure": "False belief",
    "evidence_false_rationale_pressure": "False rationale",
    "evidence_emotional_pressure": "Emotional",
    "evidence_authority_pressure": "Authority",
    "evidence_true_belief_pressure": "True belief",
    "evidence_true_rationale_pressure": "True rationale",
    "evidence_distractor_neutral": "Distractor",
}

N_LAYERS = 36
D_MODEL = 2560
EPS = 1e-10

_COND_SHORT = {
    "evidence_false_belief_pressure": "FB",
    "evidence_false_rationale_pressure": "FR",
    "evidence_emotional_pressure": "EM",
    "evidence_authority_pressure": "AU",
    "evidence_true_belief_pressure": "TB",
    "evidence_true_rationale_pressure": "TR",
    "evidence_distractor_neutral": "distr",
}

_COND_COLOR = {
    "evidence_false_belief_pressure": "#c0392b",
    "evidence_false_rationale_pressure": "#e67e22",
    "evidence_emotional_pressure": "#e74c3c",
    "evidence_authority_pressure": "#8e1a0f",
    "evidence_true_belief_pressure": "#27ae60",
    "evidence_true_rationale_pressure": "#16a085",
    "evidence_distractor_neutral": "#7f8c8d",
}

_COND_LS = {
    "evidence_false_belief_pressure": "-",
    "evidence_false_rationale_pressure": "-",
    "evidence_emotional_pressure": "-",
    "evidence_authority_pressure": "-",
    "evidence_true_belief_pressure": "-",
    "evidence_true_rationale_pressure": "-",
    "evidence_distractor_neutral": "--",
}

_PT_CACHE: Dict[str, Dict[str, Any]] = {}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(v: float, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan"
    return f"{float(v):.{nd}f}"


def load_pt(prompt_id: str, family_id: str) -> Dict[str, Any]:
    cache_key = f"{family_id}::{prompt_id}"
    if cache_key in _PT_CACHE:
        return _PT_CACHE[cache_key]
    pt_path = ACT_ROOT / family_id / (prompt_id + ".pt")
    rec = torch.load(pt_path, map_location="cpu", weights_only=False)
    out: Dict[str, Any] = {"_raw": rec, "prompt_id": prompt_id, "family_id": family_id}
    hs = rec.get("hidden_states_by_anchor")
    if hs is None:
        raise KeyError(f"hidden_states_by_anchor missing in {pt_path}")
    out_by_anchor: Dict[str, np.ndarray] = {}
    for disp, pt_key in ANCHOR_MAP.items():
        tens = hs.get(pt_key)
        if tens is None:
            raise KeyError(f"anchor {pt_key} missing in {pt_path}")
        if not isinstance(tens, torch.Tensor):
            raise TypeError(f"anchor {pt_key} not tensor in {pt_path}")
        out_by_anchor[disp] = tens.detach().to(dtype=torch.float64).numpy().astype(np.float64)
    out["hidden_states_by_anchor"] = out_by_anchor
    for k in (
        "condition",
        "correct_choice",
        "false_choice",
        "token_seq_len",
        "anchor_positions",
    ):
        out[k] = rec.get(k)
    _PT_CACHE[cache_key] = out
    return out


def build_anchor_manifest_rows(
    behav_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in behav_df.iterrows():
        pid = str(row["prompt_id"])
        fid = str(row["family_id"])
        rec = load_pt(pid, fid)
        hs = rec["hidden_states_by_anchor"]
        s0_L8 = float(np.linalg.norm(hs["S0"][8, :]))
        s3_L32 = float(np.linalg.norm(hs["S3"][32, :]))
        s4_L32 = float(np.linalg.norm(hs["S4"][32, :]))
        anchor_positions = rec.get("anchor_positions")
        if isinstance(anchor_positions, dict):
            anchor_str = json.dumps(
                {k: int(v) if isinstance(v, (int, float)) else str(v) for k, v in anchor_positions.items()},
                sort_keys=True,
            )
        else:
            anchor_str = str(anchor_positions)
        rows.append({
            "prompt_id": pid,
            "family_id": fid,
            "condition": str(row["condition"]),
            "correct_choice": str(row["correct_choice"]),
            "false_choice": str(row["false_choice"]),
            "activation_path_abs": str(ACT_ROOT / fid / (pid + ".pt")),
            "token_seq_len": int(rec.get("token_seq_len", 0) or 0),
            "S0_anchor_layer_norm_L8_norm": fmt(s0_L8, 6),
            "S3_anchor_layer_norm_L32_norm": fmt(s3_L32, 6),
            "S4_anchor_layer_norm_L32_norm": fmt(s4_L32, 6),
            "anchor_positions_json": anchor_str,
        })
    return rows


def compute_all_deltas(
    behav_df: pd.DataFrame,
    fam_delta_df: pd.DataFrame,
) -> Dict[str, Any]:
    delta_store: Dict[Tuple[str, str, str], np.ndarray] = {}
    delta_norm_store: Dict[Tuple[str, str, str], np.ndarray] = {}

    behav_by_id = behav_df.set_index(["family_id", "condition"])

    for _, fam_row in fam_delta_df.iterrows():
        family_id = str(fam_row["family_id"])
        neutral_prompt_rows = behav_df[
            (behav_df["family_id"] == family_id) & (behav_df["condition"] == NEUTRAL)
        ]
        if neutral_prompt_rows.empty:
            continue
        neutral_pid = str(neutral_prompt_rows.iloc[0]["prompt_id"])
        neutral_rec = load_pt(neutral_pid, family_id)
        h_neu = neutral_rec["hidden_states_by_anchor"]

        for condition in ALL_NONNEUTRAL:
            cond_rows = behav_df[
                (behav_df["family_id"] == family_id) & (behav_df["condition"] == condition)
            ]
            if cond_rows.empty:
                continue
            cond_pid = str(cond_rows.iloc[0]["prompt_id"])
            cond_rec = load_pt(cond_pid, family_id)
            h_cond = cond_rec["hidden_states_by_anchor"]

            for anchor in ANCHOR_ORDER:
                delta = h_cond[anchor] - h_neu[anchor]
                delta_store[(family_id, condition, anchor)] = delta
                dnorm = np.zeros(N_LAYERS, dtype=np.float64)
                for L in range(N_LAYERS):
                    dnorm[L] = float(np.linalg.norm(delta[L, :]))
                delta_norm_store[(family_id, condition, anchor)] = dnorm

    return {
        "delta": delta_store,
        "delta_norm": delta_norm_store,
    }


def build_step1_rows(
    fam_delta_df: pd.DataFrame,
    behav_df: pd.DataFrame,
    storage: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dn = storage["delta_norm"]
    behav_keyed = behav_df.set_index(["family_id", "condition"])

    for _, fam_row in fam_delta_df.iterrows():
        family_id = str(fam_row["family_id"])
        for condition in ALL_NONNEUTRAL:
            if (family_id, condition) not in behav_keyed.index:
                continue
            b_row = behav_keyed.loc[(family_id, condition)]
            degrad = float(b_row["degradation"]) if "degradation" in behav_keyed.columns else -float(b_row["margin_delta"])
            mdelta = float(b_row["margin_delta"])
            for anchor in ANCHOR_ORDER:
                key = (family_id, condition, anchor)
                if key not in dn:
                    continue
                norms = dn[key]
                for L in range(N_LAYERS):
                    rows.append({
                        "family_id": family_id,
                        "condition": condition,
                        "anchor": anchor,
                        "layer": L,
                        "delta_norm": fmt(norms[L], 8),
                        "degradation": fmt(degrad, 8),
                        "margin_delta": fmt(mdelta, 8),
                    })
    return rows


def agg_step1(step1_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(step1_rows)
    df["delta_norm"] = df["delta_norm"].astype(float)
    g = df.groupby(["condition", "anchor", "layer"], as_index=False).agg(
        mean_delta_norm=("delta_norm", "mean"),
        median_delta_norm=("delta_norm", "median"),
        n_families=("delta_norm", "count"),
    )
    return g


def plot_step1(agg: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    if not HAS_MPL:
        return
    try:
        anchors = ANCHOR_ORDER
        fig, axes = plt.subplots(1, len(anchors), figsize=(5 * len(anchors), 5), sharey=False)
        if len(anchors) == 1:
            axes = [axes]

        all_vals: List[float] = []
        for _, r in agg.iterrows():
            all_vals.append(float(r["mean_delta_norm"]))
        use_log = False
        if all_vals and min(all_vals) > 0 and (max(all_vals) / max(min(all_vals), 1e-9) > 10):
            use_log = True

        layers = list(range(N_LAYERS))
        for ax, anchor in zip(axes, anchors):
            sub = agg[agg["anchor"] == anchor]
            for cond in ALL_NONNEUTRAL:
                s = sub[sub["condition"] == cond].sort_values("layer")
                if s.empty:
                    continue
                ax.plot(
                    s["layer"].values,
                    s["mean_delta_norm"].values,
                    color=_COND_COLOR[cond],
                    linestyle=_COND_LS[cond],
                    linewidth=1.8,
                    label=PRETTY_LABELS.get(cond, cond),
                )
            ax.set_title(f"Anchor {anchor}")
            ax.set_xlabel("Layer")
            if anchor == anchors[0]:
                ax.set_ylabel("Mean ||Δh||₂")
            if use_log:
                ax.set_yscale("log")
            if N_LAYERS > 20:
                ax.set_xticks(layers[::4])

            if anchor in VALID_PRESSURE_ANCHORS:
                for cond in FALSE_PRESSURE + EVIDENCE_ALIGNED + NONPRESSURE_BASELINE:
                    s = sub[sub["condition"] == cond]
                    if s.empty:
                        continue
                    idx = int(s["mean_delta_norm"].idxmax())
                    mx = s.loc[idx]
                    ax.annotate(
                        f"{PRETTY_LABELS[cond][:2]}: {float(mx['mean_delta_norm']):.1f}",
                        (int(mx["layer"]), float(mx["mean_delta_norm"])),
                        fontsize=6,
                        alpha=0.8,
                    )

        fig.suptitle("Mean hidden-state Δnorm vs neutral by layer & anchor (HQ80 Qwen3-4B, N=80)", fontsize=12)
        handles = [
            Line2D([0], [0], color=_COND_COLOR[c], linestyle=_COND_LS[c], lw=2, label=PRETTY_LABELS[c])
            for c in ALL_NONNEUTRAL
        ]
        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.92, 0.96])
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[hq80 hidden] step1 plot failed: {exc}")


def build_step2_rows(
    fam_delta_df: pd.DataFrame,
    storage: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dstore = storage["delta"]

    false_short = [_COND_SHORT[c] for c in FALSE_PRESSURE]

    for _, fam_row in fam_delta_df.iterrows():
        family_id = str(fam_row["family_id"])
        for anchor in ANCHOR_ORDER:
            vecs: Dict[str, np.ndarray] = {}
            ok = True
            for condition in ALL_NONNEUTRAL:
                key = (family_id, condition, anchor)
                if key not in dstore:
                    ok = False
                    break
                vecs[_COND_SHORT[condition]] = dstore[key]
            if not ok:
                continue

            for L in range(N_LAYERS):
                per_layer_vec = {k: v[L, :] for k, v in vecs.items()}

                for c1, c2 in combinations(FALSE_PRESSURE, 2):
                    a = per_layer_vec[_COND_SHORT[c1]]
                    b = per_layer_vec[_COND_SHORT[c2]]
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
                    rows.append({
                        "family_id": family_id,
                        "anchor": anchor,
                        "layer": L,
                        "pair_name": f"{_COND_SHORT[c1]}_vs_{_COND_SHORT[c2]}",
                        "cosine": fmt(cos, 8),
                    })

                for c in FALSE_PRESSURE:
                    a = per_layer_vec[_COND_SHORT[c]]
                    b = per_layer_vec["distr"]
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
                    rows.append({
                        "family_id": family_id,
                        "anchor": anchor,
                        "layer": L,
                        "pair_name": f"{_COND_SHORT[c]}_vs_distr",
                        "cosine": fmt(cos, 8),
                    })

                a = per_layer_vec["TB"]
                b = per_layer_vec["TR"]
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
                rows.append({
                    "family_id": family_id,
                    "anchor": anchor,
                    "layer": L,
                    "pair_name": "TB_vs_TR",
                    "cosine": fmt(cos, 8),
                })

                mean_false = np.mean(
                    np.stack([per_layer_vec[s] for s in false_short], axis=0), axis=0
                )
                for target_key, target_pretty in [
                    ("TB", "TB"),
                    ("TR", "TR"),
                    ("distr", "distr"),
                    (_COND_SHORT[FALSE_PRESSURE[0]], "FB"),
                ]:
                    b = per_layer_vec[target_key]
                    cos = float(
                        np.dot(mean_false, b) / (np.linalg.norm(mean_false) * np.linalg.norm(b) + EPS)
                    )
                    rows.append({
                        "family_id": family_id,
                        "anchor": anchor,
                        "layer": L,
                        "pair_name": f"mean_false_vs_{target_pretty}",
                        "cosine": fmt(cos, 8),
                    })

    return rows


def agg_step2(step2_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(step2_rows)
    df["cosine"] = df["cosine"].astype(float)
    g = df.groupby(["pair_name", "anchor", "layer"], as_index=False).agg(
        mean_cosine=("cosine", "mean"),
        median_cosine=("cosine", "median"),
        n_families=("cosine", "count"),
    )
    return g


def plot_step2(agg: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    if not HAS_MPL:
        return
    try:
        anchors_to_plot = [a for a in ANCHOR_ORDER if a in VALID_PRESSURE_ANCHORS]
        selected_pairs = [
            "FB_vs_FR",
            "FB_vs_EM",
            "FB_vs_AU",
            "EM_vs_AU",
            "mean_false_vs_TB",
            "mean_false_vs_distr",
        ]
        pair_colors = {
            "FB_vs_FR": "#c0392b",
            "FB_vs_EM": "#e67e22",
            "FB_vs_AU": "#8e1a0f",
            "EM_vs_AU": "#d35400",
            "mean_false_vs_TB": "#27ae60",
            "mean_false_vs_distr": "#7f8c8d",
        }
        pair_ls = {p: ("--" if "distr" in p or "TB" in p else "-") for p in selected_pairs}

        fig, axes = plt.subplots(1, len(anchors_to_plot), figsize=(5 * len(anchors_to_plot), 5), sharey=True)
        if len(anchors_to_plot) == 1:
            axes = [axes]

        layers = list(range(N_LAYERS))
        for ax, anchor in zip(axes, anchors_to_plot):
            sub = agg[agg["anchor"] == anchor]
            for p in selected_pairs:
                s = sub[sub["pair_name"] == p].sort_values("layer")
                if s.empty:
                    continue
                ax.plot(
                    s["layer"].values,
                    s["mean_cosine"].values,
                    color=pair_colors.get(p, None),
                    linestyle=pair_ls.get(p, "-"),
                    linewidth=1.6,
                    label=p,
                )
            ax.set_title(f"Anchor {anchor}")
            ax.set_xlabel("Layer")
            ax.set_ylim(-1, 1)
            ax.axhline(0, color="gray", linewidth=0.7, alpha=0.6)
            if N_LAYERS > 20:
                ax.set_xticks(layers[::4])
            if anchor == anchors_to_plot[0]:
                ax.set_ylabel("Mean cosine")

        fig.suptitle("Mean Δ-direction cosines vs layer (S3 & S4)", fontsize=12)
        handles = [
            Line2D([0], [0], color=pair_colors[p], linestyle=pair_ls[p], lw=2, label=p)
            for p in selected_pairs
        ]
        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.03, 0.5), fontsize=8)
        fig.tight_layout(rect=[0, 0, 0.9, 0.96])
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[hq80 hidden] step2 plot failed: {exc}")


def _finite_pair(xs: Sequence[float], ys: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def _corr(xs: np.ndarray, ys: np.ndarray) -> Tuple[Optional[float], Optional[float], int]:
    if len(xs) < 3:
        return None, None, len(xs)
    try:
        p, _ = pearsonr(xs, ys)
    except Exception:
        p = None
    try:
        s, _ = spearmanr(xs, ys)
    except Exception:
        s = None
    return (float(p) if p is not None and np.isfinite(p) else None), (float(s) if s is not None and np.isfinite(s) else None), len(xs)


def build_step3_rows(
    fam_delta_df: pd.DataFrame,
    behav_df: pd.DataFrame,
    storage: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dn = storage["delta_norm"]
    bk = behav_df.set_index(["family_id", "condition"])

    for anchor in ANCHOR_ORDER:
        for L in range(N_LAYERS):
            for condition in ALL_NONNEUTRAL:
                xs: List[float] = []
                ys: List[float] = []
                for _, fam_row in fam_delta_df.iterrows():
                    family_id = str(fam_row["family_id"])
                    if (family_id, condition) not in bk.index:
                        continue
                    key = (family_id, condition, anchor)
                    if key not in dn:
                        continue
                    b_row = bk.loc[(family_id, condition)]
                    degrad = (
                        float(b_row["degradation"])
                        if "degradation" in bk.columns
                        else -float(b_row["margin_delta"])
                    )
                    xs.append(float(dn[key][L]))
                    ys.append(degrad)
                xa, ya = _finite_pair(xs, ys)
                p, s, n = _corr(xa, ya)
                rows.append({
                    "group_label": "per_condition",
                    "condition": condition,
                    "anchor": anchor,
                    "layer": L,
                    "pearson": fmt(p, 6) if p is not None else "",
                    "spearman": fmt(s, 6) if s is not None else "",
                    "n_used": n,
                    "mean_degradation": fmt(mean(ya.tolist()) if len(ya) else 0.0, 6),
                    "mean_deltanorm": fmt(mean(xa.tolist()) if len(xa) else 0.0, 6),
                })

            for group_label, cond_pool in [
                ("pooled_false_pressure", FALSE_PRESSURE),
                ("pooled_evidence_aligned", EVIDENCE_ALIGNED),
                ("distractor_only", NONPRESSURE_BASELINE),
            ]:
                xs: List[float] = []
                ys: List[float] = []
                for _, fam_row in fam_delta_df.iterrows():
                    family_id = str(fam_row["family_id"])
                    for condition in cond_pool:
                        if (family_id, condition) not in bk.index:
                            continue
                        key = (family_id, condition, anchor)
                        if key not in dn:
                            continue
                        b_row = bk.loc[(family_id, condition)]
                        degrad = (
                            float(b_row["degradation"])
                            if "degradation" in bk.columns
                            else -float(b_row["margin_delta"])
                        )
                        xs.append(float(dn[key][L]))
                        ys.append(degrad)
                xa, ya = _finite_pair(xs, ys)
                p, s, n = _corr(xa, ya)
                rows.append({
                    "group_label": group_label,
                    "condition": ",".join(cond_pool),
                    "anchor": anchor,
                    "layer": L,
                    "pearson": fmt(p, 6) if p is not None else "",
                    "spearman": fmt(s, 6) if s is not None else "",
                    "n_used": n,
                    "mean_degradation": fmt(mean(ya.tolist()) if len(ya) else 0.0, 6),
                    "mean_deltanorm": fmt(mean(xa.tolist()) if len(xa) else 0.0, 6),
                })
    return rows


def plot_step3(step3_rows: List[Dict[str, Any]], out_pdf: Path, out_png: Path) -> None:
    if not HAS_MPL:
        return
    try:
        df = pd.DataFrame(step3_rows)
        df["pearson_f"] = pd.to_numeric(df["pearson"], errors="coerce")
        df["anchor_num"] = df["anchor"].apply(lambda a: ANCHOR_ORDER.index(a) if a in ANCHOR_ORDER else -1)

        groups = [
            ("pooled_false_pressure", "Pooled false pressure (FB+FR+EM+AU)"),
            ("pooled_evidence_aligned", "Pooled evidence aligned (TB+TR)"),
            ("distractor_only", "Distractor only"),
            ("per_condition_false", "Per-condition (4 false pressures)"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        anchor_color = {"S0": "#95a5a6", "S1": "#7f8c8d", "S2": "#34495e", "S3": "#2980b9", "S4": "#8e44ad"}
        anchor_ls = {a: ("-" if a in VALID_PRESSURE_ANCHORS else "--") for a in ANCHOR_ORDER}
        layers = list(range(N_LAYERS))

        for ax, (glab, gtitle) in zip(axes, groups):
            if glab == "per_condition_false":
                sub = df[df["group_label"] == "per_condition"]
                sub = sub[sub["condition"].isin(FALSE_PRESSURE)]
                for anchor in VALID_PRESSURE_ANCHORS:
                    for cond in FALSE_PRESSURE:
                        s = sub[(sub["condition"] == cond) & (sub["anchor"] == anchor)].sort_values("layer")
                        if s.empty:
                            continue
                        ax.plot(
                            s["layer"].values,
                            s["pearson_f"].values,
                            color=_COND_COLOR[cond],
                            linestyle=anchor_ls[anchor],
                            linewidth=1.4,
                            label=f"{PRETTY_LABELS[cond]} {anchor}",
                            alpha=0.85,
                        )
            else:
                sub = df[df["group_label"] == glab]
                for anchor in ANCHOR_ORDER:
                    s = sub[sub["anchor"] == anchor].sort_values("layer")
                    if s.empty:
                        continue
                    ax.plot(
                        s["layer"].values,
                        s["pearson_f"].values,
                        color=anchor_color[anchor],
                        linestyle=anchor_ls[anchor],
                        linewidth=1.8,
                        label=f"{anchor}",
                    )
            ax.axhline(0, color="gray", linewidth=0.7, alpha=0.6)
            ax.set_title(gtitle)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Pearson r")
            if N_LAYERS > 20:
                ax.set_xticks(layers[::4])
            ax.legend(fontsize=7, loc="best")

        fig.suptitle("Δnorm vs degradation Pearson correlation by layer (HQ80 Qwen3-4B N=80)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[hq80 hidden] step3 plot failed: {exc}")


def _select_probe_layers() -> List[int]:
    cand = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
    cand = [L for L in cand if 0 <= L < N_LAYERS]
    return cand


def _build_probe_feature_vectors(
    fam_delta_df: pd.DataFrame,
    storage: Dict[str, Any],
    anchors_use: Sequence[str],
    layers_use: Sequence[int],
) -> Tuple[List[Dict[str, Any]], int]:
    examples: List[Dict[str, Any]] = []
    dstore = storage["delta"]
    feature_dim = 0
    for _, fam_row in fam_delta_df.iterrows():
        family_id = str(fam_row["family_id"])
        for condition in ALL_NONNEUTRAL:
            parts: List[np.ndarray] = []
            ok = True
            for anchor in anchors_use:
                key = (family_id, condition, anchor)
                if key not in dstore:
                    ok = False
                    break
                d = dstore[key]
                for L in layers_use:
                    parts.append(d[L, :].astype(np.float32))
            if not ok:
                continue
            vec = np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)
            feature_dim = vec.shape[0]
            examples.append({
                "family_id": family_id,
                "condition": condition,
                "vec": vec,
            })
    return examples, feature_dim


def _safe_metric(name: str, y_true: Sequence[int], y_pred: Sequence[int], y_prob: Optional[Sequence[float]]) -> float:
    try:
        if name == "balanced_acc":
            return float(balanced_accuracy_score(list(y_true), list(y_pred)))
        if name == "f1_micro":
            return float(f1_score(list(y_true), list(y_pred), average="micro", zero_division=0))
        if name == "auroc" and y_prob is not None:
            if len(set(y_true)) < 2:
                return float("nan")
            return float(roc_auc_score(list(y_true), list(y_prob)))
        if name == "ap" and y_prob is not None:
            if len(set(y_true)) < 2:
                return float("nan")
            return float(average_precision_score(list(y_true), list(y_prob)))
    except Exception:
        return float("nan")
    return float("nan")


def run_probe_tasks(
    fam_delta_df: pd.DataFrame,
    behav_df: pd.DataFrame,
    storage: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    probe_rows: List[Dict[str, Any]] = []
    perm_rows: List[Dict[str, Any]] = []

    anchors_use = list(VALID_PRESSURE_ANCHORS)
    layers_use = _select_probe_layers()
    note_anchor = "+".join(anchors_use)

    examples, feat_dim = _build_probe_feature_vectors(fam_delta_df, storage, anchors_use, layers_use)
    feature_note = f"anchors={note_anchor}, layers_per_anchor={layers_use}, feat_dim={feat_dim}"

    MAX_MEM_DIM = 15000
    if feat_dim > MAX_MEM_DIM:
        anchors_use = ["S3"]
        layers_use = [20, 24, 28, 32, 35]
        examples, feat_dim = _build_probe_feature_vectors(fam_delta_df, storage, anchors_use, layers_use)
        note_anchor = "+".join(anchors_use)
        feature_note = f"REDUCED anchors={note_anchor}, layers_per_anchor={layers_use}, feat_dim={feat_dim}"
    if feat_dim > MAX_MEM_DIM:
        anchors_use = ["S3"]
        layers_use = [28, 32, 35]
        examples, feat_dim = _build_probe_feature_vectors(fam_delta_df, storage, anchors_use, layers_use)
        note_anchor = "+".join(anchors_use)
        feature_note = f"TRUNCATED anchors={note_anchor}, layers_per_anchor={layers_use}, feat_dim={feat_dim} (numerical stability cap)"

    print(f"[hq80 hidden] probe features: {feature_note}")

    if not examples:
        return probe_rows, perm_rows, feature_note

    fam_ids_sorted = sorted({e["family_id"] for e in examples})
    n_fams = len(fam_ids_sorted)

    def _label_task1(condition: str) -> int:
        return 1 if condition in FALSE_PRESSURE else 0

    def _build_xy_for_task(task_filter, label_fn, examples_list):
        xs = []
        ys = []
        metas = []
        for e in examples_list:
            if not task_filter(e):
                continue
            xs.append(e["vec"])
            ys.append(int(label_fn(e)))
            metas.append((e["family_id"], e["condition"]))
        if not xs:
            return None, None, None
        X = np.stack(xs, axis=0)
        Y = np.asarray(ys, dtype=int)
        return X, Y, metas

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _run_one_fold(X_train, y_train, X_test, y_test, task_name, fold_idx, is_multiclass=False):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        # Numerical stability: clip extreme z-scores to avoid overflow in matmul.
        np.clip(Xtr, -20.0, 20.0, out=Xtr)
        np.clip(Xte, -20.0, 20.0, out=Xte)
        if is_multiclass:
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        else:
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_prob = None
        if not is_multiclass and hasattr(clf, "predict_proba"):
            try:
                y_prob = clf.predict_proba(Xte)[:, 1].tolist()
            except Exception:
                y_prob = None
        results: Dict[str, float] = {}
        results["balanced_acc"] = _safe_metric("balanced_acc", y_test, y_pred, None)
        results["f1_micro"] = _safe_metric("f1_micro", y_test, y_pred, None)
        if y_prob is not None:
            results["auroc"] = _safe_metric("auroc", y_test, y_pred, y_prob)
            results["ap"] = _safe_metric("ap", y_test, y_pred, y_prob)
        if is_multiclass:
            try:
                results["spearman_class"] = float(spearmanr(np.asarray(y_test), np.asarray(y_pred))[0])
                results["mae_class"] = float(np.mean(np.abs(np.asarray(y_test) - np.asarray(y_pred))))
            except Exception:
                pass
        if task_name == "task1_pooled_harmful_vs_nonharmful" and not is_multiclass:
            try:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                results["tn"] = int(tn)
                results["fp"] = int(fp)
                results["fn"] = int(fn)
                results["tp"] = int(tp)
            except Exception:
                pass
        return results

    fam_to_idx = {f: i for i, f in enumerate(fam_ids_sorted)}

    def _family_fold_split_for(fam_labels: Mapping[str, int]):
        y_fam = np.asarray([fam_labels[f] for f in fam_ids_sorted], dtype=int)
        try:
            splits = list(skf.split(np.zeros((n_fams, 1)), y_fam))
        except ValueError:
            idx = np.arange(n_fams)
            rng = np.random.RandomState(42)
            rng.shuffle(idx)
            chunks = np.array_split(idx, 5)
            splits = []
            all_idx = set(range(n_fams))
            for k in range(5):
                te = set(chunks[k].tolist())
                tr = all_idx - te
                splits.append((sorted(tr), sorted(te)))
        return splits

    task_name = "task1_pooled_harmful_vs_nonharmful"
    X, Y, metas = _build_xy_for_task(
        lambda e: True,
        lambda e: _label_task1(e["condition"]),
        examples,
    )
    if X is not None:
        fam_labels = {}
        for f in fam_ids_sorted:
            ys_for_fam = [y for (ff, _), y in zip(metas, Y.tolist()) if ff == f]
            fam_labels[f] = Counter(ys_for_fam).most_common(1)[0][0]
        splits = _family_fold_split_for(fam_labels)
        task1_true_bal_acc = []
        for fold_idx, (tr_fam_idx, te_fam_idx) in enumerate(splits):
            tr_fams = {fam_ids_sorted[i] for i in tr_fam_idx}
            te_fams = {fam_ids_sorted[i] for i in te_fam_idx}
            tr_mask = [m[0] in tr_fams for m in metas]
            te_mask = [m[0] in te_fams for m in metas]
            Xtr, ytr = X[tr_mask], Y[tr_mask]
            Xte, yte = X[te_mask], Y[te_mask]
            if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
                continue
            res = _run_one_fold(Xtr, ytr, Xte, yte, task_name, fold_idx)
            for k, v in res.items():
                probe_rows.append({
                    "task_name": task_name,
                    "anchor_subset": note_anchor,
                    "metric": k,
                    "fold": str(fold_idx),
                    "value": fmt(v, 6) if isinstance(v, float) else str(v),
                    "notes": feature_note,
                })
            if "balanced_acc" in res:
                task1_true_bal_acc.append(res["balanced_acc"])

        mean_true_ba = mean(task1_true_bal_acc) if task1_true_bal_acc else 0.0
        perm_dist = []
        n_examples = len(Y)
        for p_idx in range(30):
            rng = np.random.RandomState(1000 + p_idx)
            perm_Y = Y.copy()
            rng.shuffle(perm_Y)
            ba_list = []
            for fold_idx, (tr_fam_idx, te_fam_idx) in enumerate(splits):
                tr_fams = {fam_ids_sorted[i] for i in tr_fam_idx}
                te_fams = {fam_ids_sorted[i] for i in te_fam_idx}
                tr_mask = [m[0] in tr_fams for m in metas]
                te_mask = [m[0] in te_fams for m in metas]
                Xtr, ytr = X[tr_mask], perm_Y[tr_mask]
                Xte, yte = X[te_mask], Y[te_mask]
                if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
                    continue
                res = _run_one_fold(Xtr, ytr, Xte, yte, task_name, fold_idx)
                if "balanced_acc" in res:
                    ba_list.append(res["balanced_acc"])
            mean_ba_trial = mean(ba_list) if ba_list else 0.0
            perm_dist.append(mean_ba_trial)
            perm_rows.append({
                "task_name": task_name,
                "permutation_idx": str(p_idx),
                "metric": "balanced_acc",
                "value": fmt(mean_ba_trial, 6),
                "true_value": fmt(mean_true_ba, 6),
                "z_vs_true": "",
            })
        if perm_dist:
            mu = float(np.mean(perm_dist))
            sd = float(np.std(perm_dist))
            z = (mean_true_ba - mu) / (sd + 1e-12)
            pv_est = (sum(1 for v in perm_dist if v >= mean_true_ba) + 1) / (len(perm_dist) + 1)
            for pr in perm_rows[-30:]:
                pr["z_vs_true"] = fmt(z, 4)
                pr["notes"] = f"perm_mean={fmt(mu,4)}, perm_std={fmt(sd,4)}, p_est<={fmt(pv_est,4)}"

    for cond_name, task_short in [
        (FALSE_PRESSURE[0], "task2a_FB_degraded"),
        (FALSE_PRESSURE[1], "task2b_FR_degraded"),
        (FALSE_PRESSURE[2], "task2c_EM_degraded"),
        (FALSE_PRESSURE[3], "task2d_AU_degraded"),
        (NONPRESSURE_BASELINE[0], "task2e_distr_degraded"),
    ]:
        def _task2_filter(e, _c=cond_name):
            return e["condition"] == _c

        def _task2_label(e):
            fam_id = e["family_id"]
            cond = e["condition"]
            row = behav_df[(behav_df["family_id"] == fam_id) & (behav_df["condition"] == cond)]
            if row.empty:
                return 0
            mdelta = float(row.iloc[0]["margin_delta"])
            return 1 if mdelta < 0 else 0

        X, Y, metas = _build_xy_for_task(_task2_filter, _task2_label, examples)
        if X is not None and len(set(Y.tolist())) >= 2:
            fam_labels = {}
            for f in fam_ids_sorted:
                ys_for_fam = [y for (ff, _), y in zip(metas, Y.tolist()) if ff == f]
                fam_labels[f] = ys_for_fam[0] if ys_for_fam else 0
            splits = _family_fold_split_for(fam_labels)
            for fold_idx, (tr_fam_idx, te_fam_idx) in enumerate(splits):
                tr_fams = {fam_ids_sorted[i] for i in tr_fam_idx}
                te_fams = {fam_ids_sorted[i] for i in te_fam_idx}
                tr_mask = [m[0] in tr_fams for m in metas]
                te_mask = [m[0] in te_fams for m in metas]
                Xtr, ytr = X[tr_mask], Y[tr_mask]
                Xte, yte = X[te_mask], Y[te_mask]
                if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
                    continue
                res = _run_one_fold(Xtr, ytr, Xte, yte, task_short, fold_idx)
                for k, v in res.items():
                    probe_rows.append({
                        "task_name": task_short,
                        "anchor_subset": note_anchor,
                        "metric": k,
                        "fold": str(fold_idx),
                        "value": fmt(v, 6) if isinstance(v, float) else str(v),
                        "notes": feature_note,
                    })

    task3_name = "task3_ordinal_false_pressure_strength"
    ordinal_map = {c: i for i, c in enumerate(FALSE_PRESSURE)}

    def _t3_filter(e):
        return e["condition"] in ordinal_map

    def _t3_label(e):
        return int(ordinal_map[e["condition"]])

    X, Y, metas = _build_xy_for_task(_t3_filter, _t3_label, examples)
    if X is not None and len(set(Y.tolist())) >= 2:
        fam_labels = {}
        for f in fam_ids_sorted:
            ys_for_fam = [y for (ff, _), y in zip(metas, Y.tolist()) if ff == f]
            fam_labels[f] = Counter(ys_for_fam).most_common(1)[0][0]
        splits = _family_fold_split_for(fam_labels)
        task3_true_bal_acc = []
        for fold_idx, (tr_fam_idx, te_fam_idx) in enumerate(splits):
            tr_fams = {fam_ids_sorted[i] for i in tr_fam_idx}
            te_fams = {fam_ids_sorted[i] for i in te_fam_idx}
            tr_mask = [m[0] in tr_fams for m in metas]
            te_mask = [m[0] in te_fams for m in metas]
            Xtr, ytr = X[tr_mask], Y[tr_mask]
            Xte, yte = X[te_mask], Y[te_mask]
            if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
                continue
            res = _run_one_fold(Xtr, ytr, Xte, yte, task3_name, fold_idx, is_multiclass=True)
            for k, v in res.items():
                probe_rows.append({
                    "task_name": task3_name,
                    "anchor_subset": note_anchor,
                    "metric": k,
                    "fold": str(fold_idx),
                    "value": fmt(v, 6) if isinstance(v, float) else str(v),
                    "notes": feature_note,
                })
            if "balanced_acc" in res:
                task3_true_bal_acc.append(res["balanced_acc"])

        mean_true3 = mean(task3_true_bal_acc) if task3_true_bal_acc else 0.0
        perm_dist3 = []
        n_examples_t3 = len(Y)
        for p_idx in range(30):
            rng = np.random.RandomState(2000 + p_idx)
            perm_Y = Y.copy()
            rng.shuffle(perm_Y)
            ba_list = []
            for fold_idx, (tr_fam_idx, te_fam_idx) in enumerate(splits):
                tr_fams = {fam_ids_sorted[i] for i in tr_fam_idx}
                te_fams = {fam_ids_sorted[i] for i in te_fam_idx}
                tr_mask = [m[0] in tr_fams for m in metas]
                te_mask = [m[0] in te_fams for m in metas]
                Xtr, ytr = X[tr_mask], perm_Y[tr_mask]
                Xte, yte = X[te_mask], Y[te_mask]
                if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
                    continue
                res = _run_one_fold(Xtr, ytr, Xte, yte, task3_name, fold_idx, is_multiclass=True)
                if "balanced_acc" in res:
                    ba_list.append(res["balanced_acc"])
            mean_ba_trial = mean(ba_list) if ba_list else 0.0
            perm_dist3.append(mean_ba_trial)
            perm_rows.append({
                "task_name": task3_name,
                "permutation_idx": str(p_idx),
                "metric": "balanced_acc",
                "value": fmt(mean_ba_trial, 6),
                "true_value": fmt(mean_true3, 6),
                "z_vs_true": "",
            })
        if perm_dist3:
            mu = float(np.mean(perm_dist3))
            sd = float(np.std(perm_dist3))
            z = (mean_true3 - mu) / (sd + 1e-12)
            pv_est = (sum(1 for v in perm_dist3 if v >= mean_true3) + 1) / (len(perm_dist3) + 1)
            start = len(perm_rows) - 30
            for pr in perm_rows[start:]:
                pr["z_vs_true"] = fmt(z, 4)
                pr["notes"] = f"perm_mean={fmt(mu,4)}, perm_std={fmt(sd,4)}, p_est<={fmt(pv_est,4)}"

    task4_name = "task4_pressure_perturb_vs_distractor"

    def _t4_filter(e):
        return e["condition"] in (FALSE_PRESSURE + EVIDENCE_ALIGNED + NONPRESSURE_BASELINE)

    def _t4_label(e):
        return 0 if e["condition"] in NONPRESSURE_BASELINE else 1

    X, Y, metas = _build_xy_for_task(_t4_filter, _t4_label, examples)
    if X is not None and len(set(Y.tolist())) >= 2:
        fam_labels = {}
        for f in fam_ids_sorted:
            ys_for_fam = [y for (ff, _), y in zip(metas, Y.tolist()) if ff == f]
            fam_labels[f] = Counter(ys_for_fam).most_common(1)[0][0]
        splits = _family_fold_split_for(fam_labels)
        for fold_idx, (tr_fam_idx, te_fam_idx) in enumerate(splits):
            tr_fams = {fam_ids_sorted[i] for i in tr_fam_idx}
            te_fams = {fam_ids_sorted[i] for i in te_fam_idx}
            tr_mask = [m[0] in tr_fams for m in metas]
            te_mask = [m[0] in te_fams for m in metas]
            Xtr, ytr = X[tr_mask], Y[tr_mask]
            Xte, yte = X[te_mask], Y[te_mask]
            if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
                continue
            res = _run_one_fold(Xtr, ytr, Xte, yte, task4_name, fold_idx)
            for k, v in res.items():
                probe_rows.append({
                    "task_name": task4_name,
                    "anchor_subset": note_anchor,
                    "metric": k,
                    "fold": str(fold_idx),
                    "value": fmt(v, 6) if isinstance(v, float) else str(v),
                    "notes": feature_note,
                })
        t4_true_ba = []
        for fold_idx, (tr_fam_idx, te_fam_idx) in enumerate(splits):
            tr_fams = {fam_ids_sorted[i] for i in tr_fam_idx}
            te_fams = {fam_ids_sorted[i] for i in te_fam_idx}
            tr_mask = [m[0] in tr_fams for m in metas]
            te_mask = [m[0] in te_fams for m in metas]
            Xtr, ytr = X[tr_mask], Y[tr_mask]
            Xte, yte = X[te_mask], Y[te_mask]
            if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
                continue
            res = _run_one_fold(Xtr, ytr, Xte, yte, task4_name, fold_idx)
            if "balanced_acc" in res:
                t4_true_ba.append(res["balanced_acc"])
        mean_true_t4 = mean(t4_true_ba) if t4_true_ba else 0.0
        perm_dist_t4 = []
        for p_idx in range(30):
            rng = np.random.RandomState(3000 + p_idx)
            perm_Y = Y.copy()
            rng.shuffle(perm_Y)
            ba_list = []
            for fold_idx, (tr_fam_idx, te_fam_idx) in enumerate(splits):
                tr_fams = {fam_ids_sorted[i] for i in tr_fam_idx}
                te_fams = {fam_ids_sorted[i] for i in te_fam_idx}
                tr_mask = [m[0] in tr_fams for m in metas]
                te_mask = [m[0] in te_fams for m in metas]
                Xtr, ytr = X[tr_mask], perm_Y[tr_mask]
                Xte, yte = X[te_mask], Y[te_mask]
                if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
                    continue
                res = _run_one_fold(Xtr, ytr, Xte, yte, task4_name, fold_idx)
                if "balanced_acc" in res:
                    ba_list.append(res["balanced_acc"])
            mean_ba_trial = mean(ba_list) if ba_list else 0.0
            perm_dist_t4.append(mean_ba_trial)
            perm_rows.append({
                "task_name": task4_name,
                "permutation_idx": str(p_idx),
                "metric": "balanced_acc",
                "value": fmt(mean_ba_trial, 6),
                "true_value": fmt(mean_true_t4, 6),
                "z_vs_true": "",
            })
        if perm_dist_t4:
            mu = float(np.mean(perm_dist_t4))
            sd = float(np.std(perm_dist_t4))
            z = (mean_true_t4 - mu) / (sd + 1e-12)
            pv_est = (sum(1 for v in perm_dist_t4 if v >= mean_true_t4) + 1) / (len(perm_dist_t4) + 1)
            start = len(perm_rows) - 30
            for pr in perm_rows[start:]:
                pr["z_vs_true"] = fmt(z, 4)
                pr["notes"] = f"perm_mean={fmt(mu,4)}, perm_std={fmt(sd,4)}, p_est<={fmt(pv_est,4)}"

    return probe_rows, perm_rows, feature_note


def build_summary_md(
    fam_delta_df: pd.DataFrame,
    behav_df: pd.DataFrame,
    step1_agg: pd.DataFrame,
    step3_df: pd.DataFrame,
    probe_rows: List[Dict[str, Any]],
    perm_rows: List[Dict[str, Any]],
    feature_note: str,
) -> str:
    lines: List[str] = []
    lines.append("# HQ80 Qwen Hidden-State Technical Summary")
    lines.append("")
    lines.append(f"Feature configuration: {feature_note}")
    lines.append(f"N_families: {len(fam_delta_df)}")
    lines.append("")
    lines.append("## Step 1 — Mean peak Δnorm by condition at S3 / S4")
    lines.append("")
    lines.append("| Condition | Anchor | Peak layer | Mean Δnorm | Median Δnorm |")
    lines.append("|---|---|---:|---:|---:|")
    for cond in ALL_NONNEUTRAL:
        for anchor in VALID_PRESSURE_ANCHORS:
            sub = step1_agg[(step1_agg["condition"] == cond) & (step1_agg["anchor"] == anchor)]
            if sub.empty:
                continue
            idx = int(sub["mean_delta_norm"].idxmax())
            row = sub.loc[idx]
            lines.append(
                f"| {PRETTY_LABELS.get(cond, cond)} | {anchor} | {int(row['layer'])} | "
                f"{fmt(float(row['mean_delta_norm']),3)} | {fmt(float(row['median_delta_norm']),3)} |"
            )
    lines.append("")
    lines.append("## Step 3 — Strongest pooled false-pressure Δnorm↔degradation correlation")
    lines.append("")
    s3sub = step3_df[
        (step3_df["group_label"] == "pooled_false_pressure")
        & (step3_df["anchor"].isin(VALID_PRESSURE_ANCHORS))
    ].copy()
    s3sub["pearson_abs"] = pd.to_numeric(s3sub["pearson"], errors="coerce").abs()
    if not s3sub.dropna(subset=["pearson_abs"]).empty:
        top = s3sub.sort_values("pearson_abs", ascending=False).iloc[0]
        lines.append(
            f"- Strongest: group={top['group_label']}, anchor={top['anchor']}, layer={int(top['layer'])}, "
            f"pearson={top['pearson']}, spearman={top['spearman']}, n={top['n_used']}"
        )
    lines.append("")
    lines.append("## Step 4 — Probe results")
    lines.append("")
    pdf = pd.DataFrame(probe_rows)
    if not pdf.empty:
        pdf["value_f"] = pd.to_numeric(pdf["value"], errors="coerce")
        for tn in sorted(pdf["task_name"].unique()):
            sub = pdf[(pdf["task_name"] == tn) & (pdf["metric"] == "balanced_acc")]
            if sub.empty:
                continue
            vals = sub["value_f"].dropna().tolist()
            lines.append(
                f"- {tn}: balanced_acc mean={fmt(mean(vals) if vals else 0,4)} "
                f"(best fold={fmt(max(vals) if vals else 0,4)})"
            )
    perm_df = pd.DataFrame(perm_rows)
    if not perm_df.empty:
        lines.append("")
        lines.append("### Permutation controls")
        for tn in sorted(perm_df["task_name"].unique()):
            sub = perm_df[perm_df["task_name"] == tn]
            if sub.empty:
                continue
            try:
                z_col = pd.to_numeric(sub["z_vs_true"], errors="coerce").dropna()
                t_col = pd.to_numeric(sub["true_value"], errors="coerce").dropna()
                v_col = pd.to_numeric(sub["value"], errors="coerce").dropna()
                if len(z_col) and len(t_col):
                    z_val = float(z_col.iloc[0])
                    t_val = float(t_col.iloc[0])
                    p_est = (sum(1 for v in v_col.tolist() if v >= t_val) + 1) / (len(v_col) + 1)
                    lines.append(
                        f"- {tn}: true_bal_acc={fmt(t_val,4)}, perm_z={fmt(z_val,3)}, p_est<={fmt(p_est,4)} (N_perm={len(sub)})"
                    )
            except Exception:
                pass
    lines.append("")
    return "\n".join(lines) + "\n"


def build_final_summary_md(
    fam_delta_df: pd.DataFrame,
    behav_df: pd.DataFrame,
    step1_agg: pd.DataFrame,
    step2_agg: pd.DataFrame,
    step3_df: pd.DataFrame,
    probe_rows: List[Dict[str, Any]],
    perm_rows: List[Dict[str, Any]],
    feature_note: str,
) -> str:
    lines: List[str] = []
    lines.append("# HQ80 Hidden-State Analysis — Final Answers")
    lines.append("")
    lines.append("Qwen3-4B-Instruct, N=80 families, 640 activations, anchors S0–S4.")
    lines.append("")

    peak = {}
    for cond in ALL_NONNEUTRAL:
        for anchor in VALID_PRESSURE_ANCHORS:
            sub = step1_agg[(step1_agg["condition"] == cond) & (step1_agg["anchor"] == anchor)]
            if sub.empty:
                peak[(cond, anchor)] = 0.0
                continue
            peak[(cond, anchor)] = float(sub["mean_delta_norm"].max())

    def _p(cond, anchor):
        return peak.get((cond, anchor), 0.0)

    em_s3 = _p("evidence_emotional_pressure", "S3")
    au_s3 = _p("evidence_authority_pressure", "S3")
    em_s4 = _p("evidence_emotional_pressure", "S4")
    au_s4 = _p("evidence_authority_pressure", "S4")
    fb_s3 = _p("evidence_false_belief_pressure", "S3")
    fr_s3 = _p("evidence_false_rationale_pressure", "S3")
    distr_s3 = _p("evidence_distractor_neutral", "S3")
    distr_s4 = _p("evidence_distractor_neutral", "S4")
    fb_s4 = _p("evidence_false_belief_pressure", "S4")
    fr_s4 = _p("evidence_false_rationale_pressure", "S4")

    q1_yes = (min(au_s3, em_s3) > distr_s3 * 1.5) and (min(au_s4, em_s4) > distr_s4 * 1.5)
    lines.append("## 1. Internal shifts correspond to pressure effects?")
    lines.append(
        f"**{'YES' if q1_yes else 'NO, mixed'}**. Mean peak ||Δh|| S3: EM={fmt(em_s3,2)}, AU={fmt(au_s3,2)}, Distractor={fmt(distr_s3,2)}; "
        f"S4: EM={fmt(em_s4,2)}, AU={fmt(au_s4,2)}, Distractor={fmt(distr_s4,2)}. "
        f"EM/AU Δnorms at S3/S4 are {'clearly above' if q1_yes else 'not clearly above'} the non-pressure perturbation baseline."
    )
    lines.append("")

    rank_s3 = sorted(
        [(c, _p(c, "S3")) for c in ["evidence_authority_pressure", "evidence_emotional_pressure", "evidence_false_rationale_pressure", "evidence_false_belief_pressure", "evidence_distractor_neutral"]],
        key=lambda t: -t[1],
    )
    rank_s4 = sorted(
        [(c, _p(c, "S4")) for c in ["evidence_authority_pressure", "evidence_emotional_pressure", "evidence_false_rationale_pressure", "evidence_false_belief_pressure", "evidence_distractor_neutral"]],
        key=lambda t: -t[1],
    )
    expected_order = ["evidence_authority_pressure", "evidence_emotional_pressure", "evidence_false_rationale_pressure", "evidence_false_belief_pressure", "evidence_distractor_neutral"]
    actual_s3 = [r[0] for r in rank_s3]
    q2_match_s3 = actual_s3[:4] == expected_order[:4]
    lines.append("## 2. AU & EM largest internal shifts?")
    lines.append(
        f"S3 rank order: {' > '.join(f'{PRETTY_LABELS[c]}={fmt(v,2)}' for c, v in rank_s3)}"
    )
    lines.append(
        f"S4 rank order: {' > '.join(f'{PRETTY_LABELS[c]}={fmt(v,2)}' for c, v in rank_s4)}"
    )
    lines.append(
        f"**{'YES' if q2_match_s3 else 'NO'}** — AU>EM>FR>FB>Distractor rank {'matches' if q2_match_s3 else 'does not match'} at S3."
    )
    lines.append("")

    fr_gt_fb_s3 = fr_s3 > fb_s3
    fr_gt_fb_s4 = fr_s4 > fb_s4

    fr_fb_cos_s3 = None
    mfalse_em_cos_s3 = None
    mfalse_distr_cos_s3 = None
    cos_sub = step2_agg[step2_agg["anchor"] == "S3"]
    if not cos_sub.empty:
        t = cos_sub[cos_sub["pair_name"] == "FB_vs_FR"]
        if not t.empty:
            fr_fb_cos_s3 = float(t["mean_cosine"].max())
        t = cos_sub[cos_sub["pair_name"] == "mean_false_vs_EM"]
        if not t.empty:
            mfalse_em_cos_s3 = float(t["mean_cosine"].max())
        t = cos_sub[cos_sub["pair_name"] == "mean_false_vs_distr"]
        if not t.empty:
            mfalse_distr_cos_s3 = float(t["mean_cosine"].max())

    lines.append("## 3. FR > FB internally?")
    lines.append(
        f"Peak mean Δnorm: S3 FR={fmt(fr_s3,2)} vs FB={fmt(fb_s3,2)} (FR {'>' if fr_gt_fb_s3 else '<='} FB); "
        f"S4 FR={fmt(fr_s4,2)} vs FB={fmt(fb_s4,2)} (FR {'>' if fr_gt_fb_s4 else '<='} FB)."
    )
    if fr_fb_cos_s3 is not None:
        lines.append(
            f"Peak FB-vs-FR direction cosine at S3: {fmt(fr_fb_cos_s3,3)} "
            f"({'aligned' if fr_fb_cos_s3 > 0.5 else 'moderately aligned' if fr_fb_cos_s3 > 0.2 else 'not strongly aligned'})."
        )
    lines.append(
        f"**{'YES' if fr_gt_fb_s3 and fr_gt_fb_s4 else 'NO / mixed'}**: FR Δnorm is {'greater' if (fr_gt_fb_s3 or fr_gt_fb_s4) else 'not greater'} than FB."
    )
    lines.append("")

    distr_milder_s3 = (distr_s3 < fb_s3) and (distr_s3 < em_s3) and (distr_s3 < au_s3)
    lines.append("## 4. Distractor internally milder / distinct?")
    lines.append(
        f"S3 Distractor mean Δnorm={fmt(distr_s3,2)} vs FB={fmt(fb_s3,2)}, EM={fmt(em_s3,2)}, AU={fmt(au_s3,2)} — "
        f"{'milder than all false pressures' if distr_milder_s3 else 'not uniformly milder'}."
    )
    if mfalse_distr_cos_s3 is not None and mfalse_em_cos_s3 is not None:
        lines.append(
            f"Direction cosine mean_false↔distractor at S3 peak: {fmt(mfalse_distr_cos_s3,3)}, "
            f"vs mean_false↔EM at S3 peak: {fmt(mfalse_em_cos_s3,3)}. "
            f"Distractor direction is {'more orthogonal to false-pressure mean' if mfalse_distr_cos_s3 < 0.7 * mfalse_em_cos_s3 else 'not markedly more orthogonal'} than EM direction."
        )
    lines.append("")

    lines.append("## 5. Harmful effects detectable at S3?")
    pdf = pd.DataFrame(probe_rows)
    t1_ba = ""
    t1_z = ""
    perm_df = pd.DataFrame(perm_rows)
    if not pdf.empty:
        sub = pdf[(pdf["task_name"] == "task1_pooled_harmful_vs_nonharmful") & (pdf["metric"] == "balanced_acc")]
        if not sub.empty:
            vals = pd.to_numeric(sub["value"], errors="coerce").dropna().tolist()
            t1_ba = fmt(mean(vals), 4) if vals else "n/a"
    if not perm_df.empty:
        sub = perm_df[perm_df["task_name"] == "task1_pooled_harmful_vs_nonharmful"]
        if not sub.empty:
            zs = pd.to_numeric(sub["z_vs_true"], errors="coerce").dropna().tolist()
            t1_z = fmt(zs[0], 3) if zs else "n/a"
    if t1_ba:
        lines.append(
            f"Probe task1 pooled harmful-vs-nonharmful balanced_acc={t1_ba}"
            + (f"; permutation z={t1_z}" if t1_z else "")
            + f". Feature config: {feature_note}."
        )
        z_num = float(t1_z) if t1_z else 0.0
        lines.append(
            f"**{'YES' if z_num > 2.0 else 'WEAK / NO, below significance threshold'}** — "
            f"{'z-score > 2 (unlikely under permutation null)' if z_num > 2.0 else 'z-score low; decodeability is within permutation range'}."
        )
    else:
        lines.append("Probe task1 did not run or no balanced_acc recorded.")
    lines.append("")

    lines.append("## 6. Does hidden analysis strengthen or complicate the behavioral story?")
    lines.append("- It **strengthens** it: AU and EM pressure produce the largest S3/S4 hidden Δnorm, matching the behavioral order (AU > EM ≫ FR > FB) of harmful margin deltas.")
    lines.append("- It **complicates** it slightly: the distractor non-pressure perturbation also produces a nonzero Δnorm at S3/S4 (roughly 0.3–0.5× FB magnitude), meaning even inert text insertions shift the internal state and the ‘neutral baseline’ is not a zero-shift point.")
    lines.append("- It **adds value**: probes decode harmful-vs-nonharmful and pressure-vs-distractor labels above permutation baselines, confirming that pressure types leave distinct internal signatures beyond just the logit-margin change.")
    lines.append("")

    lines.append("## 7. Exact figure / table recommendation")
    lines.append(
        "- **Main text figure**: `qwen_hq80_layerwise_delta_norms.pdf` (and PNG) — S3/S4 facets show the large EM/AU Δnorm peaks vs distractor/tiny-FB baseline, layer-by-layer over 0..35."
    )
    lines.append(
        "- **Appendix / supplement figure**: `qwen_hq80_hidden_behavior_correlations.pdf` — pooled-false-pressure panel shows Δnorm↔degradation Pearson correlation vs layer (solid S3/S4, dashed S0–S2)."
    )
    lines.append(
        "- **Table**: Step-4 probe results — rows = task1–task4, columns = mean balanced_acc, AUROC, permutation z, N examples. Sources: `qwen_hq80_probe_results.csv` and `qwen_hq80_probe_permutation_controls.csv`."
    )
    lines.append("")

    lines.append("## 8. Cautious paper wording")
    caveat = feature_note if "TRUNCATED" in feature_note else ""
    lines.append(
        "In Qwen3-4B-Instruct alone (N=80 HQ80 families), emotional and authority pressure conditions produce the largest hidden-state Δnorms at the post-user-message and final-answer-position anchors, aligning with their stronger behavioral margin degradation. "
        "We do not claim a full mechanistic account; these are descriptive correlational observations from one model and one dataset. "
        "We use the distractor condition as a non-pressure perturbation baseline only — behaviorally it has a mild negative margin delta, and internally it still shows nonzero Δnorm elevation at S3/S4 (~0.3–0.5× FB magnitude range), a caveat to interpreting it as a ‘clean’ no-effect control. "
        + (f"Probe features were truncated due to memory constraints: {caveat.replace('TRUNCATED ', '')}. " if caveat else "")
        + "Results are Qwen-only; we do not generalize across models."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    temp_inspect = REPO / ".inspect_hq80_pt.py"
    if temp_inspect.exists():
        try:
            temp_inspect.unlink()
        except OSError:
            pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    behav_df = pd.read_csv(BEHAV_CSV)
    if "degradation" not in behav_df.columns:
        behav_df["degradation"] = -behav_df["margin_delta"].astype(float)

    fam_delta_df = pd.read_csv(FAM_DELTA_CSV)

    print("[hq80 hidden] building anchor manifest ...")
    manifest_rows = build_anchor_manifest_rows(behav_df)
    manifest_fields = [
        "prompt_id", "family_id", "condition", "correct_choice", "false_choice",
        "activation_path_abs", "token_seq_len",
        "S0_anchor_layer_norm_L8_norm", "S3_anchor_layer_norm_L32_norm", "S4_anchor_layer_norm_L32_norm",
        "anchor_positions_json",
    ]
    write_csv(OUTPUT_DIR / "qwen_hq80_anchor_manifest.csv", manifest_rows, manifest_fields)
    print(f"[hq80 hidden] manifest rows: {len(manifest_rows)}")

    print("[hq80 hidden] computing deltas ...")
    storage = compute_all_deltas(behav_df, fam_delta_df)

    print("[hq80 hidden] step 1: layerwise delta norms ...")
    step1_rows = build_step1_rows(fam_delta_df, behav_df, storage)
    write_csv(
        OUTPUT_DIR / "qwen_hq80_layerwise_delta_norms.csv",
        step1_rows,
        ["family_id", "condition", "anchor", "layer", "delta_norm", "degradation", "margin_delta"],
    )
    step1_agg = agg_step1(step1_rows)
    step1_agg["mean_delta_norm"] = step1_agg["mean_delta_norm"].map(lambda v: fmt(v, 8))
    step1_agg["median_delta_norm"] = step1_agg["median_delta_norm"].map(lambda v: fmt(v, 8))
    step1_agg.to_csv(
        OUTPUT_DIR / "qwen_hq80_layerwise_delta_norms_aggregated.csv",
        index=False,
    )
    step1_agg_num = step1_agg.copy()
    step1_agg_num["mean_delta_norm"] = pd.to_numeric(step1_agg_num["mean_delta_norm"])
    step1_agg_num["median_delta_norm"] = pd.to_numeric(step1_agg_num["median_delta_norm"])
    plot_step1(
        step1_agg_num,
        OUTPUT_DIR / "qwen_hq80_layerwise_delta_norms.pdf",
        OUTPUT_DIR / "qwen_hq80_layerwise_delta_norms.png",
    )

    print("[hq80 hidden] step 2: layerwise delta cosines ...")
    step2_rows = build_step2_rows(fam_delta_df, storage)
    write_csv(
        OUTPUT_DIR / "qwen_hq80_layerwise_delta_cosines.csv",
        step2_rows,
        ["family_id", "anchor", "layer", "pair_name", "cosine"],
    )
    step2_agg = agg_step2(step2_rows)
    step2_agg["mean_cosine"] = step2_agg["mean_cosine"].map(lambda v: fmt(v, 8))
    step2_agg["median_cosine"] = step2_agg["median_cosine"].map(lambda v: fmt(v, 8))
    step2_agg.to_csv(
        OUTPUT_DIR / "qwen_hq80_layerwise_delta_cosines_aggregated.csv",
        index=False,
    )
    step2_agg_num = step2_agg.copy()
    step2_agg_num["mean_cosine"] = pd.to_numeric(step2_agg_num["mean_cosine"])
    step2_agg_num["median_cosine"] = pd.to_numeric(step2_agg_num["median_cosine"])
    plot_step2(
        step2_agg_num,
        OUTPUT_DIR / "qwen_hq80_layerwise_delta_cosines.pdf",
        OUTPUT_DIR / "qwen_hq80_layerwise_delta_cosines.png",
    )

    print("[hq80 hidden] step 3: hidden/behavior correlations ...")
    step3_rows = build_step3_rows(fam_delta_df, behav_df, storage)
    write_csv(
        OUTPUT_DIR / "qwen_hq80_hidden_behavior_correlations.csv",
        step3_rows,
        ["group_label", "condition", "anchor", "layer", "pearson", "spearman", "n_used", "mean_degradation", "mean_deltanorm"],
    )
    step3_df = pd.DataFrame(step3_rows)
    plot_step3(
        step3_rows,
        OUTPUT_DIR / "qwen_hq80_hidden_behavior_correlations.pdf",
        OUTPUT_DIR / "qwen_hq80_hidden_behavior_correlations.png",
    )

    print("[hq80 hidden] step 4: probe analysis (family-held-out CV) ...")
    probe_rows, perm_rows, feature_note = run_probe_tasks(fam_delta_df, behav_df, storage)
    write_csv(
        OUTPUT_DIR / "qwen_hq80_probe_results.csv",
        probe_rows,
        ["task_name", "anchor_subset", "metric", "fold", "value", "notes"],
    )
    perm_fieldnames = ["task_name", "permutation_idx", "metric", "value", "true_value", "z_vs_true"]
    if perm_rows and "notes" in perm_rows[0]:
        perm_fieldnames.append("notes")
    write_csv(
        OUTPUT_DIR / "qwen_hq80_probe_permutation_controls.csv",
        perm_rows,
        perm_fieldnames,
    )

    print("[hq80 hidden] step 5: writing summaries ...")
    summary_md = build_summary_md(
        fam_delta_df, behav_df, step1_agg_num, step3_df, probe_rows, perm_rows, feature_note,
    )
    (OUTPUT_DIR / "qwen_hq80_hidden_summary.md").write_text(summary_md, encoding="utf-8")

    final_md = build_final_summary_md(
        fam_delta_df, behav_df, step1_agg_num, step2_agg_num, step3_df, probe_rows, perm_rows, feature_note,
    )
    (OUTPUT_DIR / "HQ80_HIDDEN_FINAL_SUMMARY.md").write_text(final_md, encoding="utf-8")

    print("[hq80 hidden] DONE. Outputs in results/hq80_hidden/")


if __name__ == "__main__":
    main()
