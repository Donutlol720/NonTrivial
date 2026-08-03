import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_FAMILY_DELTAS = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_ACTIVATION_ROOT = "activations/qwen3_4b_instruct_2507"
DEFAULT_LAYERWISE_OUTPUT = "results/probe5_cross_condition_layerwise.csv"
DEFAULT_PREDICTIONS_OUTPUT = "results/probe5_cross_condition_predictions.csv"
DEFAULT_SUMMARY_OUTPUT = "results/probe5_cross_condition_summary.txt"

CONDITIONS = [
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "closed_context_false_belief_pressure",
]

DELTA_CONDITION_TO_PROMPT_TYPE = {
    "evidence_false_belief_pressure": "evidence_false_belief_pressure",
    "evidence_true_belief_pressure": "evidence_true_belief_pressure",
    "evidence_emotional_pressure": "evidence_emotional_pressure",
    "evidence_distractor_neutral": "evidence_distractor_neutral",
    "closed_context_false_belief_pressure": "closed_context_false_belief_pressure",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_family_deltas(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            family_id = row.get("family_id")
            for delta_name, prompt_type in [
                ("delta_false_pressure", "evidence_false_belief_pressure"),
                ("delta_true_pressure", "evidence_true_belief_pressure"),
                ("delta_emotional_pressure", "evidence_emotional_pressure"),
                ("delta_distractor", "evidence_distractor_neutral"),
                ("delta_closed_context", "closed_context_false_belief_pressure"),
            ]:
                if delta_name in row and row[delta_name] != "":
                    try:
                        val = float(row[delta_name])
                    except ValueError:
                        continue
                    out[(family_id, prompt_type)] = {
                        "family_id": family_id,
                        "prompt_type": prompt_type,
                        "delta_margin": val,
                    }
    return out


def load_activation(path: Path) -> Mapping[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Expected dict activation at {path}, got {type(obj)}")


def group_rows_by_family(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("family_id"))].append(row)
    return dict(out)


def build_hidden_delta(
    neutral_activation_path: Path,
    comparison_activation_path: Path,
) -> np.ndarray:
    neutral = load_activation(neutral_activation_path)
    comp = load_activation(comparison_activation_path)
    h_neutral = np.asarray(neutral["hidden_states_final_token"], dtype=np.float32)
    h_comp = np.asarray(comp["hidden_states_final_token"], dtype=np.float32)
    if h_neutral.shape != h_comp.shape:
        raise ValueError(
            f"Shape mismatch: neutral {h_neutral.shape} vs comp {h_comp.shape} "
            f"({neutral_activation_path} vs {comparison_activation_path})"
        )
    return h_comp - h_neutral


def collect_dataset(
    jsonl_rows: Sequence[Mapping[str, Any]],
    family_deltas: Mapping[Tuple[str, str], Mapping[str, Any]],
    activation_root: Path,
) -> Tuple[
    Dict[Tuple[str, str], np.ndarray],
    Dict[Tuple[str, str], Dict[str, Any]],
    Tuple[int, int],
]:
    by_family_pt = group_rows_by_family(jsonl_rows)
    activation_lookup: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for family_id, family_rows in by_family_pt.items():
        for row in family_rows:
            prompt_type = str(row.get("prompt_type"))
            activation_path = Path(str(row.get("activation_path")))
            if not activation_path.is_absolute():
                activation_path = REPO_ROOT / activation_path
            activation_lookup[(family_id, prompt_type)] = {
                "activation_path": activation_path,
                "neutral_margin": None,
                "condition_margin": None,
            }

    neutral_type = "evidence_neutral"
    per_family_neutral: Dict[str, Path] = {}
    per_family_neutral_margin: Dict[str, float] = {}
    for family_id, family_rows in by_family_pt.items():
        for row in family_rows:
            if str(row.get("prompt_type")) == neutral_type:
                ap = Path(str(row.get("activation_path")))
                if not ap.is_absolute():
                    ap = REPO_ROOT / ap
                per_family_neutral[family_id] = ap
                try:
                    per_family_neutral_margin[family_id] = float(row["logit_margin"])
                except Exception:
                    pass
                break

    deltas: Dict[Tuple[str, str], np.ndarray] = {}
    metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
    shape: Tuple[int, int] | None = None

    for (family_id, prompt_type), info in activation_lookup.items():
        if prompt_type == neutral_type:
            continue
        if family_id not in per_family_neutral:
            continue
        delta = build_hidden_delta(
            per_family_neutral[family_id], info["activation_path"]
        )
        if shape is None:
            shape = (int(delta.shape[0]), int(delta.shape[1]))
        elif shape != (delta.shape[0], delta.shape[1]):
            raise ValueError(
                f"Unexpected delta shape {delta.shape} for {(family_id, prompt_type)}, expected {shape}"
            )
        key = (family_id, prompt_type)
        deltas[key] = delta

        neutral_margin = per_family_neutral_margin.get(family_id)
        delta_obj = family_deltas.get(key)
        condition_margin = None
        delta_margin = None
        if delta_obj is not None:
            delta_margin = delta_obj.get("delta_margin")
            if neutral_margin is not None and delta_margin is not None:
                condition_margin = neutral_margin + delta_margin
        if condition_margin is None:
            for row in by_family_pt[family_id]:
                if str(row.get("prompt_type")) == prompt_type:
                    try:
                        condition_margin = float(row["logit_margin"])
                        break
                    except Exception:
                        pass
        if delta_margin is None:
            if neutral_margin is not None and condition_margin is not None:
                delta_margin = condition_margin - neutral_margin

        metadata[key] = {
            "family_id": family_id,
            "condition": prompt_type,
            "neutral_margin": neutral_margin,
            "condition_margin": condition_margin,
            "delta_margin": delta_margin,
        }

    if shape is None:
        raise ValueError("No delta tensors loaded")
    return deltas, metadata, shape


def label_primary(delta_margin: float | None) -> int | None:
    if delta_margin is None:
        return None
    return 1 if delta_margin < 0 else 0


def label_strict(
    keys: Sequence[Tuple[str, str]],
    metadata: Mapping[Tuple[str, str], Mapping[str, Any]],
    condition: str,
) -> Tuple[Dict[Tuple[str, str], int | None], float, int, int]:
    condition_keys = [k for k in keys if k[1] == condition]
    deltas = []
    for k in condition_keys:
        meta = metadata[k]
        dm = meta.get("delta_margin")
        if dm is not None:
            deltas.append(dm)
    if not deltas:
        return {}, 0.0, 0, 0
    neg = [d for d in deltas if d < 0]
    pos = [d for d in deltas if d > 0]
    threshold = 0.0
    if neg:
        neg_sorted = sorted(neg)
        top_n = max(1, int(np.ceil(len(neg_sorted) / 3)))
        threshold = neg_sorted[top_n - 1]
    labels: Dict[Tuple[str, str], int | None] = {}
    harmful = 0
    nonharmful = 0
    for k in condition_keys:
        dm = metadata[k].get("delta_margin")
        if dm is None:
            labels[k] = None
            continue
        if dm > 0:
            labels[k] = 0
            nonharmful += 1
        elif dm <= threshold and dm < 0:
            labels[k] = 1
            harmful += 1
        else:
            labels[k] = None
    return labels, threshold, harmful, nonharmful


def run_pair(
    source_condition: str,
    target_condition: str,
    label_scheme: str,
    deltas: Mapping[Tuple[str, str], np.ndarray],
    metadata: Mapping[Tuple[str, str], Mapping[str, Any]],
    strict_thresholds: Mapping[str, float],
    strict_labels: Mapping[Tuple[str, str], int | None],
    layer_count: int,
    permute_rng: "np.random.Generator | None" = None,
    restrict_layers: "set[int] | None" = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_keys = list(deltas.keys())
    source_keys = [k for k in all_keys if k[1] == source_condition]
    target_keys = [k for k in all_keys if k[1] == target_condition]
    if not source_keys or not target_keys:
        return [], []

    def get_label(k: Tuple[str, str]) -> int | None:
        if label_scheme == "primary":
            return label_primary(metadata[k].get("delta_margin"))
        return strict_labels.get(k, None)

    train_keys = [k for k in source_keys if get_label(k) in (0, 1)]
    test_keys = [k for k in target_keys if get_label(k) in (0, 1)]
    if not train_keys or not test_keys:
        return [], []

    n_train = len(train_keys)
    n_test = len(test_keys)
    dim = int(deltas[train_keys[0]].shape[1])

    train_ys = np.asarray([int(get_label(k)) for k in train_keys], dtype=np.int64)
    if len(set(train_ys.tolist())) < 2:
        return [], []

    train_tensor = np.empty((n_train, layer_count, dim), dtype=np.float32)
    for i, k in enumerate(train_keys):
        train_tensor[i] = deltas[k].astype(np.float32, copy=False)

    test_tensor = np.empty((n_test, layer_count, dim), dtype=np.float32)
    test_ys = np.asarray([int(get_label(k)) for k in test_keys], dtype=np.int64)
    test_metas = [metadata[k] for k in test_keys]
    test_family_array = np.asarray([k[0] for k in test_keys], dtype=object)
    for i, k in enumerate(test_keys):
        test_tensor[i] = deltas[k].astype(np.float32, copy=False)

    test_family_ids = sorted({k[0] for k in test_keys})
    train_family_array = np.asarray([k[0] for k in train_keys], dtype=object)
    train_family_mask_per_fold: Dict[str, np.ndarray] = {}
    test_family_mask_per_fold: Dict[str, np.ndarray] = {}
    fold_labels: List[str] = []
    for fold_id, test_family in enumerate(test_family_ids):
        train_mask = train_family_array != test_family
        test_mask = test_family_array == test_family
        fold_labels.append(str(test_family))
        if train_mask.any():
            train_ys_fold = train_ys[train_mask]
            if len(set(train_ys_fold.tolist())) < 2:
                train_mask = np.zeros_like(train_mask, dtype=bool)
        train_family_mask_per_fold[str(fold_id)] = train_mask
        test_family_mask_per_fold[str(fold_id)] = test_mask

    baseline_ba = 0.5
    layerwise_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    layer_true: List[np.ndarray] = []
    layer_pred: List[np.ndarray] = []
    layer_prob: List[np.ndarray] = []
    layer_fold_ids: List[np.ndarray] = []
    layer_test_index: List[np.ndarray] = []

    for layer_index in range(layer_count):
        if restrict_layers is not None and layer_index not in restrict_layers:
            continue
        y_true_list = []
        y_pred_list = []
        y_prob_list = []
        fold_ids_list = []
        test_idx_list = []

        train_x_layer_all = train_tensor[:, layer_index, :]
        test_x_layer_all = test_tensor[:, layer_index, :]

        for fold_id, test_family in enumerate(test_family_ids):
            train_mask = train_family_mask_per_fold[str(fold_id)]
            test_mask = test_family_mask_per_fold[str(fold_id)]
            if not train_mask.any() or not test_mask.any():
                continue
            train_x = train_x_layer_all[train_mask]
            train_y = train_ys[train_mask]
            if len(set(train_y.tolist())) < 2:
                continue
            if permute_rng is not None:
                shuffled = train_y.copy()
                permute_rng.shuffle(shuffled)
                train_y = shuffled
            test_x = test_x_layer_all[test_mask]
            test_y = test_ys[test_mask]
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            model = LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=10000,
                C=1.0,
            )
            model.fit(train_x, train_y)
            prob = model.predict_proba(test_x)[:, 1]
            pred = (prob >= 0.5).astype(np.int64)
            y_true_list.append(test_y)
            y_pred_list.append(pred)
            y_prob_list.append(prob)
            fold_ids_list.append(np.asarray([fold_id] * len(test_y), dtype=np.int64))
            test_idx_list.append(np.where(test_mask)[0].astype(np.int64))

        if not y_true_list:
            continue
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        y_prob = np.concatenate(y_prob_list)
        layer_true.append(y_true)
        layer_pred.append(y_pred)
        layer_prob.append(y_prob)
        layer_fold_ids.append(np.concatenate(fold_ids_list))
        layer_test_index.append(np.concatenate(test_idx_list))
        n_examples = int(y_true.shape[0])
        unique_labels = set(y_true.tolist())
        ba = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auroc = float("nan")
        ap = float("nan")
        if len(unique_labels) == 2:
            try:
                auroc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                auroc = float("nan")
            try:
                ap = float(average_precision_score(y_true, y_prob))
            except Exception:
                ap = float("nan")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])
        n_families = len(set(np.concatenate(fold_ids_list).tolist()))
        layerwise_rows.append(
            {
                "label_scheme": label_scheme,
                "source_condition": source_condition,
                "target_condition": target_condition,
                "layer": layer_index,
                "n_examples": n_examples,
                "n_families": n_families,
                "balanced_accuracy": f"{ba:.6f}",
                "auroc": "" if np.isnan(auroc) else f"{auroc:.6f}",
                "average_precision": "" if np.isnan(ap) else f"{ap:.6f}",
                "f1": f"{f1:.6f}",
                "precision": f"{prec:.6f}",
                "recall": f"{rec:.6f}",
                "confusion_matrix_counts": f"tn={tn} fp={fp} fn={fn} tp={tp}",
                "baseline_balanced_accuracy": f"{baseline_ba:.6f}",
                "C_used": 1.0,
            }
        )

    if layer_true:
        for layer_rel, layer_index in enumerate(
            [int(r["layer"]) for r in layerwise_rows]
        ):
            y_true = layer_true[layer_rel]
            y_pred = layer_pred[layer_rel]
            y_prob = layer_prob[layer_rel]
            fold_ids = layer_fold_ids[layer_rel]
            test_idx = layer_test_index[layer_rel]
            for i in range(y_true.shape[0]):
                idx = int(test_idx[i])
                k = test_keys[idx]
                meta = test_metas[idx]
                prediction_rows.append(
                    {
                        "label_scheme": label_scheme,
                        "source_condition": source_condition,
                        "target_condition": target_condition,
                        "layer": layer_index,
                        "family_id": k[0],
                        "condition": k[1],
                        "true_label": int(y_true[i]),
                        "predicted_label": int(y_pred[i]),
                        "predicted_probability_harmful": float(y_prob[i]),
                        "delta_margin": meta.get("delta_margin"),
                        "neutral_margin": meta.get("neutral_margin"),
                        "condition_margin": meta.get("condition_margin"),
                        "fold_id": int(fold_ids[i]),
                        "C_used": 1.0,
                    }
                )

    return layerwise_rows, prediction_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_best_layers(
    layerwise_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in layerwise_rows:
        key = (
            str(row["label_scheme"]),
            str(row["source_condition"]),
            str(row["target_condition"]),
        )
        grouped[key].append(row)
    best_rows: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        def pick(field: str) -> Dict[str, Any]:
            def value(r: Mapping[str, Any]) -> float:
                v = r.get(field)
                if v in (None, "", "nan"):
                    return -1e18
                return float(v)
            return max(rows, key=value)
        ba = pick("balanced_accuracy")
        auroc_row = pick("auroc")
        ap_row = pick("average_precision")
        f1_row = pick("f1")
        best_rows.extend([
            {
                "label_scheme": key[0],
                "source_condition": key[1],
                "target_condition": key[2],
                "criterion": "balanced_accuracy",
                "best_layer": int(ba["layer"]),
                "best_value": ba["balanced_accuracy"],
            },
            {
                "label_scheme": key[0],
                "source_condition": key[1],
                "target_condition": key[2],
                "criterion": "auroc",
                "best_layer": int(auroc_row["layer"]),
                "best_value": auroc_row["auroc"],
            },
            {
                "label_scheme": key[0],
                "source_condition": key[1],
                "target_condition": key[2],
                "criterion": "average_precision",
                "best_layer": int(ap_row["layer"]),
                "best_value": ap_row["average_precision"],
            },
            {
                "label_scheme": key[0],
                "source_condition": key[1],
                "target_condition": key[2],
                "criterion": "f1",
                "best_layer": int(f1_row["layer"]),
                "best_value": f1_row["f1"],
            },
        ])
    return best_rows


def write_summary(
    path: Path,
    shape: Tuple[int, int],
    primary_counts: Dict[str, Tuple[int, int]],
    strict_counts: Dict[str, Tuple[int, int]],
    strict_thresholds: Dict[str, float],
    layerwise_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in layerwise_rows:
        key = (
            str(row["label_scheme"]),
            str(row["source_condition"]),
            str(row["target_condition"]),
        )
        grouped[key].append(dict(row))
    lines: List[str] = []
    lines.append("Probe 5 Cross-Condition Harmfulness Generalization Summary")
    lines.append("")
    lines.append(f"hidden_state_shape: ({shape[0]}, {shape[1]})")
    lines.append("split_strategy: leave-one-family-out within target, train excludes same-family source examples")
    lines.append("feature_definition: delta_to_family_neutral")
    lines.append("classifier: logistic_regression_l2_balanced_fixed_C")
    lines.append("")
    lines.append("Label distributions")
    for scheme in ("primary", "strict"):
        lines.append(f"  {scheme}:")
        for cond in CONDITIONS:
            if scheme == "primary":
                harmful, nonharmful = primary_counts[cond]
                thr = ""
            else:
                harmful, nonharmful = strict_counts[cond]
                thr = f"  threshold={strict_thresholds.get(cond, '')}"
            total = harmful + nonharmful
            hr = harmful / total if total else 0.0
            lines.append(f"    {cond}: n={total} harmful={harmful} nonharmful={nonharmful} harmful_rate={hr:.4f}{thr}")
    lines.append("")
    lines.append("Best layer per pair by balanced accuracy")
    for row in best_rows:
        if row["criterion"] != "balanced_accuracy":
            continue
        lines.append(
            f"  [{row['label_scheme']}] {row['source_condition']} -> {row['target_condition']}: "
            f"layer {row['best_layer']} balanced_accuracy={row['best_value']} (baseline=0.5)"
        )
    lines.append("")
    lines.append("Interpretation")
    cross_pairs_within = []
    cross_pairs_between = []
    for key, rows_list in grouped.items():
        scheme, src, tgt = key
        if scheme != "primary":
            continue
        ba_rows = [r for r in rows_list if r.get("balanced_accuracy") not in ("", None)]
        if not ba_rows:
            continue
        best = max(ba_rows, key=lambda r: float(r["balanced_accuracy"]))
        value = float(best["balanced_accuracy"])
        layer = int(best["layer"])
        if src == tgt:
            cross_pairs_within.append((src, tgt, layer, value))
        else:
            cross_pairs_between.append((src, tgt, layer, value))
    lines.append("  Within-condition baselines (source == target):")
    for src, tgt, layer, value in sorted(cross_pairs_within, key=lambda x: -x[3]):
        lines.append(f"    {src} -> {tgt}: best layer {layer}, balanced_accuracy={value:.4f}")
    lines.append("  Cross-condition transfer (source != target):")
    for src, tgt, layer, value in sorted(cross_pairs_between, key=lambda x: -x[3]):
        lines.append(f"    {src} -> {tgt}: best layer {layer}, balanced_accuracy={value:.4f}")
    lines.append("")
    lines.append("Key questions")
    if cross_pairs_within and cross_pairs_between:
        best_within = max(x[3] for x in cross_pairs_within)
        best_cross = max(x[3] for x in cross_pairs_between)
        if best_cross >= 0.7 and best_within - best_cross < 0.2:
            lines.append("  Harmfulness generalizes across pressure conditions: cross-condition performance is strong and close to within-condition baselines.")
        elif best_cross > 0.6:
            lines.append("  Harmfulness partially generalizes across pressure conditions: there is some cross-condition transfer, but it is weaker than within-condition performance.")
        else:
            lines.append("  Harmfulness does not reliably generalize across pressure conditions: representations look mostly condition-specific.")
    lines.append("")
    lines.append("Best layer ranges")
    layer_positions: Dict[str, List[int]] = defaultdict(list)
    for row in best_rows:
        if row["criterion"] != "balanced_accuracy":
            continue
        layer_positions[row["source_condition"] + "->" + row["target_condition"] + "[" + row["label_scheme"] + "]"].append(int(row["best_layer"]))
    early = 0
    mid = 0
    late = 0
    for layers in layer_positions.values():
        for l in layers:
            if l < shape[0] // 3:
                early += 1
            elif l < 2 * shape[0] // 3:
                mid += 1
            else:
                late += 1
    lines.append(f"  early layers (< {shape[0]//3}) best layer count: {early}")
    lines.append(f"  mid layers ({shape[0]//3}-{2*shape[0]//3}) best layer count: {mid}")
    lines.append(f"  late layers (>= {2*shape[0]//3}) best layer count: {late}")
    lines.append("")
    lines.append("Important caveat")
    lines.append("  Positive cross-condition transfer means the harmful/nonharmful representation is condition-agnostic to some degree.")
    lines.append("  It does not prove causal sycophancy mechanisms or intervention-ready handles.")
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    input_path = REPO_ROOT / DEFAULT_INPUT
    family_deltas_path = REPO_ROOT / DEFAULT_FAMILY_DELTAS
    activation_root = REPO_ROOT / DEFAULT_ACTIVATION_ROOT
    layerwise_path = REPO_ROOT / DEFAULT_LAYERWISE_OUTPUT
    predictions_path = REPO_ROOT / DEFAULT_PREDICTIONS_OUTPUT
    summary_path = REPO_ROOT / DEFAULT_SUMMARY_OUTPUT
    best_layers_path = REPO_ROOT / "results/probe5_cross_condition_best_layers.csv"

    jsonl_rows = read_jsonl(input_path)
    family_deltas = read_family_deltas(family_deltas_path)
    deltas, metadata, shape = collect_dataset(jsonl_rows, family_deltas, activation_root)

    layer_count = int(shape[0])
    all_keys = list(deltas.keys())

    primary_counts: Dict[str, Tuple[int, int]] = {}
    strict_counts: Dict[str, Tuple[int, int]] = {}
    strict_thresholds: Dict[str, float] = {}
    strict_labels_by_condition: Dict[str, Dict[Tuple[str, str], int | None]] = {}

    for cond in CONDITIONS:
        harmful = 0
        nonharmful = 0
        for (family_id, c) in all_keys:
            if c != cond:
                continue
            lbl = label_primary(metadata[(family_id, c)].get("delta_margin"))
            if lbl == 1:
                harmful += 1
            elif lbl == 0:
                nonharmful += 1
        primary_counts[cond] = (harmful, nonharmful)
        lbl_map, thr, h, n = label_strict(all_keys, metadata, cond)
        strict_labels_by_condition[cond] = lbl_map
        strict_thresholds[cond] = thr
        strict_counts[cond] = (h, n)

    start_message = {
        "status": "start",
        "n_layers": layer_count,
        "shape": [int(shape[0]), int(shape[1])],
        "primary_counts": {k: {"harmful": v[0], "nonharmful": v[1]} for k, v in primary_counts.items()},
        "strict_counts": {k: {"harmful": v[0], "nonharmful": v[1]} for k, v in strict_counts.items()},
        "strict_thresholds": strict_thresholds,
    }
    print(json.dumps(start_message, ensure_ascii=False))

    all_layerwise: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []

    pairs: List[Tuple[str, str]] = []
    for src in CONDITIONS:
        for tgt in CONDITIONS:
            pairs.append((src, tgt))

    for label_scheme in ("primary", "strict"):
        strict_labels_flat: Dict[Tuple[str, str], int | None] = {}
        if label_scheme == "strict":
            for cond, mapping in strict_labels_by_condition.items():
                for k, v in mapping.items():
                    strict_labels_flat[k] = v
        for src, tgt in pairs:
            print(
                json.dumps(
                    {
                        "label_scheme": label_scheme,
                        "source_condition": src,
                        "target_condition": tgt,
                        "status": "start_pair",
                    },
                    ensure_ascii=False,
                )
            )
            layerwise, predictions = run_pair(
                src,
                tgt,
                label_scheme,
                deltas,
                metadata,
                strict_thresholds,
                strict_labels_flat,
                layer_count,
            )
            all_layerwise.extend(layerwise)
            all_predictions.extend(predictions)

    write_csv(layerwise_path, all_layerwise)
    write_csv(predictions_path, all_predictions)
    best_rows = find_best_layers(all_layerwise)
    write_csv(best_layers_path, best_rows)
    write_summary(
        summary_path,
        shape,
        primary_counts,
        strict_counts,
        strict_thresholds,
        all_layerwise,
        best_rows,
    )

    print(
        json.dumps(
            {
                "status": "complete",
                "layerwise_output": str(layerwise_path),
                "predictions_output": str(predictions_path),
                "summary_output": str(summary_path),
                "best_layers_output": str(best_layers_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
