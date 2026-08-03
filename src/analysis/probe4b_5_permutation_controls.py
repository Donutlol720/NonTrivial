import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_DELTA_INPUT = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_ACTIVATION_ROOT = "activations/qwen3_4b_instruct_2507"

PROBE4B_CONDITIONS = (
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "closed_context_false_belief_pressure",
)
PROBE5_CONDITIONS = PROBE4B_CONDITIONS

PROMPT_TYPE_TO_DELTA_COLUMN = {
    "evidence_false_belief_pressure": "delta_false_pressure",
    "evidence_emotional_pressure": "delta_emotional_pressure",
    "closed_context_false_belief_pressure": "delta_closed_context",
    "evidence_true_belief_pressure": "delta_true_pressure",
    "evidence_distractor_neutral": "delta_distractor",
}

FIXED_C = 1.0

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'penalty' was deprecated.*",
)
sys.stdout.reconfigure(line_buffering=True)
_rng = np.random.default_rng(20260728)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def build_behavior_by_family(delta_csv_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for row in delta_csv_rows:
        family_id = str(row["family_id"])
        for prompt_type, column in PROMPT_TYPE_TO_DELTA_COLUMN.items():
            if column not in row or row[column] in (None, ""):
                continue
            try:
                out[(family_id, prompt_type)] = float(row[column])
            except ValueError:
                pass
    return out


def group_rows_by_family(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        family_id = str(row["family_id"])
        prompt_type = str(row["prompt_type"])
        grouped[family_id][prompt_type] = row
    return dict(grouped)


def load_tensor(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        return np.asarray(obj["hidden_states_final_token"], dtype=np.float32)
    raise ValueError(f"Unexpected activation payload: {type(obj)} for {path}")


def resolve_activation_path(repo_root: Path, row: Mapping[str, Any]) -> Path:
    ap = Path(str(row.get("activation_path")))
    if ap.is_absolute():
        return ap
    return repo_root / ap


def build_condition_dataset(
    repo_root: Path,
    grouped_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    behavior_by_family: Mapping[Tuple[str, str], float],
    activation_root: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Tuple[int, int]]:
    expected_shape: Tuple[int, int] | None = None
    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for family_id, family_rows in grouped_rows.items():
        if "evidence_neutral" not in family_rows:
            continue
        neutral_path = resolve_activation_path(repo_root, family_rows["evidence_neutral"])
        neutral_tensor = load_tensor(neutral_path)
        for condition in PROBE4B_CONDITIONS:
            if condition not in family_rows:
                continue
            key = (family_id, condition)
            if key not in behavior_by_family:
                continue
            delta_margin = behavior_by_family[key]
            condition_row = family_rows[condition]
            condition_path = resolve_activation_path(repo_root, condition_row)
            condition_tensor = load_tensor(condition_path)
            if expected_shape is None:
                expected_shape = (
                    int(condition_tensor.shape[0]),
                    int(condition_tensor.shape[1]),
                )
            if condition_tensor.shape != neutral_tensor.shape:
                raise ValueError(f"Shape mismatch for {family_id}/{condition}")
            delta_np = condition_tensor - neutral_tensor
            by_condition[condition].append(
                {
                    "family_id": family_id,
                    "condition": condition,
                    "delta_margin": delta_margin,
                    "delta_np": delta_np,
                }
            )
    if expected_shape is None:
        raise ValueError("No condition datasets could be built.")
    return dict(by_condition), expected_shape


def compute_primary_label(delta_margin: float) -> int:
    return 1 if delta_margin < 0 else 0


def compute_strict_threshold(delta_margins: Sequence[float]) -> float:
    negatives = sorted([d for d in delta_margins if d < 0])
    if not negatives:
        return 0.0
    top_n = max(1, int(np.ceil(len(negatives) / 3)))
    return negatives[top_n - 1]


def compute_strict_label(delta_margin: float, threshold: float) -> int | None:
    if delta_margin > 0:
        return 0
    if delta_margin <= threshold and delta_margin < 0:
        return 1
    return None


def apply_labels(
    base_examples: Sequence[Mapping[str, Any]],
    label_scheme: str,
    strict_threshold: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for example in base_examples:
        delta_margin = float(example["delta_margin"])
        if label_scheme == "primary":
            label = compute_primary_label(delta_margin)
        elif label_scheme == "strict":
            label_or_none = compute_strict_label(delta_margin, strict_threshold)
            if label_or_none is None:
                continue
            label = label_or_none
        else:
            raise ValueError(label_scheme)
        out.append(
            {
                "family_id": str(example["family_id"]),
                "condition": str(example["condition"]),
                "label": int(label),
                "delta_margin": delta_margin,
                "delta_np": example["delta_np"],
            }
        )
    return out


def prepare_cache(
    examples: Sequence[Mapping[str, Any]],
    n_layers: int,
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    n = len(examples)
    family_ids = sorted({str(e["family_id"]) for e in examples})
    family_to_index = {fid: idx for idx, fid in enumerate(family_ids)}
    per_layer = {
        layer_index: np.stack(
            [example["delta_np"][layer_index] for example in examples],
            axis=0,
        ).astype(np.float32)
        for layer_index in range(n_layers)
    }
    labels = np.asarray([int(e["label"]) for e in examples], dtype=np.int64)
    family_vec = np.asarray(
        [family_to_index[str(e["family_id"])] for e in examples],
        dtype=np.int64,
    )
    return {
        "n": int(n),
        "n_families": len(family_ids),
        "family_ids": list(family_ids),
        "per_layer": per_layer,
        "labels": labels,
        "family_vec": family_vec,
    }, family_vec


def evaluate_loo_layer_from_cache(
    cache: Mapping[str, Any],
    layer_index: int,
    rng: np.random.Generator | None = None,
) -> float:
    n_families = int(cache["n_families"])
    x = cache["per_layer"][layer_index]
    y = cache["labels"]
    family_vec = cache["family_vec"]
    pooled_true: List[int] = []
    pooled_pred: List[int] = []
    for fam_id in range(n_families):
        train_mask = family_vec != fam_id
        test_mask = family_vec == fam_id
        if not train_mask.any() or not test_mask.any():
            continue
        train_x = x[train_mask]
        test_x = x[test_mask]
        train_y = y[train_mask]
        test_y = y[test_mask]
        if len(set(train_y.tolist())) < 2:
            continue
        if rng is not None:
            perm_indices = np.arange(train_y.shape[0])
            rng.shuffle(perm_indices)
            train_y_use = train_y[perm_indices]
        else:
            train_y_use = train_y
        scaler = StandardScaler()
        train_x_s = scaler.fit_transform(train_x)
        test_x_s = scaler.transform(test_x)
        model = LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=10000,
            C=FIXED_C,
        )
        model.fit(train_x_s, train_y_use)
        pred = model.predict(test_x_s)
        pooled_true.extend(test_y.tolist())
        pooled_pred.extend(pred.tolist())
    if not pooled_true:
        return 0.0
    return float(balanced_accuracy_score(np.asarray(pooled_true), np.asarray(pooled_pred)))


def best_ba_over_layers(
    examples: Sequence[Mapping[str, Any]],
    n_layers: int,
    rng: np.random.Generator | None = None,
    cache: Mapping[str, Any] | None = None,
) -> Tuple[float, int]:
    if cache is None:
        cache, _ = prepare_cache(examples, n_layers)
    best_value = -1.0
    best_layer = -1
    for layer_index in range(n_layers):
        ba = evaluate_loo_layer_from_cache(cache, layer_index, rng=rng)
        if ba > best_value:
            best_value = ba
            best_layer = layer_index
    return float(best_value), int(best_layer)


def real_best_ba(examples, n_layers):
    return best_ba_over_layers(examples, n_layers, rng=None)


def permute_best_ba_at_layer(examples, n_layers, layer_index, rng, cache=None):
    if cache is None:
        cache, _ = prepare_cache(examples, n_layers)
    ba = evaluate_loo_layer_from_cache(cache, layer_index, rng=rng)
    return float(ba)


def evaluate_pair_layer_from_cache(
    source_cache: Mapping[str, Any],
    target_cache: Mapping[str, Any],
    layer_index: int,
    rng: np.random.Generator | None = None,
) -> float:
    src_x = source_cache["per_layer"][layer_index]
    src_y = source_cache["labels"]
    src_family = source_cache["family_vec"]
    tgt_x = target_cache["per_layer"][layer_index]
    tgt_y = target_cache["labels"]
    tgt_family = target_cache["family_vec"]
    n_target_families = int(target_cache["n_families"])
    pooled_true: List[int] = []
    pooled_pred: List[int] = []
    for fam_id in range(n_target_families):
        train_mask = src_family != fam_id
        test_mask = tgt_family == fam_id
        if not train_mask.any() or not test_mask.any():
            continue
        train_x = src_x[train_mask]
        test_x = tgt_x[test_mask]
        train_y = src_y[train_mask]
        test_y = tgt_y[test_mask]
        if len(set(train_y.tolist())) < 2:
            continue
        if rng is not None:
            perm_indices = np.arange(train_y.shape[0])
            rng.shuffle(perm_indices)
            train_y = train_y[perm_indices]
        scaler = StandardScaler()
        train_x_s = scaler.fit_transform(train_x)
        test_x_s = scaler.transform(test_x)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            C=FIXED_C,
            solver="liblinear",
        )
        model.fit(train_x_s, train_y)
        pred = model.predict(test_x_s)
        pooled_true.extend(test_y.tolist())
        pooled_pred.extend(pred.tolist())
    if not pooled_true:
        return 0.0
    return float(balanced_accuracy_score(np.asarray(pooled_true), np.asarray(pooled_pred)))


def best_ba_pair_over_layers(
    source_examples: Sequence[Mapping[str, Any]],
    target_examples: Sequence[Mapping[str, Any]],
    n_layers: int,
    rng: np.random.Generator | None = None,
    source_cache: Mapping[str, Any] | None = None,
    target_cache: Mapping[str, Any] | None = None,
) -> Tuple[float, int]:
    if source_cache is None:
        source_cache, _ = prepare_cache(source_examples, n_layers)
    if target_cache is None:
        target_cache, _ = prepare_cache(target_examples, n_layers)
    best_value = -1.0
    best_layer = -1
    for layer_index in range(n_layers):
        ba = evaluate_pair_layer_from_cache(
            source_cache, target_cache, layer_index, rng=rng
        )
        if ba > best_value:
            best_value = ba
            best_layer = layer_index
    return float(best_value), int(best_layer)


def permute_best_pair_ba_at_layer(
    source_examples, target_examples, n_layers, layer_index, rng,
    source_cache=None, target_cache=None,
):
    if source_cache is None:
        source_cache, _ = prepare_cache(source_examples, n_layers)
    if target_cache is None:
        target_cache, _ = prepare_cache(target_examples, n_layers)
    ba = evaluate_pair_layer_from_cache(
        source_cache, target_cache, layer_index, rng=rng
    )
    return float(ba)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--probe", choices=["4b", "5", "all"], default="all")
    parser.add_argument("--label-scheme", choices=["primary", "strict", "all"], default="all")
    parser.add_argument(
        "--output",
        default="results/probe4b_5_permutation_controls.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/probe4b_5_permutation_controls_summary.txt",
    )
    args = parser.parse_args()

    input_path = REPO_ROOT / DEFAULT_INPUT
    delta_path = REPO_ROOT / DEFAULT_DELTA_INPUT
    activation_root = REPO_ROOT / DEFAULT_ACTIVATION_ROOT
    output_path = REPO_ROOT / args.output
    summary_path = REPO_ROOT / args.summary_output

    json_rows = read_jsonl(input_path)
    delta_rows = read_csv_rows(delta_path)
    behavior_by_family = build_behavior_by_family(delta_rows)
    grouped_rows = group_rows_by_family(json_rows)
    base_by_condition, shape = build_condition_dataset(
        REPO_ROOT, grouped_rows, behavior_by_family, activation_root
    )
    n_layers = int(shape[0])
    repeats = int(args.repeats)

    label_schemes: Tuple[str, ...] = (
        ("primary", "strict") if args.label_scheme == "all" else (args.label_scheme,)
    )
    probes: Tuple[str, ...] = (
        ("4b", "5") if args.probe == "all" else (args.probe,)
    )

    result_rows: List[Dict[str, Any]] = []
    for label_scheme in label_schemes:
        strict_thresholds = {
            condition: compute_strict_threshold(
                [float(e["delta_margin"]) for e in base_by_condition[condition]]
            )
            for condition in PROBE4B_CONDITIONS
        }
        labeled: Dict[str, List[Dict[str, Any]]] = {}
        for condition in PROBE4B_CONDITIONS:
            labeled[condition] = apply_labels(
                base_by_condition[condition],
                label_scheme,
                strict_thresholds[condition],
            )

        if "4b" in probes:
            print(json.dumps({
                "status": "start_probe4b",
                "label_scheme": label_scheme,
                "repeats": repeats,
            }, ensure_ascii=False), flush=True)
            for condition in PROBE4B_CONDITIONS:
                examples = labeled[condition]
                if len({int(e["label"]) for e in examples}) < 2:
                    continue
                cache, _ = prepare_cache(examples, n_layers)
                real_best, best_layer = best_ba_over_layers(
                    examples, n_layers, rng=None, cache=cache
                )
                perm_values: List[float] = []
                base_seed = _rng.integers(0, 2**60, endpoint=False)
                for r in range(repeats):
                    rng = np.random.default_rng(int(base_seed) + r)
                    perm_ba = permute_best_ba_at_layer(
                        examples, n_layers, best_layer, rng=rng, cache=cache
                    )
                    perm_values.append(perm_ba)
                    if (r + 1) % 5 == 0:
                        print(json.dumps({
                            "status": "progress_probe4b",
                            "condition": condition,
                            "label_scheme": label_scheme,
                            "repeat": r + 1,
                            "total_repeats": repeats,
                            "best_layer": int(best_layer),
                        }, ensure_ascii=False), flush=True)
                mean_perm = float(np.mean(perm_values)) if perm_values else 0.0
                p95 = percentile(perm_values, 95.0)
                p_count = sum(1 for v in perm_values if v >= real_best)
                emp_p = (p_count + 1) / (len(perm_values) + 1) if perm_values else 1.0
                result_rows.append(
                    {
                        "probe": "4B",
                        "label_scheme": label_scheme,
                        "scope": "within_condition",
                        "source_condition": condition,
                        "target_condition": condition,
                        "best_layer": int(best_layer),
                        "real_best_balanced_accuracy": f"{real_best:.6f}",
                        "permute_repeats": repeats,
                        "permute_mean_balanced_accuracy_at_best_layer": f"{mean_perm:.6f}",
                        "permute_p95_balanced_accuracy_at_best_layer": f"{p95:.6f}",
                        "empirical_p_value": f"{emp_p:.6f}",
                    }
                )

        if "5" in probes:
            print(json.dumps({
                "status": "start_probe5",
                "label_scheme": label_scheme,
                "repeats": repeats,
            }, ensure_ascii=False), flush=True)
            pairs: List[Tuple[str, str]] = []
            for src in PROBE5_CONDITIONS:
                for tgt in PROBE5_CONDITIONS:
                    pairs.append((src, tgt))
            for src, tgt in pairs:
                src_examples = labeled[src]
                tgt_examples = labeled[tgt]
                if not src_examples or not tgt_examples:
                    continue
                if (
                    len({int(e["label"]) for e in src_examples}) < 2
                    or len({int(e["label"]) for e in tgt_examples}) < 2
                ):
                    continue
                src_cache, _ = prepare_cache(src_examples, n_layers)
                tgt_cache, _ = prepare_cache(tgt_examples, n_layers)
                real_best, best_layer = best_ba_pair_over_layers(
                    src_examples, tgt_examples, n_layers, rng=None,
                    source_cache=src_cache, target_cache=tgt_cache,
                )
                perm_values: List[float] = []
                pair_base_seed = _rng.integers(0, 2**60, endpoint=False)
                for r in range(repeats):
                    rng = np.random.default_rng(int(pair_base_seed) + r)
                    perm_ba = permute_best_pair_ba_at_layer(
                        src_examples, tgt_examples, n_layers, best_layer, rng=rng,
                        source_cache=src_cache, target_cache=tgt_cache,
                    )
                    perm_values.append(perm_ba)
                    if (r + 1) % 5 == 0:
                        print(json.dumps({
                            "status": "progress_probe5",
                            "source_condition": src,
                            "target_condition": tgt,
                            "label_scheme": label_scheme,
                            "repeat": r + 1,
                            "total_repeats": repeats,
                            "best_layer": int(best_layer),
                        }, ensure_ascii=False), flush=True)
                mean_perm = float(np.mean(perm_values)) if perm_values else 0.0
                p95 = percentile(perm_values, 95.0)
                p_count = sum(1 for v in perm_values if v >= real_best)
                emp_p = (p_count + 1) / (len(perm_values) + 1) if perm_values else 1.0
                result_rows.append(
                    {
                        "probe": "5",
                        "label_scheme": label_scheme,
                        "scope": "cross_condition" if src != tgt else "within_condition",
                        "source_condition": src,
                        "target_condition": tgt,
                        "best_layer": int(best_layer),
                        "real_best_balanced_accuracy": f"{real_best:.6f}",
                        "permute_repeats": repeats,
                        "permute_mean_balanced_accuracy_at_best_layer": f"{mean_perm:.6f}",
                        "permute_p95_balanced_accuracy_at_best_layer": f"{p95:.6f}",
                        "empirical_p_value": f"{emp_p:.6f}",
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result_rows:
        fieldnames = list(result_rows[0].keys())
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in result_rows:
                writer.writerow(row)
    else:
        output_path.write_text("", encoding="utf-8")

    summary_lines: List[str] = []
    summary_lines.append("Probe 4B / Probe 5 permutation-label controls")
    summary_lines.append("")
    summary_lines.append(f"repeats: {repeats}")
    summary_lines.append(f"hidden_state_shape: ({shape[0]}, {shape[1]})")
    summary_lines.append("")
    for row in result_rows:
        summary_lines.append(
            f"[{row['probe']}:{row['scope']}] "
            f"{row['label_scheme']} {row['source_condition']} -> {row['target_condition']}:"
        )
        summary_lines.append(
            f"  best_layer         = {row['best_layer']}"
        )
        summary_lines.append(
            f"  real BA (at best)  = {row['real_best_balanced_accuracy']}"
        )
        summary_lines.append(
            f"  permute mean BA    = {row['permute_mean_balanced_accuracy_at_best_layer']}"
        )
        summary_lines.append(
            f"  permute p95 BA     = {row['permute_p95_balanced_accuracy_at_best_layer']}"
        )
        summary_lines.append(
            f"  empirical p        = {row['empirical_p_value']}"
        )
        summary_lines.append("")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "complete",
        "output": str(output_path),
        "summary": str(summary_path),
        "n_rows": len(result_rows),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
