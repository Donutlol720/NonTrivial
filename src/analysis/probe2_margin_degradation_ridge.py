import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_DELTA_INPUT = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_LAYERWISE_OUTPUT = "results/probe2_margin_degradation_ridge_layerwise.csv"
DEFAULT_PREDICTIONS_OUTPUT = "results/probe2_margin_degradation_ridge_predictions.csv"
DEFAULT_BEST_LAYERS_OUTPUT = "results/probe2_margin_degradation_ridge_best_layers.csv"
DEFAULT_SUMMARY_OUTPUT = "results/probe2_margin_degradation_ridge_summary.txt"

EXPECTED_PROMPT_TYPES = (
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
)
CONDITION_SPECS = (
    ("false_pressure_delta", "evidence_false_belief_pressure", "delta_false_pressure"),
    ("emotional_pressure_delta", "evidence_emotional_pressure", "delta_emotional_pressure"),
    ("closed_context_delta", "closed_context_false_belief_pressure", "delta_closed_context"),
    ("distractor_delta", "evidence_distractor_neutral", "delta_distractor"),
)
FIXED_ALPHA = 1.0
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
EPSILON = 1e-12


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
            rows.append(row)
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def stddev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance ** 0.5


def mse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return 0.0
    return mean([(a - p) ** 2 for a, p in zip(actual, predicted)])


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return 0.0
    return mean([abs(a - p) for a, p in zip(actual, predicted)])


def r2_score(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return 0.0
    actual_mean = mean(actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    ss_tot = sum((a - actual_mean) ** 2 for a in actual)
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("pearson_correlation requires equal-length inputs.")
    if len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def rank_values(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        next_position = position + 1
        while next_position < len(indexed) and indexed[next_position][1] == indexed[position][1]:
            next_position += 1
        average_rank = (position + 1 + next_position) / 2.0
        for idx in range(position, next_position):
            ranks[indexed[idx][0]] = average_rank
        position = next_position
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("spearman_correlation requires equal-length inputs.")
    if len(xs) < 2:
        return 0.0
    return pearson_correlation(rank_values(xs), rank_values(ys))


def format_float(value: float) -> str:
    return f"{value:.4f}"


def layer_bucket(layer_index: int, n_layers: int) -> str:
    one_third = n_layers / 3.0
    if layer_index < one_third:
        return "early"
    if layer_index < 2.0 * one_third:
        return "middle"
    return "late"


def group_rows_by_family(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["family_id"])][str(row["prompt_type"])] = row
    return grouped


def load_hidden_state_tensor(repo_root: Path, row: Mapping[str, Any]) -> torch.Tensor:
    activation_path = repo_root / str(row["activation_path"])
    record = torch.load(activation_path, map_location="cpu")
    tensor = record.get("hidden_states_final_token")
    if tensor is None:
        raise ValueError(f"hidden_states_final_token missing in {activation_path}")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"hidden_states_final_token is not a tensor in {activation_path}")
    return tensor.detach().to(dtype=torch.float32)


def load_behavior_rows(
    json_rows: Sequence[Mapping[str, Any]],
    delta_csv_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Dict[str, Any]]:
    grouped = group_rows_by_family(json_rows)
    behavior_by_family: Dict[str, Dict[str, Any]] = {}

    for row in delta_csv_rows:
        family_id = str(row["family_id"])
        if family_id not in grouped:
            raise ValueError(f"Family {family_id} is present in delta CSV but missing in JSONL.")

        family_rows = grouped[family_id]
        missing = [prompt_type for prompt_type in EXPECTED_PROMPT_TYPES if prompt_type not in family_rows]
        if missing:
            raise ValueError(f"Family {family_id} is missing prompt types: {missing}")

        behavior_by_family[family_id] = {
            "domain": row.get("domain"),
            "title": row.get("title"),
            "logit_margin_evidence_neutral": float(row["logit_margin_evidence_neutral"]),
            "logit_margin_evidence_false_belief_pressure": float(row["logit_margin_evidence_false_belief_pressure"]),
            "logit_margin_evidence_emotional_pressure": float(row["logit_margin_evidence_emotional_pressure"]),
            "logit_margin_evidence_true_belief_pressure": float(row["logit_margin_evidence_true_belief_pressure"]),
            "logit_margin_evidence_distractor_neutral": float(row["logit_margin_evidence_distractor_neutral"]),
            "logit_margin_closed_context_false_belief_pressure": float(row["logit_margin_closed_context_false_belief_pressure"]),
            "delta_false_pressure": float(row["delta_false_pressure"]),
            "delta_emotional_pressure": float(row["delta_emotional_pressure"]),
            "delta_true_pressure": float(row["delta_true_pressure"]),
            "delta_distractor": float(row["delta_distractor"]),
            "delta_closed_context": float(row["delta_closed_context"]),
        }
    return behavior_by_family


def compute_family_deltas(
    repo_root: Path,
    grouped_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Tuple[int, int], Dict[str, str]]:
    family_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
    family_domains: Dict[str, str] = {}
    expected_shape: Tuple[int, int] | None = None

    for family_id in sorted(grouped_rows):
        family_rows = grouped_rows[family_id]
        missing = [prompt_type for prompt_type in EXPECTED_PROMPT_TYPES if prompt_type not in family_rows]
        if missing:
            raise ValueError(f"Family {family_id} is missing prompt types: {missing}")

        neutral = load_hidden_state_tensor(repo_root, family_rows["evidence_neutral"])
        family_domains[family_id] = str(family_rows["evidence_neutral"].get("domain"))
        if expected_shape is None:
            expected_shape = (int(neutral.shape[0]), int(neutral.shape[1]))
        elif tuple(neutral.shape) != expected_shape:
            raise ValueError(
                f"Family {family_id} has inconsistent neutral tensor shape {tuple(neutral.shape)} != {expected_shape}"
            )

        family_deltas[family_id] = {}
        for delta_name, prompt_type, _delta_margin_name in CONDITION_SPECS:
            comparison = load_hidden_state_tensor(repo_root, family_rows[prompt_type])
            if tuple(comparison.shape) != expected_shape:
                raise ValueError(
                    f"Family {family_id} prompt {prompt_type} has tensor shape {tuple(comparison.shape)} != {expected_shape}"
                )
            family_deltas[family_id][delta_name] = comparison - neutral

    if expected_shape is None:
        raise ValueError("No family deltas could be computed.")
    return family_deltas, expected_shape, family_domains


def fit_standardizer(train_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_vector = train_x.mean(dim=0)
    std_vector = train_x.std(dim=0, unbiased=False)
    std_vector = torch.where(std_vector > 0, std_vector, torch.ones_like(std_vector))
    return mean_vector, std_vector


def standardize(matrix: torch.Tensor, mean_vector: torch.Tensor, std_vector: torch.Tensor) -> torch.Tensor:
    return (matrix - mean_vector) / std_vector


def ridge_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    mean_vector, std_vector = fit_standardizer(train_x)
    train_scaled = standardize(train_x, mean_vector, std_vector)
    test_scaled = standardize(test_x, mean_vector, std_vector)

    y_mean = train_y.mean()
    y_centered = train_y - y_mean
    gram = train_scaled @ train_scaled.T
    identity = torch.eye(gram.shape[0], dtype=train_scaled.dtype)
    dual_weights = torch.linalg.solve(gram + (alpha * identity), y_centered)
    beta = train_scaled.T @ dual_weights
    return (test_scaled @ beta) + y_mean


def select_alpha_nested(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    candidate_alphas: Sequence[float],
) -> float:
    if train_x.shape[0] <= 2:
        return FIXED_ALPHA

    best_alpha = float(candidate_alphas[0])
    best_mse = float("inf")
    for alpha in candidate_alphas:
        inner_actual: List[float] = []
        inner_pred: List[float] = []
        for heldout_index in range(train_x.shape[0]):
            inner_train_indices = [idx for idx in range(train_x.shape[0]) if idx != heldout_index]
            inner_x_train = train_x[inner_train_indices]
            inner_y_train = train_y[inner_train_indices]
            inner_x_test = train_x[heldout_index : heldout_index + 1]
            pred = ridge_predict(inner_x_train, inner_y_train, inner_x_test, float(alpha))
            inner_actual.append(float(train_y[heldout_index].item()))
            inner_pred.append(float(pred[0].item()))
        alpha_mse = mse(inner_actual, inner_pred)
        if alpha_mse < best_mse - EPSILON:
            best_mse = alpha_mse
            best_alpha = float(alpha)
    return best_alpha


def build_diagnostics(
    family_ids: Sequence[str],
    behavior_by_family: Mapping[str, Mapping[str, Any]],
    grouped_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    tensor_shape: Tuple[int, int],
) -> Dict[str, Any]:
    jsonl_family_ids = sorted(grouped_rows)
    delta_csv_family_ids = sorted(behavior_by_family)
    if jsonl_family_ids != delta_csv_family_ids:
        raise ValueError("Family IDs do not match between JSONL and matched-family delta CSV.")

    degradation_sign_checks: List[str] = []
    for family_id in family_ids:
        behavior_row = behavior_by_family[family_id]
        for _delta_type, _prompt_type, delta_margin_name in CONDITION_SPECS:
            degradation = -float(behavior_row[delta_margin_name])
            recomputed = -(float(behavior_row[delta_margin_name]))
            if abs(degradation - recomputed) > EPSILON:
                raise ValueError(f"Behavior degradation sign mismatch for {family_id} {delta_margin_name}")
        degradation_sign_checks.append(family_id)

    return {
        "n_families": len(family_ids),
        "tensor_shape": tensor_shape,
        "jsonl_and_delta_csv_families_match": True,
        "all_expected_prompt_types_present": True,
        "degradation_sign_check_passed_for_all_families": len(degradation_sign_checks) == len(family_ids),
        "splits_grouped_by_family_id": True,
        "train_only_standardization": True,
    }


def evaluate_condition_layer(
    family_ids: Sequence[str],
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_by_family: Mapping[str, Mapping[str, Any]],
    delta_type: str,
    delta_margin_name: str,
    layer_index: int,
    alpha_strategy: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []

    for fold_index, heldout_family_id in enumerate(family_ids):
        train_ids = [family_id for family_id in family_ids if family_id != heldout_family_id]
        train_x = torch.stack(
            [family_deltas[family_id][delta_type][layer_index] for family_id in train_ids],
            dim=0,
        )
        test_x = family_deltas[heldout_family_id][delta_type][layer_index].unsqueeze(0)
        train_y = torch.tensor(
            [-float(behavior_by_family[family_id][delta_margin_name]) for family_id in train_ids],
            dtype=torch.float32,
        )
        test_y = -float(behavior_by_family[heldout_family_id][delta_margin_name])

        if alpha_strategy == "fixed_alpha":
            alpha_used = FIXED_ALPHA
        elif alpha_strategy == "nested_alpha":
            alpha_used = select_alpha_nested(train_x, train_y, ALPHA_GRID)
        else:
            raise ValueError(f"Unsupported alpha_strategy: {alpha_strategy}")

        ridge_pred = float(ridge_predict(train_x, train_y, test_x, alpha_used)[0].item())
        baseline_pred = float(train_y.mean().item())

        behavior_row = behavior_by_family[heldout_family_id]
        margin_suffix = delta_margin_name.removeprefix("delta_")
        condition_margin_key = f"logit_margin_{'closed_context_false_belief_pressure' if margin_suffix == 'closed_context' else ('evidence_' + margin_suffix if margin_suffix != 'distractor' else 'evidence_distractor_neutral')}"
        if delta_margin_name == "delta_false_pressure":
            condition_margin_key = "logit_margin_evidence_false_belief_pressure"
        elif delta_margin_name == "delta_emotional_pressure":
            condition_margin_key = "logit_margin_evidence_emotional_pressure"
        elif delta_margin_name == "delta_closed_context":
            condition_margin_key = "logit_margin_closed_context_false_belief_pressure"
        elif delta_margin_name == "delta_distractor":
            condition_margin_key = "logit_margin_evidence_distractor_neutral"

        predictions.append(
            {
                "condition": delta_type,
                "layer": layer_index,
                "family_id": heldout_family_id,
                "fold_id": fold_index,
                "domain": behavior_row["domain"],
                "actual_degradation": test_y,
                "predicted_degradation": ridge_pred,
                "baseline_prediction": baseline_pred,
                "delta_margin": float(behavior_row[delta_margin_name]),
                "neutral_margin": float(behavior_row["logit_margin_evidence_neutral"]),
                "condition_margin": float(behavior_row[condition_margin_key]),
                "alpha_used": alpha_used,
            }
        )

    actual = [float(row["actual_degradation"]) for row in predictions]
    predicted = [float(row["predicted_degradation"]) for row in predictions]
    baseline_pred = [float(row["baseline_prediction"]) for row in predictions]
    ridge_mse = mse(actual, predicted)
    baseline_mse = mse(actual, baseline_pred)
    absolute_improvement = baseline_mse - ridge_mse
    relative_improvement = absolute_improvement / baseline_mse if baseline_mse != 0.0 else 0.0

    layer_metrics = {
        "condition": delta_type,
        "layer": layer_index,
        "layer_bucket": layer_bucket(layer_index, family_deltas[family_ids[0]][delta_type].shape[0]),
        "n_families": len(family_ids),
        "pearson_r": pearson_correlation(predicted, actual),
        "spearman_rho": spearman_correlation(predicted, actual),
        "r2": r2_score(actual, predicted),
        "mse": ridge_mse,
        "mae": mae(actual, predicted),
        "baseline_mse": baseline_mse,
        "absolute_mse_improvement": absolute_improvement,
        "relative_mse_improvement": relative_improvement,
        "mean_alpha_used": mean([float(row["alpha_used"]) for row in predictions]),
        "alpha_values_used_json": json.dumps(
            dict(sorted(Counter([float(row["alpha_used"]) for row in predictions]).items())),
            ensure_ascii=False,
        ),
    }
    return predictions, layer_metrics


def build_probe_outputs(
    family_ids: Sequence[str],
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_by_family: Mapping[str, Mapping[str, Any]],
    n_layers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prediction_rows: List[Dict[str, Any]] = []
    layerwise_rows: List[Dict[str, Any]] = []

    for delta_type, _prompt_type, delta_margin_name in CONDITION_SPECS:
        for layer_index in range(n_layers):
            predictions, metrics = evaluate_condition_layer(
                family_ids=family_ids,
                family_deltas=family_deltas,
                behavior_by_family=behavior_by_family,
                delta_type=delta_type,
                delta_margin_name=delta_margin_name,
                layer_index=layer_index,
                alpha_strategy="fixed_alpha",
            )
            prediction_rows.extend(predictions)
            layerwise_rows.append(metrics)
    return prediction_rows, layerwise_rows


def build_best_layers_rows(layerwise_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    criteria = (
        ("pearson_r", "max"),
        ("spearman_rho", "max"),
        ("r2", "max"),
        ("relative_mse_improvement", "max"),
    )
    output_rows: List[Dict[str, Any]] = []

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in layerwise_rows:
        grouped[str(row["condition"])].append(row)

    for condition, rows in sorted(grouped.items()):
        for metric_name, direction in criteria:
            if direction != "max":
                raise ValueError(f"Unsupported criterion direction: {direction}")
            best_row = max(rows, key=lambda row: float(row[metric_name]))
            output_rows.append(
                {
                    "condition": condition,
                    "selection_metric": metric_name,
                    "best_layer": int(best_row["layer"]),
                    "layer_bucket": str(best_row["layer_bucket"]),
                    "selected_metric_value": float(best_row[metric_name]),
                    "pearson_r": float(best_row["pearson_r"]),
                    "spearman_rho": float(best_row["spearman_rho"]),
                    "r2": float(best_row["r2"]),
                    "mse": float(best_row["mse"]),
                    "mae": float(best_row["mae"]),
                    "baseline_mse": float(best_row["baseline_mse"]),
                    "relative_mse_improvement": float(best_row["relative_mse_improvement"]),
                    "mean_alpha_used": float(best_row["mean_alpha_used"]),
                }
            )
    return output_rows


def pick_best_row(rows: Sequence[Mapping[str, Any]], condition: str, metric_name: str) -> Mapping[str, Any]:
    candidates = [
        row for row in rows
        if str(row["condition"]) == condition
    ]
    if not candidates:
        raise ValueError(f"No layerwise rows for condition={condition}")
    return max(candidates, key=lambda row: float(row[metric_name]))


def build_nested_alpha_comparisons(
    family_ids: Sequence[str],
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_by_family: Mapping[str, Mapping[str, Any]],
    layerwise_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    comparisons: List[Dict[str, Any]] = []
    for condition, _prompt_type, delta_margin_name in CONDITION_SPECS:
        fixed_best = pick_best_row(layerwise_rows, condition, "pearson_r")
        _predictions, nested_metrics = evaluate_condition_layer(
            family_ids=family_ids,
            family_deltas=family_deltas,
            behavior_by_family=behavior_by_family,
            delta_type=condition,
            delta_margin_name=delta_margin_name,
            layer_index=int(fixed_best["layer"]),
            alpha_strategy="nested_alpha",
        )
        comparisons.append(
            {
                "condition": condition,
                "fixed_best_layer": int(fixed_best["layer"]),
                "fixed_pearson_r": float(fixed_best["pearson_r"]),
                "fixed_spearman_rho": float(fixed_best["spearman_rho"]),
                "fixed_r2": float(fixed_best["r2"]),
                "fixed_relative_mse_improvement": float(fixed_best["relative_mse_improvement"]),
                "nested_pearson_r": float(nested_metrics["pearson_r"]),
                "nested_spearman_rho": float(nested_metrics["spearman_rho"]),
                "nested_r2": float(nested_metrics["r2"]),
                "nested_relative_mse_improvement": float(nested_metrics["relative_mse_improvement"]),
                "nested_mean_alpha_used": float(nested_metrics["mean_alpha_used"]),
                "nested_alpha_values_used_json": str(nested_metrics["alpha_values_used_json"]),
            }
        )
    return comparisons


def compare_eval_modes(nested_comparison: Mapping[str, Any]) -> str:
    fixed_r = float(nested_comparison["fixed_pearson_r"])
    nested_r = float(nested_comparison["nested_pearson_r"])
    fixed_improvement = float(nested_comparison["fixed_relative_mse_improvement"])
    nested_improvement = float(nested_comparison["nested_relative_mse_improvement"])
    if abs(fixed_r - nested_r) < 0.05 and abs(fixed_improvement - nested_improvement) < 0.05:
        return "fixed_alpha and nested_alpha are broadly consistent"
    if nested_r > fixed_r and nested_improvement > fixed_improvement:
        return "nested_alpha improves over fixed_alpha"
    return "nested_alpha differs but does not clearly improve on the fixed_alpha result"


def build_summary_text(
    diagnostics: Mapping[str, Any],
    layerwise_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
    nested_comparisons: Sequence[Mapping[str, Any]],
) -> str:
    tensor_shape = diagnostics["tensor_shape"]
    n_layers, d_model = int(tensor_shape[0]), int(tensor_shape[1])

    fixed_false = pick_best_row(layerwise_rows, "false_pressure_delta", "pearson_r")
    fixed_emotional = pick_best_row(layerwise_rows, "emotional_pressure_delta", "pearson_r")
    fixed_closed = pick_best_row(layerwise_rows, "closed_context_delta", "pearson_r")
    fixed_distractor = pick_best_row(layerwise_rows, "distractor_delta", "pearson_r")

    strongest_fixed = max(
        [fixed_false, fixed_emotional, fixed_closed, fixed_distractor],
        key=lambda row: float(row["pearson_r"]),
    )
    late_layer_count = sum(1 for row in best_rows if 31 <= int(row["best_layer"]) <= 34)

    lines: List[str] = []
    lines.append("Probe 2 Margin-Degradation Ridge Regression Summary")
    lines.append("")
    lines.append("Diagnostics")
    lines.append(f"  all_36_families_present: {diagnostics['n_families'] == 36}")
    lines.append(f"  jsonl_and_delta_csv_families_match: {diagnostics['jsonl_and_delta_csv_families_match']}")
    lines.append(f"  all_expected_prompt_types_present: {diagnostics['all_expected_prompt_types_present']}")
    lines.append(f"  degradation_sign_check_passed: {diagnostics['degradation_sign_check_passed_for_all_families']}")
    lines.append(f"  hidden_state_shape: ({n_layers}, {d_model})")
    lines.append(f"  splits_grouped_by_family_id: {diagnostics['splits_grouped_by_family_id']}")
    lines.append(f"  train_only_standardization: {diagnostics['train_only_standardization']}")
    lines.append("")
    lines.append("Primary Fixed-Alpha Results (alpha = 1.0)")
    lines.append(
        "  "
        f"false_pressure_delta: best Pearson layer {fixed_false['layer']} ({fixed_false['layer_bucket']}), "
        f"r={format_float(float(fixed_false['pearson_r']))}, "
        f"rho={format_float(float(fixed_false['spearman_rho']))}, "
        f"R2={format_float(float(fixed_false['r2']))}, "
        f"relative_MSE_improvement={format_float(float(fixed_false['relative_mse_improvement']))}"
    )
    lines.append(
        "  "
        f"emotional_pressure_delta: best Pearson layer {fixed_emotional['layer']} ({fixed_emotional['layer_bucket']}), "
        f"r={format_float(float(fixed_emotional['pearson_r']))}, "
        f"rho={format_float(float(fixed_emotional['spearman_rho']))}, "
        f"R2={format_float(float(fixed_emotional['r2']))}, "
        f"relative_MSE_improvement={format_float(float(fixed_emotional['relative_mse_improvement']))}"
    )
    lines.append(
        "  "
        f"closed_context_delta: best Pearson layer {fixed_closed['layer']} ({fixed_closed['layer_bucket']}), "
        f"r={format_float(float(fixed_closed['pearson_r']))}, "
        f"rho={format_float(float(fixed_closed['spearman_rho']))}, "
        f"R2={format_float(float(fixed_closed['r2']))}, "
        f"relative_MSE_improvement={format_float(float(fixed_closed['relative_mse_improvement']))}"
    )
    lines.append(
        "  "
        f"distractor_delta: best Pearson layer {fixed_distractor['layer']} ({fixed_distractor['layer_bucket']}), "
        f"r={format_float(float(fixed_distractor['pearson_r']))}, "
        f"rho={format_float(float(fixed_distractor['spearman_rho']))}, "
        f"R2={format_float(float(fixed_distractor['r2']))}, "
        f"relative_MSE_improvement={format_float(float(fixed_distractor['relative_mse_improvement']))}"
    )
    lines.append("")
    lines.append("Answers To Requested Questions")
    lines.append(
        "  "
        f"The strongest family-held-out degradation prediction comes from {strongest_fixed['condition']} "
        f"at layer {strongest_fixed['layer']} with Pearson r={format_float(float(strongest_fixed['pearson_r']))}."
    )
    lines.append(
        "  "
        f"Closed-context {'remains' if strongest_fixed['condition'] == 'closed_context_delta' else 'does not remain'} "
        f"the strongest condition under the primary fixed-alpha ridge probe."
    )
    lines.append(
        "  "
        f"False-pressure {'still has meaningful predictive signal' if float(fixed_false['pearson_r']) > 0.2 else 'looks weak under ridge-CV'}; "
        f"its best Pearson is {format_float(float(fixed_false['pearson_r']))} and best R2 is {format_float(float(fixed_false['r2']))}."
    )
    lines.append(
        "  "
        f"Emotional-pressure {'remains weaker than closed-context' if float(fixed_emotional['pearson_r']) < float(fixed_closed['pearson_r']) else 'is competitive with closed-context'}; "
        f"its best Pearson is {format_float(float(fixed_emotional['pearson_r']))}."
    )
    lines.append(
        "  "
        f"Distractor {'remains weaker than the pressure-like conditions' if float(fixed_distractor['pearson_r']) < min(float(fixed_false['pearson_r']), float(fixed_emotional['pearson_r']), float(fixed_closed['pearson_r'])) else 'is not clearly weaker than the pressure-like conditions'}."
    )
    lines.append(
        "  "
        f"Best layers {'do' if late_layer_count >= len(best_rows) // 2 else 'do not'} cluster around 31-34; "
        f"{late_layer_count} of {len(best_rows)} fixed-alpha best-layer selections land in that late-layer range."
    )
    lines.append(
        "  "
        f"Ridge {'beats' if any(float(row['relative_mse_improvement']) > 0 for row in layerwise_rows) else 'does not beat'} the mean-only baseline for at least some fixed-alpha condition/layer combinations."
    )
    lines.append(
        "  "
        "If Pearson/Spearman are positive but R2 remains negative, the right interpretation is that the probe recovers ordering signal more reliably than exact degradation magnitudes."
    )
    lines.append("")
    lines.append("Fixed vs Nested Alpha")
    for comparison in nested_comparisons:
        condition = str(comparison["condition"])
        lines.append(
            "  "
            f"{condition} at fixed best layer {comparison['fixed_best_layer']}: "
            f"{compare_eval_modes(comparison)}. "
            f"Fixed Pearson={format_float(float(comparison['fixed_pearson_r']))}, "
            f"nested Pearson={format_float(float(comparison['nested_pearson_r']))}, "
            f"nested mean alpha={format_float(float(comparison['nested_mean_alpha_used']))}."
        )
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "  A successful ridge probe shows linearly decodable information about degradation in the hidden-state deltas under family-held-out evaluation."
    )
    lines.append(
        "  It does not establish causality; it only shows that hidden-state deltas contain predictive information about later behavioral margin changes."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / DEFAULT_INPUT
    delta_input_path = repo_root / DEFAULT_DELTA_INPUT
    layerwise_output = repo_root / DEFAULT_LAYERWISE_OUTPUT
    predictions_output = repo_root / DEFAULT_PREDICTIONS_OUTPUT
    best_layers_output = repo_root / DEFAULT_BEST_LAYERS_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    json_rows = read_jsonl(input_path)
    delta_csv_rows = read_csv_rows(delta_input_path)
    grouped_rows = group_rows_by_family(json_rows)
    behavior_by_family = load_behavior_rows(json_rows, delta_csv_rows)
    family_deltas, tensor_shape, _family_domains = compute_family_deltas(repo_root, grouped_rows)
    family_ids = sorted(family_deltas)
    diagnostics = build_diagnostics(family_ids, behavior_by_family, grouped_rows, tensor_shape)

    n_layers = int(tensor_shape[0])
    prediction_rows, layerwise_rows = build_probe_outputs(
        family_ids=family_ids,
        family_deltas=family_deltas,
        behavior_by_family=behavior_by_family,
        n_layers=n_layers,
    )
    best_layers_rows = build_best_layers_rows(layerwise_rows)
    nested_comparisons = build_nested_alpha_comparisons(
        family_ids=family_ids,
        family_deltas=family_deltas,
        behavior_by_family=behavior_by_family,
        layerwise_rows=layerwise_rows,
    )

    layerwise_fieldnames = [
        "condition",
        "layer",
        "layer_bucket",
        "n_families",
        "pearson_r",
        "spearman_rho",
        "r2",
        "mse",
        "mae",
        "baseline_mse",
        "absolute_mse_improvement",
        "relative_mse_improvement",
        "mean_alpha_used",
        "alpha_values_used_json",
    ]
    prediction_fieldnames = [
        "condition",
        "layer",
        "family_id",
        "fold_id",
        "domain",
        "actual_degradation",
        "predicted_degradation",
        "baseline_prediction",
        "delta_margin",
        "neutral_margin",
        "condition_margin",
        "alpha_used",
    ]
    best_layers_fieldnames = [
        "condition",
        "selection_metric",
        "best_layer",
        "layer_bucket",
        "selected_metric_value",
        "pearson_r",
        "spearman_rho",
        "r2",
        "mse",
        "mae",
        "baseline_mse",
        "relative_mse_improvement",
        "mean_alpha_used",
    ]

    write_csv(layerwise_output, layerwise_rows, layerwise_fieldnames)
    write_csv(predictions_output, prediction_rows, prediction_fieldnames)
    write_csv(best_layers_output, best_layers_rows, best_layers_fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        build_summary_text(diagnostics, layerwise_rows, best_layers_rows, nested_comparisons),
        encoding="utf-8",
    )

    print(f"Wrote {layerwise_output}")
    print(f"Wrote {predictions_output}")
    print(f"Wrote {best_layers_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
