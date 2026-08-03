import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_DELTA_INPUT = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_LAYERWISE_OUTPUT = "results/probe4b_within_condition_harmful_pressure_layerwise.csv"
DEFAULT_PREDICTIONS_OUTPUT = "results/probe4b_within_condition_harmful_pressure_predictions.csv"
DEFAULT_SUMMARY_OUTPUT = "results/probe4b_within_condition_harmful_pressure_summary.txt"

TARGET_CONDITIONS = (
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "closed_context_false_belief_pressure",
)
EXPECTED_PROMPT_TYPES = ("evidence_neutral",) + TARGET_CONDITIONS + ("evidence_true_belief_pressure",)

PROMPT_TYPE_TO_DELTA_COLUMN = {
    "evidence_false_belief_pressure": "delta_false_pressure",
    "evidence_emotional_pressure": "delta_emotional_pressure",
    "closed_context_false_belief_pressure": "delta_closed_context",
    "evidence_true_belief_pressure": "delta_true_pressure",
}

FIXED_C = 1.0

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'penalty' was deprecated.*",
)
sys.stdout.reconfigure(line_buffering=True)


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


def validate_delta_csv(json_rows: Sequence[Mapping[str, Any]], delta_csv_rows: Sequence[Mapping[str, str]]) -> None:
    jsonl_families = sorted({str(row["family_id"]) for row in json_rows})
    csv_families = sorted({str(row["family_id"]) for row in delta_csv_rows})
    if jsonl_families != csv_families:
        raise ValueError("Family IDs do not match between JSONL and matched-family delta CSV.")


def load_hidden_state_tensor(repo_root: Path, row: Mapping[str, Any]) -> torch.Tensor:
    activation_path = repo_root / str(row["activation_path"])
    record = torch.load(activation_path, map_location="cpu")
    tensor = record.get("hidden_states_final_token")
    if tensor is None:
        raise ValueError(f"hidden_states_final_token missing in {activation_path}")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"hidden_states_final_token is not a tensor in {activation_path}")
    return tensor.detach().to(dtype=torch.float32)


def build_behavior_by_family(delta_csv_rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, float]]:
    behavior: Dict[str, Dict[str, float]] = {}
    for row in delta_csv_rows:
        family_id = str(row["family_id"])
        behavior[family_id] = {
            "delta_false_pressure": float(row["delta_false_pressure"]),
            "delta_emotional_pressure": float(row["delta_emotional_pressure"]),
            "delta_closed_context": float(row["delta_closed_context"]),
            "delta_true_pressure": float(row["delta_true_pressure"]),
        }
    return behavior


def compute_primary_label(delta_margin: float) -> int:
    return 1 if delta_margin < 0.0 else 0


def compute_strict_threshold(delta_margins: Sequence[float]) -> float:
    ordered = sorted(delta_margins)
    cutoff_index = max(0, int(len(ordered) / 3) - 1)
    return float(ordered[cutoff_index])


def compute_strict_label(delta_margin: float, threshold: float) -> int | None:
    if delta_margin > 0.0:
        return 0
    if delta_margin <= threshold:
        return 1
    return None


def safe_auroc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    if len(set(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, probabilities))


def safe_average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    if len(set(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, probabilities))


def confusion_counts(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def majority_baseline(train_labels: Sequence[int], n_test: int) -> List[int]:
    counts = Counter(train_labels)
    if not counts:
        return [0] * n_test
    majority = max(sorted(counts), key=lambda label: counts[label])
    return [int(majority)] * n_test


def build_base_examples(
    repo_root: Path,
    grouped_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    behavior_by_family: Mapping[str, Mapping[str, float]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Tuple[int, int]]:
    expected_shape: Tuple[int, int] | None = None
    by_condition: Dict[str, List[Dict[str, Any]]] = {condition: [] for condition in TARGET_CONDITIONS}

    for family_id in sorted(grouped_rows):
        family_rows = grouped_rows[family_id]
        missing = [prompt_type for prompt_type in EXPECTED_PROMPT_TYPES if prompt_type not in family_rows]
        if missing:
            raise ValueError(f"Family {family_id} is missing prompt types: {missing}")
        if family_id not in behavior_by_family:
            raise ValueError(f"Family {family_id} missing from delta CSV.")

        neutral = load_hidden_state_tensor(repo_root, family_rows["evidence_neutral"])
        if expected_shape is None:
            expected_shape = (int(neutral.shape[0]), int(neutral.shape[1]))
        elif tuple(neutral.shape) != expected_shape:
            raise ValueError(
                f"Family {family_id} has inconsistent neutral tensor shape {tuple(neutral.shape)} != {expected_shape}"
            )

        for condition in TARGET_CONDITIONS:
            column = PROMPT_TYPE_TO_DELTA_COLUMN[condition]
            delta_margin = float(behavior_by_family[family_id][column])
            comparison = load_hidden_state_tensor(repo_root, family_rows[condition])
            if tuple(comparison.shape) != expected_shape:
                raise ValueError(
                    f"Family {family_id} prompt {condition} has tensor shape {tuple(comparison.shape)} != {expected_shape}"
                )
            delta_np = (comparison - neutral).numpy()
            by_condition[condition].append(
                {
                    "family_id": family_id,
                    "domain": str(family_rows[condition].get("domain")),
                    "condition": condition,
                    "delta_margin": delta_margin,
                    "delta_np": delta_np,
                }
            )

    if expected_shape is None:
        raise ValueError("No examples built for Probe 4B.")
    return by_condition, expected_shape


def apply_label_scheme(
    base_examples: Sequence[Mapping[str, Any]],
    label_scheme: str,
    strict_threshold: float,
) -> List[Dict[str, Any]]:
    labeled: List[Dict[str, Any]] = []
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
            raise ValueError(f"Unknown label scheme: {label_scheme}")
        labeled.append(
            {
                "label_scheme": label_scheme,
                "family_id": str(example["family_id"]),
                "domain": str(example["domain"]),
                "condition": str(example["condition"]),
                "delta_margin": delta_margin,
                "label": int(label),
                "delta_np": example["delta_np"],
            }
        )
    return labeled


def evaluate_layer(
    examples: Sequence[Mapping[str, Any]],
    label_scheme: str,
    condition: str,
    n_layers: int,
    layer_index: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    family_ids = sorted({str(example["family_id"]) for example in examples})
    prediction_rows: List[Dict[str, Any]] = []

    for fold_index, heldout_family_id in enumerate(family_ids):
        train_examples = [example for example in examples if str(example["family_id"]) != heldout_family_id]
        test_examples = [example for example in examples if str(example["family_id"]) == heldout_family_id]
        if not train_examples or not test_examples:
            continue

        train_x = np.stack([example["delta_np"][layer_index] for example in train_examples], axis=0)
        test_x = np.stack([example["delta_np"][layer_index] for example in test_examples], axis=0)
        train_y = [int(example["label"]) for example in train_examples]
        test_y = [int(example["label"]) for example in test_examples]

        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        test_x_scaled = scaler.transform(test_x)

        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            C=FIXED_C,
            solver="liblinear",
        )
        classifier.fit(train_x_scaled, train_y)
        probabilities = classifier.predict_proba(test_x_scaled)[:, 1]
        predicted_labels = [1 if probability >= 0.5 else 0 for probability in probabilities]
        baseline_labels = majority_baseline(train_y, len(test_y))

        for example, true_label, predicted_label, probability, baseline_label in zip(
            test_examples,
            test_y,
            predicted_labels,
            probabilities,
            baseline_labels,
        ):
            prediction_rows.append(
                {
                    "label_scheme": label_scheme,
                    "condition": condition,
                    "family_id": str(example["family_id"]),
                    "layer": layer_index,
                    "layer_bucket": layer_bucket(layer_index, n_layers),
                    "delta_margin": float(example["delta_margin"]),
                    "true_label": int(true_label),
                    "predicted_label": int(predicted_label),
                    "predicted_probability_harmful": float(probability),
                    "baseline_predicted_label": int(baseline_label),
                    "fold_id": fold_index,
                }
            )

    pooled_true = [int(row["true_label"]) for row in prediction_rows]
    pooled_pred = [int(row["predicted_label"]) for row in prediction_rows]
    pooled_prob = [float(row["predicted_probability_harmful"]) for row in prediction_rows]
    pooled_baseline = [int(row["baseline_predicted_label"]) for row in prediction_rows]
    harmful_count = sum(pooled_true)
    nonharmful_count = len(pooled_true) - harmful_count
    counts = confusion_counts(pooled_true, pooled_pred) if pooled_true else {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

    return prediction_rows, {
        "label_scheme": label_scheme,
        "condition": condition,
        "layer": layer_index,
        "layer_bucket": layer_bucket(layer_index, n_layers),
        "n_examples": len(prediction_rows),
        "n_families": len(family_ids),
        "n_harmful": harmful_count,
        "n_nonharmful": nonharmful_count,
        "balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_pred)) if pooled_true else 0.0,
        "auroc": safe_auroc(pooled_true, pooled_prob) if pooled_true else 0.0,
        "average_precision": safe_average_precision(pooled_true, pooled_prob) if pooled_true else 0.0,
        "f1": float(f1_score(pooled_true, pooled_pred, zero_division=0)) if pooled_true else 0.0,
        "precision": float(precision_score(pooled_true, pooled_pred, zero_division=0)) if pooled_true else 0.0,
        "recall": float(recall_score(pooled_true, pooled_pred, zero_division=0)) if pooled_true else 0.0,
        "confusion_matrix_counts": json.dumps(counts, ensure_ascii=False),
        "baseline_balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_baseline)) if pooled_true else 0.0,
        "C_used": FIXED_C,
    }


def build_outputs(
    by_condition_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    label_scheme: str,
    n_layers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prediction_rows: List[Dict[str, Any]] = []
    layerwise_rows: List[Dict[str, Any]] = []
    for condition in TARGET_CONDITIONS:
        examples = by_condition_examples[condition]
        print(json.dumps({
            "label_scheme": label_scheme,
            "condition": condition,
            "n_examples": len(examples),
            "status": "start_condition",
        }, ensure_ascii=False), flush=True)
        for layer_index in range(n_layers):
            if layer_index % 6 == 0:
                print(json.dumps({
                    "label_scheme": label_scheme,
                    "condition": condition,
                    "layer": layer_index,
                    "status": "running",
                }, ensure_ascii=False), flush=True)
            layer_predictions, layer_metrics = evaluate_layer(examples, label_scheme, condition, n_layers, layer_index)
            prediction_rows.extend(layer_predictions)
            layerwise_rows.append(layer_metrics)
    return prediction_rows, layerwise_rows


def best_row(rows: Sequence[Mapping[str, Any]], condition: str, metric_name: str) -> Mapping[str, Any]:
    candidates = [row for row in rows if str(row["condition"]) == condition]
    return max(candidates, key=lambda row: float(row[metric_name]))


def harmful_rate(examples: Sequence[Mapping[str, Any]]) -> float:
    if not examples:
        return 0.0
    return mean([int(example["label"]) for example in examples])


def build_summary_text(
    tensor_shape: Tuple[int, int],
    thresholds: Mapping[str, float],
    primary_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    strict_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    primary_layerwise: Sequence[Mapping[str, Any]],
    strict_layerwise: Sequence[Mapping[str, Any]],
) -> str:
    n_layers, d_model = tensor_shape
    lines: List[str] = []
    lines.append("Probe 4B Within-Condition Harmful-Pressure Classification Summary")
    lines.append("")
    lines.append(f"hidden_state_shape: ({n_layers}, {d_model})")
    lines.append("split_strategy: leave-one-family-out")
    lines.append("feature_definition: delta_to_family_neutral")
    lines.append("classifier: logistic_regression_l2_balanced_fixed_C")
    lines.append("")

    for condition in TARGET_CONDITIONS:
        primary_examples = primary_by_condition[condition]
        strict_examples = strict_by_condition[condition]
        primary_best = best_row(primary_layerwise, condition, "balanced_accuracy")
        primary_best_auroc = best_row(primary_layerwise, condition, "auroc")
        strict_best = best_row(strict_layerwise, condition, "balanced_accuracy") if strict_examples else None
        lines.append(condition)
        lines.append(f"  primary_examples: {len(primary_examples)}")
        lines.append(f"  primary_harmful_rate: {format_float(harmful_rate(primary_examples))}")
        lines.append(
            f"  primary_best_balanced_accuracy: layer {primary_best['layer']} ({primary_best['layer_bucket']}), "
            f"value={format_float(float(primary_best['balanced_accuracy']))}, baseline={format_float(float(primary_best['baseline_balanced_accuracy']))}"
        )
        lines.append(
            f"  primary_best_auroc: layer {primary_best_auroc['layer']} ({primary_best_auroc['layer_bucket']}), "
            f"value={format_float(float(primary_best_auroc['auroc']))}"
        )
        lines.append(f"  strict_threshold: {format_float(float(thresholds[condition]))}")
        lines.append(f"  strict_examples: {len(strict_examples)}")
        if strict_examples:
            strict_best_auroc = best_row(strict_layerwise, condition, "auroc")
            lines.append(f"  strict_harmful_rate: {format_float(harmful_rate(strict_examples))}")
            lines.append(
                f"  strict_best_balanced_accuracy: layer {strict_best['layer']} ({strict_best['layer_bucket']}), "
                f"value={format_float(float(strict_best['balanced_accuracy']))}, baseline={format_float(float(strict_best['baseline_balanced_accuracy']))}"
            )
            lines.append(
                f"  strict_best_auroc: layer {strict_best_auroc['layer']} ({strict_best_auroc['layer_bucket']}), "
                f"value={format_float(float(strict_best_auroc['auroc']))}"
            )
        lines.append("")

    false_best = best_row(primary_layerwise, "evidence_false_belief_pressure", "balanced_accuracy")
    emotional_best = best_row(primary_layerwise, "evidence_emotional_pressure", "balanced_accuracy")
    closed_best = best_row(primary_layerwise, "closed_context_false_belief_pressure", "balanced_accuracy")
    lines.append("Answers To Requested Questions")
    lines.append(
        f"  Harmfulness is {'decodable' if float(false_best['balanced_accuracy']) > 0.65 else 'not strongly decodable'} within false_pressure alone."
    )
    lines.append(
        f"  Harmfulness is {'decodable' if float(emotional_best['balanced_accuracy']) > 0.65 else 'not strongly decodable'} within emotional_pressure alone."
    )
    lines.append(
        f"  Harmfulness is {'decodable' if float(closed_best['balanced_accuracy']) > 0.65 else 'not strongly decodable'} within closed_context alone."
    )
    late_count = sum(
        1 for row in (false_best, emotional_best, closed_best) if str(row["layer_bucket"]) == "late"
    )
    lines.append(
        f"  Performance is {'still' if late_count >= 2 else 'not clearly'} late-layer dominated across the within-condition classifiers."
    )
    lines.append(
        "  If the within-condition classifiers are also strong, then Probe 4 was not just a cross-condition identity effect."
    )
    lines.append(
        "  The false_pressure result is the key test of retrieved-evidence pressure degradation specifically."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / DEFAULT_INPUT
    delta_path = repo_root / DEFAULT_DELTA_INPUT
    layerwise_output = repo_root / DEFAULT_LAYERWISE_OUTPUT
    predictions_output = repo_root / DEFAULT_PREDICTIONS_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    json_rows = read_jsonl(input_path)
    delta_csv_rows = read_csv_rows(delta_path)
    validate_delta_csv(json_rows, delta_csv_rows)
    grouped_rows = group_rows_by_family(json_rows)
    behavior_by_family = build_behavior_by_family(delta_csv_rows)
    base_by_condition, tensor_shape = build_base_examples(repo_root, grouped_rows, behavior_by_family)

    strict_thresholds = {
        condition: compute_strict_threshold([float(example["delta_margin"]) for example in base_by_condition[condition]])
        for condition in TARGET_CONDITIONS
    }

    primary_by_condition = {
        condition: apply_label_scheme(base_by_condition[condition], "primary", strict_thresholds[condition])
        for condition in TARGET_CONDITIONS
    }
    strict_by_condition = {
        condition: apply_label_scheme(base_by_condition[condition], "strict", strict_thresholds[condition])
        for condition in TARGET_CONDITIONS
    }

    n_layers = int(tensor_shape[0])
    print(json.dumps({
        "status": "start",
        "n_layers": n_layers,
        "primary_counts": {condition: len(primary_by_condition[condition]) for condition in TARGET_CONDITIONS},
        "strict_counts": {condition: len(strict_by_condition[condition]) for condition in TARGET_CONDITIONS},
        "strict_thresholds": strict_thresholds,
    }, ensure_ascii=False), flush=True)

    primary_predictions, primary_layerwise = build_outputs(primary_by_condition, "primary", n_layers)
    strict_predictions, strict_layerwise = build_outputs(strict_by_condition, "strict", n_layers)

    all_predictions = primary_predictions + strict_predictions
    all_layerwise = primary_layerwise + strict_layerwise

    write_csv(
        layerwise_output,
        all_layerwise,
        [
            "label_scheme",
            "condition",
            "layer",
            "layer_bucket",
            "n_examples",
            "n_families",
            "n_harmful",
            "n_nonharmful",
            "balanced_accuracy",
            "auroc",
            "average_precision",
            "f1",
            "precision",
            "recall",
            "confusion_matrix_counts",
            "baseline_balanced_accuracy",
            "C_used",
        ],
    )
    write_csv(
        predictions_output,
        all_predictions,
        [
            "label_scheme",
            "condition",
            "family_id",
            "layer",
            "layer_bucket",
            "delta_margin",
            "true_label",
            "predicted_label",
            "predicted_probability_harmful",
            "baseline_predicted_label",
            "fold_id",
        ],
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        build_summary_text(
            tensor_shape,
            strict_thresholds,
            primary_by_condition,
            strict_by_condition,
            primary_layerwise,
            strict_layerwise,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "layerwise_output": str(layerwise_output),
        "predictions_output": str(predictions_output),
        "summary_output": str(summary_output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
