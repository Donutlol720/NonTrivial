import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
DEFAULT_LAYERWISE_OUTPUT = "results/probe3_pressure_vs_distractor_layerwise.csv"
DEFAULT_PREDICTIONS_OUTPUT = "results/probe3_pressure_vs_distractor_predictions.csv"
DEFAULT_SUMMARY_OUTPUT = "results/probe3_pressure_vs_distractor_summary.txt"

EXPECTED_PROMPT_TYPES = (
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
)
CLASSIFICATION_SPECS = (
    ("evidence_false_belief_pressure", 1),
    ("evidence_true_belief_pressure", 1),
    ("evidence_emotional_pressure", 1),
    ("closed_context_false_belief_pressure", 1),
    ("evidence_distractor_neutral", 0),
)
FIXED_C = 1.0


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


def load_hidden_state_tensor(repo_root: Path, row: Mapping[str, Any]) -> torch.Tensor:
    activation_path = repo_root / str(row["activation_path"])
    record = torch.load(activation_path, map_location="cpu")
    tensor = record.get("hidden_states_final_token")
    if tensor is None:
        raise ValueError(f"hidden_states_final_token missing in {activation_path}")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"hidden_states_final_token is not a tensor in {activation_path}")
    return tensor.detach().to(dtype=torch.float32)


def validate_family_delta_csv(json_rows: Sequence[Mapping[str, Any]], delta_csv_rows: Sequence[Mapping[str, str]]) -> None:
    jsonl_families = sorted({str(row["family_id"]) for row in json_rows})
    csv_families = sorted({str(row["family_id"]) for row in delta_csv_rows})
    if jsonl_families != csv_families:
        raise ValueError("Family IDs do not match between the canonical JSONL and the family-delta CSV.")


def build_examples(
    repo_root: Path,
    grouped_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    examples: List[Dict[str, Any]] = []
    expected_shape: Tuple[int, int] | None = None

    for family_id in sorted(grouped_rows):
        family_rows = grouped_rows[family_id]
        missing = [prompt_type for prompt_type in EXPECTED_PROMPT_TYPES if prompt_type not in family_rows]
        if missing:
            raise ValueError(f"Family {family_id} is missing prompt types: {missing}")

        neutral = load_hidden_state_tensor(repo_root, family_rows["evidence_neutral"])
        if expected_shape is None:
            expected_shape = (int(neutral.shape[0]), int(neutral.shape[1]))
        elif tuple(neutral.shape) != expected_shape:
            raise ValueError(
                f"Family {family_id} has inconsistent neutral tensor shape {tuple(neutral.shape)} != {expected_shape}"
            )

        for prompt_type, label in CLASSIFICATION_SPECS:
            comparison = load_hidden_state_tensor(repo_root, family_rows[prompt_type])
            if tuple(comparison.shape) != expected_shape:
                raise ValueError(
                    f"Family {family_id} prompt {prompt_type} has tensor shape {tuple(comparison.shape)} != {expected_shape}"
                )
            delta = comparison - neutral
            examples.append(
                {
                    "family_id": family_id,
                    "domain": str(family_rows[prompt_type].get("domain")),
                    "condition": prompt_type,
                    "label": int(label),
                    "delta": delta,
                }
            )

    if expected_shape is None:
        raise ValueError("No examples were built for Probe 3.")
    return examples, expected_shape


def build_majority_baseline_predictions(train_labels: Sequence[int], n_test: int) -> List[int]:
    counts = Counter(train_labels)
    if not counts:
        return [1] * n_test
    majority_label = max(sorted(counts), key=lambda label: counts[label])
    return [int(majority_label)] * n_test


def safe_auroc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return 0.0
    return float(roc_auc_score(labels, probabilities))


def safe_average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return 0.0
    return float(average_precision_score(labels, probabilities))


def confusion_counts(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_layer(
    examples: Sequence[Mapping[str, Any]],
    n_layers: int,
    layer_index: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    family_ids = sorted({str(example["family_id"]) for example in examples})
    prediction_rows: List[Dict[str, Any]] = []

    for fold_index, heldout_family_id in enumerate(family_ids):
        train_examples = [example for example in examples if str(example["family_id"]) != heldout_family_id]
        test_examples = [example for example in examples if str(example["family_id"]) == heldout_family_id]

        train_x = torch.stack([example["delta"][layer_index] for example in train_examples], dim=0).numpy()
        test_x = torch.stack([example["delta"][layer_index] for example in test_examples], dim=0).numpy()
        train_y = [int(example["label"]) for example in train_examples]
        test_y = [int(example["label"]) for example in test_examples]

        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        test_x_scaled = scaler.transform(test_x)

        classifier = LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=10000,
            C=FIXED_C,
        )
        classifier.fit(train_x_scaled, train_y)
        probabilities = classifier.predict_proba(test_x_scaled)[:, 1]
        predicted_labels = [1 if probability >= 0.5 else 0 for probability in probabilities]
        baseline_predictions = build_majority_baseline_predictions(train_y, len(test_y))

        for example, true_label, predicted_label, probability, baseline_label in zip(
            test_examples,
            test_y,
            predicted_labels,
            probabilities,
            baseline_predictions,
        ):
            prediction_rows.append(
                {
                    "family_id": str(example["family_id"]),
                    "condition": str(example["condition"]),
                    "layer": layer_index,
                    "layer_bucket": layer_bucket(layer_index, n_layers),
                    "true_label": int(true_label),
                    "predicted_label": int(predicted_label),
                    "predicted_probability_pressure": float(probability),
                    "baseline_predicted_label": int(baseline_label),
                    "fold_id": fold_index,
                }
            )

    pooled_true = [int(row["true_label"]) for row in prediction_rows]
    pooled_pred = [int(row["predicted_label"]) for row in prediction_rows]
    pooled_prob = [float(row["predicted_probability_pressure"]) for row in prediction_rows]
    pooled_baseline = [int(row["baseline_predicted_label"]) for row in prediction_rows]
    counts = confusion_counts(pooled_true, pooled_pred)

    layer_metrics = {
        "layer": layer_index,
        "layer_bucket": layer_bucket(layer_index, n_layers),
        "n_examples": len(prediction_rows),
        "n_families": len(family_ids),
        "balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_pred)),
        "auroc": safe_auroc(pooled_true, pooled_prob),
        "average_precision": safe_average_precision(pooled_true, pooled_prob),
        "f1": float(f1_score(pooled_true, pooled_pred, zero_division=0)),
        "precision": float(precision_score(pooled_true, pooled_pred, zero_division=0)),
        "recall": float(recall_score(pooled_true, pooled_pred, zero_division=0)),
        "confusion_matrix_counts": json.dumps(counts, ensure_ascii=False),
        "baseline_balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_baseline)),
        "C_used": FIXED_C,
    }
    return prediction_rows, layer_metrics


def build_outputs(
    examples: Sequence[Mapping[str, Any]],
    n_layers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prediction_rows: List[Dict[str, Any]] = []
    layerwise_rows: List[Dict[str, Any]] = []

    for layer_index in range(n_layers):
        layer_predictions, layer_metrics = evaluate_layer(examples, n_layers, layer_index)
        prediction_rows.extend(layer_predictions)
        layerwise_rows.append(layer_metrics)
    return prediction_rows, layerwise_rows


def best_row(rows: Sequence[Mapping[str, Any]], metric_name: str) -> Mapping[str, Any]:
    return max(rows, key=lambda row: float(row[metric_name]))


def build_condition_error_summary(
    prediction_rows: Sequence[Mapping[str, Any]],
    best_layer: int,
) -> Tuple[List[str], List[str]]:
    rows = [row for row in prediction_rows if int(row["layer"]) == best_layer]
    pressure_errors: List[str] = []
    distractor_errors: List[str] = []
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    for condition, condition_rows in sorted(grouped.items()):
        mistakes = sum(1 for row in condition_rows if int(row["true_label"]) != int(row["predicted_label"]))
        total = len(condition_rows)
        if int(condition_rows[0]["true_label"]) == 1:
            pressure_errors.append(f"{condition}: {mistakes}/{total} mistaken for distractor")
        else:
            distractor_errors.append(f"{condition}: {mistakes}/{total} mistaken for pressure")
    return pressure_errors, distractor_errors


def build_summary_text(
    tensor_shape: Tuple[int, int],
    examples: Sequence[Mapping[str, Any]],
    layerwise_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> str:
    n_layers, d_model = tensor_shape
    best_balanced = best_row(layerwise_rows, "balanced_accuracy")
    best_auroc = best_row(layerwise_rows, "auroc")
    best_ap = best_row(layerwise_rows, "average_precision")
    best_f1 = best_row(layerwise_rows, "f1")
    pressure_errors, distractor_errors = build_condition_error_summary(
        prediction_rows=prediction_rows,
        best_layer=int(best_balanced["layer"]),
    )

    strong_family_held_out = (
        float(best_balanced["balanced_accuracy"]) >= 0.8
        and float(best_auroc["auroc"]) >= 0.85
    )

    lines: List[str] = []
    lines.append("Probe 3 Pressure-vs-Distractor Classification Summary")
    lines.append("")
    lines.append(f"hidden_state_shape: ({n_layers}, {d_model})")
    lines.append(f"n_examples_total: {len(examples)}")
    lines.append(f"n_families: {len({str(example['family_id']) for example in examples})}")
    lines.append("split_strategy: leave-one-family-out")
    lines.append("feature_definition: delta_to_family_neutral")
    lines.append("classifier: logistic_regression_l2_balanced_fixed_C")
    lines.append("")
    lines.append("Best Layerwise Results")
    lines.append(
        "  "
        f"balanced_accuracy: layer {best_balanced['layer']} ({best_balanced['layer_bucket']}), "
        f"value={format_float(float(best_balanced['balanced_accuracy']))}"
    )
    lines.append(
        "  "
        f"AUROC: layer {best_auroc['layer']} ({best_auroc['layer_bucket']}), "
        f"value={format_float(float(best_auroc['auroc']))}"
    )
    lines.append(
        "  "
        f"average_precision: layer {best_ap['layer']} ({best_ap['layer_bucket']}), "
        f"value={format_float(float(best_ap['average_precision']))}"
    )
    lines.append(
        "  "
        f"F1: layer {best_f1['layer']} ({best_f1['layer_bucket']}), "
        f"value={format_float(float(best_f1['f1']))}"
    )
    lines.append("")
    lines.append("Answers To Requested Questions")
    lines.append(
        "  "
        f"Pressure-like deltas are {'distinguishable' if float(best_balanced['balanced_accuracy']) > 0.7 else 'not clearly distinguishable'} from distractor deltas."
    )
    lines.append(
        "  "
        f"The best-performing layers are {best_balanced['layer_bucket']} / {best_auroc['layer_bucket']} / {best_ap['layer_bucket']} by balanced accuracy / AUROC / average precision."
    )
    lines.append(
        "  "
        f"Performance {'remains strong' if strong_family_held_out else 'is more modest'} under family-held-out evaluation."
    )
    lines.append(
        "  "
        "If family-held-out performance is well above the majority-class baseline, that supports pressure-specific representation rather than generic non-neutral movement."
    )
    lines.append("  Pressure conditions mistaken for distractor at the best balanced-accuracy layer:")
    for item in pressure_errors:
        lines.append(f"    {item}")
    lines.append("  Distractor mistaken for pressure at the best balanced-accuracy layer:")
    for item in distractor_errors:
        lines.append(f"    {item}")
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "  A successful classifier means pressure-like deltas are linearly decodable as distinct from distractor deltas under family-held-out evaluation."
    )
    lines.append(
        "  It does not establish causality or intervention success."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / DEFAULT_INPUT
    delta_input_path = repo_root / DEFAULT_DELTA_INPUT
    layerwise_output = repo_root / DEFAULT_LAYERWISE_OUTPUT
    predictions_output = repo_root / DEFAULT_PREDICTIONS_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    rows = read_jsonl(input_path)
    delta_csv_rows = read_csv_rows(delta_input_path)
    validate_family_delta_csv(rows, delta_csv_rows)
    grouped_rows = group_rows_by_family(rows)
    examples, tensor_shape = build_examples(repo_root, grouped_rows)
    n_layers, _d_model = tensor_shape

    prediction_rows, layerwise_rows = build_outputs(examples, n_layers)

    layerwise_fieldnames = [
        "layer",
        "layer_bucket",
        "n_examples",
        "n_families",
        "balanced_accuracy",
        "auroc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "confusion_matrix_counts",
        "baseline_balanced_accuracy",
        "C_used",
    ]
    predictions_fieldnames = [
        "family_id",
        "condition",
        "layer",
        "layer_bucket",
        "true_label",
        "predicted_label",
        "predicted_probability_pressure",
        "baseline_predicted_label",
        "fold_id",
    ]

    write_csv(layerwise_output, layerwise_rows, layerwise_fieldnames)
    write_csv(predictions_output, prediction_rows, predictions_fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        build_summary_text(tensor_shape, examples, layerwise_rows, prediction_rows),
        encoding="utf-8",
    )

    print(f"Wrote {layerwise_output}")
    print(f"Wrote {predictions_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
