import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_DELTA_INPUT = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_OUTPUT_CSV = "results/qwen3_4b_instruct_2507_family36_direction_projection_probe.csv"
DEFAULT_OUTPUT_TXT = "results/qwen3_4b_instruct_2507_family36_direction_projection_probe_summary.txt"

EXPECTED_PROMPT_TYPES = (
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
)
DELTA_SPECS = (
    ("false_pressure_delta", "evidence_false_belief_pressure", "delta_false_pressure", "negative"),
    ("emotional_pressure_delta", "evidence_emotional_pressure", "delta_emotional_pressure", "negative"),
    ("closed_context_delta", "closed_context_false_belief_pressure", "delta_closed_context", "negative"),
    ("distractor_delta", "evidence_distractor_neutral", "delta_distractor", "absolute"),
)


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


def format_float(value: float) -> str:
    return f"{value:.4f}"


def layer_bucket(layer_index: int, n_layers: int) -> str:
    one_third = n_layers / 3.0
    if layer_index < one_third:
        return "early"
    if layer_index < 2.0 * one_third:
        return "middle"
    return "late"


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


def compute_family_deltas(
    repo_root: Path,
    grouped_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Tuple[int, int]]:
    family_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
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

        family_deltas[family_id] = {}
        for delta_name, prompt_type, _behavior_delta, _transform in DELTA_SPECS:
            comparison = load_hidden_state_tensor(repo_root, family_rows[prompt_type])
            if tuple(comparison.shape) != expected_shape:
                raise ValueError(
                    f"Family {family_id} prompt {prompt_type} has tensor shape {tuple(comparison.shape)} != {expected_shape}"
                )
            family_deltas[family_id][delta_name] = comparison - neutral

    if expected_shape is None:
        raise ValueError("No deltas could be computed from the family activations.")
    return family_deltas, expected_shape


def load_behavior_deltas(path: Path) -> Dict[str, Dict[str, float]]:
    behavior_by_family: Dict[str, Dict[str, float]] = {}
    for row in read_csv_rows(path):
        family_id = str(row["family_id"])
        behavior_by_family[family_id] = {
            "delta_false_pressure": float(row["delta_false_pressure"]),
            "delta_emotional_pressure": float(row["delta_emotional_pressure"]),
            "delta_closed_context": float(row["delta_closed_context"]),
            "delta_distractor": float(row["delta_distractor"]),
        }
    return behavior_by_family


def transform_behavior_value(value: float, transform_kind: str) -> float:
    if transform_kind == "negative":
        return -value
    if transform_kind == "absolute":
        return abs(value)
    raise ValueError(f"Unsupported transform kind: {transform_kind}")


def fit_standardizer(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_vector = matrix.mean(dim=0)
    std_vector = matrix.std(dim=0, unbiased=False)
    std_vector = torch.where(std_vector > 0, std_vector, torch.ones_like(std_vector))
    return mean_vector, std_vector


def standardize(matrix: torch.Tensor, mean_vector: torch.Tensor, std_vector: torch.Tensor) -> torch.Tensor:
    return (matrix - mean_vector) / std_vector


def compute_projection(
    train_vectors: torch.Tensor,
    test_vector: torch.Tensor,
) -> float:
    mean_vector, std_vector = fit_standardizer(train_vectors)
    train_scaled = standardize(train_vectors, mean_vector, std_vector)
    test_scaled = standardize(test_vector.unsqueeze(0), mean_vector, std_vector).squeeze(0)

    mean_direction = train_scaled.mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(mean_direction).item())
    if direction_norm == 0.0:
        return 0.0
    unit_direction = mean_direction / direction_norm
    return float(torch.dot(test_scaled, unit_direction).item())


def build_probe_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_deltas: Mapping[str, Mapping[str, float]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    family_ids = sorted(family_deltas)
    rows: List[Dict[str, Any]] = []

    for delta_type, _prompt_type, behavior_delta_name, transform_kind in DELTA_SPECS:
        for layer_index in range(n_layers):
            heldout_projections: List[float] = []
            behavior_values: List[float] = []

            for heldout_family_id in family_ids:
                train_ids = [family_id for family_id in family_ids if family_id != heldout_family_id]
                train_vectors = torch.stack(
                    [family_deltas[family_id][delta_type][layer_index] for family_id in train_ids],
                    dim=0,
                )
                test_vector = family_deltas[heldout_family_id][delta_type][layer_index]
                projection = compute_projection(train_vectors, test_vector)
                heldout_projections.append(projection)
                behavior_values.append(
                    transform_behavior_value(
                        float(behavior_deltas[heldout_family_id][behavior_delta_name]),
                        transform_kind,
                    )
                )

            rows.append(
                {
                    "delta_type": delta_type,
                    "behavior_delta_name": behavior_delta_name,
                    "behavior_metric": (
                        f"-{behavior_delta_name}" if transform_kind == "negative" else f"abs({behavior_delta_name})"
                    ),
                    "layer_index": layer_index,
                    "layer_bucket": layer_bucket(layer_index, n_layers),
                    "n_families": len(family_ids),
                    "projection_mean": mean(heldout_projections),
                    "projection_median": median(heldout_projections),
                    "projection_std": stddev(heldout_projections),
                    "projection_min": min(heldout_projections),
                    "projection_max": max(heldout_projections),
                    "degradation_mean": mean(behavior_values),
                    "degradation_std": stddev(behavior_values),
                    "pearson_correlation": pearson_correlation(heldout_projections, behavior_values),
                    "spearman_correlation": spearman_correlation(heldout_projections, behavior_values),
                }
            )
    return rows


def best_row(rows: Sequence[Mapping[str, Any]], delta_type: str, metric_field: str) -> Mapping[str, Any]:
    candidates = [row for row in rows if str(row["delta_type"]) == delta_type]
    if not candidates:
        raise ValueError(f"No rows found for delta_type={delta_type}")
    return max(candidates, key=lambda row: float(row[metric_field]))


def build_summary_text(
    input_path: Path,
    tensor_shape: Tuple[int, int],
    probe_rows: Sequence[Mapping[str, Any]],
) -> str:
    n_layers, d_model = tensor_shape
    best_false_pearson = best_row(probe_rows, "false_pressure_delta", "pearson_correlation")
    best_false_spearman = best_row(probe_rows, "false_pressure_delta", "spearman_correlation")
    best_emotional_pearson = best_row(probe_rows, "emotional_pressure_delta", "pearson_correlation")
    best_emotional_spearman = best_row(probe_rows, "emotional_pressure_delta", "spearman_correlation")
    best_closed_pearson = best_row(probe_rows, "closed_context_delta", "pearson_correlation")
    best_closed_spearman = best_row(probe_rows, "closed_context_delta", "spearman_correlation")
    best_distractor_pearson = best_row(probe_rows, "distractor_delta", "pearson_correlation")
    best_distractor_spearman = best_row(probe_rows, "distractor_delta", "spearman_correlation")

    strongest_pearson = max(
        [best_false_pearson, best_emotional_pearson, best_closed_pearson, best_distractor_pearson],
        key=lambda row: float(row["pearson_correlation"]),
    )

    lines: List[str] = []
    lines.append("Qwen3-4B-Instruct-2507 Family-36 Direction Projection Probe")
    lines.append("")
    lines.append(f"input_jsonl: {input_path.name}")
    lines.append(f"hidden_state_shape: ({n_layers}, {d_model})")
    lines.append("split_strategy: leave-one-family-out")
    lines.append("feature_preprocessing: train-family standardization applied before mean-direction construction and held-out projection")
    lines.append("")
    lines.append("Best Layerwise Correlations")
    lines.append(
        "  "
        f"false_pressure_delta projection vs -delta_false_pressure: "
        f"best Pearson layer {best_false_pearson['layer_index']} ({best_false_pearson['layer_bucket']}), "
        f"r={format_float(float(best_false_pearson['pearson_correlation']))}; "
        f"best Spearman layer {best_false_spearman['layer_index']} ({best_false_spearman['layer_bucket']}), "
        f"rho={format_float(float(best_false_spearman['spearman_correlation']))}"
    )
    lines.append(
        "  "
        f"emotional_pressure_delta projection vs -delta_emotional_pressure: "
        f"best Pearson layer {best_emotional_pearson['layer_index']} ({best_emotional_pearson['layer_bucket']}), "
        f"r={format_float(float(best_emotional_pearson['pearson_correlation']))}; "
        f"best Spearman layer {best_emotional_spearman['layer_index']} ({best_emotional_spearman['layer_bucket']}), "
        f"rho={format_float(float(best_emotional_spearman['spearman_correlation']))}"
    )
    lines.append(
        "  "
        f"closed_context_delta projection vs -delta_closed_context: "
        f"best Pearson layer {best_closed_pearson['layer_index']} ({best_closed_pearson['layer_bucket']}), "
        f"r={format_float(float(best_closed_pearson['pearson_correlation']))}; "
        f"best Spearman layer {best_closed_spearman['layer_index']} ({best_closed_spearman['layer_bucket']}), "
        f"rho={format_float(float(best_closed_spearman['spearman_correlation']))}"
    )
    lines.append(
        "  "
        f"distractor_delta projection vs abs(delta_distractor): "
        f"best Pearson layer {best_distractor_pearson['layer_index']} ({best_distractor_pearson['layer_bucket']}), "
        f"r={format_float(float(best_distractor_pearson['pearson_correlation']))}; "
        f"best Spearman layer {best_distractor_spearman['layer_index']} ({best_distractor_spearman['layer_bucket']}), "
        f"rho={format_float(float(best_distractor_spearman['spearman_correlation']))}"
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "  "
        f"The strongest direction-projection alignment appears for {strongest_pearson['delta_type']} at "
        f"layer {strongest_pearson['layer_index']} ({strongest_pearson['layer_bucket']}) with "
        f"Pearson r={format_float(float(strongest_pearson['pearson_correlation']))}."
    )
    lines.append(
        "  "
        "Positive correlation means families that move farther along the shared training-family direction also show larger held-out behavioral degradation."
    )
    lines.append(
        "  "
        "Because the direction and scaler are fitted without the held-out family, this probe directly tests whether the shared pressure direction generalizes across families rather than merely describing the full dataset."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / DEFAULT_INPUT
    delta_input_path = repo_root / DEFAULT_DELTA_INPUT
    output_csv = repo_root / DEFAULT_OUTPUT_CSV
    output_txt = repo_root / DEFAULT_OUTPUT_TXT

    rows = read_jsonl(input_path)
    grouped_rows = group_rows_by_family(rows)
    family_deltas, tensor_shape = compute_family_deltas(repo_root, grouped_rows)
    n_layers, _d_model = tensor_shape
    behavior_deltas = load_behavior_deltas(delta_input_path)

    probe_rows = build_probe_rows(family_deltas, behavior_deltas, n_layers)
    fieldnames = [
        "delta_type",
        "behavior_delta_name",
        "behavior_metric",
        "layer_index",
        "layer_bucket",
        "n_families",
        "projection_mean",
        "projection_median",
        "projection_std",
        "projection_min",
        "projection_max",
        "degradation_mean",
        "degradation_std",
        "pearson_correlation",
        "spearman_correlation",
    ]
    write_csv(output_csv, probe_rows, fieldnames)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(build_summary_text(input_path, tensor_shape, probe_rows), encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_txt}")


if __name__ == "__main__":
    main()
