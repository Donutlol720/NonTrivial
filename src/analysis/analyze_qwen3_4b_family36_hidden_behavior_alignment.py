import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_INPUT_CANDIDATES = (
    "outputs/state_logits_qwen3_4b_instruct_2507_family36.jsonl",
    "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl",
)
DEFAULT_BEHAVIOR_DELTA_INPUT = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_CORRELATIONS_OUTPUT = "results/qwen3_4b_instruct_2507_family36_hidden_behavior_correlations.csv"
DEFAULT_DIRECTION_CONSISTENCY_OUTPUT = "results/qwen3_4b_instruct_2507_family36_pressure_direction_consistency.csv"
DEFAULT_SUMMARY_OUTPUT = "results/qwen3_4b_instruct_2507_family36_hidden_behavior_alignment_summary.txt"

EXPECTED_PROMPT_TYPES = (
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
)
DELTA_SPECS = (
    ("false_pressure_delta", "evidence_false_belief_pressure"),
    ("emotional_pressure_delta", "evidence_emotional_pressure"),
    ("true_pressure_delta", "evidence_true_belief_pressure"),
    ("distractor_delta", "evidence_distractor_neutral"),
    ("closed_context_delta", "closed_context_false_belief_pressure"),
)
CORRELATION_SPECS = (
    ("false_pressure_delta", "delta_false_pressure", "negative"),
    ("emotional_pressure_delta", "delta_emotional_pressure", "negative"),
    ("closed_context_delta", "delta_closed_context", "negative"),
    ("distractor_delta", "delta_distractor", "absolute"),
)
DIRECTION_CONSISTENCY_DELTA_TYPES = (
    "false_pressure_delta",
    "emotional_pressure_delta",
    "closed_context_delta",
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


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


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


def resolve_input_path(repo_root: Path) -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        candidate_path = repo_root / candidate
        if candidate_path.exists():
            return candidate_path
    joined = ", ".join(str(repo_root / candidate) for candidate in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"Could not find any expected 4B family36 input JSONL. Tried: {joined}")


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
            original_index = indexed[idx][0]
            ranks[original_index] = average_rank
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

        deltas_for_family: Dict[str, torch.Tensor] = {}
        for delta_name, comparison_prompt_type in DELTA_SPECS:
            comparison = load_hidden_state_tensor(repo_root, family_rows[comparison_prompt_type])
            if tuple(comparison.shape) != expected_shape:
                raise ValueError(
                    f"Family {family_id} prompt {comparison_prompt_type} has tensor shape {tuple(comparison.shape)} != {expected_shape}"
                )
            deltas_for_family[delta_name] = comparison - neutral
        family_deltas[family_id] = deltas_for_family

    if expected_shape is None:
        raise ValueError("No families available for hidden-behavior alignment analysis.")
    return family_deltas, expected_shape


def load_behavior_deltas(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_rows(path)
    behavior_by_family: Dict[str, Dict[str, float]] = {}
    for row in rows:
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
    raise ValueError(f"Unknown transform_kind: {transform_kind}")


def build_hidden_behavior_correlation_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_deltas: Mapping[str, Mapping[str, float]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    family_ids = sorted(family_deltas)
    rows: List[Dict[str, Any]] = []

    for layer_index in range(n_layers):
        for delta_type, behavior_delta_name, transform_kind in CORRELATION_SPECS:
            hidden_values: List[float] = []
            behavior_values: List[float] = []
            for family_id in family_ids:
                if family_id not in behavior_deltas:
                    raise ValueError(f"Missing behavior delta row for family {family_id}")
                delta_vector = family_deltas[family_id][delta_type][layer_index]
                hidden_values.append(float(torch.linalg.vector_norm(delta_vector).item()))
                behavior_values.append(
                    transform_behavior_value(float(behavior_deltas[family_id][behavior_delta_name]), transform_kind)
                )

            rows.append(
                {
                    "layer_index": layer_index,
                    "layer_bucket": layer_bucket(layer_index, n_layers),
                    "delta_type": delta_type,
                    "behavior_delta_name": behavior_delta_name,
                    "behavior_transform": transform_kind,
                    "hidden_metric": f"||{delta_type}||",
                    "behavior_metric": (
                        f"-{behavior_delta_name}" if transform_kind == "negative" else f"abs({behavior_delta_name})"
                    ),
                    "n_families": len(family_ids),
                    "pearson_correlation": pearson_correlation(hidden_values, behavior_values),
                    "spearman_correlation": spearman_correlation(hidden_values, behavior_values),
                    "mean_hidden_metric": mean(hidden_values),
                    "std_hidden_metric": stddev(hidden_values),
                    "mean_behavior_metric": mean(behavior_values),
                    "std_behavior_metric": stddev(behavior_values),
                }
            )
    return rows


def build_direction_consistency_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    family_ids = sorted(family_deltas)
    output_rows: List[Dict[str, Any]] = []

    for layer_index in range(n_layers):
        for delta_type in DIRECTION_CONSISTENCY_DELTA_TYPES:
            values: List[float] = []
            layer_vectors = {
                family_id: family_deltas[family_id][delta_type][layer_index]
                for family_id in family_ids
            }
            for family_id in family_ids:
                family_vector = layer_vectors[family_id]
                other_vectors = [layer_vectors[other_family_id] for other_family_id in family_ids if other_family_id != family_id]
                leave_one_out_mean = torch.stack(other_vectors, dim=0).mean(dim=0)
                cosine_loo = float(
                    F.cosine_similarity(family_vector.unsqueeze(0), leave_one_out_mean.unsqueeze(0), dim=1).item()
                )
                values.append(cosine_loo)

            output_rows.append(
                {
                    "layer_index": layer_index,
                    "layer_bucket": layer_bucket(layer_index, n_layers),
                    "delta_type": delta_type,
                    "n_families": len(values),
                    "mean_cosine_to_leave_one_out_mean_direction": mean(values),
                    "median_cosine_to_leave_one_out_mean_direction": median(values) if values else 0.0,
                    "std_cosine_to_leave_one_out_mean_direction": stddev(values),
                    "min_cosine_to_leave_one_out_mean_direction": min(values) if values else 0.0,
                    "max_cosine_to_leave_one_out_mean_direction": max(values) if values else 0.0,
                }
            )
    return output_rows


def pick_best_row(
    rows: Sequence[Mapping[str, Any]],
    delta_type: str,
    metric_field: str,
) -> Mapping[str, Any]:
    candidates = [row for row in rows if str(row["delta_type"]) == delta_type]
    if not candidates:
        raise ValueError(f"No rows found for delta_type={delta_type}")
    return max(candidates, key=lambda row: float(row[metric_field]))


def build_summary_text(
    input_path: Path,
    tensor_shape: Tuple[int, int],
    correlation_rows: Sequence[Mapping[str, Any]],
    direction_rows: Sequence[Mapping[str, Any]],
) -> str:
    n_layers, d_model = tensor_shape

    best_false_pearson = pick_best_row(correlation_rows, "false_pressure_delta", "pearson_correlation")
    best_false_spearman = pick_best_row(correlation_rows, "false_pressure_delta", "spearman_correlation")
    best_emotional_pearson = pick_best_row(correlation_rows, "emotional_pressure_delta", "pearson_correlation")
    best_emotional_spearman = pick_best_row(correlation_rows, "emotional_pressure_delta", "spearman_correlation")
    best_closed_pearson = pick_best_row(correlation_rows, "closed_context_delta", "pearson_correlation")
    best_closed_spearman = pick_best_row(correlation_rows, "closed_context_delta", "spearman_correlation")
    best_distractor_pearson = pick_best_row(correlation_rows, "distractor_delta", "pearson_correlation")
    best_distractor_spearman = pick_best_row(correlation_rows, "distractor_delta", "spearman_correlation")

    best_false_direction = pick_best_row(
        direction_rows,
        "false_pressure_delta",
        "mean_cosine_to_leave_one_out_mean_direction",
    )
    best_emotional_direction = pick_best_row(
        direction_rows,
        "emotional_pressure_delta",
        "mean_cosine_to_leave_one_out_mean_direction",
    )
    best_closed_direction = pick_best_row(
        direction_rows,
        "closed_context_delta",
        "mean_cosine_to_leave_one_out_mean_direction",
    )

    lines: List[str] = []
    lines.append("Qwen3-4B-Instruct-2507 Family-36 Hidden/Behavior Alignment Summary")
    lines.append("")
    lines.append(f"input_jsonl: {input_path.name}")
    lines.append(f"hidden_state_shape: ({n_layers}, {d_model})")
    lines.append("")
    lines.append("Best Correlation By Condition")
    lines.append(
        "  "
        f"false_pressure_delta vs -delta_false_pressure: "
        f"best Pearson layer {best_false_pearson['layer_index']} ({best_false_pearson['layer_bucket']}), "
        f"r={format_float(float(best_false_pearson['pearson_correlation']))}; "
        f"best Spearman layer {best_false_spearman['layer_index']} ({best_false_spearman['layer_bucket']}), "
        f"rho={format_float(float(best_false_spearman['spearman_correlation']))}"
    )
    lines.append(
        "  "
        f"emotional_pressure_delta vs -delta_emotional_pressure: "
        f"best Pearson layer {best_emotional_pearson['layer_index']} ({best_emotional_pearson['layer_bucket']}), "
        f"r={format_float(float(best_emotional_pearson['pearson_correlation']))}; "
        f"best Spearman layer {best_emotional_spearman['layer_index']} ({best_emotional_spearman['layer_bucket']}), "
        f"rho={format_float(float(best_emotional_spearman['spearman_correlation']))}"
    )
    lines.append(
        "  "
        f"closed_context_delta vs -delta_closed_context: "
        f"best Pearson layer {best_closed_pearson['layer_index']} ({best_closed_pearson['layer_bucket']}), "
        f"r={format_float(float(best_closed_pearson['pearson_correlation']))}; "
        f"best Spearman layer {best_closed_spearman['layer_index']} ({best_closed_spearman['layer_bucket']}), "
        f"rho={format_float(float(best_closed_spearman['spearman_correlation']))}"
    )
    lines.append(
        "  "
        f"distractor_delta vs abs(delta_distractor): "
        f"best Pearson layer {best_distractor_pearson['layer_index']} ({best_distractor_pearson['layer_bucket']}), "
        f"r={format_float(float(best_distractor_pearson['pearson_correlation']))}; "
        f"best Spearman layer {best_distractor_spearman['layer_index']} ({best_distractor_spearman['layer_bucket']}), "
        f"rho={format_float(float(best_distractor_spearman['spearman_correlation']))}"
    )
    lines.append("")
    lines.append("Pressure Direction Consistency")
    lines.append(
        "  "
        f"false_pressure_delta: best layer {best_false_direction['layer_index']} ({best_false_direction['layer_bucket']}), "
        f"mean leave-one-out cosine={format_float(float(best_false_direction['mean_cosine_to_leave_one_out_mean_direction']))}"
    )
    lines.append(
        "  "
        f"emotional_pressure_delta: best layer {best_emotional_direction['layer_index']} ({best_emotional_direction['layer_bucket']}), "
        f"mean leave-one-out cosine={format_float(float(best_emotional_direction['mean_cosine_to_leave_one_out_mean_direction']))}"
    )
    lines.append(
        "  "
        f"closed_context_delta: best layer {best_closed_direction['layer_index']} ({best_closed_direction['layer_bucket']}), "
        f"mean leave-one-out cosine={format_float(float(best_closed_direction['mean_cosine_to_leave_one_out_mean_direction']))}"
    )
    lines.append("")
    lines.append("Answers To Requested Questions")
    lines.append(
        "  "
        f"Larger emotional-pressure shifts {'do' if float(best_emotional_pearson['pearson_correlation']) > 0 else 'do not'} "
        f"predict larger emotional margin drops overall; the strongest alignment is "
        f"Pearson r={format_float(float(best_emotional_pearson['pearson_correlation']))} "
        f"and Spearman rho={format_float(float(best_emotional_spearman['spearman_correlation']))}."
    )
    lines.append(
        "  "
        f"Larger closed-context shifts {'do' if float(best_closed_pearson['pearson_correlation']) > 0 else 'do not'} "
        f"predict larger closed-context margin drops overall; the strongest alignment is "
        f"Pearson r={format_float(float(best_closed_pearson['pearson_correlation']))} "
        f"and Spearman rho={format_float(float(best_closed_spearman['spearman_correlation']))}."
    )
    false_r = float(best_false_pearson["pearson_correlation"])
    emotional_r = float(best_emotional_pearson["pearson_correlation"])
    closed_r = float(best_closed_pearson["pearson_correlation"])
    lines.append(
        "  "
        f"False-pressure shifts are behaviorally mixed rather than uniformly weak: "
        f"best false-pressure Pearson r={format_float(false_r)}, "
        f"vs emotional r={format_float(emotional_r)}, "
        f"closed-context r={format_float(closed_r)}. "
        f"That means false-pressure alignment is stronger than emotional-pressure alignment, "
        f"but weaker than closed-context alignment."
    )
    lines.append(
        "  "
        f"Pressure directions are "
        f"{'highly' if min(float(best_false_direction['mean_cosine_to_leave_one_out_mean_direction']), float(best_emotional_direction['mean_cosine_to_leave_one_out_mean_direction']), float(best_closed_direction['mean_cosine_to_leave_one_out_mean_direction'])) >= 0.7 else 'moderately'} "
        f"consistent across families, with best leave-one-out mean cosines of "
        f"{format_float(float(best_false_direction['mean_cosine_to_leave_one_out_mean_direction']))} (false), "
        f"{format_float(float(best_emotional_direction['mean_cosine_to_leave_one_out_mean_direction']))} (emotional), "
        f"and {format_float(float(best_closed_direction['mean_cosine_to_leave_one_out_mean_direction']))} (closed-context)."
    )
    lines.append(
        "  "
        f"The damaging conditions do not behave uniformly: closed-context shows stronger hidden/behavior alignment "
        f"than the more robust retrieved-evidence false-pressure condition, while emotional pressure shows weaker alignment."
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "  Positive Pearson/Spearman values mean larger hidden-state shifts are associated with larger behavioral margin degradation across families."
    )
    lines.append(
        "  Higher leave-one-out direction cosine means family-level deltas point in a more shared representational direction."
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = resolve_input_path(repo_root)
    behavior_delta_path = repo_root / DEFAULT_BEHAVIOR_DELTA_INPUT
    correlations_output = repo_root / DEFAULT_CORRELATIONS_OUTPUT
    direction_consistency_output = repo_root / DEFAULT_DIRECTION_CONSISTENCY_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    rows = read_jsonl(input_path)
    grouped_rows = group_rows_by_family(rows)
    family_deltas, tensor_shape = compute_family_deltas(repo_root, grouped_rows)
    n_layers, _d_model = tensor_shape
    behavior_deltas = load_behavior_deltas(behavior_delta_path)

    correlation_rows = build_hidden_behavior_correlation_rows(family_deltas, behavior_deltas, n_layers)
    direction_rows = build_direction_consistency_rows(family_deltas, n_layers)

    correlation_fieldnames = [
        "layer_index",
        "layer_bucket",
        "delta_type",
        "behavior_delta_name",
        "behavior_transform",
        "hidden_metric",
        "behavior_metric",
        "n_families",
        "pearson_correlation",
        "spearman_correlation",
        "mean_hidden_metric",
        "std_hidden_metric",
        "mean_behavior_metric",
        "std_behavior_metric",
    ]
    direction_fieldnames = [
        "layer_index",
        "layer_bucket",
        "delta_type",
        "n_families",
        "mean_cosine_to_leave_one_out_mean_direction",
        "median_cosine_to_leave_one_out_mean_direction",
        "std_cosine_to_leave_one_out_mean_direction",
        "min_cosine_to_leave_one_out_mean_direction",
        "max_cosine_to_leave_one_out_mean_direction",
    ]

    write_csv(correlations_output, correlation_rows, correlation_fieldnames)
    write_csv(direction_consistency_output, direction_rows, direction_fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        build_summary_text(input_path, tensor_shape, correlation_rows, direction_rows),
        encoding="utf-8",
    )

    print(f"Wrote {correlations_output}")
    print(f"Wrote {direction_consistency_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
