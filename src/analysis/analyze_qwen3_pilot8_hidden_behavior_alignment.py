import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_INPUT = "outputs/state_logits_qwen3_1_7b_subset.jsonl"
DEFAULT_BEHAVIOR_DELTA_INPUT = "results/qwen3_1_7b_pilot8_family_margin_deltas.csv"
DEFAULT_CORRELATIONS_OUTPUT = "results/qwen3_1_7b_pilot8_hidden_behavior_correlations.csv"
DEFAULT_DIRECTION_CONSISTENCY_OUTPUT = "results/qwen3_1_7b_pilot8_false_pressure_direction_consistency.csv"
DEFAULT_SUMMARY_OUTPUT = "results/qwen3_1_7b_pilot8_phase3e_summary.txt"

EXPECTED_PROMPT_TYPES = (
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
)
ALIGNMENT_SPECS = (
    ("false_pressure_delta_norm", "delta_false_pressure"),
    ("closed_context_delta_norm", "delta_closed_context"),
    ("distractor_delta_norm", "delta_distractor"),
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


def format_float(value: float) -> str:
    return f"{value:.4f}"


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

        family_deltas[family_id] = {
            "false_pressure_delta": load_hidden_state_tensor(repo_root, family_rows["evidence_false_belief_pressure"]) - neutral,
            "closed_context_delta": load_hidden_state_tensor(repo_root, family_rows["closed_context_false_belief_pressure"]) - neutral,
            "distractor_delta": load_hidden_state_tensor(repo_root, family_rows["evidence_distractor_neutral"]) - neutral,
        }

    if expected_shape is None:
        raise ValueError("No families available for hidden-state alignment analysis.")
    return family_deltas, expected_shape


def load_behavior_deltas(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_rows(path)
    behavior_by_family: Dict[str, Dict[str, float]] = {}
    for row in rows:
        family_id = str(row["family_id"])
        behavior_by_family[family_id] = {
            "delta_false_pressure": float(row["delta_false_pressure"]),
            "delta_closed_context": float(row["delta_closed_context"]),
            "delta_distractor": float(row["delta_distractor"]),
        }
    return behavior_by_family


def build_hidden_behavior_correlation_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_deltas: Mapping[str, Mapping[str, float]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    family_ids = sorted(family_deltas)
    rows: List[Dict[str, Any]] = []

    for layer_index in range(n_layers):
        for hidden_metric_name, behavior_delta_name in ALIGNMENT_SPECS:
            hidden_values: List[float] = []
            behavior_values: List[float] = []
            for family_id in family_ids:
                if family_id not in behavior_deltas:
                    raise ValueError(f"Missing behavior delta row for family {family_id}")
                if hidden_metric_name == "false_pressure_delta_norm":
                    vector = family_deltas[family_id]["false_pressure_delta"][layer_index]
                elif hidden_metric_name == "closed_context_delta_norm":
                    vector = family_deltas[family_id]["closed_context_delta"][layer_index]
                elif hidden_metric_name == "distractor_delta_norm":
                    vector = family_deltas[family_id]["distractor_delta"][layer_index]
                else:
                    raise ValueError(f"Unknown hidden metric {hidden_metric_name}")
                hidden_values.append(float(torch.linalg.vector_norm(vector).item()))
                behavior_values.append(abs(float(behavior_deltas[family_id][behavior_delta_name])))

            rows.append(
                {
                    "layer_index": layer_index,
                    "hidden_metric": hidden_metric_name,
                    "behavior_metric": f"abs({behavior_delta_name})",
                    "n_families": len(family_ids),
                    "pearson_correlation": pearson_correlation(hidden_values, behavior_values),
                    "mean_hidden_metric": mean(hidden_values),
                    "mean_abs_behavior_metric": mean(behavior_values),
                }
            )
    return rows


def build_direction_consistency_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    family_ids = sorted(family_deltas)
    rows: List[Dict[str, Any]] = []

    for layer_index in range(n_layers):
        layer_vectors = {
            family_id: family_deltas[family_id]["false_pressure_delta"][layer_index]
            for family_id in family_ids
        }
        mean_direction_all = torch.stack([layer_vectors[family_id] for family_id in family_ids], dim=0).mean(dim=0)

        all_cosines: List[float] = []
        loo_cosines: List[float] = []
        for family_id in family_ids:
            family_vector = layer_vectors[family_id]
            cosine_all = float(
                F.cosine_similarity(family_vector.unsqueeze(0), mean_direction_all.unsqueeze(0), dim=1).item()
            )

            other_vectors = [layer_vectors[other_family_id] for other_family_id in family_ids if other_family_id != family_id]
            leave_one_out_mean = torch.stack(other_vectors, dim=0).mean(dim=0)
            cosine_loo = float(
                F.cosine_similarity(family_vector.unsqueeze(0), leave_one_out_mean.unsqueeze(0), dim=1).item()
            )

            all_cosines.append(cosine_all)
            loo_cosines.append(cosine_loo)
            rows.append(
                {
                    "layer_index": layer_index,
                    "family_id": family_id,
                    "cosine_to_all8_mean_direction": cosine_all,
                    "cosine_to_leave_one_out_mean_direction": cosine_loo,
                }
            )

        rows.append(
            {
                "layer_index": layer_index,
                "family_id": "__summary__",
                "cosine_to_all8_mean_direction": mean(all_cosines),
                "cosine_to_leave_one_out_mean_direction": mean(loo_cosines),
            }
        )
    return rows


def build_summary_text(
    tensor_shape: Tuple[int, int],
    correlation_rows: Sequence[Mapping[str, Any]],
    direction_rows: Sequence[Mapping[str, Any]],
) -> str:
    n_layers, d_model = tensor_shape
    lines: List[str] = []
    lines.append("Qwen3 1.7B Pilot-8 Phase 3E Summary")
    lines.append("")
    lines.append(f"hidden_state_shape: ({n_layers}, {d_model})")
    lines.append("")

    for hidden_metric_name, behavior_delta_name in ALIGNMENT_SPECS:
        matching_rows = [
            row
            for row in correlation_rows
            if str(row["hidden_metric"]) == hidden_metric_name
            and str(row["behavior_metric"]) == f"abs({behavior_delta_name})"
        ]
        best_positive = max(matching_rows, key=lambda row: float(row["pearson_correlation"]))
        best_negative = min(matching_rows, key=lambda row: float(row["pearson_correlation"]))
        lines.append(
            f"{hidden_metric_name} vs abs({behavior_delta_name})"
        )
        lines.append(
            "  "
            f"best positive correlation: layer {best_positive['layer_index']}, "
            f"r={format_float(float(best_positive['pearson_correlation']))}"
        )
        lines.append(
            "  "
            f"most negative correlation: layer {best_negative['layer_index']}, "
            f"r={format_float(float(best_negative['pearson_correlation']))}"
        )
        lines.append("")

    direction_summary_rows = [row for row in direction_rows if str(row["family_id"]) == "__summary__"]
    best_all = max(direction_summary_rows, key=lambda row: float(row["cosine_to_all8_mean_direction"]))
    best_loo = max(direction_summary_rows, key=lambda row: float(row["cosine_to_leave_one_out_mean_direction"]))
    worst_loo = min(direction_summary_rows, key=lambda row: float(row["cosine_to_leave_one_out_mean_direction"]))

    lines.append("False-Pressure Direction Consistency")
    lines.append(
        "  "
        f"highest mean cosine to all-8 mean direction: layer {best_all['layer_index']}, "
        f"mean_cosine={format_float(float(best_all['cosine_to_all8_mean_direction']))}"
    )
    lines.append(
        "  "
        f"highest mean cosine to leave-one-out mean direction: layer {best_loo['layer_index']}, "
        f"mean_cosine={format_float(float(best_loo['cosine_to_leave_one_out_mean_direction']))}"
    )
    lines.append(
        "  "
        f"lowest mean cosine to leave-one-out mean direction: layer {worst_loo['layer_index']}, "
        f"mean_cosine={format_float(float(worst_loo['cosine_to_leave_one_out_mean_direction']))}"
    )
    lines.append("")
    lines.append("Interpretation Notes")
    lines.append("  Positive correlation means larger hidden-state movement is associated with larger absolute behavioral margin change across families.")
    lines.append("  Higher direction-consistency cosine means the family-level false-pressure deltas point in a more shared representational direction at that layer.")
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
    input_path = repo_root / DEFAULT_INPUT
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
        "hidden_metric",
        "behavior_metric",
        "n_families",
        "pearson_correlation",
        "mean_hidden_metric",
        "mean_abs_behavior_metric",
    ]
    direction_fieldnames = [
        "layer_index",
        "family_id",
        "cosine_to_all8_mean_direction",
        "cosine_to_leave_one_out_mean_direction",
    ]

    write_csv(correlations_output, correlation_rows, correlation_fieldnames)
    write_csv(direction_consistency_output, direction_rows, direction_fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        build_summary_text(tensor_shape, correlation_rows, direction_rows),
        encoding="utf-8",
    )

    print(f"Wrote {correlations_output}")
    print(f"Wrote {direction_consistency_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
