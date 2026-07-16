import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_LAYERWISE_OUTPUT = "results/qwen3_4b_instruct_2507_family36_layerwise_delta_cosines.csv"
DEFAULT_SUMMARY_OUTPUT = "results/qwen3_4b_instruct_2507_family36_hidden_state_cosine_summary.txt"

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
COSINE_SPECS = (
    ("false_pressure_vs_true_pressure", "false_pressure_delta", "true_pressure_delta"),
    ("false_pressure_vs_emotional_pressure", "false_pressure_delta", "emotional_pressure_delta"),
    ("false_pressure_vs_closed_context", "false_pressure_delta", "closed_context_delta"),
    ("false_pressure_vs_distractor", "false_pressure_delta", "distractor_delta"),
    ("emotional_pressure_vs_closed_context", "emotional_pressure_delta", "closed_context_delta"),
    ("emotional_pressure_vs_distractor", "emotional_pressure_delta", "distractor_delta"),
    ("closed_context_vs_distractor", "closed_context_delta", "distractor_delta"),
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
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
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
        raise ValueError("No families found for cosine analysis.")
    return family_deltas, expected_shape


def compute_cosine(left_vector: torch.Tensor, right_vector: torch.Tensor) -> float:
    cosine = F.cosine_similarity(left_vector.unsqueeze(0), right_vector.unsqueeze(0), dim=1)
    return float(cosine.item())


def build_layerwise_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    family_ids = sorted(family_deltas)

    for layer_index in range(n_layers):
        for cosine_pair, left_delta_name, right_delta_name in COSINE_SPECS:
            values: List[float] = []
            for family_id in family_ids:
                left_vector = family_deltas[family_id][left_delta_name][layer_index]
                right_vector = family_deltas[family_id][right_delta_name][layer_index]
                values.append(compute_cosine(left_vector, right_vector))
            output_rows.append(
                {
                    "layer_index": layer_index,
                    "layer_bucket": layer_bucket(layer_index, n_layers),
                    "cosine_pair": cosine_pair,
                    "left_delta_type": left_delta_name,
                    "right_delta_type": right_delta_name,
                    "n_families": len(values),
                    "mean_cosine": mean(values),
                    "median_cosine": median(values) if values else 0.0,
                    "std_cosine": stddev(values),
                    "min_cosine": min(values) if values else 0.0,
                    "max_cosine": max(values) if values else 0.0,
                }
            )
    return output_rows


def pick_peak_row(rows: Sequence[Mapping[str, Any]], cosine_pair: str) -> Mapping[str, Any]:
    candidates = [row for row in rows if str(row["cosine_pair"]) == cosine_pair]
    if not candidates:
        raise ValueError(f"No rows found for cosine_pair={cosine_pair}")
    return max(candidates, key=lambda row: float(row["mean_cosine"]))


def build_pair_overall_means(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cosine_pair"])].append(float(row["mean_cosine"]))
    return {pair: mean(values) for pair, values in grouped.items()}


def strongest_pair_band(rows: Sequence[Mapping[str, Any]]) -> Tuple[Mapping[str, Any], str]:
    peak_row = max(rows, key=lambda row: float(row["mean_cosine"]))
    return peak_row, str(peak_row["layer_bucket"])


def build_summary_text(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    tensor_shape: Tuple[int, int],
    layerwise_rows: Sequence[Mapping[str, Any]],
) -> str:
    n_layers, d_model = tensor_shape
    pair_overall_means = build_pair_overall_means(layerwise_rows)

    false_pressure_candidates = {
        "true pressure": pair_overall_means["false_pressure_vs_true_pressure"],
        "emotional pressure": pair_overall_means["false_pressure_vs_emotional_pressure"],
        "closed-context pressure": pair_overall_means["false_pressure_vs_closed_context"],
        "distractor": pair_overall_means["false_pressure_vs_distractor"],
    }
    closest_false_pressure_label, closest_false_pressure_value = max(
        false_pressure_candidates.items(),
        key=lambda item: item[1],
    )

    emotional_closed_mean = pair_overall_means["emotional_pressure_vs_closed_context"]
    emotional_distractor_mean = pair_overall_means["emotional_pressure_vs_distractor"]
    closed_distractor_mean = pair_overall_means["closed_context_vs_distractor"]
    overall_peak_row, overall_peak_band = strongest_pair_band(layerwise_rows)

    lines: List[str] = []
    lines.append("Qwen3-4B-Instruct-2507 Family-36 Hidden-State Cosine Summary")
    lines.append("")
    lines.append(f"families: {len(family_deltas)}")
    lines.append(f"hidden_state_shape: ({n_layers}, {d_model})")
    lines.append("")
    lines.append("Overall Mean Cosine By Pair")
    for cosine_pair, _left_delta_name, _right_delta_name in COSINE_SPECS:
        lines.append(f"  {cosine_pair}: mean={format_float(pair_overall_means[cosine_pair])}")
    lines.append("")
    lines.append("Peak Mean Cosine By Pair")
    for cosine_pair, _left_delta_name, _right_delta_name in COSINE_SPECS:
        peak_row = pick_peak_row(layerwise_rows, cosine_pair)
        lines.append(
            "  "
            f"{cosine_pair}: layer {peak_row['layer_index']} ({peak_row['layer_bucket']}), "
            f"mean={format_float(float(peak_row['mean_cosine']))}, "
            f"median={format_float(float(peak_row['median_cosine']))}, "
            f"std={format_float(float(peak_row['std_cosine']))}, "
            f"min={format_float(float(peak_row['min_cosine']))}, "
            f"max={format_float(float(peak_row['max_cosine']))}"
        )
    lines.append("")
    lines.append("Answers To Requested Questions")
    lines.append(
        "  "
        f"Direct false pressure is directionally closest overall to {closest_false_pressure_label} "
        f"(overall mean cosine={format_float(closest_false_pressure_value)})."
    )
    lines.append(
        "  "
        f"Emotional pressure and closed-context pressure show "
        f"{'strong' if emotional_closed_mean >= 0.7 else 'moderate' if emotional_closed_mean >= 0.4 else 'weak'} "
        f"alignment overall (mean cosine={format_float(emotional_closed_mean)})."
    )
    lines.append(
        "  "
        f"Distractor effects are {'directionally separate from' if emotional_distractor_mean < emotional_closed_mean and closed_distractor_mean < emotional_closed_mean else 'not clearly separate from'} "
        f"pressure effects: emotional vs distractor mean={format_float(emotional_distractor_mean)}, "
        f"closed-context vs distractor mean={format_float(closed_distractor_mean)}, "
        f"false pressure vs distractor mean={format_float(pair_overall_means['false_pressure_vs_distractor'])}."
    )
    lines.append(
        "  "
        f"The strongest directional similarity occurs in the {overall_peak_band} layers: "
        f"{overall_peak_row['cosine_pair']} peaks at layer {overall_peak_row['layer_index']} "
        f"with mean cosine={format_float(float(overall_peak_row['mean_cosine']))}."
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append("  Cosines near +1 indicate similar directional hidden-state shifts relative to the neutral evidence prompt.")
    lines.append("  Cosines near 0 indicate weak directional alignment, even if both conditions have large delta norms.")
    lines.append("  This analysis is descriptive only; it does not establish causal mechanism or behavioral predictiveness.")
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
    layerwise_output = repo_root / DEFAULT_LAYERWISE_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    rows = read_jsonl(input_path)
    grouped_rows = group_rows_by_family(rows)
    family_deltas, tensor_shape = compute_family_deltas(repo_root, grouped_rows)
    n_layers, _d_model = tensor_shape
    layerwise_rows = build_layerwise_rows(family_deltas, n_layers)

    fieldnames = [
        "layer_index",
        "layer_bucket",
        "cosine_pair",
        "left_delta_type",
        "right_delta_type",
        "n_families",
        "mean_cosine",
        "median_cosine",
        "std_cosine",
        "min_cosine",
        "max_cosine",
    ]
    write_csv(layerwise_output, layerwise_rows, fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(build_summary_text(family_deltas, tensor_shape, layerwise_rows), encoding="utf-8")

    print(f"Wrote {layerwise_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
