import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_INPUT = "outputs/state_logits_qwen3_1_7b_subset.jsonl"
DEFAULT_DELTA_NORMS_OUTPUT = "results/qwen3_1_7b_pilot8_layerwise_delta_norms.csv"
DEFAULT_DELTA_COSINES_OUTPUT = "results/qwen3_1_7b_pilot8_layerwise_delta_cosines.csv"
DEFAULT_OUTLIER_COMPARISON_OUTPUT = "results/qwen3_1_7b_pilot8_layerwise_outlier_comparison.csv"
DEFAULT_SUMMARY_OUTPUT = "results/qwen3_1_7b_pilot8_hidden_state_delta_summary.txt"

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
    ("cosine(false_pressure_delta, emotional_pressure_delta)", "false_pressure_delta", "emotional_pressure_delta"),
    ("cosine(false_pressure_delta, true_pressure_delta)", "false_pressure_delta", "true_pressure_delta"),
    ("cosine(false_pressure_delta, distractor_delta)", "false_pressure_delta", "distractor_delta"),
    ("cosine(false_pressure_delta, closed_context_delta)", "false_pressure_delta", "closed_context_delta"),
)
OUTLIER_FAMILY_ID = "policy_library_checkout_003"
SUBSET_SPECS = (
    ("all_8_families", "all 8 families"),
    ("excluding_policy_library_checkout_003", "excluding policy_library_checkout_003"),
    ("policy_library_checkout_003_alone", "policy_library_checkout_003 alone"),
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


def format_float(value: float) -> str:
    return f"{value:.4f}"


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
        raise ValueError("No families found for delta analysis.")
    return family_deltas, expected_shape


def select_subset_family_ids(
    subset_key: str,
    family_ids: Sequence[str],
) -> List[str]:
    if subset_key == "all_8_families":
        return list(family_ids)
    if subset_key == "excluding_policy_library_checkout_003":
        return [family_id for family_id in family_ids if family_id != OUTLIER_FAMILY_ID]
    if subset_key == "policy_library_checkout_003_alone":
        return [family_id for family_id in family_ids if family_id == OUTLIER_FAMILY_ID]
    raise ValueError(f"Unknown subset_key: {subset_key}")


def build_layerwise_delta_norm_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    family_ids = sorted(family_deltas)
    rows: List[Dict[str, Any]] = []

    for subset_key, subset_label in SUBSET_SPECS:
        subset_family_ids = select_subset_family_ids(subset_key, family_ids)
        for layer_index in range(n_layers):
            for delta_name, _comparison_prompt_type in DELTA_SPECS:
                values: List[float] = []
                for family_id in subset_family_ids:
                    delta_vector = family_deltas[family_id][delta_name][layer_index]
                    values.append(float(torch.linalg.vector_norm(delta_vector).item()))
                rows.append(
                    {
                        "subset_key": subset_key,
                        "subset_label": subset_label,
                        "layer_index": layer_index,
                        "delta_type": delta_name,
                        "n_families": len(values),
                        "mean_delta_norm": mean(values),
                        "median_delta_norm": median(values) if values else 0.0,
                    }
                )
    return rows


def build_layerwise_delta_cosine_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    family_ids = sorted(family_deltas)
    rows: List[Dict[str, Any]] = []

    for subset_key, subset_label in SUBSET_SPECS:
        subset_family_ids = select_subset_family_ids(subset_key, family_ids)
        for layer_index in range(n_layers):
            for cosine_name, left_delta_name, right_delta_name in COSINE_SPECS:
                values: List[float] = []
                for family_id in subset_family_ids:
                    left_vector = family_deltas[family_id][left_delta_name][layer_index]
                    right_vector = family_deltas[family_id][right_delta_name][layer_index]
                    cosine_value = F.cosine_similarity(left_vector.unsqueeze(0), right_vector.unsqueeze(0), dim=1)
                    values.append(float(cosine_value.item()))
                rows.append(
                    {
                        "subset_key": subset_key,
                        "subset_label": subset_label,
                        "layer_index": layer_index,
                        "cosine_pair": cosine_name,
                        "n_families": len(values),
                        "mean_cosine": mean(values),
                        "median_cosine": median(values) if values else 0.0,
                    }
                )
    return rows


def build_outlier_comparison_rows(
    norm_rows: Sequence[Mapping[str, Any]],
    cosine_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    norm_map: Dict[Tuple[int, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    cosine_map: Dict[Tuple[int, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)

    for row in norm_rows:
        norm_map[(int(row["layer_index"]), str(row["delta_type"]))][str(row["subset_key"])] = row
    for row in cosine_rows:
        cosine_map[(int(row["layer_index"]), str(row["cosine_pair"]))][str(row["subset_key"])] = row

    output_rows: List[Dict[str, Any]] = []
    for (layer_index, delta_type), subset_rows in sorted(norm_map.items()):
        output_rows.append(
            {
                "analysis_kind": "delta_norm",
                "metric_name": delta_type,
                "layer_index": layer_index,
                "all_8_families_mean": subset_rows["all_8_families"]["mean_delta_norm"],
                "all_8_families_median": subset_rows["all_8_families"]["median_delta_norm"],
                "excluding_policy_library_checkout_003_mean": subset_rows["excluding_policy_library_checkout_003"]["mean_delta_norm"],
                "excluding_policy_library_checkout_003_median": subset_rows["excluding_policy_library_checkout_003"]["median_delta_norm"],
                "policy_library_checkout_003_alone_mean": subset_rows["policy_library_checkout_003_alone"]["mean_delta_norm"],
                "policy_library_checkout_003_alone_median": subset_rows["policy_library_checkout_003_alone"]["median_delta_norm"],
            }
        )

    for (layer_index, cosine_pair), subset_rows in sorted(cosine_map.items()):
        output_rows.append(
            {
                "analysis_kind": "delta_cosine",
                "metric_name": cosine_pair,
                "layer_index": layer_index,
                "all_8_families_mean": subset_rows["all_8_families"]["mean_cosine"],
                "all_8_families_median": subset_rows["all_8_families"]["median_cosine"],
                "excluding_policy_library_checkout_003_mean": subset_rows["excluding_policy_library_checkout_003"]["mean_cosine"],
                "excluding_policy_library_checkout_003_median": subset_rows["excluding_policy_library_checkout_003"]["median_cosine"],
                "policy_library_checkout_003_alone_mean": subset_rows["policy_library_checkout_003_alone"]["mean_cosine"],
                "policy_library_checkout_003_alone_median": subset_rows["policy_library_checkout_003_alone"]["median_cosine"],
            }
        )
    return output_rows


def pick_peak_row(
    rows: Sequence[Mapping[str, Any]],
    metric_field: str,
    subset_key: str,
    metric_name_field: str,
    metric_name_value: str,
) -> Mapping[str, Any]:
    candidates = [
        row
        for row in rows
        if str(row["subset_key"]) == subset_key and str(row[metric_name_field]) == metric_name_value
    ]
    if not candidates:
        raise ValueError(f"No rows found for subset={subset_key} and {metric_name_field}={metric_name_value}")
    return max(candidates, key=lambda row: float(row[metric_field]))


def build_summary_text(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    tensor_shape: Tuple[int, int],
    norm_rows: Sequence[Mapping[str, Any]],
    cosine_rows: Sequence[Mapping[str, Any]],
) -> str:
    n_layers, d_model = tensor_shape
    lines: List[str] = []
    lines.append("Qwen3 1.7B Pilot-8 Hidden-State Delta Summary")
    lines.append("")
    lines.append(f"families: {len(family_deltas)}")
    lines.append(f"hidden_state_shape: ({n_layers}, {d_model})")
    lines.append("")

    for subset_key, subset_label in SUBSET_SPECS:
        false_peak = pick_peak_row(
            norm_rows,
            metric_field="mean_delta_norm",
            subset_key=subset_key,
            metric_name_field="delta_type",
            metric_name_value="false_pressure_delta",
        )
        closed_peak = pick_peak_row(
            norm_rows,
            metric_field="mean_delta_norm",
            subset_key=subset_key,
            metric_name_field="delta_type",
            metric_name_value="closed_context_delta",
        )
        cosine_peak = pick_peak_row(
            cosine_rows,
            metric_field="mean_cosine",
            subset_key=subset_key,
            metric_name_field="cosine_pair",
            metric_name_value="cosine(false_pressure_delta, closed_context_delta)",
        )
        lines.append(subset_label)
        lines.append(
            "  "
            f"peak false_pressure_delta norm: layer {false_peak['layer_index']}, "
            f"mean={format_float(float(false_peak['mean_delta_norm']))}, "
            f"median={format_float(float(false_peak['median_delta_norm']))}"
        )
        lines.append(
            "  "
            f"peak closed_context_delta norm: layer {closed_peak['layer_index']}, "
            f"mean={format_float(float(closed_peak['mean_delta_norm']))}, "
            f"median={format_float(float(closed_peak['median_delta_norm']))}"
        )
        lines.append(
            "  "
            f"peak cosine(false_pressure_delta, closed_context_delta): layer {cosine_peak['layer_index']}, "
            f"mean={format_float(float(cosine_peak['mean_cosine']))}, "
            f"median={format_float(float(cosine_peak['median_cosine']))}"
        )
        lines.append("")

    lines.append("Interpretation Notes")
    lines.append("  Larger delta norms indicate larger hidden-state movement away from evidence_neutral at that layer.")
    lines.append("  Cosines near +1 indicate similar directional shifts; cosines near 0 indicate weak alignment; cosines near -1 indicate opposing shifts.")
    lines.append(
        "  The outlier-aware split lets you see whether policy_library_checkout_003 is driving the pilot-wide pattern or just amplifying it."
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
    input_path = repo_root / DEFAULT_INPUT
    delta_norms_output = repo_root / DEFAULT_DELTA_NORMS_OUTPUT
    delta_cosines_output = repo_root / DEFAULT_DELTA_COSINES_OUTPUT
    outlier_comparison_output = repo_root / DEFAULT_OUTLIER_COMPARISON_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    rows = read_jsonl(input_path)
    grouped_rows = group_rows_by_family(rows)
    family_deltas, tensor_shape = compute_family_deltas(repo_root, grouped_rows)
    n_layers, _d_model = tensor_shape

    norm_rows = build_layerwise_delta_norm_rows(family_deltas, n_layers)
    cosine_rows = build_layerwise_delta_cosine_rows(family_deltas, n_layers)
    outlier_rows = build_outlier_comparison_rows(norm_rows, cosine_rows)

    delta_norm_fieldnames = [
        "subset_key",
        "subset_label",
        "layer_index",
        "delta_type",
        "n_families",
        "mean_delta_norm",
        "median_delta_norm",
    ]
    delta_cosine_fieldnames = [
        "subset_key",
        "subset_label",
        "layer_index",
        "cosine_pair",
        "n_families",
        "mean_cosine",
        "median_cosine",
    ]
    outlier_fieldnames = [
        "analysis_kind",
        "metric_name",
        "layer_index",
        "all_8_families_mean",
        "all_8_families_median",
        "excluding_policy_library_checkout_003_mean",
        "excluding_policy_library_checkout_003_median",
        "policy_library_checkout_003_alone_mean",
        "policy_library_checkout_003_alone_median",
    ]

    write_csv(delta_norms_output, norm_rows, delta_norm_fieldnames)
    write_csv(delta_cosines_output, cosine_rows, delta_cosine_fieldnames)
    write_csv(outlier_comparison_output, outlier_rows, outlier_fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        build_summary_text(family_deltas, tensor_shape, norm_rows, cosine_rows),
        encoding="utf-8",
    )

    print(f"Wrote {delta_norms_output}")
    print(f"Wrote {delta_cosines_output}")
    print(f"Wrote {outlier_comparison_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
