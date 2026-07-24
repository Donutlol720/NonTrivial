import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_PER_FAMILY_OUTPUT = "results/qwen3_4b_instruct_2507_family36_per_family_layerwise_delta_norms.csv"
DEFAULT_LAYERWISE_OUTPUT = "results/qwen3_4b_instruct_2507_family36_layerwise_delta_norms.csv"
DEFAULT_SUMMARY_OUTPUT = "results/qwen3_4b_instruct_2507_family36_hidden_state_delta_norm_summary.txt"

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


def compute_per_family_norm_rows(
    repo_root: Path,
    grouped_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    per_family_rows: List[Dict[str, Any]] = []
    expected_shape: Tuple[int, int] | None = None

    for family_id in sorted(grouped_rows):
        family_rows = grouped_rows[family_id]
        missing = [prompt_type for prompt_type in EXPECTED_PROMPT_TYPES if prompt_type not in family_rows]
        if missing:
            raise ValueError(f"Family {family_id} is missing prompt types: {missing}")

        neutral_row = family_rows["evidence_neutral"]
        neutral = load_hidden_state_tensor(repo_root, neutral_row)
        if expected_shape is None:
            expected_shape = (int(neutral.shape[0]), int(neutral.shape[1]))
        elif tuple(neutral.shape) != expected_shape:
            raise ValueError(
                f"Family {family_id} has inconsistent neutral tensor shape {tuple(neutral.shape)} != {expected_shape}"
            )

        base_meta = {
            "family_id": family_id,
            "domain": neutral_row.get("domain"),
            "title": neutral_row.get("title"),
        }
        for delta_name, comparison_prompt_type in DELTA_SPECS:
            comparison = load_hidden_state_tensor(repo_root, family_rows[comparison_prompt_type])
            if tuple(comparison.shape) != expected_shape:
                raise ValueError(
                    f"Family {family_id} prompt {comparison_prompt_type} has tensor shape {tuple(comparison.shape)} != {expected_shape}"
                )
            delta = comparison - neutral
            for layer_index in range(expected_shape[0]):
                delta_norm = float(torch.linalg.vector_norm(delta[layer_index]).item())
                per_family_rows.append(
                    {
                        **base_meta,
                        "layer_index": layer_index,
                        "delta_type": delta_name,
                        "comparison_prompt_type": comparison_prompt_type,
                        "delta_norm": delta_norm,
                    }
                )

    if expected_shape is None:
        raise ValueError("No families found for delta-norm analysis.")
    return per_family_rows, expected_shape


def build_layerwise_rows(per_family_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for row in per_family_rows:
        grouped[(int(row["layer_index"]), str(row["delta_type"]))].append(float(row["delta_norm"]))

    output_rows: List[Dict[str, Any]] = []
    for layer_index, delta_type in sorted(grouped):
        values = grouped[(layer_index, delta_type)]
        output_rows.append(
            {
                "layer_index": layer_index,
                "delta_type": delta_type,
                "n_families": len(values),
                "mean_delta_norm": mean(values),
                "median_delta_norm": median(values) if values else 0.0,
                "min_delta_norm": min(values) if values else 0.0,
                "max_delta_norm": max(values) if values else 0.0,
            }
        )
    return output_rows


def pick_peak_row(
    rows: Sequence[Mapping[str, Any]],
    delta_type: str,
) -> Mapping[str, Any]:
    candidates = [row for row in rows if str(row["delta_type"]) == delta_type]
    if not candidates:
        raise ValueError(f"No rows found for delta_type={delta_type}")
    return max(candidates, key=lambda row: float(row["mean_delta_norm"]))


def build_summary_text(
    tensor_shape: Tuple[int, int],
    per_family_rows: Sequence[Mapping[str, Any]],
    layerwise_rows: Sequence[Mapping[str, Any]],
) -> str:
    n_layers, d_model = tensor_shape
    family_ids = sorted({str(row["family_id"]) for row in per_family_rows})
    lines: List[str] = []
    lines.append("Qwen3-4B-Instruct-2507 Family-36 Hidden-State Delta Norm Summary")
    lines.append("")
    lines.append(f"families: {len(family_ids)}")
    lines.append(f"hidden_state_shape: ({n_layers}, {d_model})")
    lines.append("")
    lines.append("Peak Mean Delta Norm By Condition")
    for delta_name, _comparison_prompt_type in DELTA_SPECS:
        peak_row = pick_peak_row(layerwise_rows, delta_name)
        lines.append(
            "  "
            f"{delta_name}: layer {peak_row['layer_index']}, "
            f"mean={format_float(float(peak_row['mean_delta_norm']))}, "
            f"median={format_float(float(peak_row['median_delta_norm']))}, "
            f"min={format_float(float(peak_row['min_delta_norm']))}, "
            f"max={format_float(float(peak_row['max_delta_norm']))}"
        )
    lines.append("")
    lines.append("Interpretation")
    lines.append("  Larger delta norms mean larger hidden-state movement away from the neutral evidence prompt at that layer.")
    lines.append("  This analysis is descriptive only; it does not ask whether the movement is aligned or behaviorally predictive.")
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
    per_family_output = repo_root / DEFAULT_PER_FAMILY_OUTPUT
    layerwise_output = repo_root / DEFAULT_LAYERWISE_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    rows = read_jsonl(input_path)
    grouped_rows = group_rows_by_family(rows)
    per_family_rows, tensor_shape = compute_per_family_norm_rows(repo_root, grouped_rows)
    layerwise_rows = build_layerwise_rows(per_family_rows)

    per_family_fieldnames = [
        "family_id",
        "domain",
        "title",
        "layer_index",
        "delta_type",
        "comparison_prompt_type",
        "delta_norm",
    ]
    layerwise_fieldnames = [
        "layer_index",
        "delta_type",
        "n_families",
        "mean_delta_norm",
        "median_delta_norm",
        "min_delta_norm",
        "max_delta_norm",
    ]

    write_csv(per_family_output, per_family_rows, per_family_fieldnames)
    write_csv(layerwise_output, layerwise_rows, layerwise_fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(build_summary_text(tensor_shape, per_family_rows, layerwise_rows), encoding="utf-8")

    print(f"Wrote {per_family_output}")
    print(f"Wrote {layerwise_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
