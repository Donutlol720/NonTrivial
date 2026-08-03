import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


DEFAULT_INPUT = "outputs/state_logits_qwen3_1_7b_subset.jsonl"
DEFAULT_BEHAVIOR_DELTA_INPUT = "results/qwen3_1_7b_pilot8_family_margin_deltas.csv"
DEFAULT_ROBUSTNESS_OUTPUT = "results/qwen3_1_7b_pilot8_hidden_behavior_correlation_robustness.csv"
DEFAULT_SUMMARY_OUTPUT = "results/qwen3_1_7b_pilot8_phase3e_robustness_summary.txt"

OUTLIER_FAMILY_ID = "policy_library_checkout_003"
EXPECTED_PROMPT_TYPES = (
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
)
ALIGNMENT_SPECS = (
    ("false_pressure_delta_norm", "delta_false_pressure", "false_pressure_delta"),
    ("closed_context_delta_norm", "delta_closed_context", "closed_context_delta"),
    ("distractor_delta_norm", "delta_distractor", "distractor_delta"),
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


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        start = position
        current_value = indexed[position][1]
        while position < len(indexed) and indexed[position][1] == current_value:
            position += 1
        average_rank = (start + 1 + position) / 2.0
        for original_index, _value in indexed[start:position]:
            ranks[original_index] = average_rank
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("spearman_correlation requires equal-length inputs.")
    if len(xs) < 2:
        return 0.0
    return pearson_correlation(rankdata(xs), rankdata(ys))


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
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], int]:
    family_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
    expected_n_layers: int | None = None

    for family_id in sorted(grouped_rows):
        family_rows = grouped_rows[family_id]
        missing = [prompt_type for prompt_type in EXPECTED_PROMPT_TYPES if prompt_type not in family_rows]
        if missing:
            raise ValueError(f"Family {family_id} is missing prompt types: {missing}")

        neutral = load_hidden_state_tensor(repo_root, family_rows["evidence_neutral"])
        if expected_n_layers is None:
            expected_n_layers = int(neutral.shape[0])
        elif int(neutral.shape[0]) != expected_n_layers:
            raise ValueError(f"Family {family_id} has inconsistent layer count.")

        family_deltas[family_id] = {
            "false_pressure_delta": load_hidden_state_tensor(repo_root, family_rows["evidence_false_belief_pressure"]) - neutral,
            "closed_context_delta": load_hidden_state_tensor(repo_root, family_rows["closed_context_false_belief_pressure"]) - neutral,
            "distractor_delta": load_hidden_state_tensor(repo_root, family_rows["evidence_distractor_neutral"]) - neutral,
        }

    if expected_n_layers is None:
        raise ValueError("No families available for robustness analysis.")
    return family_deltas, expected_n_layers


def load_behavior_deltas(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_rows(path)
    behavior_by_family: Dict[str, Dict[str, float]] = {}
    for row in rows:
        family_id = str(row["family_id"])
        behavior_by_family[family_id] = {
            "delta_false_pressure": abs(float(row["delta_false_pressure"])),
            "delta_closed_context": abs(float(row["delta_closed_context"])),
            "delta_distractor": abs(float(row["delta_distractor"])),
        }
    return behavior_by_family


def build_scope_specs(family_ids: Sequence[str]) -> List[Dict[str, Any]]:
    scopes: List[Dict[str, Any]] = [
        {
            "scope_type": "all_8_families",
            "scope_label": "all 8 families",
            "excluded_family_id": "",
            "family_ids": list(family_ids),
        },
        {
            "scope_type": "excluding_policy_library_checkout_003",
            "scope_label": "excluding policy_library_checkout_003",
            "excluded_family_id": OUTLIER_FAMILY_ID,
            "family_ids": [family_id for family_id in family_ids if family_id != OUTLIER_FAMILY_ID],
        },
    ]
    for family_id in family_ids:
        scopes.append(
            {
                "scope_type": "leave_one_family_out",
                "scope_label": f"leave out {family_id}",
                "excluded_family_id": family_id,
                "family_ids": [other_family_id for other_family_id in family_ids if other_family_id != family_id],
            }
        )
    return scopes


def build_robustness_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_deltas: Mapping[str, Mapping[str, float]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    family_ids = sorted(family_deltas)
    scope_specs = build_scope_specs(family_ids)
    rows: List[Dict[str, Any]] = []

    for layer_index in range(n_layers):
        for hidden_metric_name, behavior_metric_name, delta_key in ALIGNMENT_SPECS:
            for scope in scope_specs:
                included_family_ids = list(scope["family_ids"])
                hidden_values: List[float] = []
                behavior_values: List[float] = []
                for family_id in included_family_ids:
                    if family_id not in behavior_deltas:
                        raise ValueError(f"Missing behavior delta row for family {family_id}")
                    vector = family_deltas[family_id][delta_key][layer_index]
                    hidden_values.append(float(torch.linalg.vector_norm(vector).item()))
                    behavior_values.append(float(behavior_deltas[family_id][behavior_metric_name]))

                rows.append(
                    {
                        "layer_index": layer_index,
                        "hidden_metric": hidden_metric_name,
                        "behavior_metric": f"abs({behavior_metric_name})",
                        "scope_type": scope["scope_type"],
                        "scope_label": scope["scope_label"],
                        "excluded_family_id": scope["excluded_family_id"],
                        "n_families": len(included_family_ids),
                        "pearson_correlation": pearson_correlation(hidden_values, behavior_values),
                        "spearman_correlation": spearman_correlation(hidden_values, behavior_values),
                        "mean_hidden_metric": mean(hidden_values),
                        "mean_abs_behavior_metric": mean(behavior_values),
                    }
                )
    return rows


def pick_peak(rows: Sequence[Mapping[str, Any]], hidden_metric: str, scope_type: str) -> Mapping[str, Any]:
    matching_rows = [
        row
        for row in rows
        if str(row["hidden_metric"]) == hidden_metric and str(row["scope_type"]) == scope_type
    ]
    if not matching_rows:
        raise ValueError(f"No rows found for hidden_metric={hidden_metric}, scope_type={scope_type}")
    return max(matching_rows, key=lambda row: float(row["pearson_correlation"]))


def build_summary_text(rows: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("Qwen3 1.7B Pilot-8 Phase 3E.5 Robustness Summary")
    lines.append("")

    for hidden_metric_name, behavior_metric_name, _delta_key in ALIGNMENT_SPECS:
        all_peak = pick_peak(rows, hidden_metric_name, "all_8_families")
        no_outlier_peak = pick_peak(rows, hidden_metric_name, "excluding_policy_library_checkout_003")
        loo_rows = [
            row
            for row in rows
            if str(row["hidden_metric"]) == hidden_metric_name
            and str(row["scope_type"]) == "leave_one_family_out"
            and int(row["layer_index"]) == int(all_peak["layer_index"])
        ]
        min_loo_pearson = min(float(row["pearson_correlation"]) for row in loo_rows)
        max_loo_pearson = max(float(row["pearson_correlation"]) for row in loo_rows)
        min_loo_spearman = min(float(row["spearman_correlation"]) for row in loo_rows)
        max_loo_spearman = max(float(row["spearman_correlation"]) for row in loo_rows)

        lines.append(f"{hidden_metric_name} vs abs({behavior_metric_name})")
        lines.append(
            "  "
            f"best all-8 Pearson: layer {all_peak['layer_index']}, "
            f"r={format_float(float(all_peak['pearson_correlation']))}, "
            f"Spearman={format_float(float(all_peak['spearman_correlation']))}"
        )
        lines.append(
            "  "
            f"best excluding {OUTLIER_FAMILY_ID}: layer {no_outlier_peak['layer_index']}, "
            f"r={format_float(float(no_outlier_peak['pearson_correlation']))}, "
            f"Spearman={format_float(float(no_outlier_peak['spearman_correlation']))}"
        )
        lines.append(
            "  "
            f"leave-one-family-out range at the all-8 peak layer {all_peak['layer_index']}: "
            f"Pearson [{format_float(min_loo_pearson)}, {format_float(max_loo_pearson)}], "
            f"Spearman [{format_float(min_loo_spearman)}, {format_float(max_loo_spearman)}]"
        )
        lines.append("")

    lines.append("Interpretation")
    lines.append(
        "  If the excluding-outlier and leave-one-family-out correlations stay positive and reasonably sized, "
        "the alignment pattern is broad across the pilot rather than being mostly driven by policy_library_checkout_003."
    )
    lines.append(
        "  If correlations collapse or change sign after removing policy_library_checkout_003, the apparent alignment is likely fragile."
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
    behavior_delta_path = repo_root / DEFAULT_BEHAVIOR_DELTA_INPUT
    robustness_output = repo_root / DEFAULT_ROBUSTNESS_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    rows = read_jsonl(input_path)
    grouped_rows = group_rows_by_family(rows)
    family_deltas, n_layers = compute_family_deltas(repo_root, grouped_rows)
    behavior_deltas = load_behavior_deltas(behavior_delta_path)
    robustness_rows = build_robustness_rows(family_deltas, behavior_deltas, n_layers)

    fieldnames = [
        "layer_index",
        "hidden_metric",
        "behavior_metric",
        "scope_type",
        "scope_label",
        "excluded_family_id",
        "n_families",
        "pearson_correlation",
        "spearman_correlation",
        "mean_hidden_metric",
        "mean_abs_behavior_metric",
    ]
    write_csv(robustness_output, robustness_rows, fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(build_summary_text(robustness_rows), encoding="utf-8")

    print(f"Wrote {robustness_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
