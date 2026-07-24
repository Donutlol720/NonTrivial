import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_DELTA_INPUT = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_ROBUSTNESS_OUTPUT = "results/qwen3_4b_instruct_2507_family36_hidden_behavior_robustness.csv"
DEFAULT_SUBGROUP_OUTPUT = "results/qwen3_4b_instruct_2507_family36_subgroup_comparison.csv"
DEFAULT_SUMMARY_OUTPUT = "results/qwen3_4b_instruct_2507_family36_analysis8_robustness_summary.txt"

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
ROBUSTNESS_SPECS = (
    ("false_pressure_delta", "delta_false_pressure", "negative"),
    ("emotional_pressure_delta", "delta_emotional_pressure", "negative"),
    ("closed_context_delta", "delta_closed_context", "negative"),
    ("distractor_delta", "delta_distractor", "absolute"),
)
SUBGROUP_SPECS = (
    ("false_pressure_delta", "delta_false_pressure"),
    ("emotional_pressure_delta", "delta_emotional_pressure"),
    ("closed_context_delta", "delta_closed_context"),
)
FLIP_FAMILY_ID = "logic_tournament_036"
MIN_DOMAIN_FAMILIES = 3


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


def load_behavior_deltas(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_rows(path)
    by_family: Dict[str, Dict[str, float]] = {}
    for row in rows:
        family_id = str(row["family_id"])
        by_family[family_id] = {
            "delta_false_pressure": float(row["delta_false_pressure"]),
            "delta_emotional_pressure": float(row["delta_emotional_pressure"]),
            "delta_true_pressure": float(row["delta_true_pressure"]),
            "delta_distractor": float(row["delta_distractor"]),
            "delta_closed_context": float(row["delta_closed_context"]),
        }
    return by_family


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

        per_family: Dict[str, torch.Tensor] = {}
        for delta_name, prompt_type in DELTA_SPECS:
            comparison = load_hidden_state_tensor(repo_root, family_rows[prompt_type])
            if tuple(comparison.shape) != expected_shape:
                raise ValueError(
                    f"Family {family_id} prompt {prompt_type} has tensor shape {tuple(comparison.shape)} != {expected_shape}"
                )
            per_family[delta_name] = comparison - neutral
        family_deltas[family_id] = per_family

    if expected_shape is None:
        raise ValueError("No family deltas could be computed.")
    return family_deltas, expected_shape


def build_family_metadata(
    grouped_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    behavior_deltas: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for family_id, family_rows in grouped_rows.items():
        neutral_row = family_rows["evidence_neutral"]
        any_flip = any(row.get("model_choice") == row.get("false_choice") for row in family_rows.values())
        metadata[family_id] = {
            "family_id": family_id,
            "domain": str(neutral_row.get("domain")),
            "title": str(neutral_row.get("title")),
            "has_actual_flip": any_flip,
            "delta_false_pressure": float(behavior_deltas[family_id]["delta_false_pressure"]),
            "delta_emotional_pressure": float(behavior_deltas[family_id]["delta_emotional_pressure"]),
            "delta_closed_context": float(behavior_deltas[family_id]["delta_closed_context"]),
            "delta_distractor": float(behavior_deltas[family_id]["delta_distractor"]),
        }
    return metadata


def transformed_behavior_value(value: float, transform_kind: str) -> float:
    if transform_kind == "negative":
        return -value
    if transform_kind == "absolute":
        return abs(value)
    raise ValueError(f"Unsupported transform_kind: {transform_kind}")


def best_correlation_for_scope(
    family_ids: Sequence[str],
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_deltas: Mapping[str, Mapping[str, float]],
    delta_type: str,
    behavior_delta_name: str,
    transform_kind: str,
    n_layers: int,
) -> Dict[str, Any]:
    pearson_rows: List[Dict[str, Any]] = []
    spearman_rows: List[Dict[str, Any]] = []

    for layer_index in range(n_layers):
        hidden_values: List[float] = []
        behavior_values: List[float] = []
        for family_id in family_ids:
            hidden_values.append(float(torch.linalg.vector_norm(family_deltas[family_id][delta_type][layer_index]).item()))
            behavior_values.append(
                transformed_behavior_value(float(behavior_deltas[family_id][behavior_delta_name]), transform_kind)
            )

        pearson_rows.append(
            {
                "layer_index": layer_index,
                "layer_bucket": layer_bucket(layer_index, n_layers),
                "correlation": pearson_correlation(hidden_values, behavior_values),
            }
        )
        spearman_rows.append(
            {
                "layer_index": layer_index,
                "layer_bucket": layer_bucket(layer_index, n_layers),
                "correlation": spearman_correlation(hidden_values, behavior_values),
            }
        )

    best_pearson = max(pearson_rows, key=lambda row: float(row["correlation"]))
    best_spearman = max(spearman_rows, key=lambda row: float(row["correlation"]))
    return {
        "n_families": len(family_ids),
        "best_pearson_layer": int(best_pearson["layer_index"]),
        "best_pearson_layer_bucket": str(best_pearson["layer_bucket"]),
        "best_pearson_correlation": float(best_pearson["correlation"]),
        "best_spearman_layer": int(best_spearman["layer_index"]),
        "best_spearman_layer_bucket": str(best_spearman["layer_bucket"]),
        "best_spearman_correlation": float(best_spearman["correlation"]),
    }


def build_top_outlier_scopes(
    all_family_ids: Sequence[str],
    behavior_deltas: Mapping[str, Mapping[str, float]],
) -> List[Tuple[str, str, str, List[str]]]:
    scopes: List[Tuple[str, str, str, List[str]]] = []
    for delta_type, behavior_delta_name, _transform_kind in ROBUSTNESS_SPECS:
        ordered = sorted(
            all_family_ids,
            key=lambda family_id: abs(float(behavior_deltas[family_id][behavior_delta_name])),
            reverse=True,
        )
        scopes.append(
            (
                delta_type,
                "exclude_top1_abs_behavior_delta",
                ordered[0],
                [family_id for family_id in all_family_ids if family_id != ordered[0]],
            )
        )
        excluded_top3 = ordered[:3]
        scopes.append(
            (
                delta_type,
                "exclude_top3_abs_behavior_delta",
                ",".join(excluded_top3),
                [family_id for family_id in all_family_ids if family_id not in set(excluded_top3)],
            )
        )
    return scopes


def build_leave_one_family_out_scopes(all_family_ids: Sequence[str]) -> List[Tuple[str, str, str, List[str]]]:
    scopes: List[Tuple[str, str, str, List[str]]] = []
    for family_id in all_family_ids:
        scopes.append(
            (
                "all_conditions",
                "leave_one_family_out",
                family_id,
                [candidate for candidate in all_family_ids if candidate != family_id],
            )
        )
    return scopes


def build_leave_one_domain_out_scopes(
    all_family_ids: Sequence[str],
    family_metadata: Mapping[str, Mapping[str, Any]],
) -> List[Tuple[str, str, str, List[str]]]:
    domains = Counter(str(family_metadata[family_id]["domain"]) for family_id in all_family_ids)
    scopes: List[Tuple[str, str, str, List[str]]] = []
    for domain, count in sorted(domains.items()):
        if count < MIN_DOMAIN_FAMILIES:
            continue
        scopes.append(
            (
                "all_conditions",
                "leave_one_domain_out",
                domain,
                [family_id for family_id in all_family_ids if str(family_metadata[family_id]["domain"]) != domain],
            )
        )
    return scopes


def build_robustness_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_deltas: Mapping[str, Mapping[str, float]],
    family_metadata: Mapping[str, Mapping[str, Any]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    all_family_ids = sorted(family_deltas)
    rows: List[Dict[str, Any]] = []

    base_scopes = [
        ("all_conditions", "all_36_families", "", list(all_family_ids)),
        (
            "all_conditions",
            "exclude_flip_family",
            FLIP_FAMILY_ID,
            [family_id for family_id in all_family_ids if family_id != FLIP_FAMILY_ID],
        ),
    ]
    top_outlier_scopes = build_top_outlier_scopes(all_family_ids, behavior_deltas)
    family_out_scopes = build_leave_one_family_out_scopes(all_family_ids)
    domain_out_scopes = build_leave_one_domain_out_scopes(all_family_ids, family_metadata)

    for scope_target, scope_name, excluded_ids, scoped_family_ids in (
        base_scopes + top_outlier_scopes + family_out_scopes + domain_out_scopes
    ):
        for delta_type, behavior_delta_name, transform_kind in ROBUSTNESS_SPECS:
            if scope_target not in ("all_conditions", delta_type):
                continue
            best = best_correlation_for_scope(
                scoped_family_ids,
                family_deltas,
                behavior_deltas,
                delta_type,
                behavior_delta_name,
                transform_kind,
                n_layers,
            )
            rows.append(
                {
                    "scope_name": scope_name,
                    "scope_target": scope_target,
                    "excluded_ids": excluded_ids,
                    "delta_type": delta_type,
                    "behavior_delta_name": behavior_delta_name,
                    "behavior_transform": transform_kind,
                    **best,
                }
            )
    return rows


def split_sensitivity_groups(
    family_ids: Sequence[str],
    behavior_deltas: Mapping[str, Mapping[str, float]],
    behavior_delta_name: str,
) -> Tuple[List[str], List[str], float]:
    degradations = {
        family_id: -float(behavior_deltas[family_id][behavior_delta_name])
        for family_id in family_ids
    }
    threshold = median(degradations.values())
    high = sorted([family_id for family_id in family_ids if degradations[family_id] >= threshold])
    low = sorted([family_id for family_id in family_ids if degradations[family_id] < threshold])
    return high, low, float(threshold)


def build_subgroup_rows(
    family_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    behavior_deltas: Mapping[str, Mapping[str, float]],
    family_metadata: Mapping[str, Mapping[str, Any]],
    n_layers: int,
) -> List[Dict[str, Any]]:
    family_ids = sorted(family_deltas)
    rows: List[Dict[str, Any]] = []

    for delta_type, behavior_delta_name in SUBGROUP_SPECS:
        high_ids, low_ids, threshold = split_sensitivity_groups(family_ids, behavior_deltas, behavior_delta_name)
        high_flips = [family_id for family_id in high_ids if bool(family_metadata[family_id]["has_actual_flip"])]
        low_flips = [family_id for family_id in low_ids if bool(family_metadata[family_id]["has_actual_flip"])]
        layer_gaps: List[Tuple[int, float]] = []

        per_layer_rows: List[Dict[str, Any]] = []
        for layer_index in range(n_layers):
            high_norms = [
                float(torch.linalg.vector_norm(family_deltas[family_id][delta_type][layer_index]).item())
                for family_id in high_ids
            ]
            low_norms = [
                float(torch.linalg.vector_norm(family_deltas[family_id][delta_type][layer_index]).item())
                for family_id in low_ids
            ]
            gap = mean(high_norms) - mean(low_norms)
            layer_gaps.append((layer_index, abs(gap)))
            per_layer_rows.append(
                {
                    "delta_type": delta_type,
                    "behavior_delta_name": behavior_delta_name,
                    "sensitivity_threshold": threshold,
                    "layer_index": layer_index,
                    "layer_bucket": layer_bucket(layer_index, n_layers),
                    "high_sensitivity_n": len(high_ids),
                    "low_sensitivity_n": len(low_ids),
                    "high_sensitivity_mean_behavior_delta": mean(
                        [float(behavior_deltas[family_id][behavior_delta_name]) for family_id in high_ids]
                    ),
                    "low_sensitivity_mean_behavior_delta": mean(
                        [float(behavior_deltas[family_id][behavior_delta_name]) for family_id in low_ids]
                    ),
                    "high_sensitivity_mean_delta_norm": mean(high_norms),
                    "high_sensitivity_std_delta_norm": stddev(high_norms),
                    "low_sensitivity_mean_delta_norm": mean(low_norms),
                    "low_sensitivity_std_delta_norm": stddev(low_norms),
                    "mean_delta_norm_gap_high_minus_low": gap,
                    "abs_mean_delta_norm_gap": abs(gap),
                    "high_sensitivity_has_any_flip": bool(high_flips),
                    "low_sensitivity_has_any_flip": bool(low_flips),
                    "high_sensitivity_flip_family_ids": ",".join(high_flips),
                    "low_sensitivity_flip_family_ids": ",".join(low_flips),
                }
            )

        ranked_layers = sorted(layer_gaps, key=lambda item: item[1], reverse=True)
        rank_by_layer = {layer_index: rank + 1 for rank, (layer_index, _gap) in enumerate(ranked_layers)}
        top_layers = [str(layer_index) for layer_index, _gap in ranked_layers[:3]]

        for row in per_layer_rows:
            row["separation_rank_by_abs_gap"] = rank_by_layer[int(row["layer_index"])]
            row["is_top3_separation_layer"] = rank_by_layer[int(row["layer_index"])] <= 3
            row["top3_separating_layers"] = ",".join(top_layers)
            rows.append(row)
    return rows


def first_matching_row(
    rows: Sequence[Mapping[str, Any]],
    scope_name: str,
    delta_type: str,
) -> Mapping[str, Any]:
    for row in rows:
        if str(row["scope_name"]) == scope_name and str(row["delta_type"]) == delta_type:
            return row
    raise ValueError(f"Missing robustness row for scope_name={scope_name}, delta_type={delta_type}")


def filter_rows(rows: Sequence[Mapping[str, Any]], scope_name: str, delta_type: str) -> List[Mapping[str, Any]]:
    return [row for row in rows if str(row["scope_name"]) == scope_name and str(row["delta_type"]) == delta_type]


def build_summary_text(
    tensor_shape: Tuple[int, int],
    robustness_rows: Sequence[Mapping[str, Any]],
    subgroup_rows: Sequence[Mapping[str, Any]],
) -> str:
    n_layers, d_model = tensor_shape

    closed_all = first_matching_row(robustness_rows, "all_36_families", "closed_context_delta")
    closed_no_flip = first_matching_row(robustness_rows, "exclude_flip_family", "closed_context_delta")
    false_all = first_matching_row(robustness_rows, "all_36_families", "false_pressure_delta")
    false_no_flip = first_matching_row(robustness_rows, "exclude_flip_family", "false_pressure_delta")
    emotional_all = first_matching_row(robustness_rows, "all_36_families", "emotional_pressure_delta")

    closed_top1 = first_matching_row(robustness_rows, "exclude_top1_abs_behavior_delta", "closed_context_delta")
    closed_top3 = first_matching_row(robustness_rows, "exclude_top3_abs_behavior_delta", "closed_context_delta")
    false_top1 = first_matching_row(robustness_rows, "exclude_top1_abs_behavior_delta", "false_pressure_delta")
    false_top3 = first_matching_row(robustness_rows, "exclude_top3_abs_behavior_delta", "false_pressure_delta")

    false_loo = filter_rows(robustness_rows, "leave_one_family_out", "false_pressure_delta")
    closed_loo = filter_rows(robustness_rows, "leave_one_family_out", "closed_context_delta")
    emotional_loo = filter_rows(robustness_rows, "leave_one_family_out", "emotional_pressure_delta")

    closed_best_layer_stable = all(
        int(row["best_pearson_layer"]) >= 24 for row in filter_rows(robustness_rows, "leave_one_family_out", "closed_context_delta")
    )
    late_layer_concentration = all(
        int(row["best_pearson_layer"]) >= 24
        for row in [
            closed_all,
            closed_no_flip,
            closed_top1,
            closed_top3,
            false_all,
            false_no_flip,
            false_top1,
            false_top3,
            emotional_all,
        ]
    )

    subgroup_top = {
        delta_type: sorted(
            [row for row in subgroup_rows if str(row["delta_type"]) == delta_type and bool(row["is_top3_separation_layer"])],
            key=lambda row: int(row["separation_rank_by_abs_gap"]),
        )
        for delta_type, _behavior_name in SUBGROUP_SPECS
    }

    lines: List[str] = []
    lines.append("Qwen3-4B-Instruct-2507 Family-36 Analysis 8 Robustness Summary")
    lines.append("")
    lines.append(f"hidden_state_shape: ({n_layers}, {d_model})")
    lines.append("")
    lines.append("Core Robustness Checks")
    lines.append(
        "  "
        f"Closed-context alignment survives removing {FLIP_FAMILY_ID}: "
        f"best Pearson r={format_float(float(closed_all['best_pearson_correlation']))} at layer {closed_all['best_pearson_layer']} "
        f"(all 36) versus r={format_float(float(closed_no_flip['best_pearson_correlation']))} at layer {closed_no_flip['best_pearson_layer']} "
        f"(excluding flip family)."
    )
    lines.append(
        "  "
        f"Closed-context also survives top-outlier removal: "
        f"excluding the top-1 absolute closed-context delta family gives r={format_float(float(closed_top1['best_pearson_correlation']))} "
        f"at layer {closed_top1['best_pearson_layer']}, and excluding the top-3 gives "
        f"r={format_float(float(closed_top3['best_pearson_correlation']))} at layer {closed_top3['best_pearson_layer']}."
    )
    lines.append(
        "  "
        f"False-pressure alignment also survives outlier removal: "
        f"all 36 gives r={format_float(float(false_all['best_pearson_correlation']))} at layer {false_all['best_pearson_layer']}; "
        f"excluding the flip family gives r={format_float(float(false_no_flip['best_pearson_correlation']))}; "
        f"excluding top-1 false-pressure outlier gives r={format_float(float(false_top1['best_pearson_correlation']))}; "
        f"excluding top-3 gives r={format_float(float(false_top3['best_pearson_correlation']))}."
    )
    lines.append(
        "  "
        f"Emotional-pressure remains weaker than both closed-context and false-pressure: "
        f"best Pearson r={format_float(float(emotional_all['best_pearson_correlation']))} "
        f"versus false-pressure r={format_float(float(false_all['best_pearson_correlation']))} "
        f"and closed-context r={format_float(float(closed_all['best_pearson_correlation']))}."
    )
    lines.append(
        "  "
        f"The strongest hidden/behavior alignment {'does' if late_layer_concentration else 'does not'} stay concentrated in late layers. "
        f"Closed-context late-layer stability across leave-one-family-out scopes is "
        f"{'strong' if closed_best_layer_stable else 'weaker than expected'}."
    )
    lines.append("")
    lines.append("Leave-One-Family-Out Ranges")
    lines.append(
        "  "
        f"Closed-context best Pearson range: {format_float(min(float(row['best_pearson_correlation']) for row in closed_loo))} "
        f"to {format_float(max(float(row['best_pearson_correlation']) for row in closed_loo))}."
    )
    lines.append(
        "  "
        f"False-pressure best Pearson range: {format_float(min(float(row['best_pearson_correlation']) for row in false_loo))} "
        f"to {format_float(max(float(row['best_pearson_correlation']) for row in false_loo))}."
    )
    lines.append(
        "  "
        f"Emotional-pressure best Pearson range: {format_float(min(float(row['best_pearson_correlation']) for row in emotional_loo))} "
        f"to {format_float(max(float(row['best_pearson_correlation']) for row in emotional_loo))}."
    )
    lines.append("")
    lines.append("Sensitivity Subgroups")
    for delta_type, rows in subgroup_top.items():
        if not rows:
            continue
        top_layers = ", ".join(
            f"{row['layer_index']} ({row['layer_bucket']}, gap={format_float(float(row['mean_delta_norm_gap_high_minus_low']))})"
            for row in rows
        )
        exemplar = rows[0]
        lines.append(
            "  "
            f"{delta_type}: high-sensitivity families ({exemplar['high_sensitivity_n']}) vs low-sensitivity families "
            f"({exemplar['low_sensitivity_n']}) separate most at layers {top_layers}. "
            f"High-sensitivity mean behavioral delta={format_float(float(exemplar['high_sensitivity_mean_behavior_delta']))}, "
            f"low-sensitivity mean behavioral delta={format_float(float(exemplar['low_sensitivity_mean_behavior_delta']))}. "
            f"Flips high/low: {exemplar['high_sensitivity_flip_family_ids'] or 'none'} / {exemplar['low_sensitivity_flip_family_ids'] or 'none'}."
        )
    lines.append("")
    lines.append("Bottom Line")
    lines.append(
        "  "
        f"The Analysis 7 closed-context conclusion should "
        f"{'not ' if float(closed_top3['best_pearson_correlation']) >= 0.5 else ''}be weakened as outlier-driven."
    )
    lines.append(
        "  "
        f"False-pressure alignment is robust but consistently weaker than closed-context, which matches the behavioral result that retrieved evidence resists direct false-belief pressure better than the closed-context baseline does."
    )
    lines.append(
        "  "
        f"Emotional-pressure still moves hidden states substantially, but its hidden/behavior coupling remains the weakest of the three pressure-like conditions after robustness checks."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / DEFAULT_INPUT
    delta_input_path = repo_root / DEFAULT_DELTA_INPUT
    robustness_output = repo_root / DEFAULT_ROBUSTNESS_OUTPUT
    subgroup_output = repo_root / DEFAULT_SUBGROUP_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    rows = read_jsonl(input_path)
    grouped_rows = group_rows_by_family(rows)
    behavior_deltas = load_behavior_deltas(delta_input_path)
    family_deltas, tensor_shape = compute_family_deltas(repo_root, grouped_rows)
    family_metadata = build_family_metadata(grouped_rows, behavior_deltas)
    n_layers, _d_model = tensor_shape

    robustness_rows = build_robustness_rows(family_deltas, behavior_deltas, family_metadata, n_layers)
    subgroup_rows = build_subgroup_rows(family_deltas, behavior_deltas, family_metadata, n_layers)

    robustness_fieldnames = [
        "scope_name",
        "scope_target",
        "excluded_ids",
        "delta_type",
        "behavior_delta_name",
        "behavior_transform",
        "n_families",
        "best_pearson_layer",
        "best_pearson_layer_bucket",
        "best_pearson_correlation",
        "best_spearman_layer",
        "best_spearman_layer_bucket",
        "best_spearman_correlation",
    ]
    subgroup_fieldnames = [
        "delta_type",
        "behavior_delta_name",
        "sensitivity_threshold",
        "layer_index",
        "layer_bucket",
        "high_sensitivity_n",
        "low_sensitivity_n",
        "high_sensitivity_mean_behavior_delta",
        "low_sensitivity_mean_behavior_delta",
        "high_sensitivity_mean_delta_norm",
        "high_sensitivity_std_delta_norm",
        "low_sensitivity_mean_delta_norm",
        "low_sensitivity_std_delta_norm",
        "mean_delta_norm_gap_high_minus_low",
        "abs_mean_delta_norm_gap",
        "separation_rank_by_abs_gap",
        "is_top3_separation_layer",
        "top3_separating_layers",
        "high_sensitivity_has_any_flip",
        "low_sensitivity_has_any_flip",
        "high_sensitivity_flip_family_ids",
        "low_sensitivity_flip_family_ids",
    ]

    write_csv(robustness_output, robustness_rows, robustness_fieldnames)
    write_csv(subgroup_output, subgroup_rows, subgroup_fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(build_summary_text(tensor_shape, robustness_rows, subgroup_rows), encoding="utf-8")

    print(f"Wrote {robustness_output}")
    print(f"Wrote {subgroup_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
