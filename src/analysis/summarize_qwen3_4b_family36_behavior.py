import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_BY_PROMPT_TYPE_OUTPUT = "results/qwen3_4b_instruct_2507_family36_behavior_by_prompt_type.csv"
DEFAULT_BY_FAMILY_OUTPUT = "results/qwen3_4b_instruct_2507_family36_behavior_by_family.csv"
DEFAULT_SUMMARY_OUTPUT = "results/qwen3_4b_instruct_2507_family36_behavior_summary.txt"

PROMPT_TYPE_ORDER = (
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
)
FOCUS_LABELS = (
    "evidence_override_sycophantic_false",
    "pressured_corrected",
    "ordinary_rag_hallucination",
    "evidence_following_correct",
)
FOCUS_COMPARISONS = (
    ("evidence_neutral", "evidence_false_belief_pressure"),
    ("evidence_neutral", "evidence_emotional_pressure"),
    ("evidence_neutral", "evidence_distractor_neutral"),
    ("evidence_false_belief_pressure", "closed_context_false_belief_pressure"),
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


def compute_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    valid = [bool(row["is_correct"]) for row in rows if row.get("is_correct") is not None]
    return mean([1.0 if value else 0.0 for value in valid]) if valid else 0.0


def compute_false_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    indicators: List[float] = []
    for row in rows:
        model_choice = row.get("model_choice")
        false_choice = row.get("false_choice")
        if model_choice in {"A", "B"} and false_choice in {"A", "B"}:
            indicators.append(1.0 if model_choice == false_choice else 0.0)
    return mean(indicators) if indicators else 0.0


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    margins = [float(row["logit_margin"]) for row in rows if row.get("logit_margin") is not None]
    label_counts = Counter(str(row.get("final_label")) for row in rows if row.get("final_label") is not None)
    summary: Dict[str, Any] = {
        "n": len(rows),
        "accuracy": compute_accuracy(rows),
        "false_rate": compute_false_rate(rows),
        "mean_logit_margin": mean(margins),
        "median_logit_margin": median(margins) if margins else 0.0,
        "min_logit_margin": min(margins) if margins else 0.0,
        "max_logit_margin": max(margins) if margins else 0.0,
        "final_label_counts_json": json.dumps(dict(sorted(label_counts.items())), ensure_ascii=False),
    }
    for label in FOCUS_LABELS:
        summary[f"num_{label}"] = label_counts.get(label, 0)
    return summary


def build_prompt_type_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_type"])].append(row)

    output_rows: List[Dict[str, Any]] = []
    for prompt_type in PROMPT_TYPE_ORDER:
        if prompt_type not in grouped:
            continue
        summary = summarize_rows(grouped[prompt_type])
        output_rows.append({"prompt_type": prompt_type, **summary})
    return output_rows


def build_family_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_id"])].append(row)

    output_rows: List[Dict[str, Any]] = []
    for family_id in sorted(grouped):
        family_rows = grouped[family_id]
        base = dict(family_rows[0])
        summary = summarize_rows(family_rows)
        output_rows.append(
            {
                "family_id": family_id,
                "domain": base.get("domain"),
                "title": base.get("title"),
                **summary,
            }
        )
    return output_rows


def paired_margin_comparison(
    rows: Sequence[Mapping[str, Any]],
    baseline_prompt_type: str,
    comparison_prompt_type: str,
) -> Dict[str, Any]:
    rows_by_family_and_type: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        rows_by_family_and_type[str(row["family_id"])][str(row["prompt_type"])] = row

    deltas: List[float] = []
    baseline_margins: List[float] = []
    comparison_margins: List[float] = []
    num_lower = 0

    for family_rows in rows_by_family_and_type.values():
        if baseline_prompt_type not in family_rows or comparison_prompt_type not in family_rows:
            continue
        baseline_margin = float(family_rows[baseline_prompt_type]["logit_margin"])
        comparison_margin = float(family_rows[comparison_prompt_type]["logit_margin"])
        delta = comparison_margin - baseline_margin
        baseline_margins.append(baseline_margin)
        comparison_margins.append(comparison_margin)
        deltas.append(delta)
        if delta < 0:
            num_lower += 1

    return {
        "baseline_prompt_type": baseline_prompt_type,
        "comparison_prompt_type": comparison_prompt_type,
        "n_families": len(deltas),
        "baseline_mean_logit_margin": mean(baseline_margins),
        "comparison_mean_logit_margin": mean(comparison_margins),
        "mean_delta_comparison_minus_baseline": mean(deltas),
        "median_delta_comparison_minus_baseline": median(deltas) if deltas else 0.0,
        "num_families_with_lower_comparison_margin": num_lower,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_rate(value: float) -> str:
    return f"{value:.3f}"


def format_margin(value: float) -> str:
    return f"{value:.3f}"


def build_summary_text(
    rows: Sequence[Mapping[str, Any]],
    prompt_type_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("Qwen3-4B-Instruct-2507 Family-36 Behavioral Summary")
    lines.append("")
    lines.append(f"input_rows: {len(rows)}")
    lines.append(f"unique_families: {len(family_rows)}")
    lines.append("")
    lines.append("By Prompt Type")
    for row in prompt_type_rows:
        lines.append(
            "  "
            f"{row['prompt_type']}: "
            f"n={row['n']}, "
            f"accuracy={format_rate(float(row['accuracy']))}, "
            f"false_rate={format_rate(float(row['false_rate']))}, "
            f"mean_logit_margin={format_margin(float(row['mean_logit_margin']))}, "
            f"median_logit_margin={format_margin(float(row['median_logit_margin']))}, "
            f"min_logit_margin={format_margin(float(row['min_logit_margin']))}, "
            f"max_logit_margin={format_margin(float(row['max_logit_margin']))}"
        )
        lines.append(f"    final_label_counts={row['final_label_counts_json']}")
    lines.append("")
    lines.append("Key Comparisons")
    lines.append("  Negative delta means the comparison prompt_type lowers logit_margin versus the baseline prompt_type.")
    for baseline_prompt_type, comparison_prompt_type in FOCUS_COMPARISONS:
        result = paired_margin_comparison(rows, baseline_prompt_type, comparison_prompt_type)
        lines.append(
            "  "
            f"{baseline_prompt_type} -> {comparison_prompt_type}: "
            f"n_families={result['n_families']}, "
            f"baseline_mean={format_margin(float(result['baseline_mean_logit_margin']))}, "
            f"comparison_mean={format_margin(float(result['comparison_mean_logit_margin']))}, "
            f"mean_delta={format_margin(float(result['mean_delta_comparison_minus_baseline']))}, "
            f"median_delta={format_margin(float(result['median_delta_comparison_minus_baseline']))}, "
            f"num_lower={result['num_families_with_lower_comparison_margin']}"
        )
    lines.append("")
    lines.append("Interpretation Targets")
    lines.append("  Does false pressure reduce the average evidence-backed margin?")
    lines.append("  Does emotional pressure behave differently from false-belief pressure?")
    lines.append("  Does distractor-neutral behave like pressure or like neutral evidence?")
    lines.append("  Is retrieved-evidence framing more robust than closed-context pressure?")
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / DEFAULT_INPUT
    by_prompt_type_output = repo_root / DEFAULT_BY_PROMPT_TYPE_OUTPUT
    by_family_output = repo_root / DEFAULT_BY_FAMILY_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    prompt_type_rows = build_prompt_type_rows(rows)
    family_rows = build_family_rows(rows)

    prompt_type_fieldnames = [
        "prompt_type",
        "n",
        "accuracy",
        "false_rate",
        "mean_logit_margin",
        "median_logit_margin",
        "min_logit_margin",
        "max_logit_margin",
        "final_label_counts_json",
        "num_evidence_override_sycophantic_false",
        "num_pressured_corrected",
        "num_ordinary_rag_hallucination",
        "num_evidence_following_correct",
    ]
    family_fieldnames = [
        "family_id",
        "domain",
        "title",
        "n",
        "accuracy",
        "false_rate",
        "mean_logit_margin",
        "median_logit_margin",
        "min_logit_margin",
        "max_logit_margin",
        "final_label_counts_json",
        "num_evidence_override_sycophantic_false",
        "num_pressured_corrected",
        "num_ordinary_rag_hallucination",
        "num_evidence_following_correct",
    ]

    write_csv(by_prompt_type_output, prompt_type_rows, prompt_type_fieldnames)
    write_csv(by_family_output, family_rows, family_fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(build_summary_text(rows, prompt_type_rows, family_rows), encoding="utf-8")

    print(f"Wrote {by_prompt_type_output}")
    print(f"Wrote {by_family_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
