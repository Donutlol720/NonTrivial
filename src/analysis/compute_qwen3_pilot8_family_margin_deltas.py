import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_INPUT = "outputs/state_logits_qwen3_1_7b_subset.jsonl"
DEFAULT_CSV_OUTPUT = "results/qwen3_1_7b_pilot8_family_margin_deltas.csv"
DEFAULT_SUMMARY_OUTPUT = "results/qwen3_1_7b_pilot8_family_margin_delta_summary.txt"

EXPECTED_PROMPT_TYPES = (
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
)
DELTA_SPECS = (
    ("delta_false_pressure", "evidence_false_belief_pressure"),
    ("delta_emotional_pressure", "evidence_emotional_pressure"),
    ("delta_true_pressure", "evidence_true_belief_pressure"),
    ("delta_distractor", "evidence_distractor_neutral"),
    ("delta_closed_context", "closed_context_false_belief_pressure"),
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
    return f"{value:.3f}"


def build_family_delta_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    family_meta: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        family_id = str(row["family_id"])
        prompt_type = str(row["prompt_type"])
        grouped[family_id][prompt_type] = row
        family_meta.setdefault(
            family_id,
            {
                "family_id": family_id,
                "domain": row.get("domain"),
                "title": row.get("title"),
            },
        )

    output_rows: List[Dict[str, Any]] = []
    for family_id in sorted(grouped):
        family_rows = grouped[family_id]
        missing = [prompt_type for prompt_type in EXPECTED_PROMPT_TYPES if prompt_type not in family_rows]
        if missing:
            raise ValueError(f"Family {family_id} is missing prompt types: {missing}")

        neutral_margin = float(family_rows["evidence_neutral"]["logit_margin"])
        output_row: Dict[str, Any] = {
            **family_meta[family_id],
            "logit_margin_evidence_neutral": neutral_margin,
            "logit_margin_evidence_false_belief_pressure": float(
                family_rows["evidence_false_belief_pressure"]["logit_margin"]
            ),
            "logit_margin_evidence_emotional_pressure": float(
                family_rows["evidence_emotional_pressure"]["logit_margin"]
            ),
            "logit_margin_evidence_true_belief_pressure": float(
                family_rows["evidence_true_belief_pressure"]["logit_margin"]
            ),
            "logit_margin_evidence_distractor_neutral": float(
                family_rows["evidence_distractor_neutral"]["logit_margin"]
            ),
            "logit_margin_closed_context_false_belief_pressure": float(
                family_rows["closed_context_false_belief_pressure"]["logit_margin"]
            ),
        }

        for delta_name, comparison_prompt_type in DELTA_SPECS:
            comparison_margin = float(family_rows[comparison_prompt_type]["logit_margin"])
            output_row[delta_name] = comparison_margin - neutral_margin

        output_rows.append(output_row)

    return output_rows


def summarize_delta_rows(delta_rows: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("Qwen3 1.7B Pilot-8 Family Margin Delta Summary")
    lines.append("")
    lines.append(f"input_families: {len(delta_rows)}")
    lines.append("")
    lines.append("Per-Delta Summary")

    for delta_name, _comparison_prompt_type in DELTA_SPECS:
        values = [float(row[delta_name]) for row in delta_rows]
        negative_count = sum(1 for value in values if value < 0)
        positive_count = sum(1 for value in values if value > 0)
        zero_count = sum(1 for value in values if value == 0)
        lines.append(
            "  "
            f"{delta_name}: "
            f"mean={format_float(mean(values))}, "
            f"median={format_float(median(values) if values else 0.0)}, "
            f"negative={negative_count}/{len(values)}, "
            f"positive={positive_count}/{len(values)}, "
            f"zero={zero_count}/{len(values)}"
        )

    lines.append("")
    lines.append("Requested Count View")
    false_negative_count = sum(1 for row in delta_rows if float(row["delta_false_pressure"]) < 0)
    emotional_negative_count = sum(1 for row in delta_rows if float(row["delta_emotional_pressure"]) < 0)
    lines.append(
        f"  In how many of {len(delta_rows)} families was delta_false_pressure negative? "
        f"{false_negative_count}"
    )
    lines.append(
        f"  In how many of {len(delta_rows)} families was delta_emotional_pressure negative? "
        f"{emotional_negative_count}"
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append("  Negative delta means the comparison condition weakened the model's preference for the correct answer relative to evidence_neutral.")
    lines.append("  Positive delta means the comparison condition did not weaken evidence-following relative to evidence_neutral.")
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
    csv_output = repo_root / DEFAULT_CSV_OUTPUT
    summary_output = repo_root / DEFAULT_SUMMARY_OUTPUT

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    delta_rows = build_family_delta_rows(rows)
    fieldnames = [
        "family_id",
        "domain",
        "title",
        "logit_margin_evidence_neutral",
        "logit_margin_evidence_false_belief_pressure",
        "logit_margin_evidence_emotional_pressure",
        "logit_margin_evidence_true_belief_pressure",
        "logit_margin_evidence_distractor_neutral",
        "logit_margin_closed_context_false_belief_pressure",
        "delta_false_pressure",
        "delta_emotional_pressure",
        "delta_true_pressure",
        "delta_distractor",
        "delta_closed_context",
    ]
    write_csv(csv_output, delta_rows, fieldnames)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(summarize_delta_rows(delta_rows), encoding="utf-8")

    print(f"Wrote {csv_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
