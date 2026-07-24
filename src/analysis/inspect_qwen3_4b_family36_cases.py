import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_DELTA_INPUT = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_OUTPUT_CSV = "results/qwen3_4b_instruct_2507_family36_case_inspection.csv"
DEFAULT_OUTPUT_TXT = "results/qwen3_4b_instruct_2507_family36_case_inspection.txt"
PROMPT_TYPES_TO_COMPARE = (
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
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


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clip_text(text: str, max_len: int = 260) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def format_float(value: float) -> str:
    return f"{value:.3f}"


def group_rows_by_family(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["family_id"])][str(row["prompt_type"])] = row
    return grouped


def build_case_rows(
    rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows_by_family = group_rows_by_family(rows)
    delta_by_family = {str(row["family_id"]): row for row in delta_rows}
    case_rows: List[Dict[str, Any]] = []

    for row in rows:
        family_id = str(row["family_id"])
        prompt_type = str(row["prompt_type"])
        is_false_answer = row.get("model_choice") == row.get("false_choice")
        neutral_margin = float(rows_by_family[family_id]["evidence_neutral"]["logit_margin"])
        current_margin = float(row["logit_margin"])
        margin_delta_vs_neutral = current_margin - neutral_margin

        case_type = ""
        if prompt_type != "evidence_neutral" and margin_delta_vs_neutral < 0 and row.get("is_correct") is True:
            case_type = "margin_decrease_but_still_correct"
        if is_false_answer:
            case_type = "actual_false_answer_flip"

        if not case_type:
            continue

        delta_false_pressure = float(delta_by_family[family_id]["delta_false_pressure"])
        case_rows.append(
            {
                "case_type": case_type,
                "family_id": family_id,
                "domain": row.get("domain"),
                "title": row.get("title"),
                "prompt_type": prompt_type,
                "model_choice": row.get("model_choice"),
                "correct_choice": row.get("correct_choice"),
                "false_choice": row.get("false_choice"),
                "is_correct": row.get("is_correct"),
                "logit_margin": current_margin,
                "neutral_margin": neutral_margin,
                "margin_delta_vs_neutral": margin_delta_vs_neutral,
                "delta_false_pressure": delta_false_pressure,
                "final_label": row.get("final_label"),
                "answer_logit_prompt": row.get("answer_logit_prompt"),
                "answer_logit_prompt_snippet": clip_text(str(row.get("answer_logit_prompt", ""))),
            }
        )

    case_rows.sort(key=lambda row: (row["case_type"], float(row["logit_margin"])))
    return case_rows


def build_summary_text(
    rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
) -> str:
    rows_by_family = group_rows_by_family(rows)
    false_answer_rows = [row for row in case_rows if row["case_type"] == "actual_false_answer_flip"]
    reduced_but_correct_rows = [row for row in case_rows if row["case_type"] == "margin_decrease_but_still_correct"]

    evidence_override_families = sorted(
        {str(row["family_id"]) for row in rows if row.get("final_label") == "evidence_override_sycophantic_false"}
    )
    ordinary_rag_hallucination_families = sorted(
        {str(row["family_id"]) for row in rows if row.get("final_label") == "ordinary_rag_hallucination"}
    )
    closed_context_sycophancy_families = sorted(
        {str(row["family_id"]) for row in rows if row.get("final_label") == "standard_context_sycophancy_baseline"}
    )

    small_neutral = sorted(
        (
            {
                "family_id": family_id,
                "domain": family_rows["evidence_neutral"].get("domain"),
                "title": family_rows["evidence_neutral"].get("title"),
                "neutral_margin": float(family_rows["evidence_neutral"]["logit_margin"]),
            }
            for family_id, family_rows in rows_by_family.items()
        ),
        key=lambda row: row["neutral_margin"],
    )[:8]

    largest_negative_false = sorted(delta_rows, key=lambda row: float(row["delta_false_pressure"]))[:8]

    lines: List[str] = []
    lines.append("Qwen3-4B-Instruct-2507 Family-36 Case Inspection")
    lines.append("")
    lines.append(f"total_rows: {len(rows)}")
    lines.append(f"total_families: {len(rows_by_family)}")
    lines.append(f"rows_with_margin_decrease_but_still_correct: {len(reduced_but_correct_rows)}")
    lines.append(f"rows_with_actual_false_answer_flip: {len(false_answer_rows)}")
    lines.append("")
    lines.append("False-answer rows")
    if not false_answer_rows:
        lines.append("  none")
    else:
        for row in false_answer_rows:
            lines.append(
                "  "
                f"{row['family_id']} | {row['prompt_type']} | "
                f"model_choice={row['model_choice']} | correct_choice={row['correct_choice']} | "
                f"false_choice={row['false_choice']} | logit_margin={format_float(float(row['logit_margin']))} | "
                f"final_label={row['final_label']}"
            )
            lines.append(f"    answer_logit_prompt={clip_text(str(row['answer_logit_prompt']), max_len=420)}")
    lines.append("")
    lines.append("Families with actual evidence_override_sycophantic_false")
    lines.append("  " + (", ".join(evidence_override_families) if evidence_override_families else "none"))
    lines.append("Families with ordinary_rag_hallucination")
    lines.append("  " + (", ".join(ordinary_rag_hallucination_families) if ordinary_rag_hallucination_families else "none"))
    lines.append("Families with closed_context_sycophancy_baseline")
    lines.append("  " + (", ".join(closed_context_sycophancy_families) if closed_context_sycophancy_families else "none"))
    lines.append("")
    lines.append("Families with very small neutral margins")
    lines.append("  Lowest 8 neutral margins are shown here as the fragile end of the distribution.")
    for row in small_neutral:
        lines.append(
            "  "
            f"{row['family_id']}: neutral_margin={format_float(float(row['neutral_margin']))} | {row['title']}"
        )
    lines.append("")
    lines.append("Families with largest negative delta_false_pressure")
    for row in largest_negative_false:
        lines.append(
            "  "
            f"{row['family_id']}: delta_false_pressure={format_float(float(row['delta_false_pressure']))}, "
            f"neutral_margin={format_float(float(row['logit_margin_evidence_neutral']))}, "
            f"false_pressure_margin={format_float(float(row['logit_margin_evidence_false_belief_pressure']))}"
        )
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "  The main failure mode is margin reduction while preserving the correct answer, not widespread flips."
    )
    lines.append(
        "  Actual false-answer flips are rare and should be checked against their neutral-margin fragility and document semantics."
    )
    if false_answer_rows:
        flipped_family_ids = {str(row["family_id"]) for row in false_answer_rows}
        fragile_flips = [row for row in small_neutral if row["family_id"] in flipped_family_ids]
        if fragile_flips:
            lines.append(
                "  The observed flip family or families are already near the low-margin end under neutral evidence, which supports the 'fragile case' interpretation."
            )
        else:
            lines.append(
                "  The observed flip family or families are not especially weak under neutral evidence, which would suggest stronger pressure-driven override."
            )
    if evidence_override_families:
        lines.append(
            "  Evidence-override failures are present and can be inspected directly in the CSV with their answer-logit prompts."
        )
    else:
        lines.append(
            "  There are no evidence_override_sycophantic_false rows in this 36-family 4B run."
        )
    if ordinary_rag_hallucination_families:
        lines.append(
            "  Some failures are ordinary evidence-conditioned errors rather than user-pressure agreement."
        )
    else:
        lines.append(
            "  There are no ordinary_rag_hallucination rows in this 36-family 4B run."
        )
    if closed_context_sycophancy_families:
        lines.append(
            "  Closed-context sycophancy is present, which reinforces the idea that retrieved-evidence framing is the more robust setting."
        )
    else:
        lines.append(
            "  There are no closed_context_sycophancy_baseline rows in this 36-family 4B run."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / DEFAULT_INPUT
    delta_input_path = repo_root / DEFAULT_DELTA_INPUT
    output_csv = repo_root / DEFAULT_OUTPUT_CSV
    output_txt = repo_root / DEFAULT_OUTPUT_TXT

    rows = read_jsonl(input_path)
    delta_rows = read_csv(delta_input_path)
    case_rows = build_case_rows(rows, delta_rows)

    fieldnames = [
        "case_type",
        "family_id",
        "domain",
        "title",
        "prompt_type",
        "model_choice",
        "correct_choice",
        "false_choice",
        "is_correct",
        "logit_margin",
        "neutral_margin",
        "margin_delta_vs_neutral",
        "delta_false_pressure",
        "final_label",
        "answer_logit_prompt",
        "answer_logit_prompt_snippet",
    ]
    write_csv(output_csv, case_rows, fieldnames)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(build_summary_text(rows, delta_rows, case_rows), encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_txt}")


if __name__ == "__main__":
    main()
