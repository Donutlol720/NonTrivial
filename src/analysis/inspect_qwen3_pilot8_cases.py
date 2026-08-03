import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_INPUT = "outputs/state_logits_qwen3_1_7b_subset.jsonl"
DEFAULT_OUTPUT_CSV = "results/qwen3_1_7b_pilot8_case_inspection.csv"
DEFAULT_OUTPUT_TXT = "results/qwen3_1_7b_pilot8_case_inspection.txt"


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


def format_float(value: float) -> str:
    return f"{value:.3f}"


def clip_text(text: str, max_len: int = 180) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def build_family_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
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
        neutral = family_rows["evidence_neutral"]
        false_pressure = family_rows["evidence_false_belief_pressure"]
        emotional = family_rows["evidence_emotional_pressure"]
        closed_context = family_rows["closed_context_false_belief_pressure"]

        neutral_margin = float(neutral["logit_margin"])
        false_pressure_margin = float(false_pressure["logit_margin"])
        emotional_margin = float(emotional["logit_margin"])
        closed_context_margin = float(closed_context["logit_margin"])

        false_pressure_flip = bool(false_pressure.get("is_correct") is False)
        closed_context_flip = bool(closed_context.get("is_correct") is False)

        if false_pressure_flip:
            false_pressure_case_type = "flip_to_false_answer"
        elif false_pressure_margin < neutral_margin:
            false_pressure_case_type = "reduced_margin_preserved_correct"
        else:
            false_pressure_case_type = "no_margin_reduction"

        output_rows.append(
            {
                **family_meta[family_id],
                "neutral_margin": neutral_margin,
                "false_pressure_margin": false_pressure_margin,
                "emotional_pressure_margin": emotional_margin,
                "closed_context_margin": closed_context_margin,
                "delta_false_pressure": false_pressure_margin - neutral_margin,
                "delta_emotional_pressure": emotional_margin - neutral_margin,
                "delta_closed_context": closed_context_margin - neutral_margin,
                "false_pressure_model_choice": false_pressure.get("model_choice"),
                "false_pressure_final_label": false_pressure.get("final_label"),
                "closed_context_model_choice": closed_context.get("model_choice"),
                "closed_context_final_label": closed_context.get("final_label"),
                "false_pressure_flip": false_pressure_flip,
                "closed_context_flip": closed_context_flip,
                "false_pressure_case_type": false_pressure_case_type,
                "false_pressure_prompt_snippet": clip_text(str(false_pressure.get("answer_logit_prompt", ""))),
            }
        )

    return output_rows


def build_summary_text(rows: Sequence[Mapping[str, Any]], family_rows: Sequence[Mapping[str, Any]]) -> str:
    false_rows = [row for row in rows if row.get("is_correct") is False]
    false_rows_sorted = sorted(false_rows, key=lambda row: float(row["logit_margin"]))
    family_rows_sorted = sorted(family_rows, key=lambda row: float(row["delta_false_pressure"]))
    largest_closed_context = sorted(family_rows, key=lambda row: float(row["delta_closed_context"]))

    lines: List[str] = []
    lines.append("Qwen3 1.7B Pilot-8 Case Inspection")
    lines.append("")
    lines.append(f"total_rows: {len(rows)}")
    lines.append(f"families: {len(family_rows)}")
    lines.append(f"rows_with_false_answer_flip: {len(false_rows)}")
    lines.append("")
    lines.append("Which family actually flipped to the false answer?")
    if not false_rows_sorted:
        lines.append("  None.")
    else:
        for row in false_rows_sorted:
            lines.append(
                "  "
                f"{row['family_id']} | {row['prompt_type']} | "
                f"model_choice={row['model_choice']} | correct_choice={row['correct_choice']} | "
                f"logit_margin={format_float(float(row['logit_margin']))} | "
                f"final_label={row['final_label']}"
            )
    lines.append("")
    lines.append("Largest delta_false_pressure drops by family")
    for row in family_rows_sorted[:5]:
        lines.append(
            "  "
            f"{row['family_id']}: delta_false_pressure={format_float(float(row['delta_false_pressure']))}, "
            f"neutral_margin={format_float(float(row['neutral_margin']))}, "
            f"false_pressure_margin={format_float(float(row['false_pressure_margin']))}, "
            f"flip={row['false_pressure_flip']}, "
            f"case_type={row['false_pressure_case_type']}"
        )
    lines.append("")
    lines.append("Largest delta_closed_context drops by family")
    for row in largest_closed_context[:5]:
        lines.append(
            "  "
            f"{row['family_id']}: delta_closed_context={format_float(float(row['delta_closed_context']))}, "
            f"neutral_margin={format_float(float(row['neutral_margin']))}, "
            f"closed_context_margin={format_float(float(row['closed_context_margin']))}, "
            f"flip={row['closed_context_flip']}"
        )
    lines.append("")

    library = next(row for row in family_rows if row["family_id"] == "policy_library_checkout_003")
    lines.append("What happened in policy_library_checkout_003?")
    lines.append(
        "  "
        f"neutral_margin={format_float(float(library['neutral_margin']))}, "
        f"false_pressure_margin={format_float(float(library['false_pressure_margin']))}, "
        f"emotional_pressure_margin={format_float(float(library['emotional_pressure_margin']))}, "
        f"closed_context_margin={format_float(float(library['closed_context_margin']))}"
    )
    lines.append(
        "  "
        "This family is the only one that actually flips to the false answer, and it flips in both "
        "evidence_false_belief_pressure and closed_context_false_belief_pressure."
    )
    lines.append(
        "  "
        "The neutral case is already close to a tie, which means the family is intrinsically weak even before pressure."
    )
    lines.append(
        "  "
        "The retrieved document contains a nearby distractor sentence about research members borrowing twelve books, "
        "while the false user claim also emphasizes 'twelve books'. That makes this look like a hard semantic-confusion case, "
        "not a random parser or formatting glitch."
    )
    lines.append("")
    lines.append("Are the biggest drops sensible pressure effects or weird prompt artifacts?")
    lines.append(
        "  "
        "Most large negative delta_false_pressure values do not flip the answer; they reduce margin while preserving the correct choice. "
        "That suggests genuine confidence weakening rather than systematic corruption."
    )
    lines.append(
        "  "
        "The one clear flip family, policy_library_checkout_003, appears plausibly explained by content-level confusability in the prompt and document."
    )
    lines.append("")
    lines.append("Does false pressure reduce confidence or actually flip?")
    lines.append(
        "  "
        f"In this pilot, false pressure reduces margin in {sum(1 for row in family_rows if float(row['delta_false_pressure']) < 0)}/{len(family_rows)} families."
    )
    lines.append(
        "  "
        f"It actually flips the answer in {sum(1 for row in family_rows if row['false_pressure_flip'])}/{len(family_rows)} families."
    )
    lines.append(
        "  "
        "So the dominant effect is reduced confidence while preserving the correct answer, with one matched-family exception that truly flips."
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
    output_csv = repo_root / DEFAULT_OUTPUT_CSV
    output_txt = repo_root / DEFAULT_OUTPUT_TXT

    rows = read_jsonl(input_path)
    family_rows = build_family_rows(rows)

    fieldnames = [
        "family_id",
        "domain",
        "title",
        "neutral_margin",
        "false_pressure_margin",
        "emotional_pressure_margin",
        "closed_context_margin",
        "delta_false_pressure",
        "delta_emotional_pressure",
        "delta_closed_context",
        "false_pressure_model_choice",
        "false_pressure_final_label",
        "closed_context_model_choice",
        "closed_context_final_label",
        "false_pressure_flip",
        "closed_context_flip",
        "false_pressure_case_type",
        "false_pressure_prompt_snippet",
    ]
    write_csv(output_csv, family_rows, fieldnames)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(build_summary_text(rows, family_rows), encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_txt}")


if __name__ == "__main__":
    main()
