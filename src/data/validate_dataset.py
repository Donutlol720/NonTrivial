import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


REQUIRED_FIELDS = (
    "prompt_id",
    "family_id",
    "prompt_type",
    "prompt",
    "correct_choice",
    "false_choice",
)

PREGEN_NULL_FIELDS = (
    "final_label",
    "generated_response",
    "parsed_answer",
)

REQUIRED_EVIDENCE_FAMILY_TYPES = {
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Row {line_number} is not a JSON object")
            rows.append(obj)
    return rows


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def format_list(items: Sequence[str], max_items: int = 20) -> str:
    if not items:
        return "none"
    shown = list(items[:max_items])
    suffix = "" if len(items) <= max_items else f" ... (+{len(items) - max_items} more)"
    return ", ".join(shown) + suffix


def resolve_dataset_path(repo_root: Path, requested_path: str) -> Tuple[Path, List[str]]:
    notes: List[str] = []
    candidate = Path(requested_path)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    if candidate.exists():
        return candidate, notes

    fallback = (repo_root / "generated_prompts_v1.jsonl").resolve()
    if fallback.exists():
        notes.append(
            f"Requested dataset path not found: {candidate}. Falling back to current repo dataset: {fallback}."
        )
        return fallback, notes

    raise FileNotFoundError(f"Dataset file not found: {candidate}")


def validate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    missing_fields: Dict[str, List[str]] = {field: [] for field in REQUIRED_FIELDS}
    invalid_choice_rows: List[str] = []
    nonnull_pregen_rows: Dict[str, List[str]] = {field: [] for field in PREGEN_NULL_FIELDS}
    prompt_ids: List[str] = []
    prompt_types: Set[str] = set()
    family_types: Dict[str, Set[str]] = defaultdict(set)

    for index, row in enumerate(rows, start=1):
        prompt_id = str(row.get("prompt_id", "")).strip() or f"<row_{index}>"
        for field in REQUIRED_FIELDS:
            if is_missing(row.get(field)):
                missing_fields[field].append(prompt_id)

        correct_choice = row.get("correct_choice")
        false_choice = row.get("false_choice")
        if correct_choice not in {"A", "B"} or false_choice not in {"A", "B"} or correct_choice == false_choice:
            invalid_choice_rows.append(prompt_id)

        for field in PREGEN_NULL_FIELDS:
            if row.get(field) is not None:
                nonnull_pregen_rows[field].append(prompt_id)

        if not is_missing(row.get("prompt_id")):
            prompt_ids.append(str(row["prompt_id"]))
        if not is_missing(row.get("prompt_type")):
            prompt_types.add(str(row["prompt_type"]))
        if not is_missing(row.get("family_id")) and not is_missing(row.get("prompt_type")):
            family_types[str(row["family_id"])].add(str(row["prompt_type"]))

    duplicate_prompt_ids = sorted([pid for pid, count in Counter(prompt_ids).items() if count > 1])

    incomplete_families: Dict[str, List[str]] = {}
    evidence_family_count = 0
    for family_id, types in sorted(family_types.items()):
        if not (types & REQUIRED_EVIDENCE_FAMILY_TYPES):
            continue
        evidence_family_count += 1
        missing_types = sorted(REQUIRED_EVIDENCE_FAMILY_TYPES - types)
        if missing_types:
            incomplete_families[family_id] = missing_types

    serious_error_count = (
        sum(len(v) for v in missing_fields.values())
        + len(invalid_choice_rows)
        + len(duplicate_prompt_ids)
        + len(incomplete_families)
        + sum(len(v) for v in nonnull_pregen_rows.values())
    )

    return {
        "total_rows": len(rows),
        "unique_family_ids": len(family_types),
        "prompt_type_count": len(prompt_types),
        "prompt_types": sorted(prompt_types),
        "missing_fields": missing_fields,
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "incomplete_families": incomplete_families,
        "invalid_choice_rows": invalid_choice_rows,
        "nonnull_pregen_rows": nonnull_pregen_rows,
        "evidence_family_count": evidence_family_count,
        "serious_error_count": serious_error_count,
    }


def build_report(dataset_path: Path, notes: Iterable[str], results: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Dataset Validation Report")
    lines.append("=========================")
    lines.append("")
    lines.append(f"Dataset path: {dataset_path}")
    lines.append(f"Total rows: {results['total_rows']}")
    lines.append(f"Unique family_id values: {results['unique_family_ids']}")
    lines.append(f"Number of prompt types: {results['prompt_type_count']}")
    lines.append(f"Prompt types: {', '.join(results['prompt_types']) if results['prompt_types'] else 'none'}")
    lines.append(f"Evidence-family count checked for completeness: {results['evidence_family_count']}")
    lines.append("")

    notes = list(notes)
    if notes:
        lines.append("Path Notes")
        lines.append("----------")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("Missing Fields")
    lines.append("--------------")
    for field, rows in results["missing_fields"].items():
        lines.append(f"- {field}: {len(rows)}")
        if rows:
            lines.append(f"  Rows: {format_list(rows)}")
    lines.append("")

    lines.append("Duplicate prompt_id Values")
    lines.append("--------------------------")
    lines.append(f"Count: {len(results['duplicate_prompt_ids'])}")
    if results["duplicate_prompt_ids"]:
        lines.append(f"Values: {format_list(results['duplicate_prompt_ids'])}")
    lines.append("")

    lines.append("Incomplete Families")
    lines.append("-------------------")
    lines.append(f"Count: {len(results['incomplete_families'])}")
    for family_id, missing_types in results["incomplete_families"].items():
        lines.append(f"- {family_id}: missing {', '.join(missing_types)}")
    if not results["incomplete_families"]:
        lines.append("none")
    lines.append("")

    lines.append("Rows With Invalid correct_choice or false_choice")
    lines.append("-----------------------------------------------")
    lines.append(f"Count: {len(results['invalid_choice_rows'])}")
    if results["invalid_choice_rows"]:
        lines.append(f"Rows: {format_list(results['invalid_choice_rows'])}")
    lines.append("")

    lines.append("Rows With Non-Null Pre-Generation Fields")
    lines.append("----------------------------------------")
    for field, rows in results["nonnull_pregen_rows"].items():
        lines.append(f"- {field}: {len(rows)}")
        if rows:
            lines.append(f"  Rows: {format_list(rows)}")
    lines.append("")

    success = results["serious_error_count"] == 0
    lines.append("Validation Result")
    lines.append("-----------------")
    lines.append("PASS" if success else "FAIL")
    lines.append(
        "Dataset passes validation with no serious errors. Do not proceed to model generation until this passes."
        if success
        else "Dataset has serious validation errors. Do not proceed to model generation until they are fixed."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/generated_prompts_v1.jsonl")
    parser.add_argument("--report", default="results/dataset_validation_report.txt")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_path, notes = resolve_dataset_path(repo_root, args.dataset)
    rows = read_jsonl(dataset_path)
    results = validate_rows(rows)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = (repo_root / report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_report(dataset_path, notes, results)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    return 0 if results["serious_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
