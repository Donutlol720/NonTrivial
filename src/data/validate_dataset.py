import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


REQUIRED_FIELDS = (
    "prompt_id",
    "family_id",
    "domain",
    "prompt_type",
    "intended_condition",
    "pressure_type",
    "has_retrieved_evidence",
    "correct_choice",
    "false_choice",
    "prompt",
    "generated_response",
    "parsed_answer",
    "is_correct",
    "agrees_with_user",
    "quotes_correct_evidence",
    "final_label",
)

PREGEN_NULL_FIELDS = (
    "generated_response",
    "parsed_answer",
    "is_correct",
    "agrees_with_user",
    "quotes_correct_evidence",
    "final_label",
)

REQUIRED_NONNULL_FIELDS = tuple(field for field in REQUIRED_FIELDS if field not in PREGEN_NULL_FIELDS)

REQUIRED_EVIDENCE_FAMILY_TYPES = {
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "evidence_true_belief_pressure",
    "evidence_distractor_neutral",
    "closed_context_false_belief_pressure",
}

STANDARD_FAMILY_TYPES = {
    "standard_neutral",
    "standard_false_belief_pressure",
}

CANONICAL_DATASET_PATH = Path("data/generated_prompts_v1.jsonl")
LEGACY_DATASET_PATH = Path("generated_prompts_v1.jsonl")

AB_ANSWER_PATTERNS = (
    re.compile(r"ANSWER\s*:\s*A\s+or\s+B", re.IGNORECASE),
    re.compile(r'answer with either\s+"?A"?\s+or\s+"?B"?', re.IGNORECASE),
    re.compile(r'first answer with either\s+"?A"?\s+or\s+"?B"?', re.IGNORECASE),
)

RETRIEVED_CONTEXT_PATTERNS = (
    re.compile(r"retrieved document\s*:", re.IGNORECASE),
    re.compile(r"\bcontext\s*:", re.IGNORECASE),
)


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


def format_counter(counter: Mapping[str, int]) -> List[str]:
    if not counter:
        return ["none"]
    return [f"- {name}: {count}" for name, count in sorted(counter.items())]


def resolve_dataset_path(repo_root: Path, requested_path: str) -> Tuple[Path, List[str]]:
    notes: List[str] = []
    candidate = Path(requested_path)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    if candidate.exists():
        return candidate, notes

    canonical_dataset = (repo_root / CANONICAL_DATASET_PATH).resolve()
    requested_repo_relative: Path
    try:
        requested_repo_relative = candidate.relative_to(repo_root)
    except ValueError:
        requested_repo_relative = candidate

    if requested_repo_relative == LEGACY_DATASET_PATH and canonical_dataset.exists():
        notes.append(
            f"Requested legacy dataset path not found: {candidate}. Using canonical dataset path: {canonical_dataset}."
        )
        return canonical_dataset, notes

    if canonical_dataset.exists():
        notes.append(
            f"Requested dataset path not found: {candidate}. Falling back to current repo dataset: {canonical_dataset}."
        )
        return canonical_dataset, notes

    raise FileNotFoundError(f"Dataset file not found: {candidate}")


def prompt_id_for_row(row: Mapping[str, Any], index: int) -> str:
    prompt_id = row.get("prompt_id")
    if isinstance(prompt_id, str) and prompt_id.strip():
        return prompt_id.strip()
    return f"<row_{index}>"


def add_issue(bucket: List[str], prefix: str, message: str) -> None:
    bucket.append(f"{prefix}: {message}")


def row_has_ab_constraint(prompt: Any) -> bool:
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    return any(pattern.search(prompt) for pattern in AB_ANSWER_PATTERNS)


def row_has_context_block(prompt: Any) -> bool:
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    return any(pattern.search(prompt) for pattern in RETRIEVED_CONTEXT_PATTERNS)


def validate_prompt_type_rules(
    row: Mapping[str, Any],
    prompt_id: str,
    warnings: List[str],
) -> None:
    prompt_type = row.get("prompt_type")
    if not isinstance(prompt_type, str) or not prompt_type.strip():
        return

    pressure_type = row.get("pressure_type")
    has_retrieved_evidence = row.get("has_retrieved_evidence")
    user_claim_choice = row.get("user_claim_choice")
    user_claim_truth = row.get("user_claim_truth")
    correct_choice = row.get("correct_choice")
    false_choice = row.get("false_choice")

    def warn(message: str) -> None:
        add_issue(warnings, "WARNING", f"{prompt_id}: {message}")

    if prompt_type == "evidence_neutral":
        if pressure_type != "none":
            warn("evidence_neutral should have pressure_type=none.")
        if has_retrieved_evidence is not True:
            warn("evidence_neutral should have has_retrieved_evidence=true.")
        if user_claim_choice is not None:
            warn("evidence_neutral should have user_claim_choice=null.")
        if user_claim_truth is not None:
            warn("evidence_neutral should have user_claim_truth=null.")
        return

    if prompt_type == "evidence_false_belief_pressure":
        if pressure_type != "belief_pressure":
            warn("evidence_false_belief_pressure should have pressure_type=belief_pressure.")
        if has_retrieved_evidence is not True:
            warn("evidence_false_belief_pressure should have has_retrieved_evidence=true.")
        if user_claim_choice != false_choice:
            warn("evidence_false_belief_pressure should have user_claim_choice=false_choice.")
        if user_claim_truth is not False:
            warn("evidence_false_belief_pressure should have user_claim_truth=false.")
        return

    if prompt_type == "evidence_emotional_pressure":
        if pressure_type != "emotional_pressure":
            warn("evidence_emotional_pressure should have pressure_type=emotional_pressure.")
        if has_retrieved_evidence is not True:
            warn("evidence_emotional_pressure should have has_retrieved_evidence=true.")
        if user_claim_choice != false_choice:
            warn("evidence_emotional_pressure should have user_claim_choice=false_choice.")
        if user_claim_truth is not False:
            warn("evidence_emotional_pressure should have user_claim_truth=false.")
        return

    if prompt_type == "evidence_true_belief_pressure":
        if pressure_type != "belief_pressure":
            warn("evidence_true_belief_pressure should have pressure_type=belief_pressure.")
        if has_retrieved_evidence is not True:
            warn("evidence_true_belief_pressure should have has_retrieved_evidence=true.")
        if user_claim_choice != correct_choice:
            warn("evidence_true_belief_pressure should have user_claim_choice=correct_choice.")
        if user_claim_truth is not True:
            warn("evidence_true_belief_pressure should have user_claim_truth=true.")
        return

    if prompt_type == "evidence_distractor_neutral":
        if pressure_type != "none":
            warn("evidence_distractor_neutral should have pressure_type=none.")
        if has_retrieved_evidence is not True:
            warn("evidence_distractor_neutral should have has_retrieved_evidence=true.")
        if user_claim_choice is not None:
            warn("evidence_distractor_neutral should have user_claim_choice=null.")
        if user_claim_truth is not None:
            warn("evidence_distractor_neutral should have user_claim_truth=null.")
        return

    if prompt_type == "closed_context_false_belief_pressure":
        if pressure_type != "belief_pressure":
            warn("closed_context_false_belief_pressure should have pressure_type=belief_pressure.")
        if user_claim_choice != false_choice:
            warn("closed_context_false_belief_pressure should have user_claim_choice=false_choice.")
        if user_claim_truth is not False:
            warn("closed_context_false_belief_pressure should have user_claim_truth=false.")
        return

    if prompt_type == "standard_neutral":
        if has_retrieved_evidence is not False:
            warn("standard_neutral should have has_retrieved_evidence=false.")
        if pressure_type != "none":
            warn("standard_neutral should have pressure_type=none.")
        if user_claim_choice is not None:
            warn("standard_neutral should have user_claim_choice=null.")
        if user_claim_truth is not None:
            warn("standard_neutral should have user_claim_truth=null.")
        return

    if prompt_type == "standard_false_belief_pressure":
        if has_retrieved_evidence is not False:
            warn("standard_false_belief_pressure should have has_retrieved_evidence=false.")
        if pressure_type != "belief_pressure":
            warn("standard_false_belief_pressure should have pressure_type=belief_pressure.")
        if user_claim_choice != false_choice:
            warn("standard_false_belief_pressure should have user_claim_choice=false_choice.")
        if user_claim_truth is not False:
            warn("standard_false_belief_pressure should have user_claim_truth=false.")
        return

    warn(f"Unrecognized prompt_type={prompt_type}.")


def validate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    missing_required_fields: Dict[str, List[str]] = {field: [] for field in REQUIRED_FIELDS}
    nonnull_pregen_rows: Dict[str, List[str]] = {field: [] for field in PREGEN_NULL_FIELDS}
    prompt_type_counts: Counter = Counter()
    domain_counts: Counter = Counter()
    family_prompt_types: Dict[str, Set[str]] = defaultdict(set)
    family_ids: Set[str] = set()
    prompt_ids: List[str] = []
    invalid_choice_rows: List[str] = []
    duplicate_prompt_ids: List[str] = []
    incomplete_families: List[Dict[str, Any]] = []
    serious_errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []
    prompt_format_warning_rows: List[str] = []
    missing_evidence_rows: List[str] = []

    for index, row in enumerate(rows, start=1):
        prompt_id = prompt_id_for_row(row, index)
        prompt_type = row.get("prompt_type")
        family_id = row.get("family_id")
        domain = row.get("domain")

        for field in REQUIRED_NONNULL_FIELDS:
            if is_missing(row.get(field)):
                missing_required_fields[field].append(prompt_id)
                add_issue(serious_errors, "ERROR", f"{prompt_id}: missing required field {field}.")
        for field in PREGEN_NULL_FIELDS:
            if field not in row:
                missing_required_fields[field].append(prompt_id)
                add_issue(serious_errors, "ERROR", f"{prompt_id}: missing required field {field}.")

        if isinstance(prompt_type, str) and prompt_type.strip():
            prompt_type_counts[prompt_type] += 1
        if isinstance(domain, str) and domain.strip():
            domain_counts[domain] += 1
        if isinstance(family_id, str) and family_id.strip():
            family_ids.add(family_id)
        if isinstance(family_id, str) and family_id.strip() and isinstance(prompt_type, str) and prompt_type.strip():
            family_prompt_types[family_id].add(prompt_type)
        if isinstance(row.get("prompt_id"), str) and row["prompt_id"].strip():
            prompt_ids.append(row["prompt_id"].strip())

        correct_choice = row.get("correct_choice")
        false_choice = row.get("false_choice")
        if correct_choice not in {"A", "B"}:
            invalid_choice_rows.append(prompt_id)
            add_issue(serious_errors, "ERROR", f"{prompt_id}: correct_choice must be A or B.")
        if false_choice not in {"A", "B"}:
            invalid_choice_rows.append(prompt_id)
            add_issue(serious_errors, "ERROR", f"{prompt_id}: false_choice must be A or B.")
        if correct_choice in {"A", "B"} and false_choice in {"A", "B"} and correct_choice == false_choice:
            invalid_choice_rows.append(prompt_id)
            add_issue(serious_errors, "ERROR", f"{prompt_id}: correct_choice and false_choice must differ.")

        for field in PREGEN_NULL_FIELDS:
            if row.get(field) is not None:
                nonnull_pregen_rows[field].append(prompt_id)
                add_issue(serious_errors, "ERROR", f"{prompt_id}: {field} must be null before generation.")

        if not row_has_ab_constraint(row.get("prompt")):
            prompt_format_warning_rows.append(prompt_id)
            add_issue(warnings, "WARNING", f"{prompt_id}: prompt does not clearly force an A/B answer.")

        has_retrieved_evidence = row.get("has_retrieved_evidence")
        evidence_sentences = row.get("evidence_sentences")
        evidence_ids = row.get("evidence_sentence_ids_0_indexed")

        if has_retrieved_evidence is True:
            if "evidence_sentences" not in row:
                missing_evidence_rows.append(prompt_id)
                add_issue(serious_errors, "ERROR", f"{prompt_id}: missing evidence_sentences for retrieved-evidence row.")
            elif not isinstance(evidence_sentences, list) or not evidence_sentences:
                missing_evidence_rows.append(prompt_id)
                add_issue(serious_errors, "ERROR", f"{prompt_id}: evidence_sentences must be a non-empty list.")

            if "evidence_sentence_ids_0_indexed" not in row:
                add_issue(warnings, "WARNING", f"{prompt_id}: missing evidence_sentence_ids_0_indexed.")
            elif not isinstance(evidence_ids, list) or not evidence_ids:
                add_issue(
                    warnings,
                    "WARNING",
                    f"{prompt_id}: evidence_sentence_ids_0_indexed should usually contain at least one sentence index.",
                )

            if not row_has_context_block(row.get("prompt")):
                add_issue(
                    serious_errors,
                    "ERROR",
                    f"{prompt_id}: retrieved-evidence row is missing a retrieved document or context block in the prompt.",
                )

            if isinstance(evidence_sentences, list) and evidence_sentences:
                prompt_text = row.get("prompt")
                if isinstance(prompt_text, str) and not any(sentence in prompt_text for sentence in evidence_sentences):
                    add_issue(
                        warnings,
                        "WARNING",
                        f"{prompt_id}: no gold evidence sentence appears verbatim in the prompt text.",
                    )

        if has_retrieved_evidence is False and correct_choice not in {"A", "B"}:
            add_issue(warnings, "WARNING", f"{prompt_id}: non-retrieved row still needs valid A/B choices.")

        if prompt_type in {
            "evidence_false_belief_pressure",
            "evidence_emotional_pressure",
            "evidence_true_belief_pressure",
            "closed_context_false_belief_pressure",
            "standard_false_belief_pressure",
        }:
            if "user_claim_choice" not in row:
                add_issue(warnings, "WARNING", f"{prompt_id}: missing user_claim_choice for pressure prompt.")
            if "user_claim_truth" not in row:
                add_issue(warnings, "WARNING", f"{prompt_id}: missing user_claim_truth for pressure prompt.")

        if prompt_type in {"evidence_neutral", "evidence_distractor_neutral", "standard_neutral"}:
            if "user_claim_choice" not in row:
                add_issue(warnings, "WARNING", f"{prompt_id}: missing user_claim_choice field.")
            if "user_claim_truth" not in row:
                add_issue(warnings, "WARNING", f"{prompt_id}: missing user_claim_truth field.")

        validate_prompt_type_rules(row, prompt_id, warnings)

    duplicate_prompt_ids = sorted([prompt_id for prompt_id, count in Counter(prompt_ids).items() if count > 1])
    for prompt_id in duplicate_prompt_ids:
        add_issue(serious_errors, "ERROR", f"{prompt_id}: duplicate prompt_id.")

    for family_id, present_types in sorted(family_prompt_types.items()):
        if present_types & REQUIRED_EVIDENCE_FAMILY_TYPES:
            missing_types = sorted(REQUIRED_EVIDENCE_FAMILY_TYPES - present_types)
            if missing_types:
                incomplete_families.append(
                    {
                        "family_id": family_id,
                        "missing_prompt_types": missing_types,
                        "present_prompt_types": sorted(present_types),
                    }
                )
                add_issue(
                    warnings,
                    "WARNING",
                    f"{family_id}: incomplete evidence family. Missing {', '.join(missing_types)}. "
                    f"Present {', '.join(sorted(present_types))}.",
                )
            continue

        if present_types & STANDARD_FAMILY_TYPES:
            missing_standard_types = sorted(STANDARD_FAMILY_TYPES - present_types)
            if missing_standard_types:
                add_issue(
                    warnings,
                    "WARNING",
                    f"{family_id}: incomplete standard baseline family. Missing {', '.join(missing_standard_types)}.",
                )

    serious_errors = sorted(set(serious_errors))
    warnings = sorted(set(warnings))
    info = sorted(
        set(
            info
            + [
                f"INFO: total_rows={len(rows)}",
                f"INFO: total_unique_prompt_ids={len(set(prompt_ids))}",
                f"INFO: total_unique_family_ids={len(family_ids)}",
            ]
        )
    )

    invalid_choice_rows = sorted(set(invalid_choice_rows))
    prompt_format_warning_rows = sorted(set(prompt_format_warning_rows))
    missing_evidence_rows = sorted(set(missing_evidence_rows))

    number_of_missing_required_fields = sum(len(items) for items in missing_required_fields.values())
    number_of_invalid_choice_rows = len(invalid_choice_rows)
    number_of_rows_with_nonnull_final_label_before_generation = len(nonnull_pregen_rows["final_label"])
    number_of_rows_with_nonnull_generated_response_before_generation = len(nonnull_pregen_rows["generated_response"])
    error_count = len(serious_errors)
    warning_count = len(warnings)

    return {
        "total_rows": len(rows),
        "total_unique_prompt_ids": len(set(prompt_ids)),
        "total_unique_family_ids": len(family_ids),
        "prompt_type_counts": dict(sorted(prompt_type_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "number_of_duplicate_prompt_ids": len(duplicate_prompt_ids),
        "number_of_missing_required_fields": number_of_missing_required_fields,
        "number_of_invalid_choice_rows": number_of_invalid_choice_rows,
        "number_of_rows_with_nonnull_final_label_before_generation": number_of_rows_with_nonnull_final_label_before_generation,
        "number_of_rows_with_nonnull_generated_response_before_generation": number_of_rows_with_nonnull_generated_response_before_generation,
        "number_of_rows_with_nonnull_parsed_answer_before_generation": len(nonnull_pregen_rows["parsed_answer"]),
        "number_of_rows_with_nonnull_is_correct_before_generation": len(nonnull_pregen_rows["is_correct"]),
        "number_of_rows_with_nonnull_agrees_with_user_before_generation": len(nonnull_pregen_rows["agrees_with_user"]),
        "number_of_rows_with_nonnull_quotes_correct_evidence_before_generation": len(
            nonnull_pregen_rows["quotes_correct_evidence"]
        ),
        "number_of_incomplete_families": len(incomplete_families),
        "list_of_incomplete_families": incomplete_families,
        "list_of_serious_errors": serious_errors,
        "list_of_warnings": warnings,
        "list_of_info": info,
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "missing_required_fields": missing_required_fields,
        "invalid_choice_rows": invalid_choice_rows,
        "nonnull_pregen_rows": nonnull_pregen_rows,
        "prompt_format_warning_rows": prompt_format_warning_rows,
        "missing_evidence_rows": missing_evidence_rows,
        "error_count": error_count,
        "warning_count": warning_count,
        "serious_error_count": error_count,
    }


def build_report(dataset_path: Path, notes: Iterable[str], results: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Dataset Validation Report")
    lines.append("=========================")
    lines.append("")
    lines.append(f"Dataset path: {dataset_path}")
    lines.append("")

    if results["error_count"] == 0:
        lines.append("Validation Result: PASS")
        lines.append("The dataset has no ERROR-level issues and is structurally ready for generation.")
    else:
        lines.append("Validation Result: FAIL")
        lines.append("The dataset has ERROR-level issues and should not be used for generation yet.")
    lines.append("")

    lines.append("Summary")
    lines.append("-------")
    lines.append(f"total_rows: {results['total_rows']}")
    lines.append(f"total_unique_prompt_ids: {results['total_unique_prompt_ids']}")
    lines.append(f"total_unique_family_ids: {results['total_unique_family_ids']}")
    lines.append(f"number_of_duplicate_prompt_ids: {results['number_of_duplicate_prompt_ids']}")
    lines.append(f"number_of_missing_required_fields: {results['number_of_missing_required_fields']}")
    lines.append(f"number_of_invalid_choice_rows: {results['number_of_invalid_choice_rows']}")
    lines.append(
        "number_of_rows_with_nonnull_final_label_before_generation: "
        f"{results['number_of_rows_with_nonnull_final_label_before_generation']}"
    )
    lines.append(
        "number_of_rows_with_nonnull_generated_response_before_generation: "
        f"{results['number_of_rows_with_nonnull_generated_response_before_generation']}"
    )
    lines.append(f"number_of_incomplete_families: {results['number_of_incomplete_families']}")
    lines.append(f"error_count: {results['error_count']}")
    lines.append(f"warning_count: {results['warning_count']}")
    lines.append("")

    lines.append("Prompt Type Counts")
    lines.append("------------------")
    lines.extend(format_counter(results["prompt_type_counts"]))
    lines.append("")

    lines.append("Domain Counts")
    lines.append("-------------")
    lines.extend(format_counter(results["domain_counts"]))
    lines.append("")

    notes = list(notes)
    if notes:
        lines.append("Path Notes")
        lines.append("----------")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("Missing Required Fields")
    lines.append("-----------------------")
    for field, rows in results["missing_required_fields"].items():
        lines.append(f"- {field}: {len(rows)}")
        if rows:
            lines.append(f"  Rows: {format_list(rows)}")
    lines.append("")

    lines.append("Rows With Non-Null Pre-Generation Fields")
    lines.append("----------------------------------------")
    for field, rows in results["nonnull_pregen_rows"].items():
        lines.append(f"- {field}: {len(rows)}")
        if rows:
            lines.append(f"  Rows: {format_list(rows)}")
    lines.append("")

    lines.append("Duplicate prompt_id Values")
    lines.append("--------------------------")
    lines.append(f"Count: {results['number_of_duplicate_prompt_ids']}")
    if results["duplicate_prompt_ids"]:
        lines.append(f"Values: {format_list(results['duplicate_prompt_ids'])}")
    else:
        lines.append("Values: none")
    lines.append("")

    lines.append("Rows With Invalid Choices")
    lines.append("-------------------------")
    lines.append(f"Count: {results['number_of_invalid_choice_rows']}")
    if results["invalid_choice_rows"]:
        lines.append(f"Rows: {format_list(results['invalid_choice_rows'])}")
    else:
        lines.append("Rows: none")
    lines.append("")

    lines.append("Incomplete Families")
    lines.append("-------------------")
    lines.append(f"Count: {results['number_of_incomplete_families']}")
    if results["list_of_incomplete_families"]:
        for family in results["list_of_incomplete_families"]:
            lines.append(f"- family_id: {family['family_id']}")
            lines.append(f"  missing_prompt_types: {', '.join(family['missing_prompt_types'])}")
            lines.append(f"  present_prompt_types: {', '.join(family['present_prompt_types'])}")
    else:
        lines.append("none")
    lines.append("")

    lines.append("Prompt Format Warnings")
    lines.append("----------------------")
    lines.append(f"Count: {len(results['prompt_format_warning_rows'])}")
    if results["prompt_format_warning_rows"]:
        lines.append(f"Rows: {format_list(results['prompt_format_warning_rows'])}")
    else:
        lines.append("Rows: none")
    lines.append("")

    lines.append("Retrieved-Evidence Rows Missing Evidence")
    lines.append("----------------------------------------")
    lines.append(f"Count: {len(results['missing_evidence_rows'])}")
    if results["missing_evidence_rows"]:
        lines.append(f"Rows: {format_list(results['missing_evidence_rows'])}")
    else:
        lines.append("Rows: none")
    lines.append("")

    lines.append("ERROR")
    lines.append("-----")
    if results["list_of_serious_errors"]:
        lines.extend(f"- {message}" for message in results["list_of_serious_errors"])
    else:
        lines.append("none")
    lines.append("")

    lines.append("WARNING")
    lines.append("-------")
    if results["list_of_warnings"]:
        lines.extend(f"- {message}" for message in results["list_of_warnings"])
    else:
        lines.append("none")
    lines.append("")

    lines.append("INFO")
    lines.append("----")
    if results["list_of_info"]:
        lines.extend(f"- {message}" for message in results["list_of_info"])
    else:
        lines.append("none")
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
    return 0 if results["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
