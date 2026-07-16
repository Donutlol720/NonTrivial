import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.validate_dataset import build_report, resolve_dataset_path, validate_rows, read_jsonl as read_dataset_jsonl
from src.load_model import load_local_model, load_tokenizer, pick_device, pick_dtype


REQUIRED_OUTPUT_FIELDS = (
    "prompt_id",
    "family_id",
    "domain",
    "title",
    "prompt_type",
    "intended_condition",
    "pressure_type",
    "has_retrieved_evidence",
    "question",
    "choice_a",
    "choice_b",
    "correct_choice",
    "false_choice",
    "user_claim_choice",
    "user_claim_truth",
    "evidence_sentence_ids_0_indexed",
    "evidence_sentences",
    "prompt",
    "generated_response",
    "parsed_answer",
    "is_correct",
    "agrees_with_user",
    "quotes_correct_evidence",
    "final_label",
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


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.inference_mode()
def generate_text(
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    prompt: str,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    input_len = int(encoded["input_ids"].shape[-1])
    gen_ids = output_ids[input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def validate_output_shape(row: Dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_OUTPUT_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Output row is missing required fields: {missing}")

    for field in ("parsed_answer", "is_correct", "agrees_with_user", "quotes_correct_evidence", "final_label"):
        if row.get(field) is not None:
            raise ValueError(f"Output row has non-null field {field} during Phase 2.")


def load_existing_outputs(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    existing: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        prompt_id = row.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id.strip():
            existing[prompt_id.strip()] = row
    return existing


def should_skip_existing(row: Mapping[str, Any]) -> bool:
    return row.get("generated_response") is not None


def write_progress(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp_path, rows)
    tmp_path.replace(path)


def sanity_check(
    input_rows: Sequence[Dict[str, Any]],
    output_rows: Sequence[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if len(output_rows) != len(input_rows):
        errors.append(f"output row count {len(output_rows)} does not match input row count {len(input_rows)}")

    for row in output_rows:
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            errors.append("missing prompt_id in output row")
            continue
        if row.get("generated_response") is None and not row.get("generation_error"):
            errors.append(f"{prompt_id}: missing generated_response and generation_error")
        for field in ("parsed_answer", "is_correct", "agrees_with_user", "quotes_correct_evidence", "final_label"):
            if row.get(field) is not None:
                errors.append(f"{prompt_id}: {field} is non-null after generation")
    return (len(errors) == 0), errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated_prompts_v1.jsonl")
    parser.add_argument("--output", default="outputs/generations_qwen2_5_1_5b.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default=os.environ.get("QWEN_DEVICE", ""))
    parser.add_argument("--dtype", default=os.environ.get("QWEN_DTYPE", ""))
    parser.add_argument("--cache-dir", default=os.environ.get("QWEN_CACHE_DIR", ""))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.max_new_tokens < 80 or args.max_new_tokens > 120:
        raise ValueError("--max-new-tokens must stay within 80 to 120 for this run configuration.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be >= 1")

    repo_root = Path(__file__).resolve().parents[2]
    dataset_path, path_notes = resolve_dataset_path(repo_root, args.input)
    rows_in = read_jsonl(dataset_path)
    if args.limit and args.limit > 0:
        rows_in = rows_in[: args.limit]

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_validation:
        validation_rows = read_dataset_jsonl(dataset_path)
        validation_results = validate_rows(validation_rows)
        report_text = build_report(dataset_path, path_notes, validation_results)
        report_path = (repo_root / "results" / "dataset_validation_report.txt").resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        if validation_results.get("error_count", validation_results["serious_error_count"]) != 0:
            raise ValueError(
                "Dataset validation failed. See results/dataset_validation_report.txt before running generation."
            )

    existing_by_id = load_existing_outputs(out_path)
    completed_ids: Set[str] = set()
    for prompt_id, existing_row in existing_by_id.items():
        if should_skip_existing(existing_row):
            completed_ids.add(prompt_id)

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    model, tokenizer_for_generate = load_local_model(
        args.model,
        device=device,
        dtype=dtype,
        cache_dir=args.cache_dir,
    )
    tokenizer = load_tokenizer(args.model, cache_dir=args.cache_dir)

    generated_rows: List[Dict[str, Any]] = []
    total_read = len(rows_in)
    total_skipped = 0
    total_generated = 0
    total_failed = 0

    for index, row in enumerate(rows_in, start=1):
        prompt_id = str(row.get("prompt_id", "")).strip()
        prompt_type = str(row.get("prompt_type", "")).strip()

        if prompt_id in completed_ids:
            generated_rows.append(existing_by_id[prompt_id])
            total_skipped += 1
        else:
            out_row = dict(row)
            out_row.pop("generation_error", None)
            try:
                prompt = str(row.get("prompt", ""))
                response = generate_text(
                    model=model,
                    tokenizer=tokenizer_for_generate,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    prompt=prompt,
                )
                out_row["generated_response"] = response
                validate_output_shape(out_row)
                total_generated += 1
            except Exception as exc:
                out_row["generated_response"] = None
                out_row["generation_error"] = str(exc)
                total_failed += 1

            generated_rows.append(out_row)

        if index % args.log_every == 0 or index == total_read:
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": total_read,
                        "current_prompt_id": prompt_id,
                        "current_prompt_type": prompt_type,
                        "generated": total_generated,
                        "skipped": total_skipped,
                        "failed": total_failed,
                        "output_path": str(out_path),
                    },
                    ensure_ascii=False,
                )
            )
            write_progress(out_path, generated_rows)

    ok, sanity_errors = sanity_check(rows_in, generated_rows)
    if not ok:
        raise ValueError("Post-run sanity check failed: " + "; ".join(sanity_errors[:20]))

    print(
        json.dumps(
            {
                "total_prompts_read": total_read,
                "total_prompts_generated": total_generated,
                "total_prompts_skipped": total_skipped,
                "total_prompts_failed": total_failed,
                "output_path": str(out_path),
                "model_id": args.model,
                "temperature": 0,
                "do_sample": False,
                "top_p": 1,
                "max_new_tokens": args.max_new_tokens,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
