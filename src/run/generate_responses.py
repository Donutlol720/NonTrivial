import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.validate_dataset import build_report, resolve_dataset_path, validate_rows, read_jsonl as read_dataset_jsonl
from src.load_model import env_default_model_id, load_local_model, load_tokenizer, pick_device, pick_dtype


REQUIRED_OUTPUT_FIELDS = (
    "prompt_id",
    "family_id",
    "domain",
    "prompt_type",
    "intended_condition",
    "pressure_type",
    "has_retrieved_evidence",
    "correct_choice",
    "false_choice",
    "user_claim_choice",
    "user_claim_truth",
    "prompt",
    "generated_response",
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

def build_prompt_text(tokenizer: Any, user_prompt: str, system_prompt: str) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    parts: List[str] = []
    for message in messages:
        parts.append(f"{message['role'].upper()}: {message['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


@torch.inference_mode()
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/generated_prompts_v1.jsonl")
    parser.add_argument("--out", default="outputs/generations_qwen2_5_1_5b.jsonl")
    parser.add_argument("--model", default=env_default_model_id())
    parser.add_argument("--device", default=os.environ.get("QWEN_DEVICE", ""))
    parser.add_argument("--dtype", default=os.environ.get("QWEN_DTYPE", ""))
    parser.add_argument("--cache-dir", default=os.environ.get("QWEN_CACHE_DIR", ""))
    parser.add_argument("--system", default=os.environ.get("QWEN_SYSTEM", ""))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true", default=False)
    args = parser.parse_args()

    if args.max_new_tokens < 80 or args.max_new_tokens > 120:
        raise ValueError("--max-new-tokens must stay within 80 to 120 for this run configuration.")

    repo_root = Path(__file__).resolve().parents[2]
    dataset_path, path_notes = resolve_dataset_path(repo_root, args.dataset)
    rows = read_jsonl(dataset_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()

    if not args.skip_validation:
        validation_rows = read_dataset_jsonl(dataset_path)
        validation_results = validate_rows(validation_rows)
        report_text = build_report(dataset_path, path_notes, validation_results)
        report_path = (repo_root / "results" / "dataset_validation_report.txt").resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        if validation_results["serious_error_count"] != 0:
            raise ValueError(
                "Dataset validation failed. See results/dataset_validation_report.txt before running generation."
            )

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
    for index, row in enumerate(rows, start=1):
        prompt = str(row.get("prompt", ""))
        prompt_text = build_prompt_text(tokenizer, user_prompt=prompt, system_prompt=args.system)
        response = generate_text(
            model=model,
            tokenizer=tokenizer_for_generate,
            prompt_text=prompt_text,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        out_row = dict(row)
        out_row["generated_response"] = response
        validate_output_shape(out_row)
        generated_rows.append(out_row)

        if index % 10 == 0:
            write_jsonl(out_path, generated_rows)

    write_jsonl(out_path, generated_rows)

    summary = {
        "dataset_path": str(dataset_path),
        "output_path": str(out_path),
        "rows_written": len(generated_rows),
        "model_id": args.model,
        "temperature": 0,
        "do_sample": False,
        "top_p": 1,
        "max_new_tokens": args.max_new_tokens,
        "path_notes": path_notes,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
