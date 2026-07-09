import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_step(command: List[str], cwd: Path) -> None:
    print("RUNNING:", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/generated_prompts_v1.jsonl")
    parser.add_argument("--report", default="results/dataset_validation_report.txt")
    parser.add_argument("--generations-out", default="outputs/generations_qwen2_5_1_5b.jsonl")
    parser.add_argument("--labeled-out", default="outputs/labeled_generations_qwen2_5_1_5b.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="")
    parser.add_argument("--dtype", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--system", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-label", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    python_exe = sys.executable

    validate_cmd = [
        python_exe,
        "src/data/validate_dataset.py",
        "--dataset",
        args.dataset,
        "--report",
        args.report,
    ]
    run_step(validate_cmd, repo_root)

    generate_cmd = [
        python_exe,
        "src/run/generate_responses.py",
        "--dataset",
        args.dataset,
        "--out",
        args.generations_out,
        "--model",
        args.model,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.device:
        generate_cmd.extend(["--device", args.device])
    if args.dtype:
        generate_cmd.extend(["--dtype", args.dtype])
    if args.cache_dir:
        generate_cmd.extend(["--cache-dir", args.cache_dir])
    if args.system:
        generate_cmd.extend(["--system", args.system])
    if args.limit:
        generate_cmd.extend(["--limit", str(args.limit)])
    run_step(generate_cmd, repo_root)

    if not args.skip_label:
        label_cmd = [
            python_exe,
            "src/run/parse_and_label_responses.py",
            "--input",
            args.generations_out,
            "--out",
            args.labeled_out,
        ]
        run_step(label_cmd, repo_root)


if __name__ == "__main__":
    main()
