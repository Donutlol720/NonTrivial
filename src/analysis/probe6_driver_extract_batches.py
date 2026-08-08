import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


CONDITIONS_EXTRACT = [
    "evidence_neutral",
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "closed_context_false_belief_pressure",
    "evidence_true_belief_pressure",
]

ANCHOR_KEYS_REQUIRED = [
    "end_of_evidence_block",
    "end_of_user_pressure_sentence",
    "end_of_question",
    "end_of_answer_choices",
    "final_answer_position",
]


def default_python_executable() -> str:
    bundled = REPO_ROOT / ".venv_qwen4b" / "Scripts" / "python.exe"
    if bundled.exists():
        return str(bundled)
    return sys.executable


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def discover_needed_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    jsonl_rows = read_jsonl(Path(args.input))
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in jsonl_rows:
        fam = str(r.get("family_id", "")).strip()
        cond = str(r.get("prompt_type", "")).strip()
        if not fam or not cond:
            continue
        if cond not in CONDITIONS_EXTRACT:
            continue
        by_key[(fam, cond)] = r
    needed: List[Tuple[str, str]] = []
    out_root = Path(args.activation_output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for fam, cond in sorted(by_key.keys(), key=lambda x: (x[0], CONDITIONS_EXTRACT.index(x[1]) if x[1] in CONDITIONS_EXTRACT else 999)):
        out_path = out_root / fam / f"{fam}_{cond}.pt"
        if args.force_reextract:
            needed.append((fam, cond))
            continue
        if out_path.exists():
            try:
                import torch  # noqa
                import numpy as np
                loaded = torch.load(out_path, map_location="cpu")
                anchors = loaded.get("hidden_states_by_anchor", {})
                ok_keys = all(k in anchors for k in ANCHOR_KEYS_REQUIRED)
                ok_seq_len = False
                try:
                    ap = loaded.get("anchor_positions", {}) or {}
                    tsl = int(ap.get("_token_seq_len", 0))
                    # seq_len of 11 is the known-buggy state (just the suffix, no prompt)
                    # seq_len less than ~30 is also wrong for real prompt+choices+question content
                    ok_seq_len = tsl >= 50
                except Exception:
                    ok_seq_len = False
                ok_norm = False
                if ok_keys:
                    try:
                        fa = ANCHOR_KEYS_REQUIRED[0]
                        v = anchors.get(fa)
                        if v is not None:
                            if hasattr(v, "numpy"):
                                arr = v.to(dtype=torch.float32).numpy()
                            else:
                                arr = np.asarray(v, dtype=np.float32)
                            lyr0 = arr[0] if arr.ndim >= 2 else arr
                            # norm=0 would indicate zero-vector / duplicate / buggy extraction
                            ok_norm = float(np.linalg.norm(lyr0)) > 1e-6
                    except Exception:
                        ok_norm = False
                del loaded, anchors
                if ok_keys and ok_seq_len and ok_norm:
                    continue
            except Exception:
                pass
        needed.append((fam, cond))
    return needed


def chunk_into_batches(items: List[Any], size: int) -> List[List[Any]]:
    out: List[List[Any]] = []
    for i in range(0, len(items), size):
        out.append(items[i : i + size])
    return out


def run_batch(args: argparse.Namespace, batch: List[Tuple[str, str]], bidx: int, total: int) -> int:
    pairs_str = ",".join(f"{fam}={cond}" for fam, cond in batch)
    cmd = [
        args.python_exe,
        "-W",
        "ignore::FutureWarning",
        str(REPO_ROOT / "src" / "analysis" / "probe6_early_position_detection.py"),
        "--input",
        str(Path(args.input).resolve()),
        "--family-deltas",
        str(Path(args.family_deltas).resolve()),
        "--prompt-dataset",
        str(Path(args.prompt_dataset).resolve()),
        "--model",
        args.model,
        "--activation-output-root",
        str(Path(args.activation_output_root).resolve()),
        "--only-extract-and-exit",
        "--only-family-conditions",
        pairs_str,
        "--worker-batch-index",
        str(bidx),
        "--worker-total-batches",
        str(total),
    ]
    if args.force_reextract:
        cmd.append("--force-reextract")

    env = os.environ.copy()
    env["HF_HOME"] = str(Path(args.hf_home).resolve())
    env["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE") or "1"
    env["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE") or "1"
    env["OVERRIDE_DEVICE"] = os.environ.get("OVERRIDE_DEVICE") or "cpu"
    env["OVERRIDE_DTYPE"] = os.environ.get("OVERRIDE_DTYPE") or "float32"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    if args.cpu_max_memory_gib:
        mem = int(args.cpu_max_memory_gib)
        env["CPU_MAX_MEMORY_GIB"] = str(mem)
        offload_folder = REPO_ROOT.resolve() / "temp_offload_folder"
        offload_folder.mkdir(parents=True, exist_ok=True)
        env["CPU_OFFLOAD_FOLDER"] = str(offload_folder.resolve())
    token = env.get("HF_TOKEN") or ""
    if not token:
        try:
            from_env_user = os.environ.get("HF_TOKEN") or ""
            if from_env_user:
                token = from_env_user
                env["HF_TOKEN"] = token
        except Exception:
            pass
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token

    print(json.dumps({
        "status": "driver_spawn_batch",
        "batch_index": bidx,
        "total_batches": total,
        "batch_size": len(batch),
        "pairs": pairs_str,
    }), flush=True)
    t0 = time.time()
    completed = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    print(json.dumps({
        "status": "driver_batch_done",
        "batch_index": bidx,
        "total_batches": total,
        "returncode": completed.returncode,
        "elapsed_sec": int(elapsed),
    }), flush=True)
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / "outputs" / "state_logits_qwen3_4b_instruct_2507_all_families.jsonl"))
    parser.add_argument("--family-deltas", default=str(REPO_ROOT / "results" / "qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"))
    parser.add_argument("--prompt-dataset", default=str(REPO_ROOT / "data" / "generated_prompts_v1.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--activation-output-root", default=str(REPO_ROOT / "activations" / "qwen3_4b_instruct_2507_early_positions"))
    parser.add_argument("--python-exe", default=default_python_executable())
    parser.add_argument("--hf-home", default=str(REPO_ROOT / "model_cache"))
    parser.add_argument("--hf-token", default="", help="Optional: if set, overwrites env HF_TOKEN for subprocess workers.")
    parser.add_argument("--batch-size", type=int, default=4, help="Extractions per worker subprocess. Smaller = lower peak memory but slower.")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--cpu-max-memory-gib", type=int, default=0, help="If >0, set CPU_MAX_MEMORY_GIB on workers for low-memory offloading.")
    parser.add_argument("--skip-final-classification", action="store_true", help="After extraction, do NOT re-run the full Probe 6 classification/summary/permutation steps.")
    parser.add_argument("--skip-permutation", action="store_true", help="Pass through to probe6 final classification.")
    parser.add_argument("--permutation-pairs", type=int, default=100, help="Pass through to probe6 final classification.")
    args = parser.parse_args()

    for required_path, purpose, hint in [
        (Path(args.input), "state/logit extraction JSONL", "Run src/extraction/extract_multi_family_states_and_logits.py first."),
        (Path(args.family_deltas), "family delta CSV", "Run src/analysis/compute_qwen3_4b_family36_family_margin_deltas.py after extraction."),
        (Path(args.prompt_dataset), "prompt dataset JSONL", "The repo should already contain data/generated_prompts_v1.jsonl."),
    ]:
        path = required_path if required_path.is_absolute() else (REPO_ROOT / required_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing {purpose}: {path}\n{hint}")

    Path(args.activation_output_root).mkdir(parents=True, exist_ok=True)

    import shutil
    default_offload = REPO_ROOT.resolve() / "temp_offload_folder"
    if args.cpu_max_memory_gib and default_offload.exists():
        try:
            shutil.rmtree(default_offload, ignore_errors=True)
        except Exception:
            pass
    if args.cpu_max_memory_gib:
        default_offload.mkdir(parents=True, exist_ok=True)
    for extra in [
        REPO_ROOT.resolve() / "activations" / "temp_offload_folder",
        Path(args.activation_output_root).resolve() / "temp_offload_folder",
    ]:
        if extra.exists():
            try:
                shutil.rmtree(extra, ignore_errors=True)
            except Exception:
                pass

    needed = discover_needed_pairs(args)
    print(json.dumps({
        "status": "driver_discovery",
        "n_needed_remaining": len(needed),
        "batch_size": int(args.batch_size),
    }), flush=True)
    if not needed:
        print(json.dumps({"status": "driver_nothing_to_extract"}), flush=True)
    else:
        batches = chunk_into_batches(needed, max(1, int(args.batch_size)))
        total = len(batches)
        last_nonzero = 0
        for bidx, batch in enumerate(batches):
            rc = run_batch(args, batch, bidx, total)
            if rc != 0:
                last_nonzero = rc
                print(json.dumps({
                    "status": "driver_batch_failed",
                    "batch_index": bidx,
                    "returncode": rc,
                    "action": "stop_driver",
                }), flush=True)
                raise SystemExit(rc)

    if args.skip_final_classification:
        print(json.dumps({"status": "driver_skip_final_classification"}), flush=True)
        return

    final_cmd = [
        args.python_exe,
        "-W",
        "ignore::FutureWarning",
        str(REPO_ROOT / "src" / "analysis" / "probe6_early_position_detection.py"),
        "--input",
        str(Path(args.input).resolve()),
        "--family-deltas",
        str(Path(args.family_deltas).resolve()),
        "--prompt-dataset",
        str(Path(args.prompt_dataset).resolve()),
        "--model",
        args.model,
        "--activation-output-root",
        str(Path(args.activation_output_root).resolve()),
        "--skip-extraction",
        "--permutation-pairs",
        str(int(args.permutation_pairs)),
    ]
    if args.skip_permutation:
        final_cmd.append("--skip-permutation")
    env = os.environ.copy()
    env["HF_HOME"] = str(Path(args.hf_home).resolve())
    env["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE") or "1"
    env["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE") or "1"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    elif not env.get("HF_TOKEN"):
        user_tok = os.environ.get("HF_TOKEN") or ""
        if user_tok:
            env["HF_TOKEN"] = user_tok
    print(json.dumps({
        "status": "driver_start_final_probe6",
        "skip_permutation": bool(args.skip_permutation),
        "permutation_pairs": int(args.permutation_pairs),
    }), flush=True)
    completed = subprocess.run(final_cmd, env=env, cwd=str(REPO_ROOT))
    print(json.dumps({"status": "driver_complete", "returncode": int(completed.returncode)}), flush=True)
    raise SystemExit(int(completed.returncode))


if __name__ == "__main__":
    main()
