import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_FAMILY_DELTAS = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_PROMPT_DATASET = "data/generated_prompts_v1.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ACTIVATION_OUTPUT_ROOT = "activations/qwen3_4b_instruct_2507_early_positions"
DEFAULT_LAYERWISE_OUTPUT = "results/probe6_early_position_layerwise.csv"
DEFAULT_SUMMARY_OUTPUT = "results/probe6_early_position_summary.txt"
DEFAULT_BEST_LAYERS_OUTPUT = "results/probe6_early_position_best.csv"
DEFAULT_DETECTABLE_THRESHOLD = 0.55
MIN_TRAIN_CLASS_COUNT = 3
POST_SCALE_CLIP_ABS = 20.0
SCORE_CLIP_ABS = POST_SCALE_CLIP_ABS * POST_SCALE_CLIP_ABS * 2560 + POST_SCALE_CLIP_ABS

ANCHOR_ORDER = [
    "end_of_evidence_block",
    "end_of_user_pressure_sentence",
    "end_of_question",
    "end_of_answer_choices",
    "final_answer_position",
]
ANCHOR_DISPLAY = {
    "end_of_evidence_block": "E0: end of evidence block",
    "end_of_user_pressure_sentence": "E1: end of user-pressure sentence",
    "end_of_question": "E2: end of question",
    "end_of_answer_choices": "E3: end of answer choices",
    "final_answer_position": "E4: final ANSWER:",
}
CONDITIONS = [
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "closed_context_false_belief_pressure",
]
NEUTRAL_CONDITION = "evidence_neutral"
EXTRACTION_CONDITIONS = [
    NEUTRAL_CONDITION,
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "closed_context_false_belief_pressure",
    "evidence_true_belief_pressure",
]
CONDITION_DELTA_COLUMN_MAP: Dict[str, str] = {
    "evidence_false_belief_pressure": "delta_false_pressure",
    "evidence_emotional_pressure": "delta_emotional_pressure",
    "closed_context_false_belief_pressure": "delta_closed_context",
}


# #region debug-point A-E:probe6-runtime-instrumentation
def _debug_env() -> Tuple[str, str]:
    env_path = REPO_ROOT / ".dbg" / "probe6-numeric-overflow.env"
    debug_url = "http://127.0.0.1:7777/event"
    session_id = "probe6-numeric-overflow"
    try:
        contents = env_path.read_text(encoding="utf-8")
        for line in contents.splitlines():
            if line.startswith("DEBUG_SERVER_URL="):
                debug_url = line.split("=", 1)[1].strip() or debug_url
            elif line.startswith("DEBUG_SESSION_ID="):
                session_id = line.split("=", 1)[1].strip() or session_id
    except Exception:
        pass
    return debug_url, session_id


def _debug_post(run_id: str, hypothesis_id: str, location: str, msg: str, data: Mapping[str, Any]) -> None:
    try:
        debug_url, session_id = _debug_env()
        effective_run_id = os.environ.get("DEBUG_RUN_ID", run_id)
        payload = {
            "sessionId": session_id,
            "runId": effective_run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": dict(data),
            "ts": int(time.time() * 1000),
        }
        request = urllib.request.Request(
            debug_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=1.5).read()
    except Exception:
        pass
# #endregion


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


def resolve_required_path(path_str: str, purpose: str, hint: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}\n{hint}")
    return path


def read_family_deltas(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return out

    first = rows[0]
    if "condition" in first and "delta_margin" in first:
        for row in rows:
            family_id = str(row.get("family_id", "")).strip()
            condition = str(row.get("condition", "")).strip()
            if not family_id or not condition:
                continue
            try:
                delta_margin = float(row["delta_margin"]) if row.get("delta_margin") not in ("", None) else None
            except (TypeError, ValueError):
                delta_margin = None
            out[(family_id, condition)] = {"delta_margin": delta_margin}
        return out

    for row in rows:
        family_id = str(row.get("family_id", "")).strip()
        if not family_id:
            continue
        for condition, column_name in CONDITION_DELTA_COLUMN_MAP.items():
            value = row.get(column_name)
            try:
                delta_margin = float(value) if value not in ("", None) else None
            except (TypeError, ValueError):
                delta_margin = None
            out[(family_id, condition)] = {"delta_margin": delta_margin}
    return out


def label_primary(delta_margin: Optional[float]) -> Optional[int]:
    if delta_margin is None:
        return None
    return 1 if delta_margin < 0.0 else 0


def load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    cache_dir = os.environ.get("HF_HOME", str(REPO_ROOT / "model_cache"))
    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )


def _min_positive(values: Iterable[int]) -> int:
    positives = [value for value in values if value >= 0]
    return min(positives) if positives else -1


def find_anchor_positions(prompt_text: str, answer_logit_prompt_suffix: str, tokenizer: Any) -> Dict[str, int]:
    suffix_start = prompt_text.rfind(answer_logit_prompt_suffix)
    base_text = prompt_text if suffix_start < 0 else prompt_text[:suffix_start]
    base_text = base_text.rstrip()

    first_structural_marker = _min_positive(
        [
            base_text.find("\n\nRetrieved document:"),
            base_text.find("\n\nContext:"),
            base_text.find("\nQuestion:"),
        ]
    )
    end_prefix_pos = first_structural_marker - 1 if first_structural_marker >= 0 else len(base_text) - 1

    end_evidence_pos = -1
    for start_marker in ("\n\nRetrieved document:", "\n\nContext:"):
        start_idx = base_text.find(start_marker)
        if start_idx < 0:
            continue
        question_idx = base_text.find("\nQuestion:", start_idx + len(start_marker))
        if question_idx >= 0:
            end_evidence_pos = max(end_evidence_pos, question_idx - 1)

    question_idx = base_text.rfind("\nQuestion:")
    end_question_pos = -1
    if question_idx >= 0:
        next_idx = _min_positive(
            [
                base_text.find("\nChoices:", question_idx + len("\nQuestion:")),
                base_text.find("\n\nAnswer with exactly this format:", question_idx + len("\nQuestion:")),
            ]
        )
        if next_idx < 0:
            next_idx = len(base_text)
        end_question_pos = next_idx - 1

    choices_idx = base_text.rfind("\nChoices:")
    end_choices_pos = -1
    if choices_idx >= 0:
        next_idx = _min_positive(
            [
                base_text.find("\n\nAnswer with exactly this format:", choices_idx + len("\nChoices:")),
                base_text.find("\n\nAnswer with only A or B.", choices_idx + len("\nChoices:")),
            ]
        )
        if next_idx < 0:
            next_idx = len(base_text)
        end_choices_pos = next_idx - 1

    answer_prompt = base_text + "\n\nAnswer with only A or B.\n\nANSWER:"
    encoded = tokenizer(
        answer_prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="np",
    )
    offsets = np.asarray(encoded["offset_mapping"][0])
    token_seq_len = int(offsets.shape[0])

    def char_to_token(char_pos: int) -> int:
        if token_seq_len <= 0:
            return 0
        if char_pos < 0:
            return 0
        pos = char_pos + 1
        hits = np.where((offsets[:, 0] <= pos) & (offsets[:, 1] >= pos))[0]
        if len(hits) > 0:
            return int(hits[-1])
        fallback = np.where(offsets[:, 0] < pos)[0]
        if len(fallback) > 0:
            return int(fallback[-1])
        return 0

    return {
        "end_of_evidence_block": char_to_token(end_evidence_pos) if end_evidence_pos >= 0 else 0,
        "end_of_user_pressure_sentence": char_to_token(end_prefix_pos) if end_prefix_pos >= 0 else 0,
        "end_of_question": char_to_token(end_question_pos) if end_question_pos >= 0 else 0,
        "end_of_answer_choices": char_to_token(end_choices_pos) if end_choices_pos >= 0 else 0,
        "final_answer_position": token_seq_len - 1,
        "_token_seq_len": token_seq_len,
    }


@torch.inference_mode()
def run_forward_multi_position(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    answer_logit_suffix: str,
    device: str,
    token_positions: Mapping[str, int],
) -> Dict[str, Any]:
    import gc

    suffix_start = prompt_text.rfind(answer_logit_suffix)
    base_text = prompt_text if suffix_start < 0 else prompt_text[:suffix_start]
    base_text = base_text.rstrip()
    full_text = base_text + "\n\nAnswer with only A or B.\n\nANSWER:"
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
        attn_implementation="eager",
    )
    logits = outputs.logits[0, -1, :].detach().to("cpu", dtype=torch.float32).numpy()
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) <= 1:
        raise RuntimeError("Model did not return hidden states")

    n_layers = len(hidden_states) - 1
    hidden_dim = int(hidden_states[1].shape[-1])
    seq_len = int(inputs["input_ids"].shape[1])
    layer_vectors: Dict[str, np.ndarray] = {
        anchor: np.zeros((n_layers, hidden_dim), dtype=np.float16) for anchor in ANCHOR_ORDER
    }
    del inputs

    for anchor in ANCHOR_ORDER:
        pos = min(int(token_positions.get(anchor, 0)), seq_len - 1)
        for layer_offset in range(n_layers):
            vec = hidden_states[layer_offset + 1][0, pos, :].detach().to("cpu", dtype=torch.float16).numpy()
            layer_vectors[anchor][layer_offset] = vec

    del hidden_states, outputs
    if hasattr(torch.cuda, "empty_cache"):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()
    return {
        "logits_last_token": logits,
        "hidden_states_by_anchor": layer_vectors,
        "token_seq_len": seq_len,
    }


def build_answer_logit_suffix() -> str:
    return "\n\nAnswer with exactly this format:"


def extract_early_activations(args: argparse.Namespace) -> Path:
    from load_model import load_local_model, pick_device, pick_dtype

    input_path = resolve_required_path(
        args.input,
        "extraction JSONL",
        "Run src/extraction/extract_multi_family_states_and_logits.py to produce the state/logit dataset first.",
    )
    prompt_dataset_path = resolve_required_path(
        args.prompt_dataset,
        "prompt dataset JSONL",
        "The early-position extractor needs data/generated_prompts_v1.jsonl.",
    )

    worker_bidx = int(getattr(args, "worker_batch_index", -1))
    worker_total = int(getattr(args, "worker_total_batches", -1))

    prompts_all = read_jsonl(prompt_dataset_path)
    prompts_by_pid = {str(row["prompt_id"]): row for row in prompts_all if "prompt_id" in row}
    existing_rows = read_jsonl(input_path)

    by_family_condition: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in existing_rows:
        family_id = str(row.get("family_id", "")).strip()
        condition = str(row.get("prompt_type", "")).strip()
        if family_id and condition in EXTRACTION_CONDITIONS:
            by_family_condition[(family_id, condition)] = row

    only_pairs: Optional[set[Tuple[str, str]]] = None
    if getattr(args, "only_family_conditions", ""):
        only_pairs = set()
        for piece in str(args.only_family_conditions).split(","):
            piece = piece.strip()
            if not piece or "=" not in piece:
                continue
            family_id, condition = piece.split("=", 1)
            only_pairs.add((family_id.strip(), condition.strip()))

    families = sorted({family_id for family_id, _ in by_family_condition.keys()})

    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    device = os.environ.get("OVERRIDE_DEVICE") or pick_device("")
    if os.environ.get("OVERRIDE_DTYPE"):
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(os.environ["OVERRIDE_DTYPE"], pick_dtype(device, ""))
    else:
        dtype = torch.float32 if device == "cpu" else pick_dtype(device, "")

    cache_dir = os.environ.get("HF_HOME", str(REPO_ROOT / "model_cache"))
    cpu_max_memory_gib = int(os.environ.get("CPU_MAX_MEMORY_GIB", "0") or "0")
    offload_folder = os.environ.get("CPU_OFFLOAD_FOLDER", "")
    if cpu_max_memory_gib > 0 and offload_folder:
        os.makedirs(offload_folder, exist_ok=True)

    print(
        json.dumps(
            {
                "status": "loading_model",
                "model": args.model,
                "device": device,
                "dtype": str(dtype),
                "cpu_max_memory_gib": cpu_max_memory_gib,
                "offload_folder": offload_folder,
            }
        ),
        flush=True,
    )
    model, tokenizer = load_local_model(
        args.model,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        trust_remote_code=False,
        cpu_max_memory_gib=cpu_max_memory_gib,
        offload_folder=offload_folder,
    )

    suffix = build_answer_logit_suffix()
    output_root = Path(args.activation_output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    needed_pairs: List[Tuple[str, str]] = []
    for family_id in families:
        for condition in EXTRACTION_CONDITIONS:
            key = (family_id, condition)
            if key not in by_family_condition:
                continue
            if only_pairs is not None and key not in only_pairs:
                continue
            needed_pairs.append(key)

    manifest_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    print(
        json.dumps(
            {
                "status": "worker_batch_start",
                "worker_batch_index": worker_bidx,
                "worker_total_batches": worker_total,
                "n_pairs_in_batch": len(needed_pairs),
                "extraction_root": str(output_root),
            }
        ),
        flush=True,
    )

    import gc

    try:
        for index, (family_id, condition) in enumerate(needed_pairs, start=1):
            ref_row = by_family_condition[(family_id, condition)]
            prompt_id = str(ref_row["prompt_id"])
            prompt_row = prompts_by_pid.get(prompt_id)
            if prompt_row is None:
                raise KeyError(f"Prompt ID {prompt_id} is missing from {prompt_dataset_path}")
            prompt_text = str(prompt_row.get("prompt", ""))

            family_dir = output_root / family_id
            family_dir.mkdir(parents=True, exist_ok=True)
            out_path = family_dir / f"{family_id}_{condition}.pt"

            if out_path.exists() and not args.force_reextract:
                try:
                    loaded = torch.load(out_path, map_location="cpu")
                    hs_anchor = loaded.get("hidden_states_by_anchor", {})
                    anchor_positions = loaded.get("anchor_positions", {}) or {}
                    has_all_anchors = all(anchor in hs_anchor for anchor in ANCHOR_ORDER)
                    seq_len_ok = int(anchor_positions.get("_token_seq_len", 0)) >= 50
                    norm_ok = False
                    if has_all_anchors:
                        arr = hs_anchor[ANCHOR_ORDER[0]]
                        if isinstance(arr, torch.Tensor):
                            arr = arr.to(dtype=torch.float32).numpy()
                        else:
                            arr = np.asarray(arr, dtype=np.float32)
                        norm_ok = float(np.linalg.norm(arr[0])) > 1e-6
                    if has_all_anchors and seq_len_ok and norm_ok:
                        manifest_rows.append(
                            {
                                "family_id": family_id,
                                "condition": condition,
                                "prompt_id": prompt_id,
                                "activation_path": str(out_path.relative_to(REPO_ROOT)),
                                "skipped_existing": 1,
                            }
                        )
                        continue
                except Exception:
                    pass

            anchors = find_anchor_positions(prompt_text, suffix, tokenizer)
            result = run_forward_multi_position(model, tokenizer, prompt_text, suffix, device, anchors)
            record = {
                "family_id": family_id,
                "condition": condition,
                "prompt_id": prompt_id,
                "prompt_type": condition,
                "answer_logit_prompt": str(ref_row.get("answer_logit_prompt", "")),
                "model_name": args.model,
                "anchor_positions": dict(anchors),
                "hidden_states_by_anchor": {
                    anchor: torch.from_numpy(result["hidden_states_by_anchor"][anchor]) for anchor in ANCHOR_ORDER
                },
                "logits_last_token": torch.from_numpy(result["logits_last_token"]),
                "token_seq_len": int(result["token_seq_len"]),
            }
            torch.save(record, out_path)
            manifest_rows.append(
                {
                    "family_id": family_id,
                    "condition": condition,
                    "prompt_id": prompt_id,
                    "activation_path": str(out_path.relative_to(REPO_ROOT)),
                    "skipped_existing": 0,
                }
            )

            if index % 6 == 0 or index == len(needed_pairs):
                elapsed = max(time.time() - t0, 1e-9)
                print(
                    json.dumps(
                        {
                            "status": "extracted_batch",
                            "n_done": index,
                            "n_total": len(needed_pairs),
                            "elapsed_sec": int(elapsed),
                            "examples_per_sec": round(index / elapsed, 4),
                        }
                    ),
                    flush=True,
                )
                if hasattr(torch.cuda, "empty_cache"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
    finally:
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        if hasattr(torch.cuda, "empty_cache"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()

    if not manifest_rows:
        raise RuntimeError("No manifest rows produced.")
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def load_activation_by_anchor(path: Path) -> Dict[str, np.ndarray]:
    record = torch.load(path, map_location="cpu")
    raw = record["hidden_states_by_anchor"]
    out: Dict[str, np.ndarray] = {}
    for anchor in ANCHOR_ORDER:
        tensor = raw.get(anchor)
        if tensor is None:
            raise RuntimeError(f"Missing anchor {anchor} in {path}")
        if isinstance(tensor, torch.Tensor):
            out[anchor] = tensor.to(dtype=torch.float32).numpy()
        else:
            out[anchor] = np.asarray(tensor, dtype=np.float32)
    return out


def collect_dataset(
    jsonl_rows: Sequence[Mapping[str, Any]],
    family_deltas: Mapping[Tuple[str, str], Mapping[str, Any]],
    activation_root: Path,
) -> Tuple[Dict[Tuple[str, str, str], np.ndarray], Dict[Tuple[str, str, str], Dict[str, Any]], int]:
    by_family_condition: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in jsonl_rows:
        family_id = str(row.get("family_id", "")).strip()
        condition = str(row.get("prompt_type", "")).strip()
        if family_id and condition:
            by_family_condition[(family_id, condition)] = dict(row)

    deltas: Dict[Tuple[str, str, str], np.ndarray] = {}
    metadata: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    layer_count = 0

    for family_id in sorted({family for family, _ in by_family_condition.keys()}):
        neutral_row = by_family_condition.get((family_id, NEUTRAL_CONDITION))
        if neutral_row is None:
            continue
        neutral_path = activation_root / family_id / f"{family_id}_{NEUTRAL_CONDITION}.pt"
        if not neutral_path.exists():
            continue
        neutral_by_anchor = load_activation_by_anchor(neutral_path)
        layer_count = int(neutral_by_anchor[ANCHOR_ORDER[0]].shape[0])
        for condition in CONDITIONS:
            condition_row = by_family_condition.get((family_id, condition))
            if condition_row is None:
                continue
            condition_path = activation_root / family_id / f"{family_id}_{condition}.pt"
            if not condition_path.exists():
                continue
            condition_by_anchor = load_activation_by_anchor(condition_path)
            delta_margin = family_deltas.get((family_id, condition), {}).get("delta_margin")
            label = label_primary(delta_margin)
            for anchor in ANCHOR_ORDER:
                key = (family_id, condition, anchor)
                deltas[key] = (condition_by_anchor[anchor] - neutral_by_anchor[anchor]).astype(np.float32)
                metadata[key] = {
                    "family_id": family_id,
                    "condition": condition,
                    "anchor": anchor,
                    "delta_margin": delta_margin,
                    "harmful_label_primary": label,
                }
    # #region debug-point A:dataset-finiteness
    if deltas:
        total_keys = len(deltas)
        nonfinite_keys = 0
        worst_key = None
        worst_abs = -1.0
        by_condition_anchor: Dict[str, Dict[str, float]] = {}
        for key, arr in deltas.items():
            finite = np.isfinite(arr)
            if not finite.all():
                nonfinite_keys += 1
            max_abs = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
            if max_abs > worst_abs:
                worst_abs = max_abs
                worst_key = key
            bucket = by_condition_anchor.setdefault(f"{key[1]}::{key[2]}", {"count": 0.0, "max_abs": 0.0, "nonfinite": 0.0})
            bucket["count"] += 1.0
            bucket["max_abs"] = max(bucket["max_abs"], max_abs)
            bucket["nonfinite"] += float(not finite.all())
        top_buckets = sorted(by_condition_anchor.items(), key=lambda item: item[1]["max_abs"], reverse=True)[:5]
        _debug_post(
            "pre-fix",
            "A",
            "probe6.collect_dataset",
            "[DEBUG] Collected early-position delta tensor summary",
            {
                "total_keys": total_keys,
                "nonfinite_keys": nonfinite_keys,
                "worst_key": list(worst_key) if worst_key is not None else None,
                "worst_abs": worst_abs,
                "top_buckets": [
                    {
                        "bucket": name,
                        "count": int(stats["count"]),
                        "max_abs": stats["max_abs"],
                        "nonfinite": int(stats["nonfinite"]),
                    }
                    for name, stats in top_buckets
                ],
            },
        )
    # #endregion
    return deltas, metadata, layer_count


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) >= 2 else float("nan")


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob)) if len(set(y_true.tolist())) >= 2 else float("nan")


def majority_baseline_predictions(train_y: np.ndarray, n_test: int) -> np.ndarray:
    if train_y.size == 0:
        return np.zeros(n_test, dtype=np.int64)
    counts = np.bincount(train_y.astype(np.int64))
    majority = int(np.argmax(counts))
    return np.full(n_test, majority, dtype=np.int64)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    baseline_pred: np.ndarray,
) -> Dict[str, Any]:
    ba = float(balanced_accuracy_score(y_true, y_pred))
    baseline_ba = float(balanced_accuracy_score(y_true, baseline_pred))
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    except Exception:
        tn = fp = fn = tp = 0
    return {
        "balanced_accuracy": ba,
        "baseline_balanced_accuracy": baseline_ba,
        "auroc": safe_auroc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_counts": f"tn={tn} fp={fp} fn={fn} tp={tp}",
    }


def centroid_probe_scores(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    class0 = train_x[train_y == 0]
    class1 = train_x[train_y == 1]
    if class0.size == 0 or class1.size == 0:
        raise ValueError("Both classes are required for the centroid probe.")
    centroid0 = class0.mean(axis=0)
    centroid1 = class1.mean(axis=0)
    weight = centroid1 - centroid0
    bias = -0.5 * (float(np.dot(centroid1, centroid1)) - float(np.dot(centroid0, centroid0)))
    test_x = stabilize_scaled_features(np.asarray(test_x, dtype=np.float64))
    weight = stabilize_scaled_features(np.asarray(weight, dtype=np.float64))
    bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=POST_SCALE_CLIP_ABS, neginf=-POST_SCALE_CLIP_ABS), -POST_SCALE_CLIP_ABS, POST_SCALE_CLIP_ABS))
    if (not np.isfinite(test_x).all()) or (not np.isfinite(weight).all()):
        _debug_post(
            "post-fix",
            "B",
            "probe6.centroid_probe_scores",
            "[DEBUG] Non-finite values reached centroid scoring after stabilization",
            {
                "test_x_finite": bool(np.isfinite(test_x).all()),
                "weight_finite": bool(np.isfinite(weight).all()),
                "test_x_max_abs": float(np.nanmax(np.abs(test_x))) if test_x.size else 0.0,
                "weight_max_abs": float(np.nanmax(np.abs(weight))) if weight.size else 0.0,
            },
        )
        raise ValueError("Non-finite values reached centroid scoring.")
    # Use a bounded elementwise accumulation instead of BLAS matmul to avoid
    # platform-specific overflow warnings on tiny, degenerate high-d folds.
    scores = np.sum(test_x * weight[np.newaxis, :], axis=1, dtype=np.float64) + bias
    scores = np.clip(np.nan_to_num(scores, nan=0.0, posinf=SCORE_CLIP_ABS, neginf=-SCORE_CLIP_ABS), -SCORE_CLIP_ABS, SCORE_CLIP_ABS)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    preds = (scores >= 0.0).astype(np.int64)
    return scores, preds


def stabilize_scaled_features(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=POST_SCALE_CLIP_ABS, neginf=-POST_SCALE_CLIP_ABS)
    return np.clip(x, -POST_SCALE_CLIP_ABS, POST_SCALE_CLIP_ABS)


def run_family_heldout_classification(
    examples: Sequence[Tuple[str, str, str]],
    tensor: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    layer_count: int,
    *,
    permute_seed: Optional[int] = None,
    restrict_layers: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    logo = LeaveOneGroupOut()
    folds = list(logo.split(np.arange(len(examples)), y, groups=groups))
    rows: List[Dict[str, Any]] = []

    for layer in range(layer_count):
        if restrict_layers is not None and layer not in restrict_layers:
            continue
        X_layer = tensor[:, layer, :]
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        y_prob_all: List[np.ndarray] = []
        baseline_all: List[np.ndarray] = []
        valid_fold_ids: List[int] = []

        for fold_id, (train_idx, test_idx) in enumerate(folds):
            train_x = np.asarray(X_layer[train_idx], dtype=np.float64)
            test_x = np.asarray(X_layer[test_idx], dtype=np.float64)
            train_y = y[train_idx].copy()
            test_y = y[test_idx]
            # #region debug-point B-C:family-heldout-fold
            pre_scale_train_max_abs = float(np.nanmax(np.abs(train_x))) if train_x.size else 0.0
            pre_scale_test_max_abs = float(np.nanmax(np.abs(test_x))) if test_x.size else 0.0
            pre_scale_train_finite = bool(np.isfinite(train_x).all())
            pre_scale_test_finite = bool(np.isfinite(test_x).all())
            if (not pre_scale_train_finite) or (not pre_scale_test_finite) or pre_scale_train_max_abs > 1e4 or pre_scale_test_max_abs > 1e4:
                _debug_post(
                    "pre-fix",
                    "B",
                    "probe6.run_family_heldout_classification",
                    "[DEBUG] Suspicious feature magnitude before scaling",
                    {
                        "layer": layer,
                        "fold_id": fold_id,
                        "n_train": int(train_x.shape[0]),
                        "n_test": int(test_x.shape[0]),
                        "train_finite": pre_scale_train_finite,
                        "test_finite": pre_scale_test_finite,
                        "train_max_abs": pre_scale_train_max_abs,
                        "test_max_abs": pre_scale_test_max_abs,
                    },
                )
            label_values, label_counts = np.unique(train_y, return_counts=True)
            if len(label_values) < 2 or int(np.min(label_counts)) < MIN_TRAIN_CLASS_COUNT:
                _debug_post(
                    "pre-fix",
                    "C",
                    "probe6.run_family_heldout_classification",
                    "[DEBUG] Degenerate or near-degenerate train label split",
                    {
                        "layer": layer,
                        "fold_id": fold_id,
                        "labels": label_values.tolist(),
                        "counts": label_counts.tolist(),
                    },
                )
            # #endregion
            if len(label_values) < 2 or int(np.min(label_counts)) < MIN_TRAIN_CLASS_COUNT:
                continue
            if permute_seed is not None:
                rng = np.random.default_rng(permute_seed + layer * 1009 + fold_id)
                train_y = rng.permutation(train_y)
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            train_x = stabilize_scaled_features(train_x)
            test_x = stabilize_scaled_features(test_x)
            # #region debug-point B:family-heldout-post-scale
            post_scale_train_max_abs = float(np.nanmax(np.abs(train_x))) if train_x.size else 0.0
            post_scale_test_max_abs = float(np.nanmax(np.abs(test_x))) if test_x.size else 0.0
            if (not np.isfinite(train_x).all()) or (not np.isfinite(test_x).all()) or post_scale_train_max_abs > 1e4 or post_scale_test_max_abs > 1e4:
                _debug_post(
                    "pre-fix",
                    "B",
                    "probe6.run_family_heldout_classification",
                    "[DEBUG] Suspicious feature magnitude after scaling",
                    {
                        "layer": layer,
                        "fold_id": fold_id,
                        "train_max_abs": post_scale_train_max_abs,
                        "test_max_abs": post_scale_test_max_abs,
                        "train_finite": bool(np.isfinite(train_x).all()),
                        "test_finite": bool(np.isfinite(test_x).all()),
                        "permute_seed": permute_seed,
                    },
                )
            # #endregion
            if (not np.isfinite(train_x).all()) or (not np.isfinite(test_x).all()):
                continue
            try:
                y_prob, y_pred = centroid_probe_scores(train_x, train_y, test_x)
            except Exception:
                y_prob = np.full(len(test_x), 0.5, dtype=np.float64)
                y_pred = np.zeros(len(test_x), dtype=np.int64)
            y_true_all.append(test_y)
            y_pred_all.append(y_pred)
            y_prob_all.append(y_prob)
            baseline_all.append(majority_baseline_predictions(train_y, len(test_y)))
            valid_fold_ids.append(fold_id)

        if not y_true_all:
            continue
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        y_prob = np.concatenate(y_prob_all)
        baseline_pred = np.concatenate(baseline_all)
        metrics = evaluate_predictions(y_true, y_pred, y_prob, baseline_pred)
        rows.append(
            {
                "layer": layer,
                "n_examples": int(y_true.shape[0]),
                "n_families": len(valid_fold_ids),
                "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
                "baseline_balanced_accuracy": f"{metrics['baseline_balanced_accuracy']:.6f}",
                "auroc": "" if np.isnan(metrics["auroc"]) else f"{metrics['auroc']:.6f}",
                "average_precision": "" if np.isnan(metrics["average_precision"]) else f"{metrics['average_precision']:.6f}",
                "f1": f"{metrics['f1']:.6f}",
                "precision": f"{metrics['precision']:.6f}",
                "recall": f"{metrics['recall']:.6f}",
                "confusion_matrix_counts": metrics["confusion_matrix_counts"],
            }
        )
    return rows


def run_cross_condition_transfer(
    source_examples: Sequence[Tuple[str, str, str]],
    source_tensor: np.ndarray,
    source_y: np.ndarray,
    source_groups: np.ndarray,
    target_examples: Sequence[Tuple[str, str, str]],
    target_tensor: np.ndarray,
    target_y: np.ndarray,
    target_groups: np.ndarray,
    layer_count: int,
    *,
    permute_seed: Optional[int] = None,
    restrict_layers: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    source_family_ids = sorted(set(source_groups.tolist()))
    target_family_ids = sorted(set(target_groups.tolist()))
    eval_family_ids = [family_id for family_id in target_family_ids if family_id in set(source_family_ids)]
    rows: List[Dict[str, Any]] = []

    for layer in range(layer_count):
        if restrict_layers is not None and layer not in restrict_layers:
            continue
        source_layer = np.asarray(source_tensor[:, layer, :], dtype=np.float64)
        target_layer = np.asarray(target_tensor[:, layer, :], dtype=np.float64)
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        y_prob_all: List[np.ndarray] = []
        baseline_all: List[np.ndarray] = []
        valid_families: List[str] = []

        for fold_id, family_id in enumerate(eval_family_ids):
            train_mask = source_groups != family_id
            test_mask = target_groups == family_id
            if not train_mask.any() or not test_mask.any():
                continue
            train_x = source_layer[train_mask]
            train_y = source_y[train_mask].copy()
            test_x = target_layer[test_mask]
            test_y = target_y[test_mask]
            # #region debug-point B-C-D:cross-condition-fold
            pre_scale_train_max_abs = float(np.nanmax(np.abs(train_x))) if train_x.size else 0.0
            pre_scale_test_max_abs = float(np.nanmax(np.abs(test_x))) if test_x.size else 0.0
            if (not np.isfinite(train_x).all()) or (not np.isfinite(test_x).all()) or pre_scale_train_max_abs > 1e4 or pre_scale_test_max_abs > 1e4:
                _debug_post(
                    "pre-fix",
                    "D",
                    "probe6.run_cross_condition_transfer",
                    "[DEBUG] Suspicious cross-condition feature magnitude before scaling",
                    {
                        "layer": layer,
                        "fold_id": fold_id,
                        "heldout_family": family_id,
                        "train_max_abs": pre_scale_train_max_abs,
                        "test_max_abs": pre_scale_test_max_abs,
                        "train_finite": bool(np.isfinite(train_x).all()),
                        "test_finite": bool(np.isfinite(test_x).all()),
                    },
                )
            label_values, label_counts = np.unique(train_y, return_counts=True)
            if len(label_values) < 2 or int(np.min(label_counts)) < MIN_TRAIN_CLASS_COUNT:
                _debug_post(
                    "pre-fix",
                    "C",
                    "probe6.run_cross_condition_transfer",
                    "[DEBUG] Degenerate or near-degenerate cross-condition train label split",
                    {
                        "layer": layer,
                        "fold_id": fold_id,
                        "heldout_family": family_id,
                        "labels": label_values.tolist(),
                        "counts": label_counts.tolist(),
                    },
                )
            # #endregion
            if len(label_values) < 2 or int(np.min(label_counts)) < MIN_TRAIN_CLASS_COUNT:
                continue
            if permute_seed is not None:
                rng = np.random.default_rng(permute_seed + layer * 1009 + fold_id)
                train_y = rng.permutation(train_y)
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            train_x = stabilize_scaled_features(train_x)
            test_x = stabilize_scaled_features(test_x)
            # #region debug-point D:cross-condition-post-scale
            if (not np.isfinite(train_x).all()) or (not np.isfinite(test_x).all()) or float(np.nanmax(np.abs(train_x))) > 1e4 or float(np.nanmax(np.abs(test_x))) > 1e4:
                _debug_post(
                    "pre-fix",
                    "D",
                    "probe6.run_cross_condition_transfer",
                    "[DEBUG] Suspicious cross-condition feature magnitude after scaling",
                    {
                        "layer": layer,
                        "fold_id": fold_id,
                        "heldout_family": family_id,
                        "train_max_abs": float(np.nanmax(np.abs(train_x))) if train_x.size else 0.0,
                        "test_max_abs": float(np.nanmax(np.abs(test_x))) if test_x.size else 0.0,
                        "train_finite": bool(np.isfinite(train_x).all()),
                        "test_finite": bool(np.isfinite(test_x).all()),
                        "permute_seed": permute_seed,
                    },
                )
            # #endregion
            if (not np.isfinite(train_x).all()) or (not np.isfinite(test_x).all()):
                continue
            try:
                y_prob, y_pred = centroid_probe_scores(train_x, train_y, test_x)
            except Exception:
                y_prob = np.full(len(test_x), 0.5, dtype=np.float64)
                y_pred = np.zeros(len(test_x), dtype=np.int64)
            y_true_all.append(test_y)
            y_pred_all.append(y_pred)
            y_prob_all.append(y_prob)
            baseline_all.append(majority_baseline_predictions(train_y, len(test_y)))
            valid_families.append(family_id)

        if not y_true_all:
            continue
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        y_prob = np.concatenate(y_prob_all)
        baseline_pred = np.concatenate(baseline_all)
        metrics = evaluate_predictions(y_true, y_pred, y_prob, baseline_pred)
        rows.append(
            {
                "layer": layer,
                "n_examples": int(y_true.shape[0]),
                "n_families": len(valid_families),
                "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
                "baseline_balanced_accuracy": f"{metrics['baseline_balanced_accuracy']:.6f}",
                "auroc": "" if np.isnan(metrics["auroc"]) else f"{metrics['auroc']:.6f}",
                "average_precision": "" if np.isnan(metrics["average_precision"]) else f"{metrics['average_precision']:.6f}",
                "f1": f"{metrics['f1']:.6f}",
                "precision": f"{metrics['precision']:.6f}",
                "recall": f"{metrics['recall']:.6f}",
                "confusion_matrix_counts": metrics["confusion_matrix_counts"],
            }
        )
    return rows


def best_result(layerwise_rows: Sequence[Mapping[str, Any]]) -> Tuple[int, float, float]:
    best_layer = -1
    best_ba = -1.0
    best_baseline = 0.0
    for row in layerwise_rows:
        ba = float(row["balanced_accuracy"])
        if ba > best_ba:
            best_layer = int(row["layer"])
            best_ba = ba
            best_baseline = float(row["baseline_balanced_accuracy"])
    return best_layer, best_ba, best_baseline


def label_count_dict(y: np.ndarray) -> Dict[int, int]:
    values, counts = np.unique(np.asarray(y, dtype=np.int64), return_counts=True)
    return {int(value): int(count) for value, count in zip(values.tolist(), counts.tolist())}


def leave_one_family_out_supported(y: np.ndarray, min_train_class_count: int) -> Tuple[bool, Dict[int, int], str]:
    counts = label_count_dict(y)
    class0 = counts.get(0, 0)
    class1 = counts.get(1, 0)
    needed_total = min_train_class_count + 1
    supported = class0 >= needed_total and class1 >= needed_total
    note = (
        f"class_counts=0:{class0},1:{class1}; need at least {needed_total} examples per class "
        f"for leave-one-family-out training with min_train_class_count={min_train_class_count}"
    )
    return supported, counts, note


def source_condition_supported(y: np.ndarray, min_train_class_count: int) -> Tuple[bool, Dict[int, int], str]:
    counts = label_count_dict(y)
    class0 = counts.get(0, 0)
    class1 = counts.get(1, 0)
    needed_total = min_train_class_count + 1
    supported = class0 >= needed_total and class1 >= needed_total
    note = (
        f"source_class_counts=0:{class0},1:{class1}; need at least {needed_total} source examples per class "
        f"so the held-out-family training split keeps min_train_class_count={min_train_class_count}"
    )
    return supported, counts, note


def build_best_row(
    analysis: str,
    anchor: str,
    pair: str,
    *,
    best_layer: int = -1,
    best_ba: float = -1.0,
    best_baseline: float = 0.0,
    status: str = "ok",
    support_note: str = "",
    label_counts: Optional[Mapping[int, int]] = None,
) -> Dict[str, Any]:
    counts = dict(label_counts or {})
    return {
        "analysis": analysis,
        "anchor": anchor,
        "pair": pair,
        "best_layer": best_layer,
        "best_balanced_accuracy": f"{best_ba:.6f}",
        "best_baseline_balanced_accuracy": f"{best_baseline:.6f}",
        "status": status,
        "support_note": support_note,
        "n_label_0": counts.get(0, 0),
        "n_label_1": counts.get(1, 0),
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def is_detectable(best_ba: float, threshold: float) -> bool:
    return best_ba >= threshold


def summarize_table(
    lines: List[str],
    title: str,
    analysis_name: str,
    best_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines.append(f"== {title} ==")
    subset = [row for row in best_rows if row["analysis"] == analysis_name]
    if not subset:
        lines.append("No results.")
        lines.append("")
        return
    pair_names = sorted({str(row["pair"]) for row in subset})
    header = f"{'pair':<72s}" + "".join(f"{ANCHOR_DISPLAY[anchor][:5]:>11s}" for anchor in ANCHOR_ORDER)
    lines.append(header)
    lookup = {(str(row["pair"]), str(row["anchor"])): row for row in subset}
    for pair_name in pair_names:
        cells: List[str] = []
        for anchor in ANCHOR_ORDER:
            row = lookup.get((pair_name, anchor))
            if row is None:
                cells.append("    -    ")
            elif str(row.get("status", "ok")) != "ok":
                cells.append("  UNSUP  ")
            else:
                cells.append(f"{float(row['best_balanced_accuracy']):.3f}@L{int(row['best_layer']):02d}")
        lines.append(f"{pair_name:<72s}" + "".join(f"{cell:>11s}" for cell in cells))
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / DEFAULT_INPUT))
    parser.add_argument("--family-deltas", default=str(REPO_ROOT / DEFAULT_FAMILY_DELTAS))
    parser.add_argument("--prompt-dataset", default=str(REPO_ROOT / DEFAULT_PROMPT_DATASET))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--activation-output-root", default=str(REPO_ROOT / DEFAULT_ACTIVATION_OUTPUT_ROOT))
    parser.add_argument("--activation-root", default=None, type=str, help="If set, skip extraction and load activations from this directory.")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true", help="Use an existing activation output root directly.")
    parser.add_argument("--output-layerwise", default=str(REPO_ROOT / DEFAULT_LAYERWISE_OUTPUT))
    parser.add_argument("--output-summary", default=str(REPO_ROOT / DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--output-best", default=str(REPO_ROOT / DEFAULT_BEST_LAYERS_OUTPUT))
    parser.add_argument("--permutation-pairs", type=int, default=100)
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--detectable-threshold", type=float, default=DEFAULT_DETECTABLE_THRESHOLD)
    parser.add_argument("--only-extract-and-exit", action="store_true")
    parser.add_argument("--only-family-conditions", default="")
    parser.add_argument("--worker-batch-index", type=int, default=-1)
    parser.add_argument("--worker-total-batches", type=int, default=-1)
    args = parser.parse_args()

    input_path = resolve_required_path(
        args.input,
        "state/logit extraction JSONL",
        "Run src/extraction/extract_multi_family_states_and_logits.py first.",
    )
    family_deltas_path = resolve_required_path(
        args.family_deltas,
        "family delta CSV",
        "Run src/analysis/compute_qwen3_4b_family36_family_margin_deltas.py after the state/logit extraction step.",
    )

    jsonl_rows = read_jsonl(input_path)
    family_deltas = read_family_deltas(family_deltas_path)

    if args.activation_root:
        activation_root = Path(args.activation_root)
        if not activation_root.is_absolute():
            activation_root = (REPO_ROOT / activation_root).resolve()
    elif args.skip_extraction:
        activation_root = Path(args.activation_output_root)
        if not activation_root.is_absolute():
            activation_root = (REPO_ROOT / activation_root).resolve()
    else:
        manifest = extract_early_activations(args)
        print(json.dumps({"status": "extraction_complete", "manifest": str(manifest)}), flush=True)
        activation_root = Path(args.activation_output_root)
        if not activation_root.is_absolute():
            activation_root = (REPO_ROOT / activation_root).resolve()

    if args.only_extract_and_exit:
        print(json.dumps({"status": "worker_mode_exit_after_extraction"}), flush=True)
        return
    if not activation_root.exists():
        raise FileNotFoundError(
            f"Missing early-position activation directory: {activation_root}\n"
            "Run this script without --skip-extraction, or point --activation-root at a completed early-position extraction folder."
        )

    print(json.dumps({"status": "collecting_dataset", "activation_root": str(activation_root)}), flush=True)
    deltas, metadata, layer_count = collect_dataset(jsonl_rows, family_deltas, activation_root)
    print(json.dumps({"status": "dataset_collected", "layer_count": layer_count, "n_examples": len(deltas)}), flush=True)
    if not deltas or layer_count <= 0:
        raise RuntimeError("No early-position delta tensors were collected.")

    overall_layerwise_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for anchor in ANCHOR_ORDER:
        pooled_keys = sorted(
            key
            for key, meta in metadata.items()
            if meta["anchor"] == anchor and meta["condition"] in CONDITIONS and meta["harmful_label_primary"] is not None
        )
        if pooled_keys:
            pooled_tensor = np.stack([deltas[key] for key in pooled_keys], axis=0)
            pooled_y = np.asarray([int(metadata[key]["harmful_label_primary"]) for key in pooled_keys], dtype=np.int64)
            pooled_groups = np.asarray([str(metadata[key]["family_id"]) for key in pooled_keys], dtype=object)
            rows = run_family_heldout_classification(pooled_keys, pooled_tensor, pooled_y, pooled_groups, layer_count)
            best_layer, best_ba, best_baseline = best_result(rows)
            best_rows.append(build_best_row("overall_harmful", anchor, "all_conditions_pooled", best_layer=best_layer, best_ba=best_ba, best_baseline=best_baseline, label_counts=label_count_dict(pooled_y)))
            for row in rows:
                overall_layerwise_rows.append({"analysis": "overall_harmful", "anchor": anchor, "pair": "all_conditions_pooled", **row})

        for condition in CONDITIONS:
            condition_keys = sorted(
                key
                for key, meta in metadata.items()
                if meta["anchor"] == anchor and meta["condition"] == condition and meta["harmful_label_primary"] is not None
            )
            if not condition_keys:
                continue
            tensor = np.stack([deltas[key] for key in condition_keys], axis=0)
            y = np.asarray([int(metadata[key]["harmful_label_primary"]) for key in condition_keys], dtype=np.int64)
            groups = np.asarray([str(metadata[key]["family_id"]) for key in condition_keys], dtype=object)
            supported, counts, note = leave_one_family_out_supported(y, MIN_TRAIN_CLASS_COUNT)
            if not supported:
                best_rows.append(
                    build_best_row(
                        "within_condition",
                        anchor,
                        condition,
                        status="unsupported",
                        support_note=note,
                        label_counts=counts,
                    )
                )
                continue
            rows = run_family_heldout_classification(condition_keys, tensor, y, groups, layer_count)
            best_layer, best_ba, best_baseline = best_result(rows)
            best_rows.append(build_best_row("within_condition", anchor, condition, best_layer=best_layer, best_ba=best_ba, best_baseline=best_baseline, label_counts=counts))
            for row in rows:
                overall_layerwise_rows.append({"analysis": "within_condition", "anchor": anchor, "pair": condition, **row})

        for source_condition in CONDITIONS:
            for target_condition in CONDITIONS:
                if source_condition == target_condition:
                    continue
                source_keys = sorted(
                    key
                    for key, meta in metadata.items()
                    if meta["anchor"] == anchor
                    and meta["condition"] == source_condition
                    and meta["harmful_label_primary"] is not None
                )
                target_keys = sorted(
                    key
                    for key, meta in metadata.items()
                    if meta["anchor"] == anchor
                    and meta["condition"] == target_condition
                    and meta["harmful_label_primary"] is not None
                )
                if not source_keys or not target_keys:
                    continue
                source_tensor = np.stack([deltas[key] for key in source_keys], axis=0)
                source_y = np.asarray([int(metadata[key]["harmful_label_primary"]) for key in source_keys], dtype=np.int64)
                source_groups = np.asarray([str(metadata[key]["family_id"]) for key in source_keys], dtype=object)
                target_tensor = np.stack([deltas[key] for key in target_keys], axis=0)
                target_y = np.asarray([int(metadata[key]["harmful_label_primary"]) for key in target_keys], dtype=np.int64)
                target_groups = np.asarray([str(metadata[key]["family_id"]) for key in target_keys], dtype=object)
                pair_name = f"{source_condition}_to_{target_condition}"
                supported, counts, note = source_condition_supported(source_y, MIN_TRAIN_CLASS_COUNT)
                if not supported:
                    best_rows.append(
                        build_best_row(
                            "cross_condition",
                            anchor,
                            pair_name,
                            status="unsupported",
                            support_note=note,
                            label_counts=counts,
                        )
                    )
                    continue
                rows = run_cross_condition_transfer(
                    source_keys,
                    source_tensor,
                    source_y,
                    source_groups,
                    target_keys,
                    target_tensor,
                    target_y,
                    target_groups,
                    layer_count,
                )
                if not rows:
                    best_rows.append(
                        build_best_row(
                            "cross_condition",
                            anchor,
                            pair_name,
                            status="unsupported",
                            support_note="No valid held-out-family evaluation rows were produced.",
                            label_counts=counts,
                        )
                    )
                    continue
                best_layer, best_ba, best_baseline = best_result(rows)
                best_rows.append(build_best_row("cross_condition", anchor, pair_name, best_layer=best_layer, best_ba=best_ba, best_baseline=best_baseline, label_counts=counts))
                for row in rows:
                    overall_layerwise_rows.append({"analysis": "cross_condition", "anchor": anchor, "pair": pair_name, **row})

    output_layerwise = Path(args.output_layerwise)
    output_summary = Path(args.output_summary)
    output_best = Path(args.output_best)
    for path in (output_layerwise, output_summary, output_best):
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
    output_layerwise = output_layerwise if output_layerwise.is_absolute() else (REPO_ROOT / output_layerwise).resolve()
    output_summary = output_summary if output_summary.is_absolute() else (REPO_ROOT / output_summary).resolve()
    output_best = output_best if output_best.is_absolute() else (REPO_ROOT / output_best).resolve()
    output_layerwise.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_best.parent.mkdir(parents=True, exist_ok=True)

    if overall_layerwise_rows:
        with output_layerwise.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(overall_layerwise_rows[0].keys()))
            writer.writeheader()
            writer.writerows(overall_layerwise_rows)

    with output_best.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(best_rows[0].keys()) if best_rows else ["analysis", "anchor", "pair", "best_layer", "best_balanced_accuracy", "best_baseline_balanced_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    lines: List[str] = []
    lines.append("Probe 6: early-position harmfulness detection (Qwen3-4B-Instruct-2507 36-family)")
    lines.append("")
    lines.append("Features: Δ = h(condition, anchor) - h(evidence_neutral, anchor)")
    lines.append("Classifier: linear centroid probe on z-scored features")
    lines.append("Evaluation:")
    lines.append("  - overall and within-condition: leave-one-family-out")
    lines.append("  - cross-condition: train on source condition from non-held-out families, test on target condition for held-out family")
    lines.append("  - scaler fit only on each training fold")
    lines.append(f"  - analyses marked UNSUP require at least {MIN_TRAIN_CLASS_COUNT + 1} examples in each class for the training side of a held-out-family split")
    lines.append(f"Layer count: {layer_count}")
    lines.append(f"Detectable threshold used in summary questions: balanced accuracy >= {args.detectable_threshold:.2f}")
    lines.append("")
    lines.append("Anchor order:")
    for index, anchor in enumerate(ANCHOR_ORDER):
        lines.append(f"  {index}. {ANCHOR_DISPLAY[anchor]}")
    lines.append("")

    summarize_table(lines, "1. Overall harmful vs nonharmful (3 conditions pooled)", "overall_harmful", best_rows)
    summarize_table(lines, "2. Within-condition harmfulness", "within_condition", best_rows)
    summarize_table(lines, "3. Cross-condition harmfulness transfer", "cross_condition", best_rows)

    best_lookup = {(str(row["analysis"]), str(row["pair"]), str(row["anchor"])): row for row in best_rows}

    def best_for(analysis: str, pair: str, anchor: str) -> Optional[Tuple[int, float]]:
        row = best_lookup.get((analysis, pair, anchor))
        if row is None:
            return None
        if str(row.get("status", "ok")) != "ok":
            return None
        return int(row["best_layer"]), float(row["best_balanced_accuracy"])

    def support_row(analysis: str, pair: str, anchor: str) -> Optional[Mapping[str, Any]]:
        return best_lookup.get((analysis, pair, anchor))

    lines.append("== Summary answers ==")

    lines.append("Q1. What is the earliest anchor where harmfulness is detectable?")
    overall_detectable = [
        anchor
        for anchor in ANCHOR_ORDER
        if best_for("overall_harmful", "all_conditions_pooled", anchor)
        and is_detectable(best_for("overall_harmful", "all_conditions_pooled", anchor)[1], args.detectable_threshold)
    ]
    if overall_detectable:
        anchor = overall_detectable[0]
        layer, ba = best_for("overall_harmful", "all_conditions_pooled", anchor)
        lines.append(f"  Earliest detectable anchor: {ANCHOR_DISPLAY[anchor]} (BA {ba:.3f} @ layer {layer})")
    else:
        lines.append("  No anchor met the detectable threshold.")
    for anchor in ANCHOR_ORDER:
        value = best_for("overall_harmful", "all_conditions_pooled", anchor)
        if value is not None:
            layer, ba = value
            lines.append(f"    {ANCHOR_DISPLAY[anchor]}: BA {ba:.3f} @ layer {layer}")
    lines.append("")

    lines.append("Q2. Is false_pressure detectable before ANSWER, or only at ANSWER?")
    false_pre_final = [
        anchor
        for anchor in ANCHOR_ORDER[:-1]
        if best_for("within_condition", "evidence_false_belief_pressure", anchor)
        and is_detectable(best_for("within_condition", "evidence_false_belief_pressure", anchor)[1], args.detectable_threshold)
    ]
    false_final = best_for("within_condition", "evidence_false_belief_pressure", "final_answer_position")
    if false_pre_final:
        anchor = false_pre_final[0]
        layer, ba = best_for("within_condition", "evidence_false_belief_pressure", anchor)
        lines.append(f"  Detectable before ANSWER: {ANCHOR_DISPLAY[anchor]} (BA {ba:.3f} @ layer {layer})")
    elif false_final is not None:
        lines.append(f"  Only at ANSWER by this threshold: BA {false_final[1]:.3f} @ layer {false_final[0]}")
    else:
        support = support_row("within_condition", "evidence_false_belief_pressure", "final_answer_position")
        if support is not None and str(support.get("status", "ok")) != "ok":
            lines.append(f"  Unsupported for this dataset: {support.get('support_note', '').strip()}")
        else:
            lines.append("  No false-pressure result was available.")
    lines.append("")

    lines.append("Q3. Is emotional pressure detectable earlier than false_pressure?")
    def earliest_detectable_within(condition: str) -> Optional[str]:
        for anchor in ANCHOR_ORDER:
            value = best_for("within_condition", condition, anchor)
            if value is not None and is_detectable(value[1], args.detectable_threshold):
                return anchor
        return None

    false_anchor = earliest_detectable_within("evidence_false_belief_pressure")
    emotional_anchor = earliest_detectable_within("evidence_emotional_pressure")
    false_support = support_row("within_condition", "evidence_false_belief_pressure", "final_answer_position")
    emotional_support = support_row("within_condition", "evidence_emotional_pressure", "final_answer_position")
    lines.append(f"  False-pressure earliest detectable anchor: {ANCHOR_DISPLAY[false_anchor] if false_anchor else 'NONE'}")
    lines.append(f"  Emotional-pressure earliest detectable anchor: {ANCHOR_DISPLAY[emotional_anchor] if emotional_anchor else 'NONE'}")
    if false_support is not None and str(false_support.get("status", "ok")) != "ok":
        lines.append(f"  False-pressure within-condition analysis is unsupported: {false_support.get('support_note', '').strip()}")
    if emotional_support is not None and str(emotional_support.get("status", "ok")) != "ok":
        lines.append(f"  Emotional-pressure within-condition analysis is unsupported: {emotional_support.get('support_note', '').strip()}")
    if emotional_anchor and false_anchor:
        if ANCHOR_ORDER.index(emotional_anchor) < ANCHOR_ORDER.index(false_anchor):
            lines.append("  Yes: emotional pressure appears earlier than false pressure.")
        elif ANCHOR_ORDER.index(emotional_anchor) == ANCHOR_ORDER.index(false_anchor):
            lines.append("  They first appear at the same anchor.")
        else:
            lines.append("  No: emotional pressure does not appear earlier than false pressure.")
    elif emotional_anchor and not false_anchor:
        if false_support is not None and str(false_support.get("status", "ok")) != "ok":
            lines.append("  Emotional pressure is detectable earlier, but false pressure is unsupported rather than a clean negative.")
        else:
            lines.append("  Emotional pressure is detectable; false pressure did not reach the threshold.")
    elif false_anchor and not emotional_anchor:
        lines.append("  False pressure is detectable; emotional pressure did not reach the threshold.")
    else:
        lines.append("  Neither condition reached the threshold.")
    lines.append("")

    lines.append("Q4. Does false ⇄ emotional transfer survive before ANSWER?")
    transfer_pairs = [
        "evidence_false_belief_pressure_to_evidence_emotional_pressure",
        "evidence_emotional_pressure_to_evidence_false_belief_pressure",
    ]
    successful_transfers = []
    for pair in transfer_pairs:
        earliest_pair_anchor = None
        for anchor in ANCHOR_ORDER[:-1]:
            value = best_for("cross_condition", pair, anchor)
            if value is not None and is_detectable(value[1], args.detectable_threshold):
                earliest_pair_anchor = anchor
                successful_transfers.append(pair)
                break
        support = support_row("cross_condition", pair, "final_answer_position")
        if support is not None and str(support.get("status", "ok")) != "ok":
            lines.append(f"  {pair}: unsupported ({support.get('support_note', '').strip()})")
        else:
            lines.append(f"  {pair}: {ANCHOR_DISPLAY[earliest_pair_anchor] if earliest_pair_anchor else 'no pre-ANSWER detection'}")
    if len(successful_transfers) == 2:
        lines.append("  Yes: both transfer directions survive before ANSWER.")
    elif len(successful_transfers) == 1:
        lines.append("  Partial: one direction survives before ANSWER.")
    else:
        lines.append("  No: neither direction survives before ANSWER.")
    lines.append("")

    lines.append("Q5. Does early-position performance survive a permutation control for the strongest result?")
    best_pre_final_row: Optional[Mapping[str, Any]] = None
    best_pre_final_value = -1.0
    for row in best_rows:
        if row["anchor"] == "final_answer_position":
            continue
        if str(row.get("status", "ok")) != "ok":
            continue
        value = float(row["best_balanced_accuracy"])
        if value > best_pre_final_value:
            best_pre_final_value = value
            best_pre_final_row = row

    if best_pre_final_row is None:
        lines.append("  No pre-ANSWER result was available for a permutation control.")
    else:
        analysis = str(best_pre_final_row["analysis"])
        pair = str(best_pre_final_row["pair"])
        anchor = str(best_pre_final_row["anchor"])
        best_layer = int(best_pre_final_row["best_layer"])
        real_ba = float(best_pre_final_row["best_balanced_accuracy"])
        lines.append(
            f"  Strongest pre-ANSWER result: analysis={analysis}, pair={pair}, anchor={ANCHOR_DISPLAY[anchor]}, BA {real_ba:.3f} @ layer {best_layer}"
        )
        if not args.skip_permutation:
            permutation_values: List[float] = []
            # #region debug-point E:permutation-entry
            _debug_post(
                "pre-fix",
                "E",
                "probe6.main",
                "[DEBUG] Starting permutation control for strongest pre-answer result",
                {
                    "analysis": analysis,
                    "pair": pair,
                    "anchor": anchor,
                    "best_layer": best_layer,
                    "real_ba": real_ba,
                    "permutation_pairs": args.permutation_pairs,
                },
            )
            # #endregion
            for rep in range(args.permutation_pairs):
                permute_seed = 137_000 + rep
                if analysis in {"overall_harmful", "within_condition"}:
                    if analysis == "overall_harmful":
                        keys = sorted(
                            key
                            for key, meta in metadata.items()
                            if meta["anchor"] == anchor and meta["condition"] in CONDITIONS and meta["harmful_label_primary"] is not None
                        )
                    else:
                        keys = sorted(
                            key
                            for key, meta in metadata.items()
                            if meta["anchor"] == anchor and meta["condition"] == pair and meta["harmful_label_primary"] is not None
                        )
                    tensor = np.stack([deltas[key] for key in keys], axis=0)
                    y = np.asarray([int(metadata[key]["harmful_label_primary"]) for key in keys], dtype=np.int64)
                    groups = np.asarray([str(metadata[key]["family_id"]) for key in keys], dtype=object)
                    perm_rows = run_family_heldout_classification(
                        keys,
                        tensor,
                        y,
                        groups,
                        layer_count,
                        permute_seed=permute_seed,
                        restrict_layers={best_layer},
                    )
                else:
                    source_condition, target_condition = pair.split("_to_", 1)
                    source_keys = sorted(
                        key
                        for key, meta in metadata.items()
                        if meta["anchor"] == anchor
                        and meta["condition"] == source_condition
                        and meta["harmful_label_primary"] is not None
                    )
                    target_keys = sorted(
                        key
                        for key, meta in metadata.items()
                        if meta["anchor"] == anchor
                        and meta["condition"] == target_condition
                        and meta["harmful_label_primary"] is not None
                    )
                    source_tensor = np.stack([deltas[key] for key in source_keys], axis=0)
                    source_y = np.asarray([int(metadata[key]["harmful_label_primary"]) for key in source_keys], dtype=np.int64)
                    source_groups = np.asarray([str(metadata[key]["family_id"]) for key in source_keys], dtype=object)
                    target_tensor = np.stack([deltas[key] for key in target_keys], axis=0)
                    target_y = np.asarray([int(metadata[key]["harmful_label_primary"]) for key in target_keys], dtype=np.int64)
                    target_groups = np.asarray([str(metadata[key]["family_id"]) for key in target_keys], dtype=object)
                    perm_rows = run_cross_condition_transfer(
                        source_keys,
                        source_tensor,
                        source_y,
                        source_groups,
                        target_keys,
                        target_tensor,
                        target_y,
                        target_groups,
                        layer_count,
                        permute_seed=permute_seed,
                        restrict_layers={best_layer},
                    )
                if perm_rows:
                    permutation_values.append(float(perm_rows[0]["balanced_accuracy"]))
            if permutation_values:
                perm_mean = float(np.mean(permutation_values))
                perm_p95 = percentile(permutation_values, 95.0)
                empirical_p = (sum(1 for value in permutation_values if value >= real_ba) + 1) / (len(permutation_values) + 1)
                survives = real_ba > perm_p95
                lines.append(f"  Permutation mean BA: {perm_mean:.3f}")
                lines.append(f"  Permutation 95th percentile BA: {perm_p95:.3f}")
                lines.append(f"  Empirical p-value: {empirical_p:.4f}")
                lines.append(f"  Survives permutation control: {'YES' if survives else 'NO'}")
            else:
                lines.append("  Permutation control could not be evaluated for this result.")

    output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "done",
                "output_layerwise": str(output_layerwise),
                "output_summary": str(output_summary),
                "output_best": str(output_best),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
