import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
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
from sklearn.utils._testing import ignore_warnings  # noqa
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_FAMILY_DELTAS = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_PROMPT_DATASET = "data/generated_prompts_v1.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ACTIVATION_OUTPUT_ROOT = "activations/qwen3_4b_instruct_2507_early_positions"
DEFAULT_LAYERWISE_OUTPUT = "results/probe6_early_position_layerwise.csv"
DEFAULT_SUMMARY_OUTPUT = "results/probe6_early_position_summary.txt"
DEFAULT_BEST_LAYERS_OUTPUT = "results/probe6_early_position_best.csv"

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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


CONDITION_DELTA_COLUMN_MAP: Dict[str, str] = {
    "evidence_false_belief_pressure": "delta_false_pressure",
    "evidence_emotional_pressure": "delta_emotional_pressure",
    "closed_context_false_belief_pressure": "delta_closed_context",
}


def read_family_deltas(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return out
    first = rows[0]
    has_condition_col = "condition" in first and "delta_margin" in first
    if has_condition_col:
        for row in rows:
            fam = str(row.get("family_id", "")).strip()
            cond = str(row.get("condition", "")).strip()
            if not fam or not cond:
                continue
            dm = row.get("delta_margin")
            try:
                dm_val = float(dm) if dm is not None and dm != "" else None
            except (TypeError, ValueError):
                dm_val = None
            out[(fam, cond)] = {
                "delta_margin": dm_val,
            }
        return out
    for row in rows:
        fam = str(row.get("family_id", "")).strip()
        if not fam:
            continue
        for cond, col_name in CONDITION_DELTA_COLUMN_MAP.items():
            dm = row.get(col_name)
            try:
                dm_val = float(dm) if dm is not None and dm != "" else None
            except (TypeError, ValueError):
                dm_val = None
            out[(fam, cond)] = {
                "delta_margin": dm_val,
            }
    return out


def label_primary(delta_margin: Optional[float]) -> Optional[int]:
    if delta_margin is None:
        return None
    return 1 if delta_margin < 0 else 0


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    cache_dir = os.environ.get("HF_HOME", str(REPO_ROOT / "model_cache"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    return tokenizer


def find_anchor_positions(
    prompt_text: str,
    answer_logit_prompt_suffix: str,
    tokenizer: Any,
) -> Dict[str, int]:
    """
    Given the raw prompt and its tokenized length, return byte-level string
    positions (last-char index) and we'll map to token positions via tokenize
    with return_offsets_mapping=True.
    """
    suffix_start = prompt_text.rfind(answer_logit_prompt_suffix)
    base_text = prompt_text if suffix_start < 0 else prompt_text[:suffix_start]
    base_text = base_text.rstrip()

    has_user_pressure = False
    marker_after_pressure_list = [
        "\n\nRetrieved document:",
        "\n\nContext:",
        "\nQuestion:",
    ]
    end_pressure_pos = -1
    for marker in marker_after_pressure_list:
        idx = base_text.find(marker)
        if idx >= 0:
            end_pressure_pos = max(end_pressure_pos, idx - 1)
            has_user_pressure = True

    evidence_start_patterns = [
        ("Retrieved document:", "\nQuestion:"),
        ("Context:", "\nQuestion:"),
    ]
    end_evidence_pos = -1
    for ev_start_marker, after_ev_marker in evidence_start_patterns:
        ev_idx = base_text.find(ev_start_marker)
        if ev_idx < 0:
            continue
        after_idx = base_text.find(after_ev_marker, ev_idx + len(ev_start_marker))
        if after_idx < 0:
            continue
        end_evidence_pos = max(end_evidence_pos, after_idx - 1)

    q_marker = "\nQuestion:"
    q_idx = base_text.rfind(q_marker)
    end_question_pos = -1
    if q_idx >= 0:
        after_q = base_text.find("\nChoices:", q_idx + len(q_marker))
        if after_q < 0:
            after_q = base_text.find("\n\n", q_idx + len(q_marker))
        if after_q < 0:
            after_q = len(base_text)
        end_question_pos = after_q - 1

    choices_marker = "\nChoices:"
    end_choices_pos = -1
    c_idx = base_text.rfind(choices_marker)
    if c_idx >= 0:
        format_markers = [
            "\n\nAnswer with exactly this format:",
            "\n\nAnswer with only A or B.",
        ]
        fmt_idx = -1
        for fm in format_markers:
            i = base_text.find(fm, c_idx + len(choices_marker))
            if i >= 0 and (fmt_idx < 0 or i < fmt_idx):
                fmt_idx = i
        if fmt_idx < 0:
            fmt_idx = len(base_text)
        end_choices_pos = fmt_idx - 1

    end_prompt_pos = len(base_text) - 1

    full_text = base_text + "\n\nAnswer with only A or B.\n\nANSWER:"
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="np",
    )
    offsets = np.array(encoded["offset_mapping"][0])
    token_seq_len = offsets.shape[0]

    def char_to_token(char_pos: int) -> int:
        if char_pos < 0 or token_seq_len <= 0:
            return 0
        pos = char_pos + 1
        mask_start = offsets[:, 0] <= pos
        mask_end = offsets[:, 1] >= pos
        hits = np.where(mask_start & mask_end)[0]
        if len(hits) > 0:
            return int(hits[-1])
        fallback = np.where(offsets[:, 0] < pos)[0]
        if len(fallback) > 0:
            return int(fallback[-1])
        return 0

    out: Dict[str, int] = {
        "end_of_evidence_block": char_to_token(end_evidence_pos) if end_evidence_pos >= 0 else 0,
        "end_of_user_pressure_sentence": char_to_token(end_pressure_pos) if has_user_pressure and end_pressure_pos >= 0 else 0,
        "end_of_question": char_to_token(end_question_pos) if end_question_pos >= 0 else 0,
        "end_of_answer_choices": char_to_token(end_choices_pos) if end_choices_pos >= 0 else 0,
        "final_answer_position": token_seq_len - 1,
        "_token_seq_len": token_seq_len,
    }
    return out


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
    inputs = {k: v.to(device) for k, v in inputs.items()}
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
    last_hidden_dim = hidden_states[1].shape[-1]
    layer_vectors = {anchor: np.zeros((n_layers, last_hidden_dim), dtype=np.float16) for anchor in ANCHOR_ORDER}
    seq_len = int(inputs["input_ids"].shape[1])
    del inputs
    for anchor in ANCHOR_ORDER:
        pos = int(token_positions.get(anchor, 0))
        pos = min(pos, seq_len - 1)
        for layer_offset in range(n_layers):
            raw_state = hidden_states[layer_offset + 1]
            vec = raw_state[0, pos, :].detach().to("cpu", dtype=torch.float16).numpy()
            layer_vectors[anchor][layer_offset] = vec
            del raw_state
    del hidden_states
    if hasattr(torch.cuda, "empty_cache"):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()
    return {
        "logits_last_token": logits,
        "hidden_states_by_anchor": layer_vectors,
        "token_seq_len": int(seq_len),
    }


def build_answer_logit_suffix() -> str:
    return "\n\nAnswer with exactly this format:"


def extract_early_activations(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from load_model import load_local_model, pick_device, pick_dtype  # noqa

    worker_bidx = int(getattr(args, "worker_batch_index", -1))
    worker_total = int(getattr(args, "worker_total_batches", -1))

    prompts_all = read_jsonl(Path(args.prompt_dataset))
    prompts_by_pid: Dict[str, Dict[str, Any]] = {str(r["prompt_id"]): r for r in prompts_all if "prompt_id" in r}

    existing_rows = read_jsonl(Path(args.input))
    CONDITIONS_EXTRACT = [
        NEUTRAL_CONDITION,
        "evidence_false_belief_pressure",
        "evidence_emotional_pressure",
        "closed_context_false_belief_pressure",
        "evidence_true_belief_pressure",
    ]
    by_fam_cond: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in existing_rows:
        fam = str(r.get("family_id", "")).strip()
        cond = str(r.get("prompt_type", "")).strip()
        if not fam or not cond:
            continue
        if cond not in CONDITIONS_EXTRACT:
            continue
        by_fam_cond[(fam, cond)] = r

    only_pairs: set[Tuple[str, str]] | None = None
    only_arg = getattr(args, "only_family_conditions", "")
    if isinstance(only_arg, str) and only_arg:
        only_pairs = set()
        for piece in only_arg.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "=" in piece:
                a, b = piece.split("=", 1)
                only_pairs.add((a.strip(), b.strip()))

    families = sorted({fam for fam, _ in by_fam_cond.keys()})

    import gc
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    if os.environ.get("OVERRIDE_DEVICE"):
        device = os.environ["OVERRIDE_DEVICE"]
    else:
        device = pick_device("")
    if os.environ.get("OVERRIDE_DTYPE"):
        explicit_dt = os.environ["OVERRIDE_DTYPE"]
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(explicit_dt, pick_dtype(device, ""))
    else:
        if device == "cpu":
            dtype = torch.float32
        else:
            dtype = pick_dtype(device, "")
    cache_dir = os.environ.get("HF_HOME", str(REPO_ROOT / "model_cache"))
    mem_gib_env = os.environ.get("CPU_MAX_MEMORY_GIB", "")
    cpu_max_memory_gib = int(mem_gib_env) if mem_gib_env.isdigit() else 0
    offload_folder_env = os.environ.get("CPU_OFFLOAD_FOLDER", "")
    offload_folder = offload_folder_env or (str(REPO_ROOT / "temp_offload_folder") if cpu_max_memory_gib > 0 else "")
    if cpu_max_memory_gib > 0 and offload_folder:
        os.makedirs(offload_folder, exist_ok=True)
    print(json.dumps({"status": "loading_model", "model": args.model, "device": device, "dtype": str(dtype), "cpu_max_memory_gib": cpu_max_memory_gib, "offload_folder": offload_folder or ""}), flush=True)
    gc.collect()
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
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    n_done = 0
    n_total = 0
    needed_pairs = []
    for fam in families:
        for cond in CONDITIONS_EXTRACT:
            key = (fam, cond)
            if key not in by_fam_cond:
                continue
            if only_pairs is not None and key not in only_pairs:
                continue
            needed_pairs.append(key)
    n_total = len(needed_pairs)
    print(json.dumps({"status": "worker_batch_start", "worker_batch_index": worker_bidx, "worker_total_batches": worker_total, "n_pairs_in_batch": n_total, "extraction_root": str(output_root)}), flush=True)
    try:
        for fam, cond in needed_pairs:
            ref_row = by_fam_cond[(fam, cond)]
            pid = str(ref_row["prompt_id"])
            prompt_row = prompts_by_pid.get(pid)
            if prompt_row is None:
                continue
            prompt_text = str(prompt_row.get("prompt", ""))
            family_dir = output_root / fam
            family_dir.mkdir(parents=True, exist_ok=True)
            out_path = family_dir / f"{fam}_{cond}.pt"
            if out_path.exists() and not args.force_reextract:
                try:
                    loaded = torch.load(out_path, map_location="cpu")
                    hs_anchor = loaded.get("hidden_states_by_anchor", {})
                    has_all_keys = all(anchor in hs_anchor for anchor in ANCHOR_ORDER)
                    ap = loaded.get("anchor_positions", {}) or {}
                    try:
                        tsl = int(ap.get("_token_seq_len", 0))
                    except Exception:
                        tsl = 0
                    seq_ok = tsl >= 50
                    norm_ok = False
                    if has_all_keys:
                        try:
                            v = hs_anchor[ANCHOR_ORDER[0]]
                            if isinstance(v, torch.Tensor):
                                arr = v.to(dtype=torch.float32).numpy()
                            else:
                                import numpy as _np
                                arr = _np.asarray(v, dtype=_np.float32)
                            lyr0 = arr[0] if arr.ndim >= 2 else arr
                            norm_ok = float(np.linalg.norm(lyr0)) > 1e-6
                        except Exception:
                            norm_ok = False
                    if has_all_keys and seq_ok and norm_ok:
                        del loaded, hs_anchor
                        gc.collect()
                        n_done += 1
                        manifest_rows.append({
                            "family_id": fam,
                            "condition": cond,
                            "prompt_id": pid,
                            "activation_path": str(out_path.resolve().relative_to(REPO_ROOT.resolve())),
                            "skipped_existing": 1,
                        })
                        continue
                except Exception:
                    pass
            anchors = find_anchor_positions(prompt_text, suffix, tokenizer)
            result = run_forward_multi_position(model, tokenizer, prompt_text, suffix, device, anchors)
            hs_anchor = result["hidden_states_by_anchor"]
            record = {
                "family_id": fam,
                "condition": cond,
                "prompt_id": pid,
                "prompt_type": cond,
                "answer_logit_prompt": str(ref_row.get("answer_logit_prompt", "")),
                "model_name": args.model,
                "anchor_positions": dict(anchors),
                "hidden_states_by_anchor": {
                    anchor: torch.from_numpy(hs_anchor[anchor])
                    for anchor in ANCHOR_ORDER
                },
                "logits_last_token": torch.from_numpy(result["logits_last_token"]),
                "token_seq_len": int(result["token_seq_len"]),
            }
            torch.save(record, out_path)
            del record, hs_anchor, result, anchors, prompt_text, ref_row
            gc.collect()
            n_done += 1
            manifest_rows.append({
                "family_id": fam,
                "condition": cond,
                "prompt_id": pid,
                "activation_path": str(out_path.resolve().relative_to(REPO_ROOT.resolve())),
                "skipped_existing": 0,
            })
            if n_done % 6 == 0 or n_done == max(1, n_total):
                if hasattr(torch.cuda, "empty_cache"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                print(json.dumps({
                    "status": "extracted_batch",
                    "n_done": n_done,
                    "n_total": n_total,
                    "elapsed_sec": int(elapsed),
                    "examples_per_sec": round(rate, 4),
                }), flush=True)
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
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def load_activation_by_anchor(path: Path) -> Dict[str, np.ndarray]:
    record = torch.load(path, map_location="cpu")
    raw = record["hidden_states_by_anchor"]
    out: Dict[str, np.ndarray] = {}
    for anchor in ANCHOR_ORDER:
        t = raw.get(anchor)
        if t is None:
            raise RuntimeError(f"Missing anchor {anchor} in {path}")
        if isinstance(t, torch.Tensor):
            out[anchor] = t.to(dtype=torch.float32).numpy()
        else:
            out[anchor] = np.asarray(t, dtype=np.float32)
    return out


def collect_dataset(
    jsonl_rows: Sequence[Mapping[str, Any]],
    family_deltas: Mapping[Tuple[str, str], Mapping[str, Any]],
    activation_root: Path,
) -> Tuple[
    Dict[Tuple[str, str, str], np.ndarray],
    Dict[Tuple[str, str, str], Dict[str, Any]],
    int,
]:
    by_fam_cond: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in jsonl_rows:
        fam = str(r.get("family_id", "")).strip()
        cond = str(r.get("prompt_type", "")).strip()
        if fam and cond:
            by_fam_cond[(fam, cond)] = r
    deltas: Dict[Tuple[str, str, str], np.ndarray] = {}
    metadata: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    layer_count = 0
    families = sorted({fam for fam, _ in by_fam_cond.keys()})
    for fam in families:
        neutral_row = by_fam_cond.get((fam, NEUTRAL_CONDITION))
        if neutral_row is None:
            continue
        neutral_path = activation_root / fam / f"{fam}_{NEUTRAL_CONDITION}.pt"
        if not neutral_path.exists():
            continue
        neutral_acts = load_activation_by_anchor(neutral_path)
        layer_count = neutral_acts[ANCHOR_ORDER[0]].shape[0]
        for cond in CONDITIONS:
            row = by_fam_cond.get((fam, cond))
            if row is None:
                continue
            comp_path = activation_root / fam / f"{fam}_{cond}.pt"
            if not comp_path.exists():
                continue
            comp_acts = load_activation_by_anchor(comp_path)
            dm_info = family_deltas.get((fam, cond), {})
            delta_margin = dm_info.get("delta_margin")
            label = label_primary(delta_margin)
            for anchor in ANCHOR_ORDER:
                key = (fam, cond, anchor)
                delta = comp_acts[anchor] - neutral_acts[anchor]
                deltas[key] = delta.astype(np.float32)
                metadata[key] = {
                    "family_id": fam,
                    "condition": cond,
                    "anchor": anchor,
                    "delta_margin": delta_margin,
                    "harmful_label_primary": label,
                }
    return deltas, metadata, layer_count


def run_family_heldout_classification(
    examples: Sequence[Tuple[str, str, str]],
    X_full: Sequence[np.ndarray],
    y_full: Sequence[int],
    groups_full: Sequence[str],
    layer_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n_examples = len(examples)
    if n_examples == 0:
        return [], []
    tensor = np.stack(X_full, axis=0)
    y = np.asarray(y_full, dtype=np.int64)
    groups = np.asarray(groups_full, dtype=object)
    logo = LeaveOneGroupOut()
    fold_list = list(logo.split(np.arange(n_examples), y, groups=groups))
    layerwise_rows: List[Dict[str, Any]] = []
    per_example_rows: List[Dict[str, Any]] = []
    family_ids_unique = sorted(set(groups_full))
    n_families = len(family_ids_unique)
    example_index_to_row_id = {idx: examples[idx] for idx in range(n_examples)}
    n_folds = len(fold_list)
    n_test_per_fold = [len(test_idx) for _, test_idx in fold_list]
    max_test = max(n_test_per_fold) if n_test_per_fold else 0
    y_true_per = np.zeros((n_folds, max_test), dtype=np.int64)
    y_pred_per = np.zeros((n_folds, max_test), dtype=np.int64)
    y_prob_per = np.zeros((n_folds, max_test), dtype=np.float64)
    test_idx_per = np.zeros((n_folds, max_test), dtype=np.int64)
    fold_ids_per = np.zeros((n_folds, max_test), dtype=np.int64)
    valid_per_fold = np.asarray(n_test_per_fold, dtype=np.int64)
    for layer in range(layer_count):
        X_layer = tensor[:, layer, :]
        for fold_id, (train_idx, test_idx) in enumerate(fold_list):
            n_test = len(test_idx)
            if n_test == 0:
                continue
            train_x = X_layer[train_idx]
            test_x = X_layer[test_idx]
            train_y = y[train_idx]
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            model = LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=10000,
                C=1.0,
            )
            try:
                model.fit(train_x, train_y)
                prob = model.predict_proba(test_x)[:, 1]
                pred = (prob >= 0.5).astype(np.int64)
            except Exception:
                prob = np.full(len(test_x), 0.5, dtype=np.float64)
                pred = np.zeros(len(test_x), dtype=np.int64)
            y_true_per[fold_id, :n_test] = y[test_idx]
            y_pred_per[fold_id, :n_test] = pred
            y_prob_per[fold_id, :n_test] = prob
            test_idx_per[fold_id, :n_test] = np.asarray(test_idx, dtype=np.int64)
            fold_ids_per[fold_id, :n_test] = np.asarray([fold_id] * n_test, dtype=np.int64)
        masks_by_fold = [np.arange(max_test) < nv for nv in valid_per_fold]
        flat_mask = np.concatenate(masks_by_fold)
        y_true = y_true_per.reshape(-1)[flat_mask]
        y_pred = y_pred_per.reshape(-1)[flat_mask]
        y_prob = y_prob_per.reshape(-1)[flat_mask]
        test_idx = test_idx_per.reshape(-1)[flat_mask]
        fold_ids = fold_ids_per.reshape(-1)[flat_mask]
        ba = balanced_accuracy_score(y_true, y_pred)
        try:
            auroc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) >= 2 else float("nan")
        except Exception:
            auroc = float("nan")
        try:
            ap = average_precision_score(y_true, y_prob) if len(set(y_true.tolist())) >= 2 else float("nan")
        except Exception:
            ap = float("nan")
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn = int(cm[0, 0]) if cm.size >= 4 else 0
        fp = int(cm[0, 1]) if cm.size >= 4 else 0
        fn = int(cm[1, 0]) if cm.size >= 4 else 0
        tp = int(cm[1, 1]) if cm.size >= 4 else 0
        majority = np.bincount(y_true).argmax()
        baseline_pred = np.full_like(y_true, majority)
        baseline_ba = balanced_accuracy_score(y_true, baseline_pred)
        layerwise_rows.append({
            "layer": layer,
            "n_examples": n_examples,
            "n_families": n_families,
            "balanced_accuracy": f"{ba:.6f}",
            "auroc": "" if np.isnan(auroc) else f"{auroc:.6f}",
            "average_precision": "" if np.isnan(ap) else f"{ap:.6f}",
            "f1": f"{f1:.6f}",
            "precision": f"{prec:.6f}",
            "recall": f"{rec:.6f}",
            "confusion_matrix": f"tn={tn} fp={fp} fn={fn} tp={tp}",
            "baseline_balanced_accuracy": f"{baseline_ba:.6f}",
        })
        for i in range(len(y_true)):
            row_ex_id = example_index_to_row_id[int(test_idx[i])]
            fam, cond, anchor = row_ex_id
            per_example_rows.append({
                "layer": layer,
                "family_id": fam,
                "condition": cond,
                "anchor": anchor,
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "y_prob_harmful": float(y_prob[i]),
                "fold_id": int(fold_ids[i]),
            })
    return layerwise_rows, per_example_rows


def best_balanced_accuracy(layerwise_rows: Sequence[Mapping[str, Any]]) -> Tuple[int, float]:
    best_layer = -1
    best_val = -1.0
    for r in layerwise_rows:
        v = float(r["balanced_accuracy"])
        layer = int(r["layer"])
        if v > best_val:
            best_val = v
            best_layer = layer
    return best_layer, best_val


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / DEFAULT_INPUT))
    parser.add_argument("--family-deltas", default=str(REPO_ROOT / DEFAULT_FAMILY_DELTAS))
    parser.add_argument("--prompt-dataset", default=str(REPO_ROOT / DEFAULT_PROMPT_DATASET))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--activation-output-root", default=str(REPO_ROOT / DEFAULT_ACTIVATION_OUTPUT_ROOT))
    parser.add_argument("--activation-root", default=None, type=str, help="If set, skip extraction and load activations from this directory.")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true", help="Use existing activation output root directly.")
    parser.add_argument("--output-layerwise", default=str(REPO_ROOT / DEFAULT_LAYERWISE_OUTPUT))
    parser.add_argument("--output-summary", default=str(REPO_ROOT / DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--output-best", default=str(REPO_ROOT / DEFAULT_BEST_LAYERS_OUTPUT))
    parser.add_argument(
        "--permutation-pairs",
        type=int,
        default=100,
        help="Number of permutations for the strongest-early-result permutation control.",
    )
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--only-extract-and-exit", action="store_true", help="Worker mode: only run extraction then exit.")
    parser.add_argument("--only-family-conditions", default="", help="Comma separated list of family_id=condition pairs. Worker mode will extract ONLY these pairs, then exit.")
    parser.add_argument("--worker-batch-index", type=int, default=-1, help="0-indexed batch id (driver helper).")
    parser.add_argument("--worker-total-batches", type=int, default=-1, help="Total worker batches (driver helper).")
    args = parser.parse_args()

    jsonl_rows = read_jsonl(Path(args.input))
    family_deltas = read_family_deltas(Path(args.family_deltas))

    activation_root: Path
    if args.activation_root:
        activation_root = Path(args.activation_root)
    elif args.skip_extraction:
        activation_root = Path(args.activation_output_root)
    else:
        manifest = extract_early_activations(args)
        print(json.dumps({"status": "extraction_complete", "manifest": str(manifest)}), flush=True)
        activation_root = Path(args.activation_output_root)
    if getattr(args, "only_extract_and_exit", False):
        print(json.dumps({"status": "worker_mode_exit_after_extraction"}), flush=True)
        return

    print(json.dumps({"status": "collecting_dataset", "activation_root": str(activation_root)}), flush=True)
    deltas, metadata, layer_count = collect_dataset(jsonl_rows, family_deltas, activation_root)
    print(json.dumps({"status": "dataset_collected", "layer_count": layer_count, "n_examples": len(deltas)}), flush=True)

    overall_layerwise_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    # 1. Overall harmful vs nonharmful (all 3 conditions pooled) -- but we must do anchor separately for family structure to be preserved:
    # Actually run separate probes per (analysis_type, anchor, source, target) but structured so overall harmful pools all 3 conds as
    # rows within the same fold. For cross-condition transfer, we use only rows from source/target condition at the anchor.

    def build_rows_for_filter(filter_fn) -> List[Tuple[str, str, str]]:
        rows = [k for k in deltas.keys() if filter_fn(k, metadata[k])]
        rows.sort()
        return rows

    anchor_probe_specs: List[Dict[str, Any]] = []
    for anchor in ANCHOR_ORDER:
        anchor_probe_specs.append({
            "analysis": "overall_harmful",
            "anchor": anchor,
            "pair_name": "all_conditions_pooled",
            "source_condition": None,
            "target_condition": None,
        })
        for cond in CONDITIONS:
            anchor_probe_specs.append({
                "analysis": "within_condition",
                "anchor": anchor,
                "pair_name": cond,
                "source_condition": cond,
                "target_condition": cond,
            })
        for s in CONDITIONS:
            for t in CONDITIONS:
                if s == t:
                    continue
                anchor_probe_specs.append({
                    "analysis": "cross_condition",
                    "anchor": anchor,
                    "pair_name": f"{s}_to_{t}",
                    "source_condition": s,
                    "target_condition": t,
                })

    def filter_keys(spec: Dict[str, Any], k: Tuple[str, str, str]) -> bool:
        fam, cond, anchor = k
        if anchor != spec["anchor"]:
            return False
        analysis = spec["analysis"]
        if analysis == "overall_harmful":
            return cond in CONDITIONS
        if analysis == "within_condition":
            return cond == spec["source_condition"]
        if analysis == "cross_condition":
            s = spec["source_condition"]
            t = spec["target_condition"]
            return cond in (s, t)
        return False

    n_specs = len(anchor_probe_specs)
    for si, spec in enumerate(anchor_probe_specs):
        keys = [k for k in deltas.keys() if filter_keys(spec, k)]
        keys.sort()
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        groups_list: List[str] = []
        for k in keys:
            meta = metadata[k]
            lab = meta.get("harmful_label_primary")
            if lab is None:
                continue
            X_list.append(deltas[k])
            y_list.append(int(lab))
            groups_list.append(meta["family_id"])
        if not X_list:
            continue
        analysis = spec["analysis"]
        anchor = spec["anchor"]
        pair_name = spec["pair_name"]
        print(json.dumps({
            "status": "running_probe_spec",
            "index": si + 1,
            "total": n_specs,
            "analysis": analysis,
            "anchor": anchor,
            "pair": pair_name,
            "n": len(X_list),
            "n_families": len(set(groups_list)),
            "harmful_rate": float(np.mean(y_list)),
        }), flush=True)
        lw, pe = run_family_heldout_classification(keys, X_list, y_list, groups_list, layer_count)
        best_layer, best_ba = best_balanced_accuracy(lw)
        best_rows.append({
            "analysis": analysis,
            "anchor": anchor,
            "pair": pair_name,
            "best_layer": best_layer,
            "best_balanced_accuracy": f"{best_ba:.6f}",
        })
        for r in lw:
            overall_layerwise_rows.append({
                "analysis": analysis,
                "anchor": anchor,
                "pair": pair_name,
                **r,
            })
        if pe:
            pass  # We don't emit per-example predictions by default to keep output small; can add flag later.

    output_layerwise = Path(args.output_layerwise)
    output_summary = Path(args.output_summary)
    output_best = Path(args.output_best)
    output_layerwise.parent.mkdir(parents=True, exist_ok=True)
    if overall_layerwise_rows:
        fieldnames = list(overall_layerwise_rows[0].keys())
        with output_layerwise.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(overall_layerwise_rows)
    fieldnames_best = list(best_rows[0].keys()) if best_rows else []
    with output_best.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_best)
        w.writeheader()
        w.writerows(best_rows)

    lines: List[str] = []
    lines.append("Probe 6: early-position harmfulness detection (Qwen3-4B-Instruct-2507 36-family)")
    lines.append("")
    lines.append("Features: Δ = h(condition) - h(evidence_neutral) at anchor position")
    lines.append("Classifier: L2 logistic regression, class_weight=balanced, family-held-out (LeaveOneGroupOut)")
    lines.append(f"Layer count: {layer_count}")
    lines.append("")
    lines.append("Anchor order:")
    for i, anc in enumerate(ANCHOR_ORDER):
        lines.append(f"  {i}. {ANCHOR_DISPLAY[anc]}")
    lines.append("")

    def emit_block(analysis_filter: str, title: str) -> None:
        lines.append(f"== {title} ==")
        pairs_ordered: List[str] = []
        if analysis_filter == "overall_harmful":
            pairs_ordered = ["all_conditions_pooled"]
        elif analysis_filter == "within_condition":
            pairs_ordered = list(CONDITIONS)
        elif analysis_filter == "cross_condition":
            pairs_ordered = [f"{s}_to_{t}" for s in CONDITIONS for t in CONDITIONS if s != t]
        best_by_pair_anchor: Dict[Tuple[str, str], Tuple[int, float]] = {}
        for r in best_rows:
            if r["analysis"] != analysis_filter:
                continue
            best_by_pair_anchor[(r["pair"], r["anchor"])] = (int(r["best_layer"]), float(r["best_balanced_accuracy"]))
        lines.append("")
        header = f"{'pair':<72s}" + "".join(f"{ANCHOR_DISPLAY[anc][:5]:>9s}" for anc in ANCHOR_ORDER)
        lines.append(header)
        for pair in pairs_ordered:
            cells: List[str] = []
            for anc in ANCHOR_ORDER:
                key = (pair, anc)
                if key in best_by_pair_anchor:
                    lyr, val = best_by_pair_anchor[key]
                    cells.append(f"{val:6.3f}@L{lyr:<2d}")
                else:
                    cells.append("    -   ")
            lines.append(f"{pair:<72s}" + "".join(f"{c:>9s}" for c in cells))
        lines.append("")

    emit_block("overall_harmful", "1. Overall harmful vs nonharmful (3 conditions pooled)")
    emit_block("within_condition", "2. Within-condition harmfulness")
    emit_block("cross_condition", "3. Cross-condition harmfulness transfer")

    # Now answer the 5 specific summary questions.
    lines.append("== Summary answers ==")

    overall_by_anchor: Dict[str, Tuple[int, float]] = {}
    within_by: Dict[Tuple[str, str], Tuple[int, float]] = {}
    cross_by: Dict[Tuple[str, str], Tuple[int, float]] = {}
    for r in best_rows:
        pair = r["pair"]
        anchor = r["anchor"]
        val = (int(r["best_layer"]), float(r["best_balanced_accuracy"]))
        if r["analysis"] == "overall_harmful":
            overall_by_anchor[anchor] = val
        elif r["analysis"] == "within_condition":
            within_by[(pair, anchor)] = val
        elif r["analysis"] == "cross_condition":
            cross_by[(pair, anchor)] = val

    # Q1: earliest anchor overall where harmfulness is detectable (threshold BA > baseline, i.e. > majority)
    # For each anchor, compute majority baseline for overall analysis via a dummy: approximate by >0.5 since label balanced via balanced_weighting?
    # Actually read the stored layerwise baseline. Simpler: just take max BA per anchor over layer; if >0.55 call it detectable, and order.
    lines.append("Q1. What is the earliest anchor where harmfulness is detectable (overall pooled)?")
    detected_order: List[str] = []
    for anc in ANCHOR_ORDER:
        if anc in overall_by_anchor and overall_by_anchor[anc][1] > 0.55:
            detected_order.append(anc)
    if detected_order:
        earliest = detected_order[0]
        lyr, val = overall_by_anchor[earliest]
        lines.append(f"  Earliest detectable anchor: {ANCHOR_DISPLAY[earliest]}  (best BA {val:.3f} @ layer {lyr})")
    else:
        lines.append("  No anchor exceeded 0.55 balanced accuracy.")
    lines.append("  Per-anchor overall best balanced accuracy:")
    for anc in ANCHOR_ORDER:
        if anc in overall_by_anchor:
            lyr, val = overall_by_anchor[anc]
            lines.append(f"    {ANCHOR_DISPLAY[anc]}: BA {val:.3f} @ layer {lyr}")
    lines.append("")

    # Q2: false_pressure within: earliest detectable before final?
    lines.append("Q2. Is false_pressure within-condition harmfulness detectable before ANSWER position?")
    false_before: List[str] = []
    for anc in ANCHOR_ORDER:
        if anc == "final_answer_position":
            break
        key = ("evidence_false_belief_pressure", anc)
        if key in within_by and within_by[key][1] > 0.55:
            false_before.append(anc)
    final_key = ("evidence_false_belief_pressure", "final_answer_position")
    final_val = within_by.get(final_key, (-1, 0.0))
    if false_before:
        first = false_before[0]
        lyr, val = within_by[(final_key[0], first)]
        lines.append(f"  YES. Earliest before-final anchor: {ANCHOR_DISPLAY[first]}  (BA {val:.3f} @ layer {lyr})")
    else:
        lines.append(f"  NO. Best before-final did not exceed 0.55. Final-answer baseline BA {final_val[1]:.3f} @ layer {final_val[0]}.")
    lines.append("  False-pressure within per-anchor:")
    for anc in ANCHOR_ORDER:
        key = ("evidence_false_belief_pressure", anc)
        if key in within_by:
            lyr, val = within_by[key]
            lines.append(f"    {ANCHOR_DISPLAY[anc]}: BA {val:.3f} @ layer {lyr}")
    lines.append("")

    # Q3: emotional pressure earliest vs false pressure earliest
    lines.append("Q3. Is emotional pressure detectable earlier than false-belief pressure?")
    def earliest_within(cond: str) -> Optional[str]:
        out = None
        for anc in ANCHOR_ORDER:
            key = (cond, anc)
            if key in within_by and within_by[key][1] > 0.55:
                return anc
        return None
    f_earliest = earliest_within("evidence_false_belief_pressure")
    e_earliest = earliest_within("evidence_emotional_pressure")
    idx_f = ANCHOR_ORDER.index(f_earliest) if f_earliest else len(ANCHOR_ORDER)
    idx_e = ANCHOR_ORDER.index(e_earliest) if e_earliest else len(ANCHOR_ORDER)
    f_val = within_by.get(("evidence_false_belief_pressure", f_earliest), (-1, 0.0))[1] if f_earliest else 0.0
    e_val = within_by.get(("evidence_emotional_pressure", e_earliest), (-1, 0.0))[1] if e_earliest else 0.0
    lines.append(f"  False pressure earliest anchor: {ANCHOR_DISPLAY[f_earliest] if f_earliest else 'NONE'} (BA {f_val:.3f})")
    lines.append(f"  Emotional pressure earliest anchor: {ANCHOR_DISPLAY[e_earliest] if e_earliest else 'NONE'} (BA {e_val:.3f})")
    if idx_e < idx_f:
        lines.append("  YES: emotional pressure is detectable earlier than false-belief pressure.")
    elif idx_e == idx_f and idx_e < len(ANCHOR_ORDER):
        lines.append("  SAME: both pressures are first detectable at the same anchor.")
    else:
        lines.append("  NO: emotional pressure is NOT detectable earlier than false-belief pressure.")
    lines.append("")

    # Q4: false <-> emotional cross transfer before final ANSWER?
    lines.append("Q4. Does false⇄emotional transfer survive before the final ANSWER position?")
    pairs_false_emot = [
        "evidence_false_belief_pressure_to_evidence_emotional_pressure",
        "evidence_emotional_pressure_to_evidence_false_belief_pressure",
    ]
    for p in pairs_false_emot:
        lines.append(f"  Pair {p}:")
        for anc in ANCHOR_ORDER:
            key = (p, anc)
            if key in cross_by:
                lyr, val = cross_by[key]
                lines.append(f"    {ANCHOR_DISPLAY[anc]}: BA {val:.3f} @ layer {lyr}")
    before_final_ok: List[str] = []
    for p in pairs_false_emot:
        ok_anchors = []
        for anc in ANCHOR_ORDER:
            if anc == "final_answer_position":
                break
            key = (p, anc)
            if key in cross_by and cross_by[key][1] > 0.55:
                ok_anchors.append(anc)
        if ok_anchors:
            before_final_ok.append(f"{ok_anchors[0]}")
    if len(before_final_ok) == 2:
        lines.append("  YES: both directions of false⇄emotional transfer exceed 0.55 BA before the final ANSWER position.")
    elif len(before_final_ok) == 1:
        lines.append("  PARTIAL: one direction of false⇄emotional transfer exceeds 0.55 BA before final; the other does not.")
    else:
        lines.append("  NO: neither direction of false⇄emotional transfer exceeds 0.55 BA before the final ANSWER position.")
    lines.append("")

    # Q5: strongest early-position result permutation control. Pick anchor before final with best BA across all (analysis,pair,anchor) combos.
    best_before_final: Optional[Tuple[float, Dict[str, Any]]] = None
    for r in best_rows:
        anc = r["anchor"]
        if anc == "final_answer_position":
            continue
        val = float(r["best_balanced_accuracy"])
        if best_before_final is None or val > best_before_final[0]:
            best_before_final = (val, r)
    lines.append("Q5. Does the strongest early-position (pre-final-ANSWER) result survive the permutation control?")
    if best_before_final is None:
        lines.append("  N/A: no pre-final results available.")
        permutation_header_done = True
    else:
        r = best_before_final[1]
        analysis = r["analysis"]
        pair = r["pair"]
        anchor = r["anchor"]
        best_layer = int(r["best_layer"])
        best_ba = float(r["best_balanced_accuracy"])
        lines.append(f"  Strongest pre-final result: analysis={analysis}, pair={pair}, anchor={ANCHOR_DISPLAY[anchor]}, BA {best_ba:.3f} @ layer {best_layer}")
        permutation_header_done = False
        # Run permutations at best_layer only, inside main, below.

    if not args.skip_permutation and best_before_final is not None:
        r = best_before_final[1]
        analysis = r["analysis"]
        pair = r["pair"]
        anchor = r["anchor"]
        best_layer = int(r["best_layer"])
        real_ba = float(r["best_balanced_accuracy"])
        spec_obj = {
            "analysis": analysis,
            "pair_name": pair,
            "anchor": anchor,
        }
        if analysis == "within_condition":
            spec_obj["source_condition"] = pair
            spec_obj["target_condition"] = pair
        elif analysis == "cross_condition":
            # pair format is s_to_t
            s, t = pair.split("_to_", 1)
            spec_obj["source_condition"] = s
            spec_obj["target_condition"] = t
        else:
            spec_obj["source_condition"] = None
            spec_obj["target_condition"] = None
        keys = [k for k in deltas.keys() if filter_keys(spec_obj, k)]
        keys.sort()
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        groups_list: List[str] = []
        for k in keys:
            meta = metadata[k]
            lab = meta.get("harmful_label_primary")
            if lab is None:
                continue
            X_list.append(deltas[k])
            y_list.append(int(lab))
            groups_list.append(meta["family_id"])
        if X_list:
            tensor = np.stack(X_list, axis=0)
            X = tensor[:, best_layer, :]
            y = np.asarray(y_list, dtype=np.int64)
            groups = np.asarray(groups_list, dtype=object)
            logo = LeaveOneGroupOut()
            folds = list(logo.split(np.arange(len(X)), y, groups=groups))
            def evaluate_with_permutation(permute: bool, seed: int) -> float:
                y_true_all = []
                y_pred_all = []
                master_rng = np.random.default_rng(seed) if permute else None
                for fid, (train_idx, test_idx) in enumerate(folds):
                    train_x = X[train_idx]
                    test_x = X[test_idx]
                    train_y = y[train_idx].copy()
                    test_y = y[test_idx]
                    if permute and master_rng is not None:
                        rng = np.random.default_rng(int(master_rng.integers(0, 2**60)))
                        train_y = rng.permutation(train_y)
                    scaler = StandardScaler()
                    train_x = scaler.fit_transform(train_x)
                    test_x = scaler.transform(test_x)
                    model = LogisticRegression(
                        penalty="l2",
                        class_weight="balanced",
                        max_iter=10000,
                        C=1.0,
                    )
                    try:
                        model.fit(train_x, train_y)
                        prob = model.predict_proba(test_x)[:, 1]
                        pred = (prob >= 0.5).astype(np.int64)
                    except Exception:
                        pred = np.zeros(len(test_x), dtype=np.int64)
                    y_true_all.append(test_y)
                    y_pred_all.append(pred)
                yt = np.concatenate(y_true_all)
                yp = np.concatenate(y_pred_all)
                return float(balanced_accuracy_score(yt, yp))
            perm_values: List[float] = []
            master_seed = int(np.random.default_rng(137).integers(0, 2**60))
            for rep in range(args.permutation_pairs):
                val = evaluate_with_permutation(True, master_seed + rep)
                perm_values.append(val)
                if (rep + 1) % max(1, args.permutation_pairs // 5) == 0 or (rep + 1) == args.permutation_pairs:
                    print(json.dumps({
                        "status": "permutation_progress",
                        "repeat": rep + 1,
                        "total": args.permutation_pairs,
                    }), flush=True)
            perm_mean = float(np.mean(perm_values))
            perm_p95 = percentile(perm_values, 95.0)
            p_count = sum(1 for v in perm_values if v >= real_ba)
            emp_p = (p_count + 1) / (len(perm_values) + 1)
            exceeds = real_ba > perm_p95
            lines.append(f"  Permutation control (N={args.permutation_pairs}, training-label shuffle within each leave-one-family-out fold, fixed best layer {best_layer}):")
            lines.append(f"    permutation mean BA: {perm_mean:.3f}")
            lines.append(f"    permutation 95th percentile BA: {perm_p95:.3f}")
            lines.append(f"    empirical p-value: {emp_p:.4f}")
            lines.append(f"    real BA exceeds perm p95: {'YES' if exceeds else 'NO'}")
            lines.append(f"  Conclusion: {'SURVIVES' if exceeds else 'FAILS'} the permutation control.")

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "done",
        "output_layerwise": str(output_layerwise),
        "output_summary": str(output_summary),
        "output_best": str(output_best),
    }), flush=True)


if __name__ == "__main__":
    main()
