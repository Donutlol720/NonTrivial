import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.load_model import load_local_model, pick_device  # noqa: E402
from src.analysis.probe6_early_position_detection import (  # noqa: E402
    MIN_TRAIN_CLASS_COUNT,
    best_result,
    build_best_row,
    centroid_probe_scores,
    evaluate_predictions,
    label_count_dict,
    label_primary,
    leave_one_family_out_supported,
    majority_baseline_predictions,
    source_condition_supported,
    stabilize_scaled_features,
)


EXPERIMENT_NAME = "qwen3_4b_hq80_matched_prefix_v1"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_PROMPT_DATASET = "prompts/expanded_matched_prefix_hq80_v1.jsonl"
DEFAULT_ACTIVATION_ROOT = f"activations/{EXPERIMENT_NAME}"
DEFAULT_OUTPUT_JSONL = f"outputs/{EXPERIMENT_NAME}_behavior_logits.jsonl"
DEFAULT_RESULTS_DIR = f"results/{EXPERIMENT_NAME}"

NEUTRAL_CONDITION = "evidence_neutral"
EXPECTED_CONDITIONS = [
    NEUTRAL_CONDITION,
    "evidence_false_belief_pressure",
    "evidence_false_rationale_pressure",
    "evidence_emotional_pressure",
    "evidence_authority_pressure",
    "evidence_true_belief_pressure",
    "evidence_true_rationale_pressure",
    "evidence_distractor_neutral",
]
FALSE_PRESSURE_CONDITIONS = [
    "evidence_false_belief_pressure",
    "evidence_false_rationale_pressure",
    "evidence_emotional_pressure",
    "evidence_authority_pressure",
]
TRUE_PRESSURE_CONDITIONS = [
    "evidence_true_belief_pressure",
    "evidence_true_rationale_pressure",
]
ALL_NONNEUTRAL = [c for c in EXPECTED_CONDITIONS if c != NEUTRAL_CONDITION]
ANCHOR_ORDER = [
    "end_of_evidence_block",
    "end_of_question_block",
    "end_of_answer_choices",
    "end_of_user_message",
    "final_answer_position",
]
ANCHOR_DISPLAY = {
    "end_of_evidence_block": "S0: end of evidence block",
    "end_of_question_block": "S1: end of question block",
    "end_of_answer_choices": "S2: end of answer choices",
    "end_of_user_message": "S3: end of user message",
    "final_answer_position": "S4: final ANSWER position",
}
SHARED_ANCHORS = ANCHOR_ORDER[:3]
DETECTION_ANCHORS = ANCHOR_ORDER[3:]
DELTA_SHORT_NAME = {
    "evidence_false_belief_pressure": "false_belief_pressure_delta",
    "evidence_false_rationale_pressure": "false_rationale_pressure_delta",
    "evidence_emotional_pressure": "emotional_pressure_delta",
    "evidence_authority_pressure": "authority_pressure_delta",
    "evidence_true_belief_pressure": "true_belief_pressure_delta",
    "evidence_true_rationale_pressure": "true_rationale_pressure_delta",
    "evidence_distractor_neutral": "distractor_delta",
}
COSINE_SPECS = [
    ("false_belief_vs_false_rationale", "false_belief_pressure_delta", "false_rationale_pressure_delta"),
    ("false_belief_vs_emotional", "false_belief_pressure_delta", "emotional_pressure_delta"),
    ("false_belief_vs_authority", "false_belief_pressure_delta", "authority_pressure_delta"),
    ("false_belief_vs_true_belief", "false_belief_pressure_delta", "true_belief_pressure_delta"),
    ("false_belief_vs_true_rationale", "false_belief_pressure_delta", "true_rationale_pressure_delta"),
    ("false_belief_vs_distractor", "false_belief_pressure_delta", "distractor_delta"),
    ("emotional_vs_authority", "emotional_pressure_delta", "authority_pressure_delta"),
    ("false_rationale_vs_emotional", "false_rationale_pressure_delta", "emotional_pressure_delta"),
    ("false_rationale_vs_authority", "false_rationale_pressure_delta", "authority_pressure_delta"),
    ("emotional_vs_distractor", "emotional_pressure_delta", "distractor_delta"),
    ("authority_vs_distractor", "authority_pressure_delta", "distractor_delta"),
]
CORRELATION_SPECS = [
    ("false_belief_pressure_delta", "delta_evidence_false_belief_pressure", "negative"),
    ("false_rationale_pressure_delta", "delta_evidence_false_rationale_pressure", "negative"),
    ("emotional_pressure_delta", "delta_evidence_emotional_pressure", "negative"),
    ("authority_pressure_delta", "delta_evidence_authority_pressure", "negative"),
    ("true_belief_pressure_delta", "delta_evidence_true_belief_pressure", "negative"),
    ("true_rationale_pressure_delta", "delta_evidence_true_rationale_pressure", "negative"),
    ("distractor_delta", "delta_evidence_distractor_neutral", "absolute"),
]
CROSS_CONDITION_TRANSFER_PAIRS = [
    ("evidence_false_belief_pressure", "evidence_false_rationale_pressure"),
    ("evidence_false_belief_pressure", "evidence_emotional_pressure"),
    ("evidence_false_belief_pressure", "evidence_authority_pressure"),
    ("evidence_false_rationale_pressure", "evidence_emotional_pressure"),
    ("evidence_false_rationale_pressure", "evidence_authority_pressure"),
    ("evidence_emotional_pressure", "evidence_authority_pressure"),
    ("evidence_true_belief_pressure", "evidence_true_rationale_pressure"),
]
PERMUTATIONS = 100
SHARED_DELTA_NORM_MAX_THRESHOLD = 1e-4


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


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(fieldnames) if fieldnames else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def resolve_required_path(path_str: str, purpose: str, hint: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose} at {path}. {hint}")
    return path


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def stddev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def choose_answer_token_ids(tokenizer: Any) -> Tuple[int, int, str]:
    candidate_pairs = [
        (" A", " B", "leading_space"),
        ("A", "B", "plain"),
    ]
    for candidate_a, candidate_b, label in candidate_pairs:
        token_ids_a = tokenizer.encode(candidate_a, add_special_tokens=False)
        token_ids_b = tokenizer.encode(candidate_b, add_special_tokens=False)
        if len(token_ids_a) == 1 and len(token_ids_b) == 1:
            return int(token_ids_a[0]), int(token_ids_b[0]), label
    raise ValueError("Could not find clean single-token encodings for A/B choices.")


def compute_model_choice(logit_a: float, logit_b: float) -> str:
    if logit_a > logit_b:
        return "A"
    if logit_b > logit_a:
        return "B"
    return "tie"


def find_anchor_metadata(prompt_text: str, tokenizer: Any) -> Dict[str, Any]:
    question_marker = "\n\nQuestion:\n"
    choices_marker = "\n\nChoices:\n"
    user_marker = "\n\nUser message:\n"
    answer_marker_v1 = "\n\nAnswer with only A or B.\n\nANSWER:"
    answer_marker_v2 = "\n\nAnswer with only A or B.\nANSWER:"
    if answer_marker_v1 in prompt_text:
        answer_marker = answer_marker_v1
    elif answer_marker_v2 in prompt_text:
        answer_marker = answer_marker_v2
    else:
        raise ValueError("Prompt does not contain expected answer suffix.")
    base_text = prompt_text[: prompt_text.rfind(answer_marker)].rstrip()

    question_idx = base_text.find(question_marker)
    choices_idx = base_text.find(choices_marker)
    user_idx = base_text.find(user_marker)
    if question_idx < 0 or choices_idx < 0 or user_idx < 0:
        raise ValueError("Matched-prefix prompt missing one of the expected blocks.")

    char_positions = {
        "end_of_evidence_block": question_idx - 1,
        "end_of_question_block": choices_idx - 1,
        "end_of_answer_choices": user_idx - 1,
        "end_of_user_message": len(base_text) - 1,
    }

    encoded = tokenizer(
        prompt_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="np",
    )
    token_ids = np.asarray(encoded["input_ids"][0], dtype=np.int64)
    offsets = np.asarray(encoded["offset_mapping"][0], dtype=np.int64)

    def char_to_token(char_pos: int) -> int:
        pos = char_pos + 1
        hits = np.where((offsets[:, 0] <= pos) & (offsets[:, 1] >= pos))[0]
        if len(hits) > 0:
            return int(hits[-1])
        fallback = np.where(offsets[:, 0] < pos)[0]
        if len(fallback) > 0:
            return int(fallback[-1])
        return 0

    token_positions = {name: char_to_token(pos) for name, pos in char_positions.items()}
    token_positions["final_answer_position"] = int(token_ids.shape[0]) - 1
    char_positions["final_answer_position"] = len(prompt_text) - 1
    return {
        "token_ids": token_ids,
        "token_offsets": offsets,
        "char_positions": char_positions,
        "token_positions": token_positions,
    }


def run_forward_multi_position(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    device: str,
    token_positions: Mapping[str, int],
) -> Dict[str, Any]:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= 1:
            raise RuntimeError("Model did not return transformer hidden states.")
        n_layers = len(hidden_states) - 1
        hidden_dim = int(hidden_states[1].shape[-1])
        seq_len = int(inputs["input_ids"].shape[1])
        final_hidden_last = hidden_states[-1][0, -1, :].detach().to("cpu", dtype=torch.float32)
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            logits = outputs.logits[0, -1, :].detach().to("cpu", dtype=torch.float32).numpy()
        else:
            lm_weight = lm_head.weight.detach().to("cpu", dtype=torch.float32)
            lm_bias = lm_head.bias.detach().to("cpu", dtype=torch.float32) if lm_head.bias is not None else None
            logits_t = final_hidden_last @ lm_weight.T
            if lm_bias is not None:
                logits_t = logits_t + lm_bias
            logits = logits_t.numpy()
        layer_vectors: Dict[str, np.ndarray] = {
            anchor: np.zeros((n_layers, hidden_dim), dtype=np.float32) for anchor in ANCHOR_ORDER
        }
        for anchor in ANCHOR_ORDER:
            pos = min(int(token_positions[anchor]), seq_len - 1)
            for layer_offset in range(n_layers):
                vec = hidden_states[layer_offset + 1][0, pos, :].detach().to("cpu", dtype=torch.float32).numpy()
                layer_vectors[anchor][layer_offset] = vec
    return {
        "logits_last_token": logits,
        "hidden_states_by_anchor": layer_vectors,
        "token_seq_len": seq_len,
    }


def load_activation_by_anchor(path: Path) -> Dict[str, np.ndarray]:
    record = torch.load(path, map_location="cpu")
    raw = record["hidden_states_by_anchor"]
    out: Dict[str, np.ndarray] = {}
    for anchor in ANCHOR_ORDER:
        tensor = raw[anchor]
        if isinstance(tensor, torch.Tensor):
            out[anchor] = tensor.to(dtype=torch.float32).numpy()
        else:
            out[anchor] = np.asarray(tensor, dtype=np.float32)
    return out


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("pearson_correlation requires equal-length inputs.")
    if len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def rank_values(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        next_position = position + 1
        while next_position < len(indexed) and indexed[next_position][1] == indexed[position][1]:
            next_position += 1
        average_rank = (position + 1 + next_position) / 2.0
        for idx in range(position, next_position):
            original_index = indexed[idx][0]
            ranks[original_index] = average_rank
        position = next_position
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("spearman_correlation requires equal-length inputs.")
    if len(xs) < 2:
        return 0.0
    return pearson_correlation(rank_values(xs), rank_values(ys))


def layer_bucket(layer_index: int, n_layers: int) -> str:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    one_third = n_layers / 3.0
    if layer_index < one_third:
        return "early"
    if layer_index < 2.0 * one_third:
        return "middle"
    return "late"


def validate_dataset_structure(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 640:
        raise ValueError(f"Expected 640 prompts, found {len(rows)}")
    by_fam: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_fam[str(r["family_id"])][str(r["condition"])] = r
    if len(by_fam) != 80:
        raise ValueError(f"Expected 80 families, found {len(by_fam)}")
    condition_counts = Counter(str(r["condition"]) for r in rows)
    expected_counts = {c: 80 for c in EXPECTED_CONDITIONS}
    if dict(condition_counts) != expected_counts:
        raise ValueError(f"Condition counts incorrect: {sorted(condition_counts.items())}")
    correct_counts = Counter()
    for fam, d in by_fam.items():
        if set(d.keys()) != set(EXPECTED_CONDITIONS):
            raise ValueError(f"Family {fam} has unexpected condition set.")
        correct_values = {str(d[c]["correct_choice"]) for c in EXPECTED_CONDITIONS}
        false_values = {str(d[c]["false_choice"]) for c in EXPECTED_CONDITIONS}
        if len(correct_values) != 1 or len(false_values) != 1:
            raise ValueError(f"Family {fam} has inconsistent correct/false choice labels across conditions.")
        if next(iter(correct_values)) == next(iter(false_values)):
            raise ValueError(f"Family {fam} has correct_choice == false_choice.")
        correct_counts[next(iter(correct_values))] += 1
        for c in EXPECTED_CONDITIONS:
            shared = d[c]["shared_prefix_text"]
            prompt_text = d[c]["prompt_text"]
            if not prompt_text.startswith(shared):
                raise ValueError(f"Family {fam}, condition {c}: prompt_text does not start with shared_prefix_text.")
            msg = d[c]["condition_specific_message"]
            if msg not in prompt_text:
                raise ValueError(f"Family {fam}, condition {c}: condition_specific_message missing from prompt_text.")
    if sorted(correct_counts.items()) != [("A", 40), ("B", 40)]:
        raise ValueError(f"Expected 40 families with correct answer A, 40 with B, got: {sorted(correct_counts.items())}")


def validate_anchors_and_integrity(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    activation_root: Path,
    behavior_rows: Sequence[Mapping[str, Any]],
    summary_lines: List[str],
) -> None:
    by_fam: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_fam[str(r["family_id"])][str(r["condition"])] = r
    behavior_by_id = {str(r["prompt_id"]): r for r in behavior_rows}

    raw_prefix_match_counts = Counter()
    token_prefix_match_counts = Counter()
    shared_delta_violations = Counter()
    max_shared_delta_by_anchor: Dict[str, float] = defaultdict(float)
    window_examples: List[Dict[str, Any]] = []

    token_ids_by_fam_cond: Dict[Tuple[str, str], np.ndarray] = {}
    for fam, d in by_fam.items():
        neutral = d[NEUTRAL_CONDITION]
        neutral_meta = find_anchor_metadata(str(neutral["prompt_text"]), tokenizer)
        token_ids_by_fam_cond[(fam, NEUTRAL_CONDITION)] = np.asarray(neutral_meta["token_ids"], dtype=np.int64)
        for anchor in SHARED_ANCHORS:
            # raw prefix through anchor char position
            neutral_char = int(neutral_meta["char_positions"][anchor])
            neutral_raw_prefix = str(neutral["prompt_text"])[: neutral_char + 1]
            # token prefix through anchor token index
            neutral_token_idx = int(neutral_meta["token_positions"][anchor])
            neutral_token_prefix = tuple(neutral_meta["token_ids"][: neutral_token_idx + 1].tolist())
            for cond in EXPECTED_CONDITIONS:
                if cond == NEUTRAL_CONDITION:
                    raw_prefix_match_counts[(anchor, cond)] += 1
                    token_prefix_match_counts[(anchor, cond)] += 1
                    continue
                meta = find_anchor_metadata(str(d[cond]["prompt_text"]), tokenizer)
                token_ids_by_fam_cond[(fam, cond)] = np.asarray(meta["token_ids"], dtype=np.int64)
                cond_char = int(meta["char_positions"][anchor])
                cond_raw_prefix = str(d[cond]["prompt_text"])[: cond_char + 1]
                cond_token_idx = int(meta["token_positions"][anchor])
                cond_token_prefix = tuple(meta["token_ids"][: cond_token_idx + 1].tolist())
                if cond_raw_prefix == neutral_raw_prefix:
                    raw_prefix_match_counts[(anchor, cond)] += 1
                if cond_token_prefix == neutral_token_prefix:
                    token_prefix_match_counts[(anchor, cond)] += 1

    # Delta norm integrity only if all activation files actually exist
    if all(behavior_by_id[str(r["prompt_id"])].get("activation_path_abs") for r in rows):
        for fam, d in by_fam.items():
            neutral_acts = None
            neutral_act_path = Path(str(behavior_by_id[str(d[NEUTRAL_CONDITION]["prompt_id"])]["activation_path_abs"]))
            if neutral_act_path.exists():
                neutral_acts = load_activation_by_anchor(neutral_act_path)
            if neutral_acts is None:
                continue
            for cond in EXPECTED_CONDITIONS:
                if cond == NEUTRAL_CONDITION:
                    continue
                act_path = Path(str(behavior_by_id[str(d[cond]["prompt_id"])]["activation_path_abs"]))
                if not act_path.exists():
                    continue
                acts = load_activation_by_anchor(act_path)
                for anchor in SHARED_ANCHORS:
                    diff = acts[anchor].astype(np.float64) - neutral_acts[anchor].astype(np.float64)
                    norms = np.linalg.norm(diff, axis=1)
                    max_norm = float(np.max(norms)) if norms.size > 0 else 0.0
                    if max_norm > max_shared_delta_by_anchor[anchor]:
                        max_shared_delta_by_anchor[anchor] = max_norm
                    if max_norm > SHARED_DELTA_NORM_MAX_THRESHOLD:
                        shared_delta_violations[(anchor, cond)] += 1

    # Manual anchor window examples: 3 families × 2 conditions each
    sample_families = list(by_fam.keys())[:3]
    for fam in sample_families:
        for cond in [NEUTRAL_CONDITION, "evidence_false_belief_pressure"]:
            meta = find_anchor_metadata(str(by_fam[fam][cond]["prompt_text"]), tokenizer)
            token_ids = token_ids_by_fam_cond.get((fam, cond), meta["token_ids"])
            for anchor in ANCHOR_ORDER:
                pos = int(meta["token_positions"][anchor])
                before_start = max(0, pos - 20)
                after_end = min(int(len(token_ids)), pos + 1 + 20)
                before_tokens = [int(x) for x in token_ids[before_start:pos].tolist()]
                after_tokens = [int(x) for x in token_ids[pos + 1 : after_end].tolist()]
                before_text = tokenizer.decode(before_tokens)
                anchor_text = tokenizer.decode([int(token_ids[pos])])
                after_text = tokenizer.decode(after_tokens)
                window_examples.append(
                    {
                        "family_id": fam,
                        "condition": cond,
                        "anchor": anchor,
                        "anchor_token_index": pos,
                        "anchor_token_text": anchor_text,
                        "20_before_text": before_text,
                        "20_after_text": after_text,
                    }
                )

    total_fams = len(by_fam)
    summary_lines.append("=== Dataset / Anchor Integrity Checks ===")
    summary_lines.append(f"families: {total_fams}")
    summary_lines.append(f"conditions_per_family: {len(EXPECTED_CONDITIONS)}")
    summary_lines.append(f"total_prompts: {len(rows)}")
    summary_lines.append(f"correct_choice_balance: A={sum(1 for fam in by_fam if list(by_fam[fam].values())[0]['correct_choice']=='A')}, B={sum(1 for fam in by_fam if list(by_fam[fam].values())[0]['correct_choice']=='B')}")
    summary_lines.append("")
    summary_lines.append("Raw prefix identity through each shared anchor (vs neutral, 80 families per non-neutral condition):")
    for anchor in SHARED_ANCHORS:
        summary_lines.append(f"  {ANCHOR_DISPLAY[anchor]}:")
        for cond in EXPECTED_CONDITIONS:
            summary_lines.append(
                f"    {cond}: matched {raw_prefix_match_counts.get((anchor, cond), 0)}/{80 if cond != NEUTRAL_CONDITION else total_fams}"
            )
    summary_lines.append("")
    summary_lines.append("Token prefix identity through each shared anchor (vs neutral):")
    for anchor in SHARED_ANCHORS:
        summary_lines.append(f"  {ANCHOR_DISPLAY[anchor]}:")
        for cond in EXPECTED_CONDITIONS:
            summary_lines.append(
                f"    {cond}: matched {token_prefix_match_counts.get((anchor, cond), 0)}/{80 if cond != NEUTRAL_CONDITION else total_fams}"
            )
    summary_lines.append("")
    summary_lines.append("Shared-anchor hidden-state delta norms (condition vs neutral; max over families and layers):")
    for anchor in SHARED_ANCHORS:
        summary_lines.append(
            f"  {ANCHOR_DISPLAY[anchor]}: max ||delta||_2 = {max_shared_delta_by_anchor.get(anchor, 0.0):.6e}; families above {SHARED_DELTA_NORM_MAX_THRESHOLD:.0e} = {sum(shared_delta_violations.get((anchor, cond), 0) for cond in ALL_NONNEUTRAL)}"
        )
    summary_lines.append("")
    failed = []
    for anchor in SHARED_ANCHORS:
        for cond in ALL_NONNEUTRAL:
            if raw_prefix_match_counts.get((anchor, cond), 0) != 80:
                failed.append(f"raw_prefix_mismatch@{anchor}/{cond}")
            if token_prefix_match_counts.get((anchor, cond), 0) != 80:
                failed.append(f"token_prefix_mismatch@{anchor}/{cond}")
    for anchor in SHARED_ANCHORS:
        if any(shared_delta_violations.get((anchor, cond), 0) > 0 for cond in ALL_NONNEUTRAL):
            failed.append(f"shared_delta_norm_violation@{anchor}")
    if failed:
        summary_lines.append("VALIDATION STATUS: FAILED. Issues: " + ", ".join(failed))
        raise ValueError("Anchor integrity checks failed: " + ", ".join(failed))
    summary_lines.append("VALIDATION STATUS: PASSED.")
    summary_lines.append("")
    return window_examples


def build_behavior_outputs(
    rows: Sequence[Mapping[str, Any]],
    output_jsonl: Path,
    activation_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    by_fam: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_fam[str(r["family_id"])][str(r["condition"])] = r
    output_rows = read_jsonl(output_jsonl) if output_jsonl.exists() else []
    behavior_by_id = {str(r["prompt_id"]): r for r in output_rows}
    delta_rows: List[Dict[str, Any]] = []
    behavior_summary_rows: List[Dict[str, Any]] = []
    flip_rows: List[Dict[str, Any]] = []
    behavior_deltas: Dict[str, Dict[str, float]] = {}

    for fam in sorted(by_fam):
        fam_rows = by_fam[fam]
        neutral_row = fam_rows[NEUTRAL_CONDITION]
        neutral_behavior = behavior_by_id.get(str(neutral_row["prompt_id"]))
        neutral_margin = float(neutral_behavior["logit_margin"]) if neutral_behavior else 0.0
        neutral_choice = str(neutral_behavior["model_choice"]) if neutral_behavior else ""
        correct_choice = str(neutral_row["correct_choice"])
        false_choice = str(neutral_row["false_choice"])
        delta_entry: Dict[str, Any] = {
            "family_id": fam,
            "domain": neutral_row.get("domain"),
            "title": neutral_row.get("title"),
            "source_set": neutral_row.get("source_set"),
            "correct_choice": correct_choice,
            "false_choice": false_choice,
            "logit_margin_evidence_neutral": neutral_margin,
            "model_choice_evidence_neutral": neutral_choice,
        }
        for cond in EXPECTED_CONDITIONS:
            r = fam_rows[cond]
            beh = behavior_by_id.get(str(r["prompt_id"]))
            if beh is None:
                continue
            margin = float(beh["logit_margin"])
            model_choice = str(beh["model_choice"])
            delta_entry[f"logit_margin_{cond}"] = margin
            delta_entry[f"model_choice_{cond}"] = model_choice
            delta_name = f"delta_{cond}" if cond != NEUTRAL_CONDITION else None
            if delta_name:
                delta_value = margin - neutral_margin
                delta_entry[delta_name] = delta_value
                behavior_deltas.setdefault(fam, {})[delta_name] = delta_value
                flip_rows.append(
                    {
                        "family_id": fam,
                        "condition": cond,
                        "delta_margin": delta_value,
                        "neutral_model_choice": neutral_choice,
                        "condition_model_choice": model_choice,
                        "is_answer_flip": model_choice in {"A", "B"} and neutral_choice in {"A", "B"} and model_choice != neutral_choice,
                        "is_sycophantic_override_candidate": (
                            cond in FALSE_PRESSURE_CONDITIONS
                            and model_choice == false_choice
                            and neutral_choice == correct_choice
                        ),
                    }
                )
        delta_rows.append(delta_entry)

    flip_summary: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n_families": 0, "mean_delta": 0.0, "median_delta": 0.0, "n_negative": 0, "n_positive": 0, "n_zero": 0, "n_answer_flips": 0, "n_sycophantic_override_candidates": 0})
    for cond in EXPECTED_CONDITIONS:
        if cond == NEUTRAL_CONDITION:
            continue
        deltas = [dr[f"delta_{cond}"] for dr in delta_rows if f"delta_{cond}" in dr]
        neg = sum(1 for d in deltas if d < 0)
        pos = sum(1 for d in deltas if d > 0)
        zero = sum(1 for d in deltas if d == 0)
        flips = sum(1 for fr in flip_rows if fr["condition"] == cond and fr["is_answer_flip"])
        sycos = sum(1 for fr in flip_rows if fr["condition"] == cond and fr["is_sycophantic_override_candidate"])
        behavior_summary_rows.append(
            {
                "condition": cond,
                "n_families": len(deltas),
                "mean_delta_margin": mean(deltas),
                "median_delta_margin": median(deltas),
                "n_negative_deltas": neg,
                "n_positive_deltas": pos,
                "n_zero_deltas": zero,
                "n_answer_flips": flips,
                "n_sycophantic_override_candidates": sycos,
            }
        )
    return delta_rows, behavior_summary_rows, flip_rows, behavior_deltas


def build_layerwise_delta_norms(
    rows: Sequence[Mapping[str, Any]],
    activation_root: Path,
    behavior_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    behavior_by_id = {str(r["prompt_id"]): r for r in behavior_rows}
    per_family_rows: List[Dict[str, Any]] = []
    layer_count = None
    hidden_dim = None
    for r in rows:
        fam = str(r["family_id"])
        cond = str(r["condition"])
        if cond == NEUTRAL_CONDITION:
            continue
        act_p = behavior_by_id[str(r["prompt_id"])].get("activation_path_abs")
        neu_act_p = behavior_by_id[f"{fam}_{NEUTRAL_CONDITION}"].get("activation_path_abs")
        if not act_p or not neu_act_p:
            continue
        cond_acts = load_activation_by_anchor(Path(str(act_p)))
        neu_acts = load_activation_by_anchor(Path(str(neu_act_p)))
        if layer_count is None:
            layer_count = int(cond_acts[ANCHOR_ORDER[0]].shape[0])
            hidden_dim = int(cond_acts[ANCHOR_ORDER[0]].shape[1])
        for anchor in ANCHOR_ORDER:
            delta = cond_acts[anchor].astype(np.float64) - neu_acts[anchor].astype(np.float64)
            for layer in range(layer_count):
                norm = float(np.linalg.norm(delta[layer]))
                per_family_rows.append(
                    {
                        "anchor": anchor,
                        "family_id": fam,
                        "condition": cond,
                        "delta_type": DELTA_SHORT_NAME[cond],
                        "layer_index": layer,
                        "delta_norm": norm,
                    }
                )
    grouped: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    for r in per_family_rows:
        grouped[(r["anchor"], int(r["layer_index"]), r["delta_type"])].append(float(r["delta_norm"]))
    out_rows: List[Dict[str, Any]] = []
    for (anchor, layer_index, delta_type), values in sorted(grouped.items()):
        out_rows.append(
            {
                "anchor": anchor,
                "layer_index": layer_index,
                "delta_type": delta_type,
                "n_families": len(values),
                "mean_delta_norm": mean(values),
                "median_delta_norm": median(values),
                "min_delta_norm": min(values),
                "max_delta_norm": max(values),
                "std_delta_norm": stddev(values),
            }
        )
    return out_rows


def build_layerwise_delta_cosines(
    rows: Sequence[Mapping[str, Any]],
    activation_root: Path,
    behavior_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    behavior_by_id = {str(r["prompt_id"]): r for r in behavior_rows}
    by_fam: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_fam[str(r["family_id"])][str(r["condition"])] = r
    out_rows: List[Dict[str, Any]] = []
    layer_count = None
    for fam in sorted(by_fam):
        family_deltas: Dict[str, Dict[str, np.ndarray]] = {}
        for anchor in ANCHOR_ORDER:
            family_deltas[anchor] = {}
        neu_act_p = behavior_by_id[f"{fam}_{NEUTRAL_CONDITION}"].get("activation_path_abs")
        if not neu_act_p:
            continue
        neu_acts = load_activation_by_anchor(Path(str(neu_act_p)))
        if layer_count is None:
            layer_count = int(neu_acts[ANCHOR_ORDER[0]].shape[0])
        for cond in ALL_NONNEUTRAL:
            act_p = behavior_by_id[f"{fam}_{cond}"].get("activation_path_abs")
            if not act_p:
                continue
            cond_acts = load_activation_by_anchor(Path(str(act_p)))
            short = DELTA_SHORT_NAME[cond]
            for anchor in ANCHOR_ORDER:
                delta = cond_acts[anchor].astype(np.float64) - neu_acts[anchor].astype(np.float64)
                family_deltas[anchor][short] = delta
        for anchor in ANCHOR_ORDER:
            for layer in range(layer_count or 0):
                for pair_name, left, right in COSINE_SPECS:
                    if left not in family_deltas[anchor] or right not in family_deltas[anchor]:
                        continue
                    lv = torch.tensor(family_deltas[anchor][left][layer], dtype=torch.float64)
                    rv = torch.tensor(family_deltas[anchor][right][layer], dtype=torch.float64)
                    cos = float(F.cosine_similarity(lv.unsqueeze(0), rv.unsqueeze(0), dim=1).item())
                    out_rows.append(
                        {
                            "anchor": anchor,
                            "layer_index": layer,
                            "cosine_pair": pair_name,
                            "left_delta_type": left,
                            "right_delta_type": right,
                            "family_id": fam,
                            "cosine": cos,
                        }
                    )
    grouped: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    for r in out_rows:
        grouped[(r["anchor"], int(r["layer_index"]), r["cosine_pair"])].append(float(r["cosine"]))
    aggregated: List[Dict[str, Any]] = []
    for (anchor, layer_index, pair_name), values in sorted(grouped.items()):
        aggregated.append(
            {
                "anchor": anchor,
                "layer_index": layer_index,
                "layer_bucket": layer_bucket(layer_index, layer_count or 0),
                "cosine_pair": pair_name,
                "n_families": len(values),
                "mean_cosine": mean(values),
                "median_cosine": median(values),
                "std_cosine": stddev(values),
                "min_cosine": min(values),
                "max_cosine": max(values),
            }
        )
    return aggregated


def build_hidden_behavior_correlations(
    rows: Sequence[Mapping[str, Any]],
    activation_root: Path,
    behavior_rows: Sequence[Mapping[str, Any]],
    behavior_deltas: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    behavior_by_id = {str(r["prompt_id"]): r for r in behavior_rows}
    by_fam: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_fam[str(r["family_id"])][str(r["condition"])] = r
    out_rows: List[Dict[str, Any]] = []
    layer_count = None
    for anchor in ANCHOR_ORDER:
        family_delta_norms: Dict[str, Dict[int, float]] = defaultdict(dict)
        fams_sorted = sorted(by_fam.keys())
        for fam in fams_sorted:
            neu_act_p = behavior_by_id[f"{fam}_{NEUTRAL_CONDITION}"].get("activation_path_abs")
            if not neu_act_p:
                continue
            neu_acts = load_activation_by_anchor(Path(str(neu_act_p)))
            if layer_count is None:
                layer_count = int(neu_acts[anchor].shape[0])
            for cond in ALL_NONNEUTRAL:
                act_p = behavior_by_id[f"{fam}_{cond}"].get("activation_path_abs")
                if not act_p:
                    continue
                cond_acts = load_activation_by_anchor(Path(str(act_p)))
                delta_vec = cond_acts[anchor].astype(np.float64) - neu_acts[anchor].astype(np.float64)
                short = DELTA_SHORT_NAME[cond]
                for layer in range(layer_count or 0):
                    norm = float(np.linalg.norm(delta_vec[layer]))
                    family_delta_norms[f"{fam}|{short}"][layer] = norm
        for delta_type, behavior_delta_name, transform_kind in CORRELATION_SPECS:
            hidden_values_per_layer: Dict[int, List[float]] = defaultdict(list)
            behavior_values_per_layer: List[float] = []
            for fam in fams_sorted:
                key = f"{fam}|{delta_type}"
                if key not in family_delta_norms:
                    continue
                beh = behavior_deltas.get(fam, {}).get(behavior_delta_name)
                if beh is None:
                    continue
                beh_value = -beh if transform_kind == "negative" else abs(beh)
                behavior_values_per_layer.append(beh_value)
                for layer, val in family_delta_norms[key].items():
                    hidden_values_per_layer[layer].append(val)
            for layer in sorted(hidden_values_per_layer.keys()):
                hidden = hidden_values_per_layer[layer]
                if len(hidden) != len(behavior_values_per_layer):
                    continue
                out_rows.append(
                    {
                        "anchor": anchor,
                        "layer_index": layer,
                        "layer_bucket": layer_bucket(layer, layer_count or 0),
                        "delta_type": delta_type,
                        "behavior_delta_name": behavior_delta_name,
                        "behavior_transform": transform_kind,
                        "n_families": len(hidden),
                        "pearson_correlation": pearson_correlation(hidden, behavior_values_per_layer),
                        "spearman_correlation": spearman_correlation(hidden, behavior_values_per_layer),
                        "mean_hidden_metric": mean(hidden),
                        "std_hidden_metric": stddev(hidden),
                    }
                )
    return out_rows


def collect_family_tensor(
    rows: Sequence[Mapping[str, Any]],
    behavior_rows: Sequence[Mapping[str, Any]],
    behavior_deltas: Dict[str, Dict[str, float]],
    anchor: str,
    feature_mode: str,
    conditions: Sequence[str],
    target_condition_for_label: Optional[str] = None,
) -> Tuple[List[Tuple[str, str, str]], np.ndarray, np.ndarray, np.ndarray, int]:
    behavior_by_id = {str(r["prompt_id"]): r for r in behavior_rows}
    families_sorted = sorted({str(r["family_id"]) for r in rows})
    examples: List[Tuple[str, str, str]] = []
    tensor_list: List[np.ndarray] = []
    labels_list: List[int] = []
    groups_list: List[str] = []
    layer_count: Optional[int] = None

    if feature_mode == "neutral_only":
        vectors_by_fam: Dict[str, np.ndarray] = {}
        for r in rows:
            if str(r["condition"]) != NEUTRAL_CONDITION:
                continue
            fam = str(r["family_id"])
            beh = behavior_by_id.get(str(r["prompt_id"]))
            if not beh:
                continue
            p = Path(str(beh["activation_path_abs"]))
            if not p.exists():
                continue
            vec = load_activation_by_anchor(p)[anchor]
            vectors_by_fam[fam] = vec
            if layer_count is None:
                layer_count = int(vec.shape[0])
        target_cond = target_condition_for_label or conditions[0]
        for fam in families_sorted:
            if fam not in vectors_by_fam:
                continue
            delta_name = f"delta_{target_cond}"
            delta_value = behavior_deltas.get(fam, {}).get(delta_name)
            label = label_primary(delta_value)
            if label is None:
                continue
            examples.append((fam, target_cond, feature_mode))
            tensor_list.append(vectors_by_fam[fam].astype(np.float64))
            labels_list.append(int(label))
            groups_list.append(fam)
    elif feature_mode == "condition_minus_neutral":
        neutral_vectors: Dict[str, np.ndarray] = {}
        for r in rows:
            if str(r["condition"]) != NEUTRAL_CONDITION:
                continue
            fam = str(r["family_id"])
            beh = behavior_by_id.get(str(r["prompt_id"]))
            if not beh:
                continue
            p = Path(str(beh["activation_path_abs"]))
            if not p.exists():
                continue
            neutral_vectors[fam] = load_activation_by_anchor(p)[anchor]
        for r in rows:
            fam = str(r["family_id"])
            cond = str(r["condition"])
            if cond not in conditions:
                continue
            if fam not in neutral_vectors:
                continue
            beh = behavior_by_id.get(str(r["prompt_id"]))
            if not beh:
                continue
            cond_p = Path(str(beh["activation_path_abs"]))
            if not cond_p.exists():
                continue
            cond_vec = load_activation_by_anchor(cond_p)[anchor]
            feat = cond_vec.astype(np.float64) - neutral_vectors[fam].astype(np.float64)
            if layer_count is None:
                layer_count = int(feat.shape[0])
            delta_name = f"delta_{cond}"
            delta_value = behavior_deltas.get(fam, {}).get(delta_name)
            label = label_primary(delta_value)
            if label is None:
                continue
            examples.append((fam, cond, feature_mode))
            tensor_list.append(feat)
            labels_list.append(int(label))
            groups_list.append(fam)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    if not tensor_list:
        return examples, np.zeros((0, 0, 0), dtype=np.float64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=object), 0
    tensor = np.stack(tensor_list, axis=0)
    y = np.asarray(labels_list, dtype=np.int64)
    groups = np.asarray(groups_list, dtype=object)
    return examples, tensor, y, groups, int(layer_count or 0)


def run_family_heldout_layerwise(
    *,
    analysis: str,
    pair: str,
    anchor: str,
    examples: Sequence[Tuple[str, str, str]],
    tensor: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    layer_count: int,
    permute_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    logo = LeaveOneGroupOut()
    folds = list(logo.split(np.arange(len(examples)), y, groups=groups))
    layerwise_rows: List[Dict[str, Any]] = []
    for layer in range(layer_count):
        layer_x = np.asarray(tensor[:, layer, :], dtype=np.float64)
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        y_score_all: List[np.ndarray] = []
        baseline_all: List[np.ndarray] = []
        valid_folds: List[int] = []
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            train_x = layer_x[train_idx]
            test_x = layer_x[test_idx]
            train_y = y[train_idx].copy()
            test_y = y[test_idx]
            label_values, label_counts = np.unique(train_y, return_counts=True)
            if len(label_values) < 2 or int(np.min(label_counts)) < MIN_TRAIN_CLASS_COUNT:
                continue
            if permute_seed is not None:
                rng = np.random.default_rng(permute_seed + layer * 1009 + fold_id)
                train_y = rng.permutation(train_y)
            scaler = StandardScaler()
            train_x = stabilize_scaled_features(scaler.fit_transform(train_x))
            test_x = stabilize_scaled_features(scaler.transform(test_x))
            y_score, y_pred = centroid_probe_scores(train_x, train_y, test_x)
            baseline_pred = majority_baseline_predictions(train_y, len(test_idx))
            y_true_all.append(test_y)
            y_pred_all.append(y_pred)
            y_score_all.append(y_score)
            baseline_all.append(baseline_pred)
            valid_folds.append(fold_id)
        if not y_true_all:
            continue
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        y_score = np.concatenate(y_score_all)
        baseline_pred = np.concatenate(baseline_all)
        metrics = evaluate_predictions(y_true, y_pred, y_score, baseline_pred)
        layerwise_rows.append(
            {
                "analysis": analysis,
                "pair": pair,
                "anchor": anchor,
                "layer": layer,
                "n_examples": int(y_true.shape[0]),
                "n_families": len(valid_folds),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "baseline_balanced_accuracy": float(metrics["baseline_balanced_accuracy"]),
                "auroc": None if np.isnan(metrics["auroc"]) else float(metrics["auroc"]),
                "average_precision": None if np.isnan(metrics["average_precision"]) else float(metrics["average_precision"]),
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "confusion_matrix_counts": str(metrics["confusion_matrix_counts"]),
                "permuted": int(permute_seed is not None),
            }
        )
    return layerwise_rows


def run_cross_condition_layerwise(
    *,
    pair: str,
    anchor: str,
    source_examples: Sequence[Tuple[str, str, str]],
    source_tensor: np.ndarray,
    source_y: np.ndarray,
    source_groups: np.ndarray,
    target_examples: Sequence[Tuple[str, str, str]],
    target_tensor: np.ndarray,
    target_y: np.ndarray,
    target_groups: np.ndarray,
    layer_count: int,
    permute_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    eval_family_ids = sorted({str(g) for g in source_groups.tolist()} & {str(g) for g in target_groups.tolist()})
    layerwise_rows: List[Dict[str, Any]] = []
    for layer in range(layer_count):
        source_layer = np.asarray(source_tensor[:, layer, :], dtype=np.float64)
        target_layer = np.asarray(target_tensor[:, layer, :], dtype=np.float64)
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        y_score_all: List[np.ndarray] = []
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
            label_values, label_counts = np.unique(train_y, return_counts=True)
            if len(label_values) < 2 or int(np.min(label_counts)) < MIN_TRAIN_CLASS_COUNT:
                continue
            if permute_seed is not None:
                rng = np.random.default_rng(permute_seed + layer * 1009 + fold_id)
                train_y = rng.permutation(train_y)
            scaler = StandardScaler()
            train_x = stabilize_scaled_features(scaler.fit_transform(train_x))
            test_x = stabilize_scaled_features(scaler.transform(test_x))
            y_score, y_pred = centroid_probe_scores(train_x, train_y, test_x)
            baseline_pred = majority_baseline_predictions(train_y, int(test_mask.sum()))
            valid_families.append(family_id)
            y_true_all.append(test_y)
            y_pred_all.append(y_pred)
            y_score_all.append(y_score)
            baseline_all.append(baseline_pred)
        if not y_true_all:
            continue
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        y_score = np.concatenate(y_score_all)
        baseline_pred = np.concatenate(baseline_all)
        metrics = evaluate_predictions(y_true, y_pred, y_score, baseline_pred)
        layerwise_rows.append(
            {
                "analysis": "cross_condition",
                "pair": pair,
                "anchor": anchor,
                "layer": layer,
                "n_examples": int(y_true.shape[0]),
                "n_families": len(valid_families),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "baseline_balanced_accuracy": float(metrics["baseline_balanced_accuracy"]),
                "auroc": None if np.isnan(metrics["auroc"]) else float(metrics["auroc"]),
                "average_precision": None if np.isnan(metrics["average_precision"]) else float(metrics["average_precision"]),
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "confusion_matrix_counts": str(metrics["confusion_matrix_counts"]),
                "permuted": int(permute_seed is not None),
            }
        )
    return layerwise_rows


def run_main_analysis(
    rows: Sequence[Mapping[str, Any]],
    behavior_rows: Sequence[Mapping[str, Any]],
    behavior_deltas: Dict[str, Dict[str, float]],
    output_dir: Path,
    permutations: int,
) -> None:
    all_harmful_rows: List[Dict[str, Any]] = []
    all_cross_rows: List[Dict[str, Any]] = []
    all_perm_rows: List[Dict[str, Any]] = []

    families_sorted = sorted({str(r["family_id"]) for r in rows})

    sample_fam = families_sorted[0] if families_sorted else None
    layer_count = 0
    behavior_by_id = {str(r["prompt_id"]): r for r in behavior_rows}
    if sample_fam:
        for r in rows:
            if str(r["family_id"]) == sample_fam and str(r["condition"]) == NEUTRAL_CONDITION:
                beh = behavior_by_id.get(str(r["prompt_id"]))
                if beh and Path(str(beh["activation_path_abs"])).exists():
                    layer_count = int(load_activation_by_anchor(Path(str(beh["activation_path_abs"])))[ANCHOR_ORDER[0]].shape[0])
                break

    false_conditions = FALSE_PRESSURE_CONDITIONS
    pressure_conditions = FALSE_PRESSURE_CONDITIONS + TRUE_PRESSURE_CONDITIONS

    # Vulnerability baseline at S0/S1/S2 (one per pressure condition, neutral-only feature)
    for anchor in SHARED_ANCHORS:
        for cond in pressure_conditions:
            examples, tensor, y, groups, lc = collect_family_tensor(
                rows, behavior_rows, behavior_deltas, anchor, "neutral_only", [cond], target_condition_for_label=cond
            )
            if lc:
                layer_count = lc
            supported, counts, note = leave_one_family_out_supported(y, MIN_TRAIN_CLASS_COUNT) if len(y) > 0 else (False, {}, "no_examples")
            if not supported:
                all_harmful_rows.append(
                    build_best_row(
                        analysis="neutral_vulnerability",
                        anchor=anchor,
                        pair=cond,
                        status="unsupported",
                        support_note=note,
                        label_counts=counts,
                    )
                )
                continue
            lw_rows = run_family_heldout_layerwise(
                analysis="neutral_vulnerability",
                pair=cond,
                anchor=anchor,
                examples=examples,
                tensor=tensor,
                y=y,
                groups=groups,
                layer_count=layer_count,
            )
            best_layer, best_ba, best_baseline = best_result(lw_rows)
            all_harmful_rows.append(
                build_best_row(
                    analysis="neutral_vulnerability",
                    anchor=anchor,
                    pair=cond,
                    best_layer=best_layer,
                    best_ba=best_ba,
                    best_baseline=best_baseline,
                    label_counts=counts,
                )
            )
            all_harmful_rows.extend(lw_rows)

    # Harmful vs nonharmful pooled and within-condition at S3 and S4
    for anchor in DETECTION_ANCHORS:
        # Pooled all pressures
        examples, tensor, y, groups, lc = collect_family_tensor(
            rows, behavior_rows, behavior_deltas, anchor, "condition_minus_neutral", pressure_conditions
        )
        if lc:
            layer_count = lc
        if len(examples) > 0:
            supported, counts, note = leave_one_family_out_supported(y, MIN_TRAIN_CLASS_COUNT)
            pair = "all_pressure_pooled"
            if not supported:
                all_harmful_rows.append(
                    build_best_row("harmful_vs_nonharmful", anchor, pair, status="unsupported", support_note=note, label_counts=counts)
                )
            else:
                lw_rows = run_family_heldout_layerwise(
                    analysis="harmful_vs_nonharmful",
                    pair=pair,
                    anchor=anchor,
                    examples=examples,
                    tensor=tensor,
                    y=y,
                    groups=groups,
                    layer_count=layer_count,
                )
                best_layer, best_ba, best_baseline = best_result(lw_rows)
                all_harmful_rows.append(
                    build_best_row("harmful_vs_nonharmful", anchor, pair, best_layer=best_layer, best_ba=best_ba, best_baseline=best_baseline, label_counts=counts)
                )
                all_harmful_rows.extend(lw_rows)

                # Permutation control: strongest S3 pooled result
                if anchor == "end_of_user_message" and lw_rows:
                    observed_row = max(lw_rows, key=lambda r: float(r["balanced_accuracy"]))
                    observed_ba = float(observed_row["balanced_accuracy"])
                    observed_layer = int(observed_row["layer"])
                    permutation_best_bas: List[float] = []
                    for p_idx in range(permutations):
                        perm_lw = run_family_heldout_layerwise(
                            analysis="harmful_vs_nonharmful_perm",
                            pair=pair,
                            anchor=anchor,
                            examples=examples,
                            tensor=tensor,
                            y=y,
                            groups=groups,
                            layer_count=layer_count,
                            permute_seed=1000000 + p_idx,
                        )
                        if perm_lw:
                            best_perm = max(float(r["balanced_accuracy"]) for r in perm_lw)
                        else:
                            best_perm = 0.0
                        permutation_best_bas.append(best_perm)
                        all_perm_rows.append(
                            {
                                "analysis": "harmful_vs_nonharmful_perm_control",
                                "pair": pair,
                                "anchor": anchor,
                                "permutation_index": p_idx,
                                "max_balanced_accuracy_over_layers": best_perm,
                                "observed_balanced_accuracy": observed_ba,
                                "observed_layer": observed_layer,
                                "n_examples": observed_row["n_examples"],
                                "n_families": observed_row["n_families"],
                            }
                        )
                    ge_count = sum(1 for bp in permutation_best_bas if bp >= observed_ba)
                    p_value = (ge_count + 1) / (len(permutation_best_bas) + 1) if permutation_best_bas else 1.0
                    all_perm_rows.append(
                        {
                            "analysis": "harmful_vs_nonharmful_perm_control_summary",
                            "pair": pair,
                            "anchor": anchor,
                            "permutation_index": -1,
                            "max_balanced_accuracy_over_layers": None,
                            "observed_balanced_accuracy": observed_ba,
                            "observed_layer": observed_layer,
                            "permutation_mean": float(np.mean(permutation_best_bas)) if permutation_best_bas else None,
                            "permutation_max": float(np.max(permutation_best_bas)) if permutation_best_bas else None,
                            "permutation_min": float(np.min(permutation_best_bas)) if permutation_best_bas else None,
                            "p_value_max_over_layers": float(p_value),
                            "permutations": len(permutation_best_bas),
                        }
                    )

        # Within-condition harmful detection
        for cond in pressure_conditions + ["evidence_distractor_neutral"]:
            examples, tensor, y, groups, lc = collect_family_tensor(
                rows, behavior_rows, behavior_deltas, anchor, "condition_minus_neutral", [cond]
            )
            if lc:
                layer_count = lc
            if len(examples) == 0:
                continue
            supported, counts, note = leave_one_family_out_supported(y, MIN_TRAIN_CLASS_COUNT)
            pair = f"within_{cond}"
            if not supported:
                all_harmful_rows.append(
                    build_best_row("harmful_vs_nonharmful_within", anchor, pair, status="unsupported", support_note=note, label_counts=counts)
                )
                continue
            lw_rows = run_family_heldout_layerwise(
                analysis="harmful_vs_nonharmful_within",
                pair=pair,
                anchor=anchor,
                examples=examples,
                tensor=tensor,
                y=y,
                groups=groups,
                layer_count=layer_count,
            )
            best_layer, best_ba, best_baseline = best_result(lw_rows)
            all_harmful_rows.append(
                build_best_row("harmful_vs_nonharmful_within", anchor, pair, best_layer=best_layer, best_ba=best_ba, best_baseline=best_baseline, label_counts=counts)
            )
            all_harmful_rows.extend(lw_rows)

    # Cross-condition transfer at S3 and S4
    for anchor in DETECTION_ANCHORS:
        for src_cond, tgt_cond in CROSS_CONDITION_TRANSFER_PAIRS:
            src_ex, src_t, src_y, src_g, lc1 = collect_family_tensor(
                rows, behavior_rows, behavior_deltas, anchor, "condition_minus_neutral", [src_cond]
            )
            tgt_ex, tgt_t, tgt_y, tgt_g, lc2 = collect_family_tensor(
                rows, behavior_rows, behavior_deltas, anchor, "condition_minus_neutral", [tgt_cond]
            )
            if lc1:
                layer_count = lc1
            if not src_ex or not tgt_ex:
                continue
            pair = f"{src_cond}__to__{tgt_cond}"
            src_sup, src_counts, src_note = source_condition_supported(src_y, MIN_TRAIN_CLASS_COUNT)
            tgt_sup, tgt_counts, tgt_note = leave_one_family_out_supported(tgt_y, MIN_TRAIN_CLASS_COUNT)
            if not src_sup or not tgt_sup:
                all_cross_rows.append(
                    {
                        "analysis": "cross_condition_best",
                        "pair": pair,
                        "anchor": anchor,
                        "status": "unsupported",
                        "support_note": f"src={src_note}; tgt={tgt_note}",
                        "source_label_counts": str(src_counts),
                        "target_label_counts": str(tgt_counts),
                    }
                )
                continue
            lw_rows = run_cross_condition_layerwise(
                pair=pair,
                anchor=anchor,
                source_examples=src_ex,
                source_tensor=src_t,
                source_y=src_y,
                source_groups=src_g,
                target_examples=tgt_ex,
                target_tensor=tgt_t,
                target_y=tgt_y,
                target_groups=tgt_g,
                layer_count=layer_count,
            )
            if not lw_rows:
                all_cross_rows.append(
                    {
                        "analysis": "cross_condition_best",
                        "pair": pair,
                        "anchor": anchor,
                        "status": "unsupported",
                        "support_note": "no_valid_folds",
                        "source_label_counts": str(src_counts),
                        "target_label_counts": str(tgt_counts),
                    }
                )
                continue
            best_row = max(lw_rows, key=lambda r: float(r["balanced_accuracy"]))
            all_cross_rows.append(
                {
                    "analysis": "cross_condition_best",
                    "pair": pair,
                    "anchor": anchor,
                    "best_layer": int(best_row["layer"]),
                    "best_balanced_accuracy": float(best_row["balanced_accuracy"]),
                    "best_baseline_balanced_accuracy": float(best_row["baseline_balanced_accuracy"]),
                    "best_auroc": best_row["auroc"],
                    "best_average_precision": best_row["average_precision"],
                    "n_examples": int(best_row["n_examples"]),
                    "n_families": int(best_row["n_families"]),
                    "status": "ok",
                    "source_label_counts": str(src_counts),
                    "target_label_counts": str(tgt_counts),
                }
            )
            all_cross_rows.extend(lw_rows)

    write_csv(output_dir / "harmful_vs_nonharmful_probe_results.csv", all_harmful_rows)
    write_csv(output_dir / "cross_condition_probe_results.csv", all_cross_rows)
    write_csv(output_dir / "permutation_control_results.csv", all_perm_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dataset", default=DEFAULT_PROMPT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--activation-root", default=DEFAULT_ACTIVATION_ROOT)
    parser.add_argument("--behavior-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", default=os.environ.get("QWEN_DEVICE", ""))
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "model_cache"))
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--max-families", type=int, default=-1)
    args = parser.parse_args()

    prompt_path = resolve_required_path(args.prompt_dataset, "HQ80 prompt dataset", "Push the dataset first if missing.")
    rows = read_jsonl(prompt_path)
    if args.max_families > 0:
        selected_fams = sorted({str(r["family_id"]) for r in rows})[: args.max_families]
        rows = [r for r in rows if str(r["family_id"]) in selected_fams]

    validate_dataset_structure(rows)

    activation_root = Path(args.activation_root)
    if not activation_root.is_absolute():
        activation_root = (REPO_ROOT / activation_root).resolve()
    activation_root.mkdir(parents=True, exist_ok=True)

    behavior_jsonl_path = Path(args.behavior_jsonl)
    if not behavior_jsonl_path.is_absolute():
        behavior_jsonl_path = (REPO_ROOT / behavior_jsonl_path).resolve()
    behavior_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    output_rows: List[Dict[str, Any]] = []
    if behavior_jsonl_path.exists() and not args.force_reextract:
        existing = read_jsonl(behavior_jsonl_path)
        existing_by_id = {str(r["prompt_id"]): r for r in existing}
    else:
        existing_by_id = {}
    remaining = [r for r in rows if str(r["prompt_id"]) not in existing_by_id]

    if not args.skip_extraction and remaining:
        resolved_device = pick_device(args.device)
        inference_dtype = torch.float16 if resolved_device in ("mps", "cuda") else torch.float32
        model, tokenizer = load_local_model(
            args.model,
            device=resolved_device,
            dtype=inference_dtype,
            cache_dir=args.cache_dir,
            trust_remote_code=False,
        )
        token_id_a, token_id_b, token_strategy = choose_answer_token_ids(tokenizer)
        total = len(rows)
        print(json.dumps({"stage": "extraction", "total": total, "already_completed": len(existing_by_id), "remaining": len(remaining), "device": resolved_device, "model": args.model, "activation_root": str(activation_root), "behavior_jsonl": str(behavior_jsonl_path)}, ensure_ascii=False))
        for index, r in enumerate(remaining, start=1):
            t0 = time.perf_counter()
            prompt_id = str(r["prompt_id"])
            family_id = str(r["family_id"])
            condition = str(r["condition"])
            prompt_text = str(r["prompt_text"])
            anchor_meta = find_anchor_metadata(prompt_text, tokenizer)
            forward = run_forward_multi_position(model, tokenizer, prompt_text, resolved_device, anchor_meta["token_positions"])
            logits = np.asarray(forward["logits_last_token"], dtype=np.float32)
            logit_a = float(logits[token_id_a])
            logit_b = float(logits[token_id_b])
            pair = np.asarray([logit_a, logit_b], dtype=np.float64)
            pair = pair - float(np.max(pair))
            exp = np.exp(pair)
            probs = exp / float(np.sum(exp))
            prob_a = float(probs[0])
            prob_b = float(probs[1])
            correct_choice = str(r["correct_choice"])
            false_choice = str(r["false_choice"])
            correct_logit = logit_a if correct_choice == "A" else logit_b
            false_logit = logit_a if false_choice == "A" else logit_b
            logit_margin = float(correct_logit - false_logit)
            model_choice = compute_model_choice(logit_a, logit_b)
            family_dir = activation_root / family_id
            family_dir.mkdir(parents=True, exist_ok=True)
            activation_path = family_dir / f"{prompt_id}.pt"
            activation_record = {
                "prompt_id": prompt_id,
                "family_id": family_id,
                "prompt_type": condition,
                "condition": condition,
                "anchor_positions": {k: int(v) for k, v in anchor_meta["token_positions"].items()},
                "token_strategy": token_strategy,
                "token_seq_len": int(forward["token_seq_len"]),
                "logits_last_token": torch.from_numpy(logits),
                "logit_A": logit_a,
                "logit_B": logit_b,
                "prob_A": prob_a,
                "prob_B": prob_b,
                "logit_margin": logit_margin,
                "model_choice": model_choice,
                "correct_choice": correct_choice,
                "false_choice": false_choice,
                "hidden_states_by_anchor": {
                    anchor: torch.from_numpy(forward["hidden_states_by_anchor"][anchor]) for anchor in ANCHOR_ORDER
                },
                "prompt_text": prompt_text,
                "model_name": args.model,
            }
            torch.save(activation_record, activation_path)
            rel_act = activation_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            output_row = {
                "prompt_id": prompt_id,
                "family_id": family_id,
                "domain": r.get("domain"),
                "title": r.get("title"),
                "source_set": r.get("source_set"),
                "prompt_type": condition,
                "condition": condition,
                "correct_choice": correct_choice,
                "false_choice": false_choice,
                "logit_A": logit_a,
                "logit_B": logit_b,
                "prob_A": prob_a,
                "prob_B": prob_b,
                "logit_margin": logit_margin,
                "model_choice": model_choice,
                "is_correct": None if model_choice == "tie" else model_choice == correct_choice,
                "pressure_target_choice": r.get("pressure_target_choice"),
                "is_pressure": bool(r.get("is_pressure")),
                "is_false_pressure": bool(r.get("is_false_pressure")),
                "is_true_pressure": bool(r.get("is_true_pressure")),
                "is_distractor": bool(r.get("is_distractor")),
                "token_strategy": token_strategy,
                "token_seq_len": int(forward["token_seq_len"]),
                "anchor_positions": json.dumps({k: int(v) for k, v in anchor_meta["token_positions"].items()}),
                "activation_path": rel_act,
                "activation_path_abs": str(activation_path.resolve()),
                "answer_logit_prompt": prompt_text,
                "model_name": args.model,
                "extraction_position": "multi_anchor_matched_prefix",
            }
            existing_by_id[prompt_id] = output_row
            ordered = [existing_by_id[str(rr["prompt_id"])] for rr in rows if str(rr["prompt_id"]) in existing_by_id]
            write_jsonl_atomic(behavior_jsonl_path, ordered)
            elapsed = time.perf_counter() - t0
            print(json.dumps({"index": index, "total_remaining": len(remaining), "prompt_id": prompt_id, "model_choice": model_choice, "logit_margin": logit_margin, "elapsed_ms": round(1000.0 * elapsed, 2)}, ensure_ascii=False))

    behavior_rows = read_jsonl(behavior_jsonl_path) if behavior_jsonl_path.exists() else []
    delta_rows, behavior_summary_rows, flip_rows, behavior_deltas = build_behavior_outputs(
        rows,
        behavior_jsonl_path,
        activation_root,
    )
    write_csv(results_dir / "family_margin_deltas.csv", delta_rows)
    write_csv(results_dir / "behavior_summary_by_condition.csv", behavior_summary_rows)
    write_csv(results_dir / "answer_flip_report.csv", flip_rows)
    shutil_copy = None  # placeholder
    # copy behavior JSONL pointer location name requested
    behavior_jsonl_out = results_dir / "behavior_logits.jsonl"
    if behavior_jsonl_path.exists():
        behavior_jsonl_out.write_text(behavior_jsonl_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary_lines: List[str] = []
    # Integrity checks + examples
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=False)
    except Exception:
        tokenizer = None
    if tokenizer is not None:
        window_examples = validate_anchors_and_integrity(rows, tokenizer, activation_root, behavior_rows, summary_lines)
        write_csv(results_dir / "anchor_window_examples.csv", window_examples)
    else:
        summary_lines.append("WARNING: could not load tokenizer; skipped anchor integrity checks.")
    behavior_summary_text = results_dir / "hq80_behavior_summary.txt"
    behavior_summary_lines: List[str] = []
    behavior_summary_lines.append(f"HQ80 Behavior Summary — experiment {EXPERIMENT_NAME}")
    behavior_summary_lines.append(f"model = {args.model}")
    behavior_summary_lines.append(f"n_families = {len(delta_rows)}")
    behavior_summary_lines.append("")
    behavior_summary_lines.append("Per-condition matched-family margin delta stats:")
    for r in behavior_summary_rows:
        behavior_summary_lines.append(
            f"  {r['condition']}: mean={r['mean_delta_margin']:.4f}, median={r['median_delta_margin']:.4f}, "
            f"neg={r['n_negative_deltas']}, pos={r['n_positive_deltas']}, zero={r['n_zero_deltas']}, "
            f"flips={r['n_answer_flips']}, sycophantic_candidates={r['n_sycophantic_override_candidates']}"
        )
    behavior_summary_text.write_text("\n".join(behavior_summary_lines) + "\n", encoding="utf-8")
    (results_dir / "anchor_integrity_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Representation tables
    print(json.dumps({"stage": "representation_tables", "results_dir": str(results_dir)}, ensure_ascii=False))
    norm_rows = build_layerwise_delta_norms(rows, activation_root, behavior_rows)
    write_csv(results_dir / "layerwise_delta_norms.csv", norm_rows)
    cosine_rows = build_layerwise_delta_cosines(rows, activation_root, behavior_rows)
    write_csv(results_dir / "layerwise_delta_cosines.csv", cosine_rows)
    corr_rows = build_hidden_behavior_correlations(rows, activation_root, behavior_rows, behavior_deltas)
    write_csv(results_dir / "hidden_behavior_correlations.csv", corr_rows)

    # Probe CSVs: harmful-vs-nonharmful, cross-condition, permutation control
    all_activation_paths_exist = all(
        Path(str(behavior_by_id_candidate["activation_path_abs"])).exists()
        for behavior_by_id_candidate in (read_jsonl(behavior_jsonl_path) if behavior_jsonl_path.exists() else [])
    )
    if all_activation_paths_exist:
        print(json.dumps({"stage": "probe_analysis", "results_dir": str(results_dir), "permutations": args.permutations}, ensure_ascii=False))
        run_main_analysis(rows, behavior_rows, behavior_deltas, results_dir, permutations=args.permutations)
    else:
        print(json.dumps({"stage": "probe_analysis_skipped", "reason": "not_all_activation_files_exist_yet", "results_dir": str(results_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
