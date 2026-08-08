import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROMPT_DATASET = "data/generated_prompts_v1.jsonl"
DEFAULT_INPUT = "outputs/state_logits_qwen3_4b_instruct_2507_all_families.jsonl"
DEFAULT_FAMILY_DELTAS = "results/qwen3_4b_instruct_2507_family36_family_margin_deltas.csv"
DEFAULT_ACTIVATION_ROOT = "activations/qwen3_4b_instruct_2507_early_positions"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_SUMMARY_OUTPUT = "results/probe6_anchor_integrity_summary.txt"
DEFAULT_EXAMPLES_OUTPUT = "results/probe6_anchor_integrity_examples.txt"
DEFAULT_BASELINE_OUTPUT = "results/probe6_e0_vulnerability_baseline.csv"
DEFAULT_EXAMPLES_PER_CONDITION = 3

NEUTRAL_CONDITION = "evidence_neutral"
CONDITIONS = [
    "evidence_false_belief_pressure",
    "evidence_emotional_pressure",
    "closed_context_false_belief_pressure",
]
ANCHOR_NAME = "end_of_evidence_block"
ANCHOR_LABEL = "E0: end of evidence block"
CONDITION_DELTA_COLUMN_MAP: Dict[str, str] = {
    "evidence_false_belief_pressure": "delta_false_pressure",
    "evidence_emotional_pressure": "delta_emotional_pressure",
    "closed_context_false_belief_pressure": "delta_closed_context",
}
BASELINE_MIN_TRAIN_CLASS_COUNT = 1


def resolve_required_path(path_str: str, purpose: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    return path


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


def read_family_deltas(path: Path) -> Dict[Tuple[str, str], Optional[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: Dict[Tuple[str, str], Optional[float]] = {}
    for row in rows:
        family_id = str(row.get("family_id", "")).strip()
        if not family_id:
            continue
        for condition, column_name in CONDITION_DELTA_COLUMN_MAP.items():
            value = row.get(column_name)
            if value in ("", None):
                out[(family_id, condition)] = None
            else:
                out[(family_id, condition)] = float(value)
    return out


def label_primary(delta_margin: Optional[float]) -> Optional[int]:
    if delta_margin is None:
        return None
    return 1 if delta_margin < 0.0 else 0


def load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    cache_dir = REPO_ROOT / "model_cache"
    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )


def _min_positive(values: Sequence[int]) -> int:
    positives = [value for value in values if value >= 0]
    return min(positives) if positives else -1


def extract_base_text(prompt_text: str) -> str:
    marker_exact = "\n\nAnswer with exactly this format:"
    marker_short = "\n\nAnswer with only A or B.\n\nANSWER:"
    if marker_exact in prompt_text:
        return prompt_text[: prompt_text.rfind(marker_exact)].rstrip()
    if marker_short in prompt_text:
        return prompt_text[: prompt_text.rfind(marker_short)].rstrip()
    return prompt_text.rstrip()


def build_answer_prompt(base_text: str) -> str:
    return base_text + "\n\nAnswer with only A or B.\n\nANSWER:"


def find_anchor_metadata(prompt_text: str, tokenizer: Any) -> Dict[str, Any]:
    base_text = extract_base_text(prompt_text)

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

    end_question_pos = -1
    question_idx = base_text.rfind("\nQuestion:")
    if question_idx >= 0:
        next_idx = _min_positive(
            [
                base_text.find("\nChoices:", question_idx + len("\nQuestion:")),
                base_text.find("\n\nAnswer with exactly this format:", question_idx + len("\nQuestion:")),
                base_text.find("\n\nAnswer with only A or B.", question_idx + len("\nQuestion:")),
            ]
        )
        if next_idx < 0:
            next_idx = len(base_text)
        end_question_pos = next_idx - 1

    end_choices_pos = -1
    choices_idx = base_text.rfind("\nChoices:")
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

    answer_prompt = build_answer_prompt(base_text)
    encoded = tokenizer(
        answer_prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="np",
    )
    token_ids = np.asarray(encoded["input_ids"][0], dtype=np.int64)
    offsets = np.asarray(encoded["offset_mapping"][0], dtype=np.int64)
    token_seq_len = int(token_ids.shape[0])

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

    char_positions = {
        "end_of_evidence_block": end_evidence_pos,
        "end_of_user_pressure_sentence": end_prefix_pos,
        "end_of_question": end_question_pos,
        "end_of_answer_choices": end_choices_pos,
        "final_answer_position": len(answer_prompt) - 1,
    }
    token_positions = {name: char_to_token(pos) for name, pos in char_positions.items() if name != "final_answer_position"}
    token_positions["final_answer_position"] = token_seq_len - 1

    return {
        "base_text": base_text,
        "answer_prompt": answer_prompt,
        "token_ids": token_ids,
        "char_positions": char_positions,
        "token_positions": token_positions,
        "token_offsets": offsets,
    }


def load_e0_tensor(path: Path) -> np.ndarray:
    record = torch.load(path, map_location="cpu")
    tensor = record["hidden_states_by_anchor"][ANCHOR_NAME]
    if isinstance(tensor, torch.Tensor):
        return tensor.to(dtype=torch.float32).numpy()
    return np.asarray(tensor, dtype=np.float32)


def exact_prefix_match_rate(flags: Sequence[bool]) -> float:
    if not flags:
        return float("nan")
    return float(sum(bool(flag) for flag in flags)) / float(len(flags))


def compute_delta_norms(
    activation_root: Path,
    families: Sequence[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        per_family: List[np.ndarray] = []
        for family_id in families:
            neutral_path = activation_root / family_id / f"{family_id}_{NEUTRAL_CONDITION}.pt"
            condition_path = activation_root / family_id / f"{family_id}_{condition}.pt"
            neutral_e0 = load_e0_tensor(neutral_path)
            condition_e0 = load_e0_tensor(condition_path)
            per_family.append(np.linalg.norm(condition_e0 - neutral_e0, axis=1))
        out[condition] = np.stack(per_family, axis=0)
    return out


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) >= 2 else float("nan")


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score)) if len(set(y_true.tolist())) >= 2 else float("nan")


def majority_baseline_predictions(train_y: np.ndarray, n_test: int) -> np.ndarray:
    counts = np.bincount(train_y.astype(np.int64))
    majority = int(np.argmax(counts))
    return np.full(n_test, majority, dtype=np.int64)


def centroid_probe_scores(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    class0 = train_x[train_y == 0]
    class1 = train_x[train_y == 1]
    if class0.size == 0 or class1.size == 0:
        raise ValueError("Both classes are required.")
    centroid0 = class0.mean(axis=0)
    centroid1 = class1.mean(axis=0)
    weight = np.asarray(centroid1 - centroid0, dtype=np.float64)
    bias = -0.5 * (
        float(np.dot(centroid1.astype(np.float64), centroid1.astype(np.float64)))
        - float(np.dot(centroid0.astype(np.float64), centroid0.astype(np.float64)))
    )
    scores = np.sum(test_x * weight[np.newaxis, :], axis=1, dtype=np.float64) + bias
    preds = (scores >= 0.0).astype(np.int64)
    return scores, preds


def run_vulnerability_baseline(
    neutral_tensor: np.ndarray,
    labels_by_condition: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_layers = int(neutral_tensor.shape[1])
    loo = LeaveOneOut()

    for condition, labels in labels_by_condition.items():
        labels = np.asarray(labels, dtype=np.int64)
        label_values, label_counts = np.unique(labels, return_counts=True)
        totals = {int(v): int(c) for v, c in zip(label_values.tolist(), label_counts.tolist())}
        condition_supported = len(label_values) >= 2 and int(np.min(label_counts)) >= BASELINE_MIN_TRAIN_CLASS_COUNT + 1

        for layer in range(n_layers):
            status = "ok" if condition_supported else "unsupported"
            support_note = ""
            if not condition_supported:
                support_note = (
                    f"class_counts=0:{totals.get(0, 0)},1:{totals.get(1, 0)}; need at least "
                    f"{BASELINE_MIN_TRAIN_CLASS_COUNT + 1} total examples per class for leave-one-out training"
                )
                rows.append(
                    {
                        "condition": condition,
                        "layer": layer,
                        "n_examples": int(labels.shape[0]),
                        "n_label_0": totals.get(0, 0),
                        "n_label_1": totals.get(1, 0),
                        "status": status,
                        "support_note": support_note,
                        "balanced_accuracy": "",
                        "baseline_balanced_accuracy": "",
                        "auroc": "",
                        "average_precision": "",
                    }
                )
                continue

            layer_x = np.asarray(neutral_tensor[:, layer, :], dtype=np.float64)
            y_true_all: List[np.ndarray] = []
            y_pred_all: List[np.ndarray] = []
            y_score_all: List[np.ndarray] = []
            baseline_all: List[np.ndarray] = []

            for train_idx, test_idx in loo.split(layer_x):
                train_x = layer_x[train_idx]
                test_x = layer_x[test_idx]
                train_y = labels[train_idx]
                test_y = labels[test_idx]
                train_values, train_counts = np.unique(train_y, return_counts=True)
                if len(train_values) < 2 or int(np.min(train_counts)) < BASELINE_MIN_TRAIN_CLASS_COUNT:
                    status = "unsupported"
                    support_note = (
                        f"train fold dropped below min_train_class_count={BASELINE_MIN_TRAIN_CLASS_COUNT}"
                    )
                    break
                scaler = StandardScaler()
                train_x = scaler.fit_transform(train_x)
                test_x = scaler.transform(test_x)
                scores, preds = centroid_probe_scores(train_x, train_y, test_x)
                y_true_all.append(test_y)
                y_pred_all.append(preds)
                y_score_all.append(scores)
                baseline_all.append(majority_baseline_predictions(train_y, len(test_y)))

            if status != "ok":
                rows.append(
                    {
                        "condition": condition,
                        "layer": layer,
                        "n_examples": int(labels.shape[0]),
                        "n_label_0": totals.get(0, 0),
                        "n_label_1": totals.get(1, 0),
                        "status": status,
                        "support_note": support_note,
                        "balanced_accuracy": "",
                        "baseline_balanced_accuracy": "",
                        "auroc": "",
                        "average_precision": "",
                    }
                )
                continue

            y_true = np.concatenate(y_true_all)
            y_pred = np.concatenate(y_pred_all)
            y_score = np.concatenate(y_score_all)
            baseline_pred = np.concatenate(baseline_all)
            rows.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "n_examples": int(labels.shape[0]),
                    "n_label_0": totals.get(0, 0),
                    "n_label_1": totals.get(1, 0),
                    "status": "ok",
                    "support_note": "",
                    "balanced_accuracy": f"{balanced_accuracy_score(y_true, y_pred):.6f}",
                    "baseline_balanced_accuracy": f"{balanced_accuracy_score(y_true, baseline_pred):.6f}",
                    "auroc": "" if np.isnan(safe_auroc(y_true, y_score)) else f"{safe_auroc(y_true, y_score):.6f}",
                    "average_precision": ""
                    if np.isnan(safe_average_precision(y_true, y_score))
                    else f"{safe_average_precision(y_true, y_score):.6f}",
                }
            )
    return rows


def format_token_window(tokenizer: Any, token_ids: np.ndarray, anchor_idx: int, radius: int = 20) -> str:
    lo = max(0, anchor_idx - radius)
    hi = min(len(token_ids), anchor_idx + radius + 1)
    pieces: List[str] = []
    for idx in range(lo, hi):
        token_id = int(token_ids[idx])
        try:
            token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
        except Exception:
            token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        marker = " <ANCHOR>" if idx == anchor_idx else ""
        pieces.append(f"{idx:>4d}: {token_id:>6d} {repr(token_text)}{marker}")
    return "\n".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dataset", default=str(REPO_ROOT / DEFAULT_PROMPT_DATASET))
    parser.add_argument("--input", default=str(REPO_ROOT / DEFAULT_INPUT))
    parser.add_argument("--family-deltas", default=str(REPO_ROOT / DEFAULT_FAMILY_DELTAS))
    parser.add_argument("--activation-root", default=str(REPO_ROOT / DEFAULT_ACTIVATION_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-summary", default=str(REPO_ROOT / DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--output-examples", default=str(REPO_ROOT / DEFAULT_EXAMPLES_OUTPUT))
    parser.add_argument("--output-baseline", default=str(REPO_ROOT / DEFAULT_BASELINE_OUTPUT))
    parser.add_argument("--examples-per-condition", type=int, default=DEFAULT_EXAMPLES_PER_CONDITION)
    args = parser.parse_args()

    prompt_dataset_path = resolve_required_path(args.prompt_dataset, "prompt dataset")
    input_path = resolve_required_path(args.input, "Probe 6 input jsonl")
    family_deltas_path = resolve_required_path(args.family_deltas, "family delta csv")
    activation_root = resolve_required_path(args.activation_root, "early-position activation root")

    prompts = read_jsonl(prompt_dataset_path)
    outputs = read_jsonl(input_path)
    family_deltas = read_family_deltas(family_deltas_path)
    tokenizer = load_tokenizer(args.model)

    prompt_lookup = {
        (str(row["family_id"]), str(row["prompt_type"])): row
        for row in prompts
        if "family_id" in row and "prompt_type" in row
    }
    output_lookup = {
        (str(row["family_id"]), str(row["prompt_type"])): row
        for row in outputs
        if "family_id" in row and "prompt_type" in row
    }
    required_keys = {
        (family_id, condition)
        for (family_id, condition), delta_margin in family_deltas.items()
        if condition in CONDITIONS and delta_margin is not None
    }
    required_keys.update((family_id, NEUTRAL_CONDITION) for family_id, _ in list(required_keys))
    missing_prompt_keys = sorted(key for key in required_keys if key not in prompt_lookup)
    missing_output_keys = sorted(key for key in required_keys if key not in output_lookup)
    if missing_prompt_keys:
        raise KeyError(f"Missing prompt rows for required keys: {missing_prompt_keys[:5]}")
    if missing_output_keys:
        raise KeyError(f"Missing output rows for required keys: {missing_output_keys[:5]}")
    families = sorted({family_id for family_id, _ in required_keys})

    prefix_rows: List[Dict[str, Any]] = []
    token_rows: List[Dict[str, Any]] = []
    condition_example_sections: List[str] = []
    neutral_e0_per_family: List[np.ndarray] = []
    vulnerability_labels: Dict[str, List[int]] = {condition: [] for condition in CONDITIONS}

    for condition in CONDITIONS:
        examples_added = 0
        condition_example_sections.append(f"## {condition}")
        condition_example_sections.append("")
        for family_id in families:
            neutral_prompt = prompt_lookup[(family_id, NEUTRAL_CONDITION)]
            condition_prompt = prompt_lookup[(family_id, condition)]
            neutral_output = output_lookup[(family_id, NEUTRAL_CONDITION)]
            condition_output = output_lookup[(family_id, condition)]

            neutral_meta = find_anchor_metadata(str(neutral_prompt["prompt"]), tokenizer)
            condition_meta = find_anchor_metadata(str(condition_prompt["prompt"]), tokenizer)

            neutral_e0_char = int(neutral_meta["char_positions"][ANCHOR_NAME])
            condition_e0_char = int(condition_meta["char_positions"][ANCHOR_NAME])
            neutral_prefix = neutral_meta["base_text"][: neutral_e0_char + 1]
            condition_prefix = condition_meta["base_text"][: condition_e0_char + 1]
            raw_identical = neutral_prefix == condition_prefix
            prefix_rows.append(
                {
                    "family_id": family_id,
                    "condition": condition,
                    "neutral_prefix_len": len(neutral_prefix),
                    "condition_prefix_len": len(condition_prefix),
                    "raw_prefix_identical": raw_identical,
                }
            )

            neutral_answer_meta = find_anchor_metadata(str(neutral_output["answer_logit_prompt"]), tokenizer)
            condition_answer_meta = find_anchor_metadata(str(condition_output["answer_logit_prompt"]), tokenizer)
            neutral_e0_token = int(neutral_answer_meta["token_positions"][ANCHOR_NAME])
            condition_e0_token = int(condition_answer_meta["token_positions"][ANCHOR_NAME])
            neutral_token_prefix = neutral_answer_meta["token_ids"][: neutral_e0_token + 1]
            condition_token_prefix = condition_answer_meta["token_ids"][: condition_e0_token + 1]
            token_identical = bool(np.array_equal(neutral_token_prefix, condition_token_prefix))
            token_rows.append(
                {
                    "family_id": family_id,
                    "condition": condition,
                    "neutral_e0_token_index": neutral_e0_token,
                    "condition_e0_token_index": condition_e0_token,
                    "neutral_prefix_tokens": int(len(neutral_token_prefix)),
                    "condition_prefix_tokens": int(len(condition_token_prefix)),
                    "token_prefix_identical": token_identical,
                }
            )

            if examples_added < args.examples_per_condition:
                anchor_idx = condition_e0_token
                token_ids = condition_answer_meta["token_ids"]
                token_id = int(token_ids[anchor_idx])
                try:
                    token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
                except Exception:
                    token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                condition_example_sections.extend(
                    [
                        f"### {family_id}",
                        f"- anchor name: {ANCHOR_LABEL}",
                        f"- anchor token index: {anchor_idx}",
                        f"- anchor token id: {token_id}",
                        f"- anchor token text: {repr(token_text)}",
                        f"- raw prefix identical to neutral: {raw_identical}",
                        f"- token prefix identical to neutral: {token_identical}",
                        "- 20-token window:",
                        format_token_window(tokenizer, token_ids, anchor_idx),
                        "",
                    ]
                )
                examples_added += 1

    for family_id in families:
        neutral_path = activation_root / family_id / f"{family_id}_{NEUTRAL_CONDITION}.pt"
        neutral_e0_per_family.append(load_e0_tensor(neutral_path))
        for condition in CONDITIONS:
            delta_margin = family_deltas[(family_id, condition)]
            label = label_primary(delta_margin)
            if label is None:
                raise RuntimeError(f"Missing harmfulness label for {(family_id, condition)}")
            vulnerability_labels[condition].append(label)

    neutral_tensor = np.stack(neutral_e0_per_family, axis=0)
    baseline_rows = run_vulnerability_baseline(
        neutral_tensor=neutral_tensor,
        labels_by_condition={k: np.asarray(v, dtype=np.int64) for k, v in vulnerability_labels.items()},
    )

    delta_norms = compute_delta_norms(activation_root, families)

    output_summary = Path(args.output_summary)
    output_examples = Path(args.output_examples)
    output_baseline = Path(args.output_baseline)
    if not output_summary.is_absolute():
        output_summary = (REPO_ROOT / output_summary).resolve()
    if not output_examples.is_absolute():
        output_examples = (REPO_ROOT / output_examples).resolve()
    if not output_baseline.is_absolute():
        output_baseline = (REPO_ROOT / output_baseline).resolve()
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    with output_baseline.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(baseline_rows[0].keys()) if baseline_rows else [
            "condition",
            "layer",
            "n_examples",
            "n_label_0",
            "n_label_1",
            "status",
            "support_note",
            "balanced_accuracy",
            "baseline_balanced_accuracy",
            "auroc",
            "average_precision",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(baseline_rows)

    summary_lines: List[str] = []
    summary_lines.append("Probe 6 anchor integrity checks")
    summary_lines.append("")
    summary_lines.append(f"Anchor under test: {ANCHOR_LABEL}")
    summary_lines.append(f"Families: {len(families)}")
    summary_lines.append(f"Conditions tested: {', '.join(CONDITIONS)}")
    summary_lines.append("")

    summary_lines.append("== 1. Prefix identity check ==")
    for condition in CONDITIONS:
        rows = [row for row in prefix_rows if row["condition"] == condition]
        identical = [bool(row["raw_prefix_identical"]) for row in rows]
        summary_lines.append(
            f"- {condition}: {sum(identical)}/{len(identical)} raw prefixes identical to neutral "
            f"({100.0 * exact_prefix_match_rate(identical):.1f}%)"
        )
    summary_lines.append("")

    summary_lines.append("== 2. Token prefix identity check ==")
    for condition in CONDITIONS:
        rows = [row for row in token_rows if row["condition"] == condition]
        identical = [bool(row["token_prefix_identical"]) for row in rows]
        summary_lines.append(
            f"- {condition}: {sum(identical)}/{len(identical)} token prefixes identical through E0 "
            f"({100.0 * exact_prefix_match_rate(identical):.1f}%)"
        )
    summary_lines.append("")

    summary_lines.append("== 3. E0 delta norm check ==")
    for condition in CONDITIONS:
        norms = delta_norms[condition]
        mean_by_layer = norms.mean(axis=0)
        median_by_layer = np.median(norms, axis=0)
        max_by_layer = norms.max(axis=0)
        top_layers = np.argsort(-mean_by_layer)[:5]
        summary_lines.append(
            f"- {condition}: mean norm range {mean_by_layer.min():.4f}..{mean_by_layer.max():.4f}, "
            f"median range {median_by_layer.min():.4f}..{median_by_layer.max():.4f}"
        )
        summary_lines.append(
            "  Top mean-norm layers: "
            + ", ".join(
                f"L{int(layer):02d} mean={mean_by_layer[layer]:.4f} median={median_by_layer[layer]:.4f} max={max_by_layer[layer]:.4f}"
                for layer in top_layers.tolist()
            )
        )
    summary_lines.append("")

    summary_lines.append("== 4. Anchor location check ==")
    summary_lines.append(
        f"Detailed E0 token windows for {args.examples_per_condition} examples per condition are in {output_examples.relative_to(REPO_ROOT)}."
    )
    summary_lines.append("")

    summary_lines.append("== 5. Neutral-E0 vulnerability baseline ==")
    for condition in CONDITIONS:
        condition_rows = [row for row in baseline_rows if row["condition"] == condition]
        ok_rows = [row for row in condition_rows if row["status"] == "ok"]
        if not ok_rows:
            summary_lines.append(f"- {condition}: unsupported")
            if condition_rows:
                summary_lines.append(f"  {condition_rows[0]['support_note']}")
            continue
        best_row = max(ok_rows, key=lambda row: float(row["balanced_accuracy"]))
        summary_lines.append(
            f"- {condition}: best BA {float(best_row['balanced_accuracy']):.3f} at layer {int(best_row['layer'])} "
            f"(baseline BA {float(best_row['baseline_balanced_accuracy']):.3f}, "
            f"AUROC {best_row['auroc'] or 'NA'}, AP {best_row['average_precision'] or 'NA'})"
        )
        summary_lines.append(
            f"  class counts: nonharmful={best_row['n_label_0']}, harmful={best_row['n_label_1']}"
        )
    summary_lines.append("")

    summary_lines.append("== Interpretation ==")
    all_prefix_identity_zero = all(
        not bool(row["raw_prefix_identical"]) for row in prefix_rows
    )
    all_token_identity_zero = all(
        not bool(row["token_prefix_identical"]) for row in token_rows
    )
    if all_prefix_identity_zero and all_token_identity_zero:
        summary_lines.append(
            "- E0 is not a shared neutral-vs-pressure prefix in this dataset: raw text and token prefixes differ for every tested family/condition pair."
        )
        summary_lines.append(
            "- That means E0 delta-based results can arise from prompt-template differences before the evidence block, not only from later pressure effects."
        )
    else:
        summary_lines.append(
            "- Some E0 prefixes are shared across neutral and pressure variants; inspect the exact-match rates above before trusting E0 effects."
        )
    vulnerability_findings = []
    for condition in CONDITIONS:
        ok_rows = [row for row in baseline_rows if row["condition"] == condition and row["status"] == "ok"]
        if not ok_rows:
            continue
        best_row = max(ok_rows, key=lambda row: float(row["balanced_accuracy"]))
        if float(best_row["balanced_accuracy"]) > float(best_row["baseline_balanced_accuracy"]):
            vulnerability_findings.append(
                f"{condition} (best BA {float(best_row['balanced_accuracy']):.3f} at layer {int(best_row['layer'])})"
            )
    if vulnerability_findings:
        summary_lines.append(
            "- Neutral E0 states carry some predictive signal for later harmfulness in: " + ", ".join(vulnerability_findings) + "."
        )
        summary_lines.append(
            "- Interpret those results as family/evidence vulnerability prediction, not clean pre-pressure sycophancy detection."
        )
    else:
        summary_lines.append(
            "- Neutral E0 states did not beat the majority baseline in the supported vulnerability baselines."
        )
    summary_lines.append("")
    summary_lines.append(f"Machine-readable baseline results: {output_baseline.relative_to(REPO_ROOT)}")

    output_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    output_examples.write_text("\n".join(condition_example_sections) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "done",
                "output_summary": str(output_summary),
                "output_examples": str(output_examples),
                "output_baseline": str(output_baseline),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
