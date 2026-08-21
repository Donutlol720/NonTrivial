"""
HQ80 matched-prefix activation patching at anchors S3 (end of user message) and
S4 (final answer position) for Qwen3-4B-Instruct-2507.

Patch types:
  - rescue   : same-family neutral activation -> condition prompt
  - transfer : same-family condition activation -> neutral prompt

Controls:
  - self_patch              : patch a prompt with its own activation (sanity = 0 effect)
  - random_family_rescue    : condition prompt patched from DIFFERENT family's neutral
  - random_family_transfer  : neutral prompt patched from DIFFERENT family's condition

Output dir: results/activation_patching_qwen3_4b_hq80_s3s4/

Stages:
  0 sanity     : 2 families, 7 conditions, S3/S4, [28,32,34], rescue+transfer + self
  1 priority   : 80 families, 4 false-pressure conditions, S3/S4, [28,32,34,35], rescue+transfer
  2 controls   : add distractor, true-belief, true-rationale conditions
  3 remaining  : add layers [20,24,30]
  4 full       : all 80 families × 7 conditions × S3/S4 × 7 layers × rescue/transfer = 15,680 rows

Resumable: every CSV writer skips rows whose composite unique key already
exists in the on-disk file.  Interrupt with Ctrl+C and rerun with the same
--stage (or --full) to continue from where it left off.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.load_model import load_local_model, pick_device  # noqa: E402
from src.extraction.extract_multi_family_states_and_logits import (  # noqa: E402
    choose_answer_token_ids,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT_JSONL = REPO / "prompts" / "expanded_matched_prefix_hq80_v1.jsonl"
BEHAVIOR_CSV = REPO / "results" / "hq80_behavior" / "qwen_hq80_prompt_level_logits.csv"
ANCHOR_MANIFEST_CSV = (
    REPO / "results" / "hq80_hidden" / "qwen_hq80_anchor_manifest.csv"
)
OUT_DIR = REPO / "results" / "activation_patching_qwen3_4b_hq80_s3s4"
VAL_DIR = OUT_DIR / "validation"
SUM_DIR = OUT_DIR / "summaries"
PLOT_DIR = OUT_DIR / "plots"
CACHE_DIR = OUT_DIR / "cached_activations"

RAW_CSV = OUT_DIR / "hq80_s3s4_activation_patching_raw.csv"
RANDOM_FAMILY_CSV = OUT_DIR / "random_family_patching_raw.csv"
SELF_PATCH_CSV = VAL_DIR / "self_patch_control.csv"
UNPATCHED_REPRO_CSV = VAL_DIR / "unpatched_margin_reproduction.csv"
ANCHOR_VAL_CSV = VAL_DIR / "anchor_validation.csv"
ANCHOR_VAL_SAMPLES_MD = VAL_DIR / "anchor_validation_samples.md"
TOKEN_VAL_TXT = VAL_DIR / "token_validation.txt"
LAYER_IDX_TXT = VAL_DIR / "layer_indexing_validation.txt"
FAMILY_LEVEL_CSV = SUM_DIR / "hq80_s3s4_family_level_patch_effects.csv"

NEUTRAL = "evidence_neutral"
CONDITIONS_ORDER = [
    NEUTRAL,
    "evidence_false_belief_pressure",
    "evidence_false_rationale_pressure",
    "evidence_emotional_pressure",
    "evidence_authority_pressure",
    "evidence_true_belief_pressure",
    "evidence_true_rationale_pressure",
    "evidence_distractor_neutral",
]
FALSE_PRESSURE = [
    "evidence_false_belief_pressure",
    "evidence_false_rationale_pressure",
    "evidence_emotional_pressure",
    "evidence_authority_pressure",
]
TRUE_PRESSURE = [
    "evidence_true_belief_pressure",
    "evidence_true_rationale_pressure",
]
CONTROL_CONDS = [
    "evidence_distractor_neutral",
    *TRUE_PRESSURE,
]
ALL_NONNEUTRAL = [c for c in CONDITIONS_ORDER if c != NEUTRAL]

PATCH_LAYERS_FULL = [20, 24, 28, 30, 32, 34, 35]
PATCH_LAYERS_PRIORITY = [28, 32, 34, 35]
PATCH_LAYERS_SANITY = [28, 32, 34]
ANCHORS = ["S3", "S4"]
ANCHOR_JSON_KEY = {"S3": "end_of_user_message", "S4": "final_answer_position"}

TOLERANCE_STRICT = 1e-2
TOLERANCE_LOOSE = 5e-2

LAYER_BANDS: Dict[str, List[int]] = {
    "late_layers_L28_L35": [28, 30, 32, 34, 35],
    "late_prefinal_L28_L34": [28, 30, 32, 34],
    "all_patched_layers": PATCH_LAYERS_FULL,
}

RAW_COLUMNS = [
    "run_id",
    "prompt_id_target",
    "prompt_id_source",
    "family_id_target",
    "family_id_source",
    "condition",
    "target_condition",
    "source_condition",
    "patch_type",
    "anchor",
    "layer",
    "correct_choice",
    "false_choice",
    "answer_token_id_A",
    "answer_token_id_B",
    "original_neutral_margin",
    "original_condition_margin",
    "delta_margin",
    "original_target_margin",
    "patched_margin",
    "rescue_effect",
    "transfer_effect",
    "patch_effect",
    "expected_sign_from_delta",
    "expected_signed_rescue_effect",
    "expected_signed_transfer_effect",
    "moved_in_expected_direction",
    "target_anchor_index",
    "source_anchor_index",
    "target_token_length",
    "source_token_length",
    "dtype",
    "device",
    "model_name",
    "created_at",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_dirs() -> None:
    for d in (OUT_DIR, VAL_DIR, SUM_DIR, PLOT_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def find_transformer_block_list(model) -> Tuple[str, Any]:
    for top in ("model", "transformer", "backbone"):
        if hasattr(model, top):
            sub = getattr(model, top)
            for sub_attr in ("layers", "h", "blocks", "decoder_layers"):
                if hasattr(sub, sub_attr):
                    obj = getattr(sub, sub_attr)
                    if isinstance(obj, (list, torch.nn.ModuleList)):
                        return f"{top}.{sub_attr}", obj
    raise RuntimeError("Could not find transformer block list.")


def sign_of(x: float) -> int:
    if x > 0:
        return +1
    if x < 0:
        return -1
    return 0


def compute_margin(
    logit_A: float, logit_B: float, correct_choice: str
) -> float:
    """margin = logit(correct) - logit(false)."""
    if correct_choice == "A":
        return logit_A - logit_B
    elif correct_choice == "B":
        return logit_B - logit_A
    raise ValueError(f"unexpected correct_choice: {correct_choice!r}")


def token_window(
    tokenizer, input_ids: List[int], center: int, radius: int = 5
) -> Dict[str, Any]:
    ids = list(input_ids)
    n = len(ids)
    lo = max(0, center - radius)
    hi = min(n, center + radius + 1)
    window_ids = ids[lo:hi]
    center_in_window = center - lo
    pieces = []
    for i, tid in enumerate(window_ids):
        marker = ">>>" if i == center_in_window else "   "
        pieces.append(f"{marker}[{i+lo:4d}] {tokenizer.decode([tid])!r}")
    return {
        "center_index": center,
        "window_start": lo,
        "window_end": hi,
        "center_in_window": center_in_window,
        "window_decoded_lines": pieces,
        "decoded_joined": tokenizer.decode(window_ids),
    }


# ---------------------------------------------------------------------------
# CSV helpers with resumability
# ---------------------------------------------------------------------------
class ResumableCSVWriter:
    """Append-only CSV writer that is restart-safe.

    Before each append, the on-disk file is read (if exists) and a set of
    composite keys is kept in memory.  ``writerow`` is a no-op if the key
    already exists.  The disk file is opened in append mode for each new row,
    so crashes mid-write only leave at most one half-row (which we tolerate
    by reading the header + valid rows when re-opening).
    """

    def __init__(self, path: Path, fieldnames: List[str], key_fields: List[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        self.key_fields = list(key_fields)
        self._seen: set = set()
        self._header_written = False
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is not None:
                self._header_written = True
            for row in reader:
                try:
                    key = tuple(row[k] for k in self.key_fields)
                except KeyError:
                    continue
                self._seen.add(key)

    def _ensure_header(self) -> None:
        if self._header_written:
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writeheader()
        self._header_written = True

    def has(self, row: Dict[str, Any]) -> bool:
        key = tuple(str(row.get(k, "")) for k in self.key_fields)
        return key in self._seen

    def writerow(self, row: Dict[str, Any]) -> bool:
        """Return True if the row was newly written, False if skipped."""
        self._ensure_header()
        key = tuple(str(row.get(k, "")) for k in self.key_fields)
        if key in self._seen:
            return False
        # Write first, commit key only after successful flush to disk.
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            # Normalize: fill missing with empty string.
            norm = {fn: row.get(fn, "") for fn in self.fieldnames}
            w.writerow(norm)
            f.flush()
            os.fsync(f.fileno())
        self._seen.add(key)
        return True

    @property
    def n_written(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def load_prompt_dataset() -> Dict[str, Dict[str, Any]]:
    """Return {(family_id, condition): prompt_row}."""
    rows = read_jsonl(PROMPT_JSONL)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        fid = str(r.get("family_id"))
        cond = str(r.get("condition") or r.get("prompt_type") or "")
        out[(fid, cond)] = r
    return out


def load_behavior_reference() -> pd.DataFrame:
    df = pd.read_csv(BEHAVIOR_CSV)
    # Sanitize: ensure correct column types
    for col in ("logit_A", "logit_B", "margin", "neutral_margin", "margin_delta"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_anchor_manifest() -> pd.DataFrame:
    df = pd.read_csv(ANCHOR_MANIFEST_CSV)
    # Parse anchor_positions_json into dicts for each row.
    parsed = []
    for _, r in df.iterrows():
        j = r.get("anchor_positions_json")
        if isinstance(j, str):
            try:
                parsed.append(json.loads(j))
            except Exception:
                parsed.append({})
        elif isinstance(j, dict):
            parsed.append(j)
        else:
            parsed.append({})
    df["anchor_positions"] = parsed
    return df


def assemble_master() -> pd.DataFrame:
    """Merge prompts, behavior references, and anchor positions into one
    DataFrame keyed by (family_id, condition)."""
    prompts = load_prompt_dataset()
    behavior = load_behavior_reference()
    anchors = load_anchor_manifest()

    # Build master from behavior (it has all 640 rows with the canonical
    # prompt_id) and augment with prompt_text / anchor positions.
    rows: List[Dict[str, Any]] = []
    behavior_prompt_ids = set(behavior["prompt_id"].astype(str))
    anchor_by_pid = {
        str(r["prompt_id"]): (r.to_dict() if isinstance(r, pd.Series) else dict(r))
        for _, r in anchors.iterrows()
    }

    for _, b in behavior.iterrows():
        pid = str(b["prompt_id"])
        fid = str(b["family_id"])
        cond = str(b["condition"])
        prompt_row = prompts.get((fid, cond))
        if prompt_row is None:
            # Try fallback using pid pattern or any family match
            prompt_row = prompts.get((fid, ""), {})
            for (pfid, pc), pr in prompts.items():
                if pfid == fid:
                    prompt_row = pr
                    break
        anchor_row = anchor_by_pid.get(pid, {})
        anchor_pos = (
            anchor_row.get("anchor_positions", {})
            if isinstance(anchor_row, dict)
            else {}
        )
        rows.append({
            "prompt_id": pid,
            "family_id": fid,
            "condition": cond,
            "correct_choice": str(b.get("correct_choice", "A")),
            "false_choice": str(b.get("false_choice", "B")),
            "logit_A_ref": float(b.get("logit_A", float("nan"))),
            "logit_B_ref": float(b.get("logit_B", float("nan"))),
            "margin_ref": float(b.get("margin", float("nan"))),
            "neutral_margin_ref": float(b.get("neutral_margin", float("nan"))),
            "margin_delta_ref": float(b.get("margin_delta", float("nan"))),
            "prompt_text": (
                prompt_row.get("prompt_text")
                or prompt_row.get("answer_logit_prompt")
                or ""
            ),
            "token_seq_len_ref": (
                int(anchor_row.get("token_seq_len"))
                if (isinstance(anchor_row, dict) and anchor_row.get("token_seq_len") is not None)
                else None
            ),
            "end_of_user_message_idx": (
                int(anchor_pos["end_of_user_message"])
                if anchor_pos and anchor_pos.get("end_of_user_message") is not None
                else None
            ),
            "final_answer_position_idx": (
                int(anchor_pos["final_answer_position"])
                if anchor_pos and anchor_pos.get("final_answer_position") is not None
                else None
            ),
            "activation_path_abs": (
                anchor_row.get("activation_path_abs")
                if isinstance(anchor_row, dict)
                else None
            ),
        })
    df = pd.DataFrame(rows)
    return df


def all_families(df: pd.DataFrame) -> List[str]:
    return sorted(df["family_id"].unique().tolist())


# ---------------------------------------------------------------------------
# Validation 1: tokens
# ---------------------------------------------------------------------------
def validate_tokens(tokenizer, idA: int, idB: int, strategy: str) -> None:
    lines = []
    lines.append("Qwen HQ80 S3/S4 activation patching — answer token validation")
    lines.append("=" * 72)
    lines.append(f"timestamp           = {utc_now()}")
    lines.append(f"tokenizer source    = {MODEL_NAME}")
    lines.append(f"selection strategy  = {strategy}")
    lines.append("")
    decA = tokenizer.decode([idA])
    decB = tokenizer.decode([idB])
    lines.append(f"id_A = {idA}  decode = {decA!r}  len(tokens) = 1 (required)")
    lines.append(f"id_B = {idB}  decode = {decB!r}  len(tokens) = 1 (required)")
    # Re-verify by re-encoding the decoded strings
    encA = tokenizer.encode(decA, add_special_tokens=False)
    encB = tokenizer.encode(decB, add_special_tokens=False)
    lines.append("")
    lines.append(
        f"Round-trip re-encode ' A' : ids={tokenizer.encode(' A', add_special_tokens=False)} "
        f"(expected single id {idA})"
    )
    lines.append(
        f"Round-trip re-encode ' B' : ids={tokenizer.encode(' B', add_special_tokens=False)} "
        f"(expected single id {idB})"
    )
    leading_ok = (
        len(tokenizer.encode(" A", add_special_tokens=False)) == 1
        and len(tokenizer.encode(" B", add_special_tokens=False)) == 1
    )
    lines.append("")
    lines.append(f"leading_space single-token OK = {leading_ok}")
    lines.append(f"strategy_used = {strategy}")
    TOKEN_VAL_TXT.write_text("\n".join(lines) + "\n")
    print(f"[validate-tokens] wrote {TOKEN_VAL_TXT}")
    if not leading_ok and strategy != "leading_space":
        print(
            "[validate-tokens][warn] leading_space strategy not single-token; "
            f"using '{strategy}' instead.  Verify manually."
        )


# ---------------------------------------------------------------------------
# Validation 2: unpatched margin reproduction
# ---------------------------------------------------------------------------
def baseline_forward(
    model, tokenizer, prompt_text: str, device: str, idA: int, idB: int
) -> Tuple[float, float, int]:
    """Run unpatched forward pass.

    Returns (logit_A, logit_B, token_seq_len).
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
    logits_last = out.logits[0, -1, :].float().cpu()
    lA = float(logits_last[idA].item())
    lB = float(logits_last[idB].item())
    return lA, lB, seq_len


def validate_unpatched_margins(
    model, tokenizer, master: pd.DataFrame, device: str, idA: int, idB: int
) -> pd.DataFrame:
    ensure_dirs()
    writer = ResumableCSVWriter(
        UNPATCHED_REPRO_CSV,
        fieldnames=[
            "prompt_id",
            "family_id",
            "condition",
            "original_margin_from_behavior_file",
            "reproduced_margin",
            "abs_error",
            "pass_tolerance",
            "reproduced_logit_A",
            "reproduced_logit_B",
            "seq_len",
        ],
        key_fields=["prompt_id"],
    )
    already = writer.n_written
    total = len(master)
    failures: List[str] = []
    for i, (_, row) in enumerate(master.iterrows(), start=1):
        pid = str(row["prompt_id"])
        fid = str(row["family_id"])
        cond = str(row["condition"])
        if writer.has({"prompt_id": pid}):
            continue
        text = row["prompt_text"]
        if not text:
            print(f"[validate-unpatched {i}/{total}] SKIP empty prompt: {pid}")
            continue
        try:
            lA, lB, seq_len = baseline_forward(model, tokenizer, text, device, idA, idB)
        except Exception as e:
            print(f"[validate-unpatched {i}/{total}] FAIL {pid}: {e}")
            continue
        orig_margin = float(row["margin_ref"])
        correct_choice = str(row["correct_choice"])
        repro_margin = compute_margin(lA, lB, correct_choice)
        abs_err = abs(repro_margin - orig_margin)
        passes = abs_err <= TOLERANCE_LOOSE
        if abs_err > TOLERANCE_LOOSE:
            failures.append(
                f"{pid}  orig={orig_margin:+.4f}  repro={repro_margin:+.4f}  "
                f"abs_err={abs_err:.4f}  tol_strict={TOLERANCE_STRICT}  "
                f"tol_loose={TOLERANCE_LOOSE}"
            )
        writer.writerow({
            "prompt_id": pid,
            "family_id": fid,
            "condition": cond,
            "original_margin_from_behavior_file": f"{orig_margin:.8f}",
            "reproduced_margin": f"{repro_margin:.8f}",
            "abs_error": f"{abs_err:.8f}",
            "pass_tolerance": str(passes),
            "reproduced_logit_A": f"{lA:.8f}",
            "reproduced_logit_B": f"{lB:.8f}",
            "seq_len": seq_len,
        })
        if i % 20 == 0 or i == total:
            print(
                f"[validate-unpatched {i:3d}/{total}] "
                f"done={writer.n_written-already:4d} new  "
                f"cumulative_failures_above_loose={len(failures)}"
            )
    # Summary report
    df_out = pd.read_csv(UNPATCHED_REPRO_CSV)
    n = len(df_out)
    n_strict = int((pd.to_numeric(df_out["abs_error"]) <= TOLERANCE_STRICT).sum())
    n_loose = int((pd.to_numeric(df_out["abs_error"]) <= TOLERANCE_LOOSE).sum())
    max_err = float(pd.to_numeric(df_out["abs_error"]).max()) if n > 0 else float("nan")
    mean_err = float(pd.to_numeric(df_out["abs_error"]).mean()) if n > 0 else float("nan")
    print(
        f"[validate-unpatched] summary: n={n}  "
        f"pass_strict(≤{TOLERANCE_STRICT})={n_strict}/{n} "
        f"({100*n_strict/max(1,n):.1f}%)  "
        f"pass_loose(≤{TOLERANCE_LOOSE})={n_loose}/{n} "
        f"({100*n_loose/max(1,n):.1f}%)  "
        f"max|err|={max_err:.4f}  mean|err|={mean_err:.4f}"
    )
    if failures:
        print(f"[validate-unpatched] rows above loose tolerance ({len(failures)}):")
        for f in failures[:10]:
            print("  ", f)
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more.  Investigate before full run.")
    return df_out


# ---------------------------------------------------------------------------
# Validation 3: anchor extraction (S3/S4)
# ---------------------------------------------------------------------------
def tokenize_get_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()


def locate_anchor_indices(
    tokenizer, prompt_text: str, ref: Dict[str, Any]
) -> Dict[str, Any]:
    """Given prompt_text and reference anchors from manifest, double-check
    positions by re-tokenizing.

    Strategy: trust the manifest anchor indices as the ground truth (they were
    generated during the original extraction with the same tokenizer) but
    verify they are within 2 tokens of positions obtained by re-tokenizing the
    prompt_text and looking near the expected end-of-user-message position.

    Returns dict with keys:
      s3_idx, s4_idx, s3_window, s4_window, seq_len, match_mode
    """
    ids = tokenize_get_ids(tokenizer, prompt_text)
    n = len(ids)
    s3_ref = ref.get("end_of_user_message_idx")
    s4_ref = ref.get("final_answer_position_idx")

    # Recompute anchors from scratch using prompt text structure to verify
    # (anchors are after the user-specific "User message:" line and before
    #  "Answer with only A or B.\n\nANSWER:")
    # We use the reference indices directly; but we verify by checking the
    # token window is plausible.  If the reference is missing, we compute a
    # fallback by searching for the "ANSWER:" last token and "Answer with only
    # A or B." token position as S3.
    def _find_last_token_for_substring(sub: str) -> Optional[int]:
        # Return the last token index whose tokenization ends AFTER the
        # substring's last character in the text.
        idx = prompt_text.rfind(sub)
        if idx < 0:
            return None
        end_char = idx + len(sub)
        # Walk token boundaries via char_to_token if available
        if hasattr(tokenizer, "char_to_token"):
            # char_to_token often returns None at exact boundaries; try
            # end_char - 1 then end_char
            for off in (0, -1, 1):
                tok = tokenizer.char_to_token(end_char + off)
                if tok is not None:
                    return int(tok)
        # Fallback: encode up to end_char and count tokens
        prefix = prompt_text[: max(0, end_char)]
        return len(tokenize_get_ids(tokenizer, prefix)) - 1

    s4_fallback = _find_last_token_for_substring("ANSWER:")
    s3_fallback = _find_last_token_for_substring("User message:\n")
    if s3_fallback is not None:
        # "User message:\n" is followed by the condition-specific text.  S3 is
        # after that text and before the "\n\nAnswer with only" block.  So we
        # search backward from s4_fallback for the last token of the user
        # message itself.
        answer_block = prompt_text.rfind("\n\nAnswer with only A or B.")
        if answer_block > 0:
            s3_fallback = _find_last_token_for_substring(
                prompt_text[answer_block - 1 : answer_block]
            ) or s3_fallback
            prefix_tok = len(
                tokenize_get_ids(tokenizer, prompt_text[:answer_block])
            ) - 1
            s3_fallback = prefix_tok

    def _pick(ref_val, fallback, name) -> Tuple[int, str]:
        if ref_val is not None and 0 <= ref_val < n:
            mode = "manifest"
            if fallback is not None:
                diff = abs(ref_val - fallback)
                if diff > 2:
                    mode = f"manifest|fallback_diff={diff} (fallback={fallback})"
            return int(ref_val), mode
        if fallback is not None and 0 <= fallback < n:
            return int(fallback), "fallback"
        # last token
        return n - 1, "last_token_fallback"

    s3_idx, s3_mode = _pick(s3_ref, s3_fallback, "S3")
    s4_idx, s4_mode = _pick(s4_ref, s4_fallback, "S4")
    s3_idx = min(s3_idx, n - 1)
    s4_idx = min(s4_idx, n - 1)
    match_mode = f"S3={s3_mode}; S4={s4_mode}"

    return {
        "seq_len": n,
        "s3_idx": s3_idx,
        "s4_idx": s4_idx,
        "s3_window": token_window(tokenizer, ids, s3_idx, radius=5),
        "s4_window": token_window(tokenizer, ids, s4_idx, radius=5),
        "match_mode": match_mode,
    }


def validate_anchors(
    tokenizer, master: pd.DataFrame, n_sample_families: int = 5
) -> pd.DataFrame:
    ensure_dirs()
    writer = ResumableCSVWriter(
        ANCHOR_VAL_CSV,
        fieldnames=[
            "prompt_id",
            "family_id",
            "condition",
            "tokenized_length",
            "S3_token_index",
            "S3_window_5_before_through_4_after",
            "S4_token_index",
            "S4_window_5_before_through_4_after",
            "anchor_lookup_mode",
            "S3_center_token_decoded",
            "S4_center_token_decoded",
        ],
        key_fields=["prompt_id"],
    )
    total = len(master)
    for i, (_, row) in enumerate(master.iterrows(), start=1):
        pid = str(row["prompt_id"])
        if writer.has({"prompt_id": pid}):
            continue
        try:
            res = locate_anchor_indices(tokenizer, row["prompt_text"], row)
        except Exception as e:
            print(f"[validate-anchors {i}/{total}] FAIL {pid}: {e}")
            continue
        s3_lines = "\n".join(res["s3_window"]["window_decoded_lines"])
        s4_lines = "\n".join(res["s4_window"]["window_decoded_lines"])
        ids = tokenize_get_ids(tokenizer, row["prompt_text"])
        center_s3 = tokenizer.decode([ids[res["s3_idx"]]])
        center_s4 = tokenizer.decode([ids[res["s4_idx"]]])
        writer.writerow({
            "prompt_id": pid,
            "family_id": str(row["family_id"]),
            "condition": str(row["condition"]),
            "tokenized_length": res["seq_len"],
            "S3_token_index": res["s3_idx"],
            "S3_window_5_before_through_4_after": s3_lines,
            "S4_token_index": res["s4_idx"],
            "S4_window_5_before_through_4_after": s4_lines,
            "anchor_lookup_mode": res["match_mode"],
            "S3_center_token_decoded": center_s3,
            "S4_center_token_decoded": center_s4,
        })
        if i % 50 == 0 or i == total:
            print(f"[validate-anchors {i:3d}/{total}] new={writer.n_written}")

    df_out = pd.read_csv(ANCHOR_VAL_CSV)
    # Write human-readable markdown samples for first n_sample_families × 8 conds
    fams = sorted(df_out["family_id"].unique().tolist())[:n_sample_families]
    md_lines = []
    md_lines.append("# HQ80 S3/S4 Anchor Validation — Samples")
    md_lines.append("")
    md_lines.append(f"Generated: {utc_now()}")
    md_lines.append(f"Model tokenizer: {MODEL_NAME}")
    md_lines.append("")
    md_lines.append(
        "Per prompt we show the ~10 tokens around S3 (end of user message) and "
        "S4 (final ANSWER position).  Center token of each window is marked "
        "with `>>>`."
    )
    md_lines.append("")
    sample_df = df_out[df_out["family_id"].isin(fams)].copy()
    sample_df = sample_df.sort_values(["family_id", "condition"])
    for fam in fams:
        md_lines.append(f"## Family: `{fam}`")
        md_lines.append("")
        sub = sample_df[sample_df["family_id"] == fam]
        # Also capture the full prompt_text from master for neutral
        neut_master = master[
            (master["family_id"] == fam) & (master["condition"] == NEUTRAL)
        ]
        if len(neut_master) > 0:
            md_lines.append("<details><summary>Neutral prompt_text (shared prefix)</summary>")
            md_lines.append("")
            md_lines.append("```text")
            md_lines.append(str(neut_master.iloc[0]["prompt_text"]))
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("</details>")
            md_lines.append("")
        for _, s in sub.iterrows():
            md_lines.append(f"### Condition: `{s['condition']}`")
            md_lines.append("")
            md_lines.append(
                f"- tokenized length: **{s['tokenized_length']}**  "
                f"  S3 idx = **{s['S3_token_index']}**  "
                f"  S4 idx = **{s['S4_token_index']}**  "
                f"  mode = `{s['anchor_lookup_mode']}`"
            )
            md_lines.append("")
            md_lines.append("**S3 (end of user message) window:**")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(str(s["S3_window_5_before_through_4_after"]))
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("**S4 (final ANSWER position) window:**")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(str(s["S4_window_5_before_through_4_after"]))
            md_lines.append("```")
            md_lines.append("")
    ANCHOR_VAL_SAMPLES_MD.write_text("\n".join(md_lines) + "\n")
    print(
        f"[validate-anchors] wrote {ANCHOR_VAL_CSV} ({len(df_out)} rows) and "
        f"{ANCHOR_VAL_SAMPLES_MD} (samples for families: {fams})"
    )
    return df_out


# ---------------------------------------------------------------------------
# Validation 4: layer indexing
# ---------------------------------------------------------------------------
def validate_layer_indexing(
    model, tokenizer, sample_prompt: str, device: str, layers: List[int]
) -> None:
    """Verify that hooking block L produces the same vector as
    output_hidden_states[L+1] at the final position.

    This confirms the project-wide convention (hooks on block output =
    hidden_states[L+1] in HF tuple).  Results are written to a short .txt
    log in validation/.
    """
    ensure_dirs()
    lines = []
    lines.append("Layer indexing validation — hook vs output_hidden_states")
    lines.append("=" * 72)
    lines.append(f"timestamp = {utc_now()}")
    lines.append(f"model     = {MODEL_NAME}")
    lines.append(
        "Convention: hook(model.model.layers[L]) output should equal "
        "output_hidden_states[L+1] at a specific token position."
    )
    block_list_name, block_list = find_transformer_block_list(model)
    lines.append(f"block_list_name = {block_list_name}  n_blocks = {len(block_list)}")
    inputs = tokenizer(sample_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_tok = inputs["input_ids"].shape[1]
    pos = n_tok - 1  # S4 for simplicity
    lines.append(f"prompt tokenized len = {n_tok}, validated position = {pos} (final)")

    # 1) output_hidden_states route
    with torch.no_grad():
        out_ref = model(
            **inputs, output_hidden_states=True, use_cache=False, return_dict=True
        )
    hs = out_ref.hidden_states  # tuple
    lines.append(
        f"output_hidden_states tuple length = {len(hs)}  "
        f"(expected = 1 (embeds) + {len(block_list)} (blocks) = {len(block_list)+1})"
    )

    # 2) hook route
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(L):
        def _h(mod, args, output):
            if isinstance(output, tuple):
                hs_t = output[0]
            else:
                hs_t = output
            captured[L] = hs_t[0, pos, :].detach().float().cpu().clone()
        return _h

    handles = [block_list[L].register_forward_hook(make_hook(L)) for L in layers]
    try:
        with torch.no_grad():
            out_hook = model(
                **inputs, output_hidden_states=False, use_cache=False, return_dict=True
            )
    finally:
        for h in handles:
            h.remove()

    mismatches = []
    for L in layers:
        ref = hs[L + 1][0, pos, :].detach().float().cpu()
        cap = captured.get(L)
        if cap is None:
            lines.append(f"L={L:2d}: hook did not fire")
            mismatches.append((L, "no_hook"))
            continue
        diff = float((ref - cap).abs().max().item())
        match = diff < 1e-5
        lines.append(
            f"L={L:2d}:  max_abs_diff(hidden_states[L+1], hook_capture) = {diff:.2e}  "
            f"match={match}"
        )
        if not match:
            mismatches.append((L, diff))
    # Check logits are identical
    logits_ref = out_ref.logits[0, -1, :].float().cpu()
    logits_hook = out_hook.logits[0, -1, :].float().cpu()
    logit_diff = float((logits_ref - logits_hook).abs().max().item())
    lines.append(f"logits max_abs_diff (ref vs hook-run) = {logit_diff:.2e}")
    if logit_diff > 1e-4:
        mismatches.append(("logits", logit_diff))

    lines.append("")
    lines.append(f"mismatches (if any): {mismatches}")
    lines.append(f"RESULT: {'PASS' if not mismatches else 'FAIL — investigate'}")
    LAYER_IDX_TXT.write_text("\n".join(lines) + "\n")
    print(f"[validate-layer-indexing] wrote {LAYER_IDX_TXT}")
    if mismatches:
        print(f"[validate-layer-indexing][FAIL] mismatches = {mismatches}")
    else:
        print("[validate-layer-indexing] PASS (hook <-> output_hidden_states[L+1] convention holds)")


# ---------------------------------------------------------------------------
# Live-cache baseline (source activations for patching)
# ---------------------------------------------------------------------------
class BaselineCache:
    """
    Live cache of baseline forwards:
      keyed by prompt_id -> dict:
        lA, lB, margin, seq_len,
        anchors: {"S3": idx, "S4": idx},
        layers:  {L: {"S3": vec_cpu_f32, "S4": vec_cpu_f32}}
    """

    def __init__(self, cache_dir: Path, master: pd.DataFrame):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, Dict[str, Any]] = {}
        self._master_by_pid = {str(r["prompt_id"]): r for _, r in master.iterrows()}

    def _path(self, pid: str) -> Path:
        safe = hashlib.md5(pid.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"{pid}_{safe}.pt"

    def has(self, pid: str) -> bool:
        return pid in self._mem or self._path(pid).exists()

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        if pid in self._mem:
            return self._mem[pid]
        p = self._path(pid)
        if p.exists():
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                self._mem[pid] = obj
                return obj
            except Exception:
                return None
        return None

    def save(self, pid: str, obj: Dict[str, Any]) -> None:
        p = self._path(pid)
        # Do not keep layer tensors in memory indefinitely — only metadata and
        # hot paths.  We still write to disk for resume.
        torch.save(obj, p)
        # Drop the big tensors from memory to keep footprint bounded.
        obj_light = dict(obj)
        layers_light: Dict[int, Dict[str, torch.Tensor]] = {}
        for L, d in obj.get("layers", {}).items():
            layers_light[L] = {
                anchor: vec
                for anchor, vec in d.items()
            }
        obj_light["layers"] = layers_light
        self._mem[pid] = obj_light


def baseline_forward_with_cache_all(
    model,
    tokenizer,
    pid: str,
    prompt_text: str,
    correct_choice: str,
    device: str,
    idA: int,
    idB: int,
    layers: List[int],
    anchors_list: List[str] = ("S3", "S4"),
) -> Dict[str, Any]:
    """Run a single baseline forward with output_hidden_states=True.

    Returns dict suitable for BaselineCache.save(pid, ...):
      lA, lB, margin, seq_len, correct_choice,
      anchors: {S3: idx, S4: idx},
      layers: {L: {S3: vec_cpu_f32, S4: vec_cpu_f32}},
      prompt_sha1: str (for integrity),
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ids = inputs["input_ids"][0].tolist()
    n = len(ids)
    # Compute anchor indices: use master reference via prompt_id.
    # The caller is responsible for locating anchors; we re-derive from the
    # prompt text structure if not passed.
    # For simplicity, compute anchors here using the manifest-free fallback.
    s3_idx = _compute_s3_from_text(tokenizer, prompt_text, ids)
    s4_idx = n - 1  # S4 always = last token in prompt (position before logits)
    s3_idx = min(s3_idx, n - 1)

    with torch.no_grad():
        out = model(
            **inputs, output_hidden_states=True, use_cache=False, return_dict=True
        )
    logits_last = out.logits[0, -1, :].float().cpu()
    lA = float(logits_last[idA].item())
    lB = float(logits_last[idB].item())
    margin = compute_margin(lA, lB, correct_choice)
    hidden = out.hidden_states  # tuple: embeds + layers
    layer_caches: Dict[int, Dict[str, torch.Tensor]] = {}
    for L in layers:
        h = hidden[L + 1]  # block L output
        vec_s3 = h[0, s3_idx, :].detach().float().cpu().contiguous()
        vec_s4 = h[0, s4_idx, :].detach().float().cpu().contiguous()
        layer_caches[L] = {"S3": vec_s3, "S4": vec_s4}

    return {
        "lA": lA,
        "lB": lB,
        "margin": margin,
        "seq_len": n,
        "correct_choice": correct_choice,
        "anchors": {"S3": s3_idx, "S4": s4_idx},
        "layers": layer_caches,
        "prompt_sha1": hashlib.sha1(prompt_text.encode("utf-8")).hexdigest(),
    }


def _compute_s3_from_text(tokenizer, prompt_text: str, ids: List[int]) -> int:
    """Fallback S3 = last token of the condition-specific user message.

    We locate the last occurrence of "\n\nAnswer with only A or B." and take
    the token index of the character position immediately before that block.
    """
    marker = "\n\nAnswer with only A or B."
    pos = prompt_text.rfind(marker)
    if pos < 0:
        return len(ids) - 2
    # Tokenize prefix up to `pos` (exclusive of the marker)
    prefix = prompt_text[:pos]
    return max(0, len(tokenizer(prefix, return_tensors="pt")["input_ids"][0].tolist()) - 1)


# ---------------------------------------------------------------------------
# Patching forward (generic by anchor position)
# ---------------------------------------------------------------------------
def patchable_forward_at_pos(
    model,
    tokenizer,
    prompt_text: str,
    anchor_token_index: int,
    device: str,
    block_list_name: str,
    block_list,
    layer_idx: int,
    replacement_vector: torch.Tensor,
    idA: int,
    idB: int,
) -> Tuple[float, float, bool]:
    """
    Forward pass with a single hook replacing the hidden state at
    (layer_idx, anchor_token_index).

    Returns (logit_A, logit_B, hook_fired_bool).
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]
    pos = min(anchor_token_index, seq_len - 1)

    target_dtype = next(block_list[0].parameters()).dtype
    replacement = replacement_vector.to(device=device, dtype=target_dtype)

    fired = {"ok": False}

    def hook(mod, args, output):
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = ()
        new_hs = hs.clone()
        new_hs[0, pos, :] = replacement.to(dtype=new_hs.dtype, device=new_hs.device)
        fired["ok"] = True
        if isinstance(output, tuple):
            return (new_hs,) + rest
        return new_hs

    h = block_list[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(
                **inputs, output_hidden_states=False, use_cache=False, return_dict=True
            )
    finally:
        h.remove()
    logits_last = out.logits[0, -1, :].float().cpu()
    lA = float(logits_last[idA].item())
    lB = float(logits_last[idB].item())
    return lA, lB, fired["ok"]


# ---------------------------------------------------------------------------
# Raw patching row assembly
# ---------------------------------------------------------------------------
def _build_raw_row(
    *,
    run_id: str,
    pid_target: str,
    pid_source: str,
    fam_target: str,
    fam_source: str,
    condition_main: str,
    target_condition: str,
    source_condition: str,
    patch_type: str,
    anchor: str,
    layer: int,
    correct_choice: str,
    false_choice: str,
    idA: int,
    idB: int,
    orig_neutral_margin: float,
    orig_condition_margin: float,
    orig_target_margin: float,
    patched_margin: float,
    target_anchor_idx: int,
    source_anchor_idx: int,
    target_token_len: int,
    source_token_len: int,
    dtype: str,
    device: str,
) -> Dict[str, Any]:
    delta = orig_condition_margin - orig_neutral_margin
    if patch_type == "rescue":
        rescue_effect = patched_margin - orig_condition_margin
        transfer_effect = float("nan")
        patch_effect = rescue_effect
    elif patch_type == "transfer":
        rescue_effect = float("nan")
        transfer_effect = patched_margin - orig_neutral_margin
        patch_effect = transfer_effect
    elif patch_type == "self_patch":
        # target = source, use self margin as baseline
        rescue_effect = float("nan")
        transfer_effect = float("nan")
        patch_effect = patched_margin - orig_target_margin
    elif patch_type in ("random_family_rescue",):
        rescue_effect = patched_margin - orig_condition_margin
        transfer_effect = float("nan")
        patch_effect = rescue_effect
    elif patch_type in ("random_family_transfer",):
        rescue_effect = float("nan")
        transfer_effect = patched_margin - orig_neutral_margin
        patch_effect = transfer_effect
    else:
        rescue_effect = float("nan")
        transfer_effect = float("nan")
        patch_effect = patched_margin - orig_target_margin

    sdelta = sign_of(delta)
    expected_signed_rescue = ""
    expected_signed_transfer = ""
    moved_expected = ""
    if patch_type == "rescue":
        expected_signed_rescue = (-sdelta) * rescue_effect if sdelta != 0 else rescue_effect
        if sdelta > 0:
            moved_expected = bool(rescue_effect < 0)
        elif sdelta < 0:
            moved_expected = bool(rescue_effect > 0)
        else:
            moved_expected = bool(abs(rescue_effect) == 0)
    elif patch_type == "transfer":
        expected_signed_transfer = sdelta * transfer_effect if sdelta != 0 else transfer_effect
        if sdelta > 0:
            moved_expected = bool(transfer_effect > 0)
        elif sdelta < 0:
            moved_expected = bool(transfer_effect < 0)
        else:
            moved_expected = bool(abs(transfer_effect) == 0)
    return {
        "run_id": run_id,
        "prompt_id_target": pid_target,
        "prompt_id_source": pid_source,
        "family_id_target": fam_target,
        "family_id_source": fam_source,
        "condition": condition_main,
        "target_condition": target_condition,
        "source_condition": source_condition,
        "patch_type": patch_type,
        "anchor": anchor,
        "layer": layer,
        "correct_choice": correct_choice,
        "false_choice": false_choice,
        "answer_token_id_A": idA,
        "answer_token_id_B": idB,
        "original_neutral_margin": (
            f"{orig_neutral_margin:.8f}"
            if np.isfinite(orig_neutral_margin)
            else ""
        ),
        "original_condition_margin": (
            f"{orig_condition_margin:.8f}"
            if np.isfinite(orig_condition_margin)
            else ""
        ),
        "delta_margin": f"{delta:.8f}",
        "original_target_margin": (
            f"{orig_target_margin:.8f}" if np.isfinite(orig_target_margin) else ""
        ),
        "patched_margin": f"{patched_margin:.8f}",
        "rescue_effect": (
            f"{rescue_effect:+.8f}" if np.isfinite(rescue_effect) else ""
        ),
        "transfer_effect": (
            f"{transfer_effect:+.8f}" if np.isfinite(transfer_effect) else ""
        ),
        "patch_effect": f"{patch_effect:+.8f}",
        "expected_sign_from_delta": sdelta,
        "expected_signed_rescue_effect": (
            f"{float(expected_signed_rescue):+.8f}"
            if expected_signed_rescue != "" and np.isfinite(float(expected_signed_rescue))
            else ""
        ),
        "expected_signed_transfer_effect": (
            f"{float(expected_signed_transfer):+.8f}"
            if expected_signed_transfer != "" and np.isfinite(float(expected_signed_transfer))
            else ""
        ),
        "moved_in_expected_direction": (
            str(moved_expected) if isinstance(moved_expected, bool) else ""
        ),
        "target_anchor_index": target_anchor_idx,
        "source_anchor_index": source_anchor_idx,
        "target_token_length": target_token_len,
        "source_token_length": source_token_len,
        "dtype": dtype,
        "device": device,
        "model_name": MODEL_NAME,
        "created_at": utc_now(),
    }


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
def stage_config(
    stage: str, all_fams: List[str]
) -> Dict[str, Any]:
    """
    Return dict: {
        families: list[str],
        conditions: list[str],
        layers: list[int],
        patch_types: list[str],
        with_self_patch: bool,
        description: str,
    }
    """
    if stage == "0" or stage == "sanity":
        return {
            "families": all_fams[:2],
            "conditions": ALL_NONNEUTRAL,
            "layers": PATCH_LAYERS_SANITY,
            "patch_types": ["rescue", "transfer"],
            "with_self_patch": True,
            "description": "Stage 0 (sanity): 2 families × 7 conditions × S3/S4 × [28,32,34] × rescue/transfer + self patch",
        }
    if stage == "1" or stage == "priority":
        return {
            "families": all_fams,
            "conditions": FALSE_PRESSURE,
            "layers": PATCH_LAYERS_PRIORITY,
            "patch_types": ["rescue", "transfer"],
            "with_self_patch": False,
            "description": "Stage 1 (priority): 80 families × 4 false-pressure × S3/S4 × [28,32,34,35] × rescue/transfer",
        }
    if stage == "2" or stage == "controls":
        return {
            "families": all_fams,
            "conditions": [*CONTROL_CONDS, *FALSE_PRESSURE],
            "layers": PATCH_LAYERS_PRIORITY,
            "patch_types": ["rescue", "transfer"],
            "with_self_patch": False,
            "description": "Stage 2 (+controls): priority stage + distractor / true-belief / true-rationale",
        }
    if stage == "3" or stage == "remaining":
        return {
            "families": all_fams,
            "conditions": ALL_NONNEUTRAL,
            "layers": PATCH_LAYERS_FULL,
            "patch_types": ["rescue", "transfer"],
            "with_self_patch": False,
            "description": "Stage 3 (+remaining layers): full 15,680-row target",
        }
    if stage == "4" or stage == "full":
        return {
            "families": all_fams,
            "conditions": ALL_NONNEUTRAL,
            "layers": PATCH_LAYERS_FULL,
            "patch_types": ["rescue", "transfer"],
            "with_self_patch": False,
            "description": "Stage 4 (full): same as stage 3",
        }
    if stage == "self":
        return {
            "families": all_fams[:5],
            "conditions": CONDITIONS_ORDER,  # all 8 incl neutral
            "layers": PATCH_LAYERS_FULL,
            "patch_types": [],
            "with_self_patch": True,
            "description": "Self-patch control only: ≥5 families × 8 conditions × S3/S4 × 7 layers",
        }
    if stage == "random":
        return {
            "families": all_fams[:20],
            "conditions": ALL_NONNEUTRAL,
            "layers": PATCH_LAYERS_PRIORITY,
            "patch_types": ["random_family_rescue", "random_family_transfer"],
            "with_self_patch": False,
            "description": "Random-family control: 20 families × 7 conditions × S3/S4 × [28,32,34,35] × random rescue/transfer",
        }
    raise ValueError(f"unknown --stage: {stage}")


# ---------------------------------------------------------------------------
# Main patching driver
# ---------------------------------------------------------------------------
def _pid(master: pd.DataFrame, fid: str, cond: str) -> Optional[str]:
    sub = master[(master["family_id"] == fid) & (master["condition"] == cond)]
    if len(sub) == 0:
        return None
    return str(sub.iloc[0]["prompt_id"])


def _ensure_baseline_cached(
    baseline_cache: BaselineCache,
    pid: str,
    master_row: pd.Series,
    model,
    tokenizer,
    device: str,
    idA: int,
    idB: int,
    layers: List[int],
) -> Optional[Dict[str, Any]]:
    obj = baseline_cache.get(pid)
    if obj is not None:
        return obj
    prompt_text = str(master_row["prompt_text"])
    if not prompt_text:
        return None
    try:
        obj = baseline_forward_with_cache_all(
            model, tokenizer, pid, prompt_text,
            str(master_row["correct_choice"]), device, idA, idB, layers,
        )
    except Exception as e:
        print(f"[baseline-cache] FAIL pid={pid}: {e}")
        return None
    baseline_cache.save(pid, obj)
    return obj


def run_patching_stage(
    args,
    model, tokenizer,
    master: pd.DataFrame,
    device: str,
    idA: int, idB: int, strategy: str,
    stage_cfg: Dict[str, Any],
) -> None:
    ensure_dirs()
    run_id = new_run_id()
    block_list_name, block_list = find_transformer_block_list(model)
    target_dtype = str(next(block_list[0].parameters()).dtype).replace("torch.", "")

    families = stage_cfg["families"]
    conditions = stage_cfg["conditions"]
    layers = sorted(stage_cfg["layers"])
    patch_types = list(stage_cfg["patch_types"])
    do_self = bool(stage_cfg["with_self_patch"])

    print("[patching] stage config:")
    print(f"  description = {stage_cfg['description']}")
    print(f"  families    = {len(families)} ({families[:3]}...{families[-2:]})")
    print(f"  conditions  = {conditions}")
    print(f"  layers      = {layers}")
    print(f"  patch_types = {patch_types}")
    print(f"  self_patch? = {do_self}")
    print(f"  anchors     = {ANCHORS}")
    print(f"  run_id      = {run_id}")
    print(f"  dtype       = {target_dtype}  device = {device}")

    baseline_cache = BaselineCache(CACHE_DIR, master)
    master_by_pid = {str(r["prompt_id"]): r for _, r in master.iterrows()}

    raw_writer = ResumableCSVWriter(RAW_CSV, RAW_COLUMNS, key_fields=[
        "prompt_id_target", "prompt_id_source", "patch_type", "anchor", "layer",
        "condition", "family_id_target", "family_id_source",
    ])
    self_writer = ResumableCSVWriter(SELF_PATCH_CSV, [
        "prompt_id", "family_id", "condition", "anchor", "layer",
        "original_margin", "patched_margin", "abs_diff",
        "anchor_index", "token_length", "run_id", "device", "dtype",
    ], key_fields=["prompt_id", "anchor", "layer"])
    rand_writer = ResumableCSVWriter(RANDOM_FAMILY_CSV, RAW_COLUMNS, key_fields=[
        "prompt_id_target", "prompt_id_source", "patch_type", "anchor", "layer",
        "condition", "family_id_target", "family_id_source",
    ])

    # Random-family plan (fixed seed): for each (family_i, condition, anchor,
    # layer, ptype) pick a single family_j != family_i from the same pool of
    # eligible families.
    rng = random.Random(12345)
    family_pool = list(families)

    def pick_rand_family(i: str, condition: str) -> str:
        candidates = [f for f in family_pool if f != i]
        # Deterministic by (i, condition)
        rng2 = random.Random(hash((i, condition)) & 0xFFFFFFFF)
        return rng2.choice(candidates)

    total_todo = (
        len(families) * len(conditions) * len(ANCHORS) * len(layers)
        * (len(patch_types) + (2 if do_self else 0))
    )
    done = 0
    start = time.time()
    last_report = 0.0

    for fam_i, fid in enumerate(families, start=1):
        # Ensure baselines for all 8 conditions + this family are cached
        # (even if we only patch non-neutral conditions, we need neutral for
        # rescue source and transfer target).
        for c in CONDITIONS_ORDER:
            pid = _pid(master, fid, c)
            if pid is None:
                continue
            mrow = master_by_pid.get(pid)
            if mrow is None:
                continue
            _ensure_baseline_cached(
                baseline_cache, pid, mrow, model, tokenizer, device, idA, idB, layers,
            )

        for cond in conditions:
            pid_N = _pid(master, fid, NEUTRAL)
            pid_C = _pid(master, fid, cond)
            if pid_N is None or pid_C is None:
                print(f"[patching] SKIP fid={fid} cond={cond}: missing prompt_id")
                continue
            base_N = baseline_cache.get(pid_N)
            base_C = baseline_cache.get(pid_C)
            if base_N is None or base_C is None:
                print(f"[patching] SKIP fid={fid} cond={cond}: baseline cache miss")
                continue
            correct_choice = str(base_C["correct_choice"])
            false_choice = str(master_by_pid[pid_C]["false_choice"])
            orig_N = float(base_N["margin"])
            orig_C = float(base_C["margin"])

            for anchor in ANCHORS:
                s3s4_N = int(base_N["anchors"][anchor])
                s3s4_C = int(base_C["anchors"][anchor])
                len_N = int(base_N["seq_len"])
                len_C = int(base_C["seq_len"])
                for L in layers:
                    vec_N = base_N["layers"][L][anchor]
                    vec_C = base_C["layers"][L][anchor]
                    text_N = str(master_by_pid[pid_N]["prompt_text"])
                    text_C = str(master_by_pid[pid_C]["prompt_text"])

                    # -------- rescue (neutral -> condition) --------
                    if "rescue" in patch_types:
                        key_test = {
                            "prompt_id_target": pid_C,
                            "prompt_id_source": pid_N,
                            "patch_type": "rescue",
                            "anchor": anchor,
                            "layer": L,
                            "condition": cond,
                            "family_id_target": fid,
                            "family_id_source": fid,
                        }
                        if not raw_writer.has(key_test):
                            try:
                                lA_r, lB_r, ok = patchable_forward_at_pos(
                                    model, tokenizer, text_C, s3s4_C, device,
                                    block_list_name, block_list, L, vec_N, idA, idB,
                                )
                                if not ok:
                                    raise RuntimeError("hook not fired")
                            except Exception as e:
                                print(
                                    f"[patching] rescue FAIL fid={fid} cond={cond} "
                                    f"anchor={anchor} L={L}: {e}"
                                )
                                lA_r = lB_r = float("nan")
                            p_margin = compute_margin(lA_r, lB_r, correct_choice)
                            row = _build_raw_row(
                                run_id=run_id,
                                pid_target=pid_C, pid_source=pid_N,
                                fam_target=fid, fam_source=fid,
                                condition_main=cond,
                                target_condition=cond, source_condition=NEUTRAL,
                                patch_type="rescue",
                                anchor=anchor, layer=L,
                                correct_choice=correct_choice, false_choice=false_choice,
                                idA=idA, idB=idB,
                                orig_neutral_margin=orig_N,
                                orig_condition_margin=orig_C,
                                orig_target_margin=orig_C,
                                patched_margin=p_margin,
                                target_anchor_idx=s3s4_C,
                                source_anchor_idx=s3s4_N,
                                target_token_len=len_C,
                                source_token_len=len_N,
                                dtype=target_dtype, device=device,
                            )
                            raw_writer.writerow(row)

                    # -------- transfer (condition -> neutral) --------
                    if "transfer" in patch_types:
                        key_test = {
                            "prompt_id_target": pid_N,
                            "prompt_id_source": pid_C,
                            "patch_type": "transfer",
                            "anchor": anchor,
                            "layer": L,
                            "condition": cond,
                            "family_id_target": fid,
                            "family_id_source": fid,
                        }
                        if not raw_writer.has(key_test):
                            try:
                                lA_t, lB_t, ok = patchable_forward_at_pos(
                                    model, tokenizer, text_N, s3s4_N, device,
                                    block_list_name, block_list, L, vec_C, idA, idB,
                                )
                                if not ok:
                                    raise RuntimeError("hook not fired")
                            except Exception as e:
                                print(
                                    f"[patching] transfer FAIL fid={fid} cond={cond} "
                                    f"anchor={anchor} L={L}: {e}"
                                )
                                lA_t = lB_t = float("nan")
                            p_margin = compute_margin(lA_t, lB_t, correct_choice)
                            row = _build_raw_row(
                                run_id=run_id,
                                pid_target=pid_N, pid_source=pid_C,
                                fam_target=fid, fam_source=fid,
                                condition_main=cond,
                                target_condition=NEUTRAL, source_condition=cond,
                                patch_type="transfer",
                                anchor=anchor, layer=L,
                                correct_choice=correct_choice, false_choice=false_choice,
                                idA=idA, idB=idB,
                                orig_neutral_margin=orig_N,
                                orig_condition_margin=orig_C,
                                orig_target_margin=orig_N,
                                patched_margin=p_margin,
                                target_anchor_idx=s3s4_N,
                                source_anchor_idx=s3s4_C,
                                target_token_len=len_N,
                                source_token_len=len_C,
                                dtype=target_dtype, device=device,
                            )
                            raw_writer.writerow(row)

                    # -------- random-family rescue/transfer --------
                    for rpt in ("random_family_rescue", "random_family_transfer"):
                        if rpt not in patch_types:
                            continue
                        fid_src = pick_rand_family(fid, cond)
                        if rpt == "random_family_rescue":
                            # target = condition (family i), source = neutral (family j)
                            pid_S = _pid(master, fid_src, NEUTRAL)
                            pid_T = pid_C
                            src_cond = NEUTRAL
                            tgt_cond = cond
                            orig_target_m = orig_C
                            s3s4_src = int(baseline_cache.get(pid_S)["anchors"][anchor]) if baseline_cache.get(pid_S) else s3s4_N
                            s3s4_tgt = s3s4_C
                            len_src = int(baseline_cache.get(pid_S)["seq_len"]) if baseline_cache.get(pid_S) else len_N
                            len_tgt = len_C
                            text_tgt = text_C
                        else:
                            # target = neutral (family i), source = condition (family j)
                            pid_S = _pid(master, fid_src, cond)
                            pid_T = pid_N
                            src_cond = cond
                            tgt_cond = NEUTRAL
                            orig_target_m = orig_N
                            s3s4_src = int(baseline_cache.get(pid_S)["anchors"][anchor]) if baseline_cache.get(pid_S) else s3s4_C
                            s3s4_tgt = s3s4_N
                            len_src = int(baseline_cache.get(pid_S)["seq_len"]) if baseline_cache.get(pid_S) else len_C
                            len_tgt = len_N
                            text_tgt = text_N
                        if pid_S is None:
                            continue
                        # Ensure source cached
                        mrow_s = master_by_pid.get(pid_S)
                        if mrow_s is not None:
                            _ensure_baseline_cached(
                                baseline_cache, pid_S, mrow_s,
                                model, tokenizer, device, idA, idB, layers,
                            )
                        base_S = baseline_cache.get(pid_S)
                        if base_S is None:
                            continue
                        vec_S = base_S["layers"][L][anchor]
                        orig_S_neutral = (
                            float(baseline_cache.get(
                                _pid(master, fid_src, NEUTRAL) or "",
                            )["margin"])
                            if _pid(master, fid_src, NEUTRAL) and baseline_cache.get(
                                _pid(master, fid_src, NEUTRAL) or ""
                            )
                            else float("nan")
                        )
                        orig_S_cond = (
                            float(base_S["margin"])
                            if src_cond != NEUTRAL
                            else float(baseline_cache.get(pid_S)["margin"])
                        )
                        if src_cond == NEUTRAL:
                            orig_S_neutral = float(base_S["margin"])
                        else:
                            orig_S_cond = float(base_S["margin"])
                        if not np.isfinite(orig_S_neutral):
                            orig_S_neutral = orig_N
                        if not np.isfinite(orig_S_cond):
                            orig_S_cond = orig_C
                        key_test_r = {
                            "prompt_id_target": pid_T,
                            "prompt_id_source": pid_S,
                            "patch_type": rpt,
                            "anchor": anchor,
                            "layer": L,
                            "condition": cond,
                            "family_id_target": fid,
                            "family_id_source": fid_src,
                        }
                        if rand_writer.has(key_test_r):
                            continue
                        try:
                            lA_r, lB_r, ok = patchable_forward_at_pos(
                                model, tokenizer, text_tgt, s3s4_tgt, device,
                                block_list_name, block_list, L, vec_S, idA, idB,
                            )
                            if not ok:
                                raise RuntimeError("hook not fired")
                        except Exception as e:
                            print(
                                f"[patching] {rpt} FAIL fid={fid} cond={cond} "
                                f"anchor={anchor} L={L} src_fam={fid_src}: {e}"
                            )
                            lA_r = lB_r = float("nan")
                        p_margin = compute_margin(lA_r, lB_r, correct_choice)
                        row = _build_raw_row(
                            run_id=run_id,
                            pid_target=pid_T, pid_source=pid_S,
                            fam_target=fid, fam_source=fid_src,
                            condition_main=cond,
                            target_condition=tgt_cond, source_condition=src_cond,
                            patch_type=rpt,
                            anchor=anchor, layer=L,
                            correct_choice=correct_choice, false_choice=false_choice,
                            idA=idA, idB=idB,
                            orig_neutral_margin=orig_S_neutral,
                            orig_condition_margin=orig_S_cond,
                            orig_target_margin=orig_target_m,
                            patched_margin=p_margin,
                            target_anchor_idx=s3s4_tgt,
                            source_anchor_idx=s3s4_src,
                            target_token_len=len_tgt,
                            source_token_len=len_src,
                            dtype=target_dtype, device=device,
                        )
                        rand_writer.writerow(row)

                    # -------- self patch (sanity control) --------
                    if do_self:
                        # Self-patch on BOTH neutral and condition prompts,
                        # but condition has 8 types; we do all 8.
                        for (pid_self, txt, base_self, cond_self, s3s4_self) in (
                            (pid_C, text_C, base_C, cond, s3s4_C),
                            (pid_N, text_N, base_N, NEUTRAL, s3s4_N),
                        ):
                            if self_writer.has({
                                "prompt_id": pid_self,
                                "anchor": anchor,
                                "layer": L,
                            }):
                                continue
                            vec_self = base_self["layers"][L][anchor]
                            try:
                                lA_sp, lB_sp, ok = patchable_forward_at_pos(
                                    model, tokenizer, txt, s3s4_self, device,
                                    block_list_name, block_list, L, vec_self,
                                    idA, idB,
                                )
                                if not ok:
                                    raise RuntimeError("hook not fired")
                            except Exception as e:
                                print(
                                    f"[patching] self-patch FAIL pid={pid_self} "
                                    f"anchor={anchor} L={L}: {e}"
                                )
                                continue
                            orig_self = float(base_self["margin"])
                            p_self = compute_margin(lA_sp, lB_sp, correct_choice)
                            self_writer.writerow({
                                "prompt_id": pid_self,
                                "family_id": fid,
                                "condition": cond_self,
                                "anchor": anchor,
                                "layer": L,
                                "original_margin": f"{orig_self:.8f}",
                                "patched_margin": f"{p_self:.8f}",
                                "abs_diff": f"{abs(p_self-orig_self):.8f}",
                                "anchor_index": s3s4_self,
                                "token_length": base_self["seq_len"],
                                "run_id": run_id,
                                "device": device,
                                "dtype": target_dtype,
                            })

                    done += max(1,
                        (2 if "rescue" in patch_types else 0) +
                        (2 if "transfer" in patch_types else 0)
                    )
                    t_now = time.time()
                    if t_now - last_report > 30:
                        elapsed = t_now - start
                        pct = 100 * done / max(1, total_todo)
                        print(
                            f"[patching] progress {pct:.1f}%  "
                            f"done={done}/{total_todo}  "
                            f"raw_rows={raw_writer.n_written}  "
                            f"self_rows={self_writer.n_written}  "
                            f"random_rows={rand_writer.n_written}  "
                            f"elapsed={elapsed/60:.1f} min  "
                            f"current=family {fam_i}/{len(families)} ({fid}) cond={cond} "
                            f"anchor={anchor} L={L}"
                        )
                        last_report = t_now

    # Final self-patch report
    if SELF_PATCH_CSV.exists():
        try:
            sp_df = pd.read_csv(SELF_PATCH_CSV)
            abs_d = pd.to_numeric(sp_df["abs_diff"], errors="coerce").dropna()
            if len(abs_d) > 0:
                print(
                    f"[self-patch-control] n={len(abs_d)}  "
                    f"mean|Δmargin|={abs_d.mean():.3e}  "
                    f"max|Δmargin|={abs_d.max():.3e}  "
                    f"n_gt_1e-3={int((abs_d>1e-3).sum())}  "
                    f"n_gt_1e-2={int((abs_d>1e-2).sum())}"
                )
        except Exception as e:
            print(f"[self-patch-control] summary read fail: {e}")
    print(
        f"[patching] done.  raw={raw_writer.n_written}  "
        f"self={self_writer.n_written}  random={rand_writer.n_written}  "
        f"elapsed={(time.time()-start)/60:.1f} min"
    )


# ---------------------------------------------------------------------------
# Primary analysis: family-level summaries + Q1-Q4 stats
# ---------------------------------------------------------------------------
def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def _bootstrap_mean_ci(
    vals: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 0
) -> Tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(vals)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = vals[idx].mean()
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    return float(vals.mean()), float(np.quantile(means, lo_q)), float(np.quantile(means, hi_q))


def family_level_summary() -> pd.DataFrame:
    ensure_dirs()
    if not RAW_CSV.exists():
        raise FileNotFoundError(RAW_CSV)
    df = pd.read_csv(RAW_CSV)
    if len(df) == 0:
        raise ValueError(f"{RAW_CSV} is empty; run a patching stage first.")

    rows: List[Dict[str, Any]] = []
    grp_cols = ["family_id_target", "condition", "anchor", "patch_type"]
    valid_ptypes = {"rescue", "transfer"}
    df_f = df[df["patch_type"].isin(valid_ptypes)].copy()
    df_f["patch_effect_num"] = _to_float_series(df_f["patch_effect"])
    df_f["expected_signed_rescue_num"] = _to_float_series(
        df_f["expected_signed_rescue_effect"]
    )
    df_f["expected_signed_transfer_num"] = _to_float_series(
        df_f["expected_signed_transfer_effect"]
    )
    df_f["moved_expected_bool"] = df_f["moved_in_expected_direction"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df_f["delta_margin_num"] = _to_float_series(df_f["delta_margin"])
    df_f["original_neutral_margin_num"] = _to_float_series(df_f["original_neutral_margin"])
    df_f["original_condition_margin_num"] = _to_float_series(df_f["original_condition_margin"])

    for (fid, cond, anchor, ptype), grp in df_f.groupby(grp_cols):
        delta_margin = float(grp["delta_margin_num"].iloc[0])
        orig_N = float(grp["original_neutral_margin_num"].iloc[0])
        orig_C = float(grp["original_condition_margin_num"].iloc[0])
        for band_name, band_layers in LAYER_BANDS.items():
            sub = grp[grp["layer"].isin(band_layers)]
            if len(sub) == 0:
                continue
            effects = sub["patch_effect_num"].dropna().values
            if ptype == "rescue":
                signed = sub["expected_signed_rescue_num"].dropna().values
            else:
                signed = sub["expected_signed_transfer_num"].dropna().values
            moved_series = sub["moved_expected_bool"].dropna()
            n_layers = len(band_layers)
            rows.append({
                "family_id": fid,
                "condition": cond,
                "anchor": anchor,
                "patch_type": ptype,
                "layer_band": band_name,
                "n_layers_in_band_present": int(sub["layer"].nunique()),
                "n_layers_band_total": n_layers,
                "mean_patch_effect": float(np.mean(effects)) if len(effects) else float("nan"),
                "median_patch_effect": float(np.median(effects)) if len(effects) else float("nan"),
                "mean_expected_signed_effect": (
                    float(np.mean(signed)) if len(signed) else float("nan")
                ),
                "moved_in_expected_direction_fraction": (
                    float(moved_series.mean()) if len(moved_series) else float("nan")
                ),
                "delta_margin": delta_margin,
                "degradation_score": -delta_margin,
                "original_neutral_margin": orig_N,
                "original_condition_margin": orig_C,
            })
    out = pd.DataFrame(rows)
    out.to_csv(FAMILY_LEVEL_CSV, index=False)
    print(
        f"[analysis-family-level] wrote {FAMILY_LEVEL_CSV} "
        f"({len(out)} rows, layer_bands = {list(LAYER_BANDS)})"
    )
    return out


def answer_primary_questions(family_df: pd.DataFrame) -> None:
    """Print Q1-Q4 summaries to stdout and write a summary CSV."""
    if len(family_df) == 0:
        print("[analysis-q1-q4] family_df empty; skipping primary questions.")
        return
    # Use the main layer band only.
    main_band = "late_layers_L28_L35"
    main = family_df[family_df["layer_band"] == main_band].copy()
    if len(main) == 0:
        main_band = family_df["layer_band"].iloc[0]
        main = family_df[family_df["layer_band"] == main_band].copy()
        print(f"[analysis-q1-q4] {main_band!r} not present; using {main_band!r}")

    lines: List[str] = []
    lines.append("# HQ80 S3/S4 Activation Patching — Primary Analysis")
    lines.append("")
    lines.append(f"Generated: {utc_now()}")
    lines.append(f"Primary layer band: `{main_band}` ({LAYER_BANDS.get(main_band, [])})")
    lines.append("")

    # --- Q1 & Q2: rescue / transfer for false-pressure conditions ---
    for qname, cond_set, patch_t, sign_pred in [
        (
            "Q1 (rescue false-pressure): rescue_effect > 0 for negative-delta cases",
            FALSE_PRESSURE, "rescue", +1,
        ),
        (
            "Q2 (transfer false-pressure): transfer_effect < 0 for negative-delta cases",
            FALSE_PRESSURE, "transfer", -1,
        ),
    ]:
        lines.append(f"## {qname}")
        lines.append("")
        for anchor in ANCHORS:
            sub = main[
                (main["patch_type"] == patch_t)
                & (main["condition"].isin(cond_set))
                & (main["anchor"] == anchor)
            ].copy()
            if len(sub) == 0:
                lines.append(f"- **anchor {anchor}**: no rows")
                continue
            vals = sub["mean_patch_effect"].dropna().values
            # Only negative-delta (harmful cases) are truly expected to be
            # sign-correct. But the spec asks across the false-pressure set.
            # Direction alignment: predicted by sign_pred * delta sign.
            def _expected_pos(row) -> bool:
                d = float(row["delta_margin"])
                eff = float(row["mean_patch_effect"])
                if patch_t == "rescue":
                    # harmful (d<0) => rescue should be > 0
                    return eff * (-np.sign(d)) > 0 if d != 0 else False
                else:
                    # transfer harmful (d<0) => transfer_effect < 0
                    return eff * np.sign(d) > 0 if d != 0 else False

            sub["_expected_pos"] = sub.apply(_expected_pos, axis=1)
            mu, lo, hi = _bootstrap_mean_ci(vals, n_boot=10000)
            # one-sample test against zero via sign test (simple count of signs)
            if patch_t == "rescue":
                pred_count = int((sub["mean_patch_effect"] > 0).sum())
            else:
                pred_count = int((sub["mean_patch_effect"] < 0).sum())
            # one-sample Wilcoxon if scipy; otherwise just signs
            from scipy.stats import wilcoxon  # type: ignore
            try:
                finite = vals[np.isfinite(vals)]
                if len(finite) >= 3 and np.std(finite) > 0:
                    w_stat, w_p = wilcoxon(finite - 0.0, alternative="two-sided")
                    w_str = f"Wilcoxon W={w_stat:.1f}, p={w_p:.3e}"
                else:
                    w_str = "Wilcoxon skipped (constant or <3 obs)"
            except Exception:
                w_str = "Wilcoxon skipped"

            lines.append(f"### Anchor `{anchor}`  (n family-condition rows = {len(sub)})")
            lines.append("")
            lines.append(f"- mean   {patch_t}_effect = **{mu:+.4f}** (95% CI [{lo:+.4f}, {hi:+.4f}])")
            lines.append(f"- median {patch_t}_effect = **{float(np.median(vals)):+.4f}**")
            lines.append(
                f"- direction (family-level pred sign count): {pred_count}/{len(sub)} "
                f"({100*pred_count/max(1,len(sub)):.1f}%)"
            )
            lines.append(
                f"- expected_direction_matches_delta_sign: "
                f"{int(sub['_expected_pos'].sum())}/{len(sub)} "
                f"({100*float(sub['_expected_pos'].mean()):.1f}%)"
            )
            lines.append(f"- one-sample vs zero: {w_str}")
            lines.append("")
            # Per-condition breakdown
            lines.append("| condition | n | mean_effect | median_effect | frac_predicted_direction |")
            lines.append("|---|---:|---:|---:|---:|")
            for c in cond_set:
                ss = sub[sub["condition"] == c]
                if len(ss) == 0:
                    continue
                vs = ss["mean_patch_effect"].dropna().values
                if patch_t == "rescue":
                    frac_pos = float((vs > 0).mean())
                else:
                    frac_pos = float((vs < 0).mean())
                lines.append(
                    f"| {c} | {len(ss)} | {float(np.mean(vs)):+.4f} | "
                    f"{float(np.median(vs)):+.4f} | {frac_pos:.2f} |"
                )
            lines.append("")

    # --- Q3: S3 vs S4 comparison ---
    lines.append("## Q3 (S3 already causal? S3 vs S4)")
    lines.append("")
    for ptype in ("rescue", "transfer"):
        sub3 = main[
            (main["patch_type"] == ptype) & (main["anchor"] == "S3")
            & (main["condition"].isin(FALSE_PRESSURE))
        ]
        sub4 = main[
            (main["patch_type"] == ptype) & (main["anchor"] == "S4")
            & (main["condition"].isin(FALSE_PRESSURE))
        ]
        if len(sub3) == 0 or len(sub4) == 0:
            lines.append(f"- {ptype}: missing S3 or S4 rows")
            continue
        # Merge on (family_id, condition)
        merged = sub3.merge(
            sub4, on=["family_id", "condition"], suffixes=("_S3", "_S4"),
        )
        m3 = sub3["mean_patch_effect"].dropna().values.mean()
        m4 = sub4["mean_patch_effect"].dropna().values.mean()
        mu3, lo3, hi3 = _bootstrap_mean_ci(sub3["mean_patch_effect"].dropna().values)
        mu4, lo4, hi4 = _bootstrap_mean_ci(sub4["mean_patch_effect"].dropna().values)
        lines.append(f"### {ptype}")
        lines.append("")
        lines.append(
            f"- S3 mean = **{mu3:+.4f}** [95% CI {lo3:+.4f}, {hi3:+.4f}]  "
            f"(n={len(sub3)} family-condition rows)"
        )
        lines.append(
            f"- S4 mean = **{mu4:+.4f}** [95% CI {lo4:+.4f}, {hi4:+.4f}]  "
            f"(n={len(sub4)} family-condition rows)"
        )
        if len(merged) > 0:
            diff = (merged["mean_patch_effect_S4"] - merged["mean_patch_effect_S3"]).dropna().values
            mu_d, lo_d, hi_d = _bootstrap_mean_ci(diff)
            lines.append(
                f"- paired S4 - S3 = **{mu_d:+.4f}** "
                f"[95% CI {lo_d:+.4f}, {hi_d:+.4f}]  (n_pairs={len(merged)})"
            )
        # expected-direction fractions
        def pred_frac(s: pd.DataFrame) -> float:
            vs = s["mean_patch_effect"].dropna().values
            if ptype == "rescue":
                return float((vs > 0).mean()) if len(vs) else float("nan")
            else:
                return float((vs < 0).mean()) if len(vs) else float("nan")
        lines.append(
            f"- expected direction fraction: S3 = {pred_frac(sub3):.2f}, "
            f"S4 = {pred_frac(sub4):.2f}"
        )
        lines.append("")

    # --- Q4: scaling with delta_margin (regressions) ---
    lines.append("## Q4 (patching effect scales with behavioral delta_margin?)")
    lines.append("")
    lines.append(
        "Across all non-neutral conditions we test:\n\n"
        "- rescue_effect ≈ -delta_margin   (rescue column regressed on -delta)\n"
        "- transfer_effect ≈ delta_margin (transfer column regressed on +delta)\n"
    )
    lines.append("")
    from scipy.stats import pearsonr, spearmanr, linregress  # type: ignore
    lines.append("| anchor | patch_type | layer_band | n | Pearson r | Spearman ρ | slope | intercept | R² | p(r=0) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    all_q4_rows: List[Dict[str, Any]] = []
    for anchor in ANCHORS:
        for ptype in ("rescue", "transfer"):
            for band_name, band_layers in LAYER_BANDS.items():
                sub = family_df[
                    (family_df["anchor"] == anchor)
                    & (family_df["patch_type"] == ptype)
                    & (family_df["layer_band"] == band_name)
                    & (family_df["condition"].isin(ALL_NONNEUTRAL))
                ]
                if len(sub) < 5:
                    continue
                if ptype == "rescue":
                    x = -sub["delta_margin"].astype(float).values
                    y = sub["mean_patch_effect"].astype(float).values
                else:
                    x = sub["delta_margin"].astype(float).values
                    y = sub["mean_patch_effect"].astype(float).values
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if len(x) < 5:
                    continue
                pr, pp = pearsonr(x, y)
                sr, _ = spearmanr(x, y)
                slope, intercept, r, p_lin, _ = linregress(x, y)
                r2 = r * r
                lines.append(
                    f"| {anchor} | {ptype} | {band_name} | {len(x)} | "
                    f"{pr:+.3f} | {sr:+.3f} | {slope:+.3f} | {intercept:+.3f} | "
                    f"{r2:.3f} | {pp:.3e} |"
                )
                all_q4_rows.append({
                    "anchor": anchor, "patch_type": ptype,
                    "layer_band": band_name, "n": len(x),
                    "pearson_r": pr, "spearman_rho": sr,
                    "slope": slope, "intercept": intercept,
                    "r_squared": r2, "p_pearson": pp,
                })
    lines.append("")
    pd.DataFrame(all_q4_rows).to_csv(
        SUM_DIR / "hq80_s3s4_delta_scaling_regressions.csv", index=False,
    )

    # Also try scatter plots if matplotlib importable.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        main_band_name = "late_layers_L28_L35"
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex="row")
        for col, ptype in enumerate(("rescue", "transfer")):
            for row, anchor in enumerate(("S3", "S4")):
                ax = axes[row, col]
                sub = family_df[
                    (family_df["anchor"] == anchor)
                    & (family_df["patch_type"] == ptype)
                    & (family_df["layer_band"] == main_band_name)
                    & (family_df["condition"].isin(ALL_NONNEUTRAL))
                ]
                if len(sub) == 0:
                    continue
                if ptype == "rescue":
                    x = -sub["delta_margin"].astype(float).values
                    xlabel = "−delta_margin (harmful magnitude → right)"
                else:
                    x = sub["delta_margin"].astype(float).values
                    xlabel = "delta_margin (harmful → left)"
                y = sub["mean_patch_effect"].astype(float).values
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if len(x) < 5:
                    continue
                # Color by condition type
                cond_type = []
                for c in sub["condition"].values[mask]:
                    if c in FALSE_PRESSURE:
                        cond_type.append("tab:red")
                    elif c in TRUE_PRESSURE:
                        cond_type.append("tab:green")
                    else:  # distractor
                        cond_type.append("tab:blue")
                ax.scatter(x, y, c=cond_type, alpha=0.55, s=24)
                slope, intercept, *_ = linregress(x, y)
                xx = np.linspace(x.min(), x.max(), 50)
                ax.plot(xx, slope * xx + intercept, "k--", lw=1.2,
                        label=f"fit y = {slope:+.2f}x {intercept:+.2f}")
                ax.axline((0, 0), slope=1.0, color="gray", lw=0.7,
                          linestyle=(0, (5, 5)), label="1:1 reference")
                ax.axhline(0.0, color="k", lw=0.5)
                ax.axvline(0.0, color="k", lw=0.5)
                ax.set_title(f"{ptype} @ {anchor}  (n={len(x)})")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(f"{ptype}_effect (family-level, {main_band_name})")
                ax.legend(fontsize=7)
        fig.suptitle(
            "HQ80 S3/S4 patching: family-level patch effect vs. behavioral delta_margin",
            fontsize=12,
        )
        fig.tight_layout()
        p = PLOT_DIR / "hq80_s3s4_patch_effect_vs_delta_scatter.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[analysis-q4] wrote scatter: {p}")
        p_pdf = PLOT_DIR / "hq80_s3s4_patch_effect_vs_delta_scatter.pdf"
        fig.savefig(p_pdf, bbox_inches="tight")
    except Exception as e:
        print(f"[analysis-q4] scatter plot skipped: {e}")

    # Write combined summary markdown
    (SUM_DIR / "hq80_s3s4_primary_questions_summary.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(
        f"[analysis-q1-q4] wrote primary questions summary: "
        f"{SUM_DIR / 'hq80_s3s4_primary_questions_summary.md'}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        type=str,
        default="0",
        choices=["0", "sanity", "1", "priority", "2", "controls",
                 "3", "remaining", "4", "full", "self", "random"],
        help="Which patching stage to run (resumable).",
    )
    ap.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip token / unpatched-margin / anchor / layer-index validations.",
    )
    ap.add_argument(
        "--validation-only",
        action="store_true",
        help="Run only validations; no patching.",
    )
    ap.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip validations + patching; only re-run family-level summaries "
             "and primary questions on existing raw CSV.",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model", type=str, default=MODEL_NAME)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ensure_dirs()
    device = pick_device(args.device)
    # Per project-memory constraint: patching must use CPU float32 to avoid
    # MPS buffer aliasing bugs.
    force_cpu = True
    if force_cpu and device != "cpu":
        print(
            f"[init] device request was {device}, but HQ80 patching requires "
            "CPU float32 (project hard constraint: no MPS aliasing bugs). "
            "Overriding device = cpu."
        )
        device = "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16

    # ---- Discovered path manifest (always print + save) ----
    paths = {
        "PROMPT_JSONL": str(PROMPT_JSONL),
        "BEHAVIOR_CSV": str(BEHAVIOR_CSV),
        "ANCHOR_MANIFEST_CSV": str(ANCHOR_MANIFEST_CSV),
        "OUT_DIR": str(OUT_DIR),
        "RAW_CSV": str(RAW_CSV),
        "RANDOM_FAMILY_CSV": str(RANDOM_FAMILY_CSV),
        "SELF_PATCH_CSV": str(SELF_PATCH_CSV),
        "UNPATCHED_REPRO_CSV": str(UNPATCHED_REPRO_CSV),
        "ANCHOR_VAL_CSV": str(ANCHOR_VAL_CSV),
        "ANCHOR_VAL_SAMPLES_MD": str(ANCHOR_VAL_SAMPLES_MD),
        "TOKEN_VAL_TXT": str(TOKEN_VAL_TXT),
        "LAYER_IDX_TXT": str(LAYER_IDX_TXT),
        "FAMILY_LEVEL_CSV": str(FAMILY_LEVEL_CSV),
        "CACHE_DIR": str(CACHE_DIR),
    }
    manifest_lines = ["Discovered data / output paths:", "=" * 60]
    for k, v in paths.items():
        manifest_lines.append(f"  {k:30s} = {v}")
    manifest_lines.append(f"\ntime = {utc_now()}")
    manifest = OUT_DIR / "discovered_paths_manifest.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n")
    print("\n".join(manifest_lines))
    print()

    # ---- Load master metadata early (needed for validations + analysis) ---
    master = assemble_master()
    print(
        f"[init] master metadata = {len(master)} rows, "
        f"{master['family_id'].nunique()} families, "
        f"{master['condition'].nunique()} conditions"
    )

    # ---- Analysis-only mode ----
    if args.analysis_only:
        fam_df = family_level_summary()
        answer_primary_questions(fam_df)
        return

    # ---- Load model / tokenizer now (needed for validations and patching) --
    print(f"[model] loading {args.model} (device={device}, dtype={dtype}) ...")
    t0 = time.time()
    model, tokenizer = load_local_model(
        args.model, device=device, dtype=dtype,
    )
    model.eval()
    block_list_name, block_list = find_transformer_block_list(model)
    t1 = time.time()
    print(
        f"[model] loaded in {t1-t0:.1f} s; block_list = {block_list_name} "
        f"n_layers = {len(block_list)}"
    )
    idA, idB, token_strategy = choose_answer_token_ids(tokenizer)
    print(
        f"[tokens] strategy = {token_strategy}  idA = {idA} ({tokenizer.decode([idA])!r})  "
        f"idB = {idB} ({tokenizer.decode([idB])!r})"
    )

    # ---- Validations ----
    if not args.skip_validation:
        print("\n===== VALIDATION 1/4: token ids =====")
        validate_tokens(tokenizer, idA, idB, token_strategy)

        print("\n===== VALIDATION 2/4: unpatched margin reproduction =====")
        validate_unpatched_margins(model, tokenizer, master, device, idA, idB)

        print("\n===== VALIDATION 3/4: anchor positions S3/S4 =====")
        validate_anchors(tokenizer, master, n_sample_families=5)

        print("\n===== VALIDATION 4/4: layer indexing convention =====")
        # Pick a neutral prompt as sample (it has all structure).
        sample_neutral_row = master[master["condition"] == NEUTRAL].iloc[0]
        validate_layer_indexing(
            model, tokenizer, str(sample_neutral_row["prompt_text"]),
            device, PATCH_LAYERS_SANITY,
        )
    else:
        print("[init] --skip-validation: skipping all 4 validations")

    if args.validation_only:
        print("\n[init] --validation-only: exiting before patching.")
        return

    # ---- Stage-based patching ----
    all_fams = all_families(master)
    cfg = stage_config(args.stage, all_fams)
    print(f"\n===== STAGE {args.stage!r}: {cfg['description']} =====")
    # Stages 2-4 build incrementally: we still go through the ResumableCSV
    # so any row already written by a previous stage is skipped.
    run_patching_stage(
        args, model, tokenizer, master, device, idA, idB, token_strategy, cfg,
    )

    # ---- Final summaries (only if main raw CSV has any content) ----
    if RAW_CSV.exists() and os.path.getsize(RAW_CSV) > 100:
        print("\n===== PRIMARY ANALYSIS =====")
        try:
            fam_df = family_level_summary()
            answer_primary_questions(fam_df)
        except Exception as e:
            print(f"[analysis] FAIL: {e}")
    else:
        print("[analysis] raw CSV absent or too small; skip analysis step.")

    print("\nDone.")


if __name__ == "__main__":
    main()
