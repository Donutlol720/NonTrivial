#!/usr/bin/env bash
# ============================================================
# NonTrivial RAG-sycophancy — HQ80 S3/S4 activation patching.
# Migration / resume helper for a NEW device.
#
# Usage on the new machine (after you rsync the repo over):
#
#   chmod +x ./scripts/resume_hq80_patching_on_new_device.sh
#   ./scripts/resume_hq80_patching_on_new_device.sh \
#       --stage 0 \
#       2>&1 | tee -a results/activation_patching_qwen3_4b_hq80_s3s4/_new_device_stage0.log
#
# What this script does:
#   1. Creates a venv if missing and installs all required deps.
#   2. Verifies env (python, pandas, torch, transformers versions).
#   3. Points HF cache to ./model_cache (shared layout with the
#      original Mac; model weights only re-downloaded if you did
#      not rsync the model_cache folder).
#   4. Verifies all 4 prior validations exist on disk (so you can
#      safely pass --skip-validation).
#   5. Launches `python run_activation_patching_hq80_s3s4.py` with
#      the requested --stage, or --analysis-only if given.
# ============================================================
set -euo pipefail

STAGE_FLAG=""
EXTRA_FLAGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage|-s)
      STAGE_FLAG="--stage $2"
      shift 2
      ;;
    --analysis-only)
      EXTRA_FLAGS="$EXTRA_FLAGS --analysis-only"
      shift
      ;;
    --validation-only)
      EXTRA_FLAGS="$EXTRA_FLAGS --validation-only"
      shift
      ;;
    --skip-validation)
      EXTRA_FLAGS="$EXTRA_FLAGS --skip-validation"
      shift
      ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "[resume] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
cd "$REPO_ROOT"

echo "[resume] repo root = $REPO_ROOT"
echo "[resume] uname    = $(uname -a)"
echo "[resume] cwd      = $(pwd)"

# ------------------------------------------------------------
# 1) Venv + deps (idempotent)
# ------------------------------------------------------------
VENV_DIR="$REPO_ROOT/.venv_patching"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[resume] creating fresh venv at $VENV_DIR ..."
  PY3=""
  for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PY3="$cand"; break; fi
  done
  if [[ -z "$PY3" ]]; then
    echo "[resume] ERROR: no python3 found. Install python 3.10+." >&2; exit 3
  fi
  "$PY3" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[resume] python = $(python --version) @ $(which python)"
echo "[resume] pip    = $(which pip)"

PYVER_OK=$(python -c "
import sys
v = sys.version_info
print(1 if (v.major == 3 and v.minor >= 10) else 0)
")
if [[ "$PYVER_OK" != "1" ]]; then
  echo "[resume] ERROR: python >= 3.10 required" >&2; exit 4
fi

# Install core deps (frozen-ish versions match the Mac env)
python -m pip install --upgrade pip setuptools wheel >/dev/null
python -m pip install \
  "torch>=2.5" \
  "transformers>=4.45,<5" \
  "huggingface_hub" \
  "accelerate" \
  "safetensors" \
  "sentencepiece" \
  "protobuf" \
  "pandas>=2.2" \
  "numpy>=1.26,<2.2" \
  "scipy>=1.12" \
  "scikit-learn>=1.5" \
  "matplotlib>=3.9" \
  "tqdm" \
  "psutil" >/dev/null

# ------------------------------------------------------------
# 2) Verify env
# ------------------------------------------------------------
echo ""
echo "[resume] === Environment verification ==="
python - <<'PY'
import sys, importlib
mods = ["torch","pandas","numpy","transformers","scipy","sklearn","matplotlib","tqdm","psutil","safetensors","huggingface_hub","accelerate","sentencepiece"]
ok = True
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  OK  {m:<16s} {v}")
    except Exception as e:
        print(f"  FAIL {m:<16s} {e!r}")
        ok = False
if not ok:
    sys.exit(5)
PY

# ------------------------------------------------------------
# 3) HF cache location (prefer ./model_cache to match Mac)
# ------------------------------------------------------------
export HF_HOME="$REPO_ROOT/model_cache/huggingface"
export TRANSFORMERS_CACHE="$REPO_ROOT/model_cache/huggingface/hub"
export HF_HUB_CACHE="$REPO_ROOT/model_cache/huggingface/hub"
mkdir -p "$TRANSFORMERS_CACHE"
echo ""
echo "[resume] HF_HOME          = $HF_HOME"
echo "[resume] TRANSFORMERS_CACHE = $TRANSFORMERS_CACHE"

# ------------------------------------------------------------
# 4) Verify prior validations present (gate for --skip-validation)
# ------------------------------------------------------------
OUT="$REPO_ROOT/results/activation_patching_qwen3_4b_hq80_s3s4"
VAL_FILES=(
  "$OUT/validation/token_validation.txt"
  "$OUT/validation/unpatched_margin_reproduction.csv"
  "$OUT/validation/anchor_validation.csv"
  "$OUT/validation/layer_indexing_validation.txt"
)
echo ""
echo "[resume] === Validation artifacts check ==="
ALL_VAL=1
for f in "${VAL_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    N=$(wc -l < "$f" 2>/dev/null || echo 0)
    echo "  OK  $(basename "$f")  lines=$N"
  else
    echo "  MISSING $(basename "$f")"
    ALL_VAL=0
  fi
done

if [[ "$ALL_VAL" == "1" ]]; then
  echo "[resume] all validations present — safe to use --skip-validation."
else
  echo "[resume] WARNING: some validations missing — you may need to run --validation-only first."
fi

# Quick summary of current state on disk
echo ""
echo "[resume] === Current run state (on disk) ==="
for csv in \
  "$OUT/hq80_s3s4_activation_patching_raw.csv" \
  "$OUT/random_family_patching_raw.csv" \
  "$OUT/validation/self_patch_control.csv"; do
  if [[ -f "$csv" ]]; then
    N=$(awk 'END{print NR-1}' "$csv")
    echo "  $(basename "$csv") rows = $N"
  else
    echo "  $(basename "$csv") rows = 0 (not yet started)"
  fi
done

# ------------------------------------------------------------
# 5) Launch the pipeline
# ------------------------------------------------------------
echo ""
echo "[resume] === Launching patching script ==="
echo "[resume] cmd: python run_activation_patching_hq80_s3s4.py $STAGE_FLAG $EXTRA_FLAGS"
exec python run_activation_patching_hq80_s3s4.py $STAGE_FLAG $EXTRA_FLAGS
