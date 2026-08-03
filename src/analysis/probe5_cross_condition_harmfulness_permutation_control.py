import argparse
import csv
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

_SRC = REPO_ROOT / "src" / "analysis" / "probe5_cross_condition_generalization.py"
if not _SRC.is_file():
    raise FileNotFoundError(_SRC)
_SPEC = importlib.util.spec_from_file_location("probe5_cross_condition_generalization", _SRC)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load spec for {_SRC}")
probe5 = importlib.util.module_from_spec(_SPEC)
sys.modules["probe5_cross_condition_generalization"] = probe5
_SPEC.loader.exec_module(probe5)

read_jsonl = probe5.read_jsonl
read_family_deltas = probe5.read_family_deltas
collect_dataset = probe5.collect_dataset
label_primary = probe5.label_primary
label_strict = probe5.label_strict
run_pair = probe5.run_pair
CONDITIONS = probe5.CONDITIONS

DEFAULT_INPUT = probe5.DEFAULT_INPUT
DEFAULT_FAMILY_DELTAS = probe5.DEFAULT_FAMILY_DELTAS
DEFAULT_ACTIVATION_ROOT = probe5.DEFAULT_ACTIVATION_ROOT
DEFAULT_LAYERWISE_REFERENCE = "results/probe5_cross_condition_layerwise.csv"
DEFAULT_BEST_REFERENCE = "results/probe5_cross_condition_best_layers.csv"

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'penalty' was deprecated.*",
)
sys.stdout.reconfigure(line_buffering=True)

_rng_master = np.random.default_rng(20260729)


def load_reference_layerwise(path: Path) -> Dict[Tuple[str, str, str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (
                str(row["label_scheme"]),
                str(row["source_condition"]),
                str(row["target_condition"]),
                int(row["layer"]),
            )
            out[key] = dict(row)
    return out


def pairs_all() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for src in CONDITIONS:
        for tgt in CONDITIONS:
            out.append((src, tgt))
    return out


def build_strict_labels(
    all_keys: Sequence[Tuple[str, str]],
    metadata: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[Tuple[str, str], int | None]], Dict[str, float]]:
    labels: Dict[str, Dict[Tuple[str, str], int | None]] = {}
    thresholds: Dict[str, float] = {}
    for cond in CONDITIONS:
        lbl_map, thr, _h, _n = label_strict(all_keys, metadata, cond)
        labels[cond] = lbl_map
        thresholds[cond] = thr
    return labels, thresholds


def strict_labels_flat(
    strict_labels_by_condition: Mapping[str, Mapping[Tuple[str, str], int | None]],
) -> Dict[Tuple[str, str], int | None]:
    out: Dict[Tuple[str, str], int | None] = {}
    for mapping in strict_labels_by_condition.values():
        for k, v in mapping.items():
            out[k] = v
    return out


def best_balanced_accuracy_from_layerwise(
    layerwise_rows: Sequence[Mapping[str, Any]],
) -> Tuple[int, float]:
    if not layerwise_rows:
        return -1, 0.0
    best = max(
        layerwise_rows,
        key=lambda r: float(r.get("balanced_accuracy", 0.0)),
    )
    return int(best["layer"]), float(best["balanced_accuracy"])


def validate_real_scores(
    deltas: Mapping[Tuple[str, str], np.ndarray],
    metadata: Mapping[Tuple[str, str], Mapping[str, Any]],
    strict_thresholds: Mapping[str, float],
    strict_labels_map: Mapping[str, Mapping[Tuple[str, str], int | None]],
    layer_count: int,
    reference_layerwise_path: Path,
) -> Tuple[bool, List[Dict[str, Any]]]:
    reference = load_reference_layerwise(reference_layerwise_path)
    mismatches: List[Dict[str, Any]] = []
    all_layerwise: List[Dict[str, Any]] = []
    for label_scheme in ("primary", "strict"):
        sf = strict_labels_flat(strict_labels_map) if label_scheme == "strict" else {}
        for src, tgt in pairs_all():
            layerwise, _ = run_pair(
                src,
                tgt,
                label_scheme,
                deltas,
                metadata,
                strict_thresholds,
                sf,
                layer_count,
                permute_rng=None,
            )
            all_layerwise.extend(layerwise)
            for row in layerwise:
                key = (
                    str(row["label_scheme"]),
                    str(row["source_condition"]),
                    str(row["target_condition"]),
                    int(row["layer"]),
                )
                ref_row = reference.get(key)
                if ref_row is None:
                    mismatches.append(
                        {"type": "missing_in_reference", "key": key}
                    )
                    continue
                got = float(row["balanced_accuracy"])
                want_s = ref_row.get("balanced_accuracy", "")
                if want_s in (None, ""):
                    mismatches.append({"type": "missing_ba", "key": key})
                    continue
                want = float(want_s)
                if abs(got - want) > 1e-6:
                    mismatches.append(
                        {
                            "type": "ba_mismatch",
                            "key": key,
                            "got": got,
                            "want": want,
                            "abs_diff": abs(got - want),
                        }
                    )
    return (len(mismatches) == 0), mismatches, all_layerwise


def run_permutation_repeat(
    label_scheme: str,
    src: str,
    tgt: str,
    deltas: Mapping[Tuple[str, str], np.ndarray],
    metadata: Mapping[Tuple[str, str], Mapping[str, Any]],
    strict_thresholds: Mapping[str, float],
    strict_labels_map: Mapping[str, Mapping[Tuple[str, str], int | None]],
    layer_count: int,
    repeat_seed: int,
    restrict_layers: set[int] | None = None,
) -> Tuple[int, float]:
    sf = strict_labels_flat(strict_labels_map) if label_scheme == "strict" else {}
    rng = np.random.default_rng(int(repeat_seed))
    layerwise, _ = run_pair(
        src,
        tgt,
        label_scheme,
        deltas,
        metadata,
        strict_thresholds,
        sf,
        layer_count,
        permute_rng=rng,
        restrict_layers=restrict_layers,
    )
    return best_balanced_accuracy_from_layerwise(layerwise)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / DEFAULT_INPUT),
    )
    parser.add_argument(
        "--family-deltas",
        default=str(REPO_ROOT / DEFAULT_FAMILY_DELTAS),
    )
    parser.add_argument(
        "--activation-root",
        default=str(REPO_ROOT / DEFAULT_ACTIVATION_ROOT),
    )
    parser.add_argument(
        "--reference-layerwise",
        default=str(REPO_ROOT / DEFAULT_LAYERWISE_REFERENCE),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--also-100",
        action="store_true",
        help="If set, also run a 100-repeat sweep after the --repeats sweep.",
    )
    parser.add_argument(
        "--best-layer-null",
        action="store_true",
        help=(
            "During permutation repeats, evaluate only the real-data best layer "
            "instead of all 36 layers (~36x faster). The null is then: best-layer "
            "shuffled-label BA vs real best BA at the same layer."
        ),
    )
    parser.add_argument(
        "--max-layer-null",
        action="store_true",
        help=(
            "During permutation repeats, evaluate all 36 layers and record the "
            "maximum balanced accuracy over layers (controls for layer selection. "
            "Slower (~36x). If --best-layer-null and --max-layer-null are both "
            "false, this max-layer-null is used by default when both are false."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip real-score reproduction validation (not recommended).",
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "results" / "probe5_cross_condition_harmfulness_permutation_control.csv"),
    )
    parser.add_argument(
        "--output-summary",
        default=str(REPO_ROOT / "results" / "probe5_cross_condition_harmfulness_permutation_control_summary.txt"),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    family_deltas_path = Path(args.family_deltas)
    activation_root = Path(args.activation_root)
    reference_path = Path(args.reference_layerwise)
    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)

    jsonl_rows = read_jsonl(input_path)
    family_deltas = read_family_deltas(family_deltas_path)
    deltas, metadata, shape = collect_dataset(jsonl_rows, family_deltas, activation_root)
    layer_count = int(shape[0])
    all_keys = list(deltas.keys())
    strict_labels_by_condition, strict_thresholds = build_strict_labels(all_keys, metadata)

    print(json.dumps({
        "status": "loaded",
        "shape": [int(shape[0]), int(shape[1])],
        "n_family_condition_pairs": len(deltas),
        "conditions": list(CONDITIONS),
    }, ensure_ascii=False), flush=True)

    if not args.skip_validation:
        ok, mismatches, control_layerwise = validate_real_scores(
            deltas,
            metadata,
            strict_thresholds,
            strict_labels_by_condition,
            layer_count,
            reference_path,
        )
        validation_path = output_csv.with_name(
            "probe5_cross_condition_harmfulness_permutation_control_real_layerwise.csv"
        )
        if control_layerwise:
            with validation_path.open("w", encoding="utf-8", newline="") as f:
                fieldnames = list(control_layerwise[0].keys())
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in control_layerwise:
                    w.writerow(row)
        else:
            validation_path.write_text("", encoding="utf-8")
        mismatches_path = output_csv.with_name(
            "probe5_cross_condition_harmfulness_permutation_control_validation_mismatches.json"
        )
        with mismatches_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_mismatches": len(mismatches),
                    "mismatches": mismatches,
                    "reference": str(reference_path),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        if not ok:
            first = mismatches[:5]
            raise ValueError(
                f"Real-score validation failed with {len(mismatches)} mismatches vs {reference_path}. "
                f"First: {json.dumps(first, ensure_ascii=False)}. "
                f"Details in {mismatches_path}"
            )
        print(json.dumps({
            "status": "real_score_validation_ok",
            "reference": str(reference_path),
            "mismatches": len(mismatches),
            "real_layerwise_snapshot": str(validation_path),
        }, ensure_ascii=False), flush=True)
    else:
        print(json.dumps({
            "status": "real_score_validation_skipped",
        }, ensure_ascii=False), flush=True)
        control_layerwise = []
        for label_scheme in ("primary", "strict"):
            sf = strict_labels_flat(strict_labels_by_condition) if label_scheme == "strict" else {}
            for src, tgt in pairs_all():
                lw, _ = run_pair(
                    src, tgt, label_scheme, deltas, metadata,
                    strict_thresholds, sf, layer_count, permute_rng=None,
                )
                control_layerwise.extend(lw)

    def run_sweep(
        repeats: int,
        tag: str,
        best_layer_null: bool,
    ) -> List[Dict[str, Any]]:
        result_rows: List[Dict[str, Any]] = []
        for label_scheme in ("primary", "strict"):
            for src, tgt in pairs_all():
                print(json.dumps({
                    "status": f"start_{tag}_pair",
                    "label_scheme": label_scheme,
                    "source_condition": src,
                    "target_condition": tgt,
                    "repeats": repeats,
                    "best_layer_null": best_layer_null,
                }, ensure_ascii=False), flush=True)
                sf = strict_labels_flat(strict_labels_by_condition) if label_scheme == "strict" else {}
                real_lw, _ = run_pair(
                    src, tgt, label_scheme, deltas, metadata,
                    strict_thresholds, sf, layer_count, permute_rng=None,
                )
                real_layer, real_ba = best_balanced_accuracy_from_layerwise(real_lw)
                restrict = {int(real_layer)} if best_layer_null else None
                perm_values: List[float] = []
                base_seed = int(_rng_master.integers(0, 2**60, endpoint=False))
                for r in range(repeats):
                    _best_layer, best_ba = run_permutation_repeat(
                        label_scheme, src, tgt, deltas, metadata,
                        strict_thresholds, strict_labels_by_condition,
                        layer_count, base_seed + r,
                        restrict_layers=restrict,
                    )
                    perm_values.append(best_ba)
                    if repeats <= 20 or (r + 1) % 20 == 0 or (r + 1) == repeats:
                        print(json.dumps({
                            "status": f"progress_{tag}_pair",
                            "label_scheme": label_scheme,
                            "source_condition": src,
                            "target_condition": tgt,
                            "repeat": r + 1,
                            "total": repeats,
                        }, ensure_ascii=False), flush=True)
                perm_mean = float(np.mean(perm_values)) if perm_values else 0.0
                perm_p95 = percentile(perm_values, 95.0)
                p_count = sum(1 for v in perm_values if v >= real_ba)
                emp_p = (p_count + 1) / (len(perm_values) + 1) if perm_values else 1.0
                exceeds_p95 = bool(real_ba > perm_p95)
                result_rows.append({
                    "sweep": tag,
                    "null_type": "best_layer_fixed" if best_layer_null else "max_over_all_36_layers",
                    "label_scheme": label_scheme,
                    "source_condition": src,
                    "target_condition": tgt,
                    "within_or_cross": "within" if src == tgt else "cross",
                    "repeats": repeats,
                    "real_best_layer": int(real_layer),
                    "real_best_balanced_accuracy": f"{real_ba:.6f}",
                    "permute_mean_best_balanced_accuracy": f"{perm_mean:.6f}",
                    "permute_p95_best_balanced_accuracy": f"{perm_p95:.6f}",
                    "empirical_p_value": f"{emp_p:.6f}",
                    "exceeds_p95": "1" if exceeds_p95 else "0",
                })
        return result_rows

    rows_20 = run_sweep(int(args.repeats), "20", bool(args.best_layer_null))
    rows_all = list(rows_20)
    if args.also_100:
        rows_100 = run_sweep(100, "100", bool(args.best_layer_null))
        rows_all.extend(rows_100)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows_all[0].keys()) if rows_all else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_all:
            writer.writerow(row)

    lines: List[str] = []
    lines.append("Probe 5 cross-condition harmfulness permutation-label control")
    lines.append("")
    lines.append(f"hidden_state_shape: ({shape[0]}, {shape[1]})")
    lines.append("evaluation_path: probe5.run_pair (same as original Probe 5 run)")
    lines.append("label_shuffle_scope: training_labels_within_each_leave_one_family_out_fold")
    lines.append(f"source_ref: {reference_path}")
    lines.append("")
    for row in rows_all:
        lines.append(
            f"[{row['sweep']}:{row['label_scheme']}] "
            f"{row['source_condition']} -> {row['target_condition']} "
            f"({row['within_or_cross']}, repeats={row['repeats']}, null_type={row['null_type']}):"
        )
        lines.append(f"  real_best_layer                    = {row['real_best_layer']}")
        lines.append(f"  real_best_balanced_accuracy        = {row['real_best_balanced_accuracy']}")
        lines.append(f"  permute_mean_best_balanced_accuracy = {row['permute_mean_best_balanced_accuracy']}")
        lines.append(f"  permute_p95_best_balanced_accuracy  = {row['permute_p95_best_balanced_accuracy']}")
        lines.append(f"  empirical_p_value                  = {row['empirical_p_value']}")
        lines.append(f"  exceeds_p95                        = {row['exceeds_p95']}")
        lines.append("")
    output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "complete",
        "output_csv": str(output_csv),
        "summary": str(output_summary),
        "n_rows": len(rows_all),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
