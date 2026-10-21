#!/usr/bin/env python3
"""Aggregate the frozen 125-case post-processing outputs without inferential tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRICS = [
    "pooled_auroc", "pooled_aupr", "macro_auroc", "macro_aupr",
    "brier", "ece_10bin", "nll",
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=here / "00_admin/FORMAL_FROZEN_INPUT_MANIFEST_125CASE.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    root = manifest_path.parents[1]
    formal = root / "03_outputs/formal"
    metric_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    platt_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for case in manifest["cases"]:
        case_dir = formal / case["task"] / case["setting"] / f"seed_{case['seed']}"
        required = [
            case_dir / "metrics.csv", case_dir / "qc.json", case_dir / "platt_parameters.json",
            case_dir / "diagnostic_summary.json", case_dir / "input_manifest_snapshot.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            qc_rows.append({
                "case_id": case["case_id"], "task": case["task"], "setting": case["setting"],
                "seed": case["seed"], "status": "BLOCKED", "blocker": f"MISSING_OUTPUTS: {missing}",
            })
            continue
        qc = read_json(case_dir / "qc.json")
        qc_row: dict[str, Any] = {
            "case_id": case["case_id"], "task": case["task"], "setting": case["setting"],
            "seed": case["seed"], "status": qc.get("status"), "blocker": qc.get("blocker"),
        }
        if qc.get("historical_parity"):
            structure = qc["structure_and_alignment"]
            qc_row.update({
                "historical_parity_status": qc["historical_parity"]["status"],
                "historical_parity_max_abs_error": qc["historical_parity"]["max_abs_error"],
                "test_rows": structure["test_rows"], "query_count": structure["query_count"],
                "row_count_match": structure["row_count_match"],
                "candidate_alignment": structure["candidate_alignment"],
                "label_alignment": structure["label_alignment"],
                "raw_qrc_macro_invariance_error": qc["raw_qrc_macro_invariance_error"],
                "platt_status": qc["platt_status"],
                "platt_nonpositive_retained_by_frozen_policy": qc["platt_nonpositive_retained_by_frozen_policy"],
            })
            for method in ("RawCenter", "QRC", "SQRC", "RawZScore", "MedianResidual"):
                inv = qc["ranking_invariance"][method]
                prefix = method.lower()
                qc_row[f"{prefix}_strict_order_violations"] = inv["strict_order_violations"]
                qc_row[f"{prefix}_tie_violations"] = inv["tie_violations"]
                qc_row[f"{prefix}_max_pairwise_difference_error"] = inv["max_pairwise_difference_error"]
                qc_row[f"{prefix}_max_macro_metric_difference"] = inv["max_macro_metric_difference"]
        qc_rows.append(qc_row)
        if qc.get("status") != "PASS":
            continue

        metrics = pd.read_csv(case_dir / "metrics.csv")
        metrics.insert(0, "case_id", case["case_id"])
        metric_frames.append(metrics)

        diagnostic = read_json(case_dir / "diagnostic_summary.json")
        diagnostic_rows.append({
            "case_id": case["case_id"], "task": case["task"], "setting": case["setting"],
            "seed": case["seed"], **diagnostic,
        })

        platt = read_json(case_dir / "platt_parameters.json")
        monotonic = platt.pop("monotonic_invariance", {})
        platt_rows.append({
            "case_id": case["case_id"], "task": case["task"], "setting": case["setting"],
            "seed": case["seed"], **platt,
            **{f"monotonic_{key}": value for key, value in monotonic.items()},
        })

    all_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    all_diagnostics = pd.DataFrame(diagnostic_rows)
    all_platt = pd.DataFrame(platt_rows)
    all_qc = pd.DataFrame(qc_rows)
    all_metrics.to_csv(formal / "FORMAL_ALL_CASE_METRICS.csv", index=False, float_format="%.17g")
    all_diagnostics.to_csv(formal / "FORMAL_ALL_CASE_DIAGNOSTICS.csv", index=False, float_format="%.17g")
    all_platt.to_csv(formal / "FORMAL_PLATT_PARAMETERS.csv", index=False, float_format="%.17g")
    all_qc.to_csv(formal / "FORMAL_QC_SUMMARY.csv", index=False, float_format="%.17g")

    aggregate_rows = []
    if not all_metrics.empty:
        for (task, setting, method), group in all_metrics.groupby(["task", "setting", "method"], sort=True):
            row: dict[str, Any] = {
                "task": task, "setting": setting, "method": method, "n_cases": int(len(group)),
            }
            for metric in METRICS:
                values = group[metric].dropna().astype(float)
                row[f"{metric}_n"] = int(len(values))
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            aggregate_rows.append(row)
    aggregates = pd.DataFrame(aggregate_rows)
    aggregates.to_csv(formal / "FORMAL_SETTING_AGGREGATES.csv", index=False, float_format="%.17g")

    slope_rows = []
    if not all_platt.empty:
        for (task, setting), group in all_platt.groupby(["task", "setting"], sort=True):
            slope = group["a"].astype(float)
            slope_rows.append({
                "task": task, "setting": setting, "n_cases": int(len(group)),
                "positive_slope_count": int((slope > 0).sum()),
                "zero_slope_count": int((slope == 0).sum()),
                "negative_slope_count": int((slope < 0).sum()),
                "slope_mean": float(slope.mean()), "slope_std": float(slope.std(ddof=1)),
                "slope_min": float(slope.min()), "slope_max": float(slope.max()),
            })
    pd.DataFrame(slope_rows).to_csv(formal / "FORMAL_PLATT_SLOPE_SUMMARY.csv", index=False, float_format="%.17g")

    paired_rows = []
    qrc_summary_rows = []
    if not all_metrics.empty:
        for (task, setting), group in all_metrics.groupby(["task", "setting"], sort=True):
            pivot = group.pivot(index="seed", columns="method", values=["pooled_auroc", "pooled_aupr"])
            current = []
            for seed in pivot.index:
                row = {"task": task, "setting": setting, "seed": int(seed)}
                for metric in ("pooled_auroc", "pooled_aupr"):
                    difference = float(pivot.loc[seed, (metric, "QRC")] - pivot.loc[seed, (metric, "RawCenter")])
                    row[f"qrc_minus_rawcenter_{metric}"] = difference
                paired_rows.append(row)
                current.append(row)
            current_frame = pd.DataFrame(current)
            qrc_summary_rows.append({
                "task": task, "setting": setting, "n_seeds": len(current_frame),
                "qrc_minus_rawcenter_pooled_auroc_mean": current_frame["qrc_minus_rawcenter_pooled_auroc"].mean(),
                "qrc_minus_rawcenter_pooled_auroc_std": current_frame["qrc_minus_rawcenter_pooled_auroc"].std(ddof=1),
                "qrc_pooled_auroc_win_count": int((current_frame["qrc_minus_rawcenter_pooled_auroc"] > 0).sum()),
                "qrc_minus_rawcenter_pooled_aupr_mean": current_frame["qrc_minus_rawcenter_pooled_aupr"].mean(),
                "qrc_minus_rawcenter_pooled_aupr_std": current_frame["qrc_minus_rawcenter_pooled_aupr"].std(ddof=1),
                "qrc_pooled_aupr_win_count": int((current_frame["qrc_minus_rawcenter_pooled_aupr"] > 0).sum()),
            })
    pd.DataFrame(paired_rows).to_csv(
        formal / "FORMAL_QRC_VS_RAWCENTER_PAIRED_DIFFERENCES.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(qrc_summary_rows).to_csv(
        formal / "FORMAL_QRC_VS_RAWCENTER_SUMMARY.csv", index=False, float_format="%.17g"
    )

    robustness_pairs = []
    robustness_summary = []
    if not all_metrics.empty:
        for (task, setting), group in all_metrics.groupby(["task", "setting"], sort=True):
            pivot = group.pivot(index="seed", columns="method", values=[
                "pooled_auroc", "pooled_aupr", "macro_auroc", "macro_aupr"
            ])
            for method in ("RawZScore", "MedianResidual"):
                current = []
                for seed in pivot.index:
                    row = {"task": task, "setting": setting, "seed": int(seed), "method": method}
                    for metric in ("pooled_auroc", "pooled_aupr"):
                        row[f"{metric}"] = float(pivot.loc[seed, (metric, method)])
                        row[f"minus_qrc_{metric}"] = float(
                            pivot.loc[seed, (metric, method)] - pivot.loc[seed, (metric, "QRC")]
                        )
                    row["macro_auroc_minus_raw"] = float(
                        pivot.loc[seed, ("macro_auroc", method)] - pivot.loc[seed, ("macro_auroc", "Raw")]
                    )
                    row["macro_aupr_minus_raw"] = float(
                        pivot.loc[seed, ("macro_aupr", method)] - pivot.loc[seed, ("macro_aupr", "Raw")]
                    )
                    robustness_pairs.append(row)
                    current.append(row)
                current_frame = pd.DataFrame(current)
                robustness_summary.append({
                    "task": task, "setting": setting, "method": method, "n_seeds": len(current_frame),
                    "pooled_auroc_mean": current_frame["pooled_auroc"].mean(),
                    "pooled_auroc_std": current_frame["pooled_auroc"].std(ddof=1),
                    "pooled_aupr_mean": current_frame["pooled_aupr"].mean(),
                    "pooled_aupr_std": current_frame["pooled_aupr"].std(ddof=1),
                    "minus_qrc_pooled_auroc_mean": current_frame["minus_qrc_pooled_auroc"].mean(),
                    "minus_qrc_pooled_auroc_std": current_frame["minus_qrc_pooled_auroc"].std(ddof=1),
                    "minus_qrc_pooled_aupr_mean": current_frame["minus_qrc_pooled_aupr"].mean(),
                    "minus_qrc_pooled_aupr_std": current_frame["minus_qrc_pooled_aupr"].std(ddof=1),
                    "max_abs_macro_auroc_difference_vs_raw": current_frame["macro_auroc_minus_raw"].abs().max(),
                    "max_abs_macro_aupr_difference_vs_raw": current_frame["macro_aupr_minus_raw"].abs().max(),
                })
    pd.DataFrame(robustness_pairs).to_csv(
        formal / "FORMAL_ROBUSTNESS_PAIRED_DIFFERENCES.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(robustness_summary).to_csv(
        formal / "FORMAL_ROBUSTNESS_SUMMARY.csv", index=False, float_format="%.17g"
    )

    status_counts = all_qc["status"].value_counts().to_dict() if not all_qc.empty else {}
    summary = {
        "expected_cases": int(manifest["case_count"]),
        "qc_rows": int(len(all_qc)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "final_status": "FORMAL_POSTPROCESS_COMPLETE" if status_counts == {"PASS": 125} else "PARTIAL_PARITY_FAILURE",
        "inferential_statistics_performed": False,
    }
    (formal / "FORMAL_AGGREGATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (formal / "FORMAL_RUN_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["final_status"] == "FORMAL_POSTPROCESS_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
