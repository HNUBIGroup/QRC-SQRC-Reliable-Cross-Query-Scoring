#!/usr/bin/env python3
"""Label-free QRC/SQRC score correction and held-out DTI metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def ece(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)
    total = len(labels)
    value = 0.0
    for index in range(n_bins):
        mask = bins == index
        if not np.any(mask):
            continue
        value += mask.sum() / total * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(value)


def metric_bundle(frame: pd.DataFrame, score_column: str) -> dict:
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame[score_column].to_numpy(dtype=float)
    probabilities = sigmoid(scores)
    per_query = []
    for _, group in frame.groupby("canonical_smiles", sort=True):
        current_labels = group["label"].to_numpy(dtype=int)
        if len(np.unique(current_labels)) != 2:
            raise ValueError("Held-out query lacks both classes")
        current_scores = group[score_column].to_numpy(dtype=float)
        per_query.append(
            (
                roc_auc_score(current_labels, current_scores),
                average_precision_score(current_labels, current_scores),
            )
        )
    return {
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "pooled_aupr": float(average_precision_score(labels, scores)),
        "drug_macro_auroc": float(np.mean([value[0] for value in per_query])),
        "drug_macro_aupr": float(np.mean([value[1] for value in per_query])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10bin": ece(labels, probabilities, n_bins=10),
    }


def add_qrc(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["delta_raw_base"] = output["raw_logit"] - output["base_logit"]
    output["b_hat"] = output.groupby("canonical_smiles")["delta_raw_base"].transform("mean")
    output["qrc_logit"] = output["raw_logit"] - output["b_hat"]
    output["qrc_prob"] = sigmoid(output["qrc_logit"].to_numpy(dtype=float))
    return output


def estimate_sqrc(validation: pd.DataFrame) -> dict:
    rows = []
    for drug, group in validation.groupby("canonical_smiles", sort=True):
        delta = group["delta_raw_base"].to_numpy(dtype=float)
        sample_variance = float(np.var(delta, ddof=1)) if len(delta) > 1 else 0.0
        rows.append(
            {
                "canonical_smiles": drug,
                "b_hat": float(np.mean(delta)),
                "se2": sample_variance / len(delta),
            }
        )
    query = pd.DataFrame(rows)
    observed_variance = float(np.var(query["b_hat"], ddof=1)) if len(query) > 1 else 0.0
    mean_se2 = float(query["se2"].mean())
    tau_b2 = max(observed_variance - mean_se2, 0.0)
    return {
        "tau_b2": tau_b2,
        "observed_bias_variance": observed_variance,
        "mean_se2": mean_se2,
        "n_validation_queries": int(len(query)),
        "uses_labels": False,
    }


def add_sqrc(frame: pd.DataFrame, tau_b2: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    query_rows = []
    for drug, group in output.groupby("canonical_smiles", sort=True):
        delta = group["delta_raw_base"].to_numpy(dtype=float)
        sample_variance = float(np.var(delta, ddof=1)) if len(delta) > 1 else 0.0
        se2 = sample_variance / len(delta)
        denominator = tau_b2 + se2
        weight = tau_b2 / denominator if denominator > 0 else 0.0
        query_rows.append(
            {
                "canonical_smiles": drug,
                "n_candidates": len(group),
                "b_hat": float(np.mean(delta)),
                "delta_sd": float(np.std(delta, ddof=0)),
                "se2": se2,
                "tau_b2_from_validation": tau_b2,
                "sqrc_weight": weight,
            }
        )
        output.loc[group.index, "sqrc_weight"] = weight
    output["sqrc_logit"] = output["raw_logit"] - output["sqrc_weight"] * output["b_hat"]
    output["sqrc_prob"] = sigmoid(output["sqrc_logit"].to_numpy(dtype=float))
    return output, pd.DataFrame(query_rows)


def rank_invariance_qc(frame: pd.DataFrame, tolerance: float) -> dict:
    violations = 0
    tie_violations = 0
    max_difference_error = 0.0
    for _, group in frame.groupby("canonical_smiles", sort=True):
        raw = group["raw_logit"].to_numpy(dtype=float)
        qrc = group["qrc_logit"].to_numpy(dtype=float)
        upper = np.triu_indices(len(group), 1)
        raw_diff = (raw[:, None] - raw[None, :])[upper]
        qrc_diff = (qrc[:, None] - qrc[None, :])[upper]
        max_difference_error = max(
            max_difference_error, float(np.max(np.abs(raw_diff - qrc_diff), initial=0.0))
        )
        violations += int(np.sum(raw_diff * qrc_diff < 0))
        tie_violations += int(np.sum((raw_diff == 0) != (qrc_diff == 0)))
    raw_metrics = metric_bundle(frame, "raw_logit")
    qrc_metrics = metric_bundle(frame, "qrc_logit")
    macro_difference = max(
        abs(raw_metrics["drug_macro_auroc"] - qrc_metrics["drug_macro_auroc"]),
        abs(raw_metrics["drug_macro_aupr"] - qrc_metrics["drug_macro_aupr"]),
    )
    status = (
        "PASS"
        if violations == 0 and tie_violations == 0 and macro_difference <= tolerance
        else "FAIL"
    )
    return {
        "status": status,
        "strict_order_violations": violations,
        "tie_violations": tie_violations,
        "max_pairwise_difference_error": max_difference_error,
        "max_macro_metric_difference": macro_difference,
        "tolerance": tolerance,
    }


def cross_query_inversion_rate(frame: pd.DataFrame, candidate_column: str) -> dict:
    groups = [group for _, group in frame.groupby("canonical_smiles", sort=True)]
    inversions = 0
    comparable = 0
    induced_ties = 0
    for left_index in range(len(groups)):
        left = groups[left_index]
        left_base = left["base_logit"].to_numpy(dtype=float)
        left_candidate = left[candidate_column].to_numpy(dtype=float)
        for right in groups[left_index + 1 :]:
            base_diff = left_base[:, None] - right["base_logit"].to_numpy(dtype=float)[None, :]
            candidate_diff = (
                left_candidate[:, None]
                - right[candidate_column].to_numpy(dtype=float)[None, :]
            )
            non_tied_base = base_diff != 0
            comparable += int(non_tied_base.sum())
            inversions += int(np.sum(non_tied_base & (base_diff * candidate_diff < 0)))
            induced_ties += int(np.sum(non_tied_base & (candidate_diff == 0)))
    return {
        "inversions": inversions,
        "comparable_cross_query_pairs": comparable,
        "induced_ties": induced_ties,
        "rate": inversions / comparable if comparable else math.nan,
    }


def distribution_rows(frame: pd.DataFrame, seed: int) -> list[dict]:
    records = []
    for method, column in (
        ("Base", "base_logit"),
        ("Raw", "raw_logit"),
        ("QRC", "qrc_logit"),
        ("SQRC", "sqrc_logit"),
    ):
        values = frame[column].to_numpy(dtype=float)
        records.append(
            {
                "seed": seed,
                "method": method,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "q01": float(np.quantile(values, 0.01)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    validation = pd.read_csv(args.validation_predictions.resolve())
    test = pd.read_csv(args.test_predictions.resolve())
    required = {
        "pair_id", "canonical_smiles", "target_id", "label", "base_logit",
        "aux_logit", "fusion_logit", "raw_logit",
    }
    for name, frame in (("validation", validation), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} export missing columns: {sorted(missing)}")
        if frame["pair_id"].duplicated().any():
            raise ValueError(f"{name} export has duplicate pairs")
        if set(frame["seed"]) != {args.seed}:
            raise ValueError(f"{name} export seed mismatch")

    with args.split_manifest.resolve().open() as handle:
        split_manifest = json.load(handle)
    expected_validation = len(split_manifest["drugs"]["validation"]) * validation["target_id"].nunique()
    expected_test = len(split_manifest["drugs"]["test"]) * test["target_id"].nunique()
    if len(validation) != expected_validation or len(test) != expected_test:
        raise ValueError(
            f"Export row count mismatch: validation {len(validation)}/{expected_validation}; "
            f"test {len(test)}/{expected_test}"
        )

    validation = add_qrc(validation)
    test = add_qrc(test)
    sqrc_parameters = estimate_sqrc(validation)
    test, query_statistics = add_sqrc(test, sqrc_parameters["tau_b2"])
    qc = rank_invariance_qc(test, args.tolerance)
    if qc["status"] != "PASS":
        raise RuntimeError(f"IMPLEMENTATION_QC_FAIL: {qc}")

    delta = test["delta_raw_base"].to_numpy(dtype=float)
    bias_broadcast = test["b_hat"].to_numpy(dtype=float)
    total_variance = float(np.var(delta, ddof=0))
    between_variance = float(np.var(bias_broadcast, ddof=0))
    shift_ratio = between_variance / total_variance if total_variance > 0 else 0.0
    raw_inversion = cross_query_inversion_rate(test, "raw_logit")
    qrc_inversion = cross_query_inversion_rate(test, "qrc_logit")

    metric_records = []
    for method, column in (
        ("Base", "base_logit"),
        ("Raw", "raw_logit"),
        ("QRC", "qrc_logit"),
        ("SQRC", "sqrc_logit"),
    ):
        record = {
            "seed": args.seed,
            "method": method,
            **metric_bundle(test, column),
            "query_shift_variance_ratio": shift_ratio,
            "raw_vs_base_cross_query_inversion_rate": raw_inversion["rate"],
            "qrc_vs_base_cross_query_inversion_rate": qrc_inversion["rate"],
            "macro_invariance_qc": qc["status"],
        }
        metric_records.append(record)

    run_subdir = Path("smoke") if args.run_kind == "smoke" else Path()
    result_dir = args.output_root.resolve() / "results" / run_subdir
    qc_dir = args.output_root.resolve() / "qc" / run_subdir
    prediction_dir = args.output_root.resolve() / "predictions" / run_subdir
    for path in (result_dir, qc_dir, prediction_dir):
        path.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / f"seed_{args.seed}_metrics.csv"
    query_path = result_dir / f"seed_{args.seed}_query_shift_statistics.csv"
    distribution_path = result_dir / f"seed_{args.seed}_score_distributions.csv"
    qrc_predictions_path = prediction_dir / f"seed_{args.seed}_test_pairs_qrc.csv"
    pd.DataFrame(metric_records).to_csv(metrics_path, index=False)
    query_statistics.insert(0, "seed", args.seed)
    query_statistics.to_csv(query_path, index=False)
    pd.DataFrame(distribution_rows(test, args.seed)).to_csv(distribution_path, index=False)
    test.to_csv(qrc_predictions_path, index=False)

    qc_payload = {
        "status": "PASS",
        "seed": args.seed,
        "run_kind": args.run_kind,
        "validation_rows": len(validation),
        "test_rows": len(test),
        "score_alignment": "PASS",
        "rank_invariance": qc,
        "sqrc_validation_only_parameters": sqrc_parameters,
        "query_shift_variance": between_variance,
        "total_delta_variance": total_variance,
        "query_shift_variance_ratio": shift_ratio,
        "raw_vs_base_cross_query": raw_inversion,
        "qrc_vs_base_cross_query": qrc_inversion,
        "input_sha256": {
            "validation": sha256_file(args.validation_predictions.resolve()),
            "test": sha256_file(args.test_predictions.resolve()),
            "split_manifest": sha256_file(args.split_manifest.resolve()),
        },
        "output_sha256": {
            "metrics": sha256_file(metrics_path),
            "query_statistics": sha256_file(query_path),
            "score_distributions": sha256_file(distribution_path),
            "qrc_predictions": sha256_file(qrc_predictions_path),
        },
    }
    with (qc_dir / f"seed_{args.seed}_qrc_qc.json").open("w") as handle:
        json.dump(qc_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(qc_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
