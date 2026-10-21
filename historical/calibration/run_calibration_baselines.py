#!/usr/bin/env python3
"""Frozen five-case CPU-only QRC/SQRC baseline smoke and parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Prevent accidental accelerator use before importing numerical libraries.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score


METHOD_ORDER = [
    "Base", "Raw", "RawCenter", "QRC", "SQRC",
    "RawZScore", "MedianResidual", "Global Platt(Raw)",
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=here / "00_admin" / "FROZEN_INPUT_MANIFEST.json",
    )
    parser.add_argument("--case", action="append", default=[])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    pos = values >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-values[pos]))
    exp_value = np.exp(values[~pos])
    out[~pos] = exp_value / (1.0 + exp_value)
    return out


def ece(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)
    value = 0.0
    for index in range(n_bins):
        mask = bins == index
        if np.any(mask):
            value += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return float(value)


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    eps = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece_10bin": ece(labels, probabilities, n_bins=10),
        "nll": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
    }


def ranking_metrics(frame: pd.DataFrame, score: np.ndarray, query_col: str) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=int)
    score = np.asarray(score, dtype=float)
    pooled_auc = float(roc_auc_score(labels, score))
    pooled_aupr = float(average_precision_score(labels, score))
    macro_auc: list[float] = []
    macro_aupr: list[float] = []
    work = frame[[query_col, "label"]].copy()
    work["_score"] = score
    for _, group in work.groupby(query_col, sort=True):
        current_labels = group["label"].to_numpy(dtype=int)
        if len(np.unique(current_labels)) < 2:
            continue
        current_score = group["_score"].to_numpy(dtype=float)
        macro_auc.append(float(roc_auc_score(current_labels, current_score)))
        macro_aupr.append(float(average_precision_score(current_labels, current_score)))
    return {
        "pooled_auroc": pooled_auc,
        "pooled_aupr": pooled_aupr,
        "macro_auroc": float(np.mean(macro_auc)) if macro_auc else math.nan,
        "macro_aupr": float(np.mean(macro_aupr)) if macro_aupr else math.nan,
        "macro_queries_used": len(macro_auc),
    }


def pair_key(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    return frame[left].astype(str) + "\x1f" + frame[right].astype(str)


def read_first_column(path: Path) -> set[str]:
    frame = pd.read_csv(path, sep="\t")
    return set(frame.iloc[:, 0].astype(str))


def input_snapshot(case: dict[str, Any]) -> dict[str, Any]:
    inputs = {}
    for name, entry in case["inputs"].items():
        path = Path(entry["path"])
        actual = sha256_file(path) if path.is_file() else None
        inputs[name] = {
            "path": str(path),
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual,
            "hash_match": actual == entry["sha256"],
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "setting": case["setting"],
        "seed": case["seed"],
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "uses_test_labels_for_fitting": False,
        "inputs": inputs,
    }


def verify_hashes(snapshot: dict[str, Any]) -> None:
    failed = [name for name, value in snapshot["inputs"].items() if not value["hash_match"]]
    if failed:
        raise RuntimeError(f"FROZEN_INPUT_HASH_MISMATCH: {failed}")


def residual_query_stats(frame: pd.DataFrame, query_col: str) -> pd.DataFrame:
    rows = []
    for query, group in frame.groupby(query_col, sort=True):
        delta = group["delta"].to_numpy(dtype=float)
        n = len(delta)
        sample_variance = float(np.var(delta, ddof=1)) if n > 1 else 0.0
        rows.append({
            "query": query,
            "n": n,
            "b_hat": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "se2": sample_variance / n,
        })
    return pd.DataFrame(rows)


def add_frozen_qrc_sqrc(
    validation: pd.DataFrame, test: pd.DataFrame, query_col: str,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    validation = validation.copy()
    test = test.copy()
    validation["delta"] = validation["Raw"] - validation["Base"]
    test["delta"] = test["Raw"] - test["Base"]
    validation_stats = residual_query_stats(validation, query_col)
    observed = float(validation_stats["b_hat"].var(ddof=1)) if len(validation_stats) > 1 else 0.0
    mean_se2 = float(validation_stats["se2"].mean())
    tau2 = max(observed - mean_se2, 0.0)
    test_stats = residual_query_stats(test, query_col)
    b_map = dict(zip(test_stats["query"], test_stats["b_hat"]))
    median_map = dict(zip(test_stats["query"], test_stats["median"]))
    se2_map = dict(zip(test_stats["query"], test_stats["se2"]))
    b = test[query_col].map(b_map).to_numpy(dtype=float)
    med = test[query_col].map(median_map).to_numpy(dtype=float)
    se2 = test[query_col].map(se2_map).to_numpy(dtype=float)
    denominator = tau2 + se2
    weight = np.divide(tau2, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    raw = test["Raw"].to_numpy(dtype=float)
    test["b_hat"] = b
    test["QRC"] = raw - b
    test["SQRC"] = raw - weight * b
    test["MedianResidual"] = raw - med
    test["sqrc_weight"] = weight
    params = {
        "tau2": tau2,
        "observed_bias_variance": observed,
        "mean_validation_se2": mean_se2,
        "validation_query_count": int(len(validation_stats)),
        "uses_validation_labels": False,
    }
    return test, params, test_stats


def load_dti(case: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = case["inputs"]
    validation_raw = pd.read_csv(inputs["validation_scores"]["path"])
    test_raw = pd.read_csv(inputs["test_scores"]["path"])
    historical_qrc = pd.read_csv(inputs["historical_qrc_scores"]["path"])
    with Path(inputs["split_manifest"]["path"]).open(encoding="utf-8") as handle:
        split = json.load(handle)
    required = {"pair_id", "canonical_smiles", "target_id", "label", "base_logit", "raw_logit"}
    for name, frame in (("validation", validation_raw), ("test", test_raw), ("historical_qrc", historical_qrc)):
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{name.upper()}_MISSING_COLUMNS: {sorted(missing)}")
        if frame["pair_id"].duplicated().any():
            raise RuntimeError(f"{name.upper()}_DUPLICATE_PAIR_ID")
    expected_validation = len(split["drugs"]["validation"]) * validation_raw["target_id"].nunique()
    expected_test = len(split["drugs"]["test"]) * test_raw["target_id"].nunique()
    aligned = historical_qrc.set_index("pair_id").loc[test_raw["pair_id"]]
    candidate_alignment = (
        set(test_raw["pair_id"]) == set(historical_qrc["pair_id"])
        and np.array_equal(aligned["target_id"].astype(str).to_numpy(), test_raw["target_id"].astype(str).to_numpy())
    )
    label_alignment = np.array_equal(aligned["label"].to_numpy(dtype=int), test_raw["label"].to_numpy(dtype=int))
    validation = validation_raw.rename(columns={"base_logit": "Base", "raw_logit": "Raw", "label": "label"})
    test = test_raw.rename(columns={"base_logit": "Base", "raw_logit": "Raw", "label": "label"})
    structure = {
        "validation_rows": len(validation),
        "test_rows": len(test),
        "expected_validation_rows": expected_validation,
        "expected_test_rows": expected_test,
        "query_count": int(test[case["query_column"]].nunique()),
        "candidate_count": int(test[case["candidate_column"]].nunique()),
        "candidate_counts_per_query": sorted(test.groupby(case["query_column"])[case["candidate_column"]].nunique().unique().tolist()),
        "row_count_match": len(validation) == expected_validation and len(test) == expected_test,
        "candidate_alignment": candidate_alignment,
        "label_alignment": label_alignment,
        "historical_pair_score_max_abs_error": float(max(
            np.max(np.abs(aligned["base_logit"].to_numpy(float) - test["Base"].to_numpy(float))),
            np.max(np.abs(aligned["raw_logit"].to_numpy(float) - test["Raw"].to_numpy(float))),
        )),
    }
    if not all((structure["row_count_match"], candidate_alignment, label_alignment)):
        raise RuntimeError(f"DTI_ALIGNMENT_FAIL: {structure}")
    return validation, test, {"structure": structure, "historical_qrc_frame": aligned.reset_index(drop=False)}


def load_mda_random(case: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = case["inputs"]
    x = pd.read_csv(inputs["score_table"]["path"], sep="\t")
    if x[["drug", "miRNA"]].duplicated().any():
        raise RuntimeError("MDA_RANDOM_DUPLICATE_PAIR")
    val_pos = x[x["split"] == "val_known"].copy()
    test_pos = x[x["split"] == "test_known"].copy()
    unlabeled = x[x["split"] == "unlabeled"].copy()
    rng = np.random.default_rng(20000 + int(case["seed"]))
    permutation = rng.permutation(len(unlabeled))
    midpoint = len(permutation) // 2
    val_unlabeled = unlabeled.iloc[permutation[:midpoint]].copy()
    test_unlabeled = unlabeled.iloc[permutation[midpoint:]].copy()
    val_pos["label"], val_unlabeled["label"] = 1, 0
    test_pos["label"], test_unlabeled["label"] = 1, 0
    validation = pd.concat([val_pos, val_unlabeled], ignore_index=True)
    test = pd.concat([test_pos, test_unlabeled], ignore_index=True)
    lambda_table = pd.read_csv(inputs["lambda_table"]["path"], sep="\t")
    lambda_row = lambda_table[lambda_table["seed"].astype(int) == int(case["seed"])]
    if len(lambda_row) != 1:
        raise RuntimeError("RANDOM_FROZEN_LAMBDA_NOT_UNIQUE")
    lam = float(lambda_row.iloc[0]["lambda"])
    for frame in (validation, test):
        frame["Base"] = frame["molecular_score"].to_numpy(float)
        frame["Raw"] = (1.0 - lam) * frame["molecular_score"].to_numpy(float) + lam * frame["unified_score"].to_numpy(float)
    frozen_val_pairs = pd.read_csv(inputs["validation_positive_pairs"]["path"], sep="\t")
    frozen_test_pairs = pd.read_csv(inputs["test_positive_pairs"]["path"], sep="\t")
    candidate_alignment = (
        set(pair_key(val_pos, "drug", "miRNA")) == set(pair_key(frozen_val_pairs, "drug", "miRNA"))
        and set(pair_key(test_pos, "drug", "miRNA")) == set(pair_key(frozen_test_pairs, "drug", "miRNA"))
        and set(pair_key(val_unlabeled, "drug", "miRNA")).isdisjoint(set(pair_key(test_unlabeled, "drug", "miRNA")))
        and len(val_unlabeled) + len(test_unlabeled) == len(unlabeled)
    )
    label_alignment = (
        validation.loc[validation["label"] == 1, "split"].eq("val_known").all()
        and test.loc[test["label"] == 1, "split"].eq("test_known").all()
        and validation.loc[validation["label"] == 0, "split"].eq("unlabeled").all()
        and test.loc[test["label"] == 0, "split"].eq("unlabeled").all()
    )
    structure = {
        "source_rows": len(x),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "query_count": int(test[case["query_column"]].nunique()),
        "candidate_count": int(test[case["candidate_column"]].nunique()),
        "candidate_counts_per_query": sorted(test.groupby(case["query_column"])[case["candidate_column"]].nunique().unique().tolist()),
        "test_positive_count": int(test["label"].sum()),
        "test_unlabeled_count": int((test["label"] == 0).sum()),
        "row_count_match": len(x) == 77675 and len(test) == 37460 and len(validation) == 37460,
        "candidate_alignment": bool(candidate_alignment),
        "label_alignment": bool(label_alignment),
        "random_partition_seed": 20000 + int(case["seed"]),
        "frozen_lambda": lam,
    }
    if not all((structure["row_count_match"], candidate_alignment, label_alignment)):
        raise RuntimeError(f"MDA_RANDOM_ALIGNMENT_FAIL: {structure}")
    return validation, test, {"structure": structure}


def load_mda_entity_cold(case: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = case["inputs"]
    x = pd.read_csv(inputs["score_table"]["path"], sep="\t")
    qcol = case["query_column"]
    ccol = case["candidate_column"]
    validation_entities = read_first_column(Path(inputs["validation_entities"]["path"]))
    test_entities = read_first_column(Path(inputs["test_entities"]["path"]))
    validation = x[x[qcol].astype(str).isin(validation_entities)].copy()
    test = x[x[qcol].astype(str).isin(test_entities)].copy()
    validation["label"] = (validation["split"] == "val_known").astype(int)
    test["label"] = (test["split"] == "test_known").astype(int)
    for frame in (validation, test):
        frame["Base"] = frame["molecular_score"].to_numpy(float)
        frame["Raw"] = frame["final_score"].to_numpy(float)
    full_candidates = set(x[ccol].astype(str))
    test_candidate_sets = test.groupby(qcol)[ccol].apply(lambda values: set(values.astype(str)))
    candidate_alignment = set(test[qcol].astype(str)) == test_entities and all(values == full_candidates for values in test_candidate_sets)
    label_alignment = (
        validation.loc[validation["label"] == 1, "split"].eq("val_known").all()
        and test.loc[test["label"] == 1, "split"].eq("test_known").all()
        and not validation["split"].eq("test_known").any()
        and not test["split"].eq("val_known").any()
    )
    expected_rows = len(test_entities) * len(full_candidates)
    expected_validation_rows = len(validation_entities) * len(full_candidates)
    structure = {
        "source_rows": len(x),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "expected_validation_rows": expected_validation_rows,
        "expected_test_rows": expected_rows,
        "query_count": int(test[qcol].nunique()),
        "candidate_count": int(test[ccol].nunique()),
        "candidate_counts_per_query": sorted(test.groupby(qcol)[ccol].nunique().unique().tolist()),
        "test_positive_count": int(test["label"].sum()),
        "test_unlabeled_count": int((test["label"] == 0).sum()),
        "row_count_match": len(x) == 77675 and len(test) == expected_rows and len(validation) == expected_validation_rows,
        "candidate_alignment": bool(candidate_alignment),
        "label_alignment": bool(label_alignment),
    }
    if not all((structure["row_count_match"], candidate_alignment, label_alignment)):
        raise RuntimeError(f"MDA_ENTITY_COLD_ALIGNMENT_FAIL: {structure}")
    return validation, test, {"structure": structure}


def load_mda_strict_cold(case: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = case["inputs"]
    x = pd.read_csv(inputs["score_table"]["path"], sep="\t")
    validation_drugs = read_first_column(Path(inputs["validation_drugs"]["path"]))
    test_drugs = read_first_column(Path(inputs["test_drugs"]["path"]))
    validation_mirnas = read_first_column(Path(inputs["validation_mirnas"]["path"]))
    test_mirnas = read_first_column(Path(inputs["test_mirnas"]["path"]))
    validation = x[x["drug"].astype(str).isin(validation_drugs) & x["miRNA"].astype(str).isin(validation_mirnas)].copy()
    test = x[x["drug"].astype(str).isin(test_drugs) & x["miRNA"].astype(str).isin(test_mirnas)].copy()
    validation["label"] = (validation["split"] == "val_known").astype(int)
    test["label"] = (test["split"] == "test_known").astype(int)
    for frame in (validation, test):
        frame["Base"] = frame["molecular_score"].to_numpy(float)
        frame["Raw"] = frame["final_score"].to_numpy(float)
    test_candidate_sets = test.groupby("drug")["miRNA"].apply(lambda values: set(values.astype(str)))
    candidate_alignment = set(test["drug"].astype(str)) == test_drugs and all(values == test_mirnas for values in test_candidate_sets)
    label_alignment = (
        validation.loc[validation["label"] == 1, "split"].eq("val_known").all()
        and test.loc[test["label"] == 1, "split"].eq("test_known").all()
        and not validation["split"].eq("test_known").any()
        and not test["split"].eq("val_known").any()
    )
    expected_validation_rows = len(validation_drugs) * len(validation_mirnas)
    expected_test_rows = len(test_drugs) * len(test_mirnas)
    structure = {
        "source_rows": len(x),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "expected_validation_rows": expected_validation_rows,
        "expected_test_rows": expected_test_rows,
        "query_count": int(test["drug"].nunique()),
        "candidate_count": int(test["miRNA"].nunique()),
        "candidate_counts_per_query": sorted(test.groupby("drug")["miRNA"].nunique().unique().tolist()),
        "test_positive_count": int(test["label"].sum()),
        "test_unlabeled_count": int((test["label"] == 0).sum()),
        "row_count_match": len(x) == 77675 and len(validation) == expected_validation_rows and len(test) == expected_test_rows,
        "candidate_alignment": bool(candidate_alignment),
        "label_alignment": bool(label_alignment),
    }
    if not all((structure["row_count_match"], candidate_alignment, label_alignment)):
        raise RuntimeError(f"MDA_STRICT_ALIGNMENT_FAIL: {structure}")
    return validation, test, {"structure": structure}


def load_case(case: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    loaders = {
        "dti": load_dti,
        "mda_random": load_mda_random,
        "mda_entity_cold": load_mda_entity_cold,
        "mda_strict_cold": load_mda_strict_cold,
    }
    return loaders[case["loader"]](case)


def compare_value(checks: list[dict[str, Any]], name: str, actual: float, expected: float) -> None:
    error = abs(float(actual) - float(expected))
    checks.append({"name": name, "actual": float(actual), "expected": float(expected), "abs_error": error})


def historical_parity(
    case: dict[str, Any], test: pd.DataFrame, params: dict[str, float],
    derived_metrics: dict[str, dict[str, Any]], loader_meta: dict[str, Any], tolerance: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    seed = int(case["seed"])
    history_path = case["inputs"]["historical_metrics"]["path"]
    if case["loader"] == "dti":
        history = pd.read_csv(history_path)
        history = history[history["seed"].astype(int) == seed]
        field_map = {
            "pooled_auroc": "pooled_auroc", "pooled_aupr": "pooled_aupr",
            "macro_auroc": "drug_macro_auroc", "macro_aupr": "drug_macro_aupr",
        }
        historical_qrc = loader_meta["historical_qrc_frame"]
        for method in ("Base", "Raw", "QRC", "SQRC"):
            row = history[history["method"] == method]
            if len(row) != 1:
                raise RuntimeError(f"DTI_HISTORICAL_ROW_MISSING: {method}")
            for current, historical in field_map.items():
                compare_value(checks, f"{method}.{current}", derived_metrics[method][current], row.iloc[0][historical])
            hist_prob_col = {"Base": "base_logit", "Raw": "raw_logit", "QRC": "qrc_logit", "SQRC": "sqrc_logit"}[method]
            probabilities = sigmoid(test[method].to_numpy(float))
            calibration = calibration_metrics(test["label"].to_numpy(int), probabilities)
            compare_value(checks, f"{method}.brier", calibration["brier"], row.iloc[0]["brier"])
            compare_value(checks, f"{method}.ece_10bin", calibration["ece_10bin"], row.iloc[0]["ece_10bin"])
            compare_value(
                checks, f"{method}.historical_pair_score",
                float(np.max(np.abs(test[method].to_numpy(float) - historical_qrc[hist_prob_col].to_numpy(float)))), 0.0,
            )
        historical_qc = json.loads(Path(case["inputs"]["historical_qc"]["path"]).read_text(encoding="utf-8"))
        compare_value(checks, "SQRC.tau2", params["tau2"], historical_qc["sqrc_validation_only_parameters"]["tau_b2"])
        delta = test["delta"].to_numpy(float)
        strength = float(np.var(test["b_hat"].to_numpy(float), ddof=0) / np.var(delta, ddof=0))
        compare_value(checks, "QRC.query_shift_variance_ratio", strength, history.iloc[0]["query_shift_variance_ratio"])
    else:
        history = pd.read_csv(history_path, sep="\t")
        if case["loader"] == "mda_entity_cold":
            history = history[(history["mode"] == case["mode"]) & (history["seed"].astype(int) == seed)]
            if len(history) != 1:
                raise RuntimeError("MDA_STAGEE_HISTORICAL_ROW_MISSING")
            row = history.iloc[0]
            for method, prefix in (("Base", "Molecular"), ("Raw", "Raw"), ("QRC", "QRC"), ("SQRC", "SQRC")):
                for current, suffix in (("pooled_auroc", "Pooled_AUROC"), ("pooled_aupr", "Pooled_AUPR"), ("macro_auroc", "Macro_AUROC"), ("macro_aupr", "Macro_AUPR")):
                    compare_value(checks, f"{method}.{current}", derived_metrics[method][current], row[f"{prefix}_{suffix}"])
            compare_value(checks, "SQRC.tau2", params["tau2"], row["tau2"])
            compare_value(checks, "SQRC.mean_weight", float(test["sqrc_weight"].mean()), row["mean_weight"])
        else:
            historical_setting = case["setting"].replace("-", "_")
            history = history[(history["setting"] == historical_setting) & (history["seed"].astype(int) == seed)]
            if len(history) != 1:
                raise RuntimeError("MDA_STAGEG_HISTORICAL_ROW_MISSING")
            row = history.iloc[0]
            for method, prefix in (("Base", "Molecular"), ("Raw", "Raw"), ("QRC", "QRC"), ("SQRC", "SQRC")):
                for current, suffix in (("pooled_auroc", "AUROC"), ("pooled_aupr", "AUPR")):
                    compare_value(checks, f"{method}.{current}", derived_metrics[method][current], row[f"{prefix}_{suffix}"])
            compare_value(checks, "SQRC.tau2", params["tau2"], row["tau2"])
            compare_value(checks, "SQRC.mean_weight", float(test["sqrc_weight"].mean()), row["mean_weight"])
            compare_value(checks, "row_count", len(test), row["n_pairs"])
            compare_value(checks, "positive_count", int(test["label"].sum()), row["n_pos"])
            if case["setting"] == "random":
                compare_value(checks, "frozen_lambda", loader_meta["structure"]["frozen_lambda"], row["lambda"])
    max_error = max((entry["abs_error"] for entry in checks), default=math.inf)
    return {
        "status": "PASS" if max_error <= tolerance else "FAIL",
        "tolerance": tolerance,
        "max_abs_error": max_error,
        "checks": checks,
        "precision_blocker": max_error > tolerance,
    }


def invariance_qc(frame: pd.DataFrame, transformed: np.ndarray, query_col: str, tolerance: float) -> dict[str, Any]:
    transformed = np.asarray(transformed, dtype=float)
    strict_order_violations = 0
    tie_violations = 0
    max_difference_error = 0.0
    work = frame[[query_col, "Raw", "label"]].copy()
    work["_transformed"] = transformed
    for _, group in work.groupby(query_col, sort=True):
        raw = group["Raw"].to_numpy(float)
        current = group["_transformed"].to_numpy(float)
        upper = np.triu_indices(len(group), 1)
        raw_diff = (raw[:, None] - raw[None, :])[upper]
        current_diff = (current[:, None] - current[None, :])[upper]
        max_difference_error = max(max_difference_error, float(np.max(np.abs(raw_diff - current_diff), initial=0.0)))
        strict_order_violations += int(np.sum(raw_diff * current_diff < 0))
        tie_violations += int(np.sum((raw_diff == 0) != (current_diff == 0)))
    raw_metrics = ranking_metrics(frame, frame["Raw"].to_numpy(float), query_col)
    current_metrics = ranking_metrics(frame, transformed, query_col)
    macro_difference = max(
        abs(raw_metrics["macro_auroc"] - current_metrics["macro_auroc"]),
        abs(raw_metrics["macro_aupr"] - current_metrics["macro_aupr"]),
    )
    status = "PASS" if strict_order_violations == 0 and tie_violations == 0 and macro_difference <= tolerance else "FAIL"
    return {
        "status": status,
        "strict_order_violations": strict_order_violations,
        "tie_violations": tie_violations,
        "max_pairwise_difference_error": max_difference_error,
        "max_macro_metric_difference": macro_difference,
        "tolerance": tolerance,
    }


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    center = float(np.mean(scores))
    scale = float(np.std(scores, ddof=0))
    if scale == 0.0:
        raise RuntimeError("ZERO_STD_VALIDATION_RAW_FOR_PLATT")
    standardized = (scores - center) / scale
    prevalence = float(labels.mean())
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise RuntimeError("PLATT_VALIDATION_LACKS_BOTH_CLASSES")
    initial_intercept = float(np.log(prevalence / (1.0 - prevalence)))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = theta[0] * standardized + theta[1]
        value = float(np.sum(np.logaddexp(0.0, linear) - labels * linear))
        residual = sigmoid(linear) - labels
        gradient = np.array([np.dot(residual, standardized), residual.sum()], dtype=float)
        return value, gradient

    result = minimize(
        lambda theta: objective(theta)[0],
        np.array([1.0, initial_intercept], dtype=float),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 100},
    )
    alpha, intercept_standardized = result.x
    slope = float(alpha / scale)
    intercept = float(intercept_standardized - alpha * center / scale)
    parameters = {
        "a": slope,
        "b": intercept,
        "fit_rows": int(len(labels)),
        "positive_count": int(labels.sum()),
        "unlabeled_or_pseudo_negative_count": int((labels == 0).sum()),
        "fitting_source": "frozen_validation_labels" if "dti" else "validation",
        "uses_test_labels": False,
        "optimizer": "scipy.optimize.minimize L-BFGS-B; unregularized binomial NLL",
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "objective_nll_sum": float(result.fun),
        "validation_score_center": center,
        "validation_score_scale": scale,
        "slope_status": "POSITIVE" if slope > 0 else "NON_POSITIVE_PLATT_SLOPE",
    }
    return parameters, result.x


def direction(value: float, tolerance: float = 0.0) -> str:
    if value > tolerance:
        return "UP"
    if value < -tolerance:
        return "DOWN"
    return "TIE"


def discordance(macro_change: float, pooled_change: float) -> str:
    macro = direction(macro_change)
    pooled = direction(pooled_change)
    if macro == "UP" and pooled == "DOWN":
        return "MACRO_UP_POOLED_DOWN"
    if macro == "DOWN" and pooled == "UP":
        return "MACRO_DOWN_POOLED_UP"
    if macro == pooled:
        return "NONE"
    return f"MACRO_{macro}_POOLED_{pooled}"


def empty_outputs(
    output_dir: Path, case: dict[str, Any], snapshot: dict[str, Any], error: str, status: str,
) -> None:
    pd.DataFrame(columns=[
        "task", "setting", "seed", "method", "role", "pooled_auroc", "pooled_aupr",
        "macro_auroc", "macro_aupr", "macro_queries_used", "brier", "ece_10bin", "nll",
        "calibration_probability_source",
    ]).to_csv(output_dir / "metrics.csv", index=False)
    write_json(output_dir / "qc.json", {"status": status, "blocker": error})
    write_json(output_dir / "platt_parameters.json", {
        "status": "NOT_RUN_PARITY_BLOCKED", "uses_test_labels": False, "blocker": error,
    })
    write_json(output_dir / "diagnostic_summary.json", {
        "status": "NOT_COMPUTED_PARITY_BLOCKED", "semantic_identification_of_bN_and_bI": False,
        "blocker": error,
    })
    write_json(output_dir / "input_manifest_snapshot.json", snapshot)


def run_case(case: dict[str, Any], manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    run_kind = manifest.get("run_kind", "smoke")
    output_dir = root / "03_outputs" / run_kind / case["task"] / case["setting"] / f"seed_{case['seed']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = input_snapshot(case)
    write_json(output_dir / "input_manifest_snapshot.json", snapshot)
    try:
        verify_hashes(snapshot)
        validation, test, loader_meta = load_case(case)
        query_col = case["query_column"]
        test, sqrc_parameters, query_stats = add_frozen_qrc_sqrc(validation, test, query_col)
        raw = test["Raw"].to_numpy(float)
        raw_mean = test.groupby(query_col)["Raw"].transform("mean").to_numpy(float)
        raw_std = test.groupby(query_col)["Raw"].transform(lambda values: values.std(ddof=0)).to_numpy(float)
        zero_std_queries = sorted(test.loc[raw_std == 0.0, query_col].astype(str).unique().tolist())
        test["RawCenter"] = raw - raw_mean
        if zero_std_queries:
            test["RawZScore"] = math.nan
        else:
            test["RawZScore"] = (raw - raw_mean) / raw_std

        pre_metrics = {
            method: ranking_metrics(test, test[method].to_numpy(float), query_col)
            for method in ("Base", "Raw", "QRC", "SQRC")
        }
        parity = historical_parity(
            case, test, sqrc_parameters, pre_metrics, loader_meta,
            float(manifest["historical_metric_tolerance"]),
        )
        raw_qrc_macro_error = max(
            abs(pre_metrics["Raw"]["macro_auroc"] - pre_metrics["QRC"]["macro_auroc"]),
            abs(pre_metrics["Raw"]["macro_aupr"] - pre_metrics["QRC"]["macro_aupr"]),
        )
        if raw_qrc_macro_error > float(manifest["ranking_invariance_tolerance"]):
            parity["status"] = "FAIL"
            parity["raw_qrc_macro_invariance_error"] = raw_qrc_macro_error
        if parity["status"] != "PASS":
            raise RuntimeError(f"HISTORICAL_PARITY_FAIL: max_abs_error={parity['max_abs_error']:.17g}")

        if zero_std_queries:
            raise RuntimeError(f"ZERO_STD_QUERY: {zero_std_queries}")

        # New baselines are only evaluated after the complete historical parity gate passes.
        scores = {
            "Base": test["Base"].to_numpy(float),
            "Raw": test["Raw"].to_numpy(float),
            "RawCenter": test["RawCenter"].to_numpy(float),
            "QRC": test["QRC"].to_numpy(float),
            "SQRC": test["SQRC"].to_numpy(float),
            "RawZScore": test["RawZScore"].to_numpy(float),
            "MedianResidual": test["MedianResidual"].to_numpy(float),
        }
        invariance = {
            method: invariance_qc(test, score, query_col, float(manifest["ranking_invariance_tolerance"]))
            for method, score in scores.items() if method not in {"Base", "Raw"}
        }
        required_invariance = ("RawCenter", "QRC", "MedianResidual", "RawZScore")
        if any(invariance[name]["status"] != "PASS" for name in required_invariance):
            raise RuntimeError("RANKING_INVARIANCE_FAIL")

        validation_scores = validation["Raw"].to_numpy(float)
        validation_labels = validation["label"].to_numpy(int)
        platt, _ = fit_platt(validation_scores, validation_labels)
        platt["fitting_source"] = (
            "DTI frozen validation labels" if case["task"].lower() == "dti"
            else "MDA frozen PU validation semantics (known positives=1; unlabeled/pseudo-negatives=0)"
        )
        test_platt_probability = sigmoid(platt["a"] * raw + platt["b"])
        scores["Global Platt(Raw)"] = test_platt_probability
        platt_rank = ranking_metrics(test, test_platt_probability, query_col)
        raw_rank = ranking_metrics(test, raw, query_col)
        platt["monotonic_invariance"] = {
            "pooled_auroc_abs_diff": abs(platt_rank["pooled_auroc"] - raw_rank["pooled_auroc"]),
            "pooled_aupr_abs_diff": abs(platt_rank["pooled_aupr"] - raw_rank["pooled_aupr"]),
            "macro_auroc_abs_diff": abs(platt_rank["macro_auroc"] - raw_rank["macro_auroc"]),
            "macro_aupr_abs_diff": abs(platt_rank["macro_aupr"] - raw_rank["macro_aupr"]),
        }
        platt["status"] = "PASS" if platt["optimizer_success"] and platt["a"] > 0 else (
            "NON_POSITIVE_PLATT_SLOPE" if platt["a"] <= 0 else "OPTIMIZER_NOT_CONVERGED"
        )
        write_json(output_dir / "platt_parameters.json", platt)

        roles = {
            "Base": "main_ranking", "Raw": "main_ranking_and_calibration_control",
            "RawCenter": "main_ranking", "QRC": "main_ranking", "SQRC": "main_ranking",
            "RawZScore": "robustness_appendix", "MedianResidual": "robustness_appendix",
            "Global Platt(Raw)": "calibration_only",
        }
        metric_rows = []
        for method in METHOD_ORDER:
            ranking = ranking_metrics(test, scores[method], query_col)
            row = {
                "task": case["task"], "setting": case["setting"], "seed": case["seed"],
                "method": method, "role": roles[method], **ranking,
                "brier": math.nan, "ece_10bin": math.nan, "nll": math.nan,
                "calibration_probability_source": "not_reported_for_this_method",
            }
            if method == "Raw":
                probability = sigmoid(raw)
                row.update(calibration_metrics(test["label"].to_numpy(int), probability))
                row["calibration_probability_source"] = "sigmoid_identity(a=1,b=0) applied to Raw"
            elif method == "Global Platt(Raw)":
                row.update(calibration_metrics(test["label"].to_numpy(int), test_platt_probability))
                row["calibration_probability_source"] = "validation-fitted sigmoid(a*Raw+b)"
            metric_rows.append(row)
        pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False, float_format="%.17g")

        delta = test["delta"].to_numpy(float)
        total_delta_variance = float(np.var(delta, ddof=0))
        broadcast_variance = float(np.var(test["b_hat"].to_numpy(float), ddof=0))
        query_shift_strength = broadcast_variance / total_delta_variance if total_delta_variance > 0 else 0.0
        all_metrics = {method: ranking_metrics(test, scores[method], query_col) for method in scores}
        changes = {
            "raw_macro_auroc_vs_base": all_metrics["Raw"]["macro_auroc"] - all_metrics["Base"]["macro_auroc"],
            "raw_macro_aupr_vs_base": all_metrics["Raw"]["macro_aupr"] - all_metrics["Base"]["macro_aupr"],
            "raw_pooled_auroc_vs_base": all_metrics["Raw"]["pooled_auroc"] - all_metrics["Base"]["pooled_auroc"],
            "raw_pooled_aupr_vs_base": all_metrics["Raw"]["pooled_aupr"] - all_metrics["Base"]["pooled_aupr"],
            "qrc_pooled_auroc_vs_raw": all_metrics["QRC"]["pooled_auroc"] - all_metrics["Raw"]["pooled_auroc"],
            "qrc_pooled_aupr_vs_raw": all_metrics["QRC"]["pooled_aupr"] - all_metrics["Raw"]["pooled_aupr"],
        }
        diagnostics = {
            "status": "DESCRIPTIVE_ONLY",
            "query_shift_strength": query_shift_strength,
            "query_shift_strength_numerator_var_broadcast_b_hat": broadcast_variance,
            "query_shift_strength_denominator_var_delta": total_delta_variance,
            **changes,
            "macro_pooled_discordance_auroc": discordance(changes["raw_macro_auroc_vs_base"], changes["raw_pooled_auroc_vs_base"]),
            "macro_pooled_discordance_aupr": discordance(changes["raw_macro_aupr_vs_base"], changes["raw_pooled_aupr_vs_base"]),
            "semantic_identification_of_bN_and_bI": False,
            "decision_rule_learned_on_test": False,
            "interpretation": "These diagnostics do not identify nuisance and informative latent components and are not a baseline-selection rule.",
        }
        if case["loader"] == "dti":
            dti_history = pd.read_csv(case["inputs"]["historical_metrics"]["path"])
            dti_row = dti_history[
                (dti_history["seed"].astype(int) == int(case["seed"]))
                & (dti_history["method"] == "Raw")
            ].iloc[0]
            diagnostics["raw_vs_base_cross_query_perturbation"] = float(
                dti_row["raw_vs_base_cross_query_inversion_rate"]
            )
            diagnostics["qrc_vs_base_cross_query_perturbation"] = float(
                dti_row["qrc_vs_base_cross_query_inversion_rate"]
            )
            diagnostics["cross_query_perturbation_status"] = "FROM_FROZEN_DTI_HISTORICAL_DEFINITION"
        else:
            diagnostics["raw_vs_base_cross_query_perturbation"] = None
            diagnostics["qrc_vs_base_cross_query_perturbation"] = None
            diagnostics["cross_query_perturbation_status"] = "NOT_COMPUTED_NO_FROZEN_MDA_HISTORICAL_DEFINITION"
        write_json(output_dir / "diagnostic_summary.json", diagnostics)

        nonpositive_allowed = bool(
            manifest.get("platt_nonpositive_policy", {}).get("retain_as_descriptive", False)
        )
        case_status = "PASS" if platt["optimizer_success"] and (
            platt["a"] > 0 or nonpositive_allowed
        ) else "BLOCKED"
        qc = {
            "status": case_status,
            "case_id": case["case_id"],
            "cpu_only": True,
            "historical_parity": parity,
            "raw_qrc_macro_invariance_error": raw_qrc_macro_error,
            "structure_and_alignment": loader_meta["structure"],
            "ranking_invariance": invariance,
            "zero_std_queries": zero_std_queries,
            "sqrc_validation_only_parameters": sqrc_parameters,
            "formula_audit": {
                "RawCenter": "Raw - query_mean(Raw)",
                "QRC": "Raw - query_mean(Raw - Base)",
                "RawCenter_is_QRC": False,
                "RawZScore": "(Raw - query_mean(Raw)) / query_std_ddof0(Raw)",
                "MedianResidual": "Raw - query_median(Raw - Base)",
                "SQRC": "frozen validation-only shrinkage definition retained",
            },
            "platt_status": platt["status"],
            "platt_nonpositive_retained_by_frozen_policy": bool(platt["a"] <= 0 and nonpositive_allowed),
            "uses_test_labels_for_fitting": False,
        }
        write_json(output_dir / "qc.json", qc)
        return {
            "case_id": case["case_id"], "status": qc["status"], "output_dir": str(output_dir),
            "historical_parity_max_error": parity["max_abs_error"],
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        parity_markers = (
            "FROZEN_INPUT_HASH_MISMATCH", "ALIGNMENT_FAIL", "HISTORICAL_PARITY_FAIL",
            "HISTORICAL_ROW_MISSING", "MACRO_INVARIANCE", "DTI_ALIGNMENT_FAIL",
            "MDA_RANDOM_ALIGNMENT_FAIL", "MDA_ENTITY_COLD_ALIGNMENT_FAIL", "MDA_STRICT_ALIGNMENT_FAIL",
        )
        failure_status = "PARITY_FAIL" if any(marker in error for marker in parity_markers) else "BLOCKED"
        # Preserve any detailed QC already computed only through the exception string; do not run later baselines.
        empty_outputs(output_dir, case, snapshot, error, failure_status)
        return {"case_id": case["case_id"], "status": failure_status, "output_dir": str(output_dir), "blocker": error}


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parents[1]
    run_kind = manifest.get("run_kind", "smoke")
    if run_kind == "formal":
        policy = manifest.get("platt_nonpositive_policy", {})
        policy_path = Path(policy.get("path", ""))
        if not policy_path.is_file() or sha256_file(policy_path) != policy.get("sha256"):
            raise SystemExit("PLATT_POLICY_HASH_MISMATCH")
    selected = set(args.case)
    cases = [case for case in manifest["cases"] if not selected or case["case_id"] in selected]
    unknown = selected - {case["case_id"] for case in manifest["cases"]}
    if unknown:
        raise SystemExit(f"Unknown cases: {sorted(unknown)}")
    if run_kind == "smoke" and len(cases) > 5:
        raise SystemExit("Smoke runner refuses more than five frozen cases")
    results = []
    for case in cases:
        print(f"[{case['case_id']}] parity gate and smoke", flush=True)
        result = run_case(case, manifest, root)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    summary_name = "SMOKE_RUN_SUMMARY.json" if run_kind == "smoke" else "FORMAL_RUN_SUMMARY.json"
    summary_path = root / "03_outputs" / run_kind / summary_name
    write_json(summary_path, {"cpu_only": True, "cases": results})
    return 0 if all(result["status"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
