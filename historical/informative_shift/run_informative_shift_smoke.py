#!/usr/bin/env python3
"""CPU-only seed-0 controlled informative-shift smoke benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/home/u2508283030/ECARF_Article2_20260911/12_qrc_controlled_informative_shift_20260920")
MANIFEST_PATH = ROOT / "00_admin/FROZEN_SMOKE_MANIFEST.json"
OUTPUT = ROOT / "03_outputs/smoke_seed0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def query_means(
    queries: np.ndarray, values: np.ndarray, unique_queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.array(
        [np.mean(values[queries == query], dtype=np.longdouble) for query in unique_queries],
        dtype=np.longdouble,
    )
    lookup = dict(zip(unique_queries, means))
    broadcast = np.fromiter((lookup[query] for query in queries), dtype=np.longdouble, count=len(queries))
    return means, broadcast


def score_errors(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = candidate - target
    return {
        "mae": float(np.mean(np.abs(error), dtype=np.longdouble)),
        "rmse": float(np.sqrt(np.mean(error * error, dtype=np.longdouble))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def metric_bundle(
    queries: np.ndarray, labels: np.ndarray, score: np.ndarray,
) -> dict[str, float | int]:
    pooled_auroc = float(roc_auc_score(labels, score))
    pooled_aupr = float(average_precision_score(labels, score))
    macro_auroc: list[float] = []
    macro_aupr: list[float] = []
    for query in np.unique(queries):
        mask = queries == query
        current_labels = labels[mask]
        if np.unique(current_labels).size < 2:
            continue
        macro_auroc.append(float(roc_auc_score(current_labels, score[mask])))
        macro_aupr.append(float(average_precision_score(current_labels, score[mask])))
    return {
        "pooled_auroc": pooled_auroc,
        "pooled_aupr": pooled_aupr,
        "macro_auroc": float(np.mean(macro_auroc)),
        "macro_aupr": float(np.mean(macro_aupr)),
        "valid_macro_queries": len(macro_auroc),
    }


def ranking_invariance(
    queries: np.ndarray, raw: np.ndarray, qrc: np.ndarray,
) -> dict[str, float | int]:
    strict_order_violations = 0
    tie_violations = 0
    max_pairwise_difference_error = np.longdouble(0.0)
    chunk_size = 128
    for query in np.unique(queries):
        mask = queries == query
        raw_query = raw[mask]
        qrc_query = qrc[mask]
        n = len(raw_query)
        all_index = np.arange(n)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            left_index = np.arange(start, stop)
            select = all_index[None, :] > left_index[:, None]
            raw_diff = (raw_query[start:stop, None] - raw_query[None, :])[select]
            qrc_diff = (qrc_query[start:stop, None] - qrc_query[None, :])[select]
            if raw_diff.size:
                max_pairwise_difference_error = max(
                    max_pairwise_difference_error,
                    np.max(np.abs(raw_diff - qrc_diff)),
                )
                strict_order_violations += int(np.sum(raw_diff * qrc_diff < 0))
                tie_violations += int(np.sum((raw_diff == 0) != (qrc_diff == 0)))
    return {
        "strict_order_violations": strict_order_violations,
        "tie_violations": tie_violations,
        "max_pairwise_difference_error": float(max_pairwise_difference_error),
    }


def standardize_population(values: np.ndarray, threshold: float, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.longdouble)
    centered = values - np.mean(values, dtype=np.longdouble)
    variance = np.mean(centered * centered, dtype=np.longdouble)
    if variance <= np.longdouble(threshold):
        raise RuntimeError(f"{name}_DEGENERATE: variance={float(variance):.17g}")
    return centered / np.sqrt(variance)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["seed"] != 0 or manifest["alpha"] != 1.0:
        raise RuntimeError("FROZEN_SMOKE_SCOPE_MISMATCH")
    betas = tuple(float(value) for value in manifest["beta_grid"])
    gammas = tuple(float(value) for value in manifest["gamma_grid"])
    if betas != (0.0, 0.25, 0.5, 1.0, 2.0) or gammas != (0.0, 0.25, 0.5, 1.0, 2.0):
        raise RuntimeError("FROZEN_GRID_MISMATCH")
    tolerance = float(manifest["algebraic_tolerance"])
    macro_tolerance = float(manifest["macro_invariance_tolerance"])
    degenerate_threshold = float(manifest["degenerate_variance_threshold"])
    OUTPUT.mkdir(parents=True, exist_ok=True)

    snapshot: dict[str, Any] = {
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "seed": 0,
        "inputs": {},
    }
    for name, entry in manifest["inputs"].items():
        path = Path(entry["path"])
        actual = sha256_file(path)
        snapshot["inputs"][name] = {
            "path": str(path), "expected_sha256": entry["sha256"],
            "actual_sha256": actual, "hash_match": actual == entry["sha256"],
        }
    write_json(OUTPUT / "input_manifest_snapshot.json", snapshot)
    if not all(entry["hash_match"] for entry in snapshot["inputs"].values()):
        raise RuntimeError("FROZEN_INPUT_HASH_MISMATCH")

    old_verification = json.loads(
        Path(manifest["inputs"]["old_independent_verification"]["path"]).read_text(encoding="utf-8")
    )
    if old_verification["status"] != "INDEPENDENT_VERIFICATION_PASS":
        raise RuntimeError("OLD_CONTROLLED_VERIFICATION_NOT_PASS")

    score_path = Path(manifest["inputs"]["frozen_pair_scores"]["path"])
    entity_path = Path(manifest["inputs"]["frozen_test_entities"]["path"])
    scores = pd.read_csv(
        score_path, sep="\t",
        usecols=["drug", "miRNA", "split", "molecular_score", "final_score"],
    )
    test_queries = set(pd.read_csv(entity_path, sep="\t").iloc[:, 0].astype(str))
    test = scores[
        scores["drug"].astype(str).isin(test_queries)
        & scores["split"].isin([manifest["positive_split"], manifest["pseudo_negative_split"]])
    ].copy()
    test["drug"] = test["drug"].astype(str)
    test["miRNA"] = test["miRNA"].astype(str)
    test = test.sort_values(["drug", "miRNA"], kind="mergesort").reset_index(drop=True)
    if len(test) != len(test_queries) * 1195:
        raise RuntimeError(f"INCOMPLETE_CANDIDATE_SET: rows={len(test)}")
    if test.duplicated(["drug", "miRNA"]).any():
        raise RuntimeError("DUPLICATE_PAIR_KEY")
    if set(test["drug"].unique()) != test_queries:
        raise RuntimeError("QUERY_ALIGNMENT_FAIL")
    candidate_counts = test.groupby("drug")["miRNA"].nunique()
    if not (candidate_counts == 1195).all():
        raise RuntimeError(f"CANDIDATE_ALIGNMENT_FAIL: {candidate_counts.to_dict()}")

    test["label"] = (test["split"] == manifest["positive_split"]).astype(np.int8)
    queries = test["drug"].to_numpy(str)
    labels = test["label"].to_numpy(np.int8)
    unique_queries = np.array(sorted(np.unique(queries)))
    base = test[manifest["base_column"]].to_numpy(np.longdouble)
    historical_raw = test[manifest["historical_raw_column_for_residual"]].to_numpy(np.longdouble)
    historical_delta = historical_raw - base
    _, historical_bias_pair = query_means(queries, historical_delta, unique_queries)
    residual = historical_delta - historical_bias_pair
    residual_means, _ = query_means(queries, residual, unique_queries)

    query_rows = []
    q_raw_values = []
    for query in unique_queries:
        current_labels = labels[queries == query]
        positives = int(current_labels.sum())
        candidates = int(len(current_labels))
        negatives = candidates - positives
        empirical_logit = np.log(
            (np.longdouble(positives) + np.longdouble(0.5))
            / (np.longdouble(negatives) + np.longdouble(0.5))
        )
        q_raw_values.append(empirical_logit)
        query_rows.append({
            "seed": 0, "query": query, "candidate_count": candidates,
            "positive_count": positives, "negative_count": negatives,
            "prevalence": positives / candidates,
            "smoothed_empirical_logit": float(empirical_logit),
        })
    q_raw = np.asarray(q_raw_values, dtype=np.longdouble)
    q_raw_variance = np.var(q_raw, ddof=0)
    if q_raw_variance <= np.longdouble(degenerate_threshold):
        raise RuntimeError(f"INFORMATIVE_COMPONENT_DEGENERATE: variance={float(q_raw_variance):.17g}")
    q = standardize_population(q_raw, degenerate_threshold, "INFORMATIVE_COMPONENT")

    rng = np.random.default_rng(int(manifest["rng_seed"]))
    c_initial = standardize_population(rng.normal(0.0, 1.0, len(unique_queries)), degenerate_threshold, "NUISANCE_INITIAL")
    projection_coefficient = np.sum(c_initial * q, dtype=np.longdouble) / np.sum(q * q, dtype=np.longdouble)
    c_projected = c_initial - projection_coefficient * q
    c = standardize_population(c_projected, degenerate_threshold, "NUISANCE_POST_PROJECTION")
    corr_q_c = np.sum(q * c, dtype=np.longdouble) / np.sqrt(
        np.sum(q * q, dtype=np.longdouble) * np.sum(c * c, dtype=np.longdouble)
    )
    q_lookup = dict(zip(unique_queries, q))
    c_lookup = dict(zip(unique_queries, c))
    q_pair = np.fromiter((q_lookup[query] for query in queries), dtype=np.longdouble, count=len(queries))
    c_pair = np.fromiter((c_lookup[query] for query in queries), dtype=np.longdouble, count=len(queries))
    for index, row in enumerate(query_rows):
        row["q_i"] = float(q[index])
        row["c_initial_standardized"] = float(c_initial[index])
        row["c_i_orthogonalized"] = float(c[index])
        row["residual_mean"] = float(residual_means[index])
    query_components = pd.DataFrame(query_rows)
    query_components.to_csv(OUTPUT / "query_components.csv", index=False, float_format="%.17g")

    base_metrics = metric_bundle(queries, labels, base)
    recovery_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []

    for beta in betas:
        for gamma in gammas:
            beta_ld = np.longdouble(beta)
            gamma_ld = np.longdouble(gamma)
            target = base + residual + gamma_ld * q_pair
            raw = target + beta_ld * c_pair
            delta = raw - base
            b_hat_query, b_hat_pair = query_means(queries, delta, unique_queries)
            qrc = raw - b_hat_pair
            oracle_n = raw - beta_ld * c_pair
            true_observable_query = gamma_ld * q + beta_ld * c
            raw_error = score_errors(raw, target)
            qrc_error = score_errors(qrc, target)
            oracle_error = score_errors(oracle_n, target)
            bias_error = score_errors(b_hat_query, true_observable_query)
            informative_removal_error = score_errors(qrc - target, -gamma_ld * q_pair)
            invariance = ranking_invariance(queries, raw, qrc)

            method_scores = {
                "Base": base, "Target": target, "Raw": raw,
                "QRC": qrc, "OracleN": oracle_n,
            }
            current_metrics: dict[str, dict[str, float | int]] = {}
            for method, method_score in method_scores.items():
                metrics = metric_bundle(queries, labels, method_score)
                current_metrics[method] = metrics
                ranking_rows.append({
                    "seed": 0, "alpha": 1.0, "beta": beta, "gamma": gamma,
                    "method": method, **metrics,
                    "role": "controlled_oracle_reference" if method in {"Target", "OracleN"} else (
                        "deployment_method" if method in {"Base", "Raw", "QRC"} else ""
                    ),
                })
            macro_difference = max(
                abs(float(current_metrics["Raw"]["macro_auroc"]) - float(current_metrics["QRC"]["macro_auroc"])),
                abs(float(current_metrics["Raw"]["macro_aupr"]) - float(current_metrics["QRC"]["macro_aupr"])),
            )
            if beta > gamma:
                phase_region = "BETA_GT_GAMMA_QRC_EXPECTED_BETTER"
                phase_pass = qrc_error["rmse"] < raw_error["rmse"]
            elif beta < gamma:
                phase_region = "BETA_LT_GAMMA_RAW_EXPECTED_BETTER"
                phase_pass = raw_error["rmse"] < qrc_error["rmse"]
            else:
                phase_region = "BETA_EQ_GAMMA_EQUAL_EXPECTED"
                phase_pass = abs(raw_error["rmse"] - qrc_error["rmse"]) <= tolerance

            recovery_rows.append({
                "seed": 0, "alpha": 1.0, "beta": beta, "gamma": gamma,
                "raw_to_target_mae": raw_error["mae"],
                "raw_to_target_rmse": raw_error["rmse"],
                "raw_to_target_max_abs_error": raw_error["max_abs_error"],
                "qrc_to_target_mae": qrc_error["mae"],
                "qrc_to_target_rmse": qrc_error["rmse"],
                "qrc_to_target_max_abs_error": qrc_error["max_abs_error"],
                "oracle_n_to_target_mae": oracle_error["mae"],
                "oracle_n_to_target_rmse": oracle_error["rmse"],
                "oracle_n_to_target_max_abs_error": oracle_error["max_abs_error"],
                "bias_recovery_mae": bias_error["mae"],
                "bias_recovery_rmse": bias_error["rmse"],
                "bias_recovery_max_abs_error": bias_error["max_abs_error"],
                "informative_removal_identity_mae": informative_removal_error["mae"],
                "informative_removal_identity_rmse": informative_removal_error["rmse"],
                "informative_removal_identity_max_abs_error": informative_removal_error["max_abs_error"],
                "strict_order_violations": invariance["strict_order_violations"],
                "tie_violations": invariance["tie_violations"],
                "max_pairwise_difference_error": invariance["max_pairwise_difference_error"],
                "raw_qrc_macro_metric_difference": macro_difference,
                "phase_region": phase_region,
                "phase_boundary_status": "PASS" if phase_pass else "FAIL",
            })
            summary_rows.append({
                "seed": 0, "alpha": 1.0, "beta": beta, "gamma": gamma,
                "n_pairs": len(test), "n_queries": len(unique_queries),
                "target_mean": float(np.mean(target, dtype=np.longdouble)),
                "target_population_sd": float(np.std(target, ddof=0)),
                "raw_mean": float(np.mean(raw, dtype=np.longdouble)),
                "raw_population_sd": float(np.std(raw, ddof=0)),
                "qrc_mean": float(np.mean(qrc, dtype=np.longdouble)),
                "qrc_population_sd": float(np.std(qrc, ddof=0)),
                "oracle_n_mean": float(np.mean(oracle_n, dtype=np.longdouble)),
                "oracle_n_population_sd": float(np.std(oracle_n, ddof=0)),
                "expected_raw_rmse": beta,
                "observed_raw_rmse": raw_error["rmse"],
                "expected_qrc_rmse": gamma,
                "observed_qrc_rmse": qrc_error["rmse"],
            })
            matrix_rows.append({
                "seed": 0, "beta": beta, "gamma": gamma,
                "raw_rmse_to_target": raw_error["rmse"],
                "qrc_rmse_to_target": qrc_error["rmse"],
                "qrc_rmse_minus_raw_rmse": qrc_error["rmse"] - raw_error["rmse"],
                "qrc_pooled_auroc_minus_raw": float(current_metrics["QRC"]["pooled_auroc"]) - float(current_metrics["Raw"]["pooled_auroc"]),
                "qrc_pooled_aupr_minus_raw": float(current_metrics["QRC"]["pooled_aupr"]) - float(current_metrics["Raw"]["pooled_aupr"]),
                "phase_region": phase_region,
                "phase_boundary_status": "PASS" if phase_pass else "FAIL",
            })

    recovery = pd.DataFrame(recovery_rows).sort_values(["beta", "gamma"]).reset_index(drop=True)
    ranking = pd.DataFrame(ranking_rows).sort_values(["beta", "gamma", "method"]).reset_index(drop=True)
    summaries = pd.DataFrame(summary_rows).sort_values(["beta", "gamma"]).reset_index(drop=True)
    matrix = pd.DataFrame(matrix_rows).sort_values(["beta", "gamma"]).reset_index(drop=True)
    recovery.to_csv(OUTPUT / "ground_truth_recovery.csv", index=False, float_format="%.17g")
    ranking.to_csv(OUTPUT / "ranking_metrics.csv", index=False, float_format="%.17g")
    summaries.to_csv(OUTPUT / "controlled_scores_summary.csv", index=False, float_format="%.17g")
    matrix.to_csv(OUTPUT / "beta_gamma_matrix.csv", index=False, float_format="%.17g")

    q_mean = float(np.mean(q, dtype=np.longdouble))
    q_variance = float(np.var(q, ddof=0))
    c_mean = float(np.mean(c, dtype=np.longdouble))
    c_variance = float(np.var(c, ddof=0))
    max_residual_mean = float(np.max(np.abs(residual_means)))
    max_oracle_error = float(recovery["oracle_n_to_target_max_abs_error"].max())
    max_bias_error = float(recovery["bias_recovery_max_abs_error"].max())
    max_informative_identity_error = float(recovery["informative_removal_identity_max_abs_error"].max())
    max_macro_difference = float(recovery["raw_qrc_macro_metric_difference"].max())
    max_pairwise_difference_error = float(recovery["max_pairwise_difference_error"].max())
    phase_counts = recovery.groupby("phase_region")["phase_boundary_status"].agg(["count", lambda values: int((values == "PASS").sum())])
    phase_counts.columns = ["conditions", "pass_conditions"]
    core_checks = {
        "q_mean": abs(q_mean) <= tolerance,
        "q_variance": abs(q_variance - 1.0) <= tolerance,
        "c_mean": abs(c_mean) <= tolerance,
        "c_variance": abs(c_variance - 1.0) <= tolerance,
        "q_c_correlation": abs(float(corr_q_c)) <= tolerance,
        "residual_centering": max_residual_mean <= tolerance,
        "oracle_recovery": max_oracle_error <= tolerance,
        "bias_recovery": max_bias_error <= tolerance,
        "informative_removal_identity": max_informative_identity_error <= tolerance,
        "macro_invariance": max_macro_difference <= macro_tolerance,
        "strict_order": int(recovery["strict_order_violations"].sum()) == 0,
        "tie_invariance": int(recovery["tie_violations"].sum()) == 0,
        "phase_boundary": bool((recovery["phase_boundary_status"] == "PASS").all()),
        "no_labels_used_by_qrc": True,
        "no_test_derived_tuning": True,
    }
    qc = {
        "status": "PASS" if all(core_checks.values()) else "FAIL",
        "seed": 0,
        "conditions": len(recovery),
        "grid": {"alpha": 1.0, "beta": list(betas), "gamma": list(gammas)},
        "input_hashes_all_match": True,
        "row_count": len(test),
        "query_count": len(unique_queries),
        "candidate_count_per_query": 1195,
        "positive_count": int(labels.sum()),
        "labels": "frozen historical PU labels; used for oracle q construction and evaluation only",
        "informative_component": {
            "status": "NON_DEGENERATE",
            "q_raw_population_variance": float(q_raw_variance),
            "q_mean": q_mean,
            "q_population_variance": q_variance,
            "q_min": float(np.min(q)),
            "q_max": float(np.max(q)),
        },
        "nuisance_component": {
            "rng_seed": int(manifest["rng_seed"]),
            "projection_coefficient": float(projection_coefficient),
            "c_mean": c_mean,
            "c_population_variance": c_variance,
            "corr_q_c": float(corr_q_c),
            "c_min": float(np.min(c)),
            "c_max": float(np.max(c)),
        },
        "max_abs_residual_query_mean": max_residual_mean,
        "max_oracle_n_to_target_error": max_oracle_error,
        "max_bias_recovery_error": max_bias_error,
        "max_informative_removal_identity_error": max_informative_identity_error,
        "max_raw_qrc_macro_metric_difference": max_macro_difference,
        "strict_order_violations_total": int(recovery["strict_order_violations"].sum()),
        "tie_violations_total": int(recovery["tie_violations"].sum()),
        "max_pairwise_difference_error": max_pairwise_difference_error,
        "phase_boundary": {
            region: {
                "conditions": int(row["conditions"]),
                "pass_conditions": int(row["pass_conditions"]),
            }
            for region, row in phase_counts.iterrows()
        },
        "core_checks": core_checks,
        "q_i_is_oracle_controlled_informative_component": True,
        "qrc_accesses_q_i": False,
        "qrc_accesses_labels": False,
        "labels_used_to_construct_oracle_q_i": True,
        "decision_rule_learned_on_test": False,
        "gpu_used": False,
        "model_training_or_inference": False,
        "formal_seeds_1_to_24_run": False,
        "base_metrics": base_metrics,
    }
    write_json(OUTPUT / "qc.json", qc)
    print(json.dumps(qc, indent=2, sort_keys=True), flush=True)
    return 0 if qc["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
