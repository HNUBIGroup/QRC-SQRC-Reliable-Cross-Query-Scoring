#!/usr/bin/env python3
"""Independent numeric, structural, provenance, and reproducibility verifier."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path("/home/u2508283030/ECARF_Article2_20260911")
ROOT = PROJECT / "10_qrc_controlled_ground_truth_20260917"
RUNNER = ROOT / "src/run_controlled_ground_truth_benchmark.py"
MANIFESTS = ROOT / "manifests"
RESULTS = ROOT / "results"
REPORT = ROOT / "reports/QRC_CONTROLLED_GROUND_TRUTH_BENCHMARK.md"
QC = ROOT / "qc"
LOGS = ROOT / "logs"
ALPHAS = {0.0, 0.5, 1.0, 1.5}
BETAS = {0.0, 0.25, 0.5, 1.0, 2.0}
TOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_tsv(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_json(value: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def tracked_outputs() -> list[Path]:
    paths = sorted(MANIFESTS.glob("query_bias_vectors_seed*.tsv"))
    paths += [MANIFESTS / "input_provenance.tsv", MANIFESTS / "seed_residual_status.tsv",
              RESULTS / "controlled_alpha_beta_grid.tsv", RESULTS / "controlled_alpha_beta_summary.tsv",
              QC / "residual_centering_qc.tsv", QC / "mechanism_qc.json", REPORT,
              LOGS / "environment.json", LOGS / "controlled_benchmark.log.json"]
    return paths


def main() -> None:
    checks: list[dict[str, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    grid = pd.read_csv(RESULTS / "controlled_alpha_beta_grid.tsv", sep="\t")
    summary = pd.read_csv(RESULTS / "controlled_alpha_beta_summary.tsv", sep="\t")
    seeds = pd.read_csv(MANIFESTS / "seed_residual_status.tsv", sep="\t")
    residual_qc = pd.read_csv(QC / "residual_centering_qc.tsv", sep="\t")
    mechanism = json.loads((QC / "mechanism_qc.json").read_text(encoding="utf-8"))

    required_grid = {
        "seed", "alpha", "beta", "sigma_r", "sigma_b", "bias_residual_ratio",
        "target_pooled_auc", "synthetic_pooled_auc", "qrc_pooled_auc",
        "target_pooled_aupr", "synthetic_pooled_aupr", "qrc_pooled_aupr",
        "target_macro_auc", "synthetic_macro_auc", "qrc_macro_auc",
        "target_macro_aupr", "synthetic_macro_aupr", "qrc_macro_aupr",
        "bias_mae", "bias_rmse", "bias_pearson", "bias_spearman", "bias_max_abs_error",
        "score_recovery_mae", "score_recovery_rmse", "score_recovery_max_error",
        "macro_invariance_max_diff", "alpha0_base_recovery_max_error",
    }
    check("grid_required_columns", required_grid.issubset(grid.columns), str(sorted(required_grid - set(grid.columns))))
    check("grid_cardinality", len(grid) == 25 * 4 * 5 and not grid.duplicated(["seed", "alpha", "beta"]).any(), f"rows={len(grid)}")
    check("grid_seed_set", set(grid["seed"]) == set(range(25)), str(sorted(set(grid["seed"]))))
    check("alpha_grid_frozen", set(grid["alpha"]) == ALPHAS, str(sorted(set(grid["alpha"]))))
    check("beta_grid_frozen", set(grid["beta"]) == BETAS, str(sorted(set(grid["beta"]))))
    check("each_seed_20_cells", (grid.groupby("seed").size() == 20).all(), str(grid.groupby("seed").size().to_dict()))

    active = set(seeds.loc[seeds["residual_status"] == "RESIDUAL_ACTIVE", "seed"].astype(int))
    inactive = set(seeds.loc[seeds["residual_status"] == "RESIDUAL_INACTIVE", "seed"].astype(int))
    expected_active = {0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 24}
    check("seed_status_complete", len(seeds) == 25 and active | inactive == set(range(25)) and not (active & inactive), f"active={len(active)} inactive={len(inactive)}")
    check("residual_active_seed_set", active == expected_active, str(sorted(active)))
    check("inactive_residual_zero", (grid.loc[grid["seed"].isin(inactive), "sigma_r"].abs() <= TOL).all(), f"max={grid.loc[grid['seed'].isin(inactive),'sigma_r'].abs().max():.3e}")
    check("residual_centering", residual_qc["abs_mean_residual"].max() <= TOL and (residual_qc["status"] == "PASS").all(), f"max={residual_qc['abs_mean_residual'].max():.3e}")

    vector_files = sorted(MANIFESTS.glob("query_bias_vectors_seed*.tsv"))
    vector_ok, vector_detail = len(vector_files) == 25, []
    for seed in range(25):
        path = MANIFESTS / f"query_bias_vectors_seed{seed}.tsv"
        if not path.is_file():
            vector_ok = False
            vector_detail.append(f"missing:{seed}")
            continue
        vector = pd.read_csv(path, sep="\t")
        ok = (list(vector.columns) == ["drug", "c_i"] and len(vector) == 6 and vector["drug"].is_unique
              and abs(vector["c_i"].mean()) <= TOL and abs(vector["c_i"].std(ddof=0) - 1.0) <= TOL)
        if not ok:
            vector_ok = False
            vector_detail.append(f"seed{seed}:n={len(vector)} mean={vector['c_i'].mean():.3e} sd={vector['c_i'].std(ddof=0):.3e}")
    check("query_bias_vectors", vector_ok, ";".join(vector_detail) or "25 files; each n=6, mean=0, population SD=1")

    check("sigma_b_matches_beta", np.max(np.abs(grid["sigma_b"] - grid["beta"])) <= TOL, f"maxdiff={np.max(np.abs(grid['sigma_b']-grid['beta'])):.3e}")
    check("alpha0_sigma_r_zero", grid.loc[grid["alpha"] == 0, "sigma_r"].abs().max() <= TOL, f"max={grid.loc[grid['alpha']==0,'sigma_r'].abs().max():.3e}")
    check("bias_recovery", grid[["bias_mae", "bias_rmse", "bias_max_abs_error"]].to_numpy().max() <= TOL, f"max={grid[['bias_mae','bias_rmse','bias_max_abs_error']].to_numpy().max():.3e}")
    positive_beta = grid[grid["beta"] > 0]
    check("bias_correlations", np.min(positive_beta["bias_pearson"]) >= 1 - TOL and np.min(positive_beta["bias_spearman"]) >= 1 - TOL, f"minPearson={positive_beta['bias_pearson'].min():.15g};minSpearman={positive_beta['bias_spearman'].min():.15g}")
    check("score_recovery", grid[["score_recovery_mae", "score_recovery_rmse", "score_recovery_max_error"]].to_numpy().max() <= TOL, f"max={grid[['score_recovery_mae','score_recovery_rmse','score_recovery_max_error']].to_numpy().max():.3e}")
    check("macro_invariance", grid["macro_invariance_max_diff"].max() <= TOL and (grid["macro_invariance_status"] == "PASS").all(), f"max={grid['macro_invariance_max_diff'].max():.3e}")
    check("qrc_pooled_beta_invariance", grid[["qrc_pooled_auc_abs_deviation", "qrc_pooled_aupr_abs_deviation"]].to_numpy().max() <= TOL, f"max={grid[['qrc_pooled_auc_abs_deviation','qrc_pooled_aupr_abs_deviation']].to_numpy().max():.3e}")
    check("alpha0_recovers_base", grid.loc[grid["alpha"] == 0, "alpha0_base_recovery_max_error"].max() <= TOL, f"max={grid.loc[grid['alpha']==0,'alpha0_base_recovery_max_error'].max():.3e}")
    check("alpha_positive_preserves_residual", grid.loc[(grid["alpha"] > 0) & grid["seed"].isin(active), "score_recovery_max_error"].max() <= TOL, f"max={grid.loc[(grid['alpha']>0)&grid['seed'].isin(active),'score_recovery_max_error'].max():.3e}")
    check("raw_pooled_is_affected", grid.loc[grid["beta"] == 2, "raw_pooled_auc_abs_deviation"].mean() > grid.loc[grid["beta"] == 0, "raw_pooled_auc_abs_deviation"].mean(), f"beta0={grid.loc[grid['beta']==0,'raw_pooled_auc_abs_deviation'].mean():.3e};beta2={grid.loc[grid['beta']==2,'raw_pooled_auc_abs_deviation'].mean():.6g}")

    check("summary_scopes", set(summary["scope"]) == {"all_seeds", "residual_active_seeds"} and len(summary) == 40, f"rows={len(summary)}")
    check("summary_seed_counts", set(summary.loc[summary["scope"] == "all_seeds", "n_seeds"]) == {25} and set(summary.loc[summary["scope"] == "residual_active_seeds", "n_seeds"]) == {18}, str(summary.groupby('scope')['n_seeds'].unique().to_dict()))

    provenance = pd.read_csv(MANIFESTS / "input_provenance.tsv", sep="\t")
    check("provenance_50_inputs", len(provenance) == 50 and not provenance.duplicated(["seed", "role"]).any(), f"rows={len(provenance)}")
    checksum_bad = []
    for row in provenance.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or sha256(path) != row.sha256:
            checksum_bad.append(str(path))
    check("frozen_input_checksums", not checksum_bad, str(checksum_bad))

    report = REPORT.read_text(encoding="utf-8")
    check("report_verdict", "Final verdict: **CONTROLLED_QRC_MECHANISM_VALIDATED**" in report, "validated verdict present")
    check("report_answers", all(token in report for token in ("Known query bias recovery", "Score recovery", "Effect of beta", "alpha=0 boundary", "alpha>0", "Harmful-to-useful ratio", "Conditions where the theory holds", "Boundaries")), "required mechanism sections present")
    check("fallacy_scan_11", "Coverage: **11/11 checked**" in report, "11/11 reported")
    check("initial_precision_failure_disclosed", "initial float64 execution" in report and (LOGS / "initial_float64_qc_failure.json").is_file(), "precision remediation disclosed")

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imports = {node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)} | {(node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    forbidden = {"torch", "tensorflow", "keras", "xgboost", "lightgbm", "catboost"}
    check("no_training_framework_imports", not (imports & forbidden), str(sorted(imports)))
    environment = json.loads((LOGS / "environment.json").read_text(encoding="utf-8"))
    check("no_gpu_or_training", environment["model_training"] is False and environment["gpu_used"] is False, str(environment))

    check("mechanism_json_verdict", mechanism["final_verdict"] == "CONTROLLED_QRC_MECHANISM_VALIDATED" and mechanism["macro_invariance_status"] == "PASS", mechanism["final_verdict"])

    # Deterministic reproducibility: exact byte hashes before and after a fresh
    # CPU-only rerun of the same frozen command.
    tracked = tracked_outputs()
    before = {str(path): sha256(path) for path in tracked}
    rerun = subprocess.run([sys.executable, str(RUNNER)], cwd=str(PROJECT), text=True, capture_output=True, check=False)
    atomic_json({"returncode": rerun.returncode, "stdout": rerun.stdout, "stderr": rerun.stderr}, LOGS / "deterministic_rerun.json")
    after = {str(path): sha256(path) for path in tracked}
    comparison = [{"path": path, "sha256_before": before[path], "sha256_after": after[path], "status": "MATCH" if before[path] == after[path] else "MISMATCH"} for path in sorted(before)]
    atomic_tsv(comparison, QC / "reproducibility_hash_comparison.tsv")
    check("deterministic_rerun_exit", rerun.returncode == 0, f"returncode={rerun.returncode}")
    check("deterministic_byte_reproducibility", all(row["status"] == "MATCH" for row in comparison), f"mismatches={sum(row['status']!='MATCH' for row in comparison)}")

    failed = [row for row in checks if row["status"] == "FAIL"]
    atomic_tsv(checks, QC / "independent_verification.tsv")
    result = {"status": "INDEPENDENT_VERIFICATION_PASS" if not failed else "INDEPENDENT_VERIFICATION_FAIL",
              "checks": len(checks), "failed": len(failed), "failed_checks": [row["check"] for row in failed],
              "reproducibility": "REPRODUCIBLE" if not failed else "NOT_REPRODUCIBLE"}
    atomic_json(result, QC / "independent_verification.json")
    print(json.dumps(result, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
