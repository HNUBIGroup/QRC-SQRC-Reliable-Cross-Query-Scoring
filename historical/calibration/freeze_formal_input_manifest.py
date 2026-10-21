#!/usr/bin/env python3
"""Create the immutable 125-case input hash manifest from audited frozen assets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/u2508283030/ECARF_Article2_20260911")
TASK_ROOT = ROOT / "11_qrc_calibration_baselines_20260919"
MDA_ROOT = Path("/home/u2508283030/MDA_CellPainting_Multiview_20260910")
DTI_ROOT = ROOT / "DTI_QRC_FORMAL_20260918"
SOURCE_MANIFEST = TASK_ROOT / "00_admin/FROZEN_INPUT_MANIFEST.json"
POLICY = TASK_ROOT / "00_admin/PLATT_NONPOSITIVE_POLICY_FREEZE_20260920.md"
OUTPUT = TASK_ROOT / "00_admin/FORMAL_FROZEN_INPUT_MANIFEST_125CASE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if len(source["cases"]) != 5:
        raise RuntimeError("SOURCE_SMOKE_MANIFEST_CASE_COUNT_CHANGED")
    policy_hash = sha256_file(POLICY)
    if policy_hash != "6b73a3c4e81aef4b0522bcca183eef4be3e8500e5e54629b58d26e034c77caec":
        raise RuntimeError("PLATT_POLICY_HASH_MISMATCH_BEFORE_FORMAL_FREEZE")

    stage_e = ROOT / "02_results/stageE_shrinkage_qrc_25seed.tsv"
    stage_g = ROOT / "02_results/stageG_v2_random_strict_final_25seed.tsv"
    lambda_table = MDA_ROOT / "05_results/formal_random25/formal_random25_perseed.tsv"
    dti_metrics = DTI_ROOT / "results/per_seed_metrics.csv"
    cases: list[dict] = []

    for seed in range(25):
        cases.append({
            "case_id": f"mda_random_seed_{seed}",
            "task": "MDA", "setting": "random", "seed": seed,
            "loader": "mda_random", "query_column": "drug", "candidate_column": "miRNA",
            "inputs": {
                "score_table": file_entry(ROOT / f"03_frozen_inputs/random_current_20260912/all_pair_scores_seed{seed}.tsv"),
                "lambda_table": file_entry(lambda_table),
                "validation_positive_pairs": file_entry(MDA_ROOT / f"03_splits/seed{seed}_fourview/val_pairs.tsv"),
                "test_positive_pairs": file_entry(MDA_ROOT / f"03_splits/seed{seed}_fourview/test_pairs.tsv"),
                "historical_metrics": file_entry(stage_g),
            },
        })

    for mode, setting in (("mirna", "mirna_cold"), ("drug", "drug_cold")):
        query_column, candidate_column = ("miRNA", "drug") if mode == "mirna" else ("drug", "miRNA")
        for seed in range(25):
            tag = f"{mode}_cold_seed{seed}"
            cases.append({
                "case_id": f"mda_{setting}_seed_{seed}",
                "task": "MDA", "setting": setting, "seed": seed,
                "loader": "mda_entity_cold", "mode": mode,
                "query_column": query_column, "candidate_column": candidate_column,
                "inputs": {
                    "score_table": file_entry(MDA_ROOT / f"05_results/{tag}/all_pair_scores_{tag}.tsv"),
                    "validation_entities": file_entry(MDA_ROOT / f"03_splits/{tag}/val_entities.tsv"),
                    "test_entities": file_entry(MDA_ROOT / f"03_splits/{tag}/test_entities.tsv"),
                    "historical_metrics": file_entry(stage_e),
                },
            })

    for seed in range(25):
        tag = f"strict_cold_seed{seed}"
        cases.append({
            "case_id": f"mda_strict_cold_seed_{seed}",
            "task": "MDA", "setting": "strict_cold", "seed": seed,
            "loader": "mda_strict_cold", "query_column": "drug", "candidate_column": "miRNA",
            "inputs": {
                "score_table": file_entry(MDA_ROOT / f"05_results/{tag}/all_pair_scores_{tag}.tsv"),
                "validation_drugs": file_entry(MDA_ROOT / f"03_splits/{tag}/val_drugs.tsv"),
                "test_drugs": file_entry(MDA_ROOT / f"03_splits/{tag}/test_drugs.tsv"),
                "validation_mirnas": file_entry(MDA_ROOT / f"03_splits/{tag}/val_mirnas.tsv"),
                "test_mirnas": file_entry(MDA_ROOT / f"03_splits/{tag}/test_mirnas.tsv"),
                "historical_metrics": file_entry(stage_g),
            },
        })

    for seed in range(5, 30):
        cases.append({
            "case_id": f"dti_drug_cold_seed_{seed}",
            "task": "DTI", "setting": "drug_cold", "seed": seed,
            "loader": "dti", "query_column": "canonical_smiles", "candidate_column": "target_id",
            "inputs": {
                "validation_scores": file_entry(DTI_ROOT / f"predictions/seed_{seed}_validation_pairs.csv"),
                "test_scores": file_entry(DTI_ROOT / f"predictions/seed_{seed}_test_pairs.csv"),
                "historical_qrc_scores": file_entry(DTI_ROOT / f"predictions/seed_{seed}_test_pairs_qrc.csv"),
                "split_manifest": file_entry(DTI_ROOT / f"splits/davis_seed_{seed}.json"),
                "historical_metrics": file_entry(dti_metrics),
                "historical_qc": file_entry(DTI_ROOT / f"qc/seed_{seed}_qrc_qc.json"),
            },
        })

    counts = Counter((case["task"], case["setting"]) for case in cases)
    expected = {
        ("MDA", "random"): 25,
        ("MDA", "mirna_cold"): 25,
        ("MDA", "drug_cold"): 25,
        ("MDA", "strict_cold"): 25,
        ("DTI", "drug_cold"): 25,
    }
    if len(cases) != 125 or dict(counts) != expected:
        raise RuntimeError(f"FORMAL_CASE_MATRIX_INVALID: n={len(cases)} counts={dict(counts)}")

    payload = {
        "schema_version": 2,
        "created_at": "2026-09-20",
        "run_kind": "formal",
        "scope": "CPU-only formal 125-case post-processing; no training, inference, GPU, resampling, or new methods",
        "source_smoke_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256_file(SOURCE_MANIFEST),
        },
        "platt_nonpositive_policy": {
            "path": str(POLICY),
            "sha256": policy_hash,
            "retain_as_descriptive": True,
            "monotonic_invariance_required_when_nonpositive": False,
        },
        "historical_metric_tolerance": 1e-8,
        "ranking_invariance_tolerance": 1e-12,
        "ece_bins": 10,
        "raw_zscore_ddof": 0,
        "random_unlabeled_partition_seed_rule": "20000 + seed",
        "case_count": len(cases),
        "case_counts": {f"{task}/{setting}": count for (task, setting), count in sorted(counts.items())},
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {OUTPUT}")
    print(f"CASES {len(cases)}")
    print(f"SHA256 {sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
