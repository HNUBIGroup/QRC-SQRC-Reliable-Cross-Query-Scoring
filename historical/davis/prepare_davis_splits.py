#!/usr/bin/env python3
"""Audit the official MMDG-DTI Davis table and freeze 25 confirmatory drug-cold splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split


SEEDS = tuple(range(5, 30))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_pairs(path: Path) -> pd.DataFrame:
    columns = ["drug_id", "target_id", "smiles", "target_sequence", "label"]
    frame = pd.read_csv(path, sep=r"\s+", names=columns, dtype=str)
    if frame.isna().any().any():
        raise ValueError("Davis table contains missing fields")
    frame["label"] = frame["label"].astype(int)
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"Expected official binary labels, found {sorted(frame.label.unique())}")

    canonical = []
    for raw in frame["smiles"]:
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {raw}")
        canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    frame["canonical_smiles"] = canonical
    frame["target_sequence_hash"] = frame["target_sequence"].map(
        lambda value: hashlib.sha256(value.encode()).hexdigest()
    )
    frame["pair_id"] = frame["drug_id"] + "::" + frame["target_id"]
    return frame


def validate_identity(frame: pd.DataFrame) -> dict:
    if frame["pair_id"].duplicated().any():
        duplicated = frame.loc[frame["pair_id"].duplicated(), "pair_id"].head().tolist()
        raise ValueError(f"Duplicated drug-target identities: {duplicated}")

    drug_raw_counts = frame.groupby("drug_id")["smiles"].nunique()
    drug_canonical_counts = frame.groupby("drug_id")["canonical_smiles"].nunique()
    target_sequence_counts = frame.groupby("target_id")["target_sequence"].nunique()
    if drug_raw_counts.max() != 1 or drug_canonical_counts.max() != 1:
        raise ValueError("A drug ID maps to multiple SMILES representations")
    if target_sequence_counts.max() != 1:
        raise ValueError("A target ID maps to multiple protein sequences")

    drug_map = frame[["drug_id", "canonical_smiles"]].drop_duplicates()
    canonical_to_ids = drug_map.groupby("canonical_smiles")["drug_id"].nunique()
    canonical_collisions = canonical_to_ids[canonical_to_ids > 1]

    n_drugs = frame["canonical_smiles"].nunique()
    n_targets = frame["target_sequence_hash"].nunique()
    expected_pairs = n_drugs * n_targets
    per_drug_candidates = frame.groupby("canonical_smiles")["target_sequence_hash"].nunique()
    per_target_drugs = frame.groupby("target_sequence_hash")["canonical_smiles"].nunique()
    complete = (
        len(frame) == expected_pairs
        and per_drug_candidates.nunique() == 1
        and per_drug_candidates.iloc[0] == n_targets
        and per_target_drugs.nunique() == 1
        and per_target_drugs.iloc[0] == n_drugs
    )
    if not complete:
        raise ValueError(
            f"Davis is not a complete matrix: rows={len(frame)} expected={expected_pairs}"
        )

    return {
        "n_pairs": int(len(frame)),
        "n_drug_ids": int(frame["drug_id"].nunique()),
        "n_canonical_drugs": int(n_drugs),
        "n_target_ids": int(frame["target_id"].nunique()),
        "n_target_sequence_hashes": int(n_targets),
        "expected_complete_pairs": int(expected_pairs),
        "complete_drug_target_matrix": bool(complete),
        "candidate_count_min": int(per_drug_candidates.min()),
        "candidate_count_max": int(per_drug_candidates.max()),
        "positive_pairs": int(frame["label"].sum()),
        "negative_pairs": int((frame["label"] == 0).sum()),
        "canonical_collision_groups": int(len(canonical_collisions)),
        "canonical_collision_drug_ids": int(canonical_collisions.sum()) if len(canonical_collisions) else 0,
        "all_smiles_rdkit_parseable": True,
        "stable_drug_identity": bool(drug_raw_counts.max() == 1),
        "stable_target_identity": bool(target_sequence_counts.max() == 1),
    }


def quantile_bins(values: pd.Series, max_bins: int) -> pd.Series:
    ranks = values.rank(method="first")
    return pd.qcut(ranks, q=max_bins, labels=False, duplicates="drop")


def make_split(frame: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    per_drug = frame.groupby("canonical_smiles")["label"].agg(["sum", "count"]).reset_index()
    per_drug["negative"] = per_drug["count"] - per_drug["sum"]
    eligible = per_drug[(per_drug["sum"] > 0) & (per_drug["negative"] > 0)].copy()
    ineligible = per_drug.loc[~per_drug["canonical_smiles"].isin(eligible["canonical_smiles"]), "canonical_smiles"].tolist()

    n_total = len(per_drug)
    n_validation = int(round(0.15 * n_total))
    n_test = int(round(0.15 * n_total))
    n_train = n_total - n_validation - n_test
    if len(ineligible) > n_train:
        raise ValueError("Too many single-class drugs to keep validation/test query-valid")

    last_error = None
    for n_bins in range(5, 1, -1):
        try:
            bins = quantile_bins(eligible["sum"], n_bins)
            train_part, holdout = train_test_split(
                eligible,
                test_size=n_validation + n_test,
                random_state=seed,
                shuffle=True,
                stratify=bins,
            )
            holdout_bins = quantile_bins(holdout["sum"], min(n_bins, 5))
            validation, test = train_test_split(
                holdout,
                test_size=n_test,
                random_state=1000 + seed,
                shuffle=True,
                stratify=holdout_bins,
            )
            train_drugs = sorted(train_part["canonical_smiles"].tolist() + ineligible)
            validation_drugs = sorted(validation["canonical_smiles"].tolist())
            test_drugs = sorted(test["canonical_smiles"].tolist())
            if len(train_drugs) != n_train:
                raise ValueError(f"Unexpected train size {len(train_drugs)} != {n_train}")
            return {"train": train_drugs, "validation": validation_drugs, "test": test_drugs}
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"Could not create stratified split for seed {seed}: {last_error}")


def audit_split(frame: pd.DataFrame, split: dict[str, list[str]], seed: int) -> list[dict]:
    sets = {name: set(values) for name, values in split.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError(f"Seed {seed}: drug overlap across splits")
    if set.union(*sets.values()) != set(frame["canonical_smiles"].unique()):
        raise ValueError(f"Seed {seed}: split coverage mismatch")

    records = []
    for name in ("train", "validation", "test"):
        subset = frame[frame["canonical_smiles"].isin(sets[name])]
        per_query = subset.groupby("canonical_smiles")["label"].agg(["count", "sum"])
        if name in {"validation", "test"}:
            if (per_query["sum"] == 0).any() or (per_query["sum"] == per_query["count"]).any():
                raise ValueError(f"Seed {seed} {name}: single-class held-out query")
        records.append(
            {
                "seed": seed,
                "split": name,
                "n_drugs": len(sets[name]),
                "n_pairs": len(subset),
                "n_positive": int(subset["label"].sum()),
                "n_negative": int((subset["label"] == 0).sum()),
                "candidate_count_min": int(per_query["count"].min()),
                "candidate_count_max": int(per_query["count"].max()),
                "positive_per_drug_min": int(per_query["sum"].min()),
                "positive_per_drug_max": int(per_query["sum"].max()),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    split_dir = root / "splits"
    processed_dir = root / "data" / "processed"
    qc_dir = root / "qc"
    for path in (split_dir, processed_dir, qc_dir):
        path.mkdir(parents=True, exist_ok=True)

    frame = load_pairs(args.input.resolve())
    identity_audit = validate_identity(frame)
    source_sha = sha256_file(args.input.resolve())

    ordered_columns = [
        "pair_id",
        "drug_id",
        "smiles",
        "canonical_smiles",
        "target_id",
        "target_sequence",
        "target_sequence_hash",
        "label",
    ]
    frame[ordered_columns].to_csv(processed_dir / "davis_pairs.csv", index=False)

    summary_records = []
    pair_assignment_records = []
    for seed in SEEDS:
        split = make_split(frame, seed)
        summary_records.extend(audit_split(frame, split, seed))
        drug_lookup = (
            frame[["canonical_smiles", "drug_id", "smiles"]]
            .drop_duplicates()
            .sort_values(["canonical_smiles", "drug_id"])
        )
        entries = {}
        for name, canonical_values in split.items():
            current = drug_lookup[drug_lookup["canonical_smiles"].isin(canonical_values)]
            entries[name] = current.to_dict(orient="records")
            subset = frame[frame["canonical_smiles"].isin(canonical_values)]
            for pair_id in subset["pair_id"]:
                pair_assignment_records.append({"seed": seed, "pair_id": pair_id, "split": name})

        payload = {
            "dataset": "Davis",
            "seed": seed,
            "query": "canonical_drug",
            "candidate": "target_protein",
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "actual_drug_counts": {name: len(values) for name, values in split.items()},
            "source_path": str(args.input.resolve()),
            "source_sha256": source_sha,
            "label_protocol": "official MMDG-DTI Davis.txt binary label used unchanged",
            "affinity_threshold": "not_reapplied_official_binary_label_as_supplied",
            "stratification": "positive-count quantile strata; single-class drugs forced to train",
            "train_balancing": "none_at_manifest_stage",
            "validation_test_balancing": "forbidden_all_candidates_retained",
            "drugs": entries,
        }
        payload["content_sha256"] = canonical_json_sha256(payload)
        with (split_dir / f"davis_seed_{seed}.json").open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    summary = pd.DataFrame(summary_records)
    summary.to_csv(split_dir / "davis_formal25_split_summary.csv", index=False)
    pd.DataFrame(pair_assignment_records).to_csv(
        split_dir / "davis_formal25_pair_assignments.csv", index=False
    )
    qc_payload = {
        "source_sha256": source_sha,
        "identity_and_matrix_audit": identity_audit,
        "split_files": {
            str(seed): sha256_file(split_dir / f"davis_seed_{seed}.json") for seed in SEEDS
        },
        "split_summary_sha256": sha256_file(split_dir / "davis_formal25_split_summary.csv"),
        "status": "PASS",
    }
    with (qc_dir / "davis_data_and_split_audit.json").open("w") as handle:
        json.dump(qc_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(qc_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
