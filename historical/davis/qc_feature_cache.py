#!/usr/bin/env python3
"""Validate a completed MMDG-DTI feature cache without rebuilding it."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(args.pairs.resolve())
    feature_dir = args.features_dir.resolve()
    required = [
        feature_dir / "structural_features.pkl",
        feature_dir / "p_LM.pkl",
        feature_dir / "d_LM.pkl",
        feature_dir / "feature_cache_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete feature cache; missing: {missing}")

    with (feature_dir / "structural_features.pkl").open("rb") as handle:
        structural = pickle.load(handle)
    with (feature_dir / "p_LM.pkl").open("rb") as handle:
        protein_lm = pickle.load(handle)
    with (feature_dir / "d_LM.pkl").open("rb") as handle:
        drug_lm = pickle.load(handle)

    smiles = set(pairs["smiles"])
    sequences = set(pairs["target_sequence"])
    failures = []
    if len(pairs) != 25772:
        failures.append(f"pair_count={len(pairs)}")
    if pairs["pair_id"].duplicated().any():
        failures.append("duplicate_pair_id")
    if set(structural["drugs"]) != smiles:
        failures.append("structural_drug_alignment")
    if set(structural["targets"]) != sequences:
        failures.append("structural_target_alignment")
    if set(drug_lm) != smiles:
        failures.append("drug_lm_alignment")
    if set(protein_lm) != sequences:
        failures.append("protein_lm_alignment")

    drug_shapes = sorted({tuple(value.shape) for value in drug_lm.values()})
    protein_shapes = sorted({tuple(value.shape) for value in protein_lm.values()})
    if any(len(shape) != 2 or shape[1] != 768 for shape in drug_shapes):
        failures.append("drug_lm_hidden_dimension")
    if any(len(shape) != 2 or shape[1] != 1024 for shape in protein_shapes):
        failures.append("protein_lm_hidden_dimension")
    if any(not np.isfinite(value).all() for value in drug_lm.values()):
        failures.append("drug_lm_nonfinite")
    if any(not np.isfinite(value).all() for value in protein_lm.values()):
        failures.append("protein_lm_nonfinite")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "pair_count": int(len(pairs)),
        "drug_entities": len(smiles),
        "protein_entities": len(sequences),
        "drug_lm_shapes": [list(shape) for shape in drug_shapes],
        "protein_lm_shape_min": list(min(protein_shapes)),
        "protein_lm_shape_max": list(max(protein_shapes)),
        "sha256": {path.name: sha256_file(path) for path in required},
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
