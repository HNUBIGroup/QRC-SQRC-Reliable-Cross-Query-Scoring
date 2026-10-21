#!/usr/bin/env python3
"""Build shared MMDG-DTI structural and official PLM features once per entity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer


PROTEIN_DICT = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
    "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14,
    "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21,
    "V": 22, "Y": 23, "X": 24, "Z": 25,
}

CHAR_SMI_SET = {
    "(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
    "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15,
    "R": 16, "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22,
    "f": 23, "h": 24, "l": 25, "n": 26, "r": 27, "t": 28, "#": 29,
    "%": 30, ")": 31, "+": 32, "-": 33, "/": 34, "1": 35, "3": 36,
    "5": 37, "7": 38, "9": 39, "=": 40, "A": 41, "C": 42, "E": 43,
    "G": 44, "I": 45, "K": 46, "M": 47, "O": 48, "S": 49, "U": 50,
    "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56, "e": 57,
    "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vendor-code", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reuse-existing-structural",
        action="store_true",
        help="Require and preserve an existing structural_features.pkl instead of rebuilding it.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entity_filename(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest() + ".npy"


def smile_to_graph(smile: str, atom_features) -> tuple[np.ndarray, np.ndarray]:
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.asarray([atom_features(atom) for atom in molecule.GetAtoms()])
    return node_features, adjacency


def save_pickle(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)
    os.replace(temporary, path)


def encode_entities(
    entities: list[str],
    entity_dir: Path,
    tokenizer,
    model,
    device: torch.device,
    protein: bool,
) -> dict:
    entity_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    completed = 0
    for index, entity in enumerate(entities, start=1):
        destination = entity_dir / entity_filename(entity)
        if destination.exists():
            completed += 1
            continue
        token_text = " ".join(entity) if protein else entity
        encoded = tokenizer.batch_encode_plus(
            [token_text], add_special_tokens=True, padding=True
        )
        input_ids = torch.tensor(encoded["input_ids"], device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        array = output.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)
        temporary = destination.with_suffix(".tmp.npy")
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, destination)
        completed += 1
        print(
            f"{'protein' if protein else 'drug'} {index}/{len(entities)} "
            f"tokens={array.shape[0]} hidden={array.shape[1]}",
            flush=True,
        )
    return {"requested": len(entities), "completed": completed}


def load_entity_arrays(entities: list[str], entity_dir: Path) -> dict[str, np.ndarray]:
    arrays = {}
    for entity in entities:
        arrays[entity] = np.load(entity_dir / entity_filename(entity), allow_pickle=False)
    return arrays


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    hf_home = args.hf_home.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    vendor_code = args.vendor_code.resolve()
    sys.path.insert(0, str(vendor_code))
    from graph_features import atom_features

    pairs = pd.read_csv(args.pairs.resolve())
    drug_table = pairs[["smiles", "canonical_smiles", "drug_id"]].drop_duplicates()
    target_table = pairs[["target_sequence", "target_sequence_hash", "target_id"]].drop_duplicates()
    raw_smiles = sorted(drug_table["smiles"].tolist())
    sequences = sorted(target_table["target_sequence"].tolist())

    unknown_smiles_chars = sorted(set("".join(raw_smiles)) - set(CHAR_SMI_SET))
    unknown_protein_chars = sorted(set("".join(sequences)) - set(PROTEIN_DICT))
    if unknown_smiles_chars:
        raise ValueError(f"Official character encoder cannot encode: {unknown_smiles_chars}")
    if unknown_protein_chars:
        raise ValueError(f"Official protein encoder cannot encode: {unknown_protein_chars}")

    structural_path = output_dir / "structural_features.pkl"
    if args.reuse_existing_structural:
        if not structural_path.is_file():
            raise FileNotFoundError(
                f"--reuse-existing-structural requires {structural_path}"
            )
        print(f"reusing_structural_features={structural_path}", flush=True)
    else:
        structural = {"drugs": {}, "targets": {}}
        for row in drug_table.itertuples(index=False):
            atoms, adjacency = smile_to_graph(row.smiles, atom_features)
            structural["drugs"][row.smiles] = {
                "drug_id": row.drug_id,
                "canonical_smiles": row.canonical_smiles,
                "molecule_words": np.asarray([CHAR_SMI_SET[c] for c in row.smiles], dtype=np.int64),
                "molecule_atoms": atoms.astype(np.float32),
                "molecule_adjs": adjacency.astype(np.float32),
            }
        for row in target_table.itertuples(index=False):
            structural["targets"][row.target_sequence] = {
                "target_id": row.target_id,
                "target_sequence_hash": row.target_sequence_hash,
                "proteins": np.asarray([PROTEIN_DICT[c] for c in row.target_sequence], dtype=np.int64),
            }
        save_pickle(structural_path, structural)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    protein_model_name = "Rostlab/prot_bert_bfd"
    protein_tokenizer = AutoTokenizer.from_pretrained(
        protein_model_name, do_lower_case=False, cache_dir=hf_home
    )
    protein_model = AutoModel.from_pretrained(protein_model_name, cache_dir=hf_home).to(device)
    protein_stats = encode_entities(
        sequences,
        output_dir / "protein_lm_entities",
        protein_tokenizer,
        protein_model,
        device,
        protein=True,
    )
    protein_commit = getattr(protein_model.config, "_commit_hash", None)
    del protein_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    drug_model_name = "seyonec/PubChem10M_SMILES_BPE_450k"
    drug_tokenizer = AutoTokenizer.from_pretrained(
        drug_model_name, do_lower_case=False, cache_dir=hf_home
    )
    drug_model = AutoModel.from_pretrained(drug_model_name, cache_dir=hf_home).to(device)
    drug_stats = encode_entities(
        raw_smiles,
        output_dir / "drug_lm_entities",
        drug_tokenizer,
        drug_model,
        device,
        protein=False,
    )
    drug_commit = getattr(drug_model.config, "_commit_hash", None)
    del drug_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    protein_lm = load_entity_arrays(sequences, output_dir / "protein_lm_entities")
    drug_lm = load_entity_arrays(raw_smiles, output_dir / "drug_lm_entities")
    save_pickle(output_dir / "p_LM.pkl", protein_lm)
    save_pickle(output_dir / "d_LM.pkl", drug_lm)

    protein_shapes = [value.shape for value in protein_lm.values()]
    drug_shapes = [value.shape for value in drug_lm.values()]
    metadata = {
        "status": "PASS",
        "pairs_source": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs.resolve()),
        "protein_model": protein_model_name,
        "protein_model_commit": protein_commit,
        "drug_model": drug_model_name,
        "drug_model_commit": drug_commit,
        "official_tokenization_preserved": True,
        "structural_features_reused": bool(args.reuse_existing_structural),
        "protein_entities": protein_stats,
        "drug_entities": drug_stats,
        "protein_embedding_shape_min": [int(min(s[i] for s in protein_shapes)) for i in (0, 1)],
        "protein_embedding_shape_max": [int(max(s[i] for s in protein_shapes)) for i in (0, 1)],
        "drug_embedding_shape_min": [int(min(s[i] for s in drug_shapes)) for i in (0, 1)],
        "drug_embedding_shape_max": [int(max(s[i] for s in drug_shapes)) for i in (0, 1)],
        "p_LM_sha256": sha256_file(output_dir / "p_LM.pkl"),
        "d_LM_sha256": sha256_file(output_dir / "d_LM.pkl"),
        "structural_features_sha256": sha256_file(output_dir / "structural_features.pkl"),
    }
    with (output_dir / "feature_cache_manifest.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
