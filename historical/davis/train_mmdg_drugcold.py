#!/usr/bin/env python3
"""Leakage-free MMDG-DTI training and complete held-out pair export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--vendor-code", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--max-epochs", type=int, default=119)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.07)
    parser.add_argument("--lr-decay-interval", type=int, default=5)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--smoke-max-train-batches",
        type=int,
        default=None,
        help="Smoke-only cap per epoch; formal runs reject this option.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


class FeatureStore:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.structural = load_pickle(self.directory / "structural_features.pkl")
        self.protein_lm = load_pickle(self.directory / "p_LM.pkl")
        self.drug_lm = load_pickle(self.directory / "d_LM.pkl")

    def validate(self, pairs: pd.DataFrame) -> None:
        smiles = set(pairs["smiles"])
        sequences = set(pairs["target_sequence"])
        missing = {
            "structural_drugs": sorted(smiles - set(self.structural["drugs"])),
            "structural_targets": sorted(sequences - set(self.structural["targets"])),
            "drug_lm": sorted(smiles - set(self.drug_lm)),
            "protein_lm": sorted(sequences - set(self.protein_lm)),
        }
        if any(missing.values()):
            raise ValueError(f"Feature coverage failure: {missing}")


def split_pairs(pairs: pd.DataFrame, manifest: dict) -> dict[str, pd.DataFrame]:
    outputs = {}
    seen = set()
    for split in ("train", "validation", "test"):
        drugs = {item["canonical_smiles"] for item in manifest["drugs"][split]}
        if seen & drugs:
            raise ValueError("Drug overlap in manifest")
        seen |= drugs
        subset = pairs[pairs["canonical_smiles"].isin(drugs)].copy()
        subset = subset.sort_values(["canonical_smiles", "target_id", "pair_id"]).reset_index(drop=True)
        outputs[split] = subset
    if sum(len(value) for value in outputs.values()) != len(pairs):
        raise ValueError("Split pair coverage mismatch")
    return outputs


def pack_rows(rows: pd.DataFrame, store: FeatureStore, device: torch.device):
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError("Cannot pack an empty batch")
    drug_features = [store.structural["drugs"][value] for value in rows["smiles"]]
    target_features = [store.structural["targets"][value] for value in rows["target_sequence"]]
    max_atoms = max(value["molecule_atoms"].shape[0] for value in drug_features)

    molecule_words = torch.zeros((n_rows, 100), dtype=torch.long, device=device)
    molecule_atoms = torch.zeros((n_rows, max_atoms, 75), dtype=torch.float32, device=device)
    molecule_adjs = torch.zeros((n_rows, max_atoms, max_atoms), dtype=torch.float32, device=device)
    proteins = torch.zeros((n_rows, 1200), dtype=torch.long, device=device)
    protein_lm = torch.zeros((n_rows, 1200, 1024), dtype=torch.float32, device=device)
    molecule_lm = torch.zeros((n_rows, 100, 768), dtype=torch.float32, device=device)

    for index, (row, drug, target) in enumerate(
        zip(rows.itertuples(index=False), drug_features, target_features)
    ):
        words = torch.as_tensor(drug["molecule_words"][:100], dtype=torch.long, device=device)
        atoms = torch.as_tensor(drug["molecule_atoms"], dtype=torch.float32, device=device)
        adjacency = torch.as_tensor(drug["molecule_adjs"], dtype=torch.float32, device=device)
        protein = torch.as_tensor(target["proteins"][:1200], dtype=torch.long, device=device)
        p_lm_array = store.protein_lm[row.target_sequence][:1200]
        d_lm_array = store.drug_lm[row.smiles][:100]
        p_lm = torch.as_tensor(p_lm_array, dtype=torch.float32, device=device)
        d_lm = torch.as_tensor(d_lm_array, dtype=torch.float32, device=device)

        molecule_words[index, : len(words)] = words
        molecule_atoms[index, : atoms.shape[0]] = atoms
        molecule_adjs[index, : adjacency.shape[0], : adjacency.shape[1]] = adjacency
        molecule_adjs[index, : adjacency.shape[0], : adjacency.shape[1]] += torch.eye(
            adjacency.shape[0], device=device
        )
        proteins[index, : len(protein)] = protein
        protein_lm[index, : p_lm.shape[0]] = p_lm
        molecule_lm[index, : d_lm.shape[0]] = d_lm

    labels = torch.as_tensor(rows["label"].to_numpy(), dtype=torch.long, device=device)
    sources = torch.zeros(n_rows, dtype=torch.long, device=device)
    return (
        molecule_words,
        molecule_atoms,
        molecule_adjs,
        proteins,
        protein_lm,
        molecule_lm,
        labels,
        sources,
    )


def iter_batches(frame: pd.DataFrame, batch_size: int, order: np.ndarray | None = None):
    if order is None:
        order = np.arange(len(frame))
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        yield frame.iloc[indices]


def train_epoch(
    model, frame, store, optimizer, device, batch_size, seed, epoch, max_batches=None
):
    model.train()
    order = np.random.default_rng(seed + epoch * 100003).permutation(len(frame))
    losses = []
    for batch_index, rows in enumerate(iter_batches(frame, batch_size, order), start=1):
        batch = pack_rows(rows, store, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, epoch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if max_batches is not None and batch_index >= max_batches:
            break
    if not losses:
        raise RuntimeError("No training batches executed")
    return float(np.mean(losses)), len(losses)


def evaluate(model, frame, store, device, batch_size) -> tuple[dict, pd.DataFrame]:
    model.eval()
    outputs = []
    with torch.no_grad():
        for rows in iter_batches(frame, batch_size):
            batch = pack_rows(rows, store, device)
            inputs = batch[:-2]
            identity_index = torch.arange(len(rows), device=device)
            result = model.forward(inputs, identity_index)
            fusion_logits, base_logits, aux_logits = result[0], result[1], result[2]
            raw_logits = 0.4 * fusion_logits + 0.27 * base_logits + 0.33 * aux_logits

            exported = rows[
                [
                    "pair_id",
                    "drug_id",
                    "smiles",
                    "canonical_smiles",
                    "target_id",
                    "target_sequence_hash",
                    "label",
                ]
            ].copy()
            for name, logits in (
                ("base", base_logits),
                ("aux", aux_logits),
                ("fusion", fusion_logits),
                ("raw", raw_logits),
            ):
                array = logits.detach().cpu().numpy()
                exported[f"{name}_logit"] = array[:, 1] - array[:, 0]
                exported[f"{name}_prob"] = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            outputs.append(exported)

    predictions = pd.concat(outputs, ignore_index=True)
    if len(predictions) != len(frame):
        raise RuntimeError(f"Silent sample drop: exported={len(predictions)} expected={len(frame)}")
    if predictions["pair_id"].duplicated().any():
        raise RuntimeError("Duplicate pair IDs in export")
    if predictions["pair_id"].tolist() != frame["pair_id"].tolist():
        raise RuntimeError("Pair order mismatch in aligned score export")
    labels = predictions["label"].to_numpy()
    scores = predictions["raw_logit"].to_numpy()
    metrics = {
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "pooled_aupr": float(average_precision_score(labels, scores)),
        "n_pairs": int(len(predictions)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
    }
    return metrics, predictions


def is_better(metrics: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if metrics["pooled_aupr"] > best["pooled_aupr"] + 1e-12:
        return True
    return (
        abs(metrics["pooled_aupr"] - best["pooled_aupr"]) <= 1e-12
        and metrics["pooled_auroc"] > best["pooled_auroc"] + 1e-12
    )


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def git_metadata(repo: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--short"),
        "diff_sha256": hashlib.sha256(run("diff").encode()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    if args.seed not in set(range(5, 30)):
        raise ValueError("Only preregistered confirmatory seeds 5..29 are allowed")
    if args.max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        raise ValueError("eval-batch-size must be positive")
    if args.smoke_max_train_batches is not None:
        if args.run_kind != "smoke":
            raise ValueError("smoke-max-train-batches is prohibited for formal runs")
        if args.smoke_max_train_batches < 1:
            raise ValueError("smoke-max-train-batches must be positive")
    eval_batch_size = args.eval_batch_size or args.batch_size

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("MMDG-DTI training requires the allocated GPU")

    root = args.output_root.resolve()
    run_subdir = Path("smoke") if args.run_kind == "smoke" else Path()
    checkpoint_dir = root / "checkpoints" / run_subdir
    prediction_dir = root / "predictions" / run_subdir
    log_dir = root / "logs" / run_subdir
    for path in (checkpoint_dir, prediction_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = args.split_manifest.resolve()
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if int(manifest["seed"]) != args.seed:
        raise ValueError("Seed does not match split manifest")

    pairs = pd.read_csv(args.pairs.resolve())
    splits = split_pairs(pairs, manifest)
    store = FeatureStore(args.features_dir.resolve())
    store.validate(pairs)

    sys.path.insert(0, str(args.vendor_code.resolve()))
    from networks.model import DGMM_DTI

    model = DGMM_DTI(
        layer_gnn=3, source_number=3, device=device, dropout=0.05
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    checkpoint_path = checkpoint_dir / f"seed_{args.seed}_best.pt"
    history_path = log_dir / f"seed_{args.seed}_epoch_history.csv"
    run_manifest_path = log_dir / f"seed_{args.seed}_run_manifest.json"
    vendor_repo = args.vendor_code.resolve().parent
    command = " ".join(sys.argv)
    run_metadata = {
        "seed": args.seed,
        "run_kind": args.run_kind,
        "command": command,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "pairs_sha256": sha256_file(args.pairs.resolve()),
        "feature_manifest_sha256": sha256_file(args.features_dir.resolve() / "feature_cache_manifest.json"),
        "vendor": git_metadata(vendor_repo),
        "selection_metric": "validation pooled AUPR, then pooled AUROC, then earlier epoch",
        "test_evaluation_policy": "once after reloading validation-selected checkpoint",
        "train_balancing": "none; all train pairs retained",
        "validation_test_balancing": "none; all candidates retained",
        "hyperparameters": vars(args) | {"split_manifest": str(args.split_manifest), "pairs": str(args.pairs), "features_dir": str(args.features_dir), "vendor_code": str(args.vendor_code), "output_root": str(args.output_root)},
    }
    print(json.dumps(run_metadata, indent=2, default=str), flush=True)

    best_metrics = None
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    start_time = time.time()
    for epoch in range(1, args.max_epochs + 1):
        if epoch % args.lr_decay_interval == 0:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_decay
        train_loss, n_batches = train_epoch(
            model,
            splits["train"],
            store,
            optimizer,
            device,
            args.batch_size,
            args.seed,
            epoch,
            args.smoke_max_train_batches,
        )
        validation_metrics, _ = evaluate(
            model, splits["validation"], store, device, eval_batch_size
        )
        current = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_batches": n_batches,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_pooled_auroc": validation_metrics["pooled_auroc"],
            "validation_pooled_aupr": validation_metrics["pooled_aupr"],
            "elapsed_seconds": time.time() - start_time,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        history.append(current)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(json.dumps(current, sort_keys=True), flush=True)

        if is_better(validation_metrics, best_metrics):
            best_metrics = validation_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "run_metadata": run_metadata,
                },
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, validation_predictions = evaluate(
        model, splits["validation"], store, device, eval_batch_size
    )
    test_metrics, test_predictions = evaluate(
        model, splits["test"], store, device, eval_batch_size
    )
    validation_predictions.insert(0, "split", "validation")
    validation_predictions.insert(0, "seed", args.seed)
    test_predictions.insert(0, "split", "test")
    test_predictions.insert(0, "seed", args.seed)
    validation_path = prediction_dir / f"seed_{args.seed}_validation_pairs.csv"
    test_path = prediction_dir / f"seed_{args.seed}_test_pairs.csv"
    validation_predictions.to_csv(validation_path, index=False)
    test_predictions.to_csv(test_path, index=False)

    run_metadata.update(
        {
            "status": "COMPLETE",
            "best_epoch": int(best_epoch),
            "best_validation_metrics": validation_metrics,
            "final_test_metrics": test_metrics,
            "validation_export_rows": int(len(validation_predictions)),
            "test_export_rows": int(len(test_predictions)),
            "validation_export_sha256": sha256_file(validation_path),
            "test_export_sha256": sha256_file(test_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "walltime_seconds": time.time() - start_time,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
    )
    with run_manifest_path.open("w") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps(run_metadata, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
