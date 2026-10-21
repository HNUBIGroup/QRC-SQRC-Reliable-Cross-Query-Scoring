#!/usr/bin/env python3
"""Aggregate completed formal MMDG-DTI/QRC seed outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

local_packages = Path(__file__).resolve().parents[1] / "cache/python_packages"
if local_packages.is_dir():
    sys.path.insert(0, str(local_packages))

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    seeds = range(5, 30)

    metric_frames = [pd.read_csv(root / f"results/seed_{seed}_metrics.csv") for seed in seeds]
    per_seed = pd.concat(metric_frames, ignore_index=True)
    per_seed.to_csv(root / "results/per_seed_metrics.csv", index=False)

    numeric = [column for column in per_seed.columns if column not in {"seed", "method", "macro_invariance_qc"}]
    aggregate = per_seed.groupby("method", sort=False)[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
    aggregate.reset_index().to_csv(root / "results/aggregate_metrics.csv", index=False)

    query_frames = [
        pd.read_csv(root / f"results/seed_{seed}_query_shift_statistics.csv")
        for seed in seeds
    ]
    pd.concat(query_frames, ignore_index=True).to_csv(
        root / "results/query_shift_statistics.csv", index=False
    )

    prediction_frames = [
        pd.read_csv(root / f"predictions/seed_{seed}_test_pairs_qrc.csv")
        for seed in seeds
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(root / "results/all_test_predictions.parquet", index=False)

    qc = [json.loads((root / f"qc/seed_{seed}_qrc_qc.json").read_text()) for seed in seeds]
    if any(item["status"] != "PASS" for item in qc):
        raise RuntimeError("At least one formal QRC QC failed")
    report = [
        "# MMDG-DTI / DTI_QRC Formal 25-Seed Confirmatory Experiment",
        "",
        "Status: **COMPLETE**",
        "",
        "All 25 preregistered strict drug-cold confirmatory seeds completed with validation-selected checkpoints, complete held-out candidate exports, and QRC invariant QC PASS.",
        "",
        "See `results/per_seed_metrics.csv`, `results/aggregate_metrics.csv`, `results/query_shift_statistics.csv`, and `results/all_test_predictions.parquet`.",
    ]
    (root / "reports/formal25_report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
