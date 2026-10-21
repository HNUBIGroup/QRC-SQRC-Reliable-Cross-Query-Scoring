from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(
    "/home/u2508283030/MDA_CellPainting_Multiview_20260910"
)

OUT = Path(
    "/home/u2508283030/ECARF_Article2_20260911/02_results"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


def metric(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    return (
        roc_auc_score(y, s),
        average_precision_score(y, s),
    )


def macro_metric(
    df,
    score_col,
    query_col,
):
    aucs = []
    aprs = []

    for _, g in df.groupby(query_col):

        y = g["y"].to_numpy()

        if len(np.unique(y)) < 2:
            continue

        s = g[score_col].to_numpy()

        aucs.append(
            roc_auc_score(y, s)
        )

        aprs.append(
            average_precision_score(y, s)
        )

    return (
        np.mean(aucs)
        if aucs else np.nan,

        np.mean(aprs)
        if aprs else np.nan,
    )


def bias_stats(
    df,
    query_col,
):
    """
    Estimate per-query residual mean and its sampling variance.

    No labels are used.
    """

    stats = []

    for q, g in df.groupby(query_col):

        d = g["_delta"].to_numpy()

        n = len(d)

        b = float(
            np.mean(d)
        )

        if n > 1:
            var = float(
                np.var(
                    d,
                    ddof=1
                )
            )
        else:
            var = 0.0

        se2 = var / max(
            n,
            1
        )

        stats.append({
            "query": q,
            "n": n,
            "bias": b,
            "se2": se2,
        })

    return pd.DataFrame(stats)


records = []


for mode in [
    "drug",
    "mirna",
]:

    query_col = (
        "drug"
        if mode == "drug"
        else "miRNA"
    )

    print(
        f"\n===== {mode.upper()}-COLD =====",
        flush=True,
    )

    for seed in range(25):

        print(
            f"{mode.upper()} seed {seed}/24",
            flush=True,
        )

        tag = (
            f"{mode}_cold_seed{seed}"
        )

        score_file = (
            ROOT
            / "05_results"
            / tag
            / f"all_pair_scores_{tag}.tsv"
        )

        split_dir = (
            ROOT
            / "03_splits"
            / tag
        )

        x = pd.read_csv(
            score_file,
            sep="\t",
        )

        val_entities = set(
            pd.read_csv(
                split_dir
                / "val_entities.tsv",
                sep="\t",
            )["entity"]
        )

        test_entities = set(
            pd.read_csv(
                split_dir
                / "test_entities.tsv",
                sep="\t",
            )["entity"]
        )

        # ==================================================
        # VALIDATION QUERY SET
        # Used ONLY to estimate shrinkage hyper-variance.
        # No validation labels used.
        # ==================================================

        val = x[
            x[query_col].isin(
                val_entities
            )
        ].copy().reset_index(
            drop=True
        )

        val["_delta"] = (
            val["final_score"]
            - val["molecular_score"]
        )

        val_stats = bias_stats(
            val,
            query_col,
        )

        if len(val_stats) > 1:

            observed_var = float(
                np.var(
                    val_stats["bias"],
                    ddof=1,
                )
            )

        else:
            observed_var = 0.0

        mean_noise_var = float(
            val_stats["se2"].mean()
        )

        # Method-of-moments estimate:
        # observed between-query variance
        # minus estimation noise.
        tau2 = max(
            observed_var
            - mean_noise_var,
            0.0,
        )

        # ==================================================
        # TEST QUERY SET
        # ==================================================

        test = x[
            x[query_col].isin(
                test_entities
            )
        ].copy().reset_index(
            drop=True
        )

        test["y"] = (
            test["split"]
            == "test_known"
        ).astype(int)

        test["_delta"] = (
            test["final_score"]
            - test["molecular_score"]
        )

        test_stats = bias_stats(
            test,
            query_col,
        )

        # Standard QRC
        bias_map = dict(
            zip(
                test_stats["query"],
                test_stats["bias"],
            )
        )

        # Shrinkage weights
        weight_map = {}

        for _, row in (
            test_stats.iterrows()
        ):

            q = row["query"]
            se2 = float(
                row["se2"]
            )

            denom = (
                tau2
                + se2
            )

            if denom <= 0:
                w = 0.0
            else:
                w = (
                    tau2
                    / denom
                )

            weight_map[q] = w

        mol = (
            test[
                "molecular_score"
            ]
            .to_numpy()
        )

        raw = (
            test[
                "final_score"
            ]
            .to_numpy()
        )

        delta = (
            test["_delta"]
            .to_numpy()
        )

        b = np.array([
            bias_map[q]
            for q
            in test[query_col]
        ])

        w = np.array([
            weight_map[q]
            for q
            in test[query_col]
        ])

        qrc = (
            mol
            + delta
            - b
        )

        sqrc = (
            mol
            + delta
            - w * b
        )

        test["Molecular"] = mol
        test["Raw"] = raw
        test["QRC"] = qrc
        test["SQRC"] = sqrc

        y = test["y"].to_numpy()

        row = {
            "mode": mode,
            "seed": seed,
            "tau2": tau2,
            "observed_bias_var":
                observed_var,
            "validation_noise_var":
                mean_noise_var,
            "mean_weight":
                float(
                    test_stats[
                        "query"
                    ]
                    .map(
                        weight_map
                    )
                    .mean()
                ),
            "min_weight":
                float(
                    min(
                        weight_map.values()
                    )
                ),
            "max_weight":
                float(
                    max(
                        weight_map.values()
                    )
                ),
        }

        for name in [
            "Molecular",
            "Raw",
            "QRC",
            "SQRC",
        ]:

            auc, apr = metric(
                y,
                test[name].to_numpy()
            )

            mauc, mapr = (
                macro_metric(
                    test,
                    name,
                    query_col,
                )
            )

            row[
                f"{name}_Pooled_AUROC"
            ] = auc

            row[
                f"{name}_Pooled_AUPR"
            ] = apr

            row[
                f"{name}_Macro_AUROC"
            ] = mauc

            row[
                f"{name}_Macro_AUPR"
            ] = mapr

        records.append(row)


df = pd.DataFrame(
    records
)

df.to_csv(
    OUT
    / "stageE_shrinkage_qrc_25seed.tsv",
    sep="\t",
    index=False,
)


print(
    "\n===== SHRINKAGE QRC RESULTS ====="
)

for mode in [
    "drug",
    "mirna",
]:

    print(
        f"\n### {mode.upper()}-COLD"
    )

    z = df[
        df["mode"] == mode
    ].copy()

    for name in [
        "Molecular",
        "Raw",
        "QRC",
        "SQRC",
    ]:

        auc = z[
            f"{name}_Pooled_AUROC"
        ]

        apr = z[
            f"{name}_Pooled_AUPR"
        ]

        mauc = z[
            f"{name}_Macro_AUROC"
        ]

        print(
            f"{name:10s} | "
            f"pooled AUC="
            f"{auc.mean():.4f} ± "
            f"{auc.std(ddof=1):.4f} | "
            f"AUPR="
            f"{apr.mean():.4f} | "
            f"macro AUC="
            f"{mauc.mean():.4f}"
        )

    d_raw = (
        z["SQRC_Pooled_AUROC"]
        - z["Raw_Pooled_AUROC"]
    )

    d_qrc = (
        z["SQRC_Pooled_AUROC"]
        - z["QRC_Pooled_AUROC"]
    )

    d_mol = (
        z["SQRC_Pooled_AUROC"]
        - z["Molecular_Pooled_AUROC"]
    )

    print()
    print(
        "SQRC vs Raw:",
        f"{d_raw.mean():+.4f} ± "
        f"{d_raw.std(ddof=1):.4f}",
        f"positive={(d_raw > 0).sum()}/25",
    )

    print(
        "SQRC vs QRC:",
        f"{d_qrc.mean():+.4f} ± "
        f"{d_qrc.std(ddof=1):.4f}",
        f"positive={(d_qrc > 0).sum()}/25",
    )

    print(
        "SQRC vs Molecular:",
        f"{d_mol.mean():+.4f} ± "
        f"{d_mol.std(ddof=1):.4f}",
        f"positive={(d_mol > 0).sum()}/25",
    )

    print(
        "tau2:",
        f"{z.tau2.mean():.6e}",
    )

    print(
        "mean shrinkage weight:",
        f"{z.mean_weight.mean():.4f}",
    )


print(
    "\nSHRINKAGE_QRC_COMPLETE"
)
