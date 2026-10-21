from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = Path("/home/u2508283030/MDA_CellPainting_Multiview_20260910")
ROOT = Path("/home/u2508283030/ECARF_Article2_20260911")
OUT = ROOT / "02_results"
OUT.mkdir(parents=True, exist_ok=True)


def metric(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return (
        roc_auc_score(y, s),
        average_precision_score(y, s),
    )


def bias_stats(df, query_col):
    out = []

    for q, g in df.groupby(query_col):
        d = g["_delta"].to_numpy()
        n = len(d)

        b = float(np.mean(d))

        if n > 1:
            var = float(np.var(d, ddof=1))
        else:
            var = 0.0

        out.append({
            "query": q,
            "bias": b,
            "se2": var / max(n, 1),
        })

    return pd.DataFrame(out)


def run_one(val, test, query_col, seed, split_name):

    val = val.copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)

    val["_delta"] = (
        val["final_score"]
        - val["molecular_score"]
    )

    test["_delta"] = (
        test["final_score"]
        - test["molecular_score"]
    )

    # ----------------------------
    # Validation estimates tau^2
    # ----------------------------

    vs = bias_stats(
        val,
        query_col
    )

    if len(vs) > 1:
        observed_var = float(
            np.var(
                vs["bias"],
                ddof=1
            )
        )
    else:
        observed_var = 0.0

    noise_var = float(
        vs["se2"].mean()
    )

    tau2 = max(
        observed_var - noise_var,
        0.0
    )

    # ----------------------------
    # Test QRC / SQRC
    # ----------------------------

    ts = bias_stats(
        test,
        query_col
    )

    bias_map = dict(
        zip(
            ts["query"],
            ts["bias"]
        )
    )

    weight_map = {}

    for _, r in ts.iterrows():

        denom = (
            tau2
            + float(r["se2"])
        )

        if denom <= 0:
            w = 0.0
        else:
            w = (
                tau2
                / denom
            )

        weight_map[
            r["query"]
        ] = w

    mol = (
        test["molecular_score"]
        .to_numpy()
    )

    raw = (
        test["final_score"]
        .to_numpy()
    )

    delta = (
        test["_delta"]
        .to_numpy()
    )

    b = np.array([
        bias_map[q]
        for q in test[query_col]
    ])

    w = np.array([
        weight_map[q]
        for q in test[query_col]
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

    y = test["y"].to_numpy()

    row = {
        "split": split_name,
        "seed": seed,
        "n_pairs": len(test),
        "n_pos": int(y.sum()),
        "tau2": tau2,
        "mean_weight":
            float(np.mean(w)),
    }

    for name, score in [
        ("Molecular", mol),
        ("Raw", raw),
        ("QRC", qrc),
        ("SQRC", sqrc),
    ]:

        auc, apr = metric(
            y,
            score
        )

        row[
            f"{name}_AUROC"
        ] = auc

        row[
            f"{name}_AUPR"
        ] = apr

    return row


rows = []

# ============================================================
# RANDOM
# Query direction fixed as drug -> miRNA.
#
# Validation candidates:
# val positives + unlabeled
#
# Test candidates:
# test positives + unlabeled
# ============================================================

for seed in range(25):

    print(
        f"RANDOM seed {seed}/24",
        flush=True
    )

    f = (
        DATA
        / "05_results"
        / f"seed{seed}_fourview"
        / f"all_pair_scores_seed{seed}.tsv"
    )

    x = pd.read_csv(
        f,
        sep="\t"
    )

    val = x[
        x["split"].isin([
            "val_known",
            "unlabeled",
        ])
    ].copy()

    test = x[
        x["split"].isin([
            "test_known",
            "unlabeled",
        ])
    ].copy()

    val["y"] = (
        val["split"]
        == "val_known"
    ).astype(int)

    test["y"] = (
        test["split"]
        == "test_known"
    ).astype(int)

    rows.append(
        run_one(
            val,
            test,
            "drug",
            seed,
            "random",
        )
    )


# ============================================================
# STRICT DOUBLE-COLD
# Official query direction: drug -> miRNA.
# ============================================================

for seed in range(25):

    print(
        f"STRICT seed {seed}/24",
        flush=True
    )

    tag = (
        f"strict_cold_seed{seed}"
    )

    f = (
        DATA
        / "05_results"
        / tag
        / f"all_pair_scores_{tag}.tsv"
    )

    sd = (
        DATA
        / "03_splits"
        / tag
    )

    x = pd.read_csv(
        f,
        sep="\t"
    )

    val_drugs = set(
        pd.read_csv(
            sd / "val_drugs.tsv",
            sep="\t"
        )["drug"]
    )

    val_mirnas = set(
        pd.read_csv(
            sd / "val_mirnas.tsv",
            sep="\t"
        )["miRNA"]
    )

    test_drugs = set(
        pd.read_csv(
            sd / "test_drugs.tsv",
            sep="\t"
        )["drug"]
    )

    test_mirnas = set(
        pd.read_csv(
            sd / "test_mirnas.tsv",
            sep="\t"
        )["miRNA"]
    )

    val = x[
        x["drug"].isin(val_drugs)
        &
        x["miRNA"].isin(val_mirnas)
    ].copy()

    test = x[
        x["drug"].isin(test_drugs)
        &
        x["miRNA"].isin(test_mirnas)
    ].copy()

    val["y"] = (
        val["split"]
        == "val_known"
    ).astype(int)

    test["y"] = (
        test["split"]
        == "test_known"
    ).astype(int)

    rows.append(
        run_one(
            val,
            test,
            "drug",
            seed,
            "strict_cold",
        )
    )


res = pd.DataFrame(rows)

res.to_csv(
    OUT
    / "stageG_random_strict_final_25seed.tsv",
    sep="\t",
    index=False
)

print(
    "\n===== RANDOM + STRICT FINAL ====="
)

for split in [
    "random",
    "strict_cold",
]:

    print(
        f"\n### {split.upper()}"
    )

    z = res[
        res["split"] == split
    ]

    for name in [
        "Molecular",
        "Raw",
        "QRC",
        "SQRC",
    ]:

        auc = z[
            f"{name}_AUROC"
        ]

        apr = z[
            f"{name}_AUPR"
        ]

        print(
            f"{name:10s} | "
            f"AUROC="
            f"{auc.mean():.4f} ± "
            f"{auc.std(ddof=1):.4f} | "
            f"AUPR="
            f"{apr.mean():.4f} ± "
            f"{apr.std(ddof=1):.4f}"
        )

    print(
        "positive range:",
        int(z.n_pos.min()),
        "-",
        int(z.n_pos.max()),
        "mean=",
        f"{z.n_pos.mean():.2f}",
    )

print(
    "\nRANDOM_STRICT_FINAL_COMPLETE"
)
