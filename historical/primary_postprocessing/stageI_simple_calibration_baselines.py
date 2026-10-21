from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OLD = Path.home() / "MDA_CellPainting_Multiview_20260910"
ROOT = Path.home() / "ECARF_Article2_20260911"
OUT = ROOT / "02_results"
FROZEN = ROOT / "03_frozen_inputs/random_current_20260912"

random_meta = pd.read_csv(
    OUT / "stageG_random_frozen_current_25seed.tsv",
    sep="\t"
)

LAMBDA = dict(
    zip(
        random_meta["seed"].astype(int),
        random_meta["lambda"]
    )
)


def metric(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    return (
        roc_auc_score(y, s),
        average_precision_score(y, s)
    )


def firstcol(path):
    x = pd.read_csv(path, sep="\t")
    return set(x.iloc[:, 0])


def qstats(df, qcol):
    rows = []

    for q, g in df.groupby(qcol):
        d = g["delta"].to_numpy(float)
        n = len(d)

        b = float(np.mean(d))
        med = float(np.median(d))

        if n > 1:
            se2 = float(
                np.var(d, ddof=1) / n
            )
        else:
            se2 = 0.0

        rows.append({
            "q": q,
            "bias": b,
            "median_bias": med,
            "se2": se2
        })

    return pd.DataFrame(rows)


def evaluate_one(
    val,
    test,
    qcol,
    raw_val,
    raw_test,
    setting,
    seed
):

    val = val.copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)

    val["Raw"] = raw_val
    test["Raw"] = raw_test

    val["delta"] = (
        val["Raw"]
        - val["molecular_score"]
    )

    test["delta"] = (
        test["Raw"]
        - test["molecular_score"]
    )

    # ==========================================
    # Validation-only tau^2 for SQRC
    # ==========================================

    vs = qstats(val, qcol)

    observed = (
        float(vs["bias"].var(ddof=1))
        if len(vs) > 1
        else 0.0
    )

    noise = float(
        vs["se2"].mean()
    )

    tau2 = max(
        observed - noise,
        0.0
    )

    # ==========================================
    # Test query statistics
    # ==========================================

    ts = qstats(test, qcol)

    bmap = dict(
        zip(ts["q"], ts["bias"])
    )

    medmap = dict(
        zip(ts["q"], ts["median_bias"])
    )

    semap = dict(
        zip(ts["q"], ts["se2"])
    )

    b = test[qcol].map(
        bmap
    ).to_numpy(float)

    med = test[qcol].map(
        medmap
    ).to_numpy(float)

    se2 = test[qcol].map(
        semap
    ).to_numpy(float)

    den = tau2 + se2

    w = np.divide(
        tau2,
        den,
        out=np.zeros_like(den),
        where=den > 0
    )

    mol = test[
        "molecular_score"
    ].to_numpy(float)

    raw = test[
        "Raw"
    ].to_numpy(float)

    delta = test[
        "delta"
    ].to_numpy(float)

    # ==========================================
    # Simple calibration baselines
    # ==========================================

    raw_mean = (
        test.groupby(qcol)["Raw"]
        .transform("mean")
        .to_numpy(float)
    )

    raw_std = (
        test.groupby(qcol)["Raw"]
        .transform(
            lambda x: x.std(ddof=0)
        )
        .to_numpy(float)
    )

    raw_center = (
        raw - raw_mean
    )

    raw_zscore = (
        raw - raw_mean
    ) / np.maximum(
        raw_std,
        1e-8
    )

    median_qrc = (
        mol
        + delta
        - med
    )

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

    y = test["y"].to_numpy(int)

    methods = {
        "Molecular": mol,
        "Raw": raw,
        "RawCenter": raw_center,
        "RawZScore": raw_zscore,
        "MedianQRC": median_qrc,
        "QRC": qrc,
        "SQRC": sqrc,
    }

    row = {
        "setting": setting,
        "seed": seed,
        "tau2": tau2,
        "n_pos": int(y.sum()),
        "n_pairs": len(y),
    }

    for name, score in methods.items():

        auc, aupr = metric(
            y,
            score
        )

        row[
            f"{name}_AUROC"
        ] = auc

        row[
            f"{name}_AUPR"
        ] = aupr

    return row


rows = []

# ==========================================================
# RANDOM
# ==========================================================

for seed in range(25):

    print(
        f"RANDOM {seed}/24",
        flush=True
    )

    x = pd.read_csv(
        FROZEN
        / f"all_pair_scores_seed{seed}.tsv",
        sep="\t"
    )

    vp = x[
        x["split"] == "val_known"
    ].copy()

    tp = x[
        x["split"] == "test_known"
    ].copy()

    unl = x[
        x["split"] == "unlabeled"
    ].copy()

    rng = np.random.default_rng(
        20000 + seed
    )

    perm = rng.permutation(
        len(unl)
    )

    mid = len(perm) // 2

    vu = unl.iloc[
        perm[:mid]
    ].copy()

    tu = unl.iloc[
        perm[mid:]
    ].copy()

    vp["y"] = 1
    vu["y"] = 0
    tp["y"] = 1
    tu["y"] = 0

    val = pd.concat(
        [vp, vu],
        ignore_index=True
    )

    test = pd.concat(
        [tp, tu],
        ignore_index=True
    )

    lam = float(
        LAMBDA[seed]
    )

    raw_val = (
        (1-lam)
        * val["molecular_score"].to_numpy()
        +
        lam
        * val["unified_score"].to_numpy()
    )

    raw_test = (
        (1-lam)
        * test["molecular_score"].to_numpy()
        +
        lam
        * test["unified_score"].to_numpy()
    )

    rows.append(
        evaluate_one(
            val,
            test,
            "drug",
            raw_val,
            raw_test,
            "random",
            seed
        )
    )


# ==========================================================
# DRUG-COLD / MIRNA-COLD
# ==========================================================

for mode in [
    "drug",
    "mirna"
]:

    qcol = (
        "drug"
        if mode == "drug"
        else "miRNA"
    )

    for seed in range(25):

        print(
            f"{mode.upper()}-COLD "
            f"{seed}/24",
            flush=True
        )

        tag = (
            f"{mode}_cold_seed{seed}"
        )

        x = pd.read_csv(
            OLD
            / "05_results"
            / tag
            / f"all_pair_scores_{tag}.tsv",
            sep="\t"
        )

        sd = (
            OLD
            / "03_splits"
            / tag
        )

        ve = firstcol(
            sd / "val_entities.tsv"
        )

        te = firstcol(
            sd / "test_entities.tsv"
        )

        val = x[
            x[qcol].isin(ve)
        ].copy()

        test = x[
            x[qcol].isin(te)
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
            evaluate_one(
                val,
                test,
                qcol,
                val["final_score"].to_numpy(),
                test["final_score"].to_numpy(),
                f"{mode}_cold",
                seed
            )
        )


# ==========================================================
# STRICT DOUBLE-COLD
# drug -> miRNA
# ==========================================================

for seed in range(25):

    print(
        f"STRICT {seed}/24",
        flush=True
    )

    tag = (
        f"strict_cold_seed{seed}"
    )

    x = pd.read_csv(
        OLD
        / "05_results"
        / tag
        / f"all_pair_scores_{tag}.tsv",
        sep="\t"
    )

    sd = (
        OLD
        / "03_splits"
        / tag
    )

    vd = firstcol(
        sd / "val_drugs.tsv"
    )

    vm = firstcol(
        sd / "val_mirnas.tsv"
    )

    td = firstcol(
        sd / "test_drugs.tsv"
    )

    tm = firstcol(
        sd / "test_mirnas.tsv"
    )

    val = x[
        x["drug"].isin(vd)
        &
        x["miRNA"].isin(vm)
    ].copy()

    test = x[
        x["drug"].isin(td)
        &
        x["miRNA"].isin(tm)
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
        evaluate_one(
            val,
            test,
            "drug",
            val["final_score"].to_numpy(),
            test["final_score"].to_numpy(),
            "strict_cold",
            seed
        )
    )


res = pd.DataFrame(rows)

res.to_csv(
    OUT
    / "stageI_simple_calibration_baselines_25seed.tsv",
    sep="\t",
    index=False
)

print(
    "\n===== SIMPLE CALIBRATION BASELINES ====="
)

methods = [
    "Molecular",
    "Raw",
    "RawCenter",
    "RawZScore",
    "MedianQRC",
    "QRC",
    "SQRC",
]

for setting in [
    "random",
    "mirna_cold",
    "drug_cold",
    "strict_cold"
]:

    print(
        f"\n### {setting.upper()}"
    )

    z = res[
        res["setting"]
        == setting
    ]

    for m in methods:

        auc = z[
            f"{m}_AUROC"
        ]

        aupr = z[
            f"{m}_AUPR"
        ]

        print(
            f"{m:10s} | "
            f"AUC={auc.mean():.4f} ± "
            f"{auc.std(ddof=1):.4f} | "
            f"AUPR={aupr.mean():.4f}"
        )

print(
    "\nSTAGEI_BASELINES_COMPLETE"
)
