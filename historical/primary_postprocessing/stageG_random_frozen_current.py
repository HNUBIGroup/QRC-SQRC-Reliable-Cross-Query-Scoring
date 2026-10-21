from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path.home() / "ECARF_Article2_20260911"
SNAP = ROOT / "03_frozen_inputs/random_current_20260912"
OUT = ROOT / "02_results"

LAMBDAS = np.round(np.arange(0, 1.0001, 0.05), 2)


def metrics(y, s):
    return (
        roc_auc_score(y, s),
        average_precision_score(y, s)
    )


def qstats(df):
    rows = []

    for q, g in df.groupby("drug"):
        d = g["delta"].to_numpy(float)
        n = len(d)

        b = float(d.mean())

        if n > 1:
            se2 = float(
                d.var(ddof=1) / n
            )
        else:
            se2 = 0.0

        rows.append((q, b, se2))

    return pd.DataFrame(
        rows,
        columns=["drug", "bias", "se2"]
    )


def calibrate(val, test, raw_val, raw_test):

    val = val.copy()
    test = test.copy()

    val["raw"] = raw_val
    test["raw"] = raw_test

    val["delta"] = (
        val["raw"]
        - val["molecular_score"]
    )

    test["delta"] = (
        test["raw"]
        - test["molecular_score"]
    )

    # ------------------------------
    # Validation estimates tau^2
    # ------------------------------

    vs = qstats(val)

    observed = (
        float(vs["bias"].var(ddof=1))
        if len(vs) > 1 else 0.0
    )

    noise = float(
        vs["se2"].mean()
    )

    tau2 = max(
        observed - noise,
        0.0
    )

    # ------------------------------
    # Test query bias / uncertainty
    # ------------------------------

    ts = qstats(test)

    bmap = dict(
        zip(ts.drug, ts.bias)
    )

    semap = dict(
        zip(ts.drug, ts.se2)
    )

    b = (
        test["drug"]
        .map(bmap)
        .to_numpy(float)
    )

    se2 = (
        test["drug"]
        .map(semap)
        .to_numpy(float)
    )

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
        "raw"
    ].to_numpy(float)

    delta = test[
        "delta"
    ].to_numpy(float)

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

    out = {
        "tau2": tau2,
        "mean_weight": float(w.mean()),
    }

    for name, score in [
        ("Molecular", mol),
        ("Raw", raw),
        ("QRC", qrc),
        ("SQRC", sqrc),
    ]:

        auc, aupr = metrics(
            y,
            score
        )

        out[f"{name}_AUROC"] = auc
        out[f"{name}_AUPR"] = aupr

    return out


records = []


for seed in range(25):

    print(
        f"seed {seed}/24",
        flush=True
    )

    f = (
        SNAP
        / f"all_pair_scores_seed{seed}.tsv"
    )

    x = pd.read_csv(
        f,
        sep="\t"
    )

    val_pos = x[
        x["split"] == "val_known"
    ].copy()

    test_pos = x[
        x["split"] == "test_known"
    ].copy()

    unl = x[
        x["split"] == "unlabeled"
    ].copy()

    if len(val_pos) != 344:
        raise RuntimeError(
            f"seed={seed}: "
            f"val_pos={len(val_pos)}"
        )

    if len(test_pos) != 344:
        raise RuntimeError(
            f"seed={seed}: "
            f"test_pos={len(test_pos)}"
        )

    # Exact historical deterministic
    # unlabeled-pool protocol
    rng = np.random.default_rng(
        20000 + seed
    )

    perm = rng.permutation(
        len(unl)
    )

    mid = len(perm) // 2

    val_unl = unl.iloc[
        perm[:mid]
    ].copy()

    test_unl = unl.iloc[
        perm[mid:]
    ].copy()

    # --------------------------------
    # Lambda selection:
    # validation only
    # AUPR > AUROC > smaller lambda
    # --------------------------------

    candidates = []

    yv = np.concatenate([
        np.ones(len(val_pos)),
        np.zeros(len(val_unl))
    ]).astype(int)

    for lam in LAMBDAS:

        s = np.concatenate([
            (1 - lam)
            * val_pos[
                "molecular_score"
            ].to_numpy()
            +
            lam
            * val_pos[
                "unified_score"
            ].to_numpy(),

            (1 - lam)
            * val_unl[
                "molecular_score"
            ].to_numpy()
            +
            lam
            * val_unl[
                "unified_score"
            ].to_numpy()
        ])

        auc, aupr = metrics(
            yv,
            s
        )

        candidates.append({
            "lambda": float(lam),
            "auc": auc,
            "aupr": aupr,
        })

    candidates.sort(
        key=lambda r: (
            r["aupr"],
            r["auc"],
            -r["lambda"],
        ),
        reverse=True
    )

    lam = candidates[0]["lambda"]

    # --------------------------------
    # Assemble val / test pools
    # --------------------------------

    val_pos["y"] = 1
    val_unl["y"] = 0
    test_pos["y"] = 1
    test_unl["y"] = 0

    val = pd.concat(
        [val_pos, val_unl],
        ignore_index=True
    )

    test = pd.concat(
        [test_pos, test_unl],
        ignore_index=True
    )

    raw_val = (
        (1 - lam)
        * val["molecular_score"].to_numpy()
        +
        lam
        * val["unified_score"].to_numpy()
    )

    raw_test = (
        (1 - lam)
        * test["molecular_score"].to_numpy()
        +
        lam
        * test["unified_score"].to_numpy()
    )

    r = calibrate(
        val,
        test,
        raw_val,
        raw_test
    )

    r["seed"] = seed
    r["lambda"] = lam
    r["test_positive_n"] = int(
        test["y"].sum()
    )
    r["test_unlabeled_n"] = int(
        (test["y"] == 0).sum()
    )

    records.append(r)


res = pd.DataFrame(records)

cols = [
    "seed",
    "lambda",
    "tau2",
    "mean_weight",
    "Molecular_AUROC",
    "Molecular_AUPR",
    "Raw_AUROC",
    "Raw_AUPR",
    "QRC_AUROC",
    "QRC_AUPR",
    "SQRC_AUROC",
    "SQRC_AUPR",
    "test_positive_n",
    "test_unlabeled_n",
]

res = res[cols]

outfile = (
    OUT
    / "stageG_random_frozen_current_25seed.tsv"
)

res.to_csv(
    outfile,
    sep="\t",
    index=False
)

print(
    "\n===== FROZEN RANDOM 25-SEED ====="
)

for name in [
    "Molecular",
    "Raw",
    "QRC",
    "SQRC",
]:

    auc = res[
        f"{name}_AUROC"
    ]

    aupr = res[
        f"{name}_AUPR"
    ]

    print(
        f"{name:10s} | "
        f"AUROC="
        f"{auc.mean():.6f} ± "
        f"{auc.std(ddof=1):.6f} | "
        f"AUPR="
        f"{aupr.mean():.6f} ± "
        f"{aupr.std(ddof=1):.6f}"
    )

print(
    "\nlambda:",
    f"{res['lambda'].mean():.4f} ± "
    f"{res['lambda'].std(ddof=1):.4f}"
)

print(
    "lambda > 0:",
    int(
        (res["lambda"] > 0).sum()
    ),
    "/25"
)

print(
    "Raw > Molecular AUROC:",
    int(
        (
            res["Raw_AUROC"]
            >
            res["Molecular_AUROC"]
        ).sum()
    ),
    "/25"
)

print(
    "SQRC > Molecular AUROC:",
    int(
        (
            res["SQRC_AUROC"]
            >
            res["Molecular_AUROC"]
        ).sum()
    ),
    "/25"
)

print(
    "SQRC > Raw AUROC:",
    int(
        (
            res["SQRC_AUROC"]
            >
            res["Raw_AUROC"]
        ).sum()
    ),
    "/25"
)

print(
    "\nFROZEN_RANDOM25_COMPLETE"
)
