from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OLD = Path.home() / "MDA_CellPainting_Multiview_20260910"
NEW = Path.home() / "ECARF_Article2_20260911"
OUT = NEW / "02_results"

hist = pd.read_csv(
    OLD / "05_results/formal_random25/formal_random25_perseed.tsv",
    sep="\t"
).sort_values("seed").reset_index(drop=True)

LAM = dict(zip(hist.seed.astype(int), hist["lambda"]))


def scores(y, s):
    return roc_auc_score(y, s), average_precision_score(y, s)


def qstats(df, qcol):
    a = []
    for q, g in df.groupby(qcol):
        x = g["delta"].to_numpy(float)
        n = len(x)
        a.append((
            q,
            x.mean(),
            x.var(ddof=1) / n if n > 1 else 0.0
        ))
    return pd.DataFrame(a, columns=["q", "b", "se2"])


def calibrate(val, test, qcol, raw_val, raw_test, setting, seed):
    val = val.copy()
    test = test.copy()

    val["raw"] = raw_val
    test["raw"] = raw_test
    val["delta"] = val["raw"] - val["molecular_score"]
    test["delta"] = test["raw"] - test["molecular_score"]

    # validation-only estimate of between-query bias variance
    vs = qstats(val, qcol)
    obs = vs.b.var(ddof=1) if len(vs) > 1 else 0.0
    noise = vs.se2.mean()
    tau2 = max(float(obs - noise), 0.0)

    ts = qstats(test, qcol)
    bmap = dict(zip(ts.q, ts.b))
    semap = dict(zip(ts.q, ts.se2))

    b = test[qcol].map(bmap).to_numpy(float)
    se2 = test[qcol].map(semap).to_numpy(float)

    den = tau2 + se2
    w = np.divide(
        tau2,
        den,
        out=np.zeros_like(den),
        where=den > 0
    )

    mol = test["molecular_score"].to_numpy(float)
    raw = test["raw"].to_numpy(float)
    delta = test["delta"].to_numpy(float)

    qrc = mol + delta - b
    sqrc = mol + delta - w * b
    y = test["y"].to_numpy(int)

    r = {
        "setting": setting,
        "seed": seed,
        "n_pos": int(y.sum()),
        "n_pairs": len(y),
        "tau2": tau2,
        "mean_weight": float(w.mean())
    }

    for name, s in [
        ("Molecular", mol),
        ("Raw", raw),
        ("QRC", qrc),
        ("SQRC", sqrc),
    ]:
        auc, aupr = scores(y, s)
        r[f"{name}_AUROC"] = auc
        r[f"{name}_AUPR"] = aupr

    return r


def read_entities(path):
    x = pd.read_csv(path, sep="\t")
    return set(x.iloc[:, 0])


rows = []

# ============================================================
# RANDOM — EXACT HISTORICAL PROTOCOL
# ============================================================

for seed in range(25):
    print(f"RANDOM {seed}/24", flush=True)

    f = (
        OLD / "05_results"
        / f"seed{seed}_fourview"
        / f"all_pair_scores_seed{seed}.tsv"
    )
    x = pd.read_csv(f, sep="\t")

    vp = x[x.split == "val_known"].copy()
    tp = x[x.split == "test_known"].copy()
    unl = x[x.split == "unlabeled"].copy()

    rng = np.random.default_rng(20000 + seed)
    p = rng.permutation(len(unl))
    mid = len(p) // 2

    vu = unl.iloc[p[:mid]].copy()
    tu = unl.iloc[p[mid:]].copy()

    vp["y"], vu["y"] = 1, 0
    tp["y"], tu["y"] = 1, 0

    val = pd.concat([vp, vu], ignore_index=True)
    test = pd.concat([tp, tu], ignore_index=True)

    lam = float(LAM[seed])

    rv = (
        (1 - lam) * val["molecular_score"].to_numpy()
        + lam * val["unified_score"].to_numpy()
    )
    rt = (
        (1 - lam) * test["molecular_score"].to_numpy()
        + lam * test["unified_score"].to_numpy()
    )

    r = calibrate(
        val, test, "drug",
        rv, rt, "random", seed
    )
    r["lambda"] = lam
    rows.append(r)


# ============================================================
# STRICT DOUBLE-COLD — drug -> miRNA
# ============================================================

for seed in range(25):
    print(f"STRICT {seed}/24", flush=True)

    tag = f"strict_cold_seed{seed}"

    x = pd.read_csv(
        OLD / "05_results" / tag
        / f"all_pair_scores_{tag}.tsv",
        sep="\t"
    )

    sd = OLD / "03_splits" / tag

    vd = read_entities(sd / "val_drugs.tsv")
    vm = read_entities(sd / "val_mirnas.tsv")
    td = read_entities(sd / "test_drugs.tsv")
    tm = read_entities(sd / "test_mirnas.tsv")

    val = x[
        x["drug"].isin(vd)
        & x["miRNA"].isin(vm)
    ].copy()

    test = x[
        x["drug"].isin(td)
        & x["miRNA"].isin(tm)
    ].copy()

    val["y"] = (val["split"] == "val_known").astype(int)
    test["y"] = (test["split"] == "test_known").astype(int)

    rows.append(
        calibrate(
            val, test, "drug",
            val["final_score"].to_numpy(),
            test["final_score"].to_numpy(),
            "strict_cold", seed
        )
    )


res = pd.DataFrame(rows)
res.to_csv(
    OUT / "stageG_v2_random_strict_final_25seed.tsv",
    sep="\t",
    index=False
)

# ============================================================
# RANDOM HISTORICAL PARITY
# ============================================================

rr = (
    res[res.setting == "random"]
    .sort_values("seed")
    .reset_index(drop=True)
)

d_mol = np.max(np.abs(
    rr.Molecular_AUROC.to_numpy()
    - hist.mol_auc.to_numpy()
))

d_raw = np.max(np.abs(
    rr.Raw_AUROC.to_numpy()
    - hist.final_auc.to_numpy()
))

print("\n===== RANDOM PROTOCOL PARITY =====")
print(f"max Molecular AUROC diff = {d_mol:.3e}")
print(f"max Raw AUROC diff       = {d_raw:.3e}")

if d_mol > 1e-10 or d_raw > 1e-10:
    raise RuntimeError("RANDOM_PROTOCOL_PARITY_FAILED")

print("RANDOM_PROTOCOL_PARITY_PASS")


# ============================================================
# RANDOM + STRICT RESULTS
# ============================================================

for setting in ["random", "strict_cold"]:
    z = res[res.setting == setting]

    print(f"\n===== {setting.upper()} =====")

    for name in ["Molecular", "Raw", "QRC", "SQRC"]:
        a = z[f"{name}_AUROC"]
        p = z[f"{name}_AUPR"]

        print(
            f"{name:10s} | "
            f"AUROC={a.mean():.4f} ± {a.std(ddof=1):.4f} | "
            f"AUPR={p.mean():.4f} ± {p.std(ddof=1):.4f}"
        )

    print(
        f"positives: {z.n_pos.min()}-{z.n_pos.max()}, "
        f"mean={z.n_pos.mean():.2f}"
    )


# ============================================================
# COMPLETE FOUR-SETTING TABLE
# ============================================================

e = pd.read_csv(
    OUT / "stageE_shrinkage_qrc_25seed.tsv",
    sep="\t"
)

print("\n===== FOUR-SETTING FINAL AUROC/AUPR =====")

# random
z = res[res.setting == "random"]
print(
    "random       | "
    + " | ".join(
        f"{m}={z[f'{m}_AUROC'].mean():.4f}/"
        f"{z[f'{m}_AUPR'].mean():.4f}"
        for m in ["Molecular", "Raw", "QRC", "SQRC"]
    )
)

# mirna-cold / drug-cold
for mode, label in [
    ("mirna", "mirna_cold"),
    ("drug", "drug_cold")
]:
    z = e[e["mode"] == mode]

    print(
        f"{label:12s} | "
        + " | ".join(
            f"{m}={z[f'{m}_Pooled_AUROC'].mean():.4f}/"
            f"{z[f'{m}_Pooled_AUPR'].mean():.4f}"
            for m in ["Molecular", "Raw", "QRC", "SQRC"]
        )
    )

# strict
z = res[res.setting == "strict_cold"]
print(
    "strict_cold  | "
    + " | ".join(
        f"{m}={z[f'{m}_AUROC'].mean():.4f}/"
        f"{z[f'{m}_AUPR'].mean():.4f}"
        for m in ["Molecular", "Raw", "QRC", "SQRC"]
    )
)

print("\nSTAGEG_V2_COMPLETE")
