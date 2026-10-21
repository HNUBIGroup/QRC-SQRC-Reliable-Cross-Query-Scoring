from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/home/u2508283030/MDA_CellPainting_Multiview_20260910")
OUT  = Path("/home/u2508283030/ECARF_Article2_20260911/02_results")
OUT.mkdir(parents=True, exist_ok=True)

def metric(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, s), average_precision_score(y, s)

rows = []

for seed in range(25):
    tag = f"strict_cold_seed{seed}"

    f = ROOT / "05_results" / tag / f"all_pair_scores_{tag}.tsv"
    sd = ROOT / "03_splits" / tag

    x = pd.read_csv(f, sep="\t")
    td = set(pd.read_csv(sd / "test_drugs.tsv", sep="\t")["drug"])
    tm = set(pd.read_csv(sd / "test_mirnas.tsv", sep="\t")["miRNA"])

    x = x[
        x["drug"].isin(td) &
        x["miRNA"].isin(tm)
    ].copy()

    x["y"] = (x["split"] == "test_known").astype(int)

    mol = x["molecular_score"].to_numpy()
    raw = x["final_score"].to_numpy()

    # Already contains the validation-selected lambda.
    correction = raw - mol
    x["_corr"] = correction

    # QRC: drug as query
    drug_bias = x.groupby("drug")["_corr"].transform("mean").to_numpy()
    qrc_drug = mol + correction - drug_bias

    # QRC: miRNA as query
    mirna_bias = x.groupby("miRNA")["_corr"].transform("mean").to_numpy()
    qrc_mirna = mol + correction - mirna_bias

    y = x["y"].to_numpy()

    ma, mp = metric(y, mol)
    ra, rp = metric(y, raw)
    da, dp = metric(y, qrc_drug)
    ia, ip = metric(y, qrc_mirna)

    rows.append({
        "seed": seed,
        "n_pairs": len(x),
        "n_pos": int(y.sum()),
        "Molecular_AUROC": ma,
        "Molecular_AUPR": mp,
        "Raw_AUROC": ra,
        "Raw_AUPR": rp,
        "QRC_DrugQuery_AUROC": da,
        "QRC_DrugQuery_AUPR": dp,
        "QRC_miRNAQuery_AUROC": ia,
        "QRC_miRNAQuery_AUPR": ip,
    })

df = pd.DataFrame(rows)
df.to_csv(
    OUT / "stageC_strictcold_25seed_qrc.tsv",
    sep="\t", index=False
)

print("\n===== STRICT-COLD 25-SEED QRC =====")
for name in ["Molecular", "Raw", "QRC_DrugQuery", "QRC_miRNAQuery"]:
    for metric_name in ["AUROC", "AUPR"]:
        v = df[f"{name}_{metric_name}"]
        print(f"{name:16s} {metric_name}: {v.mean():.4f} ± {v.std(ddof=1):.4f}")
    print()

for q in ["QRC_DrugQuery", "QRC_miRNAQuery"]:
    d = df[f"{q}_AUROC"] - df["Molecular_AUROC"]
    r = df[f"{q}_AUROC"] - df["Raw_AUROC"]
    print(q)
    print(f"  vs Molecular: {d.mean():+.4f} ± {d.std(ddof=1):.4f}, positive={(d>0).sum()}/25")
    print(f"  vs Raw:       {r.mean():+.4f} ± {r.std(ddof=1):.4f}, positive={(r>0).sum()}/25")

print("\nTEST POS RANGE:",
      int(df.n_pos.min()),
      int(df.n_pos.max()),
      "mean=", round(df.n_pos.mean(), 2))

print("\nSTRICT_QRC_COMPLETE")
