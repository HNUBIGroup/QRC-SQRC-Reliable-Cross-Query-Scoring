from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path.home() / "ECARF_Article2_20260911"
OUT = ROOT / "02_results"

df = pd.read_csv(
    OUT / "stageI_simple_calibration_baselines_25seed.tsv",
    sep="\t"
)

comparisons = [
    ("SQRC", "RawCenter"),
    ("SQRC", "RawZScore"),
    ("SQRC", "MedianQRC"),
]

rows = []

for setting in [
    "random",
    "mirna_cold",
    "drug_cold",
    "strict_cold",
]:
    z = (
        df[df.setting == setting]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    for metric in ["AUROC", "AUPR"]:

        for A, B in comparisons:

            a = z[f"{A}_{metric}"].to_numpy(float)
            b = z[f"{B}_{metric}"].to_numpy(float)
            d = a - b

            if np.allclose(d, 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(
                    a,
                    b,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto"
                )

            sd = d.std(ddof=1)

            rows.append({
                "setting": setting,
                "metric": metric,
                "A": A,
                "B": B,
                "diff": d.mean(),
                "wins": int((d > 1e-12).sum()),
                "ties": int((np.abs(d) <= 1e-12).sum()),
                "losses": int((d < -1e-12).sum()),
                "p_raw": p,
                "cohen_dz":
                    d.mean()/sd if sd > 0 else np.nan,
            })

res = pd.DataFrame(rows)

# Holm correction within each setting × metric family
res["p_holm"] = np.nan

for (setting, metric), idx in res.groupby(
    ["setting", "metric"]
).groups.items():

    idx = list(idx)
    p = res.loc[idx, "p_raw"].to_numpy()
    order = np.argsort(p)
    m = len(p)

    adj = np.zeros(m)
    running = 0.0

    for rank, oi in enumerate(order):
        x = min(1.0, p[oi] * (m-rank))
        running = max(running, x)
        adj[oi] = running

    res.loc[idx, "p_holm"] = adj

res.to_csv(
    OUT / "stageI_baseline_statistics.tsv",
    sep="\t",
    index=False
)

print("\n===== BASELINE FORMAL STATISTICS =====")

for setting in [
    "random",
    "mirna_cold",
    "drug_cold",
    "strict_cold",
]:

    print(f"\n### {setting.upper()}")

    z = res[res.setting == setting]

    for _, r in z.iterrows():

        print(
            f"{r.metric:5s} | "
            f"{r.A:4s} vs {r.B:9s} | "
            f"diff={r['diff']:+.5f} | "
            f"W/T/L={r.wins}/{r.ties}/{r.losses} | "
            f"Holm={r.p_holm:.3e} | "
            f"dz={r.cohen_dz:+.3f}"
        )

print("\nSTAGEI_STATS_COMPLETE")
