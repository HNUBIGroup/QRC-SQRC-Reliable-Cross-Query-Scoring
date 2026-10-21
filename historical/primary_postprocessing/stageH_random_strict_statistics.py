from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path.home() / "ECARF_Article2_20260911"
OUT = ROOT / "02_results"

random = pd.read_csv(
    OUT / "stageG_random_frozen_current_25seed.tsv",
    sep="\t"
)

rs = pd.read_csv(
    OUT / "stageG_v2_random_strict_final_25seed.tsv",
    sep="\t"
)

strict = (
    rs[rs["setting"] == "strict_cold"]
    .sort_values("seed")
    .reset_index(drop=True)
)

random = (
    random.sort_values("seed")
    .reset_index(drop=True)
)

tests = [
    ("random", random, "Raw", "Molecular"),
    ("random", random, "SQRC", "Molecular"),
    ("random", random, "SQRC", "Raw"),

    ("strict", strict, "QRC", "Molecular"),
    ("strict", strict, "QRC", "Raw"),
    ("strict", strict, "SQRC", "Raw"),
]

rows = []

for setting, df, A, B in tests:
    for metric in ["AUROC", "AUPR"]:

        a = df[f"{A}_{metric}"].to_numpy(float)
        b = df[f"{B}_{metric}"].to_numpy(float)
        d = a - b

        if np.allclose(d, 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(
                a, b,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto"
            )

        sd = d.std(ddof=1)
        dz = d.mean() / sd if sd > 0 else np.nan

        rows.append({
            "setting": setting,
            "metric": metric,
            "A": A,
            "B": B,
            "A_mean": a.mean(),
            "B_mean": b.mean(),
            "diff": d.mean(),
            "wins": int((d > 1e-12).sum()),
            "ties": int((np.abs(d) <= 1e-12).sum()),
            "losses": int((d < -1e-12).sum()),
            "p_raw": p,
            "cohen_dz": dz,
        })

res = pd.DataFrame(rows)

# Holm correction over all 12 tests
p = res["p_raw"].to_numpy()
order = np.argsort(p)
m = len(p)
adj = np.empty(m)
running = 0.0

for rank, idx in enumerate(order):
    x = min(1.0, p[idx] * (m - rank))
    running = max(running, x)
    adj[idx] = running

res["p_holm"] = adj

res.to_csv(
    OUT / "stageH_random_strict_statistics.tsv",
    sep="\t",
    index=False
)

print("\n===== RANDOM + STRICT FORMAL STATISTICS =====")

for setting in ["random", "strict"]:
    print(f"\n### {setting.upper()}")

    z = res[res.setting == setting]

    for _, r in z.iterrows():
        print(
            f"{r.metric:5s} | "
            f"{r.A:5s} vs {r.B:9s} | "
            f"diff={r['diff']:+.5f} | "
            f"W/T/L={r.wins}/{r.ties}/{r.losses} | "
            f"p={r.p_raw:.3e} | "
            f"Holm={r.p_holm:.3e} | "
            f"dz={r.cohen_dz:+.3f}"
        )

print("\nSTAGEH_COMPLETE")
