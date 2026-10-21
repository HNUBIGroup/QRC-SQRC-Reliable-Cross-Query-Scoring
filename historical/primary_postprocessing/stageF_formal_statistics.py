from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(
    "/home/u2508283030/ECARF_Article2_20260911"
)

INFILE = (
    ROOT / "02_results"
    / "stageE_shrinkage_qrc_25seed.tsv"
)

OUT = ROOT / "02_results"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    INFILE,
    sep="\t",
)

comparisons = [
    ("SQRC", "Molecular"),
    ("SQRC", "Raw"),
    ("SQRC", "QRC"),
]

metrics = [
    "Pooled_AUROC",
    "Pooled_AUPR",
]

rows = []


def rank_biserial_from_diff(diff):
    """
    Matched-pairs rank-biserial correlation.
    Positive means method A tends to outperform B.
    """

    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    absdiff = np.abs(diff)

    order = np.argsort(absdiff)
    ranks = np.empty(len(diff), dtype=float)

    i = 0
    while i < len(diff):
        j = i
        while (
            j + 1 < len(diff)
            and absdiff[order[j + 1]]
            == absdiff[order[i]]
        ):
            j += 1

        avg_rank = (
            (i + 1 + j + 1) / 2.0
        )

        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank

        i = j + 1

    w_plus = ranks[
        diff > 0
    ].sum()

    w_minus = ranks[
        diff < 0
    ].sum()

    denom = w_plus + w_minus

    if denom == 0:
        return 0.0

    return (
        w_plus - w_minus
    ) / denom


for mode in ["drug", "mirna"]:

    z = (
        df[df["mode"] == mode]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    for metric in metrics:

        for A, B in comparisons:

            a = z[
                f"{A}_{metric}"
            ].to_numpy()

            b = z[
                f"{B}_{metric}"
            ].to_numpy()

            diff = a - b

            valid = (
                np.isfinite(a)
                & np.isfinite(b)
            )

            a = a[valid]
            b = b[valid]
            diff = diff[valid]

            wins = int(
                np.sum(diff > 1e-12)
            )

            losses = int(
                np.sum(diff < -1e-12)
            )

            ties = int(
                len(diff)
                - wins
                - losses
            )

            if np.allclose(
                diff,
                0.0,
            ):
                stat = 0.0
                p = 1.0
            else:
                try:
                    stat, p = wilcoxon(
                        a,
                        b,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    )
                except Exception:
                    stat, p = (
                        np.nan,
                        np.nan,
                    )

            sd = np.std(
                diff,
                ddof=1,
            )

            if sd > 0:
                cohen_dz = (
                    np.mean(diff)
                    / sd
                )
            else:
                cohen_dz = np.nan

            rbc = (
                rank_biserial_from_diff(
                    diff
                )
            )

            rows.append({
                "mode": mode,
                "metric": metric,
                "method_A": A,
                "method_B": B,
                "n": len(diff),

                "A_mean":
                    np.mean(a),
                "A_std":
                    np.std(a, ddof=1),

                "B_mean":
                    np.mean(b),
                "B_std":
                    np.std(b, ddof=1),

                "mean_diff":
                    np.mean(diff),
                "std_diff":
                    np.std(
                        diff,
                        ddof=1
                    ),

                "wins":
                    wins,
                "ties":
                    ties,
                "losses":
                    losses,

                "wilcoxon_stat":
                    stat,
                "p_raw":
                    p,

                "cohen_dz":
                    cohen_dz,

                "rank_biserial":
                    rbc,
            })


res = pd.DataFrame(rows)

# ============================================================
# Holm correction over all formal tests
# ============================================================

pvals = (
    res["p_raw"]
    .to_numpy()
    .copy()
)

valid = np.isfinite(pvals)

idx = np.where(valid)[0]

order = idx[
    np.argsort(
        pvals[idx]
    )
]

m = len(order)

adjusted = np.full(
    len(res),
    np.nan
)

running_max = 0.0

for rank, i in enumerate(order):

    factor = (
        m - rank
    )

    adj = min(
        1.0,
        pvals[i] * factor
    )

    running_max = max(
        running_max,
        adj
    )

    adjusted[i] = (
        running_max
    )

res["p_holm"] = adjusted

res["significant_0.05"] = (
    res["p_holm"] < 0.05
)

res.to_csv(
    OUT
    / "stageF_formal_statistics.tsv",
    sep="\t",
    index=False,
)


print(
    "\n===== FORMAL PAIRED STATISTICS ====="
)

for mode in [
    "drug",
    "mirna",
]:

    print(
        f"\n### {mode.upper()}-COLD"
    )

    z = res[
        res["mode"] == mode
    ]

    for metric in metrics:

        print(
            f"\n[{metric}]"
        )

        g = z[
            z["metric"] == metric
        ]

        for _, r in g.iterrows():

            print(
                f"{r.method_A:5s} vs "
                f"{r.method_B:9s} | "
                f"diff={r.mean_diff:+.5f} | "
                f"W/T/L="
                f"{int(r.wins)}/"
                f"{int(r.ties)}/"
                f"{int(r.losses)} | "
                f"p={r.p_raw:.3e} | "
                f"Holm={r.p_holm:.3e} | "
                f"dz={r.cohen_dz:+.3f} | "
                f"r_rb={r.rank_biserial:+.3f}"
            )

print(
    "\nFORMAL_STATISTICS_COMPLETE"
)
