from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy import stats


ROOT = (
    Path.home()
    / "ECARF_Article2_20260911"
    / "17_qrc_prism_mda_formal_20260920"
)

RESULT_ROOT = ROOT / "06_results"
OUT = ROOT / "07_qc"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def mean_sd(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1))


def paired_ci95(diff):
    diff = np.asarray(diff, dtype=float)

    n = len(diff)
    m = np.mean(diff)
    sd = np.std(diff, ddof=1)

    if n < 2:
        return float("nan"), float("nan")

    se = sd / np.sqrt(n)

    tcrit = stats.t.ppf(
        0.975,
        df=n - 1,
    )

    return (
        float(m - tcrit * se),
        float(m + tcrit * se),
    )


def cohens_dz(diff):
    diff = np.asarray(diff, dtype=float)

    sd = np.std(diff, ddof=1)

    if sd == 0:
        return float("nan")

    return float(
        np.mean(diff) / sd
    )


def wilcoxon_p(diff):
    diff = np.asarray(diff, dtype=float)

    if np.allclose(diff, 0):
        return 1.0

    try:
        res = stats.wilcoxon(
            diff,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
        )

        return float(res.pvalue)

    except Exception:
        return float("nan")


def wtl(diff, tol=1e-12):
    diff = np.asarray(diff, dtype=float)

    win = int(np.sum(diff > tol))
    tie = int(np.sum(np.abs(diff) <= tol))
    loss = int(np.sum(diff < -tol))

    return win, tie, loss


def holm_adjust(pvals):
    """
    Standard Holm step-down adjustment.
    Returns adjusted p-values in original order.
    """

    pvals = np.asarray(
        pvals,
        dtype=float,
    )

    m = len(pvals)

    order = np.argsort(pvals)

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, idx in enumerate(order):

        raw = pvals[idx]

        adj = (
            (m - rank)
            * raw
        )

        running_max = max(
            running_max,
            adj,
        )

        adjusted_sorted[rank] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]

    return adjusted


# ============================================================
# LOAD 25 SEEDS
# ============================================================

records = []

pair_qc = []

for seed in range(25):

    d = (
        RESULT_ROOT
        / f"drug_cold_seed{seed}"
    )

    summary_file = (
        d
        / f"prism_drug_cold_seed{seed}_summary.json"
    )

    pair_file = (
        d
        / f"all_pair_scores_prism_drug_cold_seed{seed}.tsv"
    )

    if not summary_file.exists():
        raise RuntimeError(
            f"Missing summary seed {seed}"
        )

    if not pair_file.exists():
        raise RuntimeError(
            f"Missing pair file seed {seed}"
        )

    with open(summary_file) as f:
        s = json.load(f)

    pair = pd.read_csv(
        pair_file,
        sep="\t",
    )

    if len(pair) != 7170:
        raise RuntimeError(
            f"seed {seed}: "
            f"expected 7170 rows, "
            f"got {len(pair)}"
        )

    if pair["drug"].nunique() != 6:
        raise RuntimeError(
            f"seed {seed}: "
            f"expected 6 test drugs"
        )

    if pair["miRNA"].nunique() != 1195:
        raise RuntimeError(
            f"seed {seed}: "
            f"expected 1195 miRNAs"
        )

    q = s["qrc"]

    macro_auc_diff = float(
        q["raw_qrc_macro_auc_abs_diff"]
    )

    macro_aupr_diff = float(
        q["raw_qrc_macro_aupr_abs_diff"]
    )

    if macro_auc_diff > 1e-12:
        raise RuntimeError(
            f"seed {seed}: "
            f"macro AUROC invariance failed"
        )

    if macro_aupr_diff > 1e-12:
        raise RuntimeError(
            f"seed {seed}: "
            f"macro AUPR invariance failed"
        )

    row = {
        "seed": seed,
    }

    for model in [
        "Base",
        "Raw",
        "QRC",
        "SQRC",
    ]:

        for metric in [
            "AUROC",
            "AUPR",
            "macro_AUROC",
            "macro_AUPR",
        ]:

            row[
                f"{model}_{metric}"
            ] = float(
                s["metrics"][model][metric]
            )

    row[
        "tau_b2"
    ] = float(
        q["tau_b2_validation_only"]
    )

    row[
        "query_shift_variance_ratio"
    ] = float(
        q["query_shift_variance_ratio"]
    )

    row[
        "raw_vs_base_mad"
    ] = float(
        q["raw_vs_base_mean_abs_change"]
    )

    row[
        "qrc_vs_base_mad"
    ] = float(
        q["qrc_vs_base_mean_abs_change"]
    )

    row[
        "qrc_identity_max_error"
    ] = float(
        q["qrc_identity_max_error"]
    )

    row[
        "sqrc_identity_max_error"
    ] = float(
        q["sqrc_identity_max_error"]
    )

    records.append(row)

    pair_qc.append({
        "seed":
            seed,
        "rows":
            len(pair),
        "test_drugs":
            pair["drug"].nunique(),
        "mirnas":
            pair["miRNA"].nunique(),
        "macro_auc_diff":
            macro_auc_diff,
        "macro_aupr_diff":
            macro_aupr_diff,
    })


df = pd.DataFrame(
    records
).sort_values("seed")

qc_df = pd.DataFrame(
    pair_qc
).sort_values("seed")


if list(df["seed"]) != list(range(25)):
    raise RuntimeError(
        "Seed set is not exactly 0-24."
    )


# ============================================================
# SAVE PER-SEED TABLE
# ============================================================

df.to_csv(
    OUT /
    "PRISM_DRUGCOLD_25SEEDS_PER_SEED.tsv",
    sep="\t",
    index=False,
)

qc_df.to_csv(
    OUT /
    "PRISM_DRUGCOLD_25SEEDS_PAIR_QC.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# DESCRIPTIVE SUMMARY
# ============================================================

summary_rows = []

for model in [
    "Base",
    "Raw",
    "QRC",
    "SQRC",
]:

    for metric in [
        "AUROC",
        "AUPR",
        "macro_AUROC",
        "macro_AUPR",
    ]:

        col = (
            f"{model}_{metric}"
        )

        mean, sd = mean_sd(
            df[col]
        )

        summary_rows.append({
            "model":
                model,
            "metric":
                metric,
            "mean":
                mean,
            "sd":
                sd,
            "mean_pm_sd":
                f"{mean:.6f} ± {sd:.6f}",
        })


desc_df = pd.DataFrame(
    summary_rows
)

desc_df.to_csv(
    OUT /
    "PRISM_DRUGCOLD_25SEEDS_MEAN_SD.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# FROZEN PAIRED CONTRASTS
# ============================================================

contrasts = [
    ("Raw", "Base", "Raw_minus_Base"),
    ("QRC", "Raw", "QRC_minus_Raw"),
    ("QRC", "Base", "QRC_minus_Base"),
]

metrics = [
    "AUROC",
    "AUPR",
]

stat_rows = []

for metric in metrics:

    metric_indices = []

    for left, right, name in contrasts:

        diff = (
            df[f"{left}_{metric}"]
            - df[f"{right}_{metric}"]
        ).to_numpy(dtype=float)

        lo, hi = paired_ci95(
            diff
        )

        win, tie, loss = wtl(
            diff
        )

        row = {
            "metric":
                metric,

            "contrast":
                name,

            "mean_diff":
                float(np.mean(diff)),

            "sd_diff":
                float(
                    np.std(
                        diff,
                        ddof=1,
                    )
                ),

            "ci95_low":
                lo,

            "ci95_high":
                hi,

            "cohens_dz":
                cohens_dz(diff),

            "wilcoxon_p":
                wilcoxon_p(diff),

            "wins":
                win,

            "ties":
                tie,

            "losses":
                loss,
        }

        stat_rows.append(row)

        metric_indices.append(
            len(stat_rows) - 1
        )

    # Holm correction only within the
    # three frozen contrasts of this metric
    raw_ps = [
        stat_rows[i]["wilcoxon_p"]
        for i in metric_indices
    ]

    adj = holm_adjust(
        raw_ps
    )

    for i, p_adj in zip(
        metric_indices,
        adj,
    ):
        stat_rows[i][
            "wilcoxon_p_holm"
        ] = float(p_adj)


stats_df = pd.DataFrame(
    stat_rows
)

stats_df.to_csv(
    OUT /
    "PRISM_DRUGCOLD_25SEEDS_PAIRED_STATS.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# MECHANISM SUMMARY
# ============================================================

mech_cols = [
    "tau_b2",
    "query_shift_variance_ratio",
    "raw_vs_base_mad",
    "qrc_vs_base_mad",
    "qrc_identity_max_error",
    "sqrc_identity_max_error",
]

mech_rows = []

for col in mech_cols:

    mean, sd = mean_sd(
        df[col]
    )

    mech_rows.append({
        "quantity":
            col,
        "mean":
            mean,
        "sd":
            sd,
        "min":
            float(df[col].min()),
        "max":
            float(df[col].max()),
    })


mech_df = pd.DataFrame(
    mech_rows
)

mech_df.to_csv(
    OUT /
    "PRISM_DRUGCOLD_25SEEDS_MECHANISM.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# W/T/L EXTRA SUMMARY
# ============================================================

extra = {}

for metric in [
    "AUROC",
    "AUPR",
]:

    for left, right, name in contrasts:

        diff = (
            df[f"{left}_{metric}"]
            - df[f"{right}_{metric}"]
        ).to_numpy()

        w, t, l = wtl(diff)

        extra[
            f"{metric}_{name}"
        ] = {
            "W":
                w,
            "T":
                t,
            "L":
                l,
        }


# ============================================================
# JSON SUMMARY
# ============================================================

final_json = {
    "n_seeds":
        25,

    "seeds":
        list(range(25)),

    "pair_rows_per_seed":
        7170,

    "all_pair_files_complete":
        True,

    "all_macro_invariance_pass":
        True,

    "mean_sd":
        summary_rows,

    "paired_stats":
        stat_rows,

    "mechanism":
        mech_rows,

    "wtl":
        extra,
}


with open(
    OUT /
    "PRISM_DRUGCOLD_25SEEDS_FINAL_SUMMARY.json",
    "w",
) as f:

    json.dump(
        final_json,
        f,
        indent=2,
    )


# ============================================================
# CONSOLE
# ============================================================

print(
    "========== PRISM DRUG-COLD 25-SEED FORMAL SUMMARY =========="
)

print("\n--- POOLED / MACRO PERFORMANCE ---")

for _, r in desc_df.iterrows():

    print(
        f"{r['model']:5s} "
        f"{r['metric']:11s} "
        f"{r['mean']:.6f} ± "
        f"{r['sd']:.6f}"
    )


print("\n--- PAIRED CONTRASTS ---")

for _, r in stats_df.iterrows():

    print(
        f"{r['metric']:5s} "
        f"{r['contrast']:15s} "
        f"diff={r['mean_diff']:+.6f} "
        f"95%CI=["
        f"{r['ci95_low']:+.6f}, "
        f"{r['ci95_high']:+.6f}] "
        f"dz={r['cohens_dz']:+.3f} "
        f"W/T/L="
        f"{int(r['wins'])}/"
        f"{int(r['ties'])}/"
        f"{int(r['losses'])} "
        f"p={r['wilcoxon_p']:.6g} "
        f"Holm={r['wilcoxon_p_holm']:.6g}"
    )


print("\n--- MECHANISM ---")

for _, r in mech_df.iterrows():

    print(
        f"{r['quantity']:30s} "
        f"{r['mean']:.6f} ± "
        f"{r['sd']:.6f} "
        f"[{r['min']:.6f}, "
        f"{r['max']:.6f}]"
    )


print("\n--- QC ---")

print(
    "seeds:",
    len(df),
)

print(
    "all pair rows == 7170:",
    bool(
        (qc_df["rows"] == 7170).all()
    ),
)

print(
    "max Raw/QRC macro AUC diff:",
    qc_df[
        "macro_auc_diff"
    ].max(),
)

print(
    "max Raw/QRC macro AUPR diff:",
    qc_df[
        "macro_aupr_diff"
    ].max(),
)

print(
    "\nPRISM_DRUGCOLD_25SEEDS_FORMAL_COMPLETE"
)
