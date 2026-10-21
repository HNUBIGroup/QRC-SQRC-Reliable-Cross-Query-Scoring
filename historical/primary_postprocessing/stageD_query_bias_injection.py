from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/home/u2508283030/MDA_CellPainting_Multiview_20260910")
OUT = Path("/home/u2508283030/ECARF_Article2_20260911/02_results")
OUT.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
N_REPEAT = 20

def metric(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return (
        roc_auc_score(y, s),
        average_precision_score(y, s)
    )

def macro_metric(df, score_col, query_col):
    aucs, aprs = [], []

    for _, g in df.groupby(query_col):
        y = g["y"].to_numpy()
        s = g[score_col].to_numpy()

        if len(np.unique(y)) < 2:
            continue

        aucs.append(roc_auc_score(y, s))
        aprs.append(average_precision_score(y, s))

    return (
        np.mean(aucs) if aucs else np.nan,
        np.mean(aprs) if aprs else np.nan
    )

records = []

for mode in ["drug", "mirna"]:

    query_col = "drug" if mode == "drug" else "miRNA"

    for seed in range(25):

        tag = f"{mode}_cold_seed{seed}"

        score_file = (
            ROOT / "05_results" / tag /
            f"all_pair_scores_{tag}.tsv"
        )

        split_file = (
            ROOT / "03_splits" / tag /
            "test_entities.tsv"
        )

        x = pd.read_csv(score_file, sep="\t")
        test_entities = set(
            pd.read_csv(split_file, sep="\t")["entity"]
        )

        x = x[
            x[query_col].isin(test_entities)
        ].copy()

        x["y"] = (
            x["split"] == "test_known"
        ).astype(int)

        mol = x["molecular_score"].to_numpy()
        raw = x["final_score"].to_numpy()

        correction = raw - mol

        base_bias = (
            pd.Series(correction)
            .groupby(x[query_col].reset_index(drop=True))
            .transform("mean")
            .to_numpy()
        )

        qrc_base = mol + correction - base_bias

        scale = np.std(raw, ddof=1)

        query_values = x[query_col].to_numpy()
        unique_queries = np.unique(query_values)

        for alpha in ALPHAS:

            for rep in range(N_REPEAT):

                rng = np.random.default_rng(
                    1000000
                    + seed * 10000
                    + rep * 100
                    + (0 if mode == "drug" else 1)
                )

                shifts = rng.normal(
                    loc=0.0,
                    scale=alpha * scale,
                    size=len(unique_queries)
                )

                shift_map = dict(
                    zip(unique_queries, shifts)
                )

                query_shift = np.array([
                    shift_map[q]
                    for q in query_values
                ])

                raw_injected = raw + query_shift

                corr_injected = raw_injected - mol

                tmp = pd.DataFrame({
                    "query": query_values,
                    "corr": corr_injected
                })

                injected_bias = (
                    tmp.groupby("query")["corr"]
                    .transform("mean")
                    .to_numpy()
                )

                qrc_injected = (
                    mol
                    + corr_injected
                    - injected_bias
                )

                work = x[
                    [query_col, "y"]
                ].copy()

                work["Raw"] = raw_injected
                work["QRC"] = qrc_injected

                raw_auc, raw_apr = metric(
                    work["y"].to_numpy(),
                    work["Raw"].to_numpy()
                )

                qrc_auc, qrc_apr = metric(
                    work["y"].to_numpy(),
                    work["QRC"].to_numpy()
                )

                raw_macro_auc, raw_macro_apr = (
                    macro_metric(
                        work,
                        "Raw",
                        query_col
                    )
                )

                qrc_macro_auc, qrc_macro_apr = (
                    macro_metric(
                        work,
                        "QRC",
                        query_col
                    )
                )

                max_qrc_diff = float(
                    np.max(
                        np.abs(
                            qrc_injected
                            - qrc_base
                        )
                    )
                )

                records.append({
                    "mode": mode,
                    "seed": seed,
                    "alpha": alpha,
                    "repeat": rep,

                    "Raw_Pooled_AUROC": raw_auc,
                    "Raw_Pooled_AUPR": raw_apr,

                    "QRC_Pooled_AUROC": qrc_auc,
                    "QRC_Pooled_AUPR": qrc_apr,

                    "Raw_Macro_AUROC": raw_macro_auc,
                    "Raw_Macro_AUPR": raw_macro_apr,

                    "QRC_Macro_AUROC": qrc_macro_auc,
                    "QRC_Macro_AUPR": qrc_macro_apr,

                    "QRC_Invariance_MaxAbsDiff":
                        max_qrc_diff,
                })

res = pd.DataFrame(records)

res.to_csv(
    OUT / "stageD_query_bias_injection_full.tsv",
    sep="\t",
    index=False
)

# First average repetitions within each seed
seed_mean = (
    res.groupby(
        ["mode", "seed", "alpha"],
        as_index=False
    )
    .mean(numeric_only=True)
)

seed_mean.to_csv(
    OUT / "stageD_query_bias_injection_seedmean.tsv",
    sep="\t",
    index=False
)

print("\n===== QUERY-BIAS INJECTION =====")

for mode in ["drug", "mirna"]:

    print(f"\n### {mode.upper()}-COLD")

    z = seed_mean[
        seed_mean["mode"] == mode
    ]

    for alpha in ALPHAS:

        g = z[z["alpha"] == alpha]

        print(
            f"alpha={alpha:4.2f} | "
            f"Raw pooled AUC="
            f"{g.Raw_Pooled_AUROC.mean():.4f} | "
            f"QRC pooled AUC="
            f"{g.QRC_Pooled_AUROC.mean():.4f} | "
            f"Raw macro AUC="
            f"{g.Raw_Macro_AUROC.mean():.4f} | "
            f"QRC maxdiff="
            f"{g.QRC_Invariance_MaxAbsDiff.max():.3e}"
        )

print("\nQUERY_BIAS_INJECTION_COMPLETE")
