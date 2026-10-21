from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/home/u2508283030/MDA_CellPainting_Multiview_20260910")
OUT = Path("/home/u2508283030/ECARF_Article2_20260911/02_results")
OUT.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
N_REPEAT = 20

def metric(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return (
        roc_auc_score(y, s),
        average_precision_score(y, s)
    )

records = []

for mode in ["drug", "mirna"]:

    query_col = "drug" if mode == "drug" else "miRNA"

    for seed in range(25):

        print(f"{mode.upper()} seed {seed}/24", flush=True)

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
            pd.read_csv(
                split_file,
                sep="\t"
            )["entity"]
        )

        x = x[
            x[query_col].isin(test_entities)
        ].copy().reset_index(drop=True)

        x["y"] = (
            x["split"] == "test_known"
        ).astype(int)

        mol = x["molecular_score"].to_numpy()
        raw = x["final_score"].to_numpy()
        delta = raw - mol
        y = x["y"].to_numpy()

        x["_delta"] = delta

        full_bias = (
            x.groupby(query_col)["_delta"]
            .transform("mean")
            .to_numpy()
        )

        qrc_full = mol + delta - full_bias

        full_auc, full_apr = metric(y, qrc_full)

        query_groups = {
            q: idx.to_numpy()
            for q, idx in x.groupby(query_col).groups.items()
        }

        full_bias_map = (
            x.groupby(query_col)["_delta"]
            .mean()
            .to_dict()
        )

        for frac in FRACTIONS:

            repeats = 1 if frac == 1.0 else N_REPEAT

            for rep in range(repeats):

                rng = np.random.default_rng(
                    4000000
                    + seed * 10000
                    + rep * 100
                    + int(frac * 100)
                    + (0 if mode == "drug" else 500000)
                )

                estimated_bias = {}
                abs_errors = []

                for q, idx in query_groups.items():

                    n = len(idx)

                    if frac == 1.0:
                        chosen = idx
                    else:
                        k = max(
                            1,
                            int(round(n * frac))
                        )

                        chosen = rng.choice(
                            idx,
                            size=k,
                            replace=False
                        )

                    b_hat = float(
                        delta[chosen].mean()
                    )

                    estimated_bias[q] = b_hat

                    abs_errors.append(
                        abs(
                            b_hat
                            - full_bias_map[q]
                        )
                    )

                b_est = np.array([
                    estimated_bias[q]
                    for q in x[query_col]
                ])

                qrc_est = (
                    mol
                    + delta
                    - b_est
                )

                auc, apr = metric(
                    y,
                    qrc_est
                )

                records.append({
                    "mode": mode,
                    "seed": seed,
                    "fraction": frac,
                    "repeat": rep,
                    "AUROC": auc,
                    "AUPR": apr,
                    "FullQRC_AUROC": full_auc,
                    "FullQRC_AUPR": full_apr,
                    "Delta_AUROC_vs_Full":
                        auc - full_auc,
                    "Delta_AUPR_vs_Full":
                        apr - full_apr,
                    "Bias_MAE":
                        float(np.mean(abs_errors)),
                })

res = pd.DataFrame(records)

res.to_csv(
    OUT / "stageD_candidate_set_stability_full.tsv",
    sep="\t",
    index=False
)

seedmean = (
    res.groupby(
        ["mode", "seed", "fraction"],
        as_index=False
    )
    .mean(numeric_only=True)
)

seedmean.to_csv(
    OUT / "stageD_candidate_set_stability_seedmean.tsv",
    sep="\t",
    index=False
)

print("\n===== CANDIDATE-SET STABILITY =====")

for mode in ["drug", "mirna"]:

    print(f"\n### {mode.upper()}-COLD")

    z = seedmean[
        seedmean["mode"] == mode
    ]

    for frac in FRACTIONS:

        g = z[
            z["fraction"] == frac
        ]

        print(
            f"fraction={frac:4.2f} | "
            f"AUC={g.AUROC.mean():.4f} | "
            f"AUPR={g.AUPR.mean():.4f} | "
            f"dAUC={g.Delta_AUROC_vs_Full.mean():+.4f} | "
            f"dAUPR={g.Delta_AUPR_vs_Full.mean():+.4f} | "
            f"bias_MAE={g.Bias_MAE.mean():.6f}"
        )

print("\nCANDIDATE_SET_STABILITY_COMPLETE")
