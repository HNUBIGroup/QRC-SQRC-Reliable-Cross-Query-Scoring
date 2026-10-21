from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/home/u2508283030/MDA_CellPainting_Multiview_20260910")
OUT = Path("/home/u2508283030/ECARF_Article2_20260911/02_results")
OUT.mkdir(parents=True, exist_ok=True)

N_REPEAT = 20

def metrics(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return (
        roc_auc_score(y, s),
        average_precision_score(y, s)
    )

def macro_auc(df, score, query):
    vals = []
    for _, g in df.groupby(query):
        y = g["y"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        vals.append(
            roc_auc_score(
                y,
                g[score].to_numpy()
            )
        )
    return np.mean(vals) if vals else np.nan

records = []

for mode in ["drug", "mirna"]:

    query_col = "drug" if mode == "drug" else "miRNA"

    for seed in range(25):

        print(
            f"{mode.upper()} seed {seed}/24",
            flush=True
        )

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

        tmp = pd.DataFrame({
            "query": x[query_col],
            "delta": delta,
        })

        b = (
            tmp.groupby("query")["delta"]
            .transform("mean")
            .to_numpy()
        )

        r = delta - b

        # Original decomposed reconstruction
        reconstructed = mol + b + r

        # QRC: remove query bias
        qrc = mol + r

        base = x[[query_col, "y"]].copy()
        base["Molecular"] = mol
        base["Raw"] = raw
        base["QRC"] = qrc

        for name in ["Molecular", "Raw", "QRC"]:
            auc, ap = metrics(
                base["y"].to_numpy(),
                base[name].to_numpy()
            )

            records.append({
                "mode": mode,
                "seed": seed,
                "repeat": -1,
                "condition": name,
                "pooled_AUROC": auc,
                "pooled_AUPR": ap,
                "macro_AUROC":
                    macro_auc(
                        base,
                        name,
                        query_col
                    ),
            })

        # --------------------------------------------------
        # Counterfactual 1:
        # Shuffle query-level bias b among queries.
        # r remains attached to original pairs.
        #
        # Expected:
        # macro ranking unchanged;
        # pooled comparability changes.
        # --------------------------------------------------
        query_values = x[query_col].to_numpy()
        unique_queries = np.unique(query_values)

        b_by_query = (
            pd.DataFrame({
                "query": query_values,
                "b": b
            })
            .groupby("query")["b"]
            .first()
        )

        for rep in range(N_REPEAT):

            rng = np.random.default_rng(
                2000000
                + seed * 1000
                + rep
                + (0 if mode == "drug" else 500000)
            )

            shuffled_b_values = (
                b_by_query.to_numpy().copy()
            )
            rng.shuffle(shuffled_b_values)

            shuffled_b_map = dict(
                zip(
                    b_by_query.index,
                    shuffled_b_values
                )
            )

            b_shuffle = np.array([
                shuffled_b_map[q]
                for q in query_values
            ])

            s = mol + b_shuffle + r

            auc, ap = metrics(
                x["y"].to_numpy(),
                s
            )

            records.append({
                "mode": mode,
                "seed": seed,
                "repeat": rep,
                "condition": "Bias_Shuffle",
                "pooled_AUROC": auc,
                "pooled_AUPR": ap,
                # theoretically equal to Raw macro
                "macro_AUROC":
                    macro_auc(
                        base,
                        "Raw",
                        query_col
                    ),
            })

        # --------------------------------------------------
        # Counterfactual 2:
        # Shuffle residual r WITHIN each query.
        # Keep b unchanged.
        #
        # Expected:
        # destroys pair-specific complementary structure,
        # therefore query ranking should deteriorate.
        # --------------------------------------------------
        group_indices = [
            g.index.to_numpy()
            for _, g in x.groupby(query_col)
        ]

        for rep in range(N_REPEAT):

            rng = np.random.default_rng(
                3000000
                + seed * 1000
                + rep
                + (0 if mode == "drug" else 500000)
            )

            r_shuffle = r.copy()

            for idx in group_indices:
                vals = r_shuffle[idx].copy()
                rng.shuffle(vals)
                r_shuffle[idx] = vals

            s = mol + b + r_shuffle

            work = x[
                [query_col, "y"]
            ].copy()
            work["score"] = s

            auc, ap = metrics(
                work["y"].to_numpy(),
                s
            )

            records.append({
                "mode": mode,
                "seed": seed,
                "repeat": rep,
                "condition":
                    "Residual_Shuffle_WithinQuery",
                "pooled_AUROC": auc,
                "pooled_AUPR": ap,
                "macro_AUROC":
                    macro_auc(
                        work,
                        "score",
                        query_col
                    ),
            })

res = pd.DataFrame(records)

res.to_csv(
    OUT / "stageD_residual_counterfactual_full.tsv",
    sep="\t",
    index=False
)

print("\n===== RESIDUAL COUNTERFACTUAL =====")

for mode in ["drug", "mirna"]:

    print(f"\n### {mode.upper()}-COLD")

    z = res[res["mode"] == mode]

    for condition in [
        "Molecular",
        "Raw",
        "QRC",
        "Bias_Shuffle",
        "Residual_Shuffle_WithinQuery",
    ]:

        g = z[
            z["condition"] == condition
        ]

        # average repeats first
        if condition in [
            "Bias_Shuffle",
            "Residual_Shuffle_WithinQuery"
        ]:
            g = (
                g.groupby("seed", as_index=False)
                .mean(numeric_only=True)
            )

        print(
            f"{condition:30s} | "
            f"pooled AUC={g.pooled_AUROC.mean():.4f} | "
            f"pooled AUPR={g.pooled_AUPR.mean():.4f} | "
            f"macro AUC={g.macro_AUROC.mean():.4f}"
        )

print("\nRESIDUAL_COUNTERFACTUAL_COMPLETE")
