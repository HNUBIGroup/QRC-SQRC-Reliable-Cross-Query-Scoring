from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.home() / "ECARF_Article2_20260911" / "14_sqrc_estimation_stress_20260920"
OUT = ROOT / "03_outputs" / "formal25"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(25))

TAU_TRUE = 1.0
N_VAL_QUERIES = 100
N_TEST_QUERIES = 100

N_GRID = [5, 10, 25, 50, 65, 100, 379, 1195]
SIGMA_GRID = [0.25, 0.5, 1.0, 2.0]


def estimate_tau_b2(b_hat, se2):
    observed_variance = float(np.var(b_hat, ddof=1)) if len(b_hat) > 1 else 0.0
    mean_se2 = float(np.mean(se2))
    tau_b2 = max(observed_variance - mean_se2, 0.0)
    return tau_b2, observed_variance, mean_se2


all_rows = []
seed_qc_rows = []

for seed in SEEDS:

    rng_bias = np.random.default_rng(100000 + seed)

    b_val_true = rng_bias.normal(
        0.0, TAU_TRUE, size=N_VAL_QUERIES
    )

    b_test_true = rng_bias.normal(
        0.0, TAU_TRUE, size=N_TEST_QUERIES
    )

    seed_rows = []
    condition_id = 0

    max_qrc_inv = 0.0
    max_sqrc_inv = 0.0

    for sigma in SIGMA_GRID:
        for n in N_GRID:

            rng_val = np.random.default_rng(
                200000 + seed * 10000 + condition_id
            )

            rng_test = np.random.default_rng(
                300000 + seed * 10000 + condition_id
            )

            # ====================================
            # Validation-only tau_b2 estimation
            # ====================================

            val_r = rng_val.normal(
                0.0,
                sigma,
                size=(N_VAL_QUERIES, n)
            )

            val_raw = b_val_true[:, None] + val_r

            val_b_hat = np.mean(
                val_raw,
                axis=1
            )

            val_sample_var = np.var(
                val_raw,
                axis=1,
                ddof=1
            )

            val_se2 = val_sample_var / n

            tau_b2, observed_var, mean_val_se2 = estimate_tau_b2(
                val_b_hat,
                val_se2
            )

            # ====================================
            # Independent test
            # ====================================

            test_r = rng_test.normal(
                0.0,
                sigma,
                size=(N_TEST_QUERIES, n)
            )

            target = test_r

            raw = (
                b_test_true[:, None]
                + test_r
            )

            test_b_hat = np.mean(
                raw,
                axis=1
            )

            test_sample_var = np.var(
                raw,
                axis=1,
                ddof=1
            )

            test_se2 = (
                test_sample_var / n
            )

            denominator = (
                tau_b2 + test_se2
            )

            weights = np.divide(
                tau_b2,
                denominator,
                out=np.zeros_like(test_se2),
                where=denominator > 0
            )

            # ====================================
            # QRC / SQRC
            # ====================================

            qrc = (
                raw
                - test_b_hat[:, None]
            )

            sqrc = (
                raw
                - weights[:, None]
                * test_b_hat[:, None]
            )

            qrc_bias_est = test_b_hat

            sqrc_bias_est = (
                weights * test_b_hat
            )

            qrc_bias_error = (
                qrc_bias_est
                - b_test_true
            )

            sqrc_bias_error = (
                sqrc_bias_est
                - b_test_true
            )

            qrc_score_error = (
                qrc - target
            )

            sqrc_score_error = (
                sqrc - target
            )

            qrc_bias_mse = float(
                np.mean(qrc_bias_error ** 2)
            )

            sqrc_bias_mse = float(
                np.mean(sqrc_bias_error ** 2)
            )

            qrc_score_rmse = float(
                np.sqrt(
                    np.mean(
                        qrc_score_error ** 2
                    )
                )
            )

            sqrc_score_rmse = float(
                np.sqrt(
                    np.mean(
                        sqrc_score_error ** 2
                    )
                )
            )

            u = float(
                sigma ** 2 / n
            )

            oracle_weight = float(
                TAU_TRUE ** 2
                /
                (
                    TAU_TRUE ** 2
                    + u
                )
            )

            # ====================================
            # Ranking invariance QC
            # ====================================

            qrc_inv = 0.0
            sqrc_inv = 0.0

            for i in range(N_TEST_QUERIES):

                raw_diff = (
                    raw[i]
                    - raw[i, 0]
                )

                qrc_diff = (
                    qrc[i]
                    - qrc[i, 0]
                )

                sqrc_diff = (
                    sqrc[i]
                    - sqrc[i, 0]
                )

                qrc_inv = max(
                    qrc_inv,
                    float(
                        np.max(
                            np.abs(
                                raw_diff
                                - qrc_diff
                            )
                        )
                    )
                )

                sqrc_inv = max(
                    sqrc_inv,
                    float(
                        np.max(
                            np.abs(
                                raw_diff
                                - sqrc_diff
                            )
                        )
                    )
                )

            max_qrc_inv = max(
                max_qrc_inv,
                qrc_inv
            )

            max_sqrc_inv = max(
                max_sqrc_inv,
                sqrc_inv
            )

            relative_reduction = (
                (
                    qrc_bias_mse
                    - sqrc_bias_mse
                )
                / qrc_bias_mse
                if qrc_bias_mse > 0
                else np.nan
            )

            row = {
                "seed": seed,
                "sigma": sigma,
                "n_candidates": n,

                "nominal_uncertainty_sigma2_over_n": u,

                "tau_true2": TAU_TRUE ** 2,

                "tau_b2_validation":
                    tau_b2,

                "tau_b2_abs_error":
                    abs(
                        tau_b2
                        - TAU_TRUE ** 2
                    ),

                "validation_observed_bhat_variance":
                    observed_var,

                "validation_mean_se2":
                    mean_val_se2,

                "test_mean_se2":
                    float(
                        np.mean(test_se2)
                    ),

                "mean_weight":
                    float(
                        np.mean(weights)
                    ),

                "median_weight":
                    float(
                        np.median(weights)
                    ),

                "min_weight":
                    float(
                        np.min(weights)
                    ),

                "max_weight":
                    float(
                        np.max(weights)
                    ),

                "oracle_weight":
                    oracle_weight,

                "qrc_bias_mse":
                    qrc_bias_mse,

                "sqrc_bias_mse":
                    sqrc_bias_mse,

                "sqrc_minus_qrc_bias_mse":
                    sqrc_bias_mse
                    - qrc_bias_mse,

                "sqrc_bias_mse_relative_reduction":
                    relative_reduction,

                "qrc_score_rmse":
                    qrc_score_rmse,

                "sqrc_score_rmse":
                    sqrc_score_rmse,

                "sqrc_minus_qrc_score_rmse":
                    sqrc_score_rmse
                    - qrc_score_rmse,

                "qrc_pairwise_invariance_error":
                    qrc_inv,

                "sqrc_pairwise_invariance_error":
                    sqrc_inv,
            }

            seed_rows.append(row)
            all_rows.append(row)

            condition_id += 1

    sdf = pd.DataFrame(seed_rows)

    rho_weight = float(
        sdf[
            "nominal_uncertainty_sigma2_over_n"
        ].corr(
            1.0 - sdf["mean_weight"],
            method="spearman"
        )
    )

    rho_gain = float(
        sdf[
            "nominal_uncertainty_sigma2_over_n"
        ].corr(
            -sdf[
                "sqrc_minus_qrc_bias_mse"
            ],
            method="spearman"
        )
    )

    seed_qc_rows.append({
        "seed": seed,
        "n_conditions": len(sdf),

        "all_tau_nonnegative":
            bool(
                (
                    sdf[
                        "tau_b2_validation"
                    ] >= 0
                ).all()
            ),

        "all_weights_in_0_1":
            bool(
                (
                    (
                        sdf["min_weight"]
                        >= 0
                    )
                    &
                    (
                        sdf["max_weight"]
                        <= 1
                    )
                ).all()
            ),

        "max_qrc_pairwise_invariance_error":
            max_qrc_inv,

        "max_sqrc_pairwise_invariance_error":
            max_sqrc_inv,

        "spearman_uncertainty_vs_one_minus_weight":
            rho_weight,

        "spearman_uncertainty_vs_sqrc_mse_gain":
            rho_gain,

        "sqrc_better_bias_mse_conditions":
            int(
                (
                    sdf["sqrc_bias_mse"]
                    <
                    sdf["qrc_bias_mse"]
                ).sum()
            ),

        "sqrc_better_score_rmse_conditions":
            int(
                (
                    sdf["sqrc_score_rmse"]
                    <
                    sdf["qrc_score_rmse"]
                ).sum()
            ),
    })


df = pd.DataFrame(all_rows)

seed_qc = pd.DataFrame(
    seed_qc_rows
)

# ====================================
# Condition-level aggregation
# ====================================

metrics = [
    "tau_b2_validation",
    "tau_b2_abs_error",
    "validation_mean_se2",
    "test_mean_se2",

    "mean_weight",
    "oracle_weight",

    "qrc_bias_mse",
    "sqrc_bias_mse",
    "sqrc_minus_qrc_bias_mse",
    "sqrc_bias_mse_relative_reduction",

    "qrc_score_rmse",
    "sqrc_score_rmse",
    "sqrc_minus_qrc_score_rmse",
]

agg_dict = {}

for metric in metrics:
    agg_dict[metric] = [
        "mean",
        "std"
    ]

condition_agg = (
    df.groupby(
        [
            "sigma",
            "n_candidates",
            "nominal_uncertainty_sigma2_over_n"
        ],
        as_index=False
    )
    .agg(agg_dict)
)

condition_agg.columns = [
    "_".join(
        [
            str(x)
            for x in col
            if str(x)
        ]
    ).rstrip("_")
    if isinstance(col, tuple)
    else col
    for col in condition_agg.columns
]

wins = (
    df.assign(
        sqrc_better_bias_mse=
            df["sqrc_bias_mse"]
            < df["qrc_bias_mse"],

        sqrc_better_score_rmse=
            df["sqrc_score_rmse"]
            < df["qrc_score_rmse"]
    )
    .groupby(
        [
            "sigma",
            "n_candidates"
        ],
        as_index=False
    )
    .agg(
        sqrc_better_bias_mse_seeds=(
            "sqrc_better_bias_mse",
            "sum"
        ),
        sqrc_better_score_rmse_seeds=(
            "sqrc_better_score_rmse",
            "sum"
        )
    )
)

condition_agg = condition_agg.merge(
    wins,
    on=[
        "sigma",
        "n_candidates"
    ],
    how="left"
)

# ====================================
# Across-condition relation
# ====================================

condition_mean = (
    df.groupby(
        [
            "sigma",
            "n_candidates",
            "nominal_uncertainty_sigma2_over_n"
        ],
        as_index=False
    )
    .mean(numeric_only=True)
)

rho_weight_condition = float(
    condition_mean[
        "nominal_uncertainty_sigma2_over_n"
    ].corr(
        1.0
        - condition_mean["mean_weight"],
        method="spearman"
    )
)

rho_gain_condition = float(
    condition_mean[
        "nominal_uncertainty_sigma2_over_n"
    ].corr(
        -condition_mean[
            "sqrc_minus_qrc_bias_mse"
        ],
        method="spearman"
    )
)

# ====================================
# Save
# ====================================

df.to_csv(
    OUT / "ALL_SEED_CONDITIONS.csv",
    index=False
)

condition_agg.to_csv(
    OUT / "CONDITION_AGGREGATE.csv",
    index=False
)

seed_qc.to_csv(
    OUT / "SEED_QC_SUMMARY.csv",
    index=False
)

summary = {
    "n_seeds": len(SEEDS),

    "n_conditions_per_seed":
        len(N_GRID) * len(SIGMA_GRID),

    "n_total_rows":
        len(df),

    "all_tau_nonnegative":
        bool(
            (
                df["tau_b2_validation"]
                >= 0
            ).all()
        ),

    "all_weights_in_0_1":
        bool(
            (
                (
                    df["min_weight"]
                    >= 0
                )
                &
                (
                    df["max_weight"]
                    <= 1
                )
            ).all()
        ),

    "max_qrc_pairwise_invariance_error":
        float(
            df[
                "qrc_pairwise_invariance_error"
            ].max()
        ),

    "max_sqrc_pairwise_invariance_error":
        float(
            df[
                "sqrc_pairwise_invariance_error"
            ].max()
        ),

    "mean_tau_b2_validation":
        float(
            df[
                "tau_b2_validation"
            ].mean()
        ),

    "mean_tau_b2_abs_error":
        float(
            df[
                "tau_b2_abs_error"
            ].mean()
        ),

    "condition_mean_spearman_uncertainty_vs_one_minus_weight":
        rho_weight_condition,

    "condition_mean_spearman_uncertainty_vs_sqrc_mse_gain":
        rho_gain_condition,

    "sqrc_better_bias_mse_total":
        int(
            (
                df["sqrc_bias_mse"]
                <
                df["qrc_bias_mse"]
            ).sum()
        ),

    "sqrc_better_score_rmse_total":
        int(
            (
                df["sqrc_score_rmse"]
                <
                df["qrc_score_rmse"]
            ).sum()
        ),

    "total_comparisons":
        int(len(df)),

    "uses_test_labels": False,
    "tau_estimated_validation_only": True,
    "sample_variance_ddof": 1,
}

with open(
    OUT / "FORMAL25_SUMMARY.json",
    "w"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )

# ====================================
# Console
# ====================================

print(
    "========== SQRC STRESS FORMAL25 =========="
)

for k, v in summary.items():
    print(f"{k}: {v}")

print(
    "\n========== CONDITION SUMMARY =========="
)

cols = [
    "sigma",
    "n_candidates",
    "nominal_uncertainty_sigma2_over_n",

    "mean_weight_mean",

    "qrc_bias_mse_mean",
    "sqrc_bias_mse_mean",

    "sqrc_bias_mse_relative_reduction_mean",

    "sqrc_better_bias_mse_seeds",
]

print(
    condition_agg[
        cols
    ].sort_values(
        [
            "sigma",
            "n_candidates"
        ]
    ).to_string(
        index=False
    )
)

print(
    "\nOutputs:",
    OUT
)
