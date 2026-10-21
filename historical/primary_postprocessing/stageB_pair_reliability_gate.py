import os
import csv
import glob
import re
import math
import random
from collections import defaultdict
from statistics import mean, stdev, median

import torch
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.expanduser(
    "~/MDA_CellPainting_Multiview_20260910"
)

OUT = os.path.expanduser(
    "~/ECARF_Article2_20260911/02_results"
)

os.makedirs(OUT, exist_ok=True)

OUT_SEED = os.path.join(
    OUT,
    "stageB_drugcold_25seed_metrics.tsv"
)

DEVICE = "cpu"

TRAIN_STEPS = 500
BATCH_SIZE = 4096
LR = 0.03
WEIGHT_DECAY = 1e-3
EVAL_INTERVAL = 20

torch.set_num_threads(2)


# ============================================================
# METRICS
# ============================================================

def rankdata(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)

    i = 0
    while i < len(order):
        j = i + 1

        while (
            j < len(order)
            and x[order[j]] == x[order[i]]
        ):
            j += 1

        r = (i + 1 + j) / 2.0

        for k in range(i, j):
            ranks[order[k]] = r

        i = j

    return ranks


def auroc(y, score):
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    if n_pos == 0 or n_neg == 0:
        return None

    ranks = rankdata(score)

    s = sum(
        ranks[i]
        for i, v in enumerate(y)
        if v == 1
    )

    return (
        s - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)


def aupr(y, score):
    n_pos = sum(y)

    if n_pos == 0:
        return None

    order = sorted(
        range(len(score)),
        key=lambda i: score[i],
        reverse=True
    )

    hit = 0
    total = 0.0

    for k, i in enumerate(order, 1):
        if y[i] == 1:
            hit += 1
            total += hit / k

    return total / n_pos


# ============================================================
# PREPARE ONE SCOPE
# ============================================================

def prepare_scope(all_rows, drugs, positive_name):

    rows = [
        r for r in all_rows
        if (
            r["drug"] in drugs
            and r["split"]
            in (positive_name, "unlabeled")
        )
    ]

    # ------------------------------------------
    # raw scores
    # ------------------------------------------

    molecular = [
        float(r["molecular_score"])
        for r in rows
    ]

    unified = [
        float(r["unified_score"])
        for r in rows
    ]

    delta = [
        u - m
        for u, m in zip(unified, molecular)
    ]

    # ------------------------------------------
    # query-conditioned residual
    # ------------------------------------------

    by_drug = defaultdict(list)

    for i, r in enumerate(rows):
        by_drug[r["drug"]].append(i)

    residual = [0.0] * len(rows)
    rank_gap = [0.0] * len(rows)

    for drug, ids in by_drug.items():

        dvals = [delta[i] for i in ids]
        dmean = mean(dvals)

        for i in ids:
            residual[i] = delta[i] - dmean

        # molecular / unified within-query ranks
        mvals = [molecular[i] for i in ids]
        uvals = [unified[i] for i in ids]

        mr = rankdata(mvals)
        ur = rankdata(uvals)

        denom = max(1, len(ids) - 1)

        for local_i, global_i in enumerate(ids):
            rank_gap[global_i] = (
                abs(
                    mr[local_i]
                    -
                    ur[local_i]
                )
                / denom
            )

    y = [
        1 if r["split"] == positive_name
        else 0
        for r in rows
    ]

    return {
        "rows": rows,
        "y": y,
        "molecular": molecular,
        "unified": unified,
        "residual": residual,
        "abs_residual": [
            abs(x) for x in residual
        ],
        "rank_gap": rank_gap,
    }


# ============================================================
# STANDARDIZATION
# ============================================================

def fit_standardizer(scope):

    x1 = scope["abs_residual"]
    x2 = scope["rank_gap"]

    mu1 = mean(x1)
    mu2 = mean(x2)

    sd1 = math.sqrt(
        mean((x - mu1) ** 2 for x in x1)
    )

    sd2 = math.sqrt(
        mean((x - mu2) ** 2 for x in x2)
    )

    sd1 = max(sd1, 1e-8)
    sd2 = max(sd2, 1e-8)

    return (mu1, sd1, mu2, sd2)


def make_features(scope, stats):

    mu1, sd1, mu2, sd2 = stats

    feats = []

    for a, g in zip(
        scope["abs_residual"],
        scope["rank_gap"]
    ):
        feats.append([
            (a - mu1) / sd1,
            (g - mu2) / sd2,
        ])

    return torch.tensor(
        feats,
        dtype=torch.float32,
        device=DEVICE
    )


# ============================================================
# SCORE WITH GATE
# ============================================================

def gated_score(scope, features, params):

    w0, w1, w2 = params

    logits = (
        w0
        +
        w1 * features[:, 0]
        +
        w2 * features[:, 1]
    )

    gate = torch.sigmoid(logits)

    mol = torch.tensor(
        scope["molecular"],
        dtype=torch.float32,
        device=DEVICE
    )

    res = torch.tensor(
        scope["residual"],
        dtype=torch.float32,
        device=DEVICE
    )

    score = mol + gate * res

    return score, gate


# ============================================================
# MACRO METRICS
# ============================================================

def macro_metrics(scope, scores):

    grouped = defaultdict(list)

    for i, r in enumerate(scope["rows"]):
        grouped[r["drug"]].append(i)

    aucs = []
    aps = []

    for drug, ids in grouped.items():

        y = [
            scope["y"][i]
            for i in ids
        ]

        s = [
            scores[i]
            for i in ids
        ]

        a = auroc(y, s)
        p = aupr(y, s)

        if a is not None:
            aucs.append(a)

        if p is not None:
            aps.append(p)

    return mean(aucs), mean(aps)


# ============================================================
# DISCOVER FILES
# ============================================================

files = glob.glob(
    os.path.join(
        ROOT,
        "05_results",
        "**",
        "all_pair_scores*.tsv"
    ),
    recursive=True
)

files = [
    f for f in files
    if "drug_cold" in f.lower()
]


def seed_of(path):
    m = re.search(r"seed(\d+)", path)
    return int(m.group(1)) if m else None


files = [
    f for f in files
    if seed_of(f) is not None
]

files.sort(key=seed_of)

print("Drug-cold score files:", len(files))

if len(files) != 25:
    print(
        "WARNING: expected 25 files."
    )


# ============================================================
# MAIN
# ============================================================

seed_results = []

for path in files:

    seed = seed_of(path)

    random.seed(1000 + seed)
    torch.manual_seed(1000 + seed)

    all_rows = []

    with open(
        path,
        encoding="utf-8-sig"
    ) as f:

        for r in csv.DictReader(
            f,
            delimiter="\t"
        ):
            all_rows.append(r)

    train_drugs = {
        r["drug"]
        for r in all_rows
        if r["split"] == "train_known"
    }

    val_drugs = {
        r["drug"]
        for r in all_rows
        if r["split"] == "val_known"
    }

    test_drugs = {
        r["drug"]
        for r in all_rows
        if r["split"] == "test_known"
    }

    train = prepare_scope(
        all_rows,
        train_drugs,
        "train_known"
    )

    val = prepare_scope(
        all_rows,
        val_drugs,
        "val_known"
    )

    test = prepare_scope(
        all_rows,
        test_drugs,
        "test_known"
    )

    stats = fit_standardizer(train)

    x_train = make_features(
        train, stats
    )

    x_val = make_features(
        val, stats
    )

    x_test = make_features(
        test, stats
    )

    y_train = torch.tensor(
        train["y"],
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------------
    # recover old global lambda
    # --------------------------------------------------------

    lambdas = []

    for r in all_rows:

        m = float(r["molecular_score"])
        u = float(r["unified_score"])
        f = float(r["final_score"])

        den = u - m

        if abs(den) > 1e-10:

            lam = (f - m) / den

            if -1e-5 <= lam <= 1.00001:
                lambdas.append(lam)

    base_lambda = median(lambdas)

    clipped = min(
        0.99,
        max(0.01, base_lambda)
    )

    init_logit = math.log(
        clipped / (1.0 - clipped)
    )

    # --------------------------------------------------------
    # only THREE trainable parameters
    # --------------------------------------------------------

    params = torch.nn.Parameter(
        torch.tensor(
            [init_logit, 0.0, 0.0],
            dtype=torch.float32,
            device=DEVICE
        )
    )

    optimizer = torch.optim.Adam(
        [params],
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    pos_idx = torch.where(
        y_train == 1
    )[0]

    neg_idx = torch.where(
        y_train == 0
    )[0]

    best_state = None
    best_key = (-1.0, -1.0)
    best_step = None

    # --------------------------------------------------------
    # pairwise ranking training
    # --------------------------------------------------------

    for step in range(
        1,
        TRAIN_STEPS + 1
    ):

        p = pos_idx[
            torch.randint(
                len(pos_idx),
                (BATCH_SIZE,)
            )
        ]

        n = neg_idx[
            torch.randint(
                len(neg_idx),
                (BATCH_SIZE,)
            )
        ]

        score_train, gate_train = gated_score(
            train,
            x_train,
            params
        )

        diff = (
            score_train[p]
            -
            score_train[n]
        )

        rank_loss = F.softplus(
            -diff
        ).mean()

        # weak prior:
        # do not let gate drift unnecessarily far
        # from validation-selected global lambda.
        gate_prior = (
            gate_train.mean()
            -
            float(base_lambda)
        ) ** 2

        loss = (
            rank_loss
            +
            0.01 * gate_prior
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (
            step % EVAL_INTERVAL == 0
            or step == TRAIN_STEPS
        ):

            with torch.no_grad():

                val_score, _ = gated_score(
                    val,
                    x_val,
                    params
                )

                val_score = (
                    val_score
                    .cpu()
                    .tolist()
                )

                va = auroc(
                    val["y"],
                    val_score
                )

                vp = aupr(
                    val["y"],
                    val_score
                )

                # same principle as old protocol:
                # AUPR first, AUROC second
                key = (
                    -1.0 if vp is None else vp,
                    -1.0 if va is None else va,
                )

                if key > best_key:

                    best_key = key

                    best_state = (
                        params
                        .detach()
                        .cpu()
                        .clone()
                    )

                    best_step = step

    if best_state is None:
        raise RuntimeError(
            f"No valid state for seed {seed}"
        )

    params.data.copy_(
        best_state.to(DEVICE)
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    with torch.no_grad():

        adaptive_score, test_gate = gated_score(
            test,
            x_test,
            params
        )

        adaptive_score = (
            adaptive_score
            .cpu()
            .tolist()
        )

        test_gate = (
            test_gate
            .cpu()
            .tolist()
        )

    # baseline molecular
    molecular = test["molecular"]

    # old raw final
    raw_final = []

    test_row_lookup = {
        (r["drug"], r["miRNA"]): r
        for r in all_rows
    }

    for r in test["rows"]:
        rr = test_row_lookup[
            (r["drug"], r["miRNA"])
        ]
        raw_final.append(
            float(rr["final_score"])
        )

    # fixed-lambda calibrated residual
    drug_cal = [
        m + base_lambda * r
        for m, r in zip(
            test["molecular"],
            test["residual"]
        )
    ]

    y = test["y"]

    methods = {
        "Molecular": molecular,
        "Raw_Final": raw_final,
        "Drug_Calibrated": drug_cal,
        "Adaptive": adaptive_score,
    }

    rec = {
        "seed": seed,
        "base_lambda": base_lambda,
        "best_step": best_step,
        "w0": float(params[0]),
        "w_abs_residual": float(params[1]),
        "w_rank_gap": float(params[2]),
        "gate_mean": mean(test_gate),
        "gate_sd": (
            stdev(test_gate)
            if len(test_gate) > 1
            else 0.0
        ),
        "gate_min": min(test_gate),
        "gate_max": max(test_gate),
    }

    for name, scores in methods.items():

        rec[
            f"{name}_pooled_AUROC"
        ] = auroc(y, scores)

        rec[
            f"{name}_pooled_AUPR"
        ] = aupr(y, scores)

        ma, mp = macro_metrics(
            test,
            scores
        )

        rec[
            f"{name}_macro_AUROC"
        ] = ma

        rec[
            f"{name}_macro_AUPR"
        ] = mp

    seed_results.append(rec)

    print(
        f"seed {seed:02d} | "
        f"lambda={base_lambda:.3f} | "
        f"gate={mean(test_gate):.3f}"
        f"±{stdev(test_gate):.3f} | "
        f"AUC: "
        f"{rec['Molecular_pooled_AUROC']:.3f} -> "
        f"{rec['Drug_Calibrated_pooled_AUROC']:.3f} -> "
        f"{rec['Adaptive_pooled_AUROC']:.3f}"
    )


# ============================================================
# SAVE
# ============================================================

with open(
    OUT_SEED,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            seed_results[0].keys()
        ),
        delimiter="\t"
    )

    w.writeheader()
    w.writerows(seed_results)


# ============================================================
# SUMMARY
# ============================================================

def smry(vals):

    return (
        f"{mean(vals):.4f}"
        f" ± {stdev(vals):.4f}"
    )


print()
print("=" * 74)
print(
    "STAGE B — PAIR-CONDITIONED RELIABILITY / DRUG-COLD 25 SEEDS"
)
print("=" * 74)

for metric in [
    "pooled_AUROC",
    "pooled_AUPR",
    "macro_AUROC",
    "macro_AUPR",
]:

    print()
    print(
        f"===== {metric} ====="
    )

    for method in [
        "Molecular",
        "Raw_Final",
        "Drug_Calibrated",
        "Adaptive",
    ]:

        vals = [
            r[f"{method}_{metric}"]
            for r in seed_results
        ]

        print(
            f"{method:20s}: "
            f"{smry(vals)}"
        )


print()
print(
    "===== ADAPTIVE - DRUG CALIBRATED ====="
)

for metric in [
    "pooled_AUROC",
    "pooled_AUPR",
]:

    vals = [
        r[f"Adaptive_{metric}"]
        -
        r[f"Drug_Calibrated_{metric}"]
        for r in seed_results
    ]

    print(
        f"{metric:14s}: "
        f"{smry(vals)} | "
        f"positive "
        f"{sum(v > 0 for v in vals)}/25"
    )


print()
print(
    "===== ADAPTIVE - MOLECULAR ====="
)

for metric in [
    "pooled_AUROC",
    "pooled_AUPR",
]:

    vals = [
        r[f"Adaptive_{metric}"]
        -
        r[f"Molecular_{metric}"]
        for r in seed_results
    ]

    print(
        f"{metric:14s}: "
        f"{smry(vals)} | "
        f"positive "
        f"{sum(v > 0 for v in vals)}/25"
    )


gate_sd = [
    r["gate_sd"]
    for r in seed_results
]

print()
print(
    "Mean test gate SD:",
    f"{mean(gate_sd):.4f}"
)

print()
print("Saved:")
print(OUT_SEED)
