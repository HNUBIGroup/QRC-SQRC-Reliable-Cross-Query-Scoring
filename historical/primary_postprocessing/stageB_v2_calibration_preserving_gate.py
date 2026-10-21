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


ROOT = os.path.expanduser(
    "~/MDA_CellPainting_Multiview_20260910"
)

OUT = os.path.expanduser(
    "~/ECARF_Article2_20260911/02_results"
)

os.makedirs(OUT, exist_ok=True)

OUT_FILE = os.path.join(
    OUT,
    "stageB_v2_drugcold_25seed_metrics.tsv"
)

DEVICE = "cpu"

TRAIN_STEPS = 600
BATCH_SIZE = 2048
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

        rr = (i + 1 + j) / 2.0

        for k in range(i, j):
            ranks[order[k]] = rr

        i = j

    return ranks


def auroc(y, s):

    np = sum(y)
    nn = len(y) - np

    if np == 0 or nn == 0:
        return None

    r = rankdata(s)

    sp = sum(
        r[i]
        for i, v in enumerate(y)
        if v == 1
    )

    return (
        sp - np * (np + 1) / 2
    ) / (np * nn)


def aupr(y, s):

    np = sum(y)

    if np == 0:
        return None

    order = sorted(
        range(len(s)),
        key=lambda i: s[i],
        reverse=True
    )

    hit = 0
    total = 0.0

    for k, i in enumerate(order, 1):

        if y[i] == 1:
            hit += 1
            total += hit / k

    return total / np


# ============================================================
# BUILD ONE QUERY SCOPE
# ============================================================

def prepare_scope(all_rows, drugs, positive_split):

    rows = [
        r for r in all_rows
        if (
            r["drug"] in drugs
            and r["split"]
            in (positive_split, "unlabeled")
        )
    ]

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
        for m, u in zip(
            molecular,
            unified
        )
    ]

    by_drug = defaultdict(list)

    for i, r in enumerate(rows):
        by_drug[r["drug"]].append(i)

    residual = [0.0] * len(rows)
    rank_gap = [0.0] * len(rows)

    # indices for within-query training
    positives_by_drug = {}
    negatives_by_drug = {}

    for drug, ids in by_drug.items():

        mu = mean(
            delta[i]
            for i in ids
        )

        for i in ids:
            residual[i] = (
                delta[i] - mu
            )

        mol_vals = [
            molecular[i]
            for i in ids
        ]

        uni_vals = [
            unified[i]
            for i in ids
        ]

        mr = rankdata(mol_vals)
        ur = rankdata(uni_vals)

        denom = max(
            1,
            len(ids) - 1
        )

        for local_i, global_i in enumerate(ids):

            rank_gap[global_i] = (
                abs(
                    mr[local_i]
                    -
                    ur[local_i]
                )
                / denom
            )

        pos = [
            i for i in ids
            if rows[i]["split"]
            == positive_split
        ]

        neg = [
            i for i in ids
            if rows[i]["split"]
            == "unlabeled"
        ]

        if pos and neg:
            positives_by_drug[drug] = pos
            negatives_by_drug[drug] = neg

    y = [
        1 if r["split"] == positive_split
        else 0
        for r in rows
    ]

    return {
        "rows": rows,
        "y": y,
        "molecular": molecular,
        "residual": residual,
        "abs_residual": [
            abs(v)
            for v in residual
        ],
        "rank_gap": rank_gap,
        "by_drug": by_drug,
        "positives_by_drug":
            positives_by_drug,
        "negatives_by_drug":
            negatives_by_drug,
    }


# ============================================================
# FEATURES
# ============================================================

def fit_stats(scope):

    a = scope["abs_residual"]
    g = scope["rank_gap"]

    ma = mean(a)
    mg = mean(g)

    sa = math.sqrt(
        mean((x-ma)**2 for x in a)
    )

    sg = math.sqrt(
        mean((x-mg)**2 for x in g)
    )

    return (
        ma,
        max(sa, 1e-8),
        mg,
        max(sg, 1e-8)
    )


def features(scope, stats):

    ma, sa, mg, sg = stats

    x = []

    for a, g in zip(
        scope["abs_residual"],
        scope["rank_gap"]
    ):
        x.append([
            (a-ma)/sa,
            (g-mg)/sg,
        ])

    return torch.tensor(
        x,
        dtype=torch.float32,
        device=DEVICE
    )


# ============================================================
# CALIBRATION-PRESERVING GATED SCORE
# ============================================================

def score_scope(scope, x, params):

    w0, w1, w2 = params

    logits = (
        w0
        + w1*x[:, 0]
        + w2*x[:, 1]
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

    correction = gate * res

    # --------------------------------------------------------
    # KEY V2 STEP:
    # reliability weighting must NOT re-introduce
    # a query-specific score offset.
    # --------------------------------------------------------

    calibrated_correction = correction.clone()

    for drug, ids in scope["by_drug"].items():

        idx = torch.tensor(
            ids,
            dtype=torch.long,
            device=DEVICE
        )

        mu = correction[idx].mean()

        calibrated_correction[idx] = (
            correction[idx] - mu
        )

    final = (
        mol
        +
        calibrated_correction
    )

    return (
        final,
        gate,
        calibrated_correction
    )


# ============================================================
# MACRO METRICS
# ============================================================

def macro(scope, scores):

    aucs = []
    aps = []

    for drug, ids in scope["by_drug"].items():

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
# FILES
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


def seedof(path):

    m = re.search(
        r"seed(\d+)",
        path
    )

    return (
        int(m.group(1))
        if m else None
    )


files = [
    f for f in files
    if seedof(f) is not None
]

files.sort(
    key=seedof
)

print(
    "Drug-cold score files:",
    len(files)
)


# ============================================================
# MAIN
# ============================================================

results = []

for path in files:

    seed = seedof(path)

    random.seed(
        2000 + seed
    )

    torch.manual_seed(
        2000 + seed
    )

    all_rows = []

    with open(
        path,
        encoding="utf-8-sig"
    ) as f:

        all_rows = list(
            csv.DictReader(
                f,
                delimiter="\t"
            )
        )

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

    stats = fit_stats(
        train
    )

    xt = features(
        train,
        stats
    )

    xv = features(
        val,
        stats
    )

    xte = features(
        test,
        stats
    )

    # --------------------------------------------------------
    # old lambda for initialization / Stage A baseline
    # --------------------------------------------------------

    lams = []

    for r in all_rows:

        m = float(
            r["molecular_score"]
        )

        u = float(
            r["unified_score"]
        )

        f = float(
            r["final_score"]
        )

        den = u - m

        if abs(den) > 1e-10:

            lam = (
                (f-m) / den
            )

            if (
                -1e-5
                <= lam
                <= 1.00001
            ):
                lams.append(lam)

    base_lambda = median(lams)

    init_lambda = min(
        0.99,
        max(
            0.01,
            base_lambda
        )
    )

    init_logit = math.log(
        init_lambda
        /
        (1-init_lambda)
    )

    params = torch.nn.Parameter(
        torch.tensor(
            [
                init_logit,
                0.0,
                0.0
            ],
            dtype=torch.float32,
            device=DEVICE
        )
    )

    opt = torch.optim.Adam(
        [params],
        lr=LR,
        weight_decay=
            WEIGHT_DECAY
    )

    trainable_drugs = sorted(
        set(
            train[
                "positives_by_drug"
            ]
        )
        &
        set(
            train[
                "negatives_by_drug"
            ]
        )
    )

    best_state = None
    best_key = (
        -1.0,
        -1.0
    )

    best_step = None

    # ========================================================
    # WITHIN-QUERY PAIRWISE TRAINING
    # ========================================================

    for step in range(
        1,
        TRAIN_STEPS + 1
    ):

        score, gate, corr = (
            score_scope(
                train,
                xt,
                params
            )
        )

        pos_idx = []
        neg_idx = []

        for _ in range(
            BATCH_SIZE
        ):

            d = random.choice(
                trainable_drugs
            )

            p = random.choice(
                train[
                    "positives_by_drug"
                ][d]
            )

            n = random.choice(
                train[
                    "negatives_by_drug"
                ][d]
            )

            pos_idx.append(p)
            neg_idx.append(n)

        p = torch.tensor(
            pos_idx,
            dtype=torch.long,
            device=DEVICE
        )

        n = torch.tensor(
            neg_idx,
            dtype=torch.long,
            device=DEVICE
        )

        loss_rank = F.softplus(
            -(
                score[p]
                -
                score[n]
            )
        ).mean()

        gate_prior = (
            gate.mean()
            -
            float(base_lambda)
        ) ** 2

        loss = (
            loss_rank
            +
            0.01 * gate_prior
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (
            step % EVAL_INTERVAL
            == 0
        ):

            with torch.no_grad():

                vs, _, _ = (
                    score_scope(
                        val,
                        xv,
                        params
                    )
                )

                vs = (
                    vs
                    .cpu()
                    .tolist()
                )

                # For gate training the target is
                # query-conditional ranking.
                ma, mp = macro(
                    val,
                    vs
                )

                key = (
                    mp,
                    ma
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
            f"No valid best state seed {seed}"
        )

    params.data.copy_(
        best_state.to(DEVICE)
    )

    # ========================================================
    # TEST
    # ========================================================

    with torch.no_grad():

        adaptive, gates, correction = (
            score_scope(
                test,
                xte,
                params
            )
        )

        adaptive = (
            adaptive
            .cpu()
            .tolist()
        )

        gates = (
            gates
            .cpu()
            .tolist()
        )

    molecular = (
        test["molecular"]
    )

    drug_cal = [
        m + base_lambda*r
        for m, r in zip(
            test["molecular"],
            test["residual"]
        )
    ]

    # recover raw final
    pair_lookup = {
        (
            r["drug"],
            r["miRNA"]
        ): r
        for r in all_rows
    }

    raw_final = [
        float(
            pair_lookup[
                (
                    r["drug"],
                    r["miRNA"]
                )
            ]["final_score"]
        )
        for r in test["rows"]
    ]

    methods = {
        "Molecular":
            molecular,
        "Raw_Final":
            raw_final,
        "Drug_Calibrated":
            drug_cal,
        "Adaptive_V2":
            adaptive,
    }

    y = test["y"]

    rec = {
        "seed": seed,
        "base_lambda":
            base_lambda,
        "best_step":
            best_step,
        "w0":
            float(params[0]),
        "w_abs_residual":
            float(params[1]),
        "w_rank_gap":
            float(params[2]),
        "gate_mean":
            mean(gates),
        "gate_sd":
            stdev(gates),
    }

    for method, s in (
        methods.items()
    ):

        rec[
            f"{method}_pooled_AUROC"
        ] = auroc(y, s)

        rec[
            f"{method}_pooled_AUPR"
        ] = aupr(y, s)

        ma, mp = macro(
            test,
            s
        )

        rec[
            f"{method}_macro_AUROC"
        ] = ma

        rec[
            f"{method}_macro_AUPR"
        ] = mp

    results.append(rec)

    print(
        f"seed {seed:02d} | "
        f"lambda={base_lambda:.3f} | "
        f"gate={mean(gates):.3f}"
        f"±{stdev(gates):.3f} | "
        f"AUC "
        f"{rec['Molecular_pooled_AUROC']:.3f}"
        f" -> "
        f"{rec['Drug_Calibrated_pooled_AUROC']:.3f}"
        f" -> "
        f"{rec['Adaptive_V2_pooled_AUROC']:.3f}"
    )


# ============================================================
# SAVE
# ============================================================

with open(
    OUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            results[0].keys()
        ),
        delimiter="\t"
    )

    w.writeheader()
    w.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

def smry(v):

    return (
        f"{mean(v):.4f}"
        f" ± {stdev(v):.4f}"
    )


print()
print("=" * 78)
print(
    "STAGE B V0.2 — CALIBRATION-PRESERVING ADAPTIVE RESIDUAL"
)
print("=" * 78)

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
        "Adaptive_V2",
    ]:

        vals = [
            r[
                f"{method}_{metric}"
            ]
            for r in results
        ]

        print(
            f"{method:20s}: "
            f"{smry(vals)}"
        )


print()
print(
    "===== Adaptive V2 - Drug Calibrated ====="
)

for metric in [
    "pooled_AUROC",
    "pooled_AUPR",
    "macro_AUROC",
    "macro_AUPR",
]:

    vals = [
        r[
            f"Adaptive_V2_{metric}"
        ]
        -
        r[
            f"Drug_Calibrated_{metric}"
        ]
        for r in results
    ]

    print(
        f"{metric:14s}: "
        f"{smry(vals)} | "
        f"positive "
        f"{sum(v > 0 for v in vals)}/25"
    )


print()
print(
    "Mean gate SD:",
    f"{mean(r['gate_sd'] for r in results):.4f}"
)

print()
print("Saved:")
print(OUT_FILE)
