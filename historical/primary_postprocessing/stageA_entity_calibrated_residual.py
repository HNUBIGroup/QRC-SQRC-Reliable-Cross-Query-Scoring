import os
import csv
import glob
import re
import math
from collections import defaultdict
from statistics import mean, stdev, median

# ============================================================
# PATHS
# ============================================================

SOURCE_ROOT = os.path.expanduser(
    "~/MDA_CellPainting_Multiview_20260910"
)

OUT_ROOT = os.path.expanduser(
    "~/ECARF_Article2_20260911/02_results"
)

os.makedirs(OUT_ROOT, exist_ok=True)

SEED_OUT = os.path.join(
    OUT_ROOT,
    "stageA_drugcold_25seed_metrics.tsv"
)

DRUG_OUT = os.path.join(
    OUT_ROOT,
    "stageA_drugcold_perdrug_metrics.tsv"
)

# ============================================================
# BASIC METRICS
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

    rank_sum = sum(
        ranks[i]
        for i, label in enumerate(y)
        if label == 1
    )

    return (
        rank_sum
        - n_pos * (n_pos + 1) / 2
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

    for rank, idx in enumerate(order, start=1):

        if y[idx] == 1:
            hit += 1
            total += hit / rank

    return total / n_pos


# ============================================================
# FILE DISCOVERY
# ============================================================

files = glob.glob(
    os.path.join(
        SOURCE_ROOT,
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
        "WARNING: expected 25 files, found",
        len(files)
    )


# ============================================================
# RUN
# ============================================================

seed_records = []
drug_records = []

METHODS = [
    "Molecular",
    "Raw_Final",
    "Drug_Calibrated",
    "BiEntity_Calibrated",
]


for path in files:

    seed = seed_of(path)

    all_rows = []

    with open(
        path,
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:
            all_rows.append(row)

    # --------------------------------------------------------
    # Held-out drugs
    # --------------------------------------------------------

    test_drugs = {
        r["drug"]
        for r in all_rows
        if r["split"] == "test_known"
    }

    # Candidate space used in cold-start evaluation:
    # hidden positives + originally unlabeled candidates.
    #
    # train/val known associations are not negatives and
    # therefore are excluded.
    rows = [
        r for r in all_rows
        if (
            r["drug"] in test_drugs
            and r["split"]
            in ("test_known", "unlabeled")
        )
    ]

    # --------------------------------------------------------
    # Recover validation-selected lambda
    #
    # Final = Molecular + lambda * (Unified - Molecular)
    # --------------------------------------------------------

    lambda_candidates = []

    for r in all_rows:

        mol = float(r["molecular_score"])
        uni = float(r["unified_score"])
        fin = float(r["final_score"])

        den = uni - mol

        if abs(den) > 1e-10:

            lam = (fin - mol) / den

            if -1e-5 <= lam <= 1.00001:
                lambda_candidates.append(lam)

    if not lambda_candidates:
        raise RuntimeError(
            f"Cannot recover lambda for seed {seed}"
        )

    lam = median(lambda_candidates)

    # --------------------------------------------------------
    # Raw scores and auxiliary correction
    # --------------------------------------------------------

    molecular = [
        float(r["molecular_score"])
        for r in rows
    ]

    unified = [
        float(r["unified_score"])
        for r in rows
    ]

    raw_final = [
        float(r["final_score"])
        for r in rows
    ]

    delta = [
        u - m
        for m, u in zip(
            molecular,
            unified
        )
    ]

    # --------------------------------------------------------
    # Drug-specific correction bias
    # --------------------------------------------------------

    drug_delta = defaultdict(list)

    for r, dlt in zip(rows, delta):
        drug_delta[r["drug"]].append(dlt)

    drug_mean = {
        d: mean(vals)
        for d, vals in drug_delta.items()
    }

    # --------------------------------------------------------
    # miRNA-specific correction bias
    # --------------------------------------------------------

    mirna_delta = defaultdict(list)

    for r, dlt in zip(rows, delta):
        mirna_delta[r["miRNA"]].append(dlt)

    mirna_mean = {
        m: mean(vals)
        for m, vals in mirna_delta.items()
    }

    grand_mean = mean(delta)

    # --------------------------------------------------------
    # Method 1:
    # drug-calibrated auxiliary residual
    # --------------------------------------------------------

    drug_calibrated = []

    # Method 2:
    # double entity calibration
    bi_calibrated = []

    for r, mol, dlt in zip(
        rows,
        molecular,
        delta
    ):

        d = r["drug"]
        m = r["miRNA"]

        residual_drug = (
            dlt
            - drug_mean[d]
        )

        residual_bi = (
            dlt
            - drug_mean[d]
            - mirna_mean[m]
            + grand_mean
        )

        drug_calibrated.append(
            mol + lam * residual_drug
        )

        bi_calibrated.append(
            mol + lam * residual_bi
        )

    score_map = {
        "Molecular":
            molecular,
        "Raw_Final":
            raw_final,
        "Drug_Calibrated":
            drug_calibrated,
        "BiEntity_Calibrated":
            bi_calibrated,
    }

    y = [
        1 if r["split"] == "test_known"
        else 0
        for r in rows
    ]

    # --------------------------------------------------------
    # Pooled metrics
    # --------------------------------------------------------

    seed_result = {
        "seed": seed,
        "lambda": lam,
        "n_test_drugs": len(test_drugs),
        "n_pairs": len(rows),
    }

    for method in METHODS:

        s = score_map[method]

        seed_result[
            f"{method}_pooled_AUROC"
        ] = auroc(y, s)

        seed_result[
            f"{method}_pooled_AUPR"
        ] = aupr(y, s)

    # --------------------------------------------------------
    # Per-drug metrics
    # --------------------------------------------------------

    per_method_auc = defaultdict(list)
    per_method_ap = defaultdict(list)

    for drug in sorted(test_drugs):

        idx = [
            i
            for i, r in enumerate(rows)
            if r["drug"] == drug
        ]

        yy = [y[i] for i in idx]

        rec = {
            "seed": seed,
            "drug": drug,
            "lambda": lam,
            "n_candidates": len(idx),
            "n_positive": sum(yy),
            "delta_mean":
                drug_mean[drug],
        }

        for method in METHODS:

            ss = [
                score_map[method][i]
                for i in idx
            ]

            a = auroc(yy, ss)
            p = aupr(yy, ss)

            rec[
                f"{method}_AUROC"
            ] = a

            rec[
                f"{method}_AUPR"
            ] = p

            if a is not None:
                per_method_auc[method].append(a)

            if p is not None:
                per_method_ap[method].append(p)

        drug_records.append(rec)

    for method in METHODS:

        seed_result[
            f"{method}_macro_AUROC"
        ] = mean(
            per_method_auc[method]
        )

        seed_result[
            f"{method}_macro_AUPR"
        ] = mean(
            per_method_ap[method]
        )

    seed_records.append(seed_result)


# ============================================================
# SAVE
# ============================================================

with open(
    SEED_OUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            seed_records[0].keys()
        ),
        delimiter="\t"
    )

    writer.writeheader()
    writer.writerows(seed_records)


with open(
    DRUG_OUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            drug_records[0].keys()
        ),
        delimiter="\t"
    )

    writer.writeheader()
    writer.writerows(drug_records)


# ============================================================
# SUMMARY
# ============================================================

def summary(values):

    if len(values) == 1:
        return f"{values[0]:.4f}"

    return (
        f"{mean(values):.4f}"
        f" ± {stdev(values):.4f}"
    )


print()
print(
    "=" * 72
)
print(
    "STAGE A — ENTITY-CALIBRATED RESIDUAL / DRUG-COLD 25 SEEDS"
)
print(
    "=" * 72
)

print(
    "Seeds:",
    len(seed_records)
)

print()

for metric in [
    "pooled_AUROC",
    "pooled_AUPR",
    "macro_AUROC",
    "macro_AUPR",
]:

    print(
        f"===== {metric} ====="
    )

    for method in METHODS:

        vals = [
            float(
                r[f"{method}_{metric}"]
            )
            for r in seed_records
        ]

        print(
            f"{method:22s}: "
            f"{summary(vals)}"
        )

    print()


# ============================================================
# Key deltas
# ============================================================

print(
    "===== KEY DELTAS: POOLED AUROC ====="
)

mol = [
    r["Molecular_pooled_AUROC"]
    for r in seed_records
]

raw = [
    r["Raw_Final_pooled_AUROC"]
    for r in seed_records
]

drugcal = [
    r["Drug_Calibrated_pooled_AUROC"]
    for r in seed_records
]

bical = [
    r["BiEntity_Calibrated_pooled_AUROC"]
    for r in seed_records
]


def diff(a, b):
    return [
        x - y
        for x, y in zip(a, b)
    ]


for label, vals in [
    (
        "Raw Final - Molecular",
        diff(raw, mol)
    ),
    (
        "Drug Cal - Molecular",
        diff(drugcal, mol)
    ),
    (
        "BiEntity Cal - Molecular",
        diff(bical, mol)
    ),
    (
        "Drug Cal - Raw Final",
        diff(drugcal, raw)
    ),
    (
        "BiEntity Cal - Raw Final",
        diff(bical, raw)
    ),
]:

    print(
        f"{label:28s}: "
        f"{summary(vals)} | "
        f"positive "
        f"{sum(v > 0 for v in vals)}/"
        f"{len(vals)}"
    )


print()
print("Saved:")
print(SEED_OUT)
print(DRUG_OUT)
