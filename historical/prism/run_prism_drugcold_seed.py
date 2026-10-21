from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

SEED = args.seed

ARTICLE = Path.home() / "ECARF_Article2_20260911"
OLD = Path.home() / "MDA_CellPainting_Multiview_20260910"
ROOT = ARTICLE / "17_qrc_prism_mda_formal_20260920"

sys.path.insert(0, str(OLD / "02_src"))

from model_fourview_staged import Stage11Molecular

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LR = 1e-4
TEMPERATURE = 0.07
EPOCHS = 500

COHORT = OLD / "01_data/fourview_complete_1195"

ASSOC_FILE = COHORT / "associations_3443.tsv"

MIRNA_FILE = (
    COHORT /
    "mirna_rnafm_cls640.tsv"
)

DRUG_STRUCT_FILE = (
    OLD /
    "01_data/strict_drug_structure_chemberta384.tsv"
)

PRISM_FILE = (
    ROOT /
    "01_data/prism65_external_identityclean_pca39_by_drug.tsv"
)

SPLIT_DIR = (
    OLD /
    f"03_splits/drug_cold_seed{SEED}"
)

RUN_DIR = (
    ROOT /
    f"05_runs/drug_cold_seed{SEED}"
)

RESULT_DIR = (
    ROOT /
    f"06_results/drug_cold_seed{SEED}"
)

RUN_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def read_tsv(path):
    return pd.read_csv(path, sep="\t")


def find_id_column(df, kind):

    if kind == "drug":
        candidates = [
            "drug",
            "Drug",
            "drug_name",
            "compound",
            "compound_name",
            "perturbation",
            "id",
        ]

    else:
        candidates = [
            "miRNA",
            "mirna",
            "miRNA_name",
            "mirna_name",
            "microRNA",
            "id",
        ]

    for c in candidates:
        if c in df.columns:
            return c

    non_numeric = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(non_numeric) == 1:
        return non_numeric[0]

    raise RuntimeError(
        f"Cannot identify ID column: {list(df.columns)}"
    )


def aligned_numeric_matrix(
    df,
    id_col,
    entity_order,
    expected_dim,
):

    work = df.copy()

    work[id_col] = (
        work[id_col]
        .astype(str)
    )

    if work[id_col].duplicated().any():
        raise RuntimeError(
            f"Duplicate IDs in {id_col}"
        )

    work = (
        work
        .set_index(id_col)
    )

    missing = [
        x for x in entity_order
        if x not in work.index
    ]

    if missing:
        raise RuntimeError(
            f"{id_col}: missing entities: "
            f"{missing[:10]}"
        )

    work = work.loc[entity_order]

    numeric = (
        work
        .select_dtypes(include=[np.number])
    )

    if numeric.shape[1] != expected_dim:
        raise RuntimeError(
            f"{id_col}: expected "
            f"{expected_dim} dims, "
            f"found {numeric.shape[1]}"
        )

    arr = numeric.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"{id_col}: non-finite input"
        )

    return arr


def load_entities(path):

    df = read_tsv(path)

    if df.shape[1] != 1:
        raise RuntimeError(
            f"Unexpected split format: {path}"
        )

    return (
        df.iloc[:, 0]
        .astype(str)
        .tolist()
    )


def symmetric_multi_positive_nce(
    left,
    right,
    positive_mask,
    temperature=0.07,
):

    logits = (
        left @ right.T
    ) / temperature

    mask = positive_mask.bool()

    # left -> right
    positive_count = (
        mask
        .sum(dim=1)
    )

    valid = positive_count > 0

    if not torch.any(valid):
        raise RuntimeError(
            "No valid left anchors."
        )

    log_prob = (
        logits
        - torch.logsumexp(
            logits,
            dim=1,
            keepdim=True,
        )
    )

    positive_log_prob = torch.where(
        mask,
        log_prob,
        torch.zeros_like(log_prob),
    )

    loss_lr = -(
        positive_log_prob.sum(dim=1)
        / positive_count.clamp_min(1)
    )[valid].mean()

    # right -> left
    mask_t = mask.T

    positive_count_t = (
        mask_t
        .sum(dim=1)
    )

    valid_t = positive_count_t > 0

    logits_t = logits.T

    log_prob_t = (
        logits_t
        - torch.logsumexp(
            logits_t,
            dim=1,
            keepdim=True,
        )
    )

    positive_log_prob_t = torch.where(
        mask_t,
        log_prob_t,
        torch.zeros_like(log_prob_t),
    )

    loss_rl = -(
        positive_log_prob_t.sum(dim=1)
        / positive_count_t.clamp_min(1)
    )[valid_t].mean()

    loss = 0.5 * (
        loss_lr + loss_rl
    )

    return loss, loss_lr, loss_rl


def pooled_metrics(y, s):

    y = np.asarray(y).reshape(-1)
    s = np.asarray(s).reshape(-1)

    return {
        "AUROC":
            float(roc_auc_score(y, s)),
        "AUPR":
            float(average_precision_score(y, s)),
        "n":
            int(len(y)),
        "positive":
            int(y.sum()),
        "prevalence":
            float(y.mean()),
    }


def macro_metrics(Y, S):

    aucs = []
    auprs = []

    for i in range(Y.shape[0]):

        y = Y[i]
        s = S[i]

        if len(np.unique(y)) < 2:
            continue

        aucs.append(
            roc_auc_score(y, s)
        )

        auprs.append(
            average_precision_score(y, s)
        )

    return {
        "macro_AUROC":
            float(np.mean(aucs)),
        "macro_AUPR":
            float(np.mean(auprs)),
        "valid_queries":
            len(aucs),
    }


# ============================================================
# LOAD DATA
# ============================================================

assoc = read_tsv(
    ASSOC_FILE
)

drug_struct_df = read_tsv(
    DRUG_STRUCT_FILE
)

mirna_df = read_tsv(
    MIRNA_FILE
)

prism_df = read_tsv(
    PRISM_FILE
)


drug_id = find_id_column(
    drug_struct_df,
    "drug",
)

mirna_id = find_id_column(
    mirna_df,
    "mirna",
)

prism_id = find_id_column(
    prism_df,
    "drug",
)


drug_order = (
    drug_struct_df[drug_id]
    .astype(str)
    .tolist()
)

mirna_order = (
    mirna_df[mirna_id]
    .astype(str)
    .tolist()
)


if len(drug_order) != 65:
    raise RuntimeError(
        f"Expected 65 drugs, got {len(drug_order)}"
    )

if len(mirna_order) != 1195:
    raise RuntimeError(
        f"Expected 1195 miRNAs, got {len(mirna_order)}"
    )


Ds_np = aligned_numeric_matrix(
    drug_struct_df,
    drug_id,
    drug_order,
    384,
)

Ms_np = aligned_numeric_matrix(
    mirna_df,
    mirna_id,
    mirna_order,
    640,
)

P_np = aligned_numeric_matrix(
    prism_df,
    prism_id,
    drug_order,
    39,
)


print("========== INPUT ==========")

print(
    "Ds:",
    Ds_np.shape
)

print(
    "PRISM:",
    P_np.shape
)

print(
    "Raw drug input:",
    (65, 384 + 39)
)

print(
    "Ms:",
    Ms_np.shape
)


# ============================================================
# ASSOCIATION MATRIX
# ============================================================

drug_to_idx = {
    x: i
    for i, x in enumerate(drug_order)
}

mirna_to_idx = {
    x: i
    for i, x in enumerate(mirna_order)
}

A = np.zeros(
    (
        len(drug_order),
        len(mirna_order),
    ),
    dtype=np.float32,
)

unknown_assoc = []

for _, r in assoc.iterrows():

    d = str(r["drug"])
    m = str(r["miRNA"])

    if (
        d not in drug_to_idx
        or m not in mirna_to_idx
    ):
        unknown_assoc.append(
            (d, m)
        )
        continue

    A[
        drug_to_idx[d],
        mirna_to_idx[m],
    ] = 1.0


if unknown_assoc:
    raise RuntimeError(
        f"Associations outside cohort: "
        f"{unknown_assoc[:5]}"
    )


print(
    "association positives:",
    int(A.sum())
)


# ============================================================
# HISTORICAL DRUG-COLD SPLIT
# ============================================================

train_drugs = load_entities(
    SPLIT_DIR /
    "train_entities.tsv"
)

val_drugs = load_entities(
    SPLIT_DIR /
    "val_entities.tsv"
)

test_drugs = load_entities(
    SPLIT_DIR /
    "test_entities.tsv"
)


train_idx = np.array(
    [
        drug_to_idx[x]
        for x in train_drugs
    ],
    dtype=int,
)

val_idx = np.array(
    [
        drug_to_idx[x]
        for x in val_drugs
    ],
    dtype=int,
)

test_idx = np.array(
    [
        drug_to_idx[x]
        for x in test_drugs
    ],
    dtype=int,
)


if (
    len(train_idx),
    len(val_idx),
    len(test_idx),
) != (
    53,
    6,
    6,
):
    raise RuntimeError(
        "Unexpected drug-cold split sizes."
    )


if (
    set(train_idx)
    & set(val_idx)
    or set(train_idx)
    & set(test_idx)
    or set(val_idx)
    & set(test_idx)
):
    raise RuntimeError(
        "Drug split overlap."
    )


print("\n========== SPLIT ==========")

print(
    "train/val/test drugs:",
    len(train_idx),
    len(val_idx),
    len(test_idx),
)

print(
    "train positives:",
    int(A[train_idx].sum())
)

print(
    "val positives:",
    int(A[val_idx].sum())
)

print(
    "test positives:",
    int(A[test_idx].sum())
)


# ============================================================
# TORCH TENSORS
# ============================================================

Ds = torch.tensor(
    Ds_np,
    dtype=torch.float32,
    device=DEVICE,
)

P = torch.tensor(
    P_np,
    dtype=torch.float32,
    device=DEVICE,
)

Ms = torch.tensor(
    Ms_np,
    dtype=torch.float32,
    device=DEVICE,
)

RawDrug = torch.cat(
    [Ds, P],
    dim=1,
)

A_train = torch.tensor(
    A[train_idx],
    dtype=torch.bool,
    device=DEVICE,
)


# ============================================================
# TRAIN FUNCTION
# Exact historical Stage1.1 protocol:
# Adam, lr=1e-4, 500 full-batch epochs
# ============================================================

def train_model(
    name,
    drug_input,
    input_dim,
):

    # Independent but paired seed
    set_seed(SEED)

    model = Stage11Molecular(
        drug_structure_dim=input_dim,
        mirna_sequence_dim=640,
        hidden_dim=256,
        latent_dim=128,
        dropout=0.1,
        temperature=TEMPERATURE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    model.train()

    final_loss = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        optimizer.zero_grad()

        out = model(
            drug_input[train_idx],
            Ms,
        )

        z_d = out["z_ds"]
        z_m = out["z_ms"]

        (
            loss,
            loss_dm,
            loss_md,
        ) = symmetric_multi_positive_nce(
            z_d,
            z_m,
            A_train,
            TEMPERATURE,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{name}: non-finite loss "
                f"epoch={epoch}"
            )

        loss.backward()
        optimizer.step()

        final_loss = float(
            loss.detach().cpu()
        )

        if (
            epoch == 1
            or epoch % 50 == 0
        ):

            print(
                f"{name} "
                f"epoch={epoch:04d} "
                f"sym={loss.item():.6f} "
                f"D->M={loss_dm.item():.6f} "
                f"M->D={loss_md.item():.6f}"
            )

    torch.save(
        model.state_dict(),
        RUN_DIR /
        f"{name.lower()}_seed{SEED}.pt",
    )

    model.eval()

    with torch.no_grad():

        out_all = model(
            drug_input,
            Ms,
        )

        score = (
            out_all["z_ds"]
            @ out_all["z_ms"].T
        )

    score = (
        score
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    return model, score, final_loss


# ============================================================
# TRAIN BASE
# ============================================================

print("\n" + "=" * 76)
print("TRAIN BASE: ChemBERTa384 + RNA-FM640")
print("=" * 76)

base_model, S_base, base_loss = train_model(
    "BASE",
    Ds,
    384,
)


# ============================================================
# TRAIN RAW
# ============================================================

print("\n" + "=" * 76)
print("TRAIN RAW: [ChemBERTa384 ; PRISM39] + RNA-FM640")
print("=" * 76)

raw_model, S_raw, raw_loss = train_model(
    "RAW",
    RawDrug,
    423,
)


# ============================================================
# QRC / SQRC
# Query = drug
# Candidate = miRNA
# ============================================================

Delta = (
    S_raw - S_base
)


# ----------------------------
# validation-only SQRC tau_b^2
# ----------------------------

delta_val = Delta[val_idx]

b_val = (
    delta_val
    .mean(axis=1)
)

s2_val = (
    delta_val
    .var(axis=1, ddof=1)
)

n_candidate = (
    delta_val.shape[1]
)

se2_val = (
    s2_val / n_candidate
)

tau_b2 = max(
    float(
        np.var(
            b_val,
            ddof=1,
        )
        - np.mean(se2_val)
    ),
    0.0,
)


# ----------------------------
# test QRC / SQRC
# ----------------------------

base_test = (
    S_base[test_idx]
)

raw_test = (
    S_raw[test_idx]
)

delta_test = (
    raw_test - base_test
)

b_test = (
    delta_test
    .mean(axis=1)
)

qrc_test = (
    raw_test
    - b_test[:, None]
)

s2_test = (
    delta_test
    .var(axis=1, ddof=1)
)

se2_test = (
    s2_test / n_candidate
)

if tau_b2 == 0.0:

    w_test = np.zeros_like(
        se2_test
    )

else:

    w_test = (
        tau_b2
        / (
            tau_b2
            + se2_test
        )
    )

sqrc_test = (
    raw_test
    - (
        w_test * b_test
    )[:, None]
)


# ============================================================
# LABELS
# ============================================================

Y_test = (
    A[test_idx]
    .astype(int)
)

Y_val = (
    A[val_idx]
    .astype(int)
)


# ============================================================
# METRICS
# ============================================================

metrics = {}

for name, score in [
    ("Base", base_test),
    ("Raw", raw_test),
    ("QRC", qrc_test),
    ("SQRC", sqrc_test),
]:

    pooled = pooled_metrics(
        Y_test,
        score,
    )

    macro = macro_metrics(
        Y_test,
        score,
    )

    metrics[name] = {
        **pooled,
        **macro,
    }


# ============================================================
# QRC EXACT INVARIANCE QC
# ============================================================

qrc_identity_error = float(
    np.max(
        np.abs(
            (
                qrc_test
                - raw_test
            )
            + b_test[:, None]
        )
    )
)

sqrc_identity_error = float(
    np.max(
        np.abs(
            (
                sqrc_test
                - raw_test
            )
            + (
                w_test
                * b_test
            )[:, None]
        )
    )
)

macro_auc_diff = abs(
    metrics["Raw"]["macro_AUROC"]
    - metrics["QRC"]["macro_AUROC"]
)

macro_aupr_diff = abs(
    metrics["Raw"]["macro_AUPR"]
    - metrics["QRC"]["macro_AUPR"]
)


# ============================================================
# SHIFT DIAGNOSTICS
# ============================================================

var_delta = float(
    np.var(
        delta_test.reshape(-1),
        ddof=0,
    )
)

var_bias = float(
    np.var(
        np.repeat(
            b_test,
            n_candidate,
        ),
        ddof=0,
    )
)

if var_delta > 0:
    query_shift_variance_ratio = (
        var_bias / var_delta
    )
else:
    query_shift_variance_ratio = 0.0


raw_base_mad = float(
    np.mean(
        np.abs(
            raw_test
            - base_test
        )
    )
)

qrc_base_mad = float(
    np.mean(
        np.abs(
            qrc_test
            - base_test
        )
    )
)


# ============================================================
# SAVE ALL 6 x 1195 TEST PAIRS
# ============================================================

rows = []

for local_i, global_i in enumerate(
    test_idx
):

    drug = drug_order[
        global_i
    ]

    for j, mirna in enumerate(
        mirna_order
    ):

        rows.append({
            "seed":
                SEED,
            "drug":
                drug,
            "miRNA":
                mirna,
            "label":
                int(Y_test[local_i, j]),
            "base_score":
                float(base_test[local_i, j]),
            "raw_score":
                float(raw_test[local_i, j]),
            "delta":
                float(delta_test[local_i, j]),
            "query_bias":
                float(b_test[local_i]),
            "qrc_score":
                float(qrc_test[local_i, j]),
            "sqrc_weight":
                float(w_test[local_i]),
            "sqrc_score":
                float(sqrc_test[local_i, j]),
        })


pair_df = pd.DataFrame(
    rows
)

PAIR_FILE = (
    RESULT_DIR /
    f"all_pair_scores_prism_drug_cold_seed{SEED}.tsv"
)

pair_df.to_csv(
    PAIR_FILE,
    sep="\t",
    index=False,
)


# ============================================================
# SAVE QUERY DIAGNOSTICS
# ============================================================

query_rows = []

for local_i, global_i in enumerate(
    test_idx
):

    query_rows.append({
        "seed":
            SEED,
        "drug":
            drug_order[global_i],
        "positive_pairs":
            int(Y_test[local_i].sum()),
        "b_i":
            float(b_test[local_i]),
        "delta_sd":
            float(
                np.std(
                    delta_test[local_i],
                    ddof=0,
                )
            ),
        "SE2":
            float(se2_test[local_i]),
        "sqrc_weight":
            float(w_test[local_i]),
    })


query_df = pd.DataFrame(
    query_rows
)

query_df.to_csv(
    RESULT_DIR /
    f"query_diagnostics_prism_drug_cold_seed{SEED}.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "seed":
        SEED,

    "setting":
        "drug_cold",

    "device":
        str(DEVICE),

    "protocol": {
        "train_drugs":
            len(train_idx),
        "val_drugs":
            len(val_idx),
        "test_drugs":
            len(test_idx),
        "candidate_mirnas":
            len(mirna_order),
        "epochs":
            EPOCHS,
        "optimizer":
            "Adam",
        "lr":
            LR,
        "weight_decay":
            0.0,
        "temperature":
            TEMPERATURE,
        "base_drug_dim":
            384,
        "prism_dim":
            39,
        "raw_drug_dim":
            423,
    },

    "training": {
        "base_final_loss":
            base_loss,
        "raw_final_loss":
            raw_loss,
    },

    "metrics":
        metrics,

    "qrc": {
        "tau_b2_validation_only":
            tau_b2,

        "test_bias_mean":
            float(
                np.mean(b_test)
            ),

        "test_bias_sd":
            float(
                np.std(
                    b_test,
                    ddof=1,
                )
            ),

        "query_shift_variance_ratio":
            query_shift_variance_ratio,

        "raw_vs_base_mean_abs_change":
            raw_base_mad,

        "qrc_vs_base_mean_abs_change":
            qrc_base_mad,

        "qrc_identity_max_error":
            qrc_identity_error,

        "sqrc_identity_max_error":
            sqrc_identity_error,

        "raw_qrc_macro_auc_abs_diff":
            macro_auc_diff,

        "raw_qrc_macro_aupr_abs_diff":
            macro_aupr_diff,
    },

    "output": {
        "pair_rows":
            len(pair_df),
        "pair_file":
            str(PAIR_FILE),
    },
}


SUMMARY_FILE = (
    RESULT_DIR /
    f"prism_drug_cold_seed{SEED}_summary.json"
)

with open(
    SUMMARY_FILE,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 76)
print("PRISM QRC DRUG-COLD RESULT")
print("=" * 76)

for name in [
    "Base",
    "Raw",
    "QRC",
    "SQRC",
]:

    x = metrics[name]

    print(
        f"{name:5s} | "
        f"AUROC={x['AUROC']:.6f} "
        f"AUPR={x['AUPR']:.6f} | "
        f"macroAUC={x['macro_AUROC']:.6f} "
        f"macroAUPR={x['macro_AUPR']:.6f}"
    )


print("\n========== QRC MECHANISM ==========")

print(
    "tau_b2(validation-only):",
    tau_b2,
)

print(
    "test b_i:",
    b_test,
)

print(
    "SQRC weights:",
    w_test,
)

print(
    "query-shift variance ratio:",
    query_shift_variance_ratio,
)

print(
    "Raw-vs-Base mean |change|:",
    raw_base_mad,
)

print(
    "QRC-vs-Base mean |change|:",
    qrc_base_mad,
)


print("\n========== INVARIANCE QC ==========")

print(
    "QRC identity max error:",
    qrc_identity_error,
)

print(
    "SQRC identity max error:",
    sqrc_identity_error,
)

print(
    "Raw/QRC macro AUROC diff:",
    macro_auc_diff,
)

print(
    "Raw/QRC macro AUPR diff:",
    macro_aupr_diff,
)


if len(pair_df) != 6 * 1195:
    raise RuntimeError(
        f"Expected 7170 test pairs, "
        f"got {len(pair_df)}"
    )


if macro_auc_diff > 1e-12:
    raise RuntimeError(
        "Raw/QRC macro AUROC invariance failed."
    )


if macro_aupr_diff > 1e-12:
    raise RuntimeError(
        "Raw/QRC macro AUPR invariance failed."
    )


print("\nPAIR ROWS:", len(pair_df))
print("SUMMARY:", SUMMARY_FILE)
print("PAIR SCORES:", PAIR_FILE)

print(
    "\nPRISM_DRUG_COLD_SEED_SMOKE_COMPLETE"
)
