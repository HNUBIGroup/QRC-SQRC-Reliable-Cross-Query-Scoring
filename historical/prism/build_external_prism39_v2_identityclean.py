from pathlib import Path
import json
import pickle
import re

import numpy as np
import pandas as pd

from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


ARTICLE = Path.home() / "ECARF_Article2_20260911"

AUDIT = ARTICLE / "16_qrc_prism_aux_audit_20260920"
OUTROOT = ARTICLE / "17_qrc_prism_mda_formal_20260920"

RAW_MATRIX = (
    AUDIT /
    "02_prism_metadata/primary-screen-replicate-collapsed-logfold-change.csv"
)

META = (
    AUDIT /
    "02_prism_metadata/primary-screen-replicate-collapsed-treatment-info.csv"
)

MAPPING = (
    AUDIT /
    "03_mapping/primary_mda65_mapping.tsv"
)

MDA_UNIVERSE = (
    ARTICLE /
    "09_transcriptomic_auxiliary_audit_20260916/audit/article2_drug_universe.tsv"
)

MDA65 = (
    AUDIT /
    "03_mapping/prism65_drug_by_cellline_STRpass_completecase.tsv"
)

OUT = OUTROOT / "01_data"
QC = OUTROOT / "07_qc"

OUT.mkdir(parents=True, exist_ok=True)
QC.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def norm_name(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = re.sub(r"[\s_\-]+", "", x)
    x = re.sub(r"[^a-z0-9]", "", x)
    return x or None


def first_valid_mol(x):
    if pd.isna(x):
        return None

    for s in str(x).split(","):
        s = s.strip()

        if not s:
            continue

        mol = Chem.MolFromSmiles(s)

        if mol is not None:
            return mol

    return None


def inchikey_from_smiles(x):
    mol = first_valid_mol(x)

    if mol is None:
        return None

    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


# ============================================================
# Frozen MDA PRISM matrix
# ============================================================

mda65 = pd.read_csv(
    MDA65,
    sep="\t",
    index_col=0,
)

mda65.index = mda65.index.astype(str)

if mda65.shape != (65, 467):
    raise RuntimeError(
        f"Expected (65,467), got {mda65.shape}"
    )

if mda65.isna().any().any():
    raise RuntimeError(
        "Frozen MDA65 phenotype matrix contains missing values."
    )

clean_cells = list(map(str, mda65.columns))


# ============================================================
# Downstream MDA identity universe
# ============================================================

mda_universe = pd.read_csv(
    MDA_UNIVERSE,
    sep="\t",
)

mda_ikey = set(
    mda_universe["inchikey"]
    .dropna()
    .astype(str)
    .str.upper()
)

mda_connectivity = {
    x.split("-")[0]
    for x in mda_ikey
}

mda_names = set()

for _, row in mda_universe.iterrows():

    for col in [
        "standard_name",
        "original_name",
    ]:
        n = norm_name(row[col])

        if n:
            mda_names.add(n)

    if pd.notna(row["synonyms"]):

        for s in str(row["synonyms"]).split(";"):

            n = norm_name(s)

            if n:
                mda_names.add(n)


# ============================================================
# PRISM metadata
# ============================================================

meta = pd.read_csv(META)

meta["generated_inchikey"] = (
    meta["smiles"]
    .map(inchikey_from_smiles)
)

meta["connectivity_block"] = (
    meta["generated_inchikey"]
    .fillna("")
    .astype(str)
    .str.split("-")
    .str[0]
)

meta["norm_name"] = (
    meta["name"]
    .map(norm_name)
)


# ============================================================
# Direct mapped columns
# ============================================================

mapping = pd.read_csv(
    MAPPING,
    sep="\t",
)

mapped_broad_ids = set()

for x in mapping["prism_broad_ids"].astype(str):

    for z in x.split(";"):

        if z:
            mapped_broad_ids.add(z)

mapped_meta = meta[
    meta["broad_id"]
    .astype(str)
    .isin(mapped_broad_ids)
]

direct_cols = set(
    mapped_meta["column_name"]
    .astype(str)
)

if len(direct_cols) != 68:
    raise RuntimeError(
        f"Expected 68 direct columns, got {len(direct_cols)}"
    )


# ============================================================
# Identity-overlap detection across ALL PRISM treatments
# ============================================================

meta["exact_ikey_overlap"] = (
    meta["generated_inchikey"]
    .fillna("")
    .astype(str)
    .str.upper()
    .isin(mda_ikey)
)

meta["connectivity_overlap"] = (
    meta["connectivity_block"]
    .isin(mda_connectivity)
)

meta["name_overlap"] = (
    meta["norm_name"]
    .isin(mda_names)
)

meta["direct_mapped"] = (
    meta["column_name"]
    .astype(str)
    .isin(direct_cols)
)

meta["exclude_identity"] = (
    meta[
        [
            "direct_mapped",
            "exact_ikey_overlap",
            "connectivity_overlap",
            "name_overlap",
        ]
    ]
    .any(axis=1)
)

excluded = meta[
    meta["exclude_identity"]
].copy()

excluded_cols = set(
    excluded["column_name"]
    .astype(str)
)

additional = excluded[
    ~excluded["direct_mapped"]
].copy()


print("========== FINAL IDENTITY EXCLUSION ==========")
print("direct mapped treatment columns:", len(direct_cols))
print("additional identity-overlap rows:", len(additional))
print("total excluded treatment columns:", len(excluded_cols))

print("\nADDITIONAL EXCLUSIONS:")

if len(additional):

    print(
        additional[
            [
                "column_name",
                "broad_id",
                "name",
                "generated_inchikey",
                "exact_ikey_overlap",
                "connectivity_overlap",
                "name_overlap",
            ]
        ]
        .sort_values(["name", "broad_id"])
        .to_string(index=False)
    )

additional.to_csv(
    QC / "PRISM39_V2_ADDITIONAL_IDENTITY_EXCLUSIONS.tsv",
    sep="\t",
    index=False,
)

excluded.to_csv(
    QC / "PRISM39_V2_ALL_IDENTITY_EXCLUSIONS.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# Raw PRISM matrix
# ============================================================

raw = pd.read_csv(
    RAW_MATRIX
)

row_col = raw.columns[0]

raw[row_col] = raw[row_col].astype(str)

raw = raw.set_index(row_col)

missing_clean = [
    c for c in clean_cells
    if c not in raw.index
]

if missing_clean:
    raise RuntimeError(
        f"Missing clean cell lines: {missing_clean}"
    )

raw = raw.loc[clean_cells]


# ============================================================
# Final external fitting universe
# ============================================================

external_cols = [
    c for c in raw.columns
    if str(c) not in excluded_cols
]

external = (
    raw[external_cols]
    .T
    .copy()
)

print("\n========== FINAL EXTERNAL PRISM UNIVERSE ==========")
print("treatments:", external.shape[0])
print("cell lines:", external.shape[1])

print(
    "raw missing fraction:",
    float(external.isna().mean().mean())
)


# ============================================================
# External-only median imputation
# ============================================================

cell_medians = (
    external
    .median(axis=0, skipna=True)
)

if cell_medians.isna().any():

    bad = list(
        cell_medians[
            cell_medians.isna()
        ].index
    )

    raise RuntimeError(
        f"Undefined external medians: {bad}"
    )

external_imp = (
    external
    .fillna(cell_medians)
)

if external_imp.isna().any().any():
    raise RuntimeError(
        "Missing remains after external imputation."
    )


# ============================================================
# StandardScaler + PCA39
# ============================================================

scaler = StandardScaler(
    with_mean=True,
    with_std=True,
)

X_ext_scaled = scaler.fit_transform(
    external_imp.to_numpy(
        dtype=np.float64
    )
)

pca = PCA(
    n_components=39,
    svd_solver="full",
)

pca.fit(
    X_ext_scaled
)

print("\n========== FINAL PCA ==========")
print("external samples:", X_ext_scaled.shape[0])
print("input dimensions:", X_ext_scaled.shape[1])
print("components:", pca.n_components_)

print(
    "explained variance ratio sum:",
    float(
        pca.explained_variance_ratio_.sum()
    )
)

print(
    "PC1 explained variance:",
    float(
        pca.explained_variance_ratio_[0]
    )
)


# ============================================================
# Transform downstream 65 MDA drugs
# ============================================================

X_mda = (
    mda65[clean_cells]
    .to_numpy(dtype=np.float64)
)

X_mda_scaled = (
    scaler.transform(X_mda)
)

X_mda_pca = (
    pca.transform(X_mda_scaled)
)

cols39 = [
    f"prism_pca_{i:02d}"
    for i in range(39)
]

repr39 = pd.DataFrame(
    X_mda_pca,
    index=mda65.index,
    columns=cols39,
)

repr39.index.name = "standard_name"


# ============================================================
# Attach EOS drug IDs
# ============================================================

name_to_drug = dict(
    zip(
        mapping["standard_name"].astype(str),
        mapping["drug"].astype(str),
    )
)

missing_names = [
    x
    for x in repr39.index
    if x not in name_to_drug
]

if missing_names:
    raise RuntimeError(
        f"Missing MDA drug IDs: {missing_names}"
    )

out = repr39.copy()

out.insert(
    0,
    "drug",
    [
        name_to_drug[x]
        for x in out.index
    ],
)

out = (
    out
    .reset_index(drop=True)
)


# ============================================================
# Save V2 final representation
# ============================================================

OUT_FILE = (
    OUT /
    "prism65_external_identityclean_pca39_by_drug.tsv"
)

out.to_csv(
    OUT_FILE,
    sep="\t",
    index=False,
)

with open(
    OUT / "prism39_identityclean_scaler.pkl",
    "wb",
) as f:
    pickle.dump(scaler, f)

with open(
    OUT / "prism39_identityclean_pca.pkl",
    "wb",
) as f:
    pickle.dump(pca, f)

cell_medians.to_csv(
    OUT / "prism39_identityclean_external_cell_medians.tsv",
    sep="\t",
    header=["median"],
)


# ============================================================
# Final QC
# ============================================================

numeric = (
    out
    .select_dtypes(include=[np.number])
)

summary = {
    "direct_mapped_exclusions": int(len(direct_cols)),
    "additional_identity_exclusions": int(len(additional)),
    "total_excluded_treatments": int(len(excluded_cols)),
    "external_treatments_final": int(external.shape[0]),
    "clean_cell_lines": int(external.shape[1]),
    "external_missing_fraction_before_imputation":
        float(external.isna().mean().mean()),
    "external_missing_after_imputation":
        int(external_imp.isna().sum().sum()),
    "pca_components": 39,
    "pca_explained_variance_ratio_sum":
        float(pca.explained_variance_ratio_.sum()),
    "pc1_explained_variance":
        float(pca.explained_variance_ratio_[0]),
    "output_drugs": int(out.shape[0]),
    "output_numeric_dimensions": int(numeric.shape[1]),
    "output_all_finite":
        bool(np.isfinite(numeric.to_numpy()).all()),
    "mda_used_to_fit_scaler": False,
    "mda_used_to_fit_pca": False,
    "downstream_labels_used": False,
}

with open(
    QC / "PRISM39_IDENTITYCLEAN_V2_QC.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print("\n========== FINAL OUTPUT ==========")
print("shape:", out.shape)
print("numeric shape:", numeric.shape)

print(
    "all finite:",
    bool(
        np.isfinite(
            numeric.to_numpy()
        ).all()
    )
)

print("\nOUTPUT:")
print(OUT_FILE)
