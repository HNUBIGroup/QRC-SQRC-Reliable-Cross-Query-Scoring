from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
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

MAPPING = AUDIT / "03_mapping/primary_mda65_mapping.tsv"

MDA65 = (
    AUDIT /
    "03_mapping/prism65_drug_by_cellline_STRpass_completecase.tsv"
)

OUT = OUTROOT / "01_data"
QC = OUTROOT / "07_qc"

OUT.mkdir(parents=True, exist_ok=True)
QC.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load frozen 65 x 467 MDA phenotype matrix
# ============================================================

mda = pd.read_csv(
    MDA65,
    sep="\t",
    index_col=0,
)

mda.index = mda.index.astype(str)

print("========== FROZEN MDA PRISM MATRIX ==========")
print("shape:", mda.shape)
print("missing:", int(mda.isna().sum().sum()))

if mda.shape != (65, 467):
    raise RuntimeError(
        f"Expected frozen MDA matrix (65,467), got {mda.shape}"
    )

if mda.isna().any().any():
    raise RuntimeError("MDA65 matrix contains missing values.")

clean_cells = list(map(str, mda.columns))


# ============================================================
# Identify ALL MDA treatment columns
# ============================================================

mapping = pd.read_csv(MAPPING, sep="\t")
meta = pd.read_csv(META)

mda_broad_ids = set()

for x in mapping["prism_broad_ids"].astype(str):
    for z in x.split(";"):
        if z:
            mda_broad_ids.add(z)

mda_meta = meta[
    meta["broad_id"].astype(str).isin(mda_broad_ids)
].copy()

mda_treatment_cols = set(
    mda_meta["column_name"].astype(str)
)

print("\n========== EXCLUDED MDA TREATMENTS ==========")
print("MDA drugs:", len(mapping))
print("MDA Broad IDs:", len(mda_broad_ids))
print("MDA treatment columns:", len(mda_treatment_cols))

if len(mda_treatment_cols) != 68:
    raise RuntimeError(
        f"Expected 68 MDA treatment columns, got {len(mda_treatment_cols)}"
    )


# ============================================================
# Load raw PRISM matrix
# rows = cell lines
# cols = treatments
# ============================================================

raw = pd.read_csv(RAW_MATRIX)

row_col = raw.columns[0]

raw[row_col] = raw[row_col].astype(str)
raw = raw.set_index(row_col)

missing_cells = [
    c for c in clean_cells
    if c not in raw.index
]

if missing_cells:
    raise RuntimeError(
        f"Missing frozen clean cells: {missing_cells[:10]}"
    )

raw = raw.loc[clean_cells]

print("\n========== RAW CLEAN-CELL MATRIX ==========")
print("shape:", raw.shape)


# ============================================================
# External PRISM universe
# Explicitly exclude all MDA treatment columns
# ============================================================

external_cols = [
    c for c in raw.columns
    if str(c) not in mda_treatment_cols
]

external = raw[external_cols].T.copy()
# treatments x cell lines

print("\n========== EXTERNAL PRISM FITTING UNIVERSE ==========")
print("treatments:", external.shape[0])
print("cell lines:", external.shape[1])
print(
    "raw missing fraction:",
    float(external.isna().mean().mean())
)


# ============================================================
# External-only median imputation
# ============================================================

cell_medians = external.median(axis=0, skipna=True)

bad = cell_medians[cell_medians.isna()]

if len(bad):
    raise RuntimeError(
        f"Cell lines with undefined external median: {list(bad.index)}"
    )

external_imp = external.fillna(cell_medians)

print(
    "missing after external-only imputation:",
    int(external_imp.isna().sum().sum())
)


# ============================================================
# External-only StandardScaler + PCA39
# ============================================================

scaler = StandardScaler(
    with_mean=True,
    with_std=True,
)

X_ext_scaled = scaler.fit_transform(
    external_imp.to_numpy(dtype=np.float64)
)

pca = PCA(
    n_components=39,
    svd_solver="full",
)

X_ext_pca = pca.fit_transform(X_ext_scaled)

print("\n========== PCA ==========")
print("external samples:", X_ext_scaled.shape[0])
print("input dims:", X_ext_scaled.shape[1])
print("components:", pca.n_components_)
print(
    "explained variance ratio sum:",
    float(pca.explained_variance_ratio_.sum())
)
print(
    "first 10 EVR:",
    pca.explained_variance_ratio_[:10]
)


# ============================================================
# Transform frozen 65 MDA drugs
# No MDA drug participated in fit
# ============================================================

X_mda = mda[clean_cells].to_numpy(dtype=np.float64)

X_mda_scaled = scaler.transform(X_mda)
X_mda_pca = pca.transform(X_mda_scaled)

cols39 = [
    f"prism_pca_{i:02d}"
    for i in range(39)
]

out = pd.DataFrame(
    X_mda_pca,
    index=mda.index,
    columns=cols39,
)

out.index.name = "standard_name"

out.to_csv(
    OUT / "prism65_external_pca39_by_standard_name.tsv",
    sep="\t",
)


# ============================================================
# Attach frozen MDA EOS drug IDs
# ============================================================

name_to_drug = dict(
    zip(
        mapping["standard_name"].astype(str),
        mapping["drug"].astype(str),
    )
)

missing_names = [
    x for x in out.index
    if x not in name_to_drug
]

if missing_names:
    raise RuntimeError(
        f"Cannot map standard names to MDA IDs: {missing_names}"
    )

out_id = out.copy()
out_id.insert(
    0,
    "drug",
    [name_to_drug[x] for x in out_id.index],
)

out_id = out_id.reset_index(drop=True)

out_id.to_csv(
    OUT / "prism65_external_pca39_by_drug.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# Save transform state / provenance
# ============================================================

with open(OUT / "prism39_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(OUT / "prism39_pca.pkl", "wb") as f:
    pickle.dump(pca, f)

cell_medians.to_csv(
    OUT / "external_prism_cellline_medians.tsv",
    sep="\t",
    header=["median"],
)

summary = {
    "mda_drugs": int(mda.shape[0]),
    "clean_cell_lines": int(mda.shape[1]),
    "excluded_mda_treatment_columns": len(mda_treatment_cols),
    "external_treatments": int(external.shape[0]),
    "external_missing_fraction_before_imputation":
        float(external.isna().mean().mean()),
    "external_missing_after_imputation":
        int(external_imp.isna().sum().sum()),
    "pca_components": 39,
    "pca_explained_variance_ratio_sum":
        float(pca.explained_variance_ratio_.sum()),
    "mda_used_to_fit_scaler": False,
    "mda_used_to_fit_pca": False,
    "mda_labels_used": False,
}

with open(
    QC / "PRISM39_EXTERNAL_PCA_QC.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

print("\n========== FINAL PRISM39 ==========")
print("shape:", out_id.shape)
print(
    "numeric shape:",
    out_id.select_dtypes(include=[np.number]).shape
)
print(
    "finite:",
    bool(np.isfinite(
        out_id.select_dtypes(include=[np.number]).to_numpy()
    ).all())
)

print("\nOUTPUT:")
print(
    OUT /
    "prism65_external_pca39_by_drug.tsv"
)
