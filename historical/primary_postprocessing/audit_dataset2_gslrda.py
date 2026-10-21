from pathlib import Path
import re
import numpy as np
import pandas as pd

HOME = Path.home()

GSL = HOME / "My_module & PLMF-MDA/external_validation/processed/GSLRDA/GSLRDA_strict_miRNA_positive_long.csv"
GBR = HOME / "My_module & PLMF-MDA/external_validation/processed/GSLRDA/GSLRDA_drug_name_to_drugbank_bridge.csv"

KNOWN = HOME / "MDA_CellPainting_Feasibility_20260909/08_hepg2_signal/02_associations/known_associations_3967.tsv"

DP = HOME / "MDA_CellPainting_Multiview_20260910/01_data/strict_drug_phenotype_39d.tsv"
MF = HOME / "MDA_CellPainting_Multiview_20260910/01_data/fourview_complete_1195/mirna_rnafm_cls640.tsv"

D1 = HOME / "MDA_CellPainting_Multiview_20260910/01_data/fourview_complete_1195/associations_3443.tsv"

OUT = HOME / "ECARF_Article2_20260911/02_results"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------
g = pd.read_csv(GSL)
bridge = pd.read_csv(GBR)
known = pd.read_csv(KNOWN, sep="\t", dtype=str, keep_default_na=False)

current_drugs = set(
    pd.read_csv(DP, sep="\t")["Metadata_EOS"].astype(str)
)

mirna_df = pd.read_csv(MF, sep="\t")
current_mirnas = set(mirna_df["miRNA"].astype(str))

d1 = pd.read_csv(D1, sep="\t")
d1_pairs = set(zip(
    d1["miRNA"].astype(str),
    d1["drug"].astype(str)
))

print("="*80)
print("INPUT")
print("="*80)
print("GSLRDA rows:", len(g))
print("GSLRDA unique drugs:", g["drug_norm_canon"].nunique())
print("GSLRDA unique ncRNA names:", g["ncRNA_Name"].nunique())
print("Current four-view drugs:", len(current_drugs))
print("Current four-view miRNAs:", len(current_mirnas))
print("Dataset1 pairs:", len(d1_pairs))

print("\nKNOWN columns:")
print(list(known.columns))

# ---------------------------------------------------------
# Drug mapping:
# GSLRDA name -> DrugBank -> Metadata_EOS
# ---------------------------------------------------------
g["drug_norm_canon"] = (
    g["drug_norm_canon"]
    .astype(str)
    .str.strip()
    .str.lower()
)

bridge["drug_norm_canon"] = (
    bridge["drug_norm_canon"]
    .astype(str)
    .str.strip()
    .str.lower()
)

g = g.merge(
    bridge[["drug_norm_canon", "DrugBank_ID"]].drop_duplicates(),
    on="drug_norm_canon",
    how="left"
)

if "DrugBank_ID" in known.columns:
    kb = known[["DrugBank_ID", "Metadata_EOS"]].copy()
    kb["DrugBank_ID"] = kb["DrugBank_ID"].astype(str).str.strip()
    kb["Metadata_EOS"] = kb["Metadata_EOS"].astype(str).str.strip()

    kb = kb[
        (kb["DrugBank_ID"] != "") &
        (kb["Metadata_EOS"] != "")
    ].drop_duplicates()

    # only retain unambiguous DrugBank -> EOS mappings
    counts = kb.groupby("DrugBank_ID")["Metadata_EOS"].nunique()
    valid_db = set(counts[counts == 1].index)

    kb = kb[kb["DrugBank_ID"].isin(valid_db)].drop_duplicates(
        subset=["DrugBank_ID"]
    )

    g = g.merge(kb, on="DrugBank_ID", how="left")

elif "Drug_Name" in known.columns:
    # conservative fallback by normalized drug name
    def norm_drug(x):
        return re.sub(r"[^a-z0-9]+", "", str(x).lower())

    known2 = known[["Drug_Name", "Metadata_EOS"]].copy()
    known2["drug_key"] = known2["Drug_Name"].map(norm_drug)
    known2 = known2[known2["Metadata_EOS"].astype(str) != ""]

    cnt = known2.groupby("drug_key")["Metadata_EOS"].nunique()
    valid = set(cnt[cnt == 1].index)

    known2 = (
        known2[known2["drug_key"].isin(valid)]
        .drop_duplicates("drug_key")
    )

    g["drug_key"] = g["GSLRDA_Drug_Name"].map(norm_drug)
    g = g.merge(
        known2[["drug_key", "Metadata_EOS"]],
        on="drug_key",
        how="left"
    )
else:
    raise RuntimeError(
        "known_associations_3967.tsv has neither DrugBank_ID nor Drug_Name"
    )

g["drug_in_current65"] = g["Metadata_EOS"].isin(current_drugs)

# ---------------------------------------------------------
# miRNA mapping
# ---------------------------------------------------------
def norm_mirna(x):
    x = str(x).strip().lower()
    x = (
        x.replace("–", "-")
         .replace("—", "-")
         .replace("_", "-")
         .replace(" ", "")
    )
    if x.startswith("hsa-"):
        x = x[4:]
    return x

current_lookup = {}
for m in current_mirnas:
    key = norm_mirna(m)
    current_lookup.setdefault(key, []).append(m)

def arm_removed(x):
    return re.sub(r"-(3p|5p)$", "", x)

def map_mirna(raw):
    q = norm_mirna(raw)

    # exact mature-name match
    exact = current_lookup.get(q, [])
    if len(exact) == 1:
        return exact[0], "exact"

    # if an explicit arm is provided but no exact match:
    # do not guess.
    if re.search(r"-(3p|5p)$", q):
        return None, "explicit_arm_unresolved"

    # arm-unspecified names:
    # accept only when exactly one current mature candidate exists
    candidates = []

    for key, vals in current_lookup.items():
        base = arm_removed(key)

        if base == q or base.startswith(q + "-"):
            candidates.extend(vals)

    candidates = sorted(set(candidates))

    if len(candidates) == 1:
        return candidates[0], "singleton_arm"

    if len(candidates) > 1:
        return None, "ambiguous_arm"

    return None, "unresolved"

mapped = g["ncRNA_Name"].map(map_mirna)

g["resolved_miRNA"] = [x[0] for x in mapped]
g["miRNA_resolution"] = [x[1] for x in mapped]
g["mirna_in_current1195"] = g["resolved_miRNA"].isin(current_mirnas)

# ---------------------------------------------------------
# Four-view intersection
# ---------------------------------------------------------
usable = g[
    g["drug_in_current65"] &
    g["mirna_in_current1195"]
].copy()

usable = usable.rename(columns={"Metadata_EOS": "drug"})
usable["miRNA"] = usable["resolved_miRNA"]

usable = usable.drop_duplicates(
    subset=["miRNA", "drug"]
).copy()

usable["pair_in_dataset1"] = [
    (m, d) in d1_pairs
    for m, d in zip(usable["miRNA"], usable["drug"])
]

external = usable[~usable["pair_in_dataset1"]].copy()

# ---------------------------------------------------------
# Reports
# ---------------------------------------------------------
print("\n" + "="*80)
print("DRUG MAPPING")
print("="*80)

print("GSLRDA unique drugs:",
      g["drug_norm_canon"].nunique())

print("DrugBank-resolved unique drugs:",
      g.loc[g["DrugBank_ID"].notna(),
            "drug_norm_canon"].nunique())

print("EOS-resolved unique drugs:",
      g.loc[g["Metadata_EOS"].notna(),
            "Metadata_EOS"].nunique())

print("Overlap current 65 drugs:",
      g.loc[g["drug_in_current65"],
            "Metadata_EOS"].nunique())

print("\n" + "="*80)
print("miRNA MAPPING")
print("="*80)

print(g["miRNA_resolution"].value_counts())

resolved_rows = g[g["resolved_miRNA"].notna()]
print("\nResolved unique current miRNAs:",
      resolved_rows["resolved_miRNA"].nunique())

print("\n" + "="*80)
print("FOUR-VIEW DATASET2")
print("="*80)

print("Drugs:", usable["drug"].nunique())
print("miRNAs:", usable["miRNA"].nunique())
print("Positive pairs:", len(usable))

print("\nPositive pairs per drug:")
deg_d = usable.groupby("drug").size()
print(deg_d.describe())

print("\nPositive pairs per miRNA:")
deg_m = usable.groupby("miRNA").size()
print(deg_m.describe())

print("\n" + "="*80)
print("INDEPENDENCE FROM DATASET1")
print("="*80)

overlap_n = int(usable["pair_in_dataset1"].sum())
external_n = len(external)

print("Dataset2 four-view pairs:", len(usable))
print("Overlap Dataset1:", overlap_n)
print("External-only:", external_n)

if len(usable):
    print("Pair overlap rate:",
          f"{overlap_n/len(usable):.4f}")

print("\nExternal-only cohort:")
print("Drugs:", external["drug"].nunique())
print("miRNAs:", external["miRNA"].nunique())
print("Positive pairs:", len(external))

if len(external):
    ext_deg = external.groupby("drug").size()
    print("\nExternal-only positives per drug:")
    print(ext_deg.describe())

    print("\nDrugs with >=5 positives:",
          int((ext_deg >= 5).sum()))
    print("Drugs with >=10 positives:",
          int((ext_deg >= 10).sum()))
    print("Drugs with >=20 positives:",
          int((ext_deg >= 20).sum()))

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
g.to_csv(
    OUT / "dataset2_gslrda_mapping_audit.tsv",
    sep="\t", index=False
)

usable.to_csv(
    OUT / "dataset2_gslrda_fourview_pairs.tsv",
    sep="\t", index=False
)

external.to_csv(
    OUT / "dataset2_gslrda_external_only_pairs.tsv",
    sep="\t", index=False
)

summary = pd.DataFrame([{
    "raw_gslrda_rows": len(g),
    "fourview_drugs": usable["drug"].nunique(),
    "fourview_mirnas": usable["miRNA"].nunique(),
    "fourview_pairs": len(usable),
    "dataset1_overlap_pairs": overlap_n,
    "external_only_pairs": external_n,
    "external_only_drugs": external["drug"].nunique(),
    "external_only_mirnas": external["miRNA"].nunique(),
}])

summary.to_csv(
    OUT / "dataset2_gslrda_summary.tsv",
    sep="\t", index=False
)

print("\nSaved:")
print(OUT / "dataset2_gslrda_mapping_audit.tsv")
print(OUT / "dataset2_gslrda_fourview_pairs.tsv")
print(OUT / "dataset2_gslrda_external_only_pairs.tsv")
print(OUT / "dataset2_gslrda_summary.tsv")
