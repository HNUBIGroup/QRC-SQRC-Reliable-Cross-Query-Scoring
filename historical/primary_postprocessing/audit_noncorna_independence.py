from pathlib import Path
import re
import pandas as pd

HOME = Path.home()

NONCO = (
    HOME / "ECARF_Article2_20260911/"
    "04_dataset2_audit_20260912/NoncoRNA_raw/"
    "NoncoRNA_2020-02-10.xlsx"
)

D1 = (
    HOME / "MDA_CellPainting_Feasibility_20260909/"
    "08_hepg2_signal/02_associations/"
    "known_associations_3967.tsv"
)

OUT = HOME / "ECARF_Article2_20260911/02_results"
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================
# Load
# =========================================================
n = pd.read_excel(NONCO)
d = pd.read_csv(D1, sep="\t", dtype=str, keep_default_na=False)

print("="*80)
print("NONCORNA RAW")
print("="*80)

print("rows:", len(n))
print("unique ncRNA names:", n["ncrna_name"].nunique())
print("unique drugs:", n["drug_name"].nunique())
print("unique PMIDs:", n["pmid"].nunique())

print("\nncrna_type:")
print(n["ncrna_type"].astype(str).str.lower().value_counts(dropna=False))

print("\nsensitive_or_resistant:")
print(
    n["sensitive_or_resistant"]
    .astype(str).str.lower().str.strip()
    .value_counts(dropna=False)
)

print("\nevidence_code:")
print(
    n["evidence_code"]
    .astype(str).str.lower().str.strip()
    .value_counts(dropna=False)
)

# =========================================================
# Keep task-compatible NoncoRNA records
# =========================================================
n["type_norm"] = n["ncrna_type"].astype(str).str.lower().str.strip()
n["effect_norm"] = (
    n["sensitive_or_resistant"]
    .astype(str).str.lower().str.strip()
)

cand = n[
    (n["type_norm"] == "mirna") &
    (n["effect_norm"].isin(["sensitive", "resistant"]))
].copy()

print("\n" + "="*80)
print("NONCORNA miRNA DRUG-RESPONSE CANDIDATES")
print("="*80)

print("records:", len(cand))
print("unique drugs:", cand["drug_name"].nunique())
print("unique miRNAs:", cand["ncrna_name"].nunique())
print("unique PMIDs:", cand["pmid"].nunique())

# =========================================================
# DrugBank parsing
# Use first/primary DB accession in NoncoRNA drug_id
# =========================================================
def first_db(x):
    hits = re.findall(r"DB\d{5}", str(x).upper())
    return hits[0] if hits else None

cand["DrugBank_ID"] = cand["drug_id"].map(first_db)

print("\nDrugBank mapping:")
print("records with DrugBank:", cand["DrugBank_ID"].notna().sum())
print(
    "unique DrugBank drugs:",
    cand.loc[cand["DrugBank_ID"].notna(), "DrugBank_ID"].nunique()
)

# =========================================================
# miRNA normalization
#
# family_key is ONLY for source-independence auditing.
# It is deliberately broader than the later mature-arm mapping.
# =========================================================
def mirna_basic(x):
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

def family_key(x):
    x = mirna_basic(x)

    # mature arm
    x = re.sub(r"-(3p|5p)$", "", x)

    # precursor locus, e.g. miR-125b-1 -> miR-125b
    x = re.sub(r"-(1|2|3)$", "", x)

    return x

cand["mirna_family"] = cand["ncrna_name"].map(family_key)

# Dataset1
d["mirna_family"] = d["ncRNA_Name"].map(family_key)
d["DrugBank_ID"] = d["DrugBank_ID"].astype(str).str.strip()
d["PMID"] = d["PMID"].astype(str).str.strip()

cand["PMID_norm"] = (
    cand["pmid"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.strip()
)

# =========================================================
# Restrict to rows with usable DB identifiers
# =========================================================
nc = cand[cand["DrugBank_ID"].notna()].copy()

d_valid = d[
    (d["DrugBank_ID"] != "") &
    (d["mirna_family"] != "")
].copy()

# unique family-level miRNA-drug pairs
nonco_pairs = set(zip(
    nc["mirna_family"],
    nc["DrugBank_ID"]
))

d1_pairs = set(zip(
    d_valid["mirna_family"],
    d_valid["DrugBank_ID"]
))

overlap = nonco_pairs & d1_pairs
external = nonco_pairs - d1_pairs

# =========================================================
# PMID-aware evidence overlap
# =========================================================
nonco_evidence = set(zip(
    nc["PMID_norm"],
    nc["mirna_family"],
    nc["DrugBank_ID"]
))

d1_evidence = set(zip(
    d_valid["PMID"],
    d_valid["mirna_family"],
    d_valid["DrugBank_ID"]
))

evidence_overlap = nonco_evidence & d1_evidence

nonco_pmids = set(nc["PMID_norm"]) - {"", "nan"}
d1_pmids = set(d_valid["PMID"]) - {"", "nan"}
pmid_overlap = nonco_pmids & d1_pmids

# =========================================================
# Report
# =========================================================
print("\n" + "="*80)
print("SOURCE-LEVEL INDEPENDENCE")
print("="*80)

print("NoncoRNA unique miRNA-drug family pairs:", len(nonco_pairs))
print("Dataset1 unique family pairs:", len(d1_pairs))

print("\nPair overlap:", len(overlap))
print("NoncoRNA-only pairs:", len(external))

if nonco_pairs:
    print(
        "Pair overlap rate:",
        f"{len(overlap)/len(nonco_pairs):.4f}"
    )

print("\nNoncoRNA PMIDs:", len(nonco_pmids))
print("Dataset1 PMIDs:", len(d1_pmids))
print("Shared PMIDs:", len(pmid_overlap))

if nonco_pmids:
    print(
        "PMID overlap rate:",
        f"{len(pmid_overlap)/len(nonco_pmids):.4f}"
    )

print("\nExact PMID + miRNA-family + drug evidence overlap:",
      len(evidence_overlap))

# =========================================================
# Save pair-level table
# =========================================================
pair_df = (
    nc[[
        "ncrna_name",
        "mirna_family",
        "drug_name",
        "DrugBank_ID"
    ]]
    .drop_duplicates()
    .copy()
)

pair_df["in_dataset1_family_pair"] = [
    (m, dr) in d1_pairs
    for m, dr in zip(
        pair_df["mirna_family"],
        pair_df["DrugBank_ID"]
    )
]

external_df = pair_df[
    ~pair_df["in_dataset1_family_pair"]
].copy()

pair_df.to_csv(
    OUT / "dataset2_noncorna_source_pair_audit.tsv",
    sep="\t",
    index=False
)

external_df.to_csv(
    OUT / "dataset2_noncorna_source_external_only.tsv",
    sep="\t",
    index=False
)

summary = pd.DataFrame([{
    "raw_records": len(n),
    "mirna_response_records": len(cand),
    "unique_response_drugs": cand["drug_name"].nunique(),
    "unique_response_mirnas": cand["ncrna_name"].nunique(),
    "noncorna_family_pairs": len(nonco_pairs),
    "dataset1_family_overlap_pairs": len(overlap),
    "noncorna_only_family_pairs": len(external),
    "pair_overlap_rate":
        len(overlap)/len(nonco_pairs) if nonco_pairs else None,
    "noncorna_pmids": len(nonco_pmids),
    "shared_pmids": len(pmid_overlap),
    "pmid_overlap_rate":
        len(pmid_overlap)/len(nonco_pmids) if nonco_pmids else None,
    "exact_evidence_overlap": len(evidence_overlap),
}])

summary.to_csv(
    OUT / "dataset2_noncorna_source_summary.tsv",
    sep="\t",
    index=False
)

print("\nSaved:")
print(OUT / "dataset2_noncorna_source_pair_audit.tsv")
print(OUT / "dataset2_noncorna_source_external_only.tsv")
print(OUT / "dataset2_noncorna_source_summary.tsv")
