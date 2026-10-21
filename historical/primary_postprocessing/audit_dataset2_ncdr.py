from pathlib import Path
import re
import pandas as pd

HOME = Path.home()

NCDR = HOME / "My_module & PLMF-MDA/external_validation/processed/ncDR_MDR_external/ncDR_Curated_DRmiRNA_standardized_all.csv"
KNOWN = HOME / "MDA_CellPainting_Feasibility_20260909/08_hepg2_signal/02_associations/known_associations_3967.tsv"

DP = HOME / "MDA_CellPainting_Multiview_20260910/01_data/strict_drug_phenotype_39d.tsv"
MF = HOME / "MDA_CellPainting_Multiview_20260910/01_data/fourview_complete_1195/mirna_rnafm_cls640.tsv"
D1 = HOME / "MDA_CellPainting_Multiview_20260910/01_data/fourview_complete_1195/associations_3443.tsv"

OUT = HOME / "ECARF_Article2_20260911/02_results"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- load ----------------
x = pd.read_csv(NCDR)
known = pd.read_csv(KNOWN, sep="\t", dtype=str, keep_default_na=False)
d1 = pd.read_csv(D1, sep="\t")

current_drugs = set(pd.read_csv(DP, sep="\t")["Metadata_EOS"].astype(str))
current_mirnas = set(pd.read_csv(MF, sep="\t")["miRNA"].astype(str))

d1_pairs = set(zip(d1["miRNA"].astype(str), d1["drug"].astype(str)))

print("="*80)
print("INPUT")
print("="*80)
print("ncDR rows:", len(x))
print("unique raw drugs:", x["Drug_Name"].nunique())
print("unique raw miRNAs:", x["miRNA_norm"].nunique())
print("Dataset1 pairs:", len(d1_pairs))

# Keep response semantics aligned with Dataset1
resp = x[(x["is_resistant"] == True) | (x["is_sensitive"] == True)].copy()

print("\nResponse rows:", len(resp))
print("Response unique drugs:", resp["Drug_Name"].nunique())
print("Response unique miRNAs:", resp["miRNA_norm"].nunique())

# ---------------- drug mapping ----------------
def norm_drug(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

# DrugBank -> EOS, only unambiguous mappings
db = known[["DrugBank_ID", "Metadata_EOS"]].copy()
db["DrugBank_ID"] = db["DrugBank_ID"].astype(str).str.strip()
db["Metadata_EOS"] = db["Metadata_EOS"].astype(str).str.strip()
db = db[(db["DrugBank_ID"] != "") & (db["Metadata_EOS"] != "")]

cnt = db.groupby("DrugBank_ID")["Metadata_EOS"].nunique()
valid = set(cnt[cnt == 1].index)
db = db[db["DrugBank_ID"].isin(valid)].drop_duplicates("DrugBank_ID")

resp["DrugBank_ID"] = resp["DrugBank_ID"].fillna("").astype(str).str.strip()
resp = resp.merge(db, on="DrugBank_ID", how="left")

# conservative unique-name fallback
nm = known[["Drug_Name", "Metadata_EOS"]].copy()
nm["drug_key"] = nm["Drug_Name"].map(norm_drug)
nm["Metadata_EOS"] = nm["Metadata_EOS"].astype(str).str.strip()
nm = nm[nm["Metadata_EOS"] != ""]

cnt = nm.groupby("drug_key")["Metadata_EOS"].nunique()
valid = set(cnt[cnt == 1].index)
nm = nm[nm["drug_key"].isin(valid)].drop_duplicates("drug_key")

resp["drug_key"] = resp["Drug_Name"].map(norm_drug)
fallback = resp[["drug_key"]].merge(
    nm[["drug_key", "Metadata_EOS"]],
    on="drug_key",
    how="left"
)["Metadata_EOS"]

resp["Metadata_EOS"] = resp["Metadata_EOS"].where(
    resp["Metadata_EOS"].notna(), fallback
)

resp["drug_in_current65"] = resp["Metadata_EOS"].isin(current_drugs)

# ---------------- miRNA mapping ----------------
def norm_mirna(s):
    s = str(s).strip().lower()
    s = s.replace("–","-").replace("—","-").replace("_","-").replace(" ","")
    if s.startswith("hsa-"):
        s = s[4:]
    return s

lookup = {}
for m in current_mirnas:
    lookup.setdefault(norm_mirna(m), []).append(m)

def strip_arm(s):
    return re.sub(r"-(3p|5p)$", "", s)

def resolve_mirna(raw):
    q = norm_mirna(raw)

    exact = lookup.get(q, [])
    if len(exact) == 1:
        return exact[0], "exact"

    if re.search(r"-(3p|5p)$", q):
        return None, "explicit_arm_unresolved"

    cands = []
    for k, vals in lookup.items():
        if strip_arm(k) == q or strip_arm(k).startswith(q + "-"):
            cands.extend(vals)

    cands = sorted(set(cands))

    if len(cands) == 1:
        return cands[0], "singleton_arm"
    if len(cands) > 1:
        return None, "ambiguous_arm"
    return None, "unresolved"

mapped = resp["miRNA_norm"].map(resolve_mirna)
resp["resolved_miRNA"] = [v[0] for v in mapped]
resp["miRNA_resolution"] = [v[1] for v in mapped]
resp["mirna_in_current1195"] = resp["resolved_miRNA"].isin(current_mirnas)

# ---------------- four-view cohort ----------------
usable = resp[
    resp["drug_in_current65"] &
    resp["mirna_in_current1195"]
].copy()

usable["drug"] = usable["Metadata_EOS"]
usable["miRNA"] = usable["resolved_miRNA"]

pairs = usable.drop_duplicates(["miRNA","drug"]).copy()

pairs["pair_in_dataset1"] = [
    (m,d) in d1_pairs
    for m,d in zip(pairs["miRNA"], pairs["drug"])
]

external = pairs[~pairs["pair_in_dataset1"]].copy()

# ---------------- report ----------------
print("\n" + "="*80)
print("DRUG MAPPING")
print("="*80)
print("Response unique drugs:", resp["Drug_Name"].nunique())
print("EOS-resolved drugs:", resp.loc[resp["Metadata_EOS"].notna(),"Metadata_EOS"].nunique())
print("Overlap current 65:", resp.loc[resp["drug_in_current65"],"Metadata_EOS"].nunique())

print("\n" + "="*80)
print("miRNA MAPPING")
print("="*80)
print(resp["miRNA_resolution"].value_counts())
print("Resolved current miRNAs:",
      resp.loc[resp["resolved_miRNA"].notna(),"resolved_miRNA"].nunique())

print("\n" + "="*80)
print("NCDR FOUR-VIEW")
print("="*80)
print("Drugs:", pairs["drug"].nunique())
print("miRNAs:", pairs["miRNA"].nunique())
print("Positive pairs:", len(pairs))

print("\n" + "="*80)
print("INDEPENDENCE FROM DATASET1")
print("="*80)
ov = int(pairs["pair_in_dataset1"].sum())
ext = len(external)

print("Overlap pairs:", ov)
print("External-only pairs:", ext)
print("Overlap rate:", f"{ov/len(pairs):.4f}" if len(pairs) else "NA")

print("\nExternal-only cohort:")
print("Drugs:", external["drug"].nunique())
print("miRNAs:", external["miRNA"].nunique())
print("Positive pairs:", len(external))

if len(external):
    deg = external.groupby("drug").size()
    print("\nExternal-only positives per drug:")
    print(deg.describe())
    print("Drugs >=5 positives:", int((deg>=5).sum()))
    print("Drugs >=10 positives:", int((deg>=10).sum()))
    print("Drugs >=20 positives:", int((deg>=20).sum()))

# save
resp.to_csv(OUT/"dataset2_ncdr_mapping_audit.tsv", sep="\t", index=False)
pairs.to_csv(OUT/"dataset2_ncdr_fourview_pairs.tsv", sep="\t", index=False)
external.to_csv(OUT/"dataset2_ncdr_external_only_pairs.tsv", sep="\t", index=False)

pd.DataFrame([{
    "fourview_drugs": pairs["drug"].nunique(),
    "fourview_mirnas": pairs["miRNA"].nunique(),
    "fourview_pairs": len(pairs),
    "dataset1_overlap_pairs": ov,
    "external_only_pairs": ext,
    "external_only_drugs": external["drug"].nunique(),
    "external_only_mirnas": external["miRNA"].nunique(),
}]).to_csv(OUT/"dataset2_ncdr_summary.tsv", sep="\t", index=False)

print("\nSaved ncDR audit files.")
