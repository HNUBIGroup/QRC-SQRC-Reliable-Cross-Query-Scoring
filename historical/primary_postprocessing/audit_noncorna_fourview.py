from pathlib import Path
import re
import pandas as pd

HOME = Path.home()

NONCO = HOME / (
    "ECARF_Article2_20260911/"
    "04_dataset2_audit_20260912/NoncoRNA_raw/"
    "NoncoRNA_2020-02-10.xlsx"
)

KNOWN = HOME / (
    "MDA_CellPainting_Feasibility_20260909/"
    "08_hepg2_signal/02_associations/"
    "known_associations_3967.tsv"
)

DP = HOME / (
    "MDA_CellPainting_Multiview_20260910/"
    "01_data/strict_drug_phenotype_39d.tsv"
)

MF = HOME / (
    "MDA_CellPainting_Multiview_20260910/"
    "01_data/fourview_complete_1195/"
    "mirna_rnafm_cls640.tsv"
)

D1 = HOME / (
    "MDA_CellPainting_Multiview_20260910/"
    "01_data/fourview_complete_1195/"
    "associations_3443.tsv"
)

OUT = HOME / "ECARF_Article2_20260911/02_results"
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================
# Utilities
# =========================================================
def first_db(x):
    hits = re.findall(r"DB\d{5}", str(x).upper())
    return hits[0] if hits else None

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

def family_key(x):
    x = norm_mirna(x)
    x = re.sub(r"-(3p|5p)$", "", x)
    x = re.sub(r"-(1|2|3)$", "", x)
    return x

def strip_arm(x):
    return re.sub(r"-(3p|5p)$", "", x)

# =========================================================
# Load
# =========================================================
n = pd.read_excel(NONCO)

known = pd.read_csv(
    KNOWN,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

current_drugs = set(
    pd.read_csv(DP, sep="\t")["Metadata_EOS"].astype(str)
)

current_mirnas = set(
    pd.read_csv(MF, sep="\t")["miRNA"].astype(str)
)

d1 = pd.read_csv(D1, sep="\t")
d1_exact = set(zip(
    d1["miRNA"].astype(str),
    d1["drug"].astype(str)
))

# Dataset1 family-level DrugBank pairs
known["mirna_family"] = known["ncRNA_Name"].map(family_key)
known["DrugBank_ID"] = known["DrugBank_ID"].astype(str).str.strip()

d1_family = set(zip(
    known.loc[known["DrugBank_ID"] != "", "mirna_family"],
    known.loc[known["DrugBank_ID"] != "", "DrugBank_ID"]
))

# =========================================================
# DrugBank -> EOS bridge
# =========================================================
bridge = known[
    ["DrugBank_ID", "Metadata_EOS"]
].copy()

bridge["DrugBank_ID"] = bridge["DrugBank_ID"].astype(str).str.strip()
bridge["Metadata_EOS"] = bridge["Metadata_EOS"].astype(str).str.strip()

bridge = bridge[
    (bridge["DrugBank_ID"] != "") &
    (bridge["Metadata_EOS"] != "")
]

cnt = bridge.groupby("DrugBank_ID")["Metadata_EOS"].nunique()
valid_db = set(cnt[cnt == 1].index)

bridge = (
    bridge[bridge["DrugBank_ID"].isin(valid_db)]
    .drop_duplicates("DrugBank_ID")
)

# =========================================================
# Current mature-miRNA lookup
# =========================================================
lookup = {}
for m in current_mirnas:
    lookup.setdefault(norm_mirna(m), []).append(m)

def resolve_mirna(raw):
    q = norm_mirna(raw)

    exact = lookup.get(q, [])
    if len(exact) == 1:
        return exact[0], "exact"

    # explicit arm but not exact => reject
    if re.search(r"-(3p|5p)$", q):
        return None, "explicit_arm_unresolved"

    # arm-unspecified family: accept only unique mature candidate
    cands = []

    for key, vals in lookup.items():
        base = strip_arm(key)

        # also remove precursor locus suffix
        base_family = re.sub(r"-(1|2|3)$", "", base)

        if (
            base == q
            or base_family == q
            or base.startswith(q + "-")
        ):
            cands.extend(vals)

    cands = sorted(set(cands))

    if len(cands) == 1:
        return cands[0], "singleton_arm"

    if len(cands) > 1:
        return None, "ambiguous_arm"

    return None, "unresolved"

# =========================================================
# Base task-compatible NoncoRNA
# =========================================================
n["type_norm"] = (
    n["ncrna_type"].astype(str)
    .str.lower().str.strip()
)

n["effect_norm"] = (
    n["sensitive_or_resistant"].astype(str)
    .str.lower().str.strip()
)

n["evidence_norm"] = (
    n["evidence_code"].astype(str)
    .str.lower().str.strip()
)

base = n[
    (n["type_norm"] == "mirna") &
    (n["effect_norm"].isin(["sensitive", "resistant"]))
].copy()

base["DrugBank_ID"] = base["drug_id"].map(first_db)
base["mirna_family"] = base["ncrna_name"].map(family_key)

# =========================================================
# Audit function
# =========================================================
def audit(frame, label):
    x = frame[frame["DrugBank_ID"].notna()].copy()

    # Remove Dataset1 family-level pairs FIRST
    x["source_family_overlap"] = [
        (m, d) in d1_family
        for m, d in zip(
            x["mirna_family"],
            x["DrugBank_ID"]
        )
    ]

    ext = x[~x["source_family_overlap"]].copy()

    source_pairs = set(zip(
        ext["mirna_family"],
        ext["DrugBank_ID"]
    ))

    # Drug mapping
    ext = ext.merge(
        bridge,
        on="DrugBank_ID",
        how="left"
    )

    ext["drug_in_current65"] = (
        ext["Metadata_EOS"].isin(current_drugs)
    )

    # miRNA mapping
    mapped = ext["ncrna_name"].map(resolve_mirna)

    ext["resolved_miRNA"] = [v[0] for v in mapped]
    ext["miRNA_resolution"] = [v[1] for v in mapped]

    ext["mirna_in_current1195"] = (
        ext["resolved_miRNA"].isin(current_mirnas)
    )

    four = ext[
        ext["drug_in_current65"] &
        ext["mirna_in_current1195"]
    ].copy()

    four["drug"] = four["Metadata_EOS"]
    four["miRNA"] = four["resolved_miRNA"]

    # Exact mature pair deduplication
    pairs = four.drop_duplicates(
        ["miRNA", "drug"]
    ).copy()

    # final safety check
    pairs["exact_pair_in_dataset1"] = [
        (m, d) in d1_exact
        for m, d in zip(
            pairs["miRNA"],
            pairs["drug"]
        )
    ]

    clean = pairs[
        ~pairs["exact_pair_in_dataset1"]
    ].copy()

    print("\n" + "="*80)
    print(label)
    print("="*80)

    print("Input records:", len(frame))
    print("DrugBank-resolved records:", len(x))

    print(
        "Source-independent family pairs:",
        len(source_pairs)
    )

    print(
        "Source-independent drugs:",
        ext["DrugBank_ID"].nunique()
    )

    print(
        "Source-independent miRNA families:",
        ext["mirna_family"].nunique()
    )

    print("\nmiRNA resolution:")
    print(ext["miRNA_resolution"].value_counts())

    print("\nFour-view mapped:")
    print("Drugs:", pairs["drug"].nunique())
    print("miRNAs:", pairs["miRNA"].nunique())
    print("Positive pairs:", len(pairs))

    residual_overlap = int(
        pairs["exact_pair_in_dataset1"].sum()
    )

    print(
        "Residual exact Dataset1 overlap:",
        residual_overlap
    )

    print("\nFINAL CLEAN EXTERNAL COHORT:")
    print("Drugs:", clean["drug"].nunique())
    print("miRNAs:", clean["miRNA"].nunique())
    print("Positive pairs:", len(clean))

    if len(clean):
        deg_d = clean.groupby("drug").size()

        print("\nPositive pairs per drug:")
        print(deg_d.describe())

        print(
            "Drugs >=5 positives:",
            int((deg_d >= 5).sum())
        )
        print(
            "Drugs >=10 positives:",
            int((deg_d >= 10).sum())
        )
        print(
            "Drugs >=20 positives:",
            int((deg_d >= 20).sum())
        )

        deg_m = clean.groupby("miRNA").size()

        print("\nPositive pairs per miRNA:")
        print(deg_m.describe())

    clean.to_csv(
        OUT / f"dataset2_noncorna_{label.lower()}_fourview_clean.tsv",
        sep="\t",
        index=False
    )

    return {
        "setting": label,
        "input_records": len(frame),
        "source_independent_family_pairs": len(source_pairs),
        "fourview_drugs": clean["drug"].nunique(),
        "fourview_mirnas": clean["miRNA"].nunique(),
        "fourview_pairs": len(clean),
    }

# =========================================================
# Run both evidence policies
# =========================================================
rows = []

rows.append(
    audit(
        base,
        "ALL_RESPONSE"
    )
)

validated = base[
    base["evidence_norm"] == "validated"
].copy()

rows.append(
    audit(
        validated,
        "VALIDATED"
    )
)

summary = pd.DataFrame(rows)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(summary.to_string(index=False))

summary.to_csv(
    OUT / "dataset2_noncorna_fourview_summary.tsv",
    sep="\t",
    index=False
)

print("\nSaved:")
print(
    OUT /
    "dataset2_noncorna_fourview_summary.tsv"
)
