from pathlib import Path
import re
import pandas as pd

H = Path.home()

NONCO = H/"ECARF_Article2_20260911/04_dataset2_audit_20260912/NoncoRNA_raw/NoncoRNA_2020-02-10.xlsx"
CPD = H/"MDA_CellPainting_Feasibility_20260909/03_cpg0036_drug/cpg0036_unique_compounds.tsv"
CPM = H/"MDA_CellPainting_Feasibility_20260909/08_hepg2_signal/01_profiles/cpg0046_mirna_aggregate_calibrated_replicated.tsv"
KNOWN = H/"MDA_CellPainting_Feasibility_20260909/08_hepg2_signal/02_associations/known_associations_3967.tsv"
OUT = H/"ECARF_Article2_20260911/02_results"

OUT.mkdir(parents=True, exist_ok=True)

def norm_drug(x):
    return re.sub(r"[^a-z0-9]+","",str(x).strip().lower())

def norm_mirna(x):
    x=str(x).strip().lower().replace("_","-").replace(" ","")
    if x.startswith("hsa-"):
        x=x[4:]
    return x

def family(x):
    x=norm_mirna(x)
    x=re.sub(r"-(3p|5p)$","",x)
    x=re.sub(r"-(1|2|3)$","",x)
    return x

# ---------- load ----------
n=pd.read_excel(NONCO)
cpd=pd.read_csv(CPD,sep="\t")
cpm=pd.read_csv(CPM,sep="\t")
known=pd.read_csv(KNOWN,sep="\t",dtype=str,keep_default_na=False)

n["type"]=n["ncrna_type"].astype(str).str.lower().str.strip()
n["effect"]=n["sensitive_or_resistant"].astype(str).str.lower().str.strip()
n["evidence"]=n["evidence_code"].astype(str).str.lower().str.strip()

base=n[
    (n["type"]=="mirna") &
    (n["effect"].isin(["sensitive","resistant"]))
].copy()

base["mirna_family"]=base["ncrna_name"].map(family)

# Dataset1 family pair keys using DrugBank
def first_db(x):
    m=re.findall(r"DB\d{5}",str(x).upper())
    return m[0] if m else None

base["DrugBank_ID"]=base["drug_id"].map(first_db)

known["mirna_family"]=known["ncRNA_Name"].map(family)
known["DrugBank_ID"]=known["DrugBank_ID"].astype(str).str.strip()

d1=set(zip(
    known.loc[known["DrugBank_ID"]!="","mirna_family"],
    known.loc[known["DrugBank_ID"]!="","DrugBank_ID"]
))

base["source_overlap"]=[
    (m,d) in d1 if d else False
    for m,d in zip(base["mirna_family"],base["DrugBank_ID"])
]

# ---------- full CPG0036 alias map ----------
alias_to_eos={}

for _,r in cpd.iterrows():
    eos=str(r["Metadata_EOS"])

    aliases=[
        r.get("standard_name",""),
        r.get("compound_original_name",""),
    ]

    syn=str(r.get("synonyms",""))
    if syn and syn.lower()!="nan":
        aliases += syn.split(";")

    for a in aliases:
        k=norm_drug(a)
        if len(k)>=3:
            alias_to_eos.setdefault(k,set()).add(eos)

unique_alias={
    k:next(iter(v))
    for k,v in alias_to_eos.items()
    if len(v)==1
}

# ---------- full CPG0046 mature map ----------
matures=sorted(set(cpm["miRNA"].astype(str)))

lookup={}
family_lookup={}

for m in matures:
    q=norm_mirna(m)
    lookup.setdefault(q,[]).append(m)
    family_lookup.setdefault(family(m),[]).append(m)

def resolve_mature(raw):
    q=norm_mirna(raw)

    ex=lookup.get(q,[])
    if len(ex)==1:
        return ex[0],"exact"

    if re.search(r"-(3p|5p)$",q):
        return None,"explicit_arm_unresolved"

    fam=family(raw)
    cands=sorted(set(family_lookup.get(fam,[])))

    if len(cands)==1:
        return cands[0],"singleton_arm"
    if len(cands)>1:
        return None,"ambiguous_arm"
    return None,"unresolved"

def audit(frame,label):
    x=frame[
        frame["DrugBank_ID"].notna() &
        (~frame["source_overlap"])
    ].copy()

    # independent source pairs first
    x=x.drop_duplicates(["mirna_family","DrugBank_ID"])

    x["drug_key"]=x["drug_name"].map(norm_drug)
    x["Metadata_EOS"]=x["drug_key"].map(unique_alias)

    mapped=x["ncrna_name"].map(resolve_mature)
    x["resolved_miRNA"]=[z[0] for z in mapped]
    x["miRNA_status"]=[z[1] for z in mapped]

    # family-level presence = upper bound
    x["family_in_CPG0046"]=x["mirna_family"].isin(family_lookup)

    drug_ok=x[x["Metadata_EOS"].notna()]
    strict=x[
        x["Metadata_EOS"].notna() &
        x["resolved_miRNA"].notna()
    ].copy()

    strict=strict.drop_duplicates(
        ["Metadata_EOS","resolved_miRNA"]
    )

    upper=x[
        x["Metadata_EOS"].notna() &
        x["family_in_CPG0046"]
    ].drop_duplicates(
        ["Metadata_EOS","mirna_family"]
    )

    print("\n"+"="*80)
    print(label)
    print("="*80)

    print("Source-independent family pairs:",len(x))
    print("Source-independent drugs:",x["drug_name"].nunique())
    print("Source-independent miRNA families:",x["mirna_family"].nunique())

    print("\nDrug mapping to full CPG0036:")
    print("Mapped drugs:",drug_ok["Metadata_EOS"].nunique())
    print("Pairs with mapped drug:",len(drug_ok))

    print("\nmiRNA mapping status:")
    print(x["miRNA_status"].value_counts())

    print("\nFAMILY-LEVEL UPPER BOUND:")
    print("Drugs:",upper["Metadata_EOS"].nunique())
    print("miRNA families:",upper["mirna_family"].nunique())
    print("Pairs:",len(upper))

    print("\nSTRICT MATURE FOUR-VIEW COHORT:")
    print("Drugs:",strict["Metadata_EOS"].nunique())
    print("miRNAs:",strict["resolved_miRNA"].nunique())
    print("Pairs:",len(strict))

    if len(strict):
        deg=strict.groupby("Metadata_EOS").size()
        print("\nPairs/drug:")
        print(deg.describe())
        print("Drugs >=5:",int((deg>=5).sum()))
        print("Drugs >=10:",int((deg>=10).sum()))
        print("Drugs >=20:",int((deg>=20).sum()))

    strict.to_csv(
        OUT/f"dataset2_noncorna_{label.lower()}_fulluniverse.tsv",
        sep="\t",index=False
    )

    return {
        "setting":label,
        "strict_drugs":strict["Metadata_EOS"].nunique(),
        "strict_mirnas":strict["resolved_miRNA"].nunique(),
        "strict_pairs":len(strict),
        "upper_drugs":upper["Metadata_EOS"].nunique(),
        "upper_mirna_families":upper["mirna_family"].nunique(),
        "upper_pairs":len(upper),
    }

rows=[
    audit(base,"ALL_RESPONSE"),
    audit(base[base["evidence"]=="validated"],"VALIDATED")
]

pd.DataFrame(rows).to_csv(
    OUT/"dataset2_noncorna_fulluniverse_summary.tsv",
    sep="\t",index=False
)

print("\nSUMMARY")
print(pd.DataFrame(rows).to_string(index=False))
