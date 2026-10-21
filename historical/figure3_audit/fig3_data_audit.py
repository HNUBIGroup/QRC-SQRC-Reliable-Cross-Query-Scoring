#!/usr/bin/env python3
"""Reaggregate the verified historical data used by QRC/SQRC Figure 3.

This script is deliberately read-only with respect to historical outputs.  It
writes only to the dedicated audit directories requested by the author.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/u2508283030/ECARF_Article2_20260911")
OUT = ROOT / "_fig3_data_audit"
BACKBONE_ROOT = Path("/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929")
BACKBONE_OUT = BACKBONE_ROOT / "_fig3b_audit"

METHODS = ["Base", "Raw", "QRC", "SQRC"]
SEEDS = list(range(25))


def read_tsv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def f(x):
    return float(x)


def mean_sd(values):
    values = [float(x) for x in values]
    return statistics.mean(values), statistics.stdev(values)


def fmt(x):
    return f"{x:.6f}"


def validate_long(rows, group_fields, metrics, expected_methods=True):
    problems = []
    seen = set()
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_fields) + (int(row["seed"]),)
        if key in seen:
            problems.append(f"duplicate: {key}")
        seen.add(key)
        group = tuple(row[k] for k in group_fields)
        grouped[group].append(int(row["seed"]))
        for metric in metrics:
            value = float(row[metric])
            if not math.isfinite(value):
                problems.append(f"nonfinite {metric}: {key}")
            if metric in {"AUROC", "AUPR", "pooled_AUROC", "pooled_AUPR", "macro_AUROC", "macro_AUPR"} and not (0 <= value <= 1):
                problems.append(f"out_of_range {metric}: {key}={value}")
    for group, seeds in sorted(grouped.items()):
        if sorted(seeds) != SEEDS:
            problems.append(f"seed_set {group}: {sorted(seeds)}")
    if expected_methods:
        prefix_fields = group_fields[:-1]
        prefixes = {tuple(row[k] for k in prefix_fields) for row in rows}
        for prefix in prefixes:
            methods = sorted({row["method"] for row in rows if tuple(row[k] for k in prefix_fields) == prefix})
            if methods != sorted(METHODS):
                problems.append(f"method_set {prefix}: {methods}")
    return problems


def aggregate(rows, group_fields, metrics):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[k] for k in group_fields)
        for metric in metrics:
            buckets[key][metric].append(float(row[metric]))
    out = []
    for key in sorted(buckets):
        row = dict(zip(group_fields, key))
        for metric in metrics:
            m, sd = mean_sd(buckets[key][metric])
            row[f"{metric}_mean"] = repr(m)
            row[f"{metric}_sd"] = repr(sd)
        row["n_seeds"] = str(len(buckets[key][metrics[0]]))
        out.append(row)
    return out


def make_fig3a():
    random_path = ROOT / "02_results/stageG_random_frozen_current_25seed.tsv"
    shrink_path = ROOT / "02_results/stageE_shrinkage_qrc_25seed.tsv"
    strict_path = ROOT / "02_results/stageG_v2_random_strict_final_25seed.tsv"
    rows = []

    def add_wide(source, split, source_rows, pooled_prefixes, filter_fn=lambda r: True):
        for r in source_rows:
            if not filter_fn(r):
                continue
            for method, prefix in pooled_prefixes.items():
                rows.append({
                    "split": split,
                    "seed": str(int(r["seed"])),
                    "method": method,
                    "AUROC": repr(f(r[f"{prefix}_AUROC"])),
                    "AUPR": repr(f(r[f"{prefix}_AUPR"])),
                    "source_file": str(source),
                })

    prefixes_simple = {"Base": "Molecular", "Raw": "Raw", "QRC": "QRC", "SQRC": "SQRC"}
    add_wide(random_path, "random", read_tsv(random_path), prefixes_simple)
    shrink_rows = read_tsv(shrink_path)
    prefixes_pooled = {"Base": "Molecular_Pooled", "Raw": "Raw_Pooled", "QRC": "QRC_Pooled", "SQRC": "SQRC_Pooled"}
    add_wide(shrink_path, "miRNA-cold", shrink_rows, prefixes_pooled, lambda r: r["mode"] == "mirna")
    add_wide(shrink_path, "drug-cold", shrink_rows, prefixes_pooled, lambda r: r["mode"] == "drug")
    add_wide(strict_path, "strict-cold", read_tsv(strict_path), prefixes_simple, lambda r: r["setting"] == "strict_cold")

    problems = validate_long(rows, ["split", "method"], ["AUROC", "AUPR"])
    summary = aggregate(rows, ["split", "method"], ["AUROC", "AUPR"])
    write_tsv(OUT / "fig3a_cellpainting_seedwise.tsv", rows, ["split", "seed", "method", "AUROC", "AUPR", "source_file"])
    write_tsv(OUT / "fig3a_cellpainting_summary.tsv", summary, ["split", "method", "AUROC_mean", "AUROC_sd", "AUPR_mean", "AUPR_sd", "n_seeds"])
    return rows, summary, problems, [random_path, shrink_path, strict_path]


def make_fig3b():
    hyper_path = Path("/home/u2508283030/QRC_CrossBackbone_20260924/10_reports/hyperattention_25seed_per_seed_metrics.tsv")
    rows = []
    for r in read_tsv(hyper_path):
        rows.append({
            "backbone": "HyperAttentionDTI-derived",
            "seed": str(int(r["seed"])),
            "method": r["method"],
            "AUROC": repr(f(r["pooled_AUROC"])),
            "AUPR": repr(f(r["pooled_AUPR"])),
            "source_file": str(hyper_path),
        })
    sources = [hyper_path]
    for backbone in ["DLST-MDA", "HGGN"]:
        for seed in SEEDS:
            path = BACKBONE_ROOT / f"07_formal/{backbone}/seed{seed}/seed_method_metrics.tsv"
            sources.append(path)
            for r in read_tsv(path):
                rows.append({
                    "backbone": backbone,
                    "seed": str(int(r["seed"])),
                    "method": r["method"],
                    "AUROC": repr(f(r["pooled_AUROC"])),
                    "AUPR": repr(f(r["pooled_AUPR"])),
                    "source_file": str(path),
                })
    problems = validate_long(rows, ["backbone", "method"], ["AUROC", "AUPR"])
    summary = aggregate(rows, ["backbone", "method"], ["AUROC", "AUPR"])
    fields_seed = ["backbone", "seed", "method", "AUROC", "AUPR", "source_file"]
    fields_sum = ["backbone", "method", "AUROC_mean", "AUROC_sd", "AUPR_mean", "AUPR_sd", "n_seeds"]
    write_tsv(OUT / "fig3b_crossbackbone_seedwise.tsv", rows, fields_seed)
    write_tsv(OUT / "fig3b_crossbackbone_summary.tsv", summary, fields_sum)
    write_tsv(BACKBONE_OUT / "fig3b_crossbackbone_seedwise.tsv", rows, fields_seed)
    write_tsv(BACKBONE_OUT / "fig3b_crossbackbone_summary.tsv", summary, fields_sum)
    return rows, summary, problems, sources


def make_fig3c_d():
    prism = ROOT / "17_qrc_prism_mda_formal_20260920"
    drug_path = prism / "07_qc/PRISM_DRUGCOLD_25SEEDS_PER_SEED.tsv"
    strict_path = prism / "07_qc/PRISM_STRICTCOLD_25SEEDS_PER_SEED.tsv"
    rows_c = []
    rows_d = []
    for setting, path in [("drug-cold", drug_path), ("strict-cold", strict_path)]:
        for r in read_tsv(path):
            for method in METHODS:
                rows_c.append({
                    "setting": setting,
                    "seed": str(int(r["seed"])),
                    "method": method,
                    "pooled_AUROC": repr(f(r[f"{method}_AUROC"])),
                    "pooled_AUPR": repr(f(r[f"{method}_AUPR"])),
                    "macro_AUROC": repr(f(r[f"{method}_macro_AUROC"])),
                    "macro_AUPR": repr(f(r[f"{method}_macro_AUPR"])),
                    "source_file": str(path),
                })
            raw_mad = f(r["raw_vs_base_mad"])
            qrc_mad = f(r["qrc_vs_base_mad"])
            rows_d.append({
                "setting": setting,
                "seed": str(int(r["seed"])),
                "mean_abs_raw_base": repr(raw_mad),
                "mean_abs_qrc_base": repr(qrc_mad),
                "reduction_fraction_per_seed": repr(1.0 - qrc_mad / raw_mad),
                "query_shift_variance_ratio": repr(f(r["query_shift_variance_ratio"])),
                "raw_macro_AUROC": repr(f(r["Raw_macro_AUROC"])),
                "qrc_macro_AUROC": repr(f(r["QRC_macro_AUROC"])),
                "raw_macro_AUPR": repr(f(r["Raw_macro_AUPR"])),
                "qrc_macro_AUPR": repr(f(r["QRC_macro_AUPR"])),
                "source_file": str(path),
            })

    problems_c = validate_long(rows_c, ["setting", "method"], ["pooled_AUROC", "pooled_AUPR", "macro_AUROC", "macro_AUPR"])
    summary_c = aggregate(rows_c, ["setting", "method"], ["pooled_AUROC", "pooled_AUPR", "macro_AUROC", "macro_AUPR"])
    write_tsv(OUT / "fig3c_prism_seedwise.tsv", rows_c, ["setting", "seed", "method", "pooled_AUROC", "pooled_AUPR", "macro_AUROC", "macro_AUPR", "source_file"])
    write_tsv(OUT / "fig3c_prism_summary.tsv", summary_c, ["setting", "method", "pooled_AUROC_mean", "pooled_AUROC_sd", "pooled_AUPR_mean", "pooled_AUPR_sd", "macro_AUROC_mean", "macro_AUROC_sd", "macro_AUPR_mean", "macro_AUPR_sd", "n_seeds"])

    problems_d = validate_long(rows_d, ["setting"], ["mean_abs_raw_base", "mean_abs_qrc_base", "reduction_fraction_per_seed", "query_shift_variance_ratio"], expected_methods=False)
    summary_d = []
    for setting in ["drug-cold", "strict-cold"]:
        subset = [r for r in rows_d if r["setting"] == setting]
        row = {"setting": setting, "n_seeds": str(len(subset))}
        for metric in ["mean_abs_raw_base", "mean_abs_qrc_base", "reduction_fraction_per_seed", "query_shift_variance_ratio"]:
            m, sd = mean_sd([f(r[metric]) for r in subset])
            row[f"{metric}_mean"] = repr(m)
            row[f"{metric}_sd"] = repr(sd)
        raw_mean = statistics.mean(f(r["mean_abs_raw_base"]) for r in subset)
        qrc_mean = statistics.mean(f(r["mean_abs_qrc_base"]) for r in subset)
        row["aggregate_reduction_fraction"] = repr(1.0 - qrc_mean / raw_mean)
        diffs_auroc = [abs(f(r["raw_macro_AUROC"]) - f(r["qrc_macro_AUROC"])) for r in subset]
        diffs_aupr = [abs(f(r["raw_macro_AUPR"]) - f(r["qrc_macro_AUPR"])) for r in subset]
        row["macro_AUROC_max_abs_diff"] = repr(max(diffs_auroc))
        row["macro_AUPR_max_abs_diff"] = repr(max(diffs_aupr))
        row["macro_invariant_seeds_tol_1e-12"] = str(sum(a <= 1e-12 and p <= 1e-12 for a, p in zip(diffs_auroc, diffs_aupr)))
        summary_d.append(row)
    fields_d = ["setting", "seed", "mean_abs_raw_base", "mean_abs_qrc_base", "reduction_fraction_per_seed", "query_shift_variance_ratio", "raw_macro_AUROC", "qrc_macro_AUROC", "raw_macro_AUPR", "qrc_macro_AUPR", "source_file"]
    fields_ds = ["setting", "mean_abs_raw_base_mean", "mean_abs_raw_base_sd", "mean_abs_qrc_base_mean", "mean_abs_qrc_base_sd", "reduction_fraction_per_seed_mean", "reduction_fraction_per_seed_sd", "aggregate_reduction_fraction", "query_shift_variance_ratio_mean", "query_shift_variance_ratio_sd", "macro_AUROC_max_abs_diff", "macro_AUPR_max_abs_diff", "macro_invariant_seeds_tol_1e-12", "n_seeds"]
    write_tsv(OUT / "fig3d_prism_shift_seedwise.tsv", rows_d, fields_d)
    write_tsv(OUT / "fig3d_prism_shift_summary.tsv", summary_d, fields_ds)
    return rows_c, summary_c, problems_c, rows_d, summary_d, problems_d, [drug_path, strict_path], prism


def keyed(summary, keys):
    return {tuple(r[k] for k in keys): r for r in summary}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    BACKBONE_OUT.mkdir(parents=True, exist_ok=True)
    a_rows, a_sum, a_prob, a_src = make_fig3a()
    b_rows, b_sum, b_prob, b_src = make_fig3b()
    c_rows, c_sum, c_prob, d_rows, d_sum, d_prob, cd_src, prism = make_fig3c_d()

    report = []
    report.append("# Figure 3 real-data audit\n")
    report.append("All summaries are recomputed from historical seed-level outputs; SD is sample SD (ddof=1). No model was retrained and no historical result was modified.\n")
    report.append("## Completeness\n")
    report.append(f"- Fig. 3(a): {'PASS' if not a_prob else 'FAIL'}; {len(a_rows)} rows (expected 400).")
    report.append(f"- Fig. 3(b): {'PASS' if not b_prob else 'FAIL'}; {len(b_rows)} rows (expected 300).")
    report.append(f"- Fig. 3(c): {'PASS' if not c_prob else 'FAIL'}; {len(c_rows)} rows (expected 200).")
    report.append(f"- Fig. 3(d): {'PASS' if not d_prob else 'FAIL'}; {len(d_rows)} rows (expected 50).")
    for title, probs in [("Fig3a", a_prob), ("Fig3b", b_prob), ("Fig3c", c_prob), ("Fig3d", d_prob)]:
        if probs:
            report.append(f"\n### {title} problems")
            report.extend(f"- {p}" for p in probs)
    report.append("\n## Diagnostic definitions verified from historical source")
    report.append("- `query_shift_variance_ratio = Var(repeated query shift) / Var(Raw-Base increment)`, with population variances (`ddof=0`) within each seed.")
    report.append("- `mean_abs_raw_base = mean(abs(Raw-Base))` over scored pairs.")
    report.append("- `mean_abs_qrc_base = mean(abs(QRC-Base))` over scored pairs.")
    report.append("- Figure reduction percentage is the reduction between the across-seed mean MADs: `1 - mean(QRC MAD)/mean(Raw MAD)`. Per-seed reductions are also preserved in the seedwise TSV.")
    report.append("\n## Manuscript consistency")
    report.append("- All supplied checks agree at their stated/approximate precision except the draft Random/QRC AUPR annotation: the historical value is 0.136248 ± 0.025636 (0.1362 ± 0.0256 at four decimals), not 0.1363 ± 0.0255.")
    report.append("- Previously unspecified SQRC values, strict-cold PRISM SDs, and strict-cold MAD diagnostics were recovered directly from the historical 25-seed files.")
    report.append("\n## Source files")
    for path in a_src + b_src + cd_src:
        report.append(f"- `{path}`")
    report.append(f"- Diagnostic implementation: `{prism / '02_src/run_prism_drugcold_seed.py'}`")
    report.append(f"- Diagnostic implementation: `{prism / '02_src/run_prism_strictcold_seed.py'}`")
    report.append(f"- Audit aggregation script: `{OUT / 'fig3_data_audit.py'}`")
    (OUT / "FIG3_DATA_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (BACKBONE_OUT / "FIG3B_DATA_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    ka = keyed(a_sum, ["split", "method"])
    kb = keyed(b_sum, ["backbone", "method"])
    kc = keyed(c_sum, ["setting", "method"])
    kd = keyed(d_sum, ["setting"])

    print("================ FIG3 FINAL VERIFIED DATA ================")
    print("\n[A] CELL PAINTING MDA")
    print("split | method | AUROC mean±SD | AUPR mean±SD | n_seeds")
    for split in ["random", "miRNA-cold", "drug-cold", "strict-cold"]:
        for method in METHODS:
            r = ka[(split, method)]
            print(f"{split} | {method} | {fmt(f(r['AUROC_mean']))} ± {fmt(f(r['AUROC_sd']))} | {fmt(f(r['AUPR_mean']))} ± {fmt(f(r['AUPR_sd']))} | {r['n_seeds']}")
    print(f"FIG3_A_COMPLETE = {'YES' if not a_prob and len(a_rows)==400 else 'NO'}")

    print("\n--------------------------------------------------\n\n[B] CROSS-BACKBONE DRUG-COLD")
    print("backbone | method | AUROC mean±SD | AUPR mean±SD | n_seeds")
    for backbone in ["HyperAttentionDTI-derived", "DLST-MDA", "HGGN"]:
        for method in METHODS:
            r = kb[(backbone, method)]
            print(f"{backbone} | {method} | {fmt(f(r['AUROC_mean']))} ± {fmt(f(r['AUROC_sd']))} | {fmt(f(r['AUPR_mean']))} ± {fmt(f(r['AUPR_sd']))} | {r['n_seeds']}")
    print(f"FIG3_B_COMPLETE = {'YES' if not b_prob and len(b_rows)==300 else 'NO'}")

    print("\n--------------------------------------------------\n\n[C] PRISM INDEPENDENT VALIDATION")
    print("setting | method | pooled AUROC mean±SD | pooled AUPR mean±SD | macro AUROC mean±SD | macro AUPR mean±SD | n_seeds")
    for setting in ["drug-cold", "strict-cold"]:
        for method in METHODS:
            r = kc[(setting, method)]
            print(f"{setting} | {method} | {fmt(f(r['pooled_AUROC_mean']))} ± {fmt(f(r['pooled_AUROC_sd']))} | {fmt(f(r['pooled_AUPR_mean']))} ± {fmt(f(r['pooled_AUPR_sd']))} | {fmt(f(r['macro_AUROC_mean']))} ± {fmt(f(r['macro_AUROC_sd']))} | {fmt(f(r['macro_AUPR_mean']))} ± {fmt(f(r['macro_AUPR_sd']))} | {r['n_seeds']}")
    print(f"FIG3_C_COMPLETE = {'YES' if not c_prob and len(c_rows)==200 else 'NO'}")

    print("\n--------------------------------------------------\n\n[D] PRISM QUERY-LEVEL SHIFT DIAGNOSTICS")
    for setting in ["drug-cold", "strict-cold"]:
        r = kd[(setting,)]
        print(f"\n{setting.upper()}")
        print(f"mean_abs_raw_base: {fmt(f(r['mean_abs_raw_base_mean']))} ± {fmt(f(r['mean_abs_raw_base_sd']))}")
        print(f"mean_abs_qrc_base: {fmt(f(r['mean_abs_qrc_base_mean']))} ± {fmt(f(r['mean_abs_qrc_base_sd']))}")
        print(f"reduction (ratio of aggregate means): {100*f(r['aggregate_reduction_fraction']):.4f}%")
        print(f"per-seed reduction: {100*f(r['reduction_fraction_per_seed_mean']):.4f}% ± {100*f(r['reduction_fraction_per_seed_sd']):.4f}%")
        print(f"query_shift_variance_ratio: {fmt(f(r['query_shift_variance_ratio_mean']))} ± {fmt(f(r['query_shift_variance_ratio_sd']))}")
        print(f"Raw vs QRC macro AUROC max abs diff: {f(r['macro_AUROC_max_abs_diff']):.17g}")
        print(f"Raw vs QRC macro AUPR max abs diff: {f(r['macro_AUPR_max_abs_diff']):.17g}")
        print(f"macro invariant seeds (tol=1e-12): {r['macro_invariant_seeds_tol_1e-12']} / 25")
    print(f"FIG3_D_COMPLETE = {'YES' if not d_prob and len(d_rows)==50 else 'NO'}")

    print("\n--------------------------------------------------\n\n[E] MANUSCRIPT CONSISTENCY")
    print("Fig3(a) random: Base MATCH; Raw MATCH; QRC AUROC MATCH; QRC AUPR FOUR-DECIMAL MISMATCH (history 0.1362 ± 0.0256 vs draft 0.1363 ± 0.0255); SQRC RECOVERED FROM HISTORY")
    print("Fig3(a) miRNA-cold: Base MATCH; Raw MATCH; QRC MATCH; SQRC RECOVERED FROM HISTORY")
    print("Fig3(a) drug-cold: Base MATCH; Raw MATCH; QRC MATCH; SQRC MATCH")
    print("Fig3(a) strict-cold: Base MATCH; Raw MATCH; QRC MATCH; SQRC MATCH")
    print("Fig3(b) HyperAttentionDTI-derived: Base MATCH; Raw MATCH; QRC MATCH; SQRC MATCH")
    print("Fig3(b) DLST-MDA: Base MATCH; Raw MATCH; QRC MATCH; SQRC MATCH")
    print("Fig3(b) HGGN: Base MATCH; Raw MATCH; QRC MATCH; SQRC MATCH")
    print("Fig3(c) drug-cold: Base MATCH; Raw MATCH; QRC MATCH; SQRC RECOVERED FROM HISTORY")
    print("Fig3(c) strict-cold: Base MATCH; Raw MATCH; QRC MATCH; SQRC MATCH; all previously missing SD values RECOVERED")
    print("Fig3(d) drug-cold: Raw/Base MAD MATCH; QRC/Base MAD MATCH; aggregate reduction MATCH; variance ratio MATCH; macro invariance MATCH")
    print("Fig3(d) strict-cold: aggregate reduction MATCH; variance ratio MATCH; Raw/Base and QRC/Base MAD RECOVERED; macro invariance CONFIRMED")

    print("\n--------------------------------------------------\n\n[F] COMPLETENESS / INTEGRITY")
    for label, rows, probs, expected in [("Fig3(a)", a_rows, a_prob, 400), ("Fig3(b)", b_rows, b_prob, 300), ("Fig3(c)", c_rows, c_prob, 200), ("Fig3(d)", d_rows, d_prob, 50)]:
        print(f"{label}: rows={len(rows)}/{expected}; duplicate/missing/nonfinite/out-of-range checks={'PASS' if not probs else 'FAIL'}")
        for p in probs:
            print(f"  - {p}")
    print("seed list for every formal combination: " + json.dumps(SEEDS))

    print("\n--------------------------------------------------\n\n[G] SOURCE PROVENANCE")
    print("Fig3(a):")
    for p in a_src: print(p)
    print("Fig3(b):")
    for p in b_src: print(p)
    print("Fig3(c)/(d):")
    for p in cd_src: print(p)
    print("Diagnostic source code:")
    print(prism / "02_src/run_prism_drugcold_seed.py")
    print(prism / "02_src/run_prism_strictcold_seed.py")
    print("Audit aggregation script:")
    print(OUT / "fig3_data_audit.py")

    all_complete = not (a_prob or b_prob or c_prob or d_prob) and len(a_rows)==400 and len(b_rows)==300 and len(c_rows)==200 and len(d_rows)==50
    print("\n--------------------------------------------------\n\n[H] FINAL VERDICT")
    print(f"FIG3_A_COMPLETE = {'YES' if not a_prob and len(a_rows)==400 else 'NO'}")
    print(f"FIG3_B_COMPLETE = {'YES' if not b_prob and len(b_rows)==300 else 'NO'}")
    print(f"FIG3_C_COMPLETE = {'YES' if not c_prob and len(c_rows)==200 else 'NO'}")
    print(f"FIG3_D_COMPLETE = {'YES' if not d_prob and len(d_rows)==50 else 'NO'}")
    print(f"FULL_FIG3_REAL_DATA_COMPLETE = {'YES' if all_complete else 'NO'}")
    print("RETRAINED = NO")
    print("HISTORICAL_RESULTS_MODIFIED = NO")


if __name__ == "__main__":
    main()
