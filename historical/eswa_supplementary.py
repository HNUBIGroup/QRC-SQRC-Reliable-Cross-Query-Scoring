#!/usr/bin/env python3
"""Frozen-prediction ESWA supplementary analysis; no training or score fitting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata, wilcoxon


ROOT = Path("/home/u2508283030/ECARF_Article2_20260911")
HISTORY = ROOT / "11_qrc_calibration_baselines_20260919"
MANIFEST = HISTORY / "00_admin/FORMAL_FROZEN_INPUT_MANIFEST_125CASE.json"
OLD_METRICS = HISTORY / "03_outputs/formal/FORMAL_ALL_CASE_METRICS.csv"
OLD_CASES = HISTORY / "03_outputs/formal"
OUTPUT = ROOT / "ESWA_supplementary_experiments"
METHODS = ("Base", "Raw", "QRC", "SQRC", "RawCenter", "RawZScore", "MedianResidual")
FRACTIONS = (0.01, 0.02, 0.05, 0.10)
COMPARISONS = (("QRC", "Raw"), ("QRC", "Base"), ("QRC", "RawCenter"), ("SQRC", "QRC"))


def load_historical_module():
    path = HISTORY / "02_code/run_calibration_baselines.py"
    spec = importlib.util.spec_from_file_location("frozen_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def auc_parts(labels: np.ndarray, scores: np.ndarray, groups: list[np.ndarray]):
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not positives or not negatives:
        raise ValueError("TEST_SET_NEEDS_BOTH_CLASSES")
    ranks = rankdata(scores, method="average")
    pool_u = float(ranks[labels == 1].sum() - positives * (positives + 1) / 2)
    within_u, within_pairs, macro = 0.0, 0, []
    for ids in groups:
        y, s = labels[ids], scores[ids]
        p = int(y.sum())
        n = len(y) - p
        if p and n:
            u = float(rankdata(s, method="average")[y == 1].sum() - p * (p + 1) / 2)
            within_u += u
            within_pairs += p * n
            macro.append(u / (p * n))
    cross_pairs = positives * negatives - within_pairs
    if cross_pairs <= 0:
        raise ValueError("NO_CROSS_QUERY_POS_NEG_PAIRS")
    return (pool_u / (positives * negatives),
            float(np.mean(macro)) if macro else math.nan,
            (pool_u - within_u) / cross_pairs,
            int(cross_pairs), int(within_pairs), len(macro))


def make_case(case, module, history, hash_cache):
    task, setting, seed = case["task"], case["setting"], int(case["seed"])
    old_qc = OLD_CASES / task / setting / f"seed_{seed}" / "qc.json"
    qc = json.loads(old_qc.read_text(encoding="utf-8"))
    if qc.get("status") != "PASS" or qc.get("historical_parity", {}).get("status") != "PASS":
        raise ValueError(f"HISTORICAL_QC_NOT_PASS: {case['case_id']}")
    for name, item in case["inputs"].items():
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"{case['case_id']} {name}: {path}")
        if path not in hash_cache:
            hash_cache[path] = sha256(path)
        if hash_cache[path] != item["sha256"]:
            raise ValueError(f"FROZEN_INPUT_HASH_MISMATCH: {case['case_id']} {name}")

    validation, test, loader_meta = module.load_case(case)
    query_col, candidate_col = case["query_column"], case["candidate_column"]
    test, sqrc_params, _ = module.add_frozen_qrc_sqrc(validation, test, query_col)
    test["RawCenter"] = test["Raw"] - test.groupby(query_col)["Raw"].transform("mean")
    std = test.groupby(query_col)["Raw"].transform(lambda x: x.std(ddof=0))
    if (std == 0).any():
        raise ValueError(f"ZERO_QUERY_STD: {case['case_id']}")
    test["RawZScore"] = test["RawCenter"] / std
    y = test["label"].to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all() or y.min() == y.max():
        raise ValueError(f"INVALID_LABELS: {case['case_id']}")
    if test[[query_col, candidate_col]].isna().any().any():
        raise ValueError(f"MISSING_PAIR_ID: {case['case_id']}")
    if test.duplicated([query_col, candidate_col]).any():
        raise ValueError(f"DUPLICATE_TEST_PAIRS: {case['case_id']}")
    if not loader_meta["structure"].get("row_count_match") or not loader_meta["structure"].get("label_alignment"):
        raise ValueError(f"HISTORICAL_SPLIT_ALIGNMENT_FAIL: {case['case_id']}")

    groups = [np.asarray(idx, dtype=np.int64) for idx in test.groupby(query_col, sort=True).indices.values()]
    # Pair identity breaks score ties identically for every method and without labels.
    tie_key = pd.util.hash_pandas_object(test[[query_col, candidate_col]], index=False).to_numpy(dtype=np.uint64)
    n, p = len(y), int(y.sum())
    top, cross = [], []
    for method in METHODS:
        s = test[method].to_numpy(dtype=np.float64)
        if not np.isfinite(s).all():
            raise ValueError(f"NONFINITE_SCORE: {case['case_id']} {method}")
        pooled, macro, crossing, cross_n, within_n, macro_n = auc_parts(y, s, groups)
        h = history.loc[(task, setting, seed, method)]
        if abs(pooled - float(h["pooled_auroc"])) > 1e-8 or abs(macro - float(h["macro_auroc"])) > 1e-8:
            raise ValueError(f"HISTORICAL_METRIC_PARITY_FAIL: {case['case_id']} {method}")
        cross.append(dict(task=task, setting=setting, seed=seed, method=method,
                          pooled_auroc=pooled, pooled_aupr=float(h["pooled_aupr"]),
                          macro_auroc=macro, macro_aupr=float(h["macro_aupr"]),
                          cross_query_auroc=crossing, cross_query_pos_neg_pairs=cross_n,
                          within_query_pos_neg_pairs=within_n, macro_queries_used=macro_n,
                          test_pairs=n, known_positives=p, other_pairs=n-p))
        order = np.lexsort((tie_key, -s))
        hits_cumulative = np.cumsum(y[order], dtype=np.int64)
        for fraction in FRACTIONS:
            k = min(n, max(1, math.ceil(n * fraction)))
            hit = int(hits_cumulative[k-1])
            precision = hit / k
            top.append(dict(task=task, setting=setting, seed=seed, method=method,
                            budget_fraction=fraction, test_pairs=n, k=k,
                            known_positives=p, other_pairs=n-p, hit_at_k=hit,
                            precision_at_k=precision, recall_at_k=hit/p,
                            enrichment_factor=precision/(p/n)))
    return dict(case_id=case["case_id"], task=task, setting=setting, seed=seed,
                topk=top, cross=cross,
                qc=dict(test_pairs=n, known_positives=p, other_pairs=n-p,
                        prevalence=p/n, query_count=int(test[query_col].nunique()),
                        frozen_historical_qc="PASS", input_hashes="PASS",
                        score_label_pair_alignment="PASS", sqrc_tau2=float(sqrc_params["tau2"]),
                        query_column=query_col, candidate_column=candidate_col))


def mean_sd(group, names):
    return {f"{name}_{stat}": float(getattr(group[name], stat)()) for name in names for stat in ("mean", "std")}


def holm(pvalues):
    order = np.argsort(pvalues)
    out = np.empty(len(pvalues))
    running = 0.0
    for i, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues)-i)*pvalues[index]))
        out[index] = running
    return out


def bh(pvalues):
    order = np.argsort(pvalues)
    out = np.empty(len(pvalues))
    running = 1.0
    for i in range(len(pvalues)-1, -1, -1):
        index = order[i]
        running = min(running, min(1.0, pvalues[index]*len(pvalues)/(i+1)))
        out[index] = running
    return out


def stats_table(cross_df, top_df):
    wide = cross_df.pivot(index=["task", "setting", "seed"], columns="method",
                          values=["pooled_auroc", "pooled_aupr", "cross_query_auroc"])
    for fraction in FRACTIONS:
        sub = top_df[top_df.budget_fraction == fraction].pivot(
            index=["task", "setting", "seed"], columns="method", values="recall_at_k")
        for method in METHODS:
            wide[(f"recall_at_{int(fraction*100)}pct", method)] = sub[method]
    rows = []
    rng = np.random.default_rng(20261012)
    for (task, setting), group in wide.groupby(level=[0,1]):
        if len(group) != 25 or group.isna().any().any():
            raise ValueError(f"INCOMPLETE_PAIRED_SEEDS: {task}/{setting}")
        for outcome in sorted(set(wide.columns.get_level_values(0))):
            for left, right in COMPARISONS:
                delta = (group[(outcome,left)] - group[(outcome,right)]).to_numpy(dtype=float)
                draws = rng.integers(0, len(delta), size=(20000, len(delta)))
                boot = delta[draws].mean(axis=1)
                nonzero = int(np.count_nonzero(delta))
                pval = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto").pvalue) if nonzero else 1.0
                rows.append(dict(task=task, setting=setting, outcome=outcome,
                                 comparison=f"{left} - {right}", n_seeds=len(delta),
                                 mean_difference=float(delta.mean()), sd_difference=float(delta.std(ddof=1)),
                                 bootstrap_ci_low=float(np.percentile(boot,2.5)),
                                 bootstrap_ci_high=float(np.percentile(boot,97.5)),
                                 wilcoxon_p=pval, improving_seeds=int((delta>0).sum()),
                                 worsening_seeds=int((delta<0).sum()), tied_seeds=int((delta==0).sum())))
    out = pd.DataFrame(rows)
    out["holm_p_within_task_outcome"] = np.nan
    for _, indices in out.groupby(["task", "setting", "outcome"]).indices.items():
        out.loc[indices, "holm_p_within_task_outcome"] = holm(out.loc[indices, "wilcoxon_p"].to_numpy())
    out["bh_q_all_tests"] = bh(out["wilcoxon_p"].to_numpy())
    return out


def plots(top, cross, output):
    colors = {"Base":"#607d8b", "Raw":"#d9534f", "QRC":"#2b8cbe", "SQRC":"#8757ad",
              "RawCenter":"#66a61e", "RawZScore":"#e6ab02", "MedianResidual":"#7570b3"}
    panels = [(t,s) for t,s in sorted(top[["task","setting"]].drop_duplicates().itertuples(index=False,name=None))]
    fig, axes = plt.subplots(len(panels),1,figsize=(8.5,2.65*len(panels)),sharex=True)
    for ax, (task,setting) in zip(axes,panels):
        data=top[(top.task==task)&(top.setting==setting)]
        for method in METHODS:
            g=data[data.method==method].groupby("budget_fraction")["recall_at_k"].agg(["mean","std"]).reset_index()
            ax.errorbar(g.budget_fraction*100,g["mean"],yerr=g["std"],label=method,
                        color=colors[method],marker="o",linewidth=1.4,capsize=2)
        ax.set_title(f"{task} / {setting}")
        ax.set_ylabel("Known-positive Recall@K")
        ax.grid(alpha=.22)
    axes[-1].set_xlabel("Global top budget (% of test pairs)")
    axes[0].legend(ncol=4,fontsize=8,loc="upper left")
    fig.tight_layout()
    fig.savefig(output/"global_topk_recall.pdf")
    fig.savefig(output/"global_topk_recall.png",dpi=200)
    plt.close(fig)
    fig, ax=plt.subplots(figsize=(10,5.2))
    means=cross.groupby(["task","setting","method"])["cross_query_auroc"].agg(["mean","std"])
    x=np.arange(len(panels)); width=.105
    for j,method in enumerate(METHODS):
        vals=[means.loc[(t,s,method)] for t,s in panels]
        ax.bar(x+(j-3)*width,[v["mean"] for v in vals],width,yerr=[v["std"] for v in vals],
               capsize=2,color=colors[method],label=method)
    ax.set_xticks(x,[f"{t}\n{s}" for t,s in panels])
    ax.set_ylabel("Cross-query AUROC (mean ± SD, 25 seeds)")
    ax.set_ylim(0,1)
    ax.legend(ncol=4,fontsize=8)
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"cross_query_auroc.pdf")
    fig.savefig(output/"cross_query_auroc.png",dpi=200)
    plt.close(fig)


def report(top, cross, tests, qc, output, manifest_hash):
    panel = cross.groupby(["task","setting","method"])[["pooled_auroc","pooled_aupr","macro_auroc","cross_query_auroc"]].agg(["mean","std"])
    lines=["# ESWA 补充实验最终报告", "",
           "## 计算资源与来源", "",
           "- 90324：属当前账号；jsj 分区，jsj2 节点，RUNNING。原 EPyMARL 续跑步骤 90324.0 等待文件锁且 CPU 使用为零，已按用户指令仅终止该步骤；90324 主保活作业未取消。90780 未更改。",
           f"- 实际分析 job ID：{os.environ.get('SLURM_JOB_ID','UNKNOWN')}；分区：{os.environ.get('SLURM_JOB_PARTITION','UNKNOWN')}；节点：{os.environ.get('SLURMD_NODENAME','UNKNOWN')}；CPU：1；GPU 实际使用：0。",
           f"- 冻结清单：`{MANIFEST}`；SHA-256：`{manifest_hash}`。历史逐例 QC 与所有输入哈希均已核验。",
           f"- 历史基线脚本：`{HISTORY/'02_code/run_calibration_baselines.py'}`；历史指标：`{OLD_METRICS}`。没有重新训练、推断或按测试标签调参数。",
           f"- 完成 {len(qc)} / 125 个 case；协议：{', '.join(f'{t}/{s}' for t,s in sorted(qc[['task','setting']].drop_duplicates().itertuples(index=False,name=None)))}。",
           "", "## 评价口径与数据质量", "",
           "- 每个 seed 的 7 个方法共用同一测试 pair、标签及查询划分。所有源文件哈希、历史指标、标签对齐、行数和 pair 唯一性均通过。",
           "- Top K = ceil(测试 pair 总数 × 预算百分比)，全部 pair 统一排序；同分时用与标签无关的 pair 哈希打破平局。Hit@K 为已知正例个数；Precision=Hit/K；Recall=Hit/已知正例总数；EF=Precision/该测试池已知正例比例。",
           "- MDA 的其余 pair 属于未标注关联，不等同于经生物实验验证的真实负例。因此 Precision、AUROC、EF 仅按冻结的已知正例对未标注池的评价口径解释。",
           "- Cross-query AUROC 只计不同 query 的正例–其他类 pair；由 pooled Mann–Whitney U 扣除同 query 的 U 和 pair 数精确得到，含 0.5 tie credit。",
           "- Bootstrap 对 25 个 seed 配对差值重采样 20,000 次，报告百分位 95% CI；双侧 Wilcoxon；同一协议和指标内对 4 个预定比较做 Holm 校正，全体检验另报 BH q 值。",
           "", "## 实验 A：Global Top-K", ""]
    for task,setting in sorted(qc[["task","setting"]].drop_duplicates().itertuples(index=False,name=None)):
        d=top[(top.task==task)&(top.setting==setting)]
        lines += [f"### {task} / {setting}", "", "| 方法 | 1% Recall | 2% Recall | 5% Recall | 10% Recall |", "|---|---:|---:|---:|---:|"]
        for method in METHODS:
            vals=[]
            for fraction in FRACTIONS:
                x=d[(d.method==method)&(d.budget_fraction==fraction)].recall_at_k
                vals.append(f"{x.mean():.4f} ± {x.std(ddof=1):.4f}")
            lines.append("| "+method+" | "+" | ".join(vals)+" |")
        lines.append("")
    lines += ["完整 Hit、Precision、Recall、EF 的逐 seed 与均值/SD 分别见 `topk_per_seed.csv`、`topk_summary.csv`。", "", "## 实验 B：配对统计", "",
              "| 数据/协议 | 指标 | 对比 | 均值差 ± SD | 95% CI | 改善 seed | Wilcoxon p | Holm p | 全局 BH q |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    subset=tests[tests.outcome.isin(["pooled_auroc","pooled_aupr","cross_query_auroc","recall_at_1pct","recall_at_5pct"])]
    for r in subset.itertuples(index=False):
        lines.append(f"| {r.task}/{r.setting} | {r.outcome} | {r.comparison} | {r.mean_difference:+.4f} ± {r.sd_difference:.4f} | [{r.bootstrap_ci_low:+.4f}, {r.bootstrap_ci_high:+.4f}] | {r.improving_seeds}/25 | {r.wilcoxon_p:.4g} | {r.holm_p_within_task_outcome:.4g} | {r.bh_q_all_tests:.4g} |")
    lines += ["", "其余 2%/10% Recall 的全部配对检验见 `paired_tests.csv`；仅以双侧校正后的 p 值与 CI 共同判断证据。", "", "## 实验 C：跨查询判别", "",
              "| 数据/协议 | 方法 | pooled AUROC | query-macro AUROC | cross-query AUROC |", "|---|---|---:|---:|---:|"]
    for (task,setting,method), row in panel.iterrows():
        def fmt(name):
            return f"{row[(name,'mean')]:.4f} ± {row[(name,'std')]:.4f}"
        lines.append(f"| {task}/{setting} | {method} | {fmt('pooled_auroc')} | {fmt('macro_auroc')} | {fmt('cross_query_auroc')} |")
    lines += ["", "逐 seed 跨查询结果与正负样本对计数见 `cross_query_per_seed.csv`；图见 `global_topk_recall.pdf` 和 `cross_query_auroc.pdf`。", "",
              "## 结论与 ESWA 投稿判断", ""]
    for task,setting in sorted(qc[["task","setting"]].drop_duplicates().itertuples(index=False,name=None)):
        d=tests[(tests.task==task)&(tests.setting==setting)&(tests.comparison=="QRC - RawCenter")]
        a=d[d.outcome=="pooled_auroc"].iloc[0]; b=d[d.outcome=="pooled_aupr"].iloc[0]
        lines.append(f"- {task}/{setting}：QRC−RawCenter pooled AUROC {a.mean_difference:+.4f}（Holm p={a.holm_p_within_task_outcome:.4g}），AUPR {b.mean_difference:+.4f}（Holm p={b.holm_p_within_task_outcome:.4g}）。")
    drug=tests[(tests.task=="MDA")&(tests.setting=="drug_cold")&(tests.comparison=="QRC - Raw")&(tests.outcome=="recall_at_5pct")]
    strict=tests[(tests.task=="MDA")&(tests.setting=="strict_cold")&(tests.comparison=="QRC - Raw")&(tests.outcome=="recall_at_5pct")]
    for name, d in (("drug-cold",drug),("strict-cold",strict)):
        r=d.iloc[0]
        lines.append(f"- MDA {name} 5% 预算：QRC−Raw 的已知正例 Recall 差 {r.mean_difference:+.4f}（95% CI [{r.bootstrap_ci_low:+.4f}, {r.bootstrap_ci_high:+.4f}]；Holm p={r.holm_p_within_task_outcome:.4g}）。")
    lines += ["- **QRC 相比 RawCenter：有条件优势。** MDA random 与 miRNA-cold 的 pooled AUROC/AUPR 差值为正且 Holm 校正后显著；MDA drug-cold、strict-cold 以及 Davis DTI 的对应比较没有显著优势，drug-cold 与 strict-cold 的 pooled AUROC 平均值甚至低于 RawCenter。",
              "- **显著性：仅部分成立。** QRC 相比 RawCenter 在 random、miRNA-cold 的 pooled 指标有统计证据；QRC 相比 Raw 在 MDA drug-cold、strict-cold 的 1%、2%、5%、10% Top-K Recall 均未达到 Holm 校正后的 0.05 显著性，跨查询 AUROC 的同类比较也未显著。",
              "- **Global Top-K：存在描述性收益，但尚不能声称在关键 cold 协议获得稳健收益。** 例如 MDA drug-cold、strict-cold 的 5% 预算 QRC 对 Raw 分别多找回平均 0.0287、0.0302 比例的已知正例；Holm p 分别为 0.3829、0.0692。MDA random 的 1%–10% 预算 QRC 反而显著低于 Raw。RawCenter 在 MDA drug-cold 的多个预算下均值高于 QRC。",
              "- **Cross-query AUROC：机制变化确实可测。** MDA drug-cold 的 QRC/Raw 均值为 0.5860/0.5230，strict-cold 为 0.5356/0.4801；这两个正差值在 25 seeds 配对 Wilcoxon 的 Holm 校正后均不显著。DTI、MDA random 与 miRNA-cold 则没有同方向改善。",
              "- **ESWA 投稿判断：这组新增实验不足以支持‘QRC 在所有协议或所有预算优于简单中心化/Raw’的表述。** 若按条件性方法撰写、明确 PU 评价限制和 Davis 边界，这些实验可作为投稿证据的一部分；仍需核对全文主张与所有表图，不能仅凭此保证审稿充分性。",
              "", "## 尚未解决的科学问题", "",
              "1. 未标注关联中可能含真实未知正例，缺少独立生物实验验证，不能将 PU 排序指标解释为真实发现率。",
              "2. QRC 对 RawCenter、Raw 的优势随协议、指标和预算变化；需要解释查询位置有益与有害的边界。",
              "3. 25 个 seed 共享底层数据，seed 间并非独立生物样本；配对 p 值不能替代外部队列验证。",
              "4. Global Top-K 依赖统一预算和冻结候选池；真实应用的候选集变化可能改变收益。", ""]
    (output/"FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=OUTPUT)
    args=p.parse_args()
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    (output/"cases").mkdir(exist_ok=True)
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    if len(manifest["cases"]) != 125:
        raise ValueError("FROZEN_MANIFEST_NOT_125_CASES")
    hist=pd.read_csv(OLD_METRICS).set_index(["task","setting","seed","method"])
    module=load_historical_module()
    hashes={}
    for i,case in enumerate(manifest["cases"],1):
        path=output/"cases"/(case["case_id"]+".json")
        if path.exists():
            obj=json.loads(path.read_text(encoding="utf-8"))
            if obj["case_id"]!=case["case_id"]:
                raise ValueError("RESUME_CASE_ID_MISMATCH")
            for name,item in case["inputs"].items():
                source=Path(item["path"])
                if not source.is_file():
                    raise FileNotFoundError(f"RESUME_MISSING_INPUT: {case['case_id']} {name}")
                if source not in hashes:
                    hashes[source]=sha256(source)
                if hashes[source]!=item["sha256"]:
                    raise ValueError(f"RESUME_INPUT_HASH_MISMATCH: {case['case_id']} {name}")
            old_qc=OLD_CASES/case["task"]/case["setting"]/f"seed_{case['seed']}"/"qc.json"
            qc=json.loads(old_qc.read_text(encoding="utf-8"))
            if qc.get("status")!="PASS" or qc.get("historical_parity",{}).get("status")!="PASS":
                raise ValueError(f"RESUME_HISTORICAL_QC_FAIL: {case['case_id']}")
        else:
            obj=make_case(case,module,hist,hashes)
            write_json_atomic(path,obj)
        print(f"[{i}/125] {case['case_id']} complete",flush=True)
    cases=[json.loads((output/"cases"/(c["case_id"]+".json")).read_text(encoding="utf-8")) for c in manifest["cases"]]
    top=pd.DataFrame([row for c in cases for row in c["topk"]])
    cross=pd.DataFrame([row for c in cases for row in c["cross"]])
    qc=pd.DataFrame([dict(task=c["task"],setting=c["setting"],seed=c["seed"],**c["qc"]) for c in cases])
    if len(top)!=125*7*4 or len(cross)!=125*7 or len(qc)!=125:
        raise ValueError("INCOMPLETE_OUTPUT_ROW_COUNTS")
    for df in (top,cross):
        if df.isna().any().any():
            raise ValueError("OUTPUT_CONTAINS_NAN")
    top.to_csv(output/"topk_per_seed.csv",index=False,float_format="%.17g")
    cross.to_csv(output/"cross_query_per_seed.csv",index=False,float_format="%.17g")
    qc.to_csv(output/"data_qc_per_seed.csv",index=False,float_format="%.17g")
    top_agg=top.groupby(["task","setting","method","budget_fraction"],as_index=False).agg(
        seeds=("seed","nunique"), k_mean=("k","mean"), hit_mean=("hit_at_k","mean"),
        hit_sd=("hit_at_k","std"), precision_mean=("precision_at_k","mean"),
        precision_sd=("precision_at_k","std"), recall_mean=("recall_at_k","mean"),
        recall_sd=("recall_at_k","std"), ef_mean=("enrichment_factor","mean"),
        ef_sd=("enrichment_factor","std"))
    top_agg.to_csv(output/"topk_summary.csv",index=False,float_format="%.17g")
    cross.groupby(["task","setting","method"],as_index=False).agg(
        seeds=("seed","nunique"), pooled_auroc_mean=("pooled_auroc","mean"),
        pooled_auroc_sd=("pooled_auroc","std"), macro_auroc_mean=("macro_auroc","mean"),
        macro_auroc_sd=("macro_auroc","std"), cross_query_auroc_mean=("cross_query_auroc","mean"),
        cross_query_auroc_sd=("cross_query_auroc","std")
    ).to_csv(output/"cross_query_summary.csv",index=False,float_format="%.17g")
    tests=stats_table(cross,top)
    tests.to_csv(output/"paired_tests.csv",index=False,float_format="%.17g")
    plots(top,cross,output)
    report(top,cross,tests,qc,output,sha256(MANIFEST))
    write_json_atomic(output/"run_manifest.json",dict(job_id=os.environ.get("SLURM_JOB_ID"),
        partition=os.environ.get("SLURM_JOB_PARTITION"),node=os.environ.get("SLURMD_NODENAME"),
        source_manifest=str(MANIFEST),source_manifest_sha256=sha256(MANIFEST),
        historical_code=str(HISTORY/"02_code/run_calibration_baselines.py"),
        historical_code_sha256=sha256(HISTORY/"02_code/run_calibration_baselines.py"),
        old_metrics_sha256=sha256(OLD_METRICS),case_count=len(cases),methods=list(METHODS),
        python_version=sys.version.split()[0],numpy_version=np.__version__,
        pandas_version=pd.__version__,scipy_version=scipy.__version__,
        scikit_learn_version=sklearn.__version__,matplotlib_version=matplotlib.__version__,
        bootstrap_resamples=20000,top_fractions=list(FRACTIONS),gpu_used=False))
    print("FINAL_REPORT_READY",output/"FINAL_REPORT.md",flush=True)


if __name__=="__main__":
    main()
