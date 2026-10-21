# Fig.4 真实原始数据审计

## 结论

只读取已有结果和逐 pair 候选身份，不训练、不启用 GPU、不改历史实验目录。
四个输出分别为 800、100、2150、4050 行；所有记录保留真实逐 seed / shuffle repeat / subset repeat，未用汇总均值模拟观测。
数值审计模式：numeric-audit；No-invention status: PASS。

## A. SQRC 压力测试

原始逐 seed：`/home/u2508283030/ECARF_Article2_20260911/14_sqrc_estimation_stress_20260920/03_outputs/formal25/ALL_SEED_CONDITIONS.csv`
条件汇总：`/home/u2508283030/ECARF_Article2_20260911/14_sqrc_estimation_stress_20260920/03_outputs/formal25/CONDITION_AGGREGATE.csv`
历史摘要：`/home/u2508283030/ECARF_Article2_20260911/14_sqrc_estimation_stress_20260920/03_outputs/formal25/FORMAL25_SUMMARY.json`

字段：原始 24 列完整保留；新添 `n` (=n_candidates)、`shrinkage_weight_w` (=mean_weight)、
`uncertainty_se2` (=test_mean_se2)、`mse_gain_qrc_minus_sqrc` (=qrc_bias_mse−sqrc_bias_mse)、
`relative_gain_qrc_minus_sqrc` (=gain/qrc_bias_mse)。这些是逐行确定性别名/算术，不是新实验观测。
其中 `mean_weight` 是测试查询的平均 shrinkage weight，不是每个候选的独立 w。
SE² 同时保留 `validation_mean_se2` 与 `test_mean_se2`；另保留名义量 `nominal_uncertainty_sigma2_over_n`。

覆盖：sigma = 0.25, 0.5, 1, 2；n = 5, 10, 25, 50, 65, 100, 379, 1195；seed = 0–24；
4 × 8 × 25 = 800，每个 sigma–n 格恰有 25 个 seed，无空值或重复键。
SQRC MSE 更低为 **563/800**，与历史摘要一致。

困难条件 sigma=2、n=5：QRC MSE 均值 **0.838862129**；SQRC MSE 均值 **0.525021573**；
平均 w **0.598001154**。重新计算的**逐 seed 相对改善率平均**为 **37.035378%**；
**均值之比** `(mean QRC−mean SQRC)/mean QRC` 为 **37.412650%**。
两者统计口径不同，历史图的 37.04% 对应前者，而不能直接由图上两个平均 MSE 相除得到。

## B. Davis DTI 正式 drug-cold

逐方法逐 seed：`/home/u2508283030/ECARF_Article2_20260911/DTI_QRC_FORMAL_20260918/results/per_seed_metrics.csv`
正式划分：`/home/u2508283030/ECARF_Article2_20260911/DTI_QRC_FORMAL_20260918/splits/davis_formal25_split_summary.csv`
正式汇总核验：`/home/u2508283030/ECARF_Article2_20260911/DTI_QRC_FORMAL_20260918/results/aggregate_metrics.csv`
同池 Top-K：`/home/u2508283030/ECARF_Article2_20260911/ESWA_supplementary_experiments/topk_per_seed.csv`

100 行 = 25 seeds (5–29) × Base/Raw/QRC/SQRC；正式 test split 的 n_pairs、n_positive、
n_negative、n_drugs 与每行合并，正例比例为 n_positive/n_pairs。
逐 seed 原始字段包括 pooled AUROC/AUPR、drug-macro AUROC/AUPR、Brier、ECE、
query-shift variance ratio、Raw/QRC 对 Base 的 cross-query inversion rate；
Top 1/2/5/10% 的 K、hits、precision、recall、enrichment factor 已按 seed+method 合并。
所有 Top-K 的 test_pairs/known_positives 与正式 test split 精确对上；不存在跨测试池拼接。
**cross-query AUROC 未在该正式逐 seed 文件中提供**；inversion rate 不可替代 cross-query AUROC。

| Method | Pooled AUROC (mean ± sample SD) | Pooled AUPR (mean ± sample SD) |
|---|---:|---:|
| Base | 0.6993 ± 0.0702 | 0.4936 ± 0.0901 |
| Raw | 0.7208 ± 0.0570 | 0.5116 ± 0.0881 |
| QRC | 0.7038 ± 0.0730 | 0.4968 ± 0.0921 |
| SQRC | 0.7039 ± 0.0728 | 0.4970 ± 0.0920 |

重算 mean、样本 SD (`ddof=1`) 与正式 aggregate_metrics.csv 的 pooled 与 drug-macro 四项逐项一致（容差 1e−12）。

## C. 反事实分解

原始逐次：`/home/u2508283030/ECARF_Article2_20260911/02_results/stageD_residual_counterfactual_full.tsv`
历史计算脚本（只读核对字段语义）：`/home/u2508283030/ECARF_Article2_20260911/01_src/stageD_residual_counterfactual.py`

2150 行 = drug-cold 与 miRNA-cold 各 1075 行；每协议 25 seeds，
Original Molecular(Base)、Raw、QRC 各 25 行，Bias_Shuffle 与 Residual_Shuffle_WithinQuery 各 25×20=500 行。
保留原始 `mode, seed, repeat, condition, pooled_AUROC, pooled_AUPR, macro_AUROC`；
新添协议名及原始/重排记录类型，未聚合 20 次 shuffle 复现。无空值、无重复复合键。

**缺失：macro_AUPR、逐查询 bias 数值、逐查询 residual 分布/统计**。
历史脚本在内存中分解 b 与 r，但此结果 TSV 只写出上述 3 个指标；不能用 pooled AUPR 代替 macro AUPR，
也不能从 AUROC/AUPR 汇总反推 b/r 的原始分布。若以后需该分布，必须另对保存的逐 pair 预测作只读派生并单独审计。

| Protocol | Condition | Pooled AUROC | Pooled AUPR | Macro AUROC |
|---|---|---:|---:|---:|
| drug-cold | Molecular | 0.5828 | 0.0577 | 0.6326 |
| drug-cold | Raw | 0.5359 | 0.0449 | 0.6923 |
| drug-cold | QRC | 0.5893 | 0.0641 | 0.6923 |
| drug-cold | Bias_Shuffle | 0.5772 | 0.0634 | 0.6923 |
| drug-cold | Residual_Shuffle_WithinQuery | 0.5275 | 0.0465 | 0.6171 |
| mirna-cold | Molecular | 0.9233 | 0.3394 | 0.9363 |
| mirna-cold | Raw | 0.9314 | 0.3649 | 0.9441 |
| mirna-cold | QRC | 0.9307 | 0.3681 | 0.9441 |
| mirna-cold | Bias_Shuffle | 0.9288 | 0.3570 | 0.9441 |
| mirna-cold | Residual_Shuffle_WithinQuery | 0.9078 | 0.2904 | 0.9213 |

这些只是对原始逐次记录的核对均值；shuffles 的 500 行是 25 seed × 20 重排，不能当作 500 个独立 seed。
Drug-cold Bias_Shuffle 约 0.5772 / 0.0634 / 0.6923，Residual_Shuffle 约 0.5275 / 0.0465 / 0.6171，
与历史 Fig.4(b) 指标口径一致。

## D. Candidate-set stability

逐次结果：`/home/u2508283030/ECARF_Article2_20260911/02_results/stageD_candidate_set_stability_full.tsv`
历史计算脚本（只读核对抽样规则）：`/home/u2508283030/ECARF_Article2_20260911/01_src/stageD_candidate_set_stability.py`

4050 行：drug-cold 与 miRNA-cold 各 2025；比例 0.1/0.25/0.5/0.75 均各 25 seed×20 repeats=500 行，
比例 1.0 各 25 seed×1=25 行。seed 0–24，无缺失/重复。
原始字段完整保留：AUROC/AUPR、FullQRC_AUROC/AUPR、
Delta_AUROC_vs_Full、Delta_AUPR_vs_Full、Bias_MAE。逐行重算配对差值，与历史 Delta 完全一致。

实际候选数量并未保存在汇总 TSV；本次只读扫描历史测试 pair 和 test_entities，
按原脚本的 `max(1, round(query_candidate_count × fraction))` 无损恢复，
附加 `actual_candidate_count`、min/max 与每 seed 的测试 query 数。没有重采样或重新计算任何预测。
候选数不统一的 seed–fraction 组合：**0**；若非零，单值列保持空白，min/max 列为真值范围。

实际每查询候选数在各 seed 内一致，逐源文件验证如下；比例到整数采用历史脚本的 Python `round` 规则：

| 协议 | 全候选 | 10% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| drug-cold | 1195 | 120 | 299 | 598 | 896 | 1195 |
| miRNA-cold | 65 | 6 | 16 | 32 | 49 | 65 |

drug-cold 10%：Bias MAE 均值 **0.004003**；
miRNA-cold 10%：Bias MAE 均值 **0.025496**。
这些均值用于对照论文图文，正式 CSV 仍保留每次真实复现。

### 用于恢复实际候选数的原始绝对路径

以下文件仅被流式读取以计数，没有修改：

- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed0/all_pair_scores_drug_cold_seed0.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed0/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed1/all_pair_scores_drug_cold_seed1.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed1/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed2/all_pair_scores_drug_cold_seed2.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed2/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed3/all_pair_scores_drug_cold_seed3.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed3/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed4/all_pair_scores_drug_cold_seed4.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed4/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed5/all_pair_scores_drug_cold_seed5.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed5/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed6/all_pair_scores_drug_cold_seed6.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed6/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed7/all_pair_scores_drug_cold_seed7.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed7/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed8/all_pair_scores_drug_cold_seed8.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed8/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed9/all_pair_scores_drug_cold_seed9.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed9/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed10/all_pair_scores_drug_cold_seed10.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed10/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed11/all_pair_scores_drug_cold_seed11.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed11/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed12/all_pair_scores_drug_cold_seed12.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed12/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed13/all_pair_scores_drug_cold_seed13.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed13/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed14/all_pair_scores_drug_cold_seed14.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed14/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed15/all_pair_scores_drug_cold_seed15.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed15/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed16/all_pair_scores_drug_cold_seed16.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed16/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed17/all_pair_scores_drug_cold_seed17.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed17/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed18/all_pair_scores_drug_cold_seed18.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed18/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed19/all_pair_scores_drug_cold_seed19.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed19/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed20/all_pair_scores_drug_cold_seed20.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed20/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed21/all_pair_scores_drug_cold_seed21.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed21/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed22/all_pair_scores_drug_cold_seed22.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed22/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed23/all_pair_scores_drug_cold_seed23.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed23/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/drug_cold_seed24/all_pair_scores_drug_cold_seed24.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/drug_cold_seed24/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed0/all_pair_scores_mirna_cold_seed0.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed0/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed1/all_pair_scores_mirna_cold_seed1.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed1/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed2/all_pair_scores_mirna_cold_seed2.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed2/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed3/all_pair_scores_mirna_cold_seed3.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed3/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed4/all_pair_scores_mirna_cold_seed4.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed4/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed5/all_pair_scores_mirna_cold_seed5.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed5/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed6/all_pair_scores_mirna_cold_seed6.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed6/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed7/all_pair_scores_mirna_cold_seed7.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed7/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed8/all_pair_scores_mirna_cold_seed8.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed8/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed9/all_pair_scores_mirna_cold_seed9.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed9/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed10/all_pair_scores_mirna_cold_seed10.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed10/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed11/all_pair_scores_mirna_cold_seed11.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed11/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed12/all_pair_scores_mirna_cold_seed12.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed12/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed13/all_pair_scores_mirna_cold_seed13.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed13/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed14/all_pair_scores_mirna_cold_seed14.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed14/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed15/all_pair_scores_mirna_cold_seed15.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed15/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed16/all_pair_scores_mirna_cold_seed16.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed16/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed17/all_pair_scores_mirna_cold_seed17.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed17/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed18/all_pair_scores_mirna_cold_seed18.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed18/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed19/all_pair_scores_mirna_cold_seed19.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed19/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed20/all_pair_scores_mirna_cold_seed20.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed20/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed21/all_pair_scores_mirna_cold_seed21.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed21/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed22/all_pair_scores_mirna_cold_seed22.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed22/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed23/all_pair_scores_mirna_cold_seed23.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed23/test_entities.tsv`
- `/home/u2508283030/MDA_CellPainting_Multiview_20260910/05_results/mirna_cold_seed24/all_pair_scores_mirna_cold_seed24.tsv`；`/home/u2508283030/MDA_CellPainting_Multiview_20260910/03_splits/mirna_cold_seed24/test_entities.tsv`

## 绘图可行性（本阶段不绘图）

| 图型 | 状态 | 真实数据支撑与限制 |
|---|---|---|
| Ridgeline 分布脊线图 | PASS | A 有 800 条逐 seed MSE/gain/uncertainty；C 的 shuffle 指标也有逐次值。需标明 20 repeats 嵌套于 seed。 |
| Uncertainty–Gain filled contour | PARTIAL | A 的 test_mean_se2 与 gain 有 800 个真实点，但 sigma×n 仅离散 4×8 条件；填色等高线需插值，禁止当作实测连续观测。可绘真实散点/离散热图。 |
| sigma × n × MSE Gain 二维响应面 | PASS | A 的 4×8 条件每格 25 seeds，可画离散热图/格点均值与离散度；不补未测参数。 |
| 三维响应曲面 | PARTIAL | 可以画 4×8 条件格点的 3D 网格/柱面，但任何光滑连续曲面都需插值，只能标为视觉化模型而非原始观测，本阶段不绘制。 |
| 配对分布图 | PASS | B 有 25 个完全配对的 Base/Raw/QRC/SQRC seed；D 有每次子采样与同 seed full-QRC 的真实差值。C shuffle 需按 seed 聚类，不可将 repeat 当独立 seed。 |

## 总审计

四份 CSV 的逐条记录均从真实现存文件提取；无 NaN、重复主键、跨协议拼接或根据 mean±SD 回填原始值。
Davis formal 的实验标签中 `negative` 是测试集构造标签；未知关联不能自动解释成实验确认的生物学阴性。
本阶段只提取与核查，不绘制 Fig.4、不启动 Slurm/GPU、不训练、不修改或删除历史结果。
