# ESWA 补充实验最终报告

## 计算资源与来源

- 90324：属当前账号；jsj 分区，jsj2 节点，RUNNING。原 EPyMARL 续跑步骤 90324.0 等待文件锁且 CPU 使用为零，已按用户指令仅终止该步骤；90324 主保活作业未取消。90780 未更改。
- 实际分析 job ID：90324；分区：jsj；节点：jsj2；CPU：1；GPU 实际使用：0。
- 冻结清单：`/home/u2508283030/ECARF_Article2_20260911/11_qrc_calibration_baselines_20260919/00_admin/FORMAL_FROZEN_INPUT_MANIFEST_125CASE.json`；SHA-256：`d735ff40332609fa91ecd2c8464b073f486c0eaf546c4e2300ab0826843fd9a7`。历史逐例 QC 与所有输入哈希均已核验。
- 历史基线脚本：`/home/u2508283030/ECARF_Article2_20260911/11_qrc_calibration_baselines_20260919/02_code/run_calibration_baselines.py`；历史指标：`/home/u2508283030/ECARF_Article2_20260911/11_qrc_calibration_baselines_20260919/03_outputs/formal/FORMAL_ALL_CASE_METRICS.csv`。没有重新训练、推断或按测试标签调参数。
- 完成 125 / 125 个 case；协议：DTI/drug_cold, MDA/drug_cold, MDA/mirna_cold, MDA/random, MDA/strict_cold。

## 评价口径与数据质量

- 每个 seed 的 7 个方法共用同一测试 pair、标签及查询划分。所有源文件哈希、历史指标、标签对齐、行数和 pair 唯一性均通过。
- Top K = ceil(测试 pair 总数 × 预算百分比)，全部 pair 统一排序；同分时用与标签无关的 pair 哈希打破平局。Hit@K 为已知正例个数；Precision=Hit/K；Recall=Hit/已知正例总数；EF=Precision/该测试池已知正例比例。
- MDA 的其余 pair 属于未标注关联，不等同于经生物实验验证的真实负例。因此 Precision、AUROC、EF 仅按冻结的已知正例对未标注池的评价口径解释。
- Cross-query AUROC 只计不同 query 的正例–其他类 pair；由 pooled Mann–Whitney U 扣除同 query 的 U 和 pair 数精确得到，含 0.5 tie credit。
- Bootstrap 对 25 个 seed 配对差值重采样 20,000 次，报告百分位 95% CI；双侧 Wilcoxon；同一协议和指标内对 4 个预定比较做 Holm 校正，全体检验另报 BH q 值。

## 实验 A：Global Top-K

### DTI / drug_cold

| 方法 | 1% Recall | 2% Recall | 5% Recall | 10% Recall |
|---|---:|---:|---:|---:|
| Base | 0.0257 ± 0.0076 | 0.0504 ± 0.0134 | 0.1175 ± 0.0272 | 0.2115 ± 0.0397 |
| Raw | 0.0264 ± 0.0090 | 0.0507 ± 0.0148 | 0.1178 ± 0.0276 | 0.2157 ± 0.0414 |
| QRC | 0.0258 ± 0.0090 | 0.0502 ± 0.0155 | 0.1166 ± 0.0295 | 0.2092 ± 0.0421 |
| SQRC | 0.0258 ± 0.0090 | 0.0502 ± 0.0154 | 0.1168 ± 0.0294 | 0.2092 ± 0.0420 |
| RawCenter | 0.0219 ± 0.0077 | 0.0429 ± 0.0135 | 0.1013 ± 0.0269 | 0.1911 ± 0.0398 |
| RawZScore | 0.0213 ± 0.0067 | 0.0411 ± 0.0116 | 0.1004 ± 0.0234 | 0.1870 ± 0.0335 |
| MedianResidual | 0.0261 ± 0.0089 | 0.0504 ± 0.0157 | 0.1164 ± 0.0291 | 0.2092 ± 0.0416 |

### MDA / drug_cold

| 方法 | 1% Recall | 2% Recall | 5% Recall | 10% Recall |
|---|---:|---:|---:|---:|
| Base | 0.0170 ± 0.0219 | 0.0325 ± 0.0376 | 0.0819 ± 0.0775 | 0.1418 ± 0.1289 |
| Raw | 0.0096 ± 0.0130 | 0.0192 ± 0.0280 | 0.0663 ± 0.0692 | 0.1114 ± 0.1093 |
| QRC | 0.0220 ± 0.0263 | 0.0378 ± 0.0453 | 0.0950 ± 0.0948 | 0.1489 ± 0.1470 |
| SQRC | 0.0220 ± 0.0263 | 0.0378 ± 0.0453 | 0.0950 ± 0.0948 | 0.1489 ± 0.1470 |
| RawCenter | 0.0484 ± 0.0482 | 0.0821 ± 0.0706 | 0.1421 ± 0.1023 | 0.2164 ± 0.1059 |
| RawZScore | 0.0403 ± 0.0383 | 0.0688 ± 0.0543 | 0.1333 ± 0.0967 | 0.2023 ± 0.0872 |
| MedianResidual | 0.0219 ± 0.0262 | 0.0385 ± 0.0453 | 0.0950 ± 0.0949 | 0.1478 ± 0.1478 |

### MDA / mirna_cold

| 方法 | 1% Recall | 2% Recall | 5% Recall | 10% Recall |
|---|---:|---:|---:|---:|
| Base | 0.0995 ± 0.0143 | 0.1920 ± 0.0170 | 0.4256 ± 0.0321 | 0.6866 ± 0.0280 |
| Raw | 0.1046 ± 0.0111 | 0.2034 ± 0.0215 | 0.4494 ± 0.0337 | 0.7170 ± 0.0246 |
| QRC | 0.1082 ± 0.0138 | 0.2028 ± 0.0201 | 0.4442 ± 0.0263 | 0.7152 ± 0.0262 |
| SQRC | 0.1088 ± 0.0127 | 0.2036 ± 0.0212 | 0.4477 ± 0.0268 | 0.7173 ± 0.0245 |
| RawCenter | 0.0921 ± 0.0094 | 0.1802 ± 0.0169 | 0.3913 ± 0.0257 | 0.6545 ± 0.0310 |
| RawZScore | 0.0986 ± 0.0119 | 0.1927 ± 0.0203 | 0.4377 ± 0.0319 | 0.7053 ± 0.0253 |
| MedianResidual | 0.1113 ± 0.0144 | 0.2052 ± 0.0213 | 0.4474 ± 0.0250 | 0.7140 ± 0.0230 |

### MDA / random

| 方法 | 1% Recall | 2% Recall | 5% Recall | 10% Recall |
|---|---:|---:|---:|---:|
| Base | 0.1999 ± 0.0302 | 0.3265 ± 0.0393 | 0.5729 ± 0.0347 | 0.7644 ± 0.0222 |
| Raw | 0.2122 ± 0.0330 | 0.3429 ± 0.0353 | 0.5903 ± 0.0303 | 0.7891 ± 0.0196 |
| QRC | 0.1978 ± 0.0328 | 0.3256 ± 0.0371 | 0.5820 ± 0.0326 | 0.7840 ± 0.0197 |
| SQRC | 0.1978 ± 0.0328 | 0.3258 ± 0.0370 | 0.5821 ± 0.0326 | 0.7840 ± 0.0197 |
| RawCenter | 0.0119 ± 0.0059 | 0.0174 ± 0.0079 | 0.0451 ± 0.0161 | 0.1048 ± 0.0232 |
| RawZScore | 0.0292 ± 0.0106 | 0.0549 ± 0.0162 | 0.1406 ± 0.0304 | 0.2513 ± 0.0345 |
| MedianResidual | 0.1978 ± 0.0325 | 0.3258 ± 0.0371 | 0.5816 ± 0.0319 | 0.7841 ± 0.0198 |

### MDA / strict_cold

| 方法 | 1% Recall | 2% Recall | 5% Recall | 10% Recall |
|---|---:|---:|---:|---:|
| Base | 0.0211 ± 0.0475 | 0.0300 ± 0.0537 | 0.0590 ± 0.0958 | 0.0977 ± 0.1492 |
| Raw | 0.0148 ± 0.0407 | 0.0217 ± 0.0476 | 0.0471 ± 0.0797 | 0.0764 ± 0.1398 |
| QRC | 0.0226 ± 0.0479 | 0.0421 ± 0.0700 | 0.0773 ± 0.1077 | 0.1269 ± 0.1862 |
| SQRC | 0.0226 ± 0.0479 | 0.0421 ± 0.0700 | 0.0773 ± 0.1077 | 0.1269 ± 0.1862 |
| RawCenter | 0.0118 ± 0.0193 | 0.0197 ± 0.0230 | 0.0654 ± 0.0617 | 0.1742 ± 0.1910 |
| RawZScore | 0.0131 ± 0.0192 | 0.0194 ± 0.0233 | 0.0627 ± 0.0613 | 0.1435 ± 0.0764 |
| MedianResidual | 0.0226 ± 0.0479 | 0.0440 ± 0.0712 | 0.0765 ± 0.1079 | 0.1250 ± 0.1847 |

完整 Hit、Precision、Recall、EF 的逐 seed 与均值/SD 分别见 `topk_per_seed.csv`、`topk_summary.csv`。

## 实验 B：配对统计

| 数据/协议 | 指标 | 对比 | 均值差 ± SD | 95% CI | 改善 seed | Wilcoxon p | Holm p | 全局 BH q |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DTI/drug_cold | cross_query_auroc | QRC - Raw | -0.0184 ± 0.0544 | [-0.0396, +0.0022] | 9/25 | 0.1485 | 0.2192 | 0.2504 |
| DTI/drug_cold | cross_query_auroc | QRC - Base | +0.0044 ± 0.0156 | [-0.0015, +0.0104] | 16/25 | 0.1096 | 0.2192 | 0.1967 |
| DTI/drug_cold | cross_query_auroc | QRC - RawCenter | +0.0359 ± 0.0829 | [+0.0033, +0.0673] | 18/25 | 0.03934 | 0.118 | 0.09179 |
| DTI/drug_cold | cross_query_auroc | SQRC - QRC | +0.0001 ± 0.0003 | [+0.0001, +0.0002] | 18/25 | 0.01145 | 0.04581 | 0.03818 |
| DTI/drug_cold | pooled_aupr | QRC - Raw | -0.0148 ± 0.0602 | [-0.0379, +0.0080] | 12/25 | 0.3254 | 0.6508 | 0.4466 |
| DTI/drug_cold | pooled_aupr | QRC - Base | +0.0032 ± 0.0325 | [-0.0088, +0.0163] | 11/25 | 0.751 | 0.751 | 0.8689 |
| DTI/drug_cold | pooled_aupr | QRC - RawCenter | +0.0521 ± 0.0909 | [+0.0177, +0.0877] | 16/25 | 0.01731 | 0.06925 | 0.05049 |
| DTI/drug_cold | pooled_aupr | SQRC - QRC | +0.0001 ± 0.0003 | [+0.0000, +0.0003] | 18/25 | 0.02748 | 0.08244 | 0.07536 |
| DTI/drug_cold | pooled_auroc | QRC - Raw | -0.0170 ± 0.0503 | [-0.0366, +0.0018] | 9/25 | 0.1485 | 0.2146 | 0.2504 |
| DTI/drug_cold | pooled_auroc | QRC - Base | +0.0045 ± 0.0155 | [-0.0016, +0.0105] | 17/25 | 0.1073 | 0.2146 | 0.1964 |
| DTI/drug_cold | pooled_auroc | QRC - RawCenter | +0.0336 ± 0.0771 | [+0.0036, +0.0632] | 18/25 | 0.03934 | 0.118 | 0.09179 |
| DTI/drug_cold | pooled_auroc | SQRC - QRC | +0.0001 ± 0.0002 | [+0.0000, +0.0002] | 18/25 | 0.01145 | 0.04581 | 0.03818 |
| DTI/drug_cold | recall_at_1pct | QRC - Raw | -0.0006 ± 0.0044 | [-0.0023, +0.0011] | 9/25 | 0.5755 | 1 | 0.6945 |
| DTI/drug_cold | recall_at_1pct | QRC - Base | +0.0001 ± 0.0071 | [-0.0026, +0.0029] | 10/25 | 0.8329 | 1 | 0.9403 |
| DTI/drug_cold | recall_at_1pct | QRC - RawCenter | +0.0039 ± 0.0084 | [+0.0007, +0.0072] | 16/25 | 0.03862 | 0.1545 | 0.09179 |
| DTI/drug_cold | recall_at_1pct | SQRC - QRC | +0.0000 ± 0.0002 | [+0.0000, +0.0001] | 1/25 | 0.3173 | 0.9519 | 0.4398 |
| DTI/drug_cold | recall_at_5pct | QRC - Raw | -0.0012 ± 0.0157 | [-0.0074, +0.0047] | 12/25 | 1 | 1 | 1 |
| DTI/drug_cold | recall_at_5pct | QRC - Base | -0.0009 ± 0.0173 | [-0.0072, +0.0057] | 9/25 | 0.4237 | 0.8474 | 0.5596 |
| DTI/drug_cold | recall_at_5pct | QRC - RawCenter | +0.0153 ± 0.0264 | [+0.0057, +0.0260] | 20/25 | 0.001844 | 0.007376 | 0.007593 |
| DTI/drug_cold | recall_at_5pct | SQRC - QRC | +0.0002 ± 0.0006 | [-0.0001, +0.0004] | 4/25 | 0.2489 | 0.7466 | 0.3746 |
| MDA/drug_cold | cross_query_auroc | QRC - Raw | +0.0629 ± 0.1520 | [+0.0064, +0.1229] | 11/25 | 0.1119 | 0.4477 | 0.1974 |
| MDA/drug_cold | cross_query_auroc | QRC - Base | +0.0012 ± 0.0247 | [-0.0087, +0.0102] | 12/25 | 0.372 | 0.7439 | 0.5054 |
| MDA/drug_cold | cross_query_auroc | QRC - RawCenter | -0.0175 ± 0.1815 | [-0.0866, +0.0519] | 15/25 | 0.7712 | 0.7712 | 0.8849 |
| MDA/drug_cold | cross_query_auroc | SQRC - QRC | -0.0000 ± 0.0001 | [-0.0001, +0.0000] | 6/25 | 0.1728 | 0.5185 | 0.2752 |
| MDA/drug_cold | pooled_aupr | QRC - Raw | +0.0192 ± 0.0408 | [+0.0050, +0.0362] | 12/25 | 0.05817 | 0.1543 | 0.1192 |
| MDA/drug_cold | pooled_aupr | QRC - Base | +0.0063 ± 0.0199 | [-0.0008, +0.0146] | 13/25 | 0.03858 | 0.1543 | 0.09179 |
| MDA/drug_cold | pooled_aupr | QRC - RawCenter | +0.0018 ± 0.0473 | [-0.0157, +0.0207] | 10/25 | 0.5782 | 0.5782 | 0.6945 |
| MDA/drug_cold | pooled_aupr | SQRC - QRC | -0.0000 ± 0.0000 | [-0.0000, -0.0000] | 5/25 | 0.04942 | 0.1543 | 0.107 |
| MDA/drug_cold | pooled_auroc | QRC - Raw | +0.0535 ± 0.1286 | [+0.0061, +0.1055] | 11/25 | 0.1024 | 0.4098 | 0.1912 |
| MDA/drug_cold | pooled_auroc | QRC - Base | +0.0066 ± 0.0260 | [-0.0036, +0.0163] | 13/25 | 0.133 | 0.4098 | 0.2298 |
| MDA/drug_cold | pooled_auroc | QRC - RawCenter | -0.0151 ± 0.1545 | [-0.0748, +0.0437] | 15/25 | 0.7915 | 0.7915 | 0.9009 |
| MDA/drug_cold | pooled_auroc | SQRC - QRC | -0.0000 ± 0.0001 | [-0.0000, +0.0000] | 6/25 | 0.1728 | 0.4098 | 0.2752 |
| MDA/drug_cold | recall_at_1pct | QRC - Raw | +0.0124 ± 0.0243 | [+0.0039, +0.0223] | 8/25 | 0.0166 | 0.06642 | 0.05049 |
| MDA/drug_cold | recall_at_1pct | QRC - Base | +0.0050 ± 0.0279 | [-0.0059, +0.0156] | 10/25 | 0.2146 | 0.4292 | 0.3338 |
| MDA/drug_cold | recall_at_1pct | QRC - RawCenter | -0.0265 ± 0.0527 | [-0.0482, -0.0071] | 7/25 | 0.02639 | 0.07917 | 0.07389 |
| MDA/drug_cold | recall_at_1pct | SQRC - QRC | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 0/25 | 1 | 1 | 1 |
| MDA/drug_cold | recall_at_5pct | QRC - Raw | +0.0287 ± 0.0817 | [+0.0004, +0.0625] | 9/25 | 0.1627 | 0.3829 | 0.2709 |
| MDA/drug_cold | recall_at_5pct | QRC - Base | +0.0130 ± 0.0441 | [-0.0039, +0.0299] | 12/25 | 0.1128 | 0.3829 | 0.1974 |
| MDA/drug_cold | recall_at_5pct | QRC - RawCenter | -0.0471 ± 0.1428 | [-0.1026, +0.0083] | 9/25 | 0.09573 | 0.3829 | 0.1811 |
| MDA/drug_cold | recall_at_5pct | SQRC - QRC | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 0/25 | 1 | 1 | 1 |
| MDA/mirna_cold | cross_query_auroc | QRC - Raw | -0.0006 ± 0.0009 | [-0.0010, -0.0003] | 7/25 | 0.002025 | 0.002025 | 0.007875 |
| MDA/mirna_cold | cross_query_auroc | QRC - Base | +0.0074 ± 0.0030 | [+0.0063, +0.0086] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/mirna_cold | cross_query_auroc | QRC - RawCenter | +0.0131 ± 0.0041 | [+0.0116, +0.0148] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/mirna_cold | cross_query_auroc | SQRC - QRC | +0.0006 ± 0.0005 | [+0.0005, +0.0008] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/mirna_cold | pooled_aupr | QRC - Raw | +0.0032 ± 0.0100 | [-0.0006, +0.0070] | 14/25 | 0.1645 | 0.1645 | 0.2709 |
| MDA/mirna_cold | pooled_aupr | QRC - Base | +0.0287 ± 0.0193 | [+0.0214, +0.0361] | 24/25 | 8.345e-07 | 2.503e-06 | 4.868e-06 |
| MDA/mirna_cold | pooled_aupr | QRC - RawCenter | +0.0543 ± 0.0205 | [+0.0466, +0.0625] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/mirna_cold | pooled_aupr | SQRC - QRC | +0.0028 ± 0.0038 | [+0.0014, +0.0042] | 21/25 | 0.0003291 | 0.0006582 | 0.00144 |
| MDA/mirna_cold | pooled_auroc | QRC - Raw | -0.0006 ± 0.0009 | [-0.0010, -0.0003] | 7/25 | 0.002025 | 0.002025 | 0.007875 |
| MDA/mirna_cold | pooled_auroc | QRC - Base | +0.0074 ± 0.0030 | [+0.0063, +0.0086] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/mirna_cold | pooled_auroc | QRC - RawCenter | +0.0130 ± 0.0040 | [+0.0115, +0.0146] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/mirna_cold | pooled_auroc | SQRC - QRC | +0.0006 ± 0.0005 | [+0.0005, +0.0008] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/mirna_cold | recall_at_1pct | QRC - Raw | +0.0037 ± 0.0100 | [-0.0002, +0.0075] | 13/25 | 0.07314 | 0.1463 | 0.1422 |
| MDA/mirna_cold | recall_at_1pct | QRC - Base | +0.0087 ± 0.0130 | [+0.0037, +0.0137] | 18/25 | 0.003903 | 0.01171 | 0.01436 |
| MDA/mirna_cold | recall_at_1pct | QRC - RawCenter | +0.0161 ± 0.0140 | [+0.0108, +0.0216] | 20/25 | 0.0001195 | 0.0004779 | 0.0005576 |
| MDA/mirna_cold | recall_at_1pct | SQRC - QRC | +0.0006 ± 0.0061 | [-0.0017, +0.0029] | 11/25 | 0.7151 | 0.7151 | 0.8343 |
| MDA/mirna_cold | recall_at_5pct | QRC - Raw | -0.0052 ± 0.0190 | [-0.0124, +0.0021] | 9/25 | 0.173 | 0.173 | 0.2752 |
| MDA/mirna_cold | recall_at_5pct | QRC - Base | +0.0187 ± 0.0215 | [+0.0105, +0.0268] | 20/25 | 0.0006068 | 0.00182 | 0.002574 |
| MDA/mirna_cold | recall_at_5pct | QRC - RawCenter | +0.0529 ± 0.0276 | [+0.0423, +0.0636] | 24/25 | 1.822e-05 | 7.286e-05 | 9.808e-05 |
| MDA/mirna_cold | recall_at_5pct | SQRC - QRC | +0.0035 ± 0.0075 | [+0.0007, +0.0064] | 14/25 | 0.03547 | 0.07093 | 0.09179 |
| MDA/random | cross_query_auroc | QRC - Raw | -0.0023 ± 0.0012 | [-0.0027, -0.0018] | 0/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/random | cross_query_auroc | QRC - Base | +0.0041 ± 0.0024 | [+0.0031, +0.0050] | 23/25 | 4.172e-07 | 4.172e-07 | 2.655e-06 |
| MDA/random | cross_query_auroc | QRC - RawCenter | +0.3038 ± 0.0162 | [+0.2978, +0.3102] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/random | cross_query_auroc | SQRC - QRC | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 24/25 | 1.788e-07 | 3.576e-07 | 1.252e-06 |
| MDA/random | pooled_aupr | QRC - Raw | -0.0109 ± 0.0145 | [-0.0172, -0.0062] | 1/25 | 1.788e-07 | 5.364e-07 | 1.252e-06 |
| MDA/random | pooled_aupr | QRC - Base | +0.0005 ± 0.0072 | [-0.0023, +0.0032] | 15/25 | 0.4742 | 0.4742 | 0.6147 |
| MDA/random | pooled_aupr | QRC - RawCenter | +0.1230 ± 0.0249 | [+0.1135, +0.1328] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/random | pooled_aupr | SQRC - QRC | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 22/25 | 8.345e-07 | 1.669e-06 | 4.868e-06 |
| MDA/random | pooled_auroc | QRC - Raw | -0.0023 ± 0.0012 | [-0.0027, -0.0018] | 0/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/random | pooled_auroc | QRC - Base | +0.0041 ± 0.0024 | [+0.0032, +0.0051] | 24/25 | 2.98e-07 | 3.576e-07 | 1.987e-06 |
| MDA/random | pooled_auroc | QRC - RawCenter | +0.3002 ± 0.0160 | [+0.2942, +0.3065] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/random | pooled_auroc | SQRC - QRC | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 24/25 | 1.788e-07 | 3.576e-07 | 1.252e-06 |
| MDA/random | recall_at_1pct | QRC - Raw | -0.0144 ± 0.0156 | [-0.0207, -0.0088] | 2/25 | 8.927e-05 | 0.0002678 | 0.0004309 |
| MDA/random | recall_at_1pct | QRC - Base | -0.0021 ± 0.0171 | [-0.0086, +0.0044] | 10/25 | 0.5804 | 1 | 0.6945 |
| MDA/random | recall_at_1pct | QRC - RawCenter | +0.1859 ± 0.0312 | [+0.1737, +0.1978] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/random | recall_at_1pct | SQRC - QRC | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 0/25 | 1 | 1 | 1 |
| MDA/random | recall_at_5pct | QRC - Raw | -0.0084 ± 0.0129 | [-0.0133, -0.0034] | 5/25 | 0.004 | 0.012 | 0.01436 |
| MDA/random | recall_at_5pct | QRC - Base | +0.0091 ± 0.0181 | [+0.0022, +0.0159] | 17/25 | 0.02054 | 0.04107 | 0.05867 |
| MDA/random | recall_at_5pct | QRC - RawCenter | +0.5369 ± 0.0362 | [+0.5235, +0.5512] | 25/25 | 5.96e-08 | 2.384e-07 | 4.909e-07 |
| MDA/random | recall_at_5pct | SQRC - QRC | +0.0001 ± 0.0006 | [+0.0000, +0.0003] | 1/25 | 0.3173 | 0.3173 | 0.4398 |
| MDA/strict_cold | cross_query_auroc | QRC - Raw | +0.0554 ± 0.1438 | [+0.0023, +0.1128] | 11/25 | 0.05347 | 0.2018 | 0.1117 |
| MDA/strict_cold | cross_query_auroc | QRC - Base | +0.0021 ± 0.0278 | [-0.0097, +0.0116] | 9/25 | 0.256 | 0.512 | 0.3813 |
| MDA/strict_cold | cross_query_auroc | QRC - RawCenter | -0.0497 ± 0.2519 | [-0.1514, +0.0448] | 12/25 | 0.4908 | 0.512 | 0.6273 |
| MDA/strict_cold | cross_query_auroc | SQRC - QRC | -0.0007 ± 0.0025 | [-0.0017, -0.0001] | 3/25 | 0.05046 | 0.2018 | 0.107 |
| MDA/strict_cold | pooled_aupr | QRC - Raw | +0.0199 ± 0.0429 | [+0.0053, +0.0381] | 11/25 | 0.0146 | 0.05839 | 0.04752 |
| MDA/strict_cold | pooled_aupr | QRC - Base | +0.0071 ± 0.0177 | [+0.0013, +0.0147] | 11/25 | 0.0356 | 0.1068 | 0.09179 |
| MDA/strict_cold | pooled_aupr | QRC - RawCenter | +0.0092 ± 0.0583 | [-0.0120, +0.0327] | 11/25 | 0.9578 | 0.9578 | 1 |
| MDA/strict_cold | pooled_aupr | SQRC - QRC | -0.0001 ± 0.0005 | [-0.0004, -0.0000] | 3/25 | 0.04139 | 0.1068 | 0.09499 |
| MDA/strict_cold | pooled_auroc | QRC - Raw | +0.0474 ± 0.1217 | [+0.0031, +0.0960] | 11/25 | 0.04683 | 0.1873 | 0.1041 |
| MDA/strict_cold | pooled_auroc | QRC - Base | +0.0037 ± 0.0312 | [-0.0089, +0.0151] | 9/25 | 0.2442 | 0.4884 | 0.3716 |
| MDA/strict_cold | pooled_auroc | QRC - RawCenter | -0.0417 ± 0.2128 | [-0.1256, +0.0388] | 12/25 | 0.4742 | 0.4884 | 0.6147 |
| MDA/strict_cold | pooled_auroc | SQRC - QRC | -0.0006 ± 0.0021 | [-0.0014, -0.0000] | 3/25 | 0.05046 | 0.1873 | 0.107 |
| MDA/strict_cold | recall_at_1pct | QRC - Raw | +0.0079 ± 0.0302 | [-0.0012, +0.0213] | 2/25 | 0.285 | 1 | 0.4101 |
| MDA/strict_cold | recall_at_1pct | QRC - Base | +0.0015 ± 0.0233 | [-0.0063, +0.0116] | 3/25 | 0.9165 | 1 | 1 |
| MDA/strict_cold | recall_at_1pct | QRC - RawCenter | +0.0109 ± 0.0555 | [-0.0081, +0.0341] | 6/25 | 0.5509 | 1 | 0.6766 |
| MDA/strict_cold | recall_at_1pct | SQRC - QRC | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 0/25 | 1 | 1 | 1 |
| MDA/strict_cold | recall_at_5pct | QRC - Raw | +0.0302 ± 0.0655 | [+0.0081, +0.0585] | 7/25 | 0.01729 | 0.06916 | 0.05049 |
| MDA/strict_cold | recall_at_5pct | QRC - Base | +0.0184 ± 0.0395 | [+0.0049, +0.0350] | 7/25 | 0.01729 | 0.06916 | 0.05049 |
| MDA/strict_cold | recall_at_5pct | QRC - RawCenter | +0.0119 ± 0.1124 | [-0.0297, +0.0571] | 8/25 | 0.9133 | 1 | 1 |
| MDA/strict_cold | recall_at_5pct | SQRC - QRC | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 0/25 | 1 | 1 | 1 |

其余 2%/10% Recall 的全部配对检验见 `paired_tests.csv`；仅以双侧校正后的 p 值与 CI 共同判断证据。

## 实验 C：跨查询判别

| 数据/协议 | 方法 | pooled AUROC | query-macro AUROC | cross-query AUROC |
|---|---|---:|---:|---:|
| DTI/drug_cold | Base | 0.6993 ± 0.0702 | 0.7157 ± 0.0543 | 0.6981 ± 0.0739 |
| DTI/drug_cold | MedianResidual | 0.7036 ± 0.0726 | 0.7200 ± 0.0550 | 0.7024 ± 0.0768 |
| DTI/drug_cold | QRC | 0.7038 ± 0.0730 | 0.7200 ± 0.0550 | 0.7025 ± 0.0772 |
| DTI/drug_cold | Raw | 0.7208 ± 0.0570 | 0.7200 ± 0.0550 | 0.7209 ± 0.0590 |
| DTI/drug_cold | RawCenter | 0.6702 ± 0.0500 | 0.7200 ± 0.0550 | 0.6666 ± 0.0499 |
| DTI/drug_cold | RawZScore | 0.6675 ± 0.0481 | 0.7200 ± 0.0550 | 0.6638 ± 0.0478 |
| DTI/drug_cold | SQRC | 0.7039 ± 0.0728 | 0.7200 ± 0.0550 | 0.7027 ± 0.0771 |
| MDA/drug_cold | Base | 0.5828 ± 0.1501 | 0.6326 ± 0.1016 | 0.5848 ± 0.1728 |
| MDA/drug_cold | MedianResidual | 0.5894 ± 0.1593 | 0.6923 ± 0.0938 | 0.5861 ± 0.1819 |
| MDA/drug_cold | QRC | 0.5893 ± 0.1591 | 0.6923 ± 0.0938 | 0.5860 ± 0.1816 |
| MDA/drug_cold | Raw | 0.5359 ± 0.1609 | 0.6923 ± 0.0938 | 0.5230 ± 0.1821 |
| MDA/drug_cold | RawCenter | 0.6045 ± 0.0924 | 0.6923 ± 0.0938 | 0.6035 ± 0.0928 |
| MDA/drug_cold | RawZScore | 0.6045 ± 0.0929 | 0.6923 ± 0.0938 | 0.6035 ± 0.0934 |
| MDA/drug_cold | SQRC | 0.5893 ± 0.1591 | 0.6923 ± 0.0938 | 0.5859 ± 0.1816 |
| MDA/mirna_cold | Base | 0.9233 ± 0.0072 | 0.9363 ± 0.0058 | 0.9233 ± 0.0072 |
| MDA/mirna_cold | MedianResidual | 0.9309 ± 0.0058 | 0.9441 ± 0.0047 | 0.9309 ± 0.0058 |
| MDA/mirna_cold | QRC | 0.9307 ± 0.0060 | 0.9441 ± 0.0047 | 0.9307 ± 0.0060 |
| MDA/mirna_cold | Raw | 0.9314 ± 0.0059 | 0.9441 ± 0.0047 | 0.9313 ± 0.0059 |
| MDA/mirna_cold | RawCenter | 0.9177 ± 0.0078 | 0.9441 ± 0.0047 | 0.9176 ± 0.0078 |
| MDA/mirna_cold | RawZScore | 0.9257 ± 0.0077 | 0.9441 ± 0.0047 | 0.9256 ± 0.0077 |
| MDA/mirna_cold | SQRC | 0.9314 ± 0.0058 | 0.9441 ± 0.0047 | 0.9313 ± 0.0058 |
| MDA/random | Base | 0.9286 ± 0.0057 | 0.6671 ± 0.0337 | 0.9320 ± 0.0056 |
| MDA/random | MedianResidual | 0.9327 ± 0.0051 | 0.6852 ± 0.0273 | 0.9360 ± 0.0050 |
| MDA/random | QRC | 0.9327 ± 0.0051 | 0.6852 ± 0.0273 | 0.9360 ± 0.0050 |
| MDA/random | Raw | 0.9350 ± 0.0056 | 0.6852 ± 0.0273 | 0.9383 ± 0.0055 |
| MDA/random | RawCenter | 0.6326 ± 0.0172 | 0.6852 ± 0.0273 | 0.6323 ± 0.0171 |
| MDA/random | RawZScore | 0.6549 ± 0.0231 | 0.6852 ± 0.0273 | 0.6548 ± 0.0230 |
| MDA/random | SQRC | 0.9327 ± 0.0051 | 0.6852 ± 0.0273 | 0.9360 ± 0.0050 |
| MDA/strict_cold | Base | 0.5401 ± 0.1721 | 0.5924 ± 0.1030 | 0.5334 ± 0.2052 |
| MDA/strict_cold | MedianResidual | 0.5438 ± 0.1811 | 0.6010 ± 0.1171 | 0.5355 ± 0.2134 |
| MDA/strict_cold | QRC | 0.5438 ± 0.1806 | 0.6010 ± 0.1171 | 0.5356 ± 0.2130 |
| MDA/strict_cold | Raw | 0.4964 ± 0.1609 | 0.6010 ± 0.1171 | 0.4801 ± 0.1886 |
| MDA/strict_cold | RawCenter | 0.5855 ± 0.1187 | 0.6010 ± 0.1171 | 0.5853 ± 0.1203 |
| MDA/strict_cold | RawZScore | 0.5831 ± 0.1146 | 0.6010 ± 0.1171 | 0.5824 ± 0.1153 |
| MDA/strict_cold | SQRC | 0.5433 ± 0.1805 | 0.6010 ± 0.1171 | 0.5349 ± 0.2129 |

逐 seed 跨查询结果与正负样本对计数见 `cross_query_per_seed.csv`；图见 `global_topk_recall.pdf` 和 `cross_query_auroc.pdf`。

## 结论与 ESWA 投稿判断

- DTI/drug_cold：QRC−RawCenter pooled AUROC +0.0336（Holm p=0.118），AUPR +0.0521（Holm p=0.06925）。
- MDA/drug_cold：QRC−RawCenter pooled AUROC -0.0151（Holm p=0.7915），AUPR +0.0018（Holm p=0.5782）。
- MDA/mirna_cold：QRC−RawCenter pooled AUROC +0.0130（Holm p=2.384e-07），AUPR +0.0543（Holm p=2.384e-07）。
- MDA/random：QRC−RawCenter pooled AUROC +0.3002（Holm p=2.384e-07），AUPR +0.1230（Holm p=2.384e-07）。
- MDA/strict_cold：QRC−RawCenter pooled AUROC -0.0417（Holm p=0.4884），AUPR +0.0092（Holm p=0.9578）。
- MDA drug-cold 5% 预算：QRC−Raw 的已知正例 Recall 差 +0.0287（95% CI [+0.0004, +0.0625]；Holm p=0.3829）。
- MDA strict-cold 5% 预算：QRC−Raw 的已知正例 Recall 差 +0.0302（95% CI [+0.0081, +0.0585]；Holm p=0.06916）。
- **QRC 相比 RawCenter：有条件优势。** MDA random 与 miRNA-cold 的 pooled AUROC/AUPR 差值为正且 Holm 校正后显著；MDA drug-cold、strict-cold 以及 Davis DTI 的对应比较没有显著优势，drug-cold 与 strict-cold 的 pooled AUROC 平均值甚至低于 RawCenter。
- **显著性：仅部分成立。** QRC 相比 RawCenter 在 random、miRNA-cold 的 pooled 指标有统计证据；QRC 相比 Raw 在 MDA drug-cold、strict-cold 的 1%、2%、5%、10% Top-K Recall 均未达到 Holm 校正后的 0.05 显著性，跨查询 AUROC 的同类比较也未显著。
- **Global Top-K：存在描述性收益，但尚不能声称在关键 cold 协议获得稳健收益。** 例如 MDA drug-cold、strict-cold 的 5% 预算 QRC 对 Raw 分别多找回平均 0.0287、0.0302 比例的已知正例；Holm p 分别为 0.3829、0.0692。MDA random 的 1%–10% 预算 QRC 反而显著低于 Raw。RawCenter 在 MDA drug-cold 的多个预算下均值高于 QRC。
- **Cross-query AUROC：机制变化确实可测。** MDA drug-cold 的 QRC/Raw 均值为 0.5860/0.5230，strict-cold 为 0.5356/0.4801；这两个正差值在 25 seeds 配对 Wilcoxon 的 Holm 校正后均不显著。DTI、MDA random 与 miRNA-cold 则没有同方向改善。
- **ESWA 投稿判断：这组新增实验不足以支持‘QRC 在所有协议或所有预算优于简单中心化/Raw’的表述。** 若按条件性方法撰写、明确 PU 评价限制和 Davis 边界，这些实验可作为投稿证据的一部分；仍需核对全文主张与所有表图，不能仅凭此保证审稿充分性。

## 尚未解决的科学问题

1. 未标注关联中可能含真实未知正例，缺少独立生物实验验证，不能将 PU 排序指标解释为真实发现率。
2. QRC 对 RawCenter、Raw 的优势随协议、指标和预算变化；需要解释查询位置有益与有害的边界。
3. 25 个 seed 共享底层数据，seed 间并非独立生物样本；配对 p 值不能替代外部队列验证。
4. Global Top-K 依赖统一预算和冻结候选池；真实应用的候选集变化可能改变收益。
