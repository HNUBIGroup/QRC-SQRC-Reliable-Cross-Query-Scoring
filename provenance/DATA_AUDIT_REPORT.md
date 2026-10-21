# QRC/SQRC ESWA figure-data audit

All exported rows originate from existing project files. No model inference, training, smoothing, interpolation, or value imputation was performed. SD below is sample SD (ddof=1).

## 1. Seed-level AUROC/AUPR

- Output: `/home/u2508283030/QRC_ESWA_FigureData/seed_metrics.csv`; rows: 100; expected 4 × 25 = 100.
- Grain: one (setting, seed) record. `Molecular` in the historical files is exported as `base`; all AUROC/AUPR columns are **pooled**, not query-macro.
- Sources:
  - `/home/u2508283030/ECARF_Article2_20260911/02_results/stageE_shrinkage_qrc_25seed.tsv` (drug-cold and miRNA-cold)
  - `/home/u2508283030/ECARF_Article2_20260911/02_results/stageG_random_frozen_current_25seed.tsv` (random)
  - `/home/u2508283030/ECARF_Article2_20260911/02_results/stageG_v2_random_strict_final_25seed.tsv` (strict-cold only)
  - Cross-check: `/home/u2508283030/ECARF_Article2_20260911/02_results/stageI_simple_calibration_baselines_25seed.tsv` (same four settings, per-seed max absolute difference 0)
  - Historical Fig. 3 audit comparison: `/home/u2508283030/ECARF_Article2_20260911/_fig3_data_audit/fig3a_cellpainting_summary.tsv` (64 mean/SD comparisons, max absolute difference 0).
- Fields: `setting`, `seed`, Base/Raw/QRC/SQRC pooled AUROC and AUPR, `source_file`.
- Missing numeric values: {}; nonfinite values: {}; duplicate (setting, seed): 0.
- Seed coverage:
  - random: 25 rows; seeds 0–24; missing []; duplicate seeds 0.
  - mirna_cold: 25 rows; seeds 0–24; missing []; duplicate seeds 0.
  - drug_cold: 25 rows; seeds 0–24; missing []; duplicate seeds 0.
  - strict_cold: 25 rows; seeds 0–24; missing []; duplicate seeds 0.

| Setting | Method | AUROC mean ± SD | AUPR mean ± SD |
|---|---|---:|---:|
| random | base | 0.9286 ± 0.0057 | 0.1358 ± 0.0248 |
| random | raw | 0.9350 ± 0.0055 | 0.1472 ± 0.0274 |
| random | qrc | 0.9327 ± 0.0051 | 0.1362 ± 0.0256 |
| random | sqrc | 0.9328 ± 0.0051 | 0.1363 ± 0.0256 |
| mirna_cold | base | 0.9233 ± 0.0072 | 0.3394 ± 0.0191 |
| mirna_cold | raw | 0.9314 ± 0.0059 | 0.3649 ± 0.0222 |
| mirna_cold | qrc | 0.9307 ± 0.0060 | 0.3681 ± 0.0254 |
| mirna_cold | sqrc | 0.9314 ± 0.0058 | 0.3708 ± 0.0265 |
| drug_cold | base | 0.5828 ± 0.1501 | 0.0577 ± 0.0429 |
| drug_cold | raw | 0.5359 ± 0.1609 | 0.0449 ± 0.0361 |
| drug_cold | qrc | 0.5893 ± 0.1591 | 0.0641 ± 0.0537 |
| drug_cold | sqrc | 0.5893 ± 0.1591 | 0.0641 ± 0.0537 |
| strict_cold | base | 0.5401 ± 0.1721 | 0.0627 ± 0.0512 |
| strict_cold | raw | 0.4964 ± 0.1609 | 0.0499 ± 0.0420 |
| strict_cold | qrc | 0.5438 ± 0.1806 | 0.0698 ± 0.0618 |
| strict_cold | sqrc | 0.5433 ± 0.1805 | 0.0696 ± 0.0618 |

**Plot support:** raincloud/violin plots and paired differences can use the 25 original seed values. Confidence intervals require an explicitly stated resampling or interval method; no interval has been precomputed or invented here.

## 2. Drug-cold pair scores

- Output: `/home/u2508283030/QRC_ESWA_FigureData/drug_pair_scores.csv.gz`; rows: 179250; source files: 25/25 score files and 25/25 query-diagnostic files.
- Grain: one (seed, drug, miRNA) test pair. Base/Raw/QRC/SQRC values coexist in the same source row, so method-wise pair/label alignment is exact within a seed.
- Fields: `seed`, `drug`, `miRNA`, `label`, `base_score`, `raw_score`, `delta`, `query_bias`, `qrc_score`, `sqrc_weight`, `sqrc_score`; scores and IDs are copied unchanged.
- Label counts: {'0': 171803, '1': 7447}. Historical 0 labels mean unlabeled/PU comparison records, not experimentally verified biological negatives.
- Rows by seed (seed, pair rows, drug diagnostics): [(0, 7170, 6), (1, 7170, 6), (2, 7170, 6), (3, 7170, 6), (4, 7170, 6), (5, 7170, 6), (6, 7170, 6), (7, 7170, 6), (8, 7170, 6), (9, 7170, 6), (10, 7170, 6), (11, 7170, 6), (12, 7170, 6), (13, 7170, 6), (14, 7170, 6), (15, 7170, 6), (16, 7170, 6), (17, 7170, 6), (18, 7170, 6), (19, 7170, 6), (20, 7170, 6), (21, 7170, 6), (22, 7170, 6), (23, 7170, 6), (24, 7170, 6)].
- Missing fields: {}; nonfinite: {}; duplicate (drug, miRNA) within seed: 0; file/header issues: [].
- Score-identity max absolute discrepancy: 0; query-diagnostic bias/weight max discrepancy: 0.
- Across-seed pair sets identical: False; intersection of pair IDs across all available seeds: 0. Do not treat cross-seed rows as identically paired if this is false.
- **Plot support:** true per-drug shift distributions and score-block movement; within-drug ranking is invariant to constant query correction. Cross-drug/global ranking changes can be computed from these saved scores.
- **Experiment boundary:** these pair scores belong to the independent **PRISM** auxiliary-source experiment. `seed_metrics.csv` belongs to the primary Cell Painting MDA experiment. Their AUROC/AUPR aggregates should not be compared as if they were calculated from the same score matrix.
- Original score paths:
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed0/all_pair_scores_prism_drug_cold_seed0.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed1/all_pair_scores_prism_drug_cold_seed1.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed2/all_pair_scores_prism_drug_cold_seed2.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed3/all_pair_scores_prism_drug_cold_seed3.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed4/all_pair_scores_prism_drug_cold_seed4.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed5/all_pair_scores_prism_drug_cold_seed5.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed6/all_pair_scores_prism_drug_cold_seed6.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed7/all_pair_scores_prism_drug_cold_seed7.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed8/all_pair_scores_prism_drug_cold_seed8.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed9/all_pair_scores_prism_drug_cold_seed9.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed10/all_pair_scores_prism_drug_cold_seed10.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed11/all_pair_scores_prism_drug_cold_seed11.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed12/all_pair_scores_prism_drug_cold_seed12.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed13/all_pair_scores_prism_drug_cold_seed13.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed14/all_pair_scores_prism_drug_cold_seed14.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed15/all_pair_scores_prism_drug_cold_seed15.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed16/all_pair_scores_prism_drug_cold_seed16.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed17/all_pair_scores_prism_drug_cold_seed17.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed18/all_pair_scores_prism_drug_cold_seed18.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed19/all_pair_scores_prism_drug_cold_seed19.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed20/all_pair_scores_prism_drug_cold_seed20.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed21/all_pair_scores_prism_drug_cold_seed21.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed22/all_pair_scores_prism_drug_cold_seed22.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed23/all_pair_scores_prism_drug_cold_seed23.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed24/all_pair_scores_prism_drug_cold_seed24.tsv`
- Original query-diagnostic paths:
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed0/query_diagnostics_prism_drug_cold_seed0.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed1/query_diagnostics_prism_drug_cold_seed1.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed2/query_diagnostics_prism_drug_cold_seed2.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed3/query_diagnostics_prism_drug_cold_seed3.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed4/query_diagnostics_prism_drug_cold_seed4.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed5/query_diagnostics_prism_drug_cold_seed5.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed6/query_diagnostics_prism_drug_cold_seed6.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed7/query_diagnostics_prism_drug_cold_seed7.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed8/query_diagnostics_prism_drug_cold_seed8.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed9/query_diagnostics_prism_drug_cold_seed9.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed10/query_diagnostics_prism_drug_cold_seed10.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed11/query_diagnostics_prism_drug_cold_seed11.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed12/query_diagnostics_prism_drug_cold_seed12.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed13/query_diagnostics_prism_drug_cold_seed13.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed14/query_diagnostics_prism_drug_cold_seed14.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed15/query_diagnostics_prism_drug_cold_seed15.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed16/query_diagnostics_prism_drug_cold_seed16.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed17/query_diagnostics_prism_drug_cold_seed17.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed18/query_diagnostics_prism_drug_cold_seed18.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed19/query_diagnostics_prism_drug_cold_seed19.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed20/query_diagnostics_prism_drug_cold_seed20.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed21/query_diagnostics_prism_drug_cold_seed21.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed22/query_diagnostics_prism_drug_cold_seed22.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed23/query_diagnostics_prism_drug_cold_seed23.tsv`
  - `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/06_results/drug_cold_seed24/query_diagnostics_prism_drug_cold_seed24.tsv`

## 3. Beta-gamma recovery

- Output: `/home/u2508283030/QRC_ESWA_FigureData/beta_gamma_recovery.csv`; an exact file copy of `/home/u2508283030/ECARF_Article2_20260911/12_qrc_controlled_informative_shift_20260920/03_outputs/formal25/ALL_CONDITIONS_PER_SEED.csv`; rows: 625; expected 625.
- Source protocol report: `/home/u2508283030/ECARF_Article2_20260911/12_qrc_controlled_informative_shift_20260920/04_reports/CONTROLLED_INFORMATIVE_SHIFT_FORMAL25_REPORT.md`.
- Grain: one (seed, beta, gamma) case. Raw observed RMSE/MAE are `raw_to_target_rmse` / `raw_to_target_mae`; QRC observed RMSE/MAE are `qrc_to_target_rmse` / `qrc_to_target_mae`. Use these actual columns, not `expected_*` theory columns, for plots.
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]; beta grid: [0.0, 0.25, 0.5, 1.0, 2.0]; gamma grid: [0.0, 0.25, 0.5, 1.0, 2.0].
- Missing expected keys: []; duplicate keys: 0; missing values: {}; nonfinite: {}.
- Actual Raw RMSE vs beta max absolute deviation: 0; actual QRC RMSE vs gamma max absolute deviation: 2.46e-19.
- Phase check: QRC better for beta>gamma 250/250; Raw better for beta<gamma 250/250; tie for beta=gamma 125/125 (tolerance 1e-12).
- **Plot support:** observed 5×5 RMSE/MAE heatmaps and phase boundary, with 25 seed values per cell.

## 4. SQRC stress test

- Output: `/home/u2508283030/QRC_ESWA_FigureData/sqrc_stress_800.csv`; rows: 800; source `/home/u2508283030/ECARF_Article2_20260911/14_sqrc_estimation_stress_20260920/03_outputs/formal25/ALL_SEED_CONDITIONS.csv`; source protocol report `/home/u2508283030/ECARF_Article2_20260911/14_sqrc_estimation_stress_20260920/04_reports/SQRC_STRESS_FORMAL25_REPORT.md`.
- Grain: one (seed, sigma, n_candidates) condition. These are 800 distinct condition records, not 800 statistically independent cohorts; conditions reuse each seed.
- All original source columns are preserved. `nominal_uncertainty_sigma2_over_n` is nominal uncertainty; `mean_weight` is the shrinkage weight; `qrc_bias_mse` and `sqrc_bias_mse` are bias-estimation MSEs.
- Added `mse_gain = -sqrc_minus_qrc_bias_mse = qrc_bias_mse - sqrc_bias_mse` using exact decimal sign reversal of the saved per-case difference. Positive gain means lower SQRC MSE.
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]; sigma grid: [0.25, 0.5, 1.0, 2.0]; candidate counts: [5, 10, 25, 50, 65, 100, 379, 1195].
- Missing expected keys: []; duplicate keys: 0; missing values: {}; nonfinite: {}.
- Saved MSE-difference identity max absolute discrepancy: 0; SQRC lower bias MSE: 563/800.
- Hardest saved condition (sigma=2, n=5): mean weight 0.598001, QRC bias MSE 0.838862, SQRC bias MSE 0.525022; compare `/home/u2508283030/ECARF_Article2_20260911/14_sqrc_estimation_stress_20260920/04_reports/SQRC_STRESS_FORMAL25_REPORT.md`.
- **Plot support:** uncertainty vs MSE-gain scatter/heatmap; plot each case or aggregate only with an explicit stated rule.

## Manuscript and recovery limitations

The historical Fig. 3 summary and the two formal experiment reports agree with recomputation at their stated precision. I also inspected the available local submission file `D:\文献\文献阅读\组会\QRC_SQRC\投稿\QRC_SQRC_TCYB_Anonymous.pdf` (last modified 2026-10-05). It is an available submission version, not proof that a later manuscript has the same annotations. Three discrepancies remain in that PDF:

1. **Strict-cold Base pooled AUPR:** saved 25-seed mean±SD is **0.0627 ± 0.0512** at four decimals. The PDF prints **0.0626 ± 0.0512** in its main text/Table II and uses 0.0626 in Fig. 3(a) and later tables. This is a mean-rounding/data mismatch, not missing seed data.
2. **Random QRC pooled AUPR:** saved 25-seed mean±SD is **0.136248 ± 0.025636**, hence **0.1362 ± 0.0256** at four decimals. Fig. 3(a) prints **0.1363** for the mean. The earlier Fig. 3 audit had also flagged the older draft notation 0.1363 ± 0.0255.
3. **Drug-cold Base pooled AUPR:** saved mean is **0.0577** at four decimals and the PDF text/Table II/Table IV print 0.0577, but the Fig. 3(a) graphic prints **0.0557**. The graphic annotation is inconsistent with its own paper tables and the source records.

No manuscript or figure source file was modified. Other displayed values sampled in the relevant PDF table/figure agree at the shown precision; this is not a certification of every number in the full manuscript.
All four requested datasets are available as original row-level records; no aggregate-only substitution or lossless recovery from means was needed. True negatives cannot be recovered from unlabeled labels without external evidence.

## Output checksums (SHA-256)

- `seed_metrics.csv`: `45c10626bc1d2d4715ccb69f0959cc8dacae867bcd2b4e8afba3946d13481119`
- `drug_pair_scores.csv.gz`: `1614db47e654e46127161d099162e7375a1913c42a405884dac7ec4d82e7d2ea`
- `beta_gamma_recovery.csv`: `719a95197bcae0edf6ad8a5dedfbf93384238859d61d2dee584e74b618f4306f`
- `sqrc_stress_800.csv`: `1d28ba78d98c23a975ddb197bcddda525c0f46885985fa189724a850a5c21d4c`
