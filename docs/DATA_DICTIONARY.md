# Data dictionary

Scores retain the historical numeric values and headers. AUROC/AUPR are on the 0-1 scale. Sample SD uses ddof=1. Empty historical fields are not imputed.

| File | Records | Interpretation |
|---|---:|---|
| [beta_gamma_recovery.csv](../data/beta_gamma_recovery.csv) | 625 | Controlled beta-gamma recovery; one seed/parameter setting per row. |
| [candidate_stability_25seeds.csv](../data/candidate_stability_25seeds.csv) | 4,050 | Candidate-subset stability; retain seed, fraction and subset repeat. See audit for actual candidate counts. |
| [counterfactual_25seeds.csv](../data/counterfactual_25seeds.csv) | 2,150 | Original and shuffled decompositions; retain seed and shuffle repeat, which are not independent seeds. |
| [cross_query_per_seed.csv](../data/cross_query_per_seed.csv) | 875 | Cross-query positive-negative AUROC, with pooled/macro comparators and pair counts. |
| [davis_boundary_25seeds.csv](../data/davis_boundary_25seeds.csv) | 100 | Davis DTI boundary; 25 seeds (5-29) x four methods. Includes merged shared-budget statistics. |
| [drug_pair_scores.csv.gz](../data/drug_pair_scores.csv.gz) | 179,250 | Independent PRISM drug-cold; one seed/drug/miRNA pair per row. label=0 denotes unlabeled comparison pairs, not validated biological negatives. |
| [fig3a_cellpainting_seedwise.tsv](../data/fig3a_cellpainting_seedwise.tsv) | 400 | Primary Cell Painting; long-form representation overlapping seed_metrics.csv. Not independent data. |
| [fig3b_crossbackbone_seedwise.tsv](../data/fig3b_crossbackbone_seedwise.tsv) | 300 | Cross-backbone, method and seed-level AUROC/AUPR. |
| [fig3c_prism_seedwise.tsv](../data/fig3c_prism_seedwise.tsv) | 200 | PRISM drug-cold and strict-cold; pooled and query-macro metrics. |
| [fig3d_prism_shift_seedwise.tsv](../data/fig3d_prism_shift_seedwise.tsv) | 50 | PRISM query-shift and Raw/QRC macro-invariance diagnostics. |
| [seed_metrics.csv](../data/seed_metrics.csv) | 100 | Primary Cell Painting MDA; four protocols, wide method columns; one protocol/seed per row. Pooled metrics only. |
| [sqrc_stress_800_raw.csv](../data/sqrc_stress_800_raw.csv) | 800 | SQRC stress; 25 seeds x 4 sigma values x 8 candidate counts. Added alias columns do not represent additional observations. |
| [topk_per_seed.csv](../data/topk_per_seed.csv) | 3,500 | Shared-pool Top-K budgets; preserve task, setting, seed, method and budget_fraction. |

## beta_gamma_recovery.csv

Controlled beta-gamma recovery; one seed/parameter setting per row.

Fields (original order): `seed`, `rng_seed`, `alpha`, `beta`, `gamma`, `raw_to_target_mae`, `raw_to_target_rmse`, `raw_to_target_max_abs_error`, `qrc_to_target_mae`, `qrc_to_target_rmse`, `qrc_to_target_max_abs_error`, `oracle_n_to_target_mae`, `oracle_n_to_target_rmse`, `oracle_n_to_target_max_abs_error`, `bias_recovery_mae`, `bias_recovery_rmse`, `bias_recovery_max_abs_error`, `informative_removal_identity_mae`, `informative_removal_identity_rmse`, `informative_removal_identity_max_abs_error`, `strict_order_violations`, `tie_violations`, `max_pairwise_difference_error`, `raw_qrc_macro_metric_difference`, `expected_raw_rmse`, `raw_rmse_theory_error`, `expected_qrc_rmse`, `qrc_rmse_theory_error`, `phase_region`, `phase_boundary_status`, `algebraic_qc_status`, `ranking_invariance_status`, `theory_error_status`, `condition_status`, `qrc_uses_labels`, `test_derived_tuning`, `base_pooled_auroc`, `base_pooled_aupr`, `base_macro_auroc`, `base_macro_aupr`, `target_pooled_auroc`, `target_pooled_aupr`, `target_macro_auroc`, `target_macro_aupr`, `raw_pooled_auroc`, `raw_pooled_aupr`, `raw_macro_auroc`, `raw_macro_aupr`, `qrc_pooled_auroc`, `qrc_pooled_aupr`, `qrc_macro_auroc`, `qrc_macro_aupr`, `oraclen_pooled_auroc`, `oraclen_pooled_aupr`, `oraclen_macro_auroc`, `oraclen_macro_aupr`.

## candidate_stability_25seeds.csv

Candidate-subset stability; retain seed, fraction and subset repeat. See audit for actual candidate counts.

Fields (original order): `mode`, `seed`, `fraction`, `repeat`, `AUROC`, `AUPR`, `FullQRC_AUROC`, `FullQRC_AUPR`, `Delta_AUROC_vs_Full`, `Delta_AUPR_vs_Full`, `Bias_MAE`, `protocol`, `n_test_queries`, `full_candidate_count_min`, `full_candidate_count_max`, `actual_candidate_count_min`, `actual_candidate_count_max`, `actual_candidate_count`, `paired_delta_auroc`, `paired_delta_aupr`.

## counterfactual_25seeds.csv

Original and shuffled decompositions; retain seed and shuffle repeat, which are not independent seeds.

Fields (original order): `mode`, `seed`, `repeat`, `condition`, `pooled_AUROC`, `pooled_AUPR`, `macro_AUROC`, `protocol`, `method_or_intervention`, `record_type`.

## cross_query_per_seed.csv

Cross-query positive-negative AUROC, with pooled/macro comparators and pair counts.

Fields (original order): `task`, `setting`, `seed`, `method`, `pooled_auroc`, `pooled_aupr`, `macro_auroc`, `macro_aupr`, `cross_query_auroc`, `cross_query_pos_neg_pairs`, `within_query_pos_neg_pairs`, `macro_queries_used`, `test_pairs`, `known_positives`, `other_pairs`.

## davis_boundary_25seeds.csv

Davis DTI boundary; 25 seeds (5-29) x four methods. Includes merged shared-budget statistics.

Fields (original order): `seed`, `method`, `pooled_auroc`, `pooled_aupr`, `drug_macro_auroc`, `drug_macro_aupr`, `brier`, `ece_10bin`, `query_shift_variance_ratio`, `raw_vs_base_cross_query_inversion_rate`, `qrc_vs_base_cross_query_inversion_rate`, `macro_invariance_qc`, `protocol`, `test_pairs`, `known_positives`, `other_pairs`, `positive_fraction`, `test_drug_queries`, `top1pct_k`, `top1pct_hits`, `top1pct_precision`, `top1pct_recall`, `top1pct_enrichment_factor`, `top2pct_k`, `top2pct_hits`, `top2pct_precision`, `top2pct_recall`, `top2pct_enrichment_factor`, `top5pct_k`, `top5pct_hits`, `top5pct_precision`, `top5pct_recall`, `top5pct_enrichment_factor`, `top10pct_k`, `top10pct_hits`, `top10pct_precision`, `top10pct_recall`, `top10pct_enrichment_factor`.

## drug_pair_scores.csv.gz

Independent PRISM drug-cold; one seed/drug/miRNA pair per row. label=0 denotes unlabeled comparison pairs, not validated biological negatives.

Fields (original order): `seed`, `drug`, `miRNA`, `label`, `base_score`, `raw_score`, `delta`, `query_bias`, `qrc_score`, `sqrc_weight`, `sqrc_score`.

## fig3a_cellpainting_seedwise.tsv

Primary Cell Painting; long-form representation overlapping seed_metrics.csv. Not independent data.

Fields (original order): `split`, `seed`, `method`, `AUROC`, `AUPR`, `source_file`.

## fig3b_crossbackbone_seedwise.tsv

Cross-backbone, method and seed-level AUROC/AUPR.

Fields (original order): `backbone`, `seed`, `method`, `AUROC`, `AUPR`, `source_file`.

## fig3c_prism_seedwise.tsv

PRISM drug-cold and strict-cold; pooled and query-macro metrics.

Fields (original order): `setting`, `seed`, `method`, `pooled_AUROC`, `pooled_AUPR`, `macro_AUROC`, `macro_AUPR`, `source_file`.

## fig3d_prism_shift_seedwise.tsv

PRISM query-shift and Raw/QRC macro-invariance diagnostics.

Fields (original order): `setting`, `seed`, `mean_abs_raw_base`, `mean_abs_qrc_base`, `reduction_fraction_per_seed`, `query_shift_variance_ratio`, `raw_macro_AUROC`, `qrc_macro_AUROC`, `raw_macro_AUPR`, `qrc_macro_AUPR`, `source_file`.

## seed_metrics.csv

Primary Cell Painting MDA; four protocols, wide method columns; one protocol/seed per row. Pooled metrics only.

Fields (original order): `setting`, `seed`, `base_pooled_auroc`, `base_pooled_aupr`, `raw_pooled_auroc`, `raw_pooled_aupr`, `qrc_pooled_auroc`, `qrc_pooled_aupr`, `sqrc_pooled_auroc`, `sqrc_pooled_aupr`, `source_file`.

## sqrc_stress_800_raw.csv

SQRC stress; 25 seeds x 4 sigma values x 8 candidate counts. Added alias columns do not represent additional observations.

Fields (original order): `seed`, `sigma`, `n_candidates`, `nominal_uncertainty_sigma2_over_n`, `tau_true2`, `tau_b2_validation`, `tau_b2_abs_error`, `validation_observed_bhat_variance`, `validation_mean_se2`, `test_mean_se2`, `mean_weight`, `median_weight`, `min_weight`, `max_weight`, `oracle_weight`, `qrc_bias_mse`, `sqrc_bias_mse`, `sqrc_minus_qrc_bias_mse`, `sqrc_bias_mse_relative_reduction`, `qrc_score_rmse`, `sqrc_score_rmse`, `sqrc_minus_qrc_score_rmse`, `qrc_pairwise_invariance_error`, `sqrc_pairwise_invariance_error`, `n`, `shrinkage_weight_w`, `uncertainty_se2`, `mse_gain_qrc_minus_sqrc`, `relative_gain_qrc_minus_sqrc`.

## topk_per_seed.csv

Shared-pool Top-K budgets; preserve task, setting, seed, method and budget_fraction.

Fields (original order): `task`, `setting`, `seed`, `method`, `budget_fraction`, `test_pairs`, `k`, `known_positives`, `other_pairs`, `hit_at_k`, `precision_at_k`, `recall_at_k`, `enrichment_factor`.

## Shared field meanings

- `seed`: historical random split/run identifier; seed ranges differ by task.
- `base_score`, `raw_score`: matched predictions without/with the target auxiliary source.
- `delta = raw_score - base_score`; `query_bias`: mean delta for the query.
- `qrc_score = raw_score - query_bias`; `sqrc_score = raw_score - sqrc_weight * query_bias`.
- `pooled_*`: metrics over the shared pair pool; `macro_*` / `drug_macro_*`: per-query metrics averaged over eligible queries.
- `cross_query_auroc`: AUROC restricted to positive-negative pairs from different queries; inversion rate is a different diagnostic.
- `mean_weight`, `shrinkage_weight_w`: mean test-query shrinkage weight in one stress condition.
- `validation_mean_se2`, `test_mean_se2`: distinct validation/test mean-shift estimation variances.
- `repeat`: repeated shuffle/subset within a seed; do not count these as new independent seeds.
- `source_file`: original provenance path, retained unchanged; it is not a local dependency.
- Stress aliases `n`, `shrinkage_weight_w`, `uncertainty_se2` copy `n_candidates`, `mean_weight`, `test_mean_se2`; gain fields are deterministic arithmetic.

The original audit reports describe column derivations, missing measures and experimental limits.
