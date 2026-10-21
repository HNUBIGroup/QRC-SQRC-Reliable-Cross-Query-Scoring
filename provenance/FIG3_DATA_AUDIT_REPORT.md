# Figure 3 real-data audit

All summaries are recomputed from historical seed-level outputs; SD is sample SD (ddof=1). No model was retrained and no historical result was modified.

## Completeness

- Fig. 3(a): PASS; 400 rows (expected 400).
- Fig. 3(b): PASS; 300 rows (expected 300).
- Fig. 3(c): PASS; 200 rows (expected 200).
- Fig. 3(d): PASS; 50 rows (expected 50).

## Diagnostic definitions verified from historical source
- `query_shift_variance_ratio = Var(repeated query shift) / Var(Raw-Base increment)`, with population variances (`ddof=0`) within each seed.
- `mean_abs_raw_base = mean(abs(Raw-Base))` over scored pairs.
- `mean_abs_qrc_base = mean(abs(QRC-Base))` over scored pairs.
- Figure reduction percentage is the reduction between the across-seed mean MADs: `1 - mean(QRC MAD)/mean(Raw MAD)`. Per-seed reductions are also preserved in the seedwise TSV.

## Manuscript consistency
- All supplied checks agree at their stated/approximate precision except the draft Random/QRC AUPR annotation: the historical value is 0.136248 ± 0.025636 (0.1362 ± 0.0256 at four decimals), not 0.1363 ± 0.0255.
- Previously unspecified SQRC values, strict-cold PRISM SDs, and strict-cold MAD diagnostics were recovered directly from the historical 25-seed files.

## Source files
- `/home/u2508283030/ECARF_Article2_20260911/02_results/stageG_random_frozen_current_25seed.tsv`
- `/home/u2508283030/ECARF_Article2_20260911/02_results/stageE_shrinkage_qrc_25seed.tsv`
- `/home/u2508283030/ECARF_Article2_20260911/02_results/stageG_v2_random_strict_final_25seed.tsv`
- `/home/u2508283030/QRC_CrossBackbone_20260924/10_reports/hyperattention_25seed_per_seed_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed0/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed1/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed2/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed3/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed4/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed5/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed6/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed7/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed8/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed9/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed10/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed11/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed12/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed13/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed14/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed15/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed16/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed17/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed18/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed19/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed20/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed21/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed22/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed23/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/DLST-MDA/seed24/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed0/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed1/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed2/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed3/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed4/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed5/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed6/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed7/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed8/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed9/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed10/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed11/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed12/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed13/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed14/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed15/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed16/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed17/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed18/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed19/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed20/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed21/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed22/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed23/seed_method_metrics.tsv`
- `/home/u2508283030/QRC_CrossBackbone_DLST_HGGN_20260929/07_formal/HGGN/seed24/seed_method_metrics.tsv`
- `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/07_qc/PRISM_DRUGCOLD_25SEEDS_PER_SEED.tsv`
- `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/07_qc/PRISM_STRICTCOLD_25SEEDS_PER_SEED.tsv`
- Diagnostic implementation: `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/02_src/run_prism_drugcold_seed.py`
- Diagnostic implementation: `/home/u2508283030/ECARF_Article2_20260911/17_qrc_prism_mda_formal_20260920/02_src/run_prism_strictcold_seed.py`
- Audit aggregation script: `/home/u2508283030/ECARF_Article2_20260911/_fig3_data_audit/fig3_data_audit.py`
