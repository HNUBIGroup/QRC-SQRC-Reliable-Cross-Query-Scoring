# QRC / SQRC: Reliable Cross-Query Scoring

Research code and experiment records for **Cross-Query Score Comparability in Sparse Pairwise Learning: Residual Centering with Uncertainty-Aware Shrinkage**.

Yun Zhou, Min Zhang, Maoqi Lian, Wei Wang, and Dong Liu.

[Manuscript](paper/QRC_SQRC.pdf) | [Data dictionary](docs/DATA_DICTIONARY.md) | [Source manifest](data/manifest.json) | [Historical sources](historical/README.md)

## Method

QRC estimates each query's mean matched Raw-minus-Base increment and subtracts it from Raw scores. SQRC scales that shift by `tau2 / (tau2 + SE2)`, estimating `tau2` from independent validation queries. Both subtract a constant within each query, preserving within-query score differences. Correction is conditional: query shifts may contain useful signal, as demonstrated by Davis DTI.

## Quick start

Python 3.10 or newer is recommended. Run from the repository root:

```sh
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/analyze_records.py
python scripts/reproduce_stress.py --output results/stress_reproduced
python scripts/compare_stress.py results/stress_reproduced/ALL_SEED_CONDITIONS.csv
```

`verify_release.py` checks file hashes, row counts, unique experimental keys, saved score identities and the 563/800 stress result. `analyze_records.py` regenerates seed summaries and PRISM pooled/macro metrics and tests QRC score reproduction. The stress entry point reruns the original random-number experiment on CPU; comparison checks the 24 original fields against the archived rows. Generated outputs do not replace archived data.

## Included data

The release contains 13 CSV/TSV tables, preserving record-level results and source paths. See the data dictionary for row counts, fields and overlap.

- Primary Cell Painting MDA: four protocols and 25 seeds each.
- Independent PRISM: 179,250 drug-cold pair predictions, plus seed metrics and query-shift diagnostics.
- Cross-backbone experiments and the controlled beta-gamma recovery study.
- SQRC stress: one canonical 800-row export, `sqrc_stress_800_raw.csv`.
- Davis DTI boundary, counterfactual decomposition, candidate stability, shared-budget Top-K, and cross-query AUROC.

`drug_pair_scores.csv.gz` is PRISM, whereas `seed_metrics.csv` is Cell Painting; their metrics must not be treated as originating from the same predictions. The long-form Fig. 3a table overlaps `seed_metrics.csv`. The alternative historical `sqrc_stress_800.csv` is intentionally omitted because it describes the same 800 experiments.

Historical label 0 in the PRISM pair table means an unlabeled comparison record, not an experimentally confirmed negative. Repeated shuffles/subsets within a seed are not independent seeds. Sample SD is calculated with ddof=1.

## Reproducibility scope

This package supports reanalysis of the released results, post-hoc QRC/SQRC correction, and rerunning the synthetic stress experiment. It also archives 45 original experiment scripts, including PRISM and Davis training/postprocessing entry points. It does not contain every backbone's upstream source code, raw biological source dataset, pretrained checkpoint, split file or every experiment's full prediction matrix. It must not be described as a complete end-to-end retraining release. Historical training scripts require their external inputs and original environment; `requirements.txt` covers the portable analysis entry points, not GPU training or upstream feature extraction.

Original audit reports are preserved in `provenance/`, including their historical ESWA names and paths. Those labels indicate provenance rather than current publication status. See `results/VERIFICATION.json` and `results/summary.json` for checks actually executed for this package.

For the prepared package, all 800 stress records reproduced exactly across the 24 original columns. PRISM QRC score reproduction differed by at most 2.23e-16 and archived metric reproduction by less than 1e-15. Exact floating-point equality is not guaranteed across different library versions; the comparison entry point allows an explicit 1e-10 tolerance. `results/SOURCE_TRANSFER_CHECK.json` records successful server/local hash comparisons for 58 code/data entries, including one duplicated historical script reference.

## Data sources and rights

Only experiment-result exports and the supplied manuscript are bundled. Third-party raw Cell Painting, PRISM, MDA and Davis datasets are not redistributed as raw source collections. No blanket license is imposed on third-party inputs or the manuscript. Reuse remains subject to applicable source terms; see [rights notice](docs/RIGHTS.md).

## Citation

Please cite the manuscript by its title and authors above. A journal, DOI and publication year have not been assigned in this package because publication metadata have not been verified.

The PRISM pair-score CSV is stored with lossless gzip compression to fit GitHub browser upload limits. pandas reads it directly; `verify_release.py` also checks the decompressed source SHA-256. No records or numeric values were changed.
