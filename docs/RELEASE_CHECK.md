# Release checks and limitations

Prepared on 2026-10-21 for the manuscript repository `HNUBIGroup/QRC-SQRC-Reliable-Cross-Query-Scoring`.

Checks executed:

- 13 CSV/TSV data files: 192,400 rows in total. Some tables overlap; this sum is not a count of independent experimental observations.
- All source-data and historical-code hashes agree with independent SHA-256 calculations on the source server: 58 entries, 57 unique remote files.
- 45 historical Python scripts compile for syntax; their full training workflows were not rerun.
- Experimental record keys, primary 25-seed coverage, PRISM labels, constant within-query shifts/weights, and saved QRC/SQRC identities pass.
- Independent recomputation of PRISM pooled/macro metrics agrees with the saved seed metrics within 1e-15.
- The portable QRC implementation matches PRISM saved predictions within 2.23e-16. Within-query Raw/QRC/SQRC score-difference errors are below 2.78e-16 and query-macro metrics match exactly on these data.
- Original synthetic SQRC stress experiment rerun: 800 cases, all 24 original fields exactly reproduced; SQRC wins 563/800.
- Historical source files scanned for obvious embedded credential patterns; no such pattern was found. This is a limited check, not a security audit.

Portable checks were run with Python 3.9.0, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1 and scikit-learn 1.5.2. Four portable API numerical/input-validation tests also passed. Python 3.10+ is recommended for new installations. Exact check-environment versions are also saved in `results/environment.json`.

Limits: full backbone training, third-party vendor dependencies, raw dataset redistribution rights and manuscript publication rights have not been validated by these numeric checks. Existing audit reports retain their recorded limitations. No publication DOI, acceptance status or repository-wide reuse license has been invented.
