# Historical source scripts

The archive contains 45 scripts copied unchanged from the original experiment directories. `manifest.json` records each source path, byte count and SHA-256 hash. Remote and local hashes were compared independently for all scripts. Three top-level reference entry points are:

| File | Original location under ECARF_Article2_20260911 |
|---|---|
| stageE_shrinkage_qrc.py | 01_src/stageE_shrinkage_qrc.py |
| run_sqrc_stress_formal25.py | 14_sqrc_estimation_stress_20260920/02_code/run_sqrc_stress_formal25.py |
| eswa_supplementary.py | ESWA_supplementary_experiments/eswa_supplementary.py |

The remaining subdirectories preserve primary postprocessing (21 scripts), PRISM (6), Davis DTI (7), controlled ground truth (2), informative shifts (2), calibration (3), and Fig. 3 auditing (1). The primary postprocessing copy of `stageE_shrinkage_qrc.py` overlaps the top-level reference copy and does not represent an additional experiment.

These sources preserve the historical computations, including absolute server paths. They are reference sources, not portable entry points; do not run them without configuring their paths and obtaining their required upstream inputs. Supplementary analysis imports `run_calibration_baselines.py`, archived under `calibration/`, through its original directory layout. PRISM additionally imports the external `model_fourview_staged` module; Davis requires the upstream MMDG-DTI vendor code, feature files, and splits. These external dependencies are not bundled.

The portable stress entry point ../scripts/reproduce_stress.py changes only output path handling. The portable src/qrc.py is a newly packaged implementation checked against saved predictions; it is not represented as the original training code.
