# QGECAF Quality-Stratified Performance and Gate-Behaviour Analysis

This package performs the reviewer-requested quality-stratified analysis on the
ten completed QGECAF patient-level outer folds. It does **not** retrain QGECAF
and does **not** use the untouched final-test cohort.

## Files

- `QGECAF_Quality_Stratified_Gate_Analysis.py` — complete automated analysis.
- `Run_QGECAF_Quality_Stratified_Analysis.ipynb` — Jupyter runner with the
  required Windows paths already configured.

## Required completed artifacts

The default root is:

```text
D:\chaitali\qgacef\QGECAF_REVIEWER_REVISION_NESTED_BGWO
```

The program requires:

```text
locked_patient_fold_assignments.csv
raw_features_presplit.npz
cross_validation\fold_01\patient_partition_audit.csv
cross_validation\fold_01\preprocessing.joblib
cross_validation\fold_01\outer_holdout_predictions.csv
...
cross_validation\fold_10\patient_partition_audit.csv
cross_validation\fold_10\preprocessing.joblib
cross_validation\fold_10\outer_holdout_predictions.csv
```

## Recommended Jupyter procedure

1. Put the Python file and notebook in the same folder on the Windows machine.
2. Open **Anaconda Prompt**.
3. Run:

```bat
conda activate chaitalienv
cd /d D:\chaitali\qgacef
jupyter notebook
```

4. Open `Run_QGECAF_Quality_Stratified_Analysis.ipynb`.
5. Select the `chaitalienv` kernel.
6. Run all cells in order.
7. Confirm that the preflight cell reports zero missing files.

The default analysis call is:

```python
from QGECAF_Quality_Stratified_Gate_Analysis import Config, run

cfg = Config(
    qgecaf_root=(
        r"D:\chaitali\qgacef"
        r"\QGECAF_REVIEWER_REVISION_NESTED_BGWO"
    ),
    output=(
        r"D:\chaitali\qgacef"
        r"\QGECAF_REVIEWER_REVISION_NESTED_BGWO"
        r"\quality_stratified_gate_analysis"
    ),
    bootstrap_repetitions=2000,
    seed=42,
    overwrite=False,
)
run(cfg)
```

## What the program verifies

- All ten locked patient folds are present.
- Inner-training and outer-holdout records are disjoint at patient level.
- No final-test record (`outer_fold == -1`) enters the analysis.
- The saved quality scaler belongs to the fold under analysis.
- The reconstructed outer quality scores match the saved prediction file.
- The 33rd and 67th percentile thresholds are calculated exclusively from
  that fold's inner-training images.
- Every development image receives exactly one out-of-fold prediction.
- Macro F1 uses all six labels with `zero_division=0`; missing classes in a
  quality stratum are explicitly recorded.

## Main outputs

Results are written to:

```text
D:\chaitali\qgacef\QGECAF_REVIEWER_REVISION_NESTED_BGWO
\quality_stratified_gate_analysis
```

Important outputs include:

- `quality_thresholds_by_fold.csv`
- `quality_stratified_fold_metrics.csv`
- `quality_stratified_10fold_summary.csv`
- `quality_stratified_pooled_oof_metrics.csv`
- `quality_stratified_oof_predictions.csv`
- `foldwise_quality_gate_correlations.csv`
- `quality_group_friedman_tests.csv`
- `quality_group_paired_wilcoxon_holm.csv`
- `quality_gate_distribution_boxplot.png`
- `quality_stratified_metric_panels.png`
- `quality_score_vs_gate_lambda.png`
- `QGECAF_quality_stratified_analysis.xlsx`
- `automated_interpretation.txt`
- `RUN_COMPLETE.json`

For each fold and quality group, the output includes image and patient counts,
class coverage, accuracy, six-class macro F1, NLL, multiclass Brier score, ECE,
quality thresholds, and mean/SD/median/IQR/range of the learned gate lambda.
Patient-clustered bootstrap 95% confidence intervals are also generated for
the primary performance, calibration, and gate metrics.

## Runtime and resumption

No GPU training is performed. Most runtime is due to 2,000 patient-clustered
bootstrap repetitions for 30 fold-by-quality strata. A rerun with the same
configuration is deterministic. For a quick pipeline test, temporarily use
`bootstrap_repetitions=100`; use 2,000 or more for the reported analysis.

If an existing output folder was produced using an incompatible protocol, the
program stops rather than mixing results. Use a new output folder when changing
the percentile definition or another analysis setting.

## Adding the shuffled-quality control later

After matched shuffled-quality gate (G3) folds have been trained and their
outer-holdout prediction files saved, set:

```python
cfg.shuffled_gate_root = (
    r"D:\path\to\matched_G3_shuffled_quality_results"
)
```

The program applies the **same G2-derived inner-training thresholds** to the G3
holdouts and creates additional matched G2-versus-G3 files. Do not interpret
quality-group differences as causal evidence until the matched G3 comparison
is available.

## Reporting caution

The pooled out-of-fold table is descriptive. Inferential comparisons use the
ten matched fold-level observations. If a quality group lacks a class, the
program records the missing class and retains all six labels in macro F1. ECE
is sample-size-sensitive, so group sizes and patient-clustered confidence
intervals should be reported with it.
