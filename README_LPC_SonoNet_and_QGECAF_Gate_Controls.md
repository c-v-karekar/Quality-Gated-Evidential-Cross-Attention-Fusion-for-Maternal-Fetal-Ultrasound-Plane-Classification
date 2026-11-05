# Automated LPC-SonoNet Benchmark and Matched QGECAF Gate Controls

This package contains the two reviewer-revision experiments that follow the
completed calibrated MobileNetV2 benchmark.

## Included experiments

### 1. Calibrated LPC-SonoNet

Files:

- `LPC_SonoNet_Calibrated_10Fold_Automated.py`
- `Run_LPC_SonoNet_Calibrated_10Fold.ipynb`

The program trains one fresh LPC-SonoNet model on each of the same ten saved
QGECAF patient partitions. It saves inner-validation and outer-holdout logits,
fits temperature scaling only from inner-validation logits, and reports the
same metric, prediction, calibration, timing and paired-comparison schema used
by the calibrated MobileNetV2 experiment.

The implementation follows the published four-block LPC design: two successive
3x3 branches are concatenated with the block input and fused with a 1x1
convolution, followed by a SonoNet-style class-activation head. The configurable
TensorFlow implementation contains approximately 4.47 million parameters with
the default channel widths. It is an architecture reimplementation under the
locked patient-fold protocol, not a reproduction of the paper's random-split
result.

Reference: Yu et al., *Sensors*, 2024, 24, 7510,
https://doi.org/10.3390/s24237510.

Default output:

```text
D:\chaitali\qgacef\LPC_SONONET_CALIBRATED_SAME_PATIENT_FOLDS
```

### 2. Matched QGECAF gate controls

Files:

- `QGECAF_Matched_Gate_Controls_Automated.py`
- `Run_QGECAF_Matched_Gate_Controls.ipynb`
- `QGECAF_Reviewer_Revision_Nested_BGWO.py`

The control program evaluates:

| Control | Gate definition | Newly trained? |
|---|---|---:|
| G0 | Fixed lambda = 0.5 | Yes, ten folds |
| G1 | One learned global scalar | Yes, ten folds |
| G2 | Correct sample-specific quality gate | No; completed QGECAF is reused |
| G3 | G2 gate architecture with shuffled quality rows | Yes, ten folds |

For every fold, the control program loads the already-fitted
`preprocessing.joblib`. It does not rerun BGWO and does not refit a raw-feature
mask, StandardScaler, PCA or quality scaler. The saved patient audit is reused
and the preprocessing-file checksum is recorded with every completed control.

G3 independently permutes quality rows inside inner training, inner validation
and outer holdout using saved deterministic seeds. It does not move images,
labels or patients between partitions. Permutation manifests are saved so the
negative control is reproducible.

Default output:

```text
D:\chaitali\qgacef\QGECAF_MATCHED_GATE_CONTROLS
```

The program also reconstructs the leakage-free fold-specific 33rd and 67th
quality percentiles from inner-training quality scores and produces
quality-stratified gate/performance results for G0--G3.

## Required QGECAF result root

Both experiments use:

```text
D:\chaitali\qgacef\QGECAF_REVIEWER_REVISION_NESTED_BGWO
```

Required files include:

```text
locked_patient_fold_assignments.csv
raw_features_presplit.npz
fold_metrics.csv
development_out_of_fold_predictions.csv
class_names.json
cross_validation\fold_01\patient_partition_audit.csv
cross_validation\fold_01\preprocessing.joblib
cross_validation\fold_01\outer_holdout_predictions.csv
...
cross_validation\fold_10\...
```

## Execution order

Run the two notebooks separately and restart the Jupyter kernel between them.

### LPC-SonoNet

```bat
conda activate chaitalienv
cd /d "D:\chaitali\qgacef\QGECAF_Reviewer_Controls_Package"
jupyter notebook "Run_LPC_SonoNet_Calibrated_10Fold.ipynb"
```

Select the `chaitalienv` TensorFlow 2.10 GPU kernel and run all cells.

### Gate controls

First choose **Kernel > Restart Kernel and Clear All Outputs**, then open:

```text
Run_QGECAF_Matched_Gate_Controls.ipynb
```

Run all cells in order. The preflight cell must report zero missing files.

## Resumption

Both programs are resumable. A completed fold is skipped only after its saved
prediction keys are checked against the locked outer holdout. An incomplete
fold is retrained. Do not change the output directory, seed, architecture or
protocol settings during a partially completed run.

## Principal LPC-SonoNet outputs

```text
fold_metrics_uncalibrated.csv
fold_metrics_temperature_scaled.csv
summary_uncalibrated_mean_sd_ci95.csv
summary_temperature_scaled_mean_sd_ci95.csv
development_oof_predictions_uncalibrated.csv
development_oof_predictions_temperature_scaled.csv
paired_qgecaf_vs_calibrated_lpc_sononet.csv
fold_temperatures.csv
reliability_diagram_oof.png
model_complexity.json
RUN_COMPLETE.json
```

## Principal gate-control outputs

```text
all_gate_control_fold_metrics.csv
gate_control_mean_sd_ci95.csv
gate_controls_friedman_tests.csv
G2_vs_controls_wilcoxon_holm.csv
quality_thresholds_by_fold.csv
quality_stratified_gate_controls.csv
G0_fixed_0p5\fold_metrics.csv
G1_learned_global\fold_metrics.csv
G2_correct_quality\fold_metrics.csv
G3_shuffled_quality\fold_metrics.csv
RUN_COMPLETE.json
```

Each newly trained control fold additionally saves its checkpoint, epoch log,
epoch timing, confusion matrix, reliability diagram, uncertainty predictions,
patient counts and, for G3, all quality permutation mappings.

## Important safeguards and interpretation

- The untouched final-test cohort is never evaluated.
- Calibration is fitted only from inner-validation logits.
- Temperature scaling must not change predicted classes.
- G2 is not rerun or selectively replaced.
- Gate controls use identical outer holdouts and therefore support paired
  fold-level testing.
- G2 superiority over G3, together with different gate distributions across
  quality groups, supports meaningful quality conditioning. Similar G2 and G3
  results would require moderation of the quality-adaptive claim.
- Run final-test comparisons only after all architectures and analysis rules
  have been frozen.
