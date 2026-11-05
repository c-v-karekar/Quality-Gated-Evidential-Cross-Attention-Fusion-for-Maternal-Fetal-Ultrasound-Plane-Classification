# Automated calibrated MobileNetV2 benchmark

This package performs the mandatory matched lightweight-CNN comparison for the
completed reviewer-revision QGECAF experiment.

## Included files

- `MobileNetV2_Calibrated_10Fold_Automated.py`: complete resumable experiment.
- `Run_MobileNetV2_Calibrated_10Fold.ipynb`: small Jupyter runner.

## Required existing QGECAF result folder

The default configuration reads:

```text
D:\chaitali\qgacef\QGECAF_REVIEWER_REVISION_NESTED_BGWO
```

It requires:

```text
locked_patient_fold_assignments.csv
fold_metrics.csv
development_out_of_fold_predictions.csv
cross_validation\fold_01\patient_partition_audit.csv
...
cross_validation\fold_10\patient_partition_audit.csv
```

## Output folder

```text
D:\chaitali\qgacef\MOBILENETV2_CALIBRATED_SAME_PATIENT_FOLDS
```

Do not point the output at the QGECAF results directory.

## Run in Jupyter

1. Put the Python file and notebook in the same Windows folder.
2. Open Anaconda Prompt and activate `chaitalienv`.
3. Start Jupyter Notebook from that folder.
4. Open `Run_MobileNetV2_Calibrated_10Fold.ipynb`.
5. Select the TensorFlow 2.10 GPU kernel.
6. Run the cells in order.
7. The cell containing `run(cfg)` performs or safely resumes the experiment.

### TensorFlow 2.10 GPU compatibility

This corrected package keeps the Python, NumPy and TensorFlow seeds fixed and
uses deterministic input-pipeline ordering, but it does not enable strict
deterministic GPU operations. TensorFlow 2.10 has no deterministic GPU kernel
for the `UnsortedSegmentSum` gradient used by sparse categorical
cross-entropy. Enabling strict operation determinism therefore stops training
with an `UnimplementedError`.

If an earlier version produced that error, replace both the Python file and
the notebook with the corrected versions, then select **Kernel > Restart**
before running the notebook from the first cell. A kernel restart is necessary
because TensorFlow's deterministic-operation setting cannot be disabled after
it has been enabled in the same Python process.

## Protocol safeguards

- Exact saved QGECAF inner-training, inner-validation and outer-holdout
  partitions are reused.
- Pairwise patient overlap is checked for every fold.
- Every row with `outer_fold=-1` is excluded.
- Temperature is fitted from inner-validation logits only.
- A positive scalar temperature is required to preserve every predicted class.
- Existing folds are skipped only after their holdout keys are verified.
- A protocol hash prevents incompatible runs from sharing an output directory.

## Principal outputs

```text
fold_metrics_uncalibrated.csv
fold_metrics_temperature_scaled.csv
summary_uncalibrated_mean_sd_ci95.csv
summary_temperature_scaled_mean_sd_ci95.csv
development_oof_predictions_uncalibrated.csv
development_oof_predictions_temperature_scaled.csv
paired_qgecaf_vs_calibrated_mobilenetv2.csv
fold_temperatures.csv
reliability_diagram_oof.png
model_complexity.json
RUN_COMPLETE.json
```

Each `cross_validation\fold_XX` directory also contains the checkpoint,
training history, epoch timing, validation logits, fitted temperature,
uncalibrated/calibrated holdout predictions and class-wise metrics.

## Important interpretation

Temperature scaling should change NLL, Brier score, ECE, confidence and
uncertainty-based measures. It must not change accuracy, balanced accuracy,
macro F1, weighted F1, MCC or the confusion matrix.

This package does not evaluate the untouched final test cohort. Final-test
benchmarking should be performed only after all models and calibration rules
have been frozen.

LPC-SonoNet and matched gate controls should be executed in separate output
folders after this benchmark is complete. Keeping them separate prevents
architecture-specific code from changing the matched MobileNetV2 control.
