# QGECAF reviewer-revision experiment

This package contains the corrected primary QGECAF experiment requested by the
reviewer. It uses the original locked patient-level outer folds and performs
feature selection without allowing a BGWO validation patient to influence the
candidate scaler or PCA.

## Critical design

The original physical `train` and `val` folders are combined into one
development cohort (11,265 images from 1,612 patients). The existing manifest
is then used exactly as follows:

- `outer_fold = 1..10`: development patient holdouts;
- `outer_fold = -1`: untouched final test cohort;
- for fold `f`, `outer_fold == f` is the outer holdout and every other
  development fold is the outer-training pool;
- an inner patient-grouped split is made only inside that outer-training pool
  for early stopping and learning-rate/checkpoint decisions;
- the final test cohort never participates in BGWO, scaling, PCA, class
  weighting, checkpoint selection, or epoch selection.

For every feature family and every candidate BGWO raw-feature mask, each
three-fold BGWO split executes this pipeline independently:

1. select candidate raw feature columns;
2. fit `StandardScaler` on BGWO-training patients only;
3. transform BGWO-training and BGWO-validation observations;
4. fit PCA on BGWO-training patients only (98% retained variance, where used);
5. transform both partitions;
6. train an RBF-SVM (`C=2`, `gamma='scale'`, balanced class weights);
7. calculate six-class macro F1 on BGWO-validation patients;
8. optimize `0.99*(1-macro_F1) + 0.01*(selected/raw features)`.

After BGWO selects a mask, its scaler and PCA are fitted once on the complete
inner-training partition and applied unchanged to the inner validation and
outer holdout. During final refitting, BGWO/scaler/PCA are fitted on the entire
development cohort, and the fixed epoch count is the rounded median of the ten
outer-fold best epochs.

## How this differs from the previous program

| Issue | Previous workflow | Reviewer-revision workflow |
|---|---|---|
| BGWO search space | Could select already transformed/PCA coordinates | Selects original raw family columns |
| Scaling during BGWO | Could be fitted before internal BGWO validation | Refit only on BGWO-training patients for every candidate split |
| PCA during BGWO | Could be fitted before internal BGWO validation | Refit only on BGWO-training patients for every candidate split |
| BGWO validation | RBF-SVM macro F1, but preprocessing provenance was unclear | Patient-grouped RBF-SVM macro F1 with explicit leakage checks |
| Outer folds | Could be regenerated from a seed | Loaded from the original locked manifest using patient ID + image stem |
| Artifacts | Limited preprocessing audit trail | Raw masks, PCA dimensions, selector history, fit partitions, fitted objects, predictions, seeds and manifests are saved |
| Timing | Network-only measurements from different contexts | Both model-only and uncached raw-image-to-prediction timings are saved with warm-up, repetitions and batch size |

Because scaler and PCA are refitted for every candidate mask and BGWO split,
this corrected selector is substantially slower than the earlier workflow,
especially for HOG. Do not write results into the old result directory.

## Run from Anaconda Prompt

Activate the same TensorFlow 2.10 environment used for the original study:

```bat
conda activate chaitalienv
cd /d D:\chaitali\qgacef
python QGECAF_Reviewer_Revision_Nested_BGWO.py
```

To provide paths explicitly:

```bat
python QGECAF_Reviewer_Revision_Nested_BGWO.py ^
  --dataset "D:\chaitali\patientsplit801010_clean_v2" ^
  --metadata-csv "D:\chaitali\patientsplit801010_clean_v2\patient_split_80_10_10.csv" ^
  --locked-manifest "D:\chaitali\qgacef\QGECAF_FROM_SCRATCH_RUN_01\locked_patient_fold_assignments.csv" ^
  --output "D:\chaitali\qgacef\QGECAF_REVIEWER_REVISION_NESTED_BGWO"
```

## Run from Jupyter

1. Open Anaconda Prompt, activate `chaitalienv`, change to the folder, and run
   `jupyter notebook`.
2. Open `QGECAF_Reviewer_Revision_Run.ipynb`.
3. Verify all four Windows paths in the configuration cell.
4. Run the preflight cell, then the training cell.
5. If interrupted, restart the kernel, rerun the definition/configuration
   cells, and call `main(cfg)` again. Completed and verified outer folds are
   skipped. Never point this script at an old result directory.

## Principal outputs

- `locked_patient_fold_assignments.csv`: normalized copy of the original folds;
- `cross_validation/fold_XX/patient_partition_audit.csv`: fold-level patient audit;
- `cross_validation/fold_XX/nested_preprocessing/`: raw masks, BGWO traces,
  PCA dimensions and exact fit population;
- `cross_validation/fold_XX/preprocessing.joblib`: fitted fold objects;
- `fold_metrics.csv` and `development_out_of_fold_predictions.csv`;
- `final_untouched_test/final_model.keras` and `preprocessing.joblib`;
- `final_untouched_test/parameter_accounting_by_variable.csv`;
- `final_untouched_test/model_complexity.json`: total, trainable and
  non-trainable parameters, network GFLOPs/image, approximate GMACs/image,
  weight memory and an estimated Adam parameter-memory footprint;
- `final_untouched_test/final_model_summary.txt`: exact Keras model summary;
- `final_untouched_test/raw_image_to_prediction_benchmark/`: raw timings and
  summaries for batch sizes 1, 8 and 16;
- `environment_manifest.json`, `method_manifest.json`, and `protocol.json`.

## Which reviewer requests this file resolves

| Reviewer concern | Status after this run |
|---|---|
| Explain how old train/validation folders enter nested CV | Resolved and auditable |
| Keep final 180-patient test cohort untouched | Resolved and checked |
| Fit BGWO scaler/PCA inside its patient-grouped splits | Resolved |
| Identify BGWO classifier, objective and final mask refit | Resolved |
| Save masks, PCA dimensions, fitted objects, predictions and seeds | Resolved locally; publish them in a versioned repository/release |
| Measure raw-image-to-prediction latency | Resolved by the included benchmark |
| Properly calibrated lightweight CNN with inner-validation temperature scaling | **Not resolved by this primary-model file** |
| Matched fixed/global/quality gates with identical head and loss | **Not resolved by this primary-model file** |
| Randomized/shuffled-quality negative control and quality-stratified gate analysis | **Not resolved by this primary-model file** |
| External multicentre clinical validation | **Not resolved by this dataset** |

Therefore, this corrected QGECAF run is necessary but not sufficient to answer
every reviewer query. A second locked-fold control experiment must compare:

1. MobileNetV2 + softmax;
2. the same CNN with temperature scaling fitted only on inner-validation data;
3. bidirectional fusion with fixed averaging;
4. bidirectional fusion with a learned global gate;
5. bidirectional fusion with the real sample-specific quality gate;
6. the same quality-gated architecture with quality vectors shuffled within
   the inner-training set and kept fixed for validation/holdout evaluation;
7. optionally, the same-capacity gate driven by random noise.

All six control configurations must use the same head/loss within the gate
comparison, the same locked patients, and paired fold-wise analysis. Report
macro F1, Brier score, NLL and ECE with patient-level bootstrap confidence
intervals; examine gate distributions and performance within quality strata
whose thresholds are defined from inner-training data only.

