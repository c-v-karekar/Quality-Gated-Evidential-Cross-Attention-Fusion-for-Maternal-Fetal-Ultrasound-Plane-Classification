# QGECAF G2 matched rerun and gate-behaviour analysis

## Purpose

This verification package trains a new `G2_matched_rerun` on the exact ten
locked patient folds used by G0, G1 and G3. It does not overwrite the original
G2 or any existing control. It does not evaluate the untouched final-test
cohort.

The runner loads each fold's saved BGWO masks, feature scalers, PCA objects and
quality scaler without refitting them. G2-R uses the same fold seed
(`42 + fold`), MobileNetV2 configuration, handcrafted feature dimensions,
bidirectional cross-attention, sample-specific quality MLP, evidential head,
KL weight, optimizer, checkpoint rule, early stopping and learning-rate
schedule used by the matched controls. The only distinction from G3 is that
G2-R retains the correct image-to-quality correspondence.

## Files

- `Run_QGECAF_G2_Matched_Rerun_and_Gate_Analysis.ipynb`
- `QGECAF_G2_Matched_Rerun_and_Gate_Analysis.py`
- `QGECAF_Reviewer_Revision_Nested_BGWO.py`

Keep all three in the same directory.

## Required completed results

The default paths expect:

```text
D:\chaitali\qgacef\QGECAF_REVIEWER_REVISION_NESTED_BGWO
D:\chaitali\qgacef\QGECAF_MATCHED_GATE_CONTROLS
```

The new results are written separately to:

```text
D:\chaitali\qgacef\QGECAF_G2_MATCHED_RERUN
```

Do not set the new output path equal to either existing results directory.

## Execution

1. Activate `chaitalienv` in Anaconda Prompt.
2. Change to the extracted package folder.
3. Start Jupyter Notebook.
4. Select the `chaitalienv` kernel.
5. Restart the kernel before running the notebook.
6. Run every cell in order.

The run is resumable. A completed G2-R fold is checked against its saved
preprocessing checksum and outer-holdout records before it is skipped.

## Principal outputs

| Output | Purpose |
|---|---|
| `all_gate_controls_with_G2R_fold_metrics.csv` | Original controls, original G2 and G2-R fold metrics |
| `gate_controls_with_G2R_mean_sd_ci95.csv` | Mean, sample SD and 95% CI |
| `G2R_vs_controls_wilcoxon_holm.csv` | Paired G2-R comparisons with G0, G1 and G3 |
| `G2R_vs_original_G2_verification.csv` | Verification comparison; not a model-selection table |
| `quality_stratified_gate_controls.csv` | Fold × quality-group performance and gate statistics |
| `quality_stratified_mean_sd_ci95.csv` | Fold-level quality-stratified summary |
| `all_outer_holdout_gate_values.csv` | Image-level outer-holdout quality and gate values |
| `pooled_gate_distribution_summary.csv` | Gate mean, SD, median, quartiles and IQR |
| `gate_quality_spearman_by_fold.csv` | Fold-wise quality--gate association |
| `gate_quality_spearman_summary.csv` | Ten-fold correlation summary |
| `gate_lambda_quality_group_friedman.csv` | Omnibus low/medium/high gate test |
| `gate_lambda_quality_group_wilcoxon_holm.csv` | Holm-adjusted paired group comparisons |
| `gate_lambda_by_quality_group.png` | 600-dpi gate-distribution figure |

## Interpretation rule

Use G2-R only to check reproducibility under the matched control pipeline. Do
not choose whichever of the original G2 and G2-R has the better result. Report
both transparently. A meaningful quality-adaptation claim requires systematic
changes in gate weight across leakage-free quality groups and preferably a
performance or calibration advantage over G3. Non-significant comparisons
must not be described as superiority or equivalence.
