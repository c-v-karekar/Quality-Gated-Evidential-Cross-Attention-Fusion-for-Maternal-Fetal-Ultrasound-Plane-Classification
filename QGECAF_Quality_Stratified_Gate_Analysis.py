"""Automated leakage-free quality-stratified analysis for completed QGECAF.

This program does not train or modify a model. For each of the ten completed
QGECAF folds it:

1. loads the exact saved inner-training partition;
2. loads the quality scaler fitted only on that inner-training partition;
3. reconstructs the same composite quality score used by QGECAF;
4. calculates the 33rd and 67th percentiles from inner-training scores only;
5. applies those fixed thresholds to the corresponding outer holdout;
6. reports discrimination, probability quality, calibration, patient counts,
   class coverage and learned gate-lambda distributions; and
7. calculates patient-clustered bootstrap confidence intervals.

The untouched final-test cohort is never used in thresholding or evaluation.
An optional shuffled-gate result root can be supplied later for G2-versus-G3
comparison without changing the quality thresholds.

Run from Jupyter:

    from QGECAF_Quality_Stratified_Gate_Analysis import Config, run
    cfg = Config()
    run(cfg)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, rankdata, spearmanr, t, wilcoxon
from sklearn.metrics import accuracy_score, f1_score


PROTOCOL_VERSION = "qgecaf_quality_stratified_gate_analysis_v1"
GROUP_ORDER = ("Low", "Medium", "High")
QUALITY_NAMES = (
    "blur",
    "brightness",
    "contrast",
    "entropy",
    "noise",
    "background",
)
PRIMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "brier_multiclass",
    "ece_15_bins",
    "gate_lambda_mean",
)


@dataclass
class Config:
    qgecaf_root: str = (
        r"D:\chaitali\qgacef"
        r"\QGECAF_REVIEWER_REVISION_NESTED_BGWO"
    )
    output: str = (
        r"D:\chaitali\qgacef"
        r"\QGECAF_REVIEWER_REVISION_NESTED_BGWO"
        r"\quality_stratified_gate_analysis"
    )
    shuffled_gate_root: str = ""
    outer_folds: int = 10
    num_classes: int = 6
    lower_percentile: float = 33.0
    upper_percentile: float = 67.0
    ece_bins: int = 15
    bootstrap_repetitions: int = 2000
    seed: int = 42
    overwrite: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qgecaf-root", default=Config.qgecaf_root)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--shuffled-gate-root", default="")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", dest="overwrite")
    return Config(**vars(parser.parse_args()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_patient(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def record_key(patient_id: object, path: object) -> str:
    return f"{normalize_patient(patient_id)}|{Path(str(path)).stem.casefold()}"


def add_record_key(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    required = {"patient_id", "path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{description} lacks columns {sorted(missing)}")
    result = frame.copy()
    result["patient_id"] = result["patient_id"].map(normalize_patient)
    result["record_key"] = [
        record_key(patient, path)
        for patient, path in zip(result["patient_id"], result["path"])
    ]
    duplicated = result["record_key"].duplicated(keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, "record_key"].head(10).tolist()
        raise RuntimeError(f"Duplicate records in {description}: {examples}")
    return result


def quality_score_from_standardized(quality_z: np.ndarray) -> np.ndarray:
    """Exact composite quality-score definition used by completed QGECAF."""
    quality_z = np.asarray(quality_z, dtype=np.float64)
    if quality_z.ndim != 2 or quality_z.shape[1] != len(QUALITY_NAMES):
        raise ValueError(
            f"Expected standardized quality shape (N, 6), got {quality_z.shape}."
        )
    components = np.column_stack(
        [
            quality_z[:, 0],
            -np.abs(quality_z[:, 1]),
            quality_z[:, 2],
            quality_z[:, 3],
            -quality_z[:, 4],
            -quality_z[:, 5],
        ]
    )
    raw_score = components.mean(axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(raw_score, -20.0, 20.0)))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> Tuple[float, int]:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    occupied = 0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            occupied += 1
            ece += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(ece), occupied


def metric_bundle(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
    ece_bins: int,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), num_classes):
        raise ValueError(
            f"Probability shape {probabilities.shape} does not match "
            f"({len(labels)}, {num_classes})."
        )
    row_sums = probabilities.sum(axis=1)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise RuntimeError("Probabilities contain invalid values.")
    if not np.allclose(row_sums, 1.0, atol=2e-5):
        raise RuntimeError("Probability rows do not sum to one.")
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    true_probability = probabilities[np.arange(len(labels)), labels]
    ece, occupied = expected_calibration_error(labels, probabilities, ece_bins)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(
            f1_score(
                labels,
                prediction,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "negative_log_likelihood": float(
            -np.mean(np.log(np.clip(true_probability, 1e-12, 1.0)))
        ),
        "brier_multiclass": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "ece_15_bins": ece,
        "ece_occupied_bins": int(occupied),
    }


def detect_probability_columns(
    predictions: pd.DataFrame,
    class_names: Sequence[str],
) -> List[str]:
    expected = [f"probability_{name}" for name in class_names]
    if all(column in predictions.columns for column in expected):
        return expected
    generic = [f"probability_{index}" for index in range(len(class_names))]
    if all(column in predictions.columns for column in generic):
        return generic
    available = [
        column for column in predictions.columns
        if str(column).startswith("probability_")
    ]
    raise ValueError(
        "Could not identify ordered probability columns. "
        f"Expected {expected} or {generic}; found {available}."
    )


def read_reference(root: Path, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    manifest_file = root / "locked_patient_fold_assignments.csv"
    cache_file = root / "raw_features_presplit.npz"
    if not root.is_dir():
        raise FileNotFoundError(f"QGECAF result directory not found: {root}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Locked manifest not found: {manifest_file}")
    if not cache_file.is_file():
        raise FileNotFoundError(f"Raw feature cache not found: {cache_file}")
    manifest = pd.read_csv(manifest_file)
    required = {
        "path", "patient_id", "label", "class_name",
        "original_split", "outer_fold",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest lacks columns {sorted(missing)}")
    manifest = add_record_key(manifest, "locked patient manifest")
    manifest["label"] = pd.to_numeric(manifest["label"], errors="raise").astype(int)
    manifest["outer_fold"] = pd.to_numeric(
        manifest["outer_fold"], errors="raise"
    ).astype(int)
    observed_folds = set(manifest["outer_fold"].unique())
    expected_folds = {-1, *range(1, cfg.outer_folds + 1)}
    if observed_folds != expected_folds:
        raise RuntimeError(
            f"Manifest folds {sorted(observed_folds)} != {sorted(expected_folds)}."
        )
    patient_folds = manifest[["patient_id", "outer_fold"]].drop_duplicates()
    inconsistent = patient_folds.groupby("patient_id")["outer_fold"].nunique()
    if (inconsistent > 1).any():
        raise RuntimeError("One or more patients occur in multiple outer folds.")
    mapping = manifest[["label", "class_name"]].drop_duplicates().sort_values("label")
    if mapping["label"].duplicated().any() or mapping["class_name"].duplicated().any():
        raise RuntimeError("Class labels and class names are not one-to-one.")
    class_names = mapping["class_name"].astype(str).tolist()
    if len(class_names) != cfg.num_classes:
        raise RuntimeError(
            f"Expected {cfg.num_classes} classes, found {len(class_names)}."
        )
    return manifest, class_names


def load_raw_quality(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_file = root / "raw_features_presplit.npz"
    with np.load(cache_file, allow_pickle=True) as cache:
        required = {"quality", "paths", "labels"}
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(f"Raw cache lacks arrays: {sorted(missing)}")
        quality = np.asarray(cache["quality"], dtype=np.float32)
        paths = np.asarray(cache["paths"], dtype=object)
        labels = np.asarray(cache["labels"], dtype=int)
    if quality.ndim != 2 or quality.shape[1] != 6:
        raise RuntimeError(f"Unexpected raw quality shape: {quality.shape}")
    if not (len(quality) == len(paths) == len(labels)):
        raise RuntimeError("Raw quality cache arrays have inconsistent lengths.")
    return quality, paths, labels


def load_fold_inputs(
    root: Path,
    fold: int,
    raw_quality: np.ndarray,
    raw_paths: np.ndarray,
    raw_labels: np.ndarray,
    manifest: pd.DataFrame,
    class_names: Sequence[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float, List[str]]:
    fold_dir = root / "cross_validation" / f"fold_{fold:02d}"
    audit_file = fold_dir / "patient_partition_audit.csv"
    preprocessing_file = fold_dir / "preprocessing.joblib"
    prediction_file = fold_dir / "outer_holdout_predictions.csv"
    for path in (audit_file, preprocessing_file, prediction_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required fold artifact not found: {path}")

    audit = pd.read_csv(audit_file)
    required_audit = {"partition", "image_index", "patient_id", "path"}
    missing = required_audit.difference(audit.columns)
    if missing:
        raise ValueError(f"{audit_file} lacks columns {sorted(missing)}")
    audit = add_record_key(audit, str(audit_file))
    audit["image_index"] = pd.to_numeric(
        audit["image_index"], errors="raise"
    ).astype(int)
    if np.any(audit["image_index"].to_numpy() < 0) or np.any(
        audit["image_index"].to_numpy() >= len(raw_paths)
    ):
        raise RuntimeError(f"Fold {fold} audit contains invalid image indices.")
    for row in audit.itertuples(index=False):
        raw_index = int(row.image_index)
        if Path(str(raw_paths[raw_index])).stem.casefold() != Path(str(row.path)).stem.casefold():
            raise RuntimeError(
                f"Fold {fold} cache/audit path mismatch at image_index={raw_index}."
            )

    partitions = set(audit["partition"].astype(str))
    expected_partitions = {"inner_train", "inner_validation", "outer_holdout"}
    if partitions != expected_partitions:
        raise RuntimeError(
            f"Fold {fold} partitions {sorted(partitions)} != "
            f"{sorted(expected_partitions)}."
        )
    patient_sets = {
        name: set(audit.loc[audit["partition"] == name, "patient_id"])
        for name in expected_partitions
    }
    for left, right in (
        ("inner_train", "inner_validation"),
        ("inner_train", "outer_holdout"),
        ("inner_validation", "outer_holdout"),
    ):
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            raise RuntimeError(
                f"Fold {fold}: patient leakage between {left} and {right}: "
                f"{sorted(overlap)[:10]}"
            )

    audit_manifest = audit.merge(
        manifest[["record_key", "outer_fold", "label", "class_name"]],
        on="record_key",
        how="left",
        validate="one_to_one",
    )
    if audit_manifest["outer_fold"].isna().any():
        raise RuntimeError(f"Fold {fold}: audit records missing from manifest.")
    audit_manifest["outer_fold"] = audit_manifest["outer_fold"].astype(int)
    if np.any(audit_manifest["outer_fold"].to_numpy() == -1):
        raise RuntimeError(
            f"Fold {fold}: untouched final-test data entered quality analysis."
        )
    outer_check = audit_manifest["partition"] == "outer_holdout"
    if not np.all(audit_manifest.loc[outer_check, "outer_fold"].to_numpy() == fold):
        raise RuntimeError(f"Fold {fold}: outer-holdout assignment is incorrect.")

    processors = joblib.load(preprocessing_file)
    if "quality_scaler" not in processors:
        raise RuntimeError(f"Fold {fold} preprocessing lacks quality_scaler.")
    scaler = processors["quality_scaler"]
    inner_audit = audit.loc[audit["partition"] == "inner_train"].copy()
    inner_indices = inner_audit["image_index"].to_numpy(int)
    inner_quality_z = scaler.transform(raw_quality[inner_indices])
    inner_scores = quality_score_from_standardized(inner_quality_z)
    q33, q67 = np.percentile(
        inner_scores, [cfg.lower_percentile, cfg.upper_percentile]
    ).astype(float)
    if not 0.0 <= q33 <= q67 <= 1.0:
        raise RuntimeError(f"Fold {fold}: invalid thresholds {q33}, {q67}.")

    predictions = pd.read_csv(prediction_file)
    predictions = add_record_key(predictions, str(prediction_file))
    required_predictions = {
        "true_index", "predicted_index", "quality_score", "quality_gate_lambda"
    }
    missing_predictions = required_predictions.difference(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"{prediction_file} lacks columns {sorted(missing_predictions)}"
        )
    outer_audit = audit.loc[audit["partition"] == "outer_holdout", [
        "record_key", "image_index", "patient_id", "path"
    ]].copy()
    outer = outer_audit.merge(
        predictions,
        on=["record_key", "patient_id", "path"],
        how="inner",
        validate="one_to_one",
        suffixes=("_audit", ""),
    )
    if len(outer) != len(outer_audit) or len(outer) != len(predictions):
        raise RuntimeError(
            f"Fold {fold}: outer predictions do not exactly match partition audit."
        )
    outer_indices = outer["image_index"].to_numpy(int)
    outer_quality_z = scaler.transform(raw_quality[outer_indices])
    recomputed_score = quality_score_from_standardized(outer_quality_z)
    saved_score = pd.to_numeric(outer["quality_score"], errors="raise").to_numpy(float)
    max_difference = float(np.max(np.abs(recomputed_score - saved_score)))
    if not np.allclose(recomputed_score, saved_score, rtol=1e-5, atol=1e-6):
        raise RuntimeError(
            f"Fold {fold}: recomputed/saved quality score mismatch; "
            f"maximum absolute difference={max_difference:.8g}."
        )
    outer["quality_score_recomputed"] = recomputed_score
    outer["quality_group"] = np.select(
        [recomputed_score < q33, recomputed_score <= q67],
        ["Low", "Medium"],
        default="High",
    )
    outer["quality_group"] = pd.Categorical(
        outer["quality_group"], categories=GROUP_ORDER, ordered=True
    )
    probability_columns = detect_probability_columns(predictions, class_names)
    outer["true_index"] = pd.to_numeric(
        outer["true_index"], errors="raise"
    ).astype(int)
    if not np.array_equal(raw_labels[outer_indices], outer["true_index"].to_numpy()):
        raise RuntimeError(f"Fold {fold}: raw-cache and prediction labels differ.")
    if outer["true_index"].min() < 0 or outer["true_index"].max() >= cfg.num_classes:
        raise RuntimeError(f"Fold {fold}: invalid class labels.")
    return outer, inner_scores, recomputed_score, q33, q67, probability_columns


def stratified_row(
    fold: int,
    group_name: str,
    group: pd.DataFrame,
    probability_columns: Sequence[str],
    class_names: Sequence[str],
    q33: float,
    q67: float,
    cfg: Config,
) -> dict:
    labels = group["true_index"].to_numpy(int)
    probabilities = group[list(probability_columns)].to_numpy(float)
    metrics = metric_bundle(labels, probabilities, cfg.num_classes, cfg.ece_bins)
    gate = pd.to_numeric(group["quality_gate_lambda"], errors="raise").to_numpy(float)
    score = pd.to_numeric(
        group["quality_score_recomputed"], errors="raise"
    ).to_numpy(float)
    classes_present = sorted(np.unique(labels).tolist())
    missing_classes = sorted(set(range(cfg.num_classes)) - set(classes_present))
    q1, median, q3 = np.quantile(gate, [0.25, 0.50, 0.75])
    row = {
        "fold": fold,
        "quality_group": group_name,
        "inner_train_q33": q33,
        "inner_train_q67": q67,
        "threshold_lower": float("-inf") if group_name == "Low" else q33 if group_name == "Medium" else q67,
        "threshold_upper": q33 if group_name == "Low" else q67 if group_name == "Medium" else float("inf"),
        "threshold_lower_inclusive": bool(group_name == "Medium"),
        "threshold_upper_inclusive": bool(group_name == "Medium"),
        "threshold_rule": (
            "score < q33" if group_name == "Low" else
            "q33 <= score <= q67" if group_name == "Medium" else
            "score > q67"
        ),
        "images": len(group),
        "patients": int(group["patient_id"].nunique()),
        "classes_present": len(classes_present),
        "class_indices_present": json.dumps(classes_present),
        "class_names_present": json.dumps([class_names[i] for i in classes_present]),
        "missing_class_indices": json.dumps(missing_classes),
        "missing_class_names": json.dumps([class_names[i] for i in missing_classes]),
        "quality_score_mean": float(score.mean()),
        "quality_score_sd": float(score.std(ddof=1)) if len(score) > 1 else float("nan"),
        "quality_score_min": float(score.min()),
        "quality_score_max": float(score.max()),
        "gate_lambda_mean": float(gate.mean()),
        "gate_lambda_sd": float(gate.std(ddof=1)) if len(gate) > 1 else float("nan"),
        "gate_lambda_q1": float(q1),
        "gate_lambda_median": float(median),
        "gate_lambda_q3": float(q3),
        "gate_lambda_iqr": float(q3 - q1),
        "gate_lambda_min": float(gate.min()),
        "gate_lambda_max": float(gate.max()),
    }
    row.update(metrics)
    return row


def patient_cluster_bootstrap(
    group: pd.DataFrame,
    probability_columns: Sequence[str],
    cfg: Config,
    seed: int,
) -> Dict[str, float]:
    patients = group["patient_id"].astype(str).unique()
    if len(patients) < 2 or cfg.bootstrap_repetitions <= 0:
        return {
            f"{metric}_patient_bootstrap_ci95_{side}": float("nan")
            for metric in PRIMARY_METRICS
            for side in ("low", "high")
        }
    patient_rows = {
        patient: np.flatnonzero(group["patient_id"].astype(str).to_numpy() == patient)
        for patient in patients
    }
    labels_all = group["true_index"].to_numpy(int)
    probabilities_all = group[list(probability_columns)].to_numpy(float)
    gate_all = group["quality_gate_lambda"].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in PRIMARY_METRICS}
    for _ in range(cfg.bootstrap_repetitions):
        selected_patients = rng.choice(patients, size=len(patients), replace=True)
        selected_rows = np.concatenate([patient_rows[patient] for patient in selected_patients])
        labels = labels_all[selected_rows]
        probabilities = probabilities_all[selected_rows]
        metrics = metric_bundle(labels, probabilities, cfg.num_classes, cfg.ece_bins)
        for metric in PRIMARY_METRICS:
            value = (
                float(gate_all[selected_rows].mean())
                if metric == "gate_lambda_mean"
                else float(metrics[metric])
            )
            samples[metric].append(value)
    result: Dict[str, float] = {}
    for metric, values in samples.items():
        low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
        result[f"{metric}_patient_bootstrap_ci95_low"] = float(low)
        result[f"{metric}_patient_bootstrap_ci95_high"] = float(high)
    return result


def foldwise_correlations(
    fold: int,
    outer: pd.DataFrame,
) -> dict:
    score = outer["quality_score_recomputed"].to_numpy(float)
    gate = outer["quality_gate_lambda"].to_numpy(float)
    uncertainty = outer["uncertainty"].to_numpy(float)
    gate_rho, gate_p = spearmanr(score, gate)
    uncertainty_rho, uncertainty_p = spearmanr(score, uncertainty)
    return {
        "fold": fold,
        "images": len(outer),
        "patients": outer["patient_id"].nunique(),
        "quality_gate_spearman_rho": float(gate_rho),
        "quality_gate_spearman_p": float(gate_p),
        "quality_uncertainty_spearman_rho": float(uncertainty_rho),
        "quality_uncertainty_spearman_p": float(uncertainty_p),
    }


def aggregate_fold_summary(fold_rows: pd.DataFrame) -> pd.DataFrame:
    numeric_metrics = [
        "images", "patients", "classes_present", "accuracy", "macro_f1",
        "negative_log_likelihood", "brier_multiclass", "ece_15_bins",
        "quality_score_mean", "gate_lambda_mean", "gate_lambda_sd",
        "gate_lambda_median", "gate_lambda_iqr",
    ]
    rows = []
    for group_name in GROUP_ORDER:
        group = fold_rows.loc[fold_rows["quality_group"] == group_name]
        for metric in numeric_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            n = len(values)
            mean = float(values.mean()) if n else float("nan")
            sd = float(values.std(ddof=1)) if n > 1 else float("nan")
            half = (
                float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
                if n > 1 else float("nan")
            )
            rows.append({
                "quality_group": group_name,
                "metric": metric,
                "folds": n,
                "mean": mean,
                "standard_deviation": sd,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
            })
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    result = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index]
        running = max(running, candidate)
        result[index] = min(1.0, running)
    return result


def rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[values != 0]
    if len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values))
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    return float((positive - negative) / ranks.sum())


def quality_group_statistics(fold_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tested_metrics = (
        "gate_lambda_mean",
        "accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "brier_multiclass",
        "ece_15_bins",
    )
    omnibus_rows, pairwise_rows = [], []
    pairs = (("Low", "Medium"), ("Medium", "High"), ("Low", "High"))
    for metric in tested_metrics:
        pivot = fold_rows.pivot(index="fold", columns="quality_group", values=metric)
        pivot = pivot[list(GROUP_ORDER)].dropna()
        if len(pivot) < 3:
            continue
        friedman = friedmanchisquare(
            *[pivot[group].to_numpy(float) for group in GROUP_ORDER]
        )
        omnibus_rows.append({
            "metric": metric,
            "folds": len(pivot),
            "friedman_statistic": float(friedman.statistic),
            "friedman_p": float(friedman.pvalue),
        })
        local_rows = []
        for left, right in pairs:
            differences = pivot[left].to_numpy(float) - pivot[right].to_numpy(float)
            try:
                test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
                statistic, raw_p = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, raw_p = 0.0, 1.0
            local_rows.append({
                "metric": metric,
                "comparison": f"{left} vs {right}",
                "folds": len(pivot),
                "mean_difference_left_minus_right": float(differences.mean()),
                "median_difference_left_minus_right": float(np.median(differences)),
                "wilcoxon_statistic": statistic,
                "raw_p": raw_p,
                "rank_biserial_left_minus_right": rank_biserial(differences),
            })
        adjusted = holm_adjust([row["raw_p"] for row in local_rows])
        for row, adjusted_p in zip(local_rows, adjusted):
            row["holm_adjusted_p_within_metric"] = float(adjusted_p)
            row["significant_at_0.05"] = bool(adjusted_p < 0.05)
            pairwise_rows.append(row)
    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def plot_gate_distribution(oof: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(7.5, 6.0))
    sns.boxplot(
        data=oof,
        x="quality_group",
        y="quality_gate_lambda",
        order=GROUP_ORDER,
        showfliers=False,
        palette="Set2",
    )
    sns.stripplot(
        data=oof.sample(min(len(oof), 1500), random_state=42),
        x="quality_group",
        y="quality_gate_lambda",
        order=GROUP_ORDER,
        color="black",
        size=1.5,
        alpha=0.25,
    )
    plt.xlabel("Image-quality group")
    plt.ylabel(r"Learned quality gate $\lambda$")
    plt.title("Quality-gate distribution across leakage-free quality strata")
    plt.tight_layout()
    plt.savefig(output, dpi=600, bbox_inches="tight")
    plt.close()


def plot_metric_panels(fold_rows: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("accuracy", "Accuracy", True),
        ("macro_f1", "Macro F1", True),
        ("negative_log_likelihood", "Negative log-likelihood", False),
        ("ece_15_bins", "ECE (15 bins)", False),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (metric, label, _) in zip(axes.ravel(), metrics):
        sns.boxplot(
            data=fold_rows,
            x="quality_group",
            y=metric,
            order=GROUP_ORDER,
            ax=axis,
            color="#9ecae1",
        )
        sns.stripplot(
            data=fold_rows,
            x="quality_group",
            y=metric,
            order=GROUP_ORDER,
            ax=axis,
            color="black",
            size=4,
        )
        axis.set_xlabel("Quality group")
        axis.set_ylabel(label)
    figure.suptitle("Fold-wise QGECAF performance by image-quality group", y=1.01)
    figure.tight_layout()
    figure.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(figure)


def plot_quality_gate_scatter(oof: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(7.5, 6.0))
    plt.hexbin(
        oof["quality_score_recomputed"].to_numpy(float),
        oof["quality_gate_lambda"].to_numpy(float),
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    plt.colorbar(label="Images per hexagon")
    plt.xlabel("Composite image-quality score")
    plt.ylabel(r"Learned quality gate $\lambda$")
    plt.title("Quality score and learned fusion gate")
    plt.tight_layout()
    plt.savefig(output, dpi=600, bbox_inches="tight")
    plt.close()


def pooled_group_metrics(
    oof: pd.DataFrame,
    probability_columns: Sequence[str],
    class_names: Sequence[str],
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    for group_name in GROUP_ORDER:
        group = oof.loc[oof["quality_group"] == group_name]
        row = stratified_row(
            fold=0,
            group_name=group_name,
            group=group,
            probability_columns=probability_columns,
            class_names=class_names,
            q33=float("nan"),
            q67=float("nan"),
            cfg=cfg,
        )
        row["fold"] = "pooled_OOF_descriptive_only"
        rows.append(row)
    return pd.DataFrame(rows)


def optional_shuffled_comparison(
    cfg: Config,
    thresholds: pd.DataFrame,
    original_oof: pd.DataFrame,
    class_names: Sequence[str],
    output: Path,
) -> None:
    status_file = output / "shuffled_quality_control_status.json"
    if not cfg.shuffled_gate_root:
        status_file.write_text(
            json.dumps({
                "available": False,
                "reason": "No shuffled_gate_root was supplied.",
                "required_next_step": "Train matched G3 and rerun with its result root.",
            }, indent=2),
            encoding="utf-8",
        )
        return
    shuffled_root = Path(cfg.shuffled_gate_root)
    if not shuffled_root.is_dir():
        raise FileNotFoundError(f"Shuffled-gate root not found: {shuffled_root}")
    rows, predictions_all = [], []
    original_groups = original_oof[["record_key", "outer_fold", "quality_group"]]
    for fold in range(1, cfg.outer_folds + 1):
        candidates = [
            shuffled_root / "cross_validation" / f"fold_{fold:02d}" / "outer_holdout_predictions.csv",
            shuffled_root / "cross_validation" / f"fold_{fold:02d}" / "outer_holdout_predictions_shuffled.csv",
        ]
        prediction_file = next((path for path in candidates if path.is_file()), None)
        if prediction_file is None:
            raise FileNotFoundError(
                f"No shuffled-gate holdout prediction file found for fold {fold}."
            )
        predictions = add_record_key(pd.read_csv(prediction_file), str(prediction_file))
        predictions = predictions.merge(
            original_groups.loc[original_groups["outer_fold"] == fold],
            on="record_key",
            how="inner",
            validate="one_to_one",
            suffixes=("", "_original"),
        )
        if len(predictions) != len(original_groups.loc[original_groups["outer_fold"] == fold]):
            raise RuntimeError(f"G3 fold {fold} does not match G2 outer holdout.")
        predictions["outer_fold"] = fold
        probability_columns = detect_probability_columns(predictions, class_names)
        label_column = "true_index" if "true_index" in predictions else "true_label"
        predictions["true_index"] = predictions[label_column].astype(int)
        for group_name in GROUP_ORDER:
            group = predictions.loc[predictions["quality_group"] == group_name]
            rows.append(stratified_row(
                fold, group_name, group, probability_columns,
                class_names, float("nan"), float("nan"), cfg,
            ))
        predictions_all.append(predictions)
    shuffled_rows = pd.DataFrame(rows)
    shuffled_rows.to_csv(output / "G3_shuffled_quality_stratified_fold_metrics.csv", index=False)
    comparison_rows = []
    original_fold = pd.read_csv(output / "quality_stratified_fold_metrics.csv")
    for metric in (
        "accuracy", "macro_f1", "negative_log_likelihood",
        "brier_multiclass", "ece_15_bins",
    ):
        left = original_fold.pivot(index="fold", columns="quality_group", values=metric)
        right = shuffled_rows.pivot(index="fold", columns="quality_group", values=metric)
        for group_name in GROUP_ORDER:
            differences = left[group_name].to_numpy(float) - right[group_name].to_numpy(float)
            try:
                test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
                statistic, raw_p = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, raw_p = 0.0, 1.0
            comparison_rows.append({
                "metric": metric,
                "quality_group": group_name,
                "mean_difference_G2_minus_G3": float(differences.mean()),
                "wilcoxon_statistic": statistic,
                "raw_p": raw_p,
                "rank_biserial_G2_minus_G3": rank_biserial(differences),
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison["holm_adjusted_p_all_G2_G3_tests"] = holm_adjust(
        comparison["raw_p"].to_numpy(float)
    )
    comparison["significant_at_0.05"] = (
        comparison["holm_adjusted_p_all_G2_G3_tests"] < 0.05
    )
    comparison.to_csv(output / "G2_vs_G3_quality_stratified_comparison.csv", index=False)
    pd.concat(predictions_all, ignore_index=True).to_csv(
        output / "G3_shuffled_quality_oof_predictions_with_G2_strata.csv", index=False
    )
    status_file.write_text(
        json.dumps({"available": True, "shuffled_gate_root": str(shuffled_root)}, indent=2),
        encoding="utf-8",
    )


def write_excel_workbook(output: Path, tables: Dict[str, pd.DataFrame]) -> str:
    workbook = output / "QGECAF_quality_stratified_analysis.xlsx"
    try:
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            for sheet_name, frame in tables.items():
                frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        return str(workbook)
    except Exception as exc:
        (output / "excel_export_warning.txt").write_text(repr(exc), encoding="utf-8")
        return "unavailable; inspect excel_export_warning.txt"


def write_interpretation(
    fold_rows: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    correlations: pd.DataFrame,
    output: Path,
) -> None:
    pivot = fold_rows.pivot(
        index="fold", columns="quality_group", values="gate_lambda_mean"
    )
    high_minus_low = float((pivot["High"] - pivot["Low"]).mean())
    gate_test = omnibus.loc[omnibus["metric"] == "gate_lambda_mean"]
    gate_p = float(gate_test["friedman_p"].iloc[0]) if len(gate_test) else float("nan")
    rho_mean = float(correlations["quality_gate_spearman_rho"].mean())
    lines = [
        "AUTOMATED DESCRIPTIVE INTERPRETATION — VERIFY BEFORE MANUSCRIPT USE",
        "",
        f"Mean fold-wise gate difference (High minus Low quality): {high_minus_low:.6f}",
        f"Friedman p-value for mean gate lambda across quality groups: {gate_p:.6g}",
        f"Mean fold-wise Spearman rho (quality score vs gate lambda): {rho_mean:.6f}",
        "",
    ]
    if np.isfinite(gate_p) and gate_p < 0.05:
        lines.append(
            "Gate values differed across quality strata at the unadjusted omnibus "
            "0.05 level. This supports quality-dependent gate behaviour, but does "
            "not by itself establish that the gate improves classification."
        )
    else:
        lines.append(
            "The analysis did not detect a statistically significant systematic "
            "difference in mean gate values across quality strata. Strong claims "
            "of quality-adaptive weighting should therefore be avoided unless the "
            "matched G2-versus-G3 control supplies additional evidence."
        )
    lines.extend([
        "",
        "Causal attribution requires matched fixed/global/shuffled gate controls.",
        "Pooled OOF results are descriptive; inferential tests use fold-wise values.",
        "Patient-clustered bootstrap intervals account for multiple images per patient.",
    ])
    (output / "automated_interpretation.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def validate_output_protocol(cfg: Config, root: Path, output: Path) -> None:
    output_preexisted = output.exists()
    output_was_nonempty = (
        output_preexisted and output.is_dir() and any(output.iterdir())
    )
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output / "analysis_protocol.json"
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "qgecaf_root": str(root),
        "locked_manifest_sha256": sha256_file(root / "locked_patient_fold_assignments.csv"),
        "raw_feature_cache_sha256": sha256_file(root / "raw_features_presplit.npz"),
        "quality_score_formula": (
            "sigmoid(mean([blur_z,-abs(brightness_z),contrast_z,entropy_z,"
            "-noise_z,-background_z]))"
        ),
        "threshold_source": "each fold inner-training partition only",
        "lower_percentile": cfg.lower_percentile,
        "upper_percentile": cfg.upper_percentile,
        "macro_f1_labels": list(range(cfg.num_classes)),
        "ece_bins": cfg.ece_bins,
        "bootstrap_unit": "patient",
        "bootstrap_repetitions": cfg.bootstrap_repetitions,
        "untouched_test_policy": "not accessed",
    }
    if output_was_nonempty and not protocol_file.is_file() and not cfg.overwrite:
        raise RuntimeError(
            f"Output directory is non-empty and lacks analysis_protocol.json: {output}"
        )
    if protocol_file.is_file():
        observed = json.loads(protocol_file.read_text(encoding="utf-8"))
        if observed != expected and not cfg.overwrite:
            raise RuntimeError(
                "Existing analysis uses a different protocol. Select a new output "
                "directory or explicitly enable overwrite."
            )
    protocol_file.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    (output / "configuration.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output / "analysis_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


def run(cfg: Config | None = None) -> None:
    cfg = Config() if cfg is None else cfg
    if not 0 < cfg.lower_percentile < cfg.upper_percentile < 100:
        raise ValueError("Quality percentiles must satisfy 0 < lower < upper < 100.")
    if cfg.bootstrap_repetitions < 0:
        raise ValueError("bootstrap_repetitions must be non-negative.")
    root = Path(cfg.qgecaf_root)
    output = Path(cfg.output)
    manifest, class_names = read_reference(root, cfg)
    validate_output_protocol(cfg, root, output)
    raw_quality, raw_paths, raw_labels = load_raw_quality(root)
    if len(raw_paths) != len(manifest):
        raise RuntimeError(
            f"Raw cache has {len(raw_paths)} images but manifest has {len(manifest)}."
        )

    threshold_rows, fold_rows, correlation_rows, oof_frames = [], [], [], []
    reference_probability_columns: List[str] | None = None
    for fold in range(1, cfg.outer_folds + 1):
        print("\n" + "=" * 78)
        print(f"QUALITY-STRATIFIED ANALYSIS — OUTER FOLD {fold}/{cfg.outer_folds}")
        print("=" * 78)
        outer, inner_scores, outer_scores, q33, q67, probability_columns = load_fold_inputs(
            root, fold, raw_quality, raw_paths, raw_labels,
            manifest, class_names, cfg,
        )
        if reference_probability_columns is None:
            reference_probability_columns = list(probability_columns)
        elif list(probability_columns) != reference_probability_columns:
            raise RuntimeError("Probability column ordering differs across folds.")
        counts = outer["quality_group"].value_counts().reindex(GROUP_ORDER, fill_value=0)
        threshold_rows.append({
            "fold": fold,
            "inner_train_images": len(inner_scores),
            "inner_train_q33": q33,
            "inner_train_q67": q67,
            "inner_train_quality_min": float(inner_scores.min()),
            "inner_train_quality_max": float(inner_scores.max()),
            "outer_holdout_images": len(outer),
            "outer_low_images": int(counts["Low"]),
            "outer_medium_images": int(counts["Medium"]),
            "outer_high_images": int(counts["High"]),
        })
        outer["outer_fold"] = fold
        oof_frames.append(outer)
        correlation_rows.append(foldwise_correlations(fold, outer))
        for group_index, group_name in enumerate(GROUP_ORDER):
            group = outer.loc[outer["quality_group"] == group_name].copy()
            if group.empty:
                raise RuntimeError(
                    f"Fold {fold} produced an empty {group_name} quality group."
                )
            row = stratified_row(
                fold, group_name, group, probability_columns,
                class_names, q33, q67, cfg,
            )
            row.update(patient_cluster_bootstrap(
                group,
                probability_columns,
                cfg,
                seed=cfg.seed + fold * 100 + group_index,
            ))
            fold_rows.append(row)
        print(
            f"Thresholds: q33={q33:.6f}, q67={q67:.6f}; "
            f"holdout groups={counts.to_dict()}"
        )

    thresholds = pd.DataFrame(threshold_rows).sort_values("fold")
    stratified = pd.DataFrame(fold_rows)
    stratified["quality_group"] = pd.Categorical(
        stratified["quality_group"], categories=GROUP_ORDER, ordered=True
    )
    stratified = stratified.sort_values(["fold", "quality_group"])
    correlations = pd.DataFrame(correlation_rows).sort_values("fold")
    oof = pd.concat(oof_frames, ignore_index=True)
    oof["quality_group"] = pd.Categorical(
        oof["quality_group"], categories=GROUP_ORDER, ordered=True
    )
    if len(oof) != int(np.sum(manifest["outer_fold"] != -1)):
        raise RuntimeError("Quality-stratified OOF predictions are incomplete.")
    if oof["record_key"].duplicated().any():
        raise RuntimeError("Quality-stratified OOF predictions contain duplicates.")
    if set(oof["outer_fold"].astype(int)) != set(range(1, cfg.outer_folds + 1)):
        raise RuntimeError("Quality-stratified OOF predictions have incomplete folds.")

    summary = aggregate_fold_summary(stratified)
    omnibus, pairwise = quality_group_statistics(stratified)
    pooled = pooled_group_metrics(
        oof,
        reference_probability_columns or [],
        class_names,
        cfg,
    )
    thresholds.to_csv(output / "quality_thresholds_by_fold.csv", index=False)
    stratified.to_csv(output / "quality_stratified_fold_metrics.csv", index=False)
    summary.to_csv(output / "quality_stratified_10fold_summary.csv", index=False)
    pooled.to_csv(output / "quality_stratified_pooled_oof_metrics.csv", index=False)
    correlations.to_csv(output / "foldwise_quality_gate_correlations.csv", index=False)
    omnibus.to_csv(output / "quality_group_friedman_tests.csv", index=False)
    pairwise.to_csv(output / "quality_group_paired_wilcoxon_holm.csv", index=False)
    oof.to_csv(output / "quality_stratified_oof_predictions.csv", index=False)

    plot_gate_distribution(oof, output / "quality_gate_distribution_boxplot.png")
    plot_metric_panels(stratified, output / "quality_stratified_metric_panels.png")
    plot_quality_gate_scatter(oof, output / "quality_score_vs_gate_lambda.png")
    write_interpretation(stratified, omnibus, pairwise, correlations, output)
    optional_shuffled_comparison(
        cfg, thresholds, oof, class_names, output
    )
    workbook = write_excel_workbook(output, {
        "thresholds": thresholds,
        "fold_metrics": stratified,
        "tenfold_summary": summary,
        "pooled_descriptive": pooled,
        "fold_correlations": correlations,
        "friedman": omnibus,
        "wilcoxon_holm": pairwise,
    })
    (output / "RUN_COMPLETE.json").write_text(
        json.dumps({
            "protocol_version": PROTOCOL_VERSION,
            "folds": cfg.outer_folds,
            "development_oof_images": len(oof),
            "untouched_final_test_used_for_thresholding_or_evaluation": False,
            "model_retrained": False,
            "quality_threshold_source": "inner training only",
            "patient_clustered_bootstrap_repetitions": cfg.bootstrap_repetitions,
            "excel_workbook": workbook,
        }, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 78)
    print("QUALITY-STRATIFIED GATE ANALYSIS COMPLETE")
    print("=" * 78)
    print("Results:", output)
    print("QGECAF retrained: NO")
    print("Untouched final test used for thresholding/evaluation: NO")
    print("\nTen-fold summary:")
    print(summary)
    print("\nGate/performance omnibus tests:")
    print(omnibus)


if __name__ == "__main__":
    run(parse_args())
