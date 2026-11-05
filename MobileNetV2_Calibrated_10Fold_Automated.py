"""Leakage-safe MobileNetV2 benchmark for the revised QGECAF experiment.

This program performs the mandatory reviewer control experiment:

1. Reuses the exact saved QGECAF inner-training, inner-validation, and
   outer-holdout patient partitions for each of ten folds.
2. Trains a fresh ImageNet-initialized MobileNetV2 softmax/logit classifier.
3. Saves inner-validation and outer-holdout logits.
4. Fits one positive temperature using inner-validation logits only.
5. Evaluates uncalibrated and temperature-scaled probabilities on the same
   outer holdout.
6. Produces fold summaries, OOF predictions, reliability diagrams, model
   complexity, and paired comparisons against the completed QGECAF run.

The untouched cohort (outer_fold == -1) is deliberately rejected by every
training, validation, calibration, and evaluation partition in this file.

Designed for Python 3.10 / TensorFlow 2.10 on Windows. Run from Jupyter:

    from MobileNetV2_Calibrated_10Fold_Automated import Config, run
    cfg = Config()
    run(cfg)

or from Anaconda Prompt:

    python MobileNetV2_Calibrated_10Fold_Automated.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# TensorFlow 2.10 does not provide a deterministic GPU implementation for the
# UnsortedSegmentSum gradient used by sparse categorical cross-entropy.  Fixed
# seeds and deterministic tf.data ordering are retained, but strict GPU-op
# determinism must remain disabled for this benchmark to train on the GPU.
os.environ["TF_DETERMINISTIC_OPS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, t, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROTOCOL_VERSION = "mobilenetv2_temperature_same_qgecaf_partitions_v1"
METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
    "mcc",
    "macro_specificity",
    "ece_15_bins",
    "brier_multiclass",
    "negative_log_likelihood",
    "uncertainty_error_auroc",
)


@dataclass
class Config:
    qgecaf_root: str = (
        r"D:\chaitali\qgacef"
        r"\QGECAF_REVIEWER_REVISION_NESTED_BGWO"
    )
    output: str = (
        r"D:\chaitali\qgacef"
        r"\MOBILENETV2_CALIBRATED_SAME_PATIENT_FOLDS"
    )
    image_size: int = 128
    num_classes: int = 6
    outer_folds: int = 10
    seed: int = 42
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    fine_tune_last: int = 30
    dropout: float = 0.30
    early_stopping_patience: int = 8
    temperature_min: float = 0.05
    temperature_max: float = 100.0
    ece_bins: int = 15
    bootstrap_repetitions: int = 10000
    resume: bool = True
    smoke_test: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qgecaf-root", default=Config.qgecaf_root)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--fine-tune-last", type=int, default=Config.fine_tune_last)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--smoke-test", action="store_true")
    return Config(**vars(parser.parse_args()))


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"TensorFlow {tf.__version__}; GPUs detected: {len(gpus)}")
    print(
        "Reproducibility: fixed seeds and deterministic input ordering; "
        "strict GPU-op determinism disabled for TensorFlow 2.10 compatibility."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_patient(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def record_key(patient_id: object, path: object) -> str:
    return f"{normalized_patient(patient_id)}|{Path(str(path)).stem.casefold()}"


def ensure_unique_keys(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    result = frame.copy()
    result["patient_id"] = result["patient_id"].map(normalized_patient)
    result["record_key"] = [
        record_key(patient, path)
        for patient, path in zip(result["patient_id"], result["path"])
    ]
    duplicated = result["record_key"].duplicated(keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, "record_key"].head(10).tolist()
        raise RuntimeError(f"Duplicate keys in {description}: {examples}")
    return result


def load_reference(cfg: Config) -> Tuple[pd.DataFrame, List[str], Path]:
    root = Path(cfg.qgecaf_root)
    manifest_file = root / "locked_patient_fold_assignments.csv"
    metrics_file = root / "fold_metrics.csv"
    if not root.is_dir():
        raise FileNotFoundError(f"QGECAF result root not found: {root}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Locked manifest not found: {manifest_file}")
    if not metrics_file.is_file():
        raise FileNotFoundError(f"QGECAF fold metrics not found: {metrics_file}")

    manifest = pd.read_csv(manifest_file)
    required = {
        "path", "patient_id", "label", "class_name",
        "original_split", "outer_fold",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Locked manifest is missing columns: {sorted(missing)}")
    manifest = ensure_unique_keys(manifest, "locked manifest")
    manifest["label"] = pd.to_numeric(manifest["label"], errors="raise").astype(int)
    manifest["outer_fold"] = pd.to_numeric(
        manifest["outer_fold"], errors="raise"
    ).astype(int)

    observed = set(manifest["outer_fold"].unique())
    expected = {-1, *range(1, cfg.outer_folds + 1)}
    if observed != expected:
        raise RuntimeError(
            f"Manifest folds are {sorted(observed)}; expected {sorted(expected)}."
        )
    patient_folds = manifest[["patient_id", "outer_fold"]].drop_duplicates()
    inconsistent = patient_folds.groupby("patient_id")["outer_fold"].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if len(inconsistent):
        raise RuntimeError(
            "A patient belongs to multiple outer folds: "
            f"{inconsistent.index[:10].tolist()}"
        )

    label_name = manifest[["label", "class_name"]].drop_duplicates()
    if label_name["label"].duplicated().any() or label_name["class_name"].duplicated().any():
        raise RuntimeError("Class labels and class names are not one-to-one.")
    label_name = label_name.sort_values("label")
    class_names = label_name["class_name"].astype(str).tolist()
    if len(class_names) != cfg.num_classes:
        raise RuntimeError(
            f"Manifest has {len(class_names)} classes; expected {cfg.num_classes}."
        )

    missing_paths = [value for value in manifest["path"] if not Path(str(value)).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} manifest image paths are unavailable. "
            f"First missing path: {missing_paths[0]}"
        )
    return manifest, class_names, manifest_file


def load_fold_partition(
    cfg: Config,
    manifest: pd.DataFrame,
    fold: int,
) -> Dict[str, pd.DataFrame]:
    audit_file = (
        Path(cfg.qgecaf_root)
        / "cross_validation"
        / f"fold_{fold:02d}"
        / "patient_partition_audit.csv"
    )
    if not audit_file.is_file():
        raise FileNotFoundError(f"Partition audit not found: {audit_file}")
    audit = pd.read_csv(audit_file)
    required = {"partition", "patient_id", "path"}
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"{audit_file} lacks columns: {sorted(missing)}")
    audit = ensure_unique_keys(audit, str(audit_file))

    merged = audit.merge(
        manifest[[
            "record_key", "label", "class_name", "outer_fold", "original_split"
        ]],
        on="record_key",
        how="left",
        validate="one_to_one",
    )
    if merged["label"].isna().any():
        examples = merged.loc[merged["label"].isna(), "record_key"].head().tolist()
        raise RuntimeError(f"Fold {fold} audit rows absent from manifest: {examples}")
    merged["label"] = merged["label"].astype(int)
    merged["outer_fold"] = merged["outer_fold"].astype(int)

    expected_partitions = {"inner_train", "inner_validation", "outer_holdout"}
    observed_partitions = set(merged["partition"].astype(str))
    if observed_partitions != expected_partitions:
        raise RuntimeError(
            f"Fold {fold} partitions are {sorted(observed_partitions)}; "
            f"expected {sorted(expected_partitions)}."
        )
    result = {
        name: merged.loc[merged["partition"] == name].copy().reset_index(drop=True)
        for name in expected_partitions
    }
    patient_sets = {
        name: set(frame["patient_id"].astype(str))
        for name, frame in result.items()
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
    if not np.all(result["outer_holdout"]["outer_fold"].to_numpy() == fold):
        raise RuntimeError(f"Fold {fold}: incorrect outer-holdout assignment.")
    for name in ("inner_train", "inner_validation"):
        values = result[name]["outer_fold"].to_numpy()
        if np.any(values == fold) or np.any(values == -1):
            raise RuntimeError(f"Fold {fold}: invalid rows in {name}.")
    combined = pd.concat(result.values(), ignore_index=True)
    if np.any(combined["outer_fold"].to_numpy() == -1):
        raise RuntimeError(
            f"Fold {fold}: untouched final-test rows entered the experiment."
        )
    return result


def decode_image(path: tf.Tensor, label: tf.Tensor, image_size: int):
    content = tf.io.read_file(path)
    image = tf.io.decode_image(content, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(
        image, [image_size, image_size], method=tf.image.ResizeMethod.AREA
    )
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(
    frame: pd.DataFrame,
    cfg: Config,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label"].astype(np.int32).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        lambda path, label: decode_image(path, label, cfg.image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(cfg.batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def compute_class_weights(labels: np.ndarray, num_classes: int) -> Dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(f"Training partition is missing classes: counts={counts}")
    weights = len(labels) / (num_classes * counts)
    weights = weights / weights.mean()
    return {index: float(value) for index, value in enumerate(weights)}


def build_mobilenetv2(cfg: Config) -> tf.keras.Model:
    image = tf.keras.layers.Input(
        (cfg.image_size, cfg.image_size, 1), name="image"
    )
    rgb = tf.keras.layers.Concatenate(name="grayscale_to_rgb")(
        [image, image, image]
    )
    rgb = tf.keras.layers.Rescaling(
        scale=2.0, offset=-1.0, name="mobilenet_preprocess"
    )(rgb)
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=None if cfg.smoke_test else "imagenet",
        input_shape=(cfg.image_size, cfg.image_size, 3),
    )
    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - cfg.fine_tune_last)
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    features = backbone(rgb)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(features)
    features = tf.keras.layers.Dense(
        128, activation="gelu", name="classifier_dense"
    )(features)
    features = tf.keras.layers.Dropout(
        cfg.dropout, name="classifier_dropout"
    )(features)
    logits = tf.keras.layers.Dense(
        cfg.num_classes, activation=None, name="class_logits"
    )(features)
    return tf.keras.Model(image, logits, name="MobileNetV2_Softmax_Baseline")


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self, output: Path):
        super().__init__()
        self.output = output
        self.rows: List[dict] = []
        self.started = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        row = {
            "epoch": int(epoch + 1),
            "epoch_time_seconds": float(time.perf_counter() - self.started),
        }
        row.update({key: float(value) for key, value in (logs or {}).items()})
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.output, index=False)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    labels: np.ndarray, probabilities: np.ndarray
) -> float:
    selected = probabilities[np.arange(len(labels)), labels.astype(int)]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def fit_temperature(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    cfg: Config,
) -> dict:
    before = negative_log_likelihood(
        validation_labels, softmax(validation_logits)
    )

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probabilities = softmax(validation_logits, temperature)
        return negative_log_likelihood(validation_labels, probabilities)

    result = minimize_scalar(
        objective,
        bounds=(math.log(cfg.temperature_min), math.log(cfg.temperature_max)),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    after = float(result.fun)
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(f"Invalid fitted temperature: {temperature}")
    return {
        "temperature": temperature,
        "validation_nll_before": before,
        "validation_nll_after": after,
        "optimization_success": bool(result.success),
        "optimization_iterations": int(result.nfev),
        "fit_partition": "inner_validation_only",
    }


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == 0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(
                correctness[selected].mean() - confidence[selected].mean()
            )
    return float(result)


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    cfg: Config,
) -> Tuple[dict, pd.DataFrame, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    uncertainty = 1.0 - confidence
    one_hot = np.eye(cfg.num_classes, dtype=np.float64)[labels]
    cm = confusion_matrix(labels, predictions, labels=np.arange(cfg.num_classes))
    total = cm.sum()
    class_rows = []
    specificities = []
    for class_index in range(cfg.num_classes):
        tp = int(cm[class_index, class_index])
        fn = int(cm[class_index, :].sum() - tp)
        fp = int(cm[:, class_index].sum() - tp)
        tn = int(total - tp - fn - fp)
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity
            else 0.0
        )
        specificities.append(specificity)
        class_rows.append({
            "class_index": class_index,
            "support": int(cm[class_index, :].sum()),
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "precision": precision,
            "sensitivity_recall": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
        })
    incorrect = (predictions != labels).astype(int)
    try:
        error_auc = float(roc_auc_score(incorrect, uncertainty))
    except ValueError:
        error_auc = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "macro_specificity": float(np.nanmean(specificities)),
        "ece_15_bins": expected_calibration_error(
            labels, probabilities, cfg.ece_bins
        ),
        "brier_multiclass": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "negative_log_likelihood": negative_log_likelihood(
            labels, probabilities
        ),
        "uncertainty_error_auroc": error_auc,
        "mean_uncertainty_correct": float(
            np.mean(uncertainty[predictions == labels])
        ),
        "mean_uncertainty_incorrect": (
            float(np.mean(uncertainty[predictions != labels]))
            if np.any(predictions != labels)
            else float("nan")
        ),
    }
    return metrics, pd.DataFrame(class_rows), cm


def prediction_frame(
    partition: pd.DataFrame,
    logits: np.ndarray,
    probabilities: np.ndarray,
    fold: int,
    calibration: str,
) -> pd.DataFrame:
    result = pd.DataFrame({
        "outer_fold": fold,
        "path": partition["path"].astype(str).to_numpy(),
        "patient_id": partition["patient_id"].astype(str).to_numpy(),
        "true_label": partition["label"].astype(int).to_numpy(),
        "class_name": partition["class_name"].astype(str).to_numpy(),
        "predicted_label": probabilities.argmax(axis=1),
        "confidence": probabilities.max(axis=1),
        "uncertainty": 1.0 - probabilities.max(axis=1),
        "calibration": calibration,
    })
    for index in range(logits.shape[1]):
        result[f"logit_{index}"] = logits[:, index]
        result[f"probability_{index}"] = probabilities[:, index]
    return result


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    output: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_reliability_comparison(
    labels: np.ndarray,
    probability_sets: Dict[str, np.ndarray],
    output: Path,
    bins: int = 15,
) -> None:
    edges = np.linspace(0.0, 1.0, bins + 1)
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    for name, probabilities in probability_sets.items():
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == labels
        mean_confidence, empirical_accuracy = [], []
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            selected = (
                (confidence >= lower) & (confidence <= upper)
                if index == 0
                else (confidence > lower) & (confidence <= upper)
            )
            if selected.any():
                mean_confidence.append(float(confidence[selected].mean()))
                empirical_accuracy.append(float(correct[selected].mean()))
        plt.plot(mean_confidence, empirical_accuracy, "o-", label=name)
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("MobileNetV2 out-of-fold reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def save_model_summary(model: tf.keras.Model, output: Path) -> None:
    lines: List[str] = []
    model.summary(print_fn=lines.append)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_complexity(model: tf.keras.Model, cfg: Config) -> dict:
    total = int(model.count_params())
    trainable = int(sum(np.prod(value.shape) for value in model.trainable_weights))
    nontrainable = int(
        sum(np.prod(value.shape) for value in model.non_trainable_weights)
    )
    flops = 0
    error = None
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )

        @tf.function
        def forward(image):
            return model(image, training=False)

        concrete = forward.get_concrete_function(
            tf.TensorSpec([1, cfg.image_size, cfg.image_size, 1], tf.float32)
        )
        frozen = convert_variables_to_constants_v2(concrete)
        options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        options["output"] = "none"
        profile = tf.compat.v1.profiler.profile(
            graph=frozen.graph,
            run_meta=tf.compat.v1.RunMetadata(),
            cmd="op",
            options=options,
        )
        flops = int(profile.total_float_ops) if profile is not None else 0
    except Exception as exc:
        error = repr(exc)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "nontrainable_parameters": nontrainable,
        "network_flops_per_image": flops,
        "network_gflops_per_image": flops / 1e9 if flops else float("nan"),
        "approximate_gmacs_per_image": flops / 2e9 if flops else float("nan"),
        "weight_memory_mib_float32": total * 4 / (1024 ** 2),
        "scope": "MobileNetV2 TensorFlow forward pass only",
        "profiler_error": error,
    }


def predict_logits(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, float]:
    warmup = dataset.take(1)
    _ = model.predict(warmup, verbose=0)
    started = time.perf_counter()
    logits = model.predict(dataset, verbose=1)
    elapsed = float(time.perf_counter() - started)
    return np.asarray(logits), elapsed


def summarize(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for metric in METRIC_COLUMNS:
        if metric not in frame.columns:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy()
        n = len(values)
        mean = float(np.mean(values)) if n else float("nan")
        sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
        half = (
            float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
            if n > 1
            else float("nan")
        )
        rows.append({
            "metric": metric,
            "folds": n,
            "mean": mean,
            "standard_deviation": sd,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
        })
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_bootstrap_ci(
    differences: np.ndarray,
    repetitions: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        differences, size=(repetitions, len(differences)), replace=True
    )
    means = sampled.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[values != 0]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values))
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    return (positive - negative) / float(ranks.sum())


def paired_comparison(
    cfg: Config,
    baseline_metrics: pd.DataFrame,
    qgecaf_metrics: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    selected_metrics = (
        "macro_f1",
        "negative_log_likelihood",
        "brier_multiclass",
        "ece_15_bins",
    )
    baseline = baseline_metrics.set_index("fold")
    qgecaf = qgecaf_metrics.set_index("fold")
    if set(baseline.index) != set(qgecaf.index) or len(baseline) != cfg.outer_folds:
        raise RuntimeError("Baseline and QGECAF folds are not identically paired.")
    rows = []
    for metric in selected_metrics:
        if metric not in baseline or metric not in qgecaf:
            raise ValueError(f"Paired comparison metric is missing: {metric}")
        q_values = qgecaf.loc[baseline.index, metric].to_numpy(float)
        b_values = baseline[metric].to_numpy(float)
        differences = q_values - b_values
        try:
            test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
            statistic, raw_p = float(test.statistic), float(test.pvalue)
        except ValueError:
            statistic, raw_p = 0.0, 1.0
        ci_low, ci_high = paired_bootstrap_ci(
            differences, cfg.bootstrap_repetitions, cfg.seed + 9000
        )
        rows.append({
            "metric": metric,
            "qgecaf_mean": float(q_values.mean()),
            "calibrated_mobilenetv2_mean": float(b_values.mean()),
            "mean_difference_qgecaf_minus_baseline": float(differences.mean()),
            "paired_bootstrap_ci95_low": ci_low,
            "paired_bootstrap_ci95_high": ci_high,
            "wilcoxon_statistic": statistic,
            "raw_p": raw_p,
            "rank_biserial_qgecaf_minus_baseline": rank_biserial(differences),
            "preferred_direction": (
                "higher" if metric == "macro_f1" else "lower"
            ),
        })
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = holm_adjust(result["raw_p"].to_numpy())
    result["significant_at_0.05"] = result["holm_adjusted_p"] < 0.05
    result.to_csv(output, index=False)
    return result


def validate_protocol(output: Path, cfg: Config, manifest_file: Path) -> None:
    output_preexisted = output.exists()
    output_was_nonempty = (
        output_preexisted and output.is_dir() and any(output.iterdir())
    )
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output / "protocol.json"
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "qgecaf_root": str(Path(cfg.qgecaf_root)),
        "locked_manifest_sha256": sha256_file(manifest_file),
        "outer_folds": cfg.outer_folds,
        "seed": cfg.seed,
        "model": "MobileNetV2 alpha=1.0 ImageNet",
        "head": "GAP-Dense128(GELU)-Dropout-Dense6(logits)",
        "temperature_fit_partition": "inner_validation_only",
        "untouched_test_policy": "outer_fold=-1 excluded from all operations",
    }
    if output_was_nonempty and not protocol_file.is_file():
        raise RuntimeError(
            f"Output directory is not empty but has no protocol.json: {output}\n"
            "Use a new empty directory to prevent mixing unrelated results."
        )
    if protocol_file.is_file():
        observed = json.loads(protocol_file.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                "Output directory belongs to a different protocol. Use a new "
                "empty output directory; do not mix experiments."
            )
    protocol_file.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    (output / "configuration.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gpu_devices": [
            device.name for device in tf.config.list_physical_devices("GPU")
        ],
        "strict_gpu_op_determinism": False,
        "reproducibility_note": (
            "Fixed Python/NumPy/TensorFlow seeds and deterministic tf.data "
            "ordering; strict deterministic GPU reductions are unavailable "
            "for the TensorFlow 2.10 sparse-cross-entropy gradient."
        ),
    }
    (output / "environment_manifest.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


def run(cfg: Config | None = None) -> None:
    cfg = Config() if cfg is None else cfg
    if cfg.smoke_test:
        cfg.epochs = 1
    set_reproducibility(cfg.seed)
    configure_gpu()
    manifest, class_names, manifest_file = load_reference(cfg)
    output = Path(cfg.output)
    validate_protocol(output, cfg, manifest_file)
    qgecaf_metrics = pd.read_csv(Path(cfg.qgecaf_root) / "fold_metrics.csv")
    expected_qgecaf_folds = set(range(1, cfg.outer_folds + 1))
    if set(qgecaf_metrics["fold"].astype(int)) != expected_qgecaf_folds:
        raise RuntimeError("QGECAF fold_metrics.csv does not contain folds 1-10.")

    existing_uncalibrated: Dict[int, dict] = {}
    existing_calibrated: Dict[int, dict] = {}
    uncalibrated_file = output / "fold_metrics_uncalibrated.csv"
    calibrated_file = output / "fold_metrics_temperature_scaled.csv"
    if cfg.resume and uncalibrated_file.is_file():
        saved = pd.read_csv(uncalibrated_file)
        existing_uncalibrated = {
            int(row["fold"]): row.to_dict() for _, row in saved.iterrows()
        }
    if cfg.resume and calibrated_file.is_file():
        saved = pd.read_csv(calibrated_file)
        existing_calibrated = {
            int(row["fold"]): row.to_dict() for _, row in saved.iterrows()
        }

    uncalibrated_rows, calibrated_rows = [], []
    oof_uncalibrated, oof_calibrated = [], []
    for fold in range(1, cfg.outer_folds + 1):
        print("\n" + "=" * 80)
        print(f"MOBILENETV2 OUTER FOLD {fold}/{cfg.outer_folds}")
        print("=" * 80)
        fold_dir = output / "cross_validation" / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        completion_file = fold_dir / "FOLD_COMPLETE.json"
        uncal_pred_file = fold_dir / "outer_holdout_predictions_uncalibrated.csv"
        cal_pred_file = fold_dir / "outer_holdout_predictions_calibrated.csv"
        if (
            cfg.resume
            and fold in existing_uncalibrated
            and fold in existing_calibrated
            and completion_file.is_file()
            and uncal_pred_file.is_file()
            and cal_pred_file.is_file()
        ):
            partitions = load_fold_partition(cfg, manifest, fold)
            expected_keys = set(partitions["outer_holdout"]["record_key"])
            saved_prediction = ensure_unique_keys(
                pd.read_csv(cal_pred_file), str(cal_pred_file)
            )
            if set(saved_prediction["record_key"]) != expected_keys:
                raise RuntimeError(
                    f"Fold {fold} saved predictions do not match the locked holdout."
                )
            uncalibrated_rows.append(existing_uncalibrated[fold])
            calibrated_rows.append(existing_calibrated[fold])
            oof_uncalibrated.append(pd.read_csv(uncal_pred_file))
            oof_calibrated.append(pd.read_csv(cal_pred_file))
            print("Completed fold verified; skipping retraining.")
            continue

        partitions = load_fold_partition(cfg, manifest, fold)
        train_frame = partitions["inner_train"]
        validation_frame = partitions["inner_validation"]
        holdout_frame = partitions["outer_holdout"]
        fold_seed = cfg.seed + fold
        set_reproducibility(fold_seed)
        train_ds = make_dataset(train_frame, cfg, True, fold_seed)
        validation_ds = make_dataset(validation_frame, cfg, False, fold_seed)
        holdout_ds = make_dataset(holdout_frame, cfg, False, fold_seed)

        tf.keras.backend.clear_session()
        model = build_mobilenetv2(cfg)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        save_model_summary(model, fold_dir / "model_summary.txt")
        if fold == 1:
            complexity = model_complexity(model, cfg)
            (output / "model_complexity.json").write_text(
                json.dumps(complexity, indent=2), encoding="utf-8"
            )
        timer = EpochTimer(fold_dir / "epoch_timing.csv")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(fold_dir / "best_model.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
            ),
            tf.keras.callbacks.CSVLogger(str(fold_dir / "training_history.csv")),
            timer,
        ]
        started = time.perf_counter()
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=cfg.epochs,
            class_weight=compute_class_weights(
                train_frame["label"].to_numpy(int), cfg.num_classes
            ),
            callbacks=callbacks,
            verbose=1,
        )
        training_seconds = float(time.perf_counter() - started)
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        # Always evaluate the checkpoint with the minimum inner-validation
        # loss, including runs that reached the maximum epoch before the
        # EarlyStopping callback fired.
        del model
        model = tf.keras.models.load_model(str(fold_dir / "best_model.keras"))
        validation_logits, _ = predict_logits(model, validation_ds)
        holdout_logits, inference_seconds = predict_logits(model, holdout_ds)
        validation_labels = validation_frame["label"].to_numpy(int)
        holdout_labels = holdout_frame["label"].to_numpy(int)

        validation_logit_frame = prediction_frame(
            validation_frame,
            validation_logits,
            softmax(validation_logits),
            fold,
            "uncalibrated_inner_validation",
        )
        validation_logit_frame.to_csv(
            fold_dir / "validation_logits.csv", index=False
        )
        temperature_details = fit_temperature(
            validation_logits, validation_labels, cfg
        )
        (fold_dir / "temperature.json").write_text(
            json.dumps(temperature_details, indent=2), encoding="utf-8"
        )
        temperature = temperature_details["temperature"]
        uncalibrated_probabilities = softmax(holdout_logits)
        calibrated_probabilities = softmax(holdout_logits, temperature)
        if not np.array_equal(
            uncalibrated_probabilities.argmax(axis=1),
            calibrated_probabilities.argmax(axis=1),
        ):
            raise RuntimeError(
                f"Fold {fold}: positive scalar temperature changed class predictions."
            )
        uncal_metrics, uncal_class, uncal_cm = calculate_metrics(
            holdout_labels, uncalibrated_probabilities, cfg
        )
        cal_metrics, cal_class, cal_cm = calculate_metrics(
            holdout_labels, calibrated_probabilities, cfg
        )
        shared = {
            "fold": fold,
            "seed": fold_seed,
            "best_epoch": best_epoch,
            "epochs_run": len(history.history["loss"]),
            "training_time_seconds": training_seconds,
            "mean_epoch_time_seconds": float(
                np.mean([row["epoch_time_seconds"] for row in timer.rows])
            ),
            "inner_train_images": len(train_frame),
            "inner_validation_images": len(validation_frame),
            "outer_holdout_images": len(holdout_frame),
            "inner_train_patients": train_frame["patient_id"].nunique(),
            "inner_validation_patients": validation_frame["patient_id"].nunique(),
            "outer_holdout_patients": holdout_frame["patient_id"].nunique(),
            "inference_time_seconds": inference_seconds,
            "inference_time_ms_per_image": (
                1000.0 * inference_seconds / len(holdout_frame)
            ),
            "temperature": temperature,
        }
        uncal_metrics.update(shared)
        cal_metrics.update(shared)
        uncalibrated_rows.append(uncal_metrics)
        calibrated_rows.append(cal_metrics)

        uncal_predictions = prediction_frame(
            holdout_frame,
            holdout_logits,
            uncalibrated_probabilities,
            fold,
            "uncalibrated",
        )
        cal_predictions = prediction_frame(
            holdout_frame,
            holdout_logits,
            calibrated_probabilities,
            fold,
            "temperature_scaled",
        )
        uncal_predictions.to_csv(uncal_pred_file, index=False)
        cal_predictions.to_csv(cal_pred_file, index=False)
        uncal_class.insert(0, "class_name", class_names)
        cal_class.insert(0, "class_name", class_names)
        uncal_class.to_csv(
            fold_dir / "classwise_metrics_uncalibrated.csv", index=False
        )
        cal_class.to_csv(
            fold_dir / "classwise_metrics_calibrated.csv", index=False
        )
        plot_confusion_matrix(
            uncal_cm,
            class_names,
            fold_dir / "confusion_matrix.png",
            f"MobileNetV2 outer fold {fold}",
        )
        if not np.array_equal(uncal_cm, cal_cm):
            raise RuntimeError("Calibration unexpectedly changed the confusion matrix.")
        pd.DataFrame(uncalibrated_rows).sort_values("fold").to_csv(
            uncalibrated_file, index=False
        )
        pd.DataFrame(calibrated_rows).sort_values("fold").to_csv(
            calibrated_file, index=False
        )
        completion_file.write_text(
            json.dumps({
                "fold": fold,
                "outer_holdout_images": len(holdout_frame),
                "outer_holdout_patients": holdout_frame["patient_id"].nunique(),
                "best_epoch": best_epoch,
                "temperature": temperature,
            }, indent=2),
            encoding="utf-8",
        )
        oof_uncalibrated.append(uncal_predictions)
        oof_calibrated.append(cal_predictions)
        del model, train_ds, validation_ds, holdout_ds
        gc.collect()
        tf.keras.backend.clear_session()

    uncal_frame = pd.DataFrame(uncalibrated_rows).sort_values("fold")
    cal_frame = pd.DataFrame(calibrated_rows).sort_values("fold")
    if len(uncal_frame) != cfg.outer_folds or len(cal_frame) != cfg.outer_folds:
        raise RuntimeError("The ten-fold metric tables are incomplete.")
    uncal_frame.to_csv(uncalibrated_file, index=False)
    cal_frame.to_csv(calibrated_file, index=False)
    summarize(
        uncal_frame, output / "summary_uncalibrated_mean_sd_ci95.csv"
    )
    summarize(
        cal_frame, output / "summary_temperature_scaled_mean_sd_ci95.csv"
    )

    oof_uncal = pd.concat(oof_uncalibrated, ignore_index=True)
    oof_cal = pd.concat(oof_calibrated, ignore_index=True)
    development_images = int(np.sum(manifest["outer_fold"] != -1))
    for name, frame in (("uncalibrated", oof_uncal), ("calibrated", oof_cal)):
        # record_key is reconstructed only for validation and is not retained.
        checked = ensure_unique_keys(frame, f"OOF {name}")
        if len(checked) != development_images:
            raise RuntimeError(f"OOF {name} predictions are incomplete.")
    oof_uncal.to_csv(
        output / "development_oof_predictions_uncalibrated.csv", index=False
    )
    oof_cal.to_csv(
        output / "development_oof_predictions_temperature_scaled.csv", index=False
    )
    labels = oof_uncal["true_label"].to_numpy(int)
    probability_columns = [f"probability_{index}" for index in range(cfg.num_classes)]
    plot_reliability_comparison(
        labels,
        {
            "Uncalibrated MobileNetV2": oof_uncal[probability_columns].to_numpy(),
            "Temperature-scaled MobileNetV2": oof_cal[probability_columns].to_numpy(),
        },
        output / "reliability_diagram_oof.png",
        bins=cfg.ece_bins,
    )
    paired = paired_comparison(
        cfg,
        cal_frame,
        qgecaf_metrics,
        output / "paired_qgecaf_vs_calibrated_mobilenetv2.csv",
    )
    temperatures = cal_frame[["fold", "temperature"]].copy()
    temperatures.to_csv(output / "fold_temperatures.csv", index=False)
    (output / "RUN_COMPLETE.json").write_text(
        json.dumps({
            "protocol_version": PROTOCOL_VERSION,
            "folds": cfg.outer_folds,
            "development_oof_images": development_images,
            "untouched_test_evaluated": False,
            "primary_discrimination_endpoint": "macro_f1",
            "primary_calibration_endpoint": "negative_log_likelihood",
        }, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 80)
    print("MOBILENETV2 CALIBRATED BENCHMARK COMPLETE")
    print("=" * 80)
    print("Results:", output)
    print("Untouched final test evaluated: NO")
    print("\nTemperature-scaled ten-fold summary:")
    print(pd.read_csv(output / "summary_temperature_scaled_mean_sd_ci95.csv"))
    print("\nPaired QGECAF comparison:")
    print(paired)


def smoke_test() -> None:
    cfg = Config(smoke_test=True)
    tf.keras.backend.clear_session()
    model = build_mobilenetv2(cfg)
    batch = np.random.default_rng(0).random(
        (2, cfg.image_size, cfg.image_size, 1), dtype=np.float32
    )
    output = model(batch, training=False).numpy()
    assert output.shape == (2, cfg.num_classes)
    details = fit_temperature(
        output, np.asarray([0, 1]), cfg
    )
    assert details["temperature"] > 0
    print("Smoke test passed.")


if __name__ == "__main__":
    run(parse_args())
