"""Reviewer-revision QGECAF experiment with fully nested BGWO preprocessing.

The original train and validation patients are combined into the development
cohort.  The untouched test patients remain excluded from preprocessing,
feature selection, checkpoint selection, epoch selection and calibration.

This revision changes the handcrafted preprocessing order used inside BGWO.
For every candidate raw-feature mask and every patient-grouped BGWO split it:

1. selects the candidate raw feature columns;
2. fits StandardScaler only on the BGWO-training patients;
3. transforms the BGWO-training and BGWO-validation patients;
4. fits PCA only on the BGWO-training patients;
5. transforms both partitions;
6. fits an RBF-SVM on the BGWO-training patients; and
7. evaluates macro F1 on the BGWO-validation patients.

After the mask is selected, the scaler and PCA are refitted on the complete
inner-training partition.  The same procedure is repeated on the complete
development cohort before the final fixed-epoch refit.  Masks, fitted objects,
fitness traces, patient partitions, predictions and timing outputs are saved.

Run this file from Jupyter with
``%run QGECAF_Reviewer_Revision_Nested_BGWO.py`` or execute it from a terminal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import warnings
import hashlib
import gc
from itertools import combinations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
import tensorflow as tf
from scipy.stats import spearmanr, t, friedmanchisquare, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.measure import label, regionprops, shannon_entropy

try:
    from kymatio.numpy import Scattering2D
except Exception:
    Scattering2D = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FAMILY_NAMES = ("wst", "gabor", "lbp", "hog", "shape")
QUALITY_NAMES = ("blur", "brightness", "contrast", "entropy", "noise", "background")
PROTOCOL_VERSION = "reviewer_revision_nested_bgwo_v1"


@dataclass
class Config:
    dataset: str = r"D:\chaitali\patientsplit801010_clean_v2"
    metadata_csv: str = r"D:\chaitali\patientsplit801010_clean_v2\patient_split_80_10_10.csv"
    locked_manifest: str = (
        r"D:\chaitali\qgacef\QGECAF_FROM_SCRATCH_RUN_01"
        r"\locked_patient_fold_assignments.csv"
    )
    patient_column: str = "Patient_num"
    image_column: str = "Image_name"
    output: str = r"D:\chaitali\qgacef\QGECAF_REVIEWER_REVISION_NESTED_BGWO"
    image_size: int = 128
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    pca_variance: float = 0.98
    bgwo_agents: int = 12
    bgwo_iters: int = 25
    bgwo_cv: int = 3
    bgwo_alpha: float = 0.99
    embed_dim: int = 64
    attention_heads: int = 4
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    kl_weight: float = 1e-3
    dropout: float = 0.30
    fine_tune_last: int = 30
    force_features: bool = False
    smoke_test: bool = False
    outer_folds: int = 10
    inner_folds: int = 9
    early_stopping_patience: int = 8
    resume: bool = True
    require_kymatio: bool = True
    run_end_to_end_latency: bool = True
    latency_warmup_images: int = 50
    latency_repetitions: int = 5
    latency_batch_sizes: Tuple[int, ...] = (1, 8, 16)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=Config.dataset)
    parser.add_argument("--metadata-csv", default=Config.metadata_csv)
    parser.add_argument("--locked-manifest", default=Config.locked_manifest)
    parser.add_argument("--patient-column", default=Config.patient_column)
    parser.add_argument("--image-column", default=Config.image_column)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--pca-variance", type=float, default=Config.pca_variance)
    parser.add_argument("--bgwo-agents", type=int, default=Config.bgwo_agents)
    parser.add_argument("--bgwo-iters", type=int, default=Config.bgwo_iters)
    parser.add_argument("--bgwo-cv", type=int, default=Config.bgwo_cv)
    parser.add_argument("--embed-dim", type=int, default=Config.embed_dim)
    parser.add_argument("--attention-heads", type=int, default=Config.attention_heads)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--fine-tune-last", type=int, default=Config.fine_tune_last)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--outer-folds", type=int, default=Config.outer_folds)
    parser.add_argument("--inner-folds", type=int, default=Config.inner_folds)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--allow-wavelet-fallback", action="store_false", dest="require_kymatio")
    parser.add_argument("--skip-end-to-end-latency", action="store_false", dest="run_end_to_end_latency")
    parser.add_argument("--latency-warmup-images", type=int, default=Config.latency_warmup_images)
    parser.add_argument("--latency-repetitions", type=int, default=Config.latency_repetitions)
    return Config(**vars(parser.parse_args()))


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"TensorFlow {tf.__version__}; GPUs detected: {len(gpus)}")


def write_environment_manifest(output_dir: Path) -> None:
    """Record the software/hardware context needed to interpret timings."""
    gpus = tf.config.list_physical_devices("GPU")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "gpu_devices": [device.name for device in gpus],
    }
    (output_dir / "environment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_method_manifest(output_dir: Path, cfg: Config) -> None:
    """Write the descriptor and selector settings required for replication."""
    method = {
        "image": {"size": [cfg.image_size, cfg.image_size], "channels": 1},
        "wst": {"implementation": "kymatio.numpy.Scattering2D", "J": 2,
                "L": 8, "max_order": 2, "aggregation": "channel_spatial_mean"},
        "gabor": {"orientations_radians": [0.0, float(np.pi / 4),
                  float(np.pi / 2), float(3 * np.pi / 4)],
                  "wavelengths": [4.0, 8.0, 16.0], "kernel_size": [15, 15],
                  "sigma": 3.0, "gamma": 0.5},
        "lbp": {"method": "uniform", "radius_points": [[1, 8], [2, 16], [3, 24]]},
        "hog": {"orientations": 9, "pixels_per_cell": [16, 16],
                "cells_per_block": [2, 2], "block_norm": "L2-Hys"},
        "shape": {"threshold": "Otsu", "closing_kernel": [5, 5],
                  "region": "largest_connected_component"},
        "quality": list(QUALITY_NAMES),
        "bgwo": {"agents": cfg.bgwo_agents, "iterations": cfg.bgwo_iters,
                 "patient_grouped_folds": cfg.bgwo_cv, "objective_alpha": cfg.bgwo_alpha,
                 "classifier": "RBF-SVM", "C": 2.0, "gamma": "scale",
                 "class_weight": "balanced", "metric": "six-class macro F1"},
        "pca": {"retained_variance": cfg.pca_variance, "solver": "full"},
    }
    (output_dir / "method_manifest.json").write_text(
        json.dumps(method, indent=2), encoding="utf-8"
    )


def discover_dataset(dataset_dir: Path) -> Tuple[List[Path], np.ndarray, List[str]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found: {dataset_dir}\n"
            "Check the drive letter and folder spelling."
        )
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if len(class_dirs) < 2:
        raise ValueError("The dataset folder must contain one subfolder per class.")

    paths: List[Path] = []
    labels: List[int] = []
    names = [p.name for p in class_dirs]
    for class_index, class_dir in enumerate(class_dirs):
        class_images = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"{class_dir.name}: {len(class_images)} images")
        paths.extend(class_images)
        labels.extend([class_index] * len(class_images))

    if not paths:
        raise ValueError("No supported images were found in the class folders.")
    return paths, np.asarray(labels, dtype=np.int32), names


def discover_presplit_dataset(
    dataset_root: Path,
) -> Tuple[List[Path], np.ndarray, List[str], np.ndarray]:
    """Read a fixed train/val/test directory without creating a new split.

    Expected layout::

        dataset_root/train/<class_name>/*.png
        dataset_root/val/<class_name>/*.png
        dataset_root/test/<class_name>/*.png

    A single class-to-index mapping is learned from the training folders and
    reused for validation and testing, preventing label-index inconsistencies.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Pre-split dataset root was not found: {dataset_root}")

    split_aliases = {"train": "train", "val": "validation", "test": "test"}
    for folder in split_aliases:
        if not (dataset_root / folder).is_dir():
            raise FileNotFoundError(
                f"Required split folder is missing: {dataset_root / folder}"
            )

    train_class_dirs = sorted(
        [p for p in (dataset_root / "train").iterdir() if p.is_dir()],
        key=lambda p: p.name.casefold(),
    )
    if len(train_class_dirs) < 2:
        raise ValueError("The train folder must contain one subfolder per class.")

    class_names = [p.name for p in train_class_dirs]
    class_key_to_index = {name.casefold(): i for i, name in enumerate(class_names)}

    paths: List[Path] = []
    labels: List[int] = []
    split_names: List[str] = []

    print("Fixed patient-level split detected")
    for folder_name, split_name in split_aliases.items():
        split_dir = dataset_root / folder_name
        available = {
            p.name.casefold(): p for p in split_dir.iterdir() if p.is_dir()
        }
        unknown = sorted(set(available) - set(class_key_to_index))
        missing = sorted(set(class_key_to_index) - set(available))
        if unknown:
            raise ValueError(f"Unknown class folders in {split_dir}: {unknown}")
        if missing:
            raise ValueError(f"Missing class folders in {split_dir}: {missing}")

        print(f"\n{split_name.upper()}")
        for class_name in class_names:
            class_dir = available[class_name.casefold()]
            images = sorted(
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise ValueError(f"No supported images found in {class_dir}")
            class_index = class_key_to_index[class_name.casefold()]
            print(f"  {class_name}: {len(images)} images")
            paths.extend(images)
            labels.extend([class_index] * len(images))
            split_names.extend([split_name] * len(images))

    normalized = [str(p.resolve()).casefold() for p in paths]
    if len(normalized) != len(set(normalized)):
        raise ValueError("The same image path was discovered more than once across splits.")

    return (
        paths,
        np.asarray(labels, dtype=np.int32),
        class_names,
        np.asarray(split_names, dtype=object),
    )


def recover_presplit_indices(
    raw_paths: np.ndarray, paths: Sequence[Path], split_names: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover fixed split indices after unreadable images have been skipped."""
    split_lookup = {str(path): split for path, split in zip(paths, split_names)}
    recovered = np.asarray([split_lookup[str(path)] for path in raw_paths], dtype=object)
    train_idx = np.flatnonzero(recovered == "train")
    val_idx = np.flatnonzero(recovered == "validation")
    test_idx = np.flatnonzero(recovered == "test")
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("At least one fixed split became empty after image loading.")
    return train_idx, val_idx, test_idx


def load_patient_ids(
    raw_paths: np.ndarray,
    metadata_csv: Path,
    patient_column: str,
    image_column: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Map every loaded image to a patient ID with fail-fast validation.

    ``image_column`` may contain a basename or a path. Basename matching is
    intentionally rejected when it is ambiguous across different patients.
    """
    if not metadata_csv.is_file():
        raise FileNotFoundError(
            f"Patient metadata CSV was not found: {metadata_csv}\n"
            "Set cfg.metadata_csv to the CSV used for your patient split."
        )
    # Auto-detect comma, semicolon, tab or other conventional CSV delimiters.
    # FETAL_PLANES_DB metadata is commonly semicolon-separated.
    meta = pd.read_csv(metadata_csv, sep=None, engine="python")
    missing_columns = [c for c in (patient_column, image_column) if c not in meta.columns]
    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}. Available columns: {list(meta.columns)}"
        )
    if meta[[patient_column, image_column]].isna().any().any():
        raise ValueError("Patient ID and image-name columns must not contain missing values.")

    meta = meta.copy()
    meta[patient_column] = meta[patient_column].astype(str).str.strip()
    # The original FETAL_PLANES_DB CSV stores Image_name without an extension,
    # whereas the organized folders normally contain .png files. Match stems.
    meta["_image_key"] = meta[image_column].astype(str).str.strip().map(
        lambda value: Path(value).stem.casefold()
    )
    ambiguous = meta.groupby("_image_key")[patient_column].nunique()
    ambiguous = ambiguous[ambiguous > 1]
    if len(ambiguous):
        raise ValueError(
            "The same filename is assigned to multiple patients. Use a unique relative-path "
            f"column as cfg.image_column. Examples: {ambiguous.index[:10].tolist()}"
        )
    lookup = meta.drop_duplicates("_image_key").set_index("_image_key")[patient_column].to_dict()
    keys = np.asarray([Path(str(path)).stem.casefold() for path in raw_paths], dtype=object)
    missing = sorted(set(keys) - set(lookup))
    if missing:
        raise ValueError(
            f"{len(missing)} loaded image filenames are absent from the metadata CSV. "
            f"Examples: {missing[:10]}"
        )
    patient_ids = np.asarray([lookup[key] for key in keys], dtype=object)
    if np.any(patient_ids == ""):
        raise ValueError("Blank patient identifiers were detected.")
    return patient_ids, meta


def validate_patient_separation(
    patient_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    groups = {
        "train": set(patient_ids[train_idx]),
        "validation": set(patient_ids[val_idx]),
        "test": set(patient_ids[test_idx]),
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = groups[left].intersection(groups[right])
        if overlap:
            raise ValueError(
                f"Patient leakage between {left} and {right}: "
                f"{sorted(map(str, overlap))[:10]}"
            )
    print("Patient leakage audit passed: train, validation and test are disjoint.")


def dataset_fingerprint(paths: np.ndarray, labels: np.ndarray, patient_ids: np.ndarray) -> str:
    """Stable audit fingerprint for the ordered analysis cohort."""
    digest = hashlib.sha256()
    for path, label_value, patient in zip(paths, labels, patient_ids):
        digest.update(f"{Path(str(path)).name.casefold()}|{int(label_value)}|{patient}\n".encode("utf-8"))
    return digest.hexdigest()


def load_locked_fold_assignments(
    manifest_path: Path,
    raw_paths: np.ndarray,
    patient_ids: np.ndarray,
    labels: np.ndarray,
    development_idx: np.ndarray,
    test_idx: np.ndarray,
    outer_folds: int,
) -> np.ndarray:
    """Recover the original locked folds without depending on drive prefixes.

    The key combines patient ID and image filename stem.  This permits the same
    manifest to be used after the physical dataset is copied to a different
    root, while still preventing accidental matches between patients.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Locked fold manifest not found: {manifest_path}\n"
            "The reviewer-revision run must reuse the original patient folds."
        )
    manifest = pd.read_csv(manifest_path)
    required = {"patient_id", "outer_fold"}
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Locked manifest is missing columns {sorted(missing_columns)}. "
            f"Available columns: {list(manifest.columns)}"
        )
    if "path" in manifest.columns:
        manifest_stems = manifest["path"].astype(str).map(
            lambda value: Path(value).stem.casefold()
        )
    else:
        image_candidates = [
            column for column in ("image_name", "Image_name", "filename")
            if column in manifest.columns
        ]
        if not image_candidates:
            raise ValueError(
                "Locked manifest requires either 'path' or an image-name column."
            )
        manifest_stems = manifest[image_candidates[0]].astype(str).map(
            lambda value: Path(value).stem.casefold()
        )

    manifest_patient = manifest["patient_id"].astype(str).str.strip()
    manifest_keys = manifest_patient + "|" + manifest_stems
    duplicated = manifest_keys[manifest_keys.duplicated(keep=False)]
    if len(duplicated):
        raise ValueError(
            "Locked manifest contains duplicate patient/image keys. Examples: "
            f"{duplicated.head(10).tolist()}"
        )
    fold_lookup = dict(zip(manifest_keys, manifest["outer_fold"].astype(int)))

    loaded_keys = np.asarray(
        [
            f"{str(patient).strip()}|{Path(str(path)).stem.casefold()}"
            for patient, path in zip(patient_ids, raw_paths)
        ],
        dtype=object,
    )
    missing = [key for key in loaded_keys if key not in fold_lookup]
    if missing:
        raise RuntimeError(
            f"{len(missing)} loaded images are absent from the locked manifest. "
            f"Examples: {missing[:10]}"
        )
    assignments = np.asarray([fold_lookup[key] for key in loaded_keys], dtype=np.int32)

    expected_folds = set(range(1, outer_folds + 1))
    observed_folds = set(assignments[development_idx].tolist())
    if observed_folds != expected_folds:
        raise RuntimeError(
            f"Development folds are {sorted(observed_folds)}; expected "
            f"{sorted(expected_folds)}."
        )
    if not np.all(assignments[test_idx] == -1):
        raise RuntimeError("Every untouched test image must have outer_fold=-1.")
    if np.any(assignments[development_idx] == -1):
        raise RuntimeError("A development image is incorrectly marked outer_fold=-1.")

    patient_fold = pd.DataFrame(
        {"patient_id": patient_ids, "outer_fold": assignments}
    ).drop_duplicates()
    inconsistent = patient_fold.groupby("patient_id")["outer_fold"].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if len(inconsistent):
        raise RuntimeError(
            "A patient appears in more than one locked outer fold. Examples: "
            f"{inconsistent.index[:10].tolist()}"
        )
    for fold in expected_folds:
        holdout = np.flatnonzero(assignments == fold)
        train = np.flatnonzero(
            np.isin(np.arange(len(assignments)), development_idx)
            & (assignments != fold)
        )
        if set(patient_ids[holdout]).intersection(set(patient_ids[train])):
            raise RuntimeError(f"Patient leakage detected in locked fold {fold}.")
    return assignments


def read_grayscale(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.uint8)


def wavelet_scattering_features(image: np.ndarray, scattering) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    if scattering is not None:
        coeff = np.asarray(scattering(x), dtype=np.float32)
        # Spatially average each scattering channel to create a stable vector.
        return coeff.reshape(coeff.shape[0], -1).mean(axis=1)

    # Explicit fallback if Kymatio cannot be imported. This is a wavelet-statistic
    # representation, not a true scattering transform; the warning is printed once.
    features: List[float] = []
    coeffs = pywt.wavedec2(x, wavelet="db2", level=3)
    for item in coeffs:
        arrays = (item,) if isinstance(item, np.ndarray) else item
        for arr in arrays:
            features.extend([float(np.mean(arr)), float(np.std(arr)), float(np.mean(arr ** 2))])
    return np.asarray(features, dtype=np.float32)


def gabor_features(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    values: List[float] = []
    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        for wavelength in (4.0, 8.0, 16.0):
            kernel = cv2.getGaborKernel((15, 15), 3.0, theta, wavelength, 0.5, 0, ktype=cv2.CV_32F)
            response = cv2.filter2D(x, cv2.CV_32F, kernel)
            values.extend(
                [
                    float(response.mean()),
                    float(response.std()),
                    float(np.mean(np.abs(response))),
                    float(np.mean(response ** 2)),
                ]
            )
    return np.asarray(values, dtype=np.float32)


def lbp_features(image: np.ndarray) -> np.ndarray:
    all_histograms = []
    for radius, points in ((1, 8), (2, 16), (3, 24)):
        lbp = local_binary_pattern(image, points, radius, method="uniform")
        bins = points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(bins + 1), range=(0, bins), density=True)
        all_histograms.append(hist.astype(np.float32))
    return np.concatenate(all_histograms)


def hog_features(image: np.ndarray) -> np.ndarray:
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def shape_features(image: np.ndarray) -> np.ndarray:
    # Approximate foreground morphology. For clinical shape analysis, replace
    # this with features computed from validated organ segmentations.
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    regions = regionprops(label(binary > 0))
    if not regions:
        return np.zeros(10, dtype=np.float32)
    region = max(regions, key=lambda r: r.area)
    perimeter = max(float(region.perimeter), 1e-6)
    circularity = 4.0 * np.pi * float(region.area) / (perimeter ** 2)
    major = max(float(region.major_axis_length), 1e-6)
    return np.asarray(
        [
            region.area / image.size,
            perimeter / (2 * (image.shape[0] + image.shape[1])),
            circularity,
            float(region.eccentricity),
            float(region.solidity),
            float(region.extent),
            major / image.shape[1],
            float(region.minor_axis_length) / image.shape[0],
            float(region.minor_axis_length) / major,
            float(region.orientation) / np.pi,
        ],
        dtype=np.float32,
    )


def quality_features(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    blur = cv2.Laplacian(x, cv2.CV_32F).var()
    brightness = x.mean()
    contrast = x.std()
    entropy = shannon_entropy(image)
    median = cv2.medianBlur(image, 3).astype(np.float32)
    noise = np.median(np.abs(x - median)) / 0.6745
    background = np.mean(image < 10)
    return np.asarray([blur, brightness, contrast, entropy, noise, background], dtype=np.float32)


def glcm_auxiliary_features(image: np.ndarray) -> np.ndarray:
    """Not used by the model; retained as an optional baseline helper."""
    quantized = (image // 16).astype(np.uint8)
    matrix = graycomatrix(quantized, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=16,
                          symmetric=True, normed=True)
    return np.asarray(
        [np.mean(graycoprops(matrix, name)) for name in ("contrast", "correlation", "energy", "homogeneity")],
        dtype=np.float32,
    )


def extract_all_features(paths: Sequence[Path], labels: np.ndarray, cfg: Config, cache: Path) -> Dict[str, np.ndarray]:
    if cache.exists() and not cfg.force_features:
        with np.load(cache, allow_pickle=True) as data:
            cached = {key: data[key] for key in data.files}
        requested_paths = np.asarray([str(path) for path in paths], dtype=object)
        if np.array_equal(cached.get("requested_paths"), requested_paths):
            print(f"Loading cached features: {cache}")
            return cached
        print("Dataset paths changed; rebuilding the feature cache.")

    if Scattering2D is None and cfg.require_kymatio:
        raise ImportError(
            "Kymatio is required for the reviewer-revision experiment. Install "
            "a TensorFlow/Python-3.10-compatible Kymatio release, or explicitly "
            "use --allow-wavelet-fallback only for a non-publication smoke test."
        )
    if Scattering2D is None:
        warnings.warn(
            "Kymatio is unavailable. WST will use a wavelet-statistics fallback. "
            "Install kymatio for the final experiment.",
            RuntimeWarning,
        )
        scattering = None
    else:
        scattering = Scattering2D(J=2, shape=(cfg.image_size, cfg.image_size), L=8, max_order=2)

    storage: Dict[str, List[np.ndarray]] = {name: [] for name in FAMILY_NAMES + ("quality", "images")}
    valid_labels: List[int] = []
    valid_paths: List[str] = []
    for i, (path, y) in enumerate(zip(paths, labels), start=1):
        try:
            image = read_grayscale(path, cfg.image_size)
            storage["images"].append(image[..., None])
            storage["wst"].append(wavelet_scattering_features(image, scattering))
            storage["gabor"].append(gabor_features(image))
            storage["lbp"].append(lbp_features(image))
            storage["hog"].append(hog_features(image))
            storage["shape"].append(shape_features(image))
            storage["quality"].append(quality_features(image))
            valid_labels.append(int(y))
            valid_paths.append(str(path))
        except Exception as exc:
            warnings.warn(f"Skipping {path}: {exc}")
        if i % 100 == 0 or i == len(paths):
            print(f"Feature extraction: {i}/{len(paths)}")

    result = {name: np.stack(values).astype(np.float32) for name, values in storage.items()}
    result["labels"] = np.asarray(valid_labels, dtype=np.int32)
    result["paths"] = np.asarray(valid_paths, dtype=object)
    result["requested_paths"] = np.asarray([str(path) for path in paths], dtype=object)
    np.savez_compressed(cache, **result)
    return result


def build_splits(labels: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_val, test = train_test_split(
        indices, test_size=cfg.test_size, stratify=labels, random_state=cfg.seed
    )
    relative_val = cfg.val_size / (1.0 - cfg.test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=labels[train_val],
        random_state=cfg.seed,
    )
    return train, val, test


class NestedBinaryGreyWolfSelector:
    """Patient-grouped BGWO with candidate-specific scaler and PCA fitting.

    The binary mask always refers to the original raw coordinates of one
    handcrafted family.  This is important because independently fitted PCA
    axes are not interchangeable across validation folds.
    """

    def __init__(
        self,
        n_agents: int = 12,
        n_iterations: int = 25,
        cv: int = 3,
        alpha: float = 0.99,
        pca_variance: float = 0.98,
        use_pca: bool = True,
        seed: int = 42,
    ):
        self.n_agents = int(n_agents)
        self.n_iterations = int(n_iterations)
        self.cv = int(cv)
        self.alpha = float(alpha)
        self.pca_variance = float(pca_variance)
        self.use_pca = bool(use_pca)
        self.seed = int(seed)
        self.mask_: np.ndarray | None = None
        self.history_: List[dict] = []
        self.best_fitness_: float | None = None
        self.best_macro_f1_: float | None = None
        self.best_fold_macro_f1_: List[float] = []
        self.best_fold_pca_dimensions_: List[int] = []
        self._splits: List[Tuple[np.ndarray, np.ndarray]] = []
        self._class_labels: np.ndarray | None = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _evaluate_mask(
        self,
        mask: np.ndarray,
        X_raw: np.ndarray,
        y: np.ndarray,
        cache: dict,
    ) -> Tuple[float, List[float], List[int]]:
        key = np.packbits(mask.astype(np.uint8)).tobytes()
        if key in cache:
            return cache[key]

        selected_count = int(mask.sum())
        if selected_count == 0:
            result = (1.0, [0.0] * self.cv, [0] * self.cv)
            cache[key] = result
            return result

        scores: List[float] = []
        pca_dimensions: List[int] = []
        for split_number, (tr, va) in enumerate(self._splits, start=1):
            # The candidate mask is applied to raw coordinates first.
            X_train_selected = X_raw[tr][:, mask]
            X_validation_selected = X_raw[va][:, mask]

            # Every fitted object is local to the BGWO-training patients.
            scaler = StandardScaler().fit(X_train_selected)
            X_train_scaled = scaler.transform(X_train_selected)
            X_validation_scaled = scaler.transform(X_validation_selected)

            if self.use_pca and selected_count > 10:
                pca = PCA(
                    n_components=self.pca_variance,
                    svd_solver="full",
                    random_state=self.seed + split_number,
                )
                X_train_model = pca.fit_transform(X_train_scaled)
                X_validation_model = pca.transform(X_validation_scaled)
                pca_dimensions.append(int(pca.n_components_))
            else:
                X_train_model = X_train_scaled
                X_validation_model = X_validation_scaled
                pca_dimensions.append(int(X_train_model.shape[1]))

            classifier = SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                class_weight="balanced",
            )
            classifier.fit(X_train_model, y[tr])
            prediction = classifier.predict(X_validation_model)
            scores.append(
                float(
                    f1_score(
                        y[va],
                        prediction,
                        labels=self._class_labels,
                        average="macro",
                        zero_division=0,
                    )
                )
            )

        mean_macro_f1 = float(np.mean(scores))
        subset_fraction = selected_count / X_raw.shape[1]
        fitness = self.alpha * (1.0 - mean_macro_f1) + (1.0 - self.alpha) * subset_fraction
        result = (float(fitness), scores, pca_dimensions)
        cache[key] = result
        return result

    def fit(
        self, X_raw: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> "NestedBinaryGreyWolfSelector":
        X_raw = np.asarray(X_raw, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        groups = np.asarray(groups)
        if len(X_raw) != len(y) or len(groups) != len(y):
            raise ValueError("BGWO features, labels and groups have different lengths.")
        if np.isnan(X_raw).any() or np.isinf(X_raw).any():
            raise ValueError("BGWO raw features contain NaN or infinite values.")

        self._class_labels = np.unique(y)
        splitter = StratifiedGroupKFold(
            n_splits=self.cv, shuffle=True, random_state=self.seed
        )
        self._splits = list(splitter.split(X_raw, y, groups=groups))
        for split_number, (tr, va) in enumerate(self._splits, start=1):
            overlap = set(groups[tr]).intersection(set(groups[va]))
            if overlap:
                raise RuntimeError(
                    f"Patient leakage in BGWO split {split_number}: "
                    f"{sorted(map(str, overlap))[:10]}"
                )

        dimensions = int(X_raw.shape[1])
        if dimensions <= 2:
            self.mask_ = np.ones(dimensions, dtype=bool)
            cache: dict = {}
            details = self._evaluate_mask(self.mask_, X_raw, y, cache)
            self.best_fitness_, self.best_fold_macro_f1_, self.best_fold_pca_dimensions_ = details
            self.best_macro_f1_ = float(np.mean(self.best_fold_macro_f1_))
            return self

        rng = np.random.default_rng(self.seed)
        positions = rng.integers(
            0, 2, size=(self.n_agents, dimensions), dtype=np.int8
        ).astype(np.float32)
        for agent in range(self.n_agents):
            if not positions[agent].any():
                positions[agent, rng.integers(0, dimensions)] = 1.0

        fitness_cache: dict = {}
        alpha_pos = np.ones(dimensions, dtype=np.float32)
        beta_pos = alpha_pos.copy()
        delta_pos = alpha_pos.copy()
        alpha_score = beta_score = delta_score = np.inf

        for iteration in range(self.n_iterations):
            iteration_start = time.perf_counter()
            for agent in range(self.n_agents):
                mask = positions[agent] >= 0.5
                score, _, _ = self._evaluate_mask(mask, X_raw, y, fitness_cache)
                if score < alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = score, positions[agent].copy()
                elif score < beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = score, positions[agent].copy()
                elif score < delta_score:
                    delta_score, delta_pos = score, positions[agent].copy()

            a = 2.0 - 2.0 * iteration / max(self.n_iterations - 1, 1)
            for agent in range(self.n_agents):
                proposals = []
                for leader in (alpha_pos, beta_pos, delta_pos):
                    r1 = rng.random(dimensions)
                    r2 = rng.random(dimensions)
                    coefficient_a = 2.0 * a * r1 - a
                    coefficient_c = 2.0 * r2
                    distance = np.abs(coefficient_c * leader - positions[agent])
                    proposals.append(leader - coefficient_a * distance)
                probabilities = self._sigmoid(np.mean(proposals, axis=0))
                positions[agent] = (
                    rng.random(dimensions) < probabilities
                ).astype(np.float32)
                if not positions[agent].any():
                    positions[agent, rng.integers(0, dimensions)] = 1.0

            best_mask = alpha_pos >= 0.5
            best_details = self._evaluate_mask(best_mask, X_raw, y, fitness_cache)
            elapsed = float(time.perf_counter() - iteration_start)
            row = {
                "iteration": iteration + 1,
                "fitness": float(alpha_score),
                "macro_f1": float(np.mean(best_details[1])),
                "selected_raw_features": int(best_mask.sum()),
                "raw_features": dimensions,
                "unique_masks_evaluated": len(fitness_cache),
                "iteration_seconds": elapsed,
            }
            self.history_.append(row)
            print(
                f"  Nested BGWO {iteration + 1:03d}/{self.n_iterations}: "
                f"fitness={row['fitness']:.6f}, macro_F1={row['macro_f1']:.6f}, "
                f"selected={row['selected_raw_features']}/{dimensions}, "
                f"time={elapsed:.1f}s"
            )

        self.mask_ = alpha_pos >= 0.5
        if not self.mask_.any():
            self.mask_[int(np.argmax(alpha_pos))] = True
        details = self._evaluate_mask(self.mask_, X_raw, y, fitness_cache)
        self.best_fitness_ = float(details[0])
        self.best_fold_macro_f1_ = [float(value) for value in details[1]]
        self.best_macro_f1_ = float(np.mean(self.best_fold_macro_f1_))
        self.best_fold_pca_dimensions_ = [int(value) for value in details[2]]
        return self


def fit_preprocessing(
    raw: Dict[str, np.ndarray], train_idx: np.ndarray, labels: np.ndarray,
    patient_ids: np.ndarray, cfg: Config, artifact_dir: Path | None = None,
) -> Tuple[dict, Dict[str, np.ndarray], np.ndarray]:
    """Fit the fully nested selector and refit final fold preprocessing.

    ``train_idx`` is either an outer fold's inner-training partition or the
    complete development cohort during final refitting.  No validation,
    outer-holdout or final-test observation is used to fit returned objects.
    """
    processors = {}
    transformed: Dict[str, np.ndarray] = {}
    summary_rows: List[dict] = []
    masks: Dict[str, np.ndarray] = {}
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    for family in FAMILY_NAMES:
        print(f"\nFully nested preprocessing family: {family.upper()}")
        X_train_raw = np.asarray(raw[family][train_idx], dtype=np.float32)
        use_pca = family != "shape" and X_train_raw.shape[1] > 10
        selector = NestedBinaryGreyWolfSelector(
            n_agents=cfg.bgwo_agents,
            n_iterations=cfg.bgwo_iters,
            cv=cfg.bgwo_cv,
            alpha=cfg.bgwo_alpha,
            pca_variance=cfg.pca_variance,
            use_pca=use_pca,
            seed=cfg.seed,
        ).fit(X_train_raw, labels[train_idx], patient_ids[train_idx])

        raw_mask = np.asarray(selector.mask_, dtype=bool)
        masks[family] = raw_mask
        selected_train = raw[family][train_idx][:, raw_mask]
        selected_all = raw[family][:, raw_mask]

        scaler = StandardScaler().fit(selected_train)
        scaled_train = scaler.transform(selected_train)
        scaled_all = scaler.transform(selected_all)
        if use_pca and selected_train.shape[1] > 10:
            pca = PCA(
                n_components=cfg.pca_variance,
                svd_solver="full",
                random_state=cfg.seed,
            )
            pca.fit(scaled_train)
            family_all = pca.transform(scaled_all)
            explained_variance = float(np.sum(pca.explained_variance_ratio_))
            final_dimension = int(pca.n_components_)
        else:
            pca = None
            family_all = scaled_all
            explained_variance = 1.0
            final_dimension = int(family_all.shape[1])

        transformed[family] = family_all.astype(np.float32)
        processors[family] = {
            "raw_mask": raw_mask,
            "selected_raw_indices": np.flatnonzero(raw_mask),
            "scaler": scaler,
            "pca": pca,
            "preprocessing_order": "raw_mask_then_scaler_then_pca",
        }
        summary_rows.append(
            {
                "family": family,
                "raw_dimension": int(raw[family].shape[1]),
                "selected_raw_dimension": int(raw_mask.sum()),
                "final_pca_dimension": final_dimension,
                "pca_explained_variance": explained_variance,
                "bgwo_best_fitness": selector.best_fitness_,
                "bgwo_best_macro_f1": selector.best_macro_f1_,
                "bgwo_fold_macro_f1": json.dumps(selector.best_fold_macro_f1_),
                "bgwo_fold_pca_dimensions": json.dumps(
                    selector.best_fold_pca_dimensions_
                ),
            }
        )
        if artifact_dir is not None:
            pd.DataFrame(selector.history_).to_csv(
                artifact_dir / f"bgwo_history_{family}.csv", index=False
            )
        print(
            f"  Raw {raw[family].shape[1]} -> selected {raw_mask.sum()} "
            f"-> final {final_dimension}; nested BGWO macro F1="
            f"{selector.best_macro_f1_:.6f}"
        )

    quality_scaler = StandardScaler().fit(raw["quality"][train_idx])
    quality_all = quality_scaler.transform(raw["quality"]).astype(np.float32)
    processors["quality_scaler"] = quality_scaler
    processors["protocol"] = {
        "version": "reviewer_revision_nested_bgwo_v1",
        "preprocessing_order": "raw_mask_then_scaler_then_pca",
        "fitted_images": int(len(train_idx)),
        "fitted_patients": int(len(set(patient_ids[train_idx]))),
        "seed": int(cfg.seed),
    }
    if artifact_dir is not None:
        pd.DataFrame(summary_rows).to_csv(
            artifact_dir / "nested_bgwo_preprocessing_summary.csv", index=False
        )
        np.savez_compressed(artifact_dir / "bgwo_raw_masks.npz", **masks)
        pd.DataFrame(
            {
                "image_index": train_idx,
                "patient_id": patient_ids[train_idx],
                "label": labels[train_idx],
            }
        ).to_csv(artifact_dir / "preprocessing_fit_partition.csv", index=False)
    return processors, transformed, quality_all


# Jupyter users commonly rerun the definitions cell. TensorFlow/Keras 2.10
# otherwise rejects the same serialized names on the second execution.
for _registered_name in (
    "QGECAF>EvidentialLoss",
    "QGECAF>DirichletMeanAccuracy",
    "QGECAF>OneMinus",
    "QGECAF>EvidenceToDirichlet",
):
    tf.keras.utils.get_custom_objects().pop(_registered_name, None)


@tf.keras.utils.register_keras_serializable(package="QGECAF")
class EvidentialLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes: int, kl_weight: float = 1e-3, class_weights=None, name="evidential_loss"):
        super().__init__(name=name)
        self.num_classes = int(num_classes)
        self.kl_weight = float(kl_weight)
        self.class_weights = None if class_weights is None else [float(v) for v in class_weights]

    @staticmethod
    def dirichlet_kl(alpha: tf.Tensor, num_classes: int) -> tf.Tensor:
        beta = tf.ones((1, num_classes), dtype=alpha.dtype)
        sum_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        sum_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        ln_b_alpha = tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True) - tf.math.lgamma(sum_alpha)
        ln_b_beta = tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) - tf.math.lgamma(sum_beta)
        digamma_term = tf.reduce_sum(
            (alpha - beta) * (tf.math.digamma(alpha) - tf.math.digamma(sum_alpha)), axis=1, keepdims=True
        )
        return tf.squeeze(ln_b_beta - ln_b_alpha + digamma_term, axis=1)

    def call(self, y_true: tf.Tensor, alpha: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, alpha.dtype)
        strength = tf.reduce_sum(alpha, axis=1, keepdims=True)
        expected_ce = tf.reduce_sum(
            y_true * (tf.math.digamma(strength) - tf.math.digamma(alpha)), axis=1
        )
        if self.class_weights is not None:
            weights = tf.constant(self.class_weights, dtype=alpha.dtype)
            expected_ce *= tf.reduce_sum(y_true * weights, axis=1)
        alpha_tilde = y_true + (1.0 - y_true) * alpha
        kl = self.dirichlet_kl(alpha_tilde, self.num_classes)
        return expected_ce + self.kl_weight * kl

    def get_config(self):
        return {
            **super().get_config(),
            "num_classes": self.num_classes,
            "kl_weight": self.kl_weight,
            "class_weights": self.class_weights,
        }


@tf.keras.utils.register_keras_serializable(package="QGECAF")
class DirichletMeanAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.metric = tf.keras.metrics.CategoricalAccuracy()

    def update_state(self, y_true, alpha, sample_weight=None):
        probabilities = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)
        self.metric.update_state(y_true, probabilities, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()


@tf.keras.utils.register_keras_serializable(package="QGECAF")
class OneMinus(tf.keras.layers.Layer):
    """Serializable replacement for a Lambda layer computing 1-x."""

    def call(self, inputs):
        return 1.0 - inputs


@tf.keras.utils.register_keras_serializable(package="QGECAF")
class EvidenceToDirichlet(tf.keras.layers.Layer):
    """Convert non-negative evidence into Dirichlet parameters alpha=e+1."""

    def call(self, evidence):
        return evidence + 1.0


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = len(labels) / (num_classes * np.maximum(counts, 1.0))
    return weights / weights.mean()


def build_model(feature_dims: Dict[str, int], num_classes: int, cfg: Config) -> tf.keras.Model:
    image_input = tf.keras.layers.Input((cfg.image_size, cfg.image_size, 1), name="image")
    rgb = tf.keras.layers.Concatenate(name="grayscale_to_rgb")([image_input, image_input, image_input])
    rgb = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_preprocess")(rgb)

    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=None if cfg.smoke_test else "imagenet",
        input_shape=(cfg.image_size, cfg.image_size, 3),
    )
    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - cfg.fine_tune_last)
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    feature_map = backbone(rgb)
    deep_projected = tf.keras.layers.Conv2D(cfg.embed_dim, 1, padding="same", name="deep_projection")(feature_map)
    spatial_shape = deep_projected.shape[1:3]
    if spatial_shape[0] is None or spatial_shape[1] is None:
        raise ValueError("MobileNetV2 spatial dimensions must be statically known.")
    deep_tokens = tf.keras.layers.Reshape((int(spatial_shape[0] * spatial_shape[1]), cfg.embed_dim),
                                          name="deep_spatial_tokens")(deep_projected)
    deep_tokens = tf.keras.layers.LayerNormalization(name="deep_token_norm")(deep_tokens)

    family_inputs = []
    family_tokens = []
    for family in FAMILY_NAMES:
        inp = tf.keras.layers.Input((feature_dims[family],), name=family)
        token = tf.keras.layers.Dense(cfg.embed_dim, activation="gelu", name=f"{family}_token_projection")(inp)
        token = tf.keras.layers.LayerNormalization(name=f"{family}_token_norm")(token)
        token = tf.keras.layers.Reshape((1, cfg.embed_dim), name=f"{family}_token")(token)
        family_inputs.append(inp)
        family_tokens.append(token)
    handcrafted_tokens = tf.keras.layers.Concatenate(axis=1, name="handcrafted_family_tokens")(family_tokens)

    key_dim = cfg.embed_dim // cfg.attention_heads
    deep_from_hand = tf.keras.layers.MultiHeadAttention(
        num_heads=cfg.attention_heads, key_dim=key_dim, dropout=0.1, name="deep_to_handcrafted_attention"
    )(query=deep_tokens, value=handcrafted_tokens, key=handcrafted_tokens)
    deep_enriched = tf.keras.layers.Add(name="deep_attention_residual")([deep_tokens, deep_from_hand])
    deep_enriched = tf.keras.layers.LayerNormalization(name="deep_attention_norm")(deep_enriched)

    hand_from_deep = tf.keras.layers.MultiHeadAttention(
        num_heads=cfg.attention_heads, key_dim=key_dim, dropout=0.1, name="handcrafted_to_deep_attention"
    )(query=handcrafted_tokens, value=deep_tokens, key=deep_tokens)
    hand_enriched = tf.keras.layers.Add(name="handcrafted_attention_residual")(
        [handcrafted_tokens, hand_from_deep]
    )
    hand_enriched = tf.keras.layers.LayerNormalization(name="handcrafted_attention_norm")(hand_enriched)

    deep_vector = tf.keras.layers.GlobalAveragePooling1D(name="deep_cross_attention_pool")(deep_enriched)
    hand_vector = tf.keras.layers.GlobalAveragePooling1D(name="handcrafted_cross_attention_pool")(hand_enriched)

    quality_input = tf.keras.layers.Input((len(QUALITY_NAMES),), name="quality")
    gate_hidden = tf.keras.layers.Dense(16, activation="relu", name="quality_gate_hidden")(quality_input)
    gate = tf.keras.layers.Dense(1, activation="sigmoid", name="quality_gate_lambda")(gate_hidden)
    inverse_gate = OneMinus(name="inverse_quality_gate")(gate)
    gated_deep = tf.keras.layers.Multiply(name="gated_deep_cross_attention")([deep_vector, gate])
    gated_hand = tf.keras.layers.Multiply(name="gated_handcrafted_cross_attention")([hand_vector, inverse_gate])
    fused = tf.keras.layers.Add(name="quality_gated_fusion")([gated_deep, gated_hand])
    fused = tf.keras.layers.LayerNormalization(name="fusion_norm")(fused)
    fused = tf.keras.layers.Dense(128, activation="gelu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(cfg.dropout, name="fusion_dropout")(fused)

    evidence = tf.keras.layers.Dense(num_classes, activation="softplus", name="class_evidence")(fused)
    alpha = EvidenceToDirichlet(name="dirichlet_alpha")(evidence)

    model = tf.keras.Model(
        inputs=[image_input, *family_inputs, quality_input], outputs=alpha, name="QGECAF_MaternalFetal"
    )
    return model


def make_inputs(
    raw: Dict[str, np.ndarray], transformed: Dict[str, np.ndarray], quality: np.ndarray, idx: np.ndarray
) -> Dict[str, np.ndarray]:
    result = {"image": raw["images"][idx].astype(np.float32) / 255.0, "quality": quality[idx]}
    for family in FAMILY_NAMES:
        result[family] = transformed[family][idx]
    return result


def quality_score_from_standardized(quality_z: np.ndarray) -> np.ndarray:
    # Higher means better. Brightness quality is closeness to the training mean (z=0).
    components = np.column_stack(
        [quality_z[:, 0], -np.abs(quality_z[:, 1]), quality_z[:, 2], quality_z[:, 3],
         -quality_z[:, 4], -quality_z[:, 5]]
    )
    raw_score = np.mean(components, axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(raw_score, -20, 20)))


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = prediction == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
    return float(ece)


def plot_reliability(y_true: np.ndarray, probabilities: np.ndarray, output: Path, bins: int = 10) -> None:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = prediction == y_true
    edges = np.linspace(0, 1, bins + 1)
    centers, accuracy, conf = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            centers.append((lower + upper) / 2)
            accuracy.append(correctness[selected].mean())
            conf.append(confidence[selected].mean())
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(conf, accuracy, "o-", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def evaluate(
    model: tf.keras.Model,
    inputs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    class_names: Sequence[str],
    quality_z: np.ndarray,
    paths: np.ndarray,
    output_dir: Path,
) -> dict:
    # Exclude graph tracing and initialization from network-only timing.
    warmup_count = min(16, len(y_true))
    warmup = {name: value[:warmup_count] for name, value in inputs.items()}
    _ = model.predict(warmup, batch_size=16, verbose=0)
    prediction_start = time.perf_counter()
    alpha = model.predict(inputs, batch_size=16, verbose=1)
    inference_seconds = float(time.perf_counter() - prediction_start)
    strength = alpha.sum(axis=1, keepdims=True)
    probabilities = alpha / strength
    uncertainty = len(class_names) / strength[:, 0]
    prediction = probabilities.argmax(axis=1)

    gate_model = tf.keras.Model(model.inputs, model.get_layer("quality_gate_lambda").output)
    gate_values = gate_model.predict(inputs, batch_size=16, verbose=0).reshape(-1)
    q_score = quality_score_from_standardized(quality_z)

    one_hot = tf.keras.utils.to_categorical(y_true, len(class_names))
    nll = -np.mean(np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-8, 1.0)))
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    ece = expected_calibration_error(y_true, probabilities)
    incorrect = (prediction != y_true).astype(int)
    try:
        error_auc = roc_auc_score(incorrect, uncertainty)
    except ValueError:
        error_auc = float("nan")
    rho, rho_p = spearmanr(q_score, uncertainty)

    metrics = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted")),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "ece_15_bins": ece,
        "brier_multiclass": float(brier),
        "negative_log_likelihood": float(nll),
        "uncertainty_error_auroc": float(error_auc),
        "quality_uncertainty_spearman_rho": float(rho),
        "quality_uncertainty_spearman_p": float(rho_p),
        "mean_uncertainty_correct": float(uncertainty[prediction == y_true].mean()),
        "mean_uncertainty_incorrect": float(uncertainty[prediction != y_true].mean())
        if np.any(prediction != y_true) else float("nan"),
        "inference_time_seconds": inference_seconds,
        "inference_time_ms_per_image": 1000.0 * inference_seconds / len(y_true),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(
        y_true, prediction, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, prediction, labels=np.arange(len(class_names)))
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    plot_reliability(y_true, probabilities, output_dir / "reliability_diagram.png")

    table = {
        "path": paths,
        "true_index": y_true,
        "true_class": [class_names[i] for i in y_true],
        "predicted_index": prediction,
        "predicted_class": [class_names[i] for i in prediction],
        "correct": prediction == y_true,
        "confidence": probabilities.max(axis=1),
        "uncertainty": uncertainty,
        "quality_score": q_score,
        "quality_gate_lambda": gate_values,
    }
    for i, name in enumerate(class_names):
        table[f"probability_{name}"] = probabilities[:, i]
        table[f"evidence_{name}"] = alpha[:, i] - 1.0
    pd.DataFrame(table).to_csv(output_dir / "test_predictions_uncertainty.csv", index=False)

    coverages = [1.0, 0.95, 0.90, 0.80]
    order = np.argsort(uncertainty)
    selective = []
    for coverage in coverages:
        retained = order[: max(1, int(round(len(order) * coverage)))]
        selective.append(
            {
                "coverage": coverage,
                "retained_samples": len(retained),
                "accuracy": accuracy_score(y_true[retained], prediction[retained]),
                "macro_f1": f1_score(y_true[retained], prediction[retained], average="macro"),
            }
        )
    pd.DataFrame(selective).to_csv(output_dir / "selective_prediction.csv", index=False)
    return metrics


def plot_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    frame = pd.DataFrame(history.history)
    frame.to_csv(output_dir / "training_history.csv", index=False)
    for metric in ("loss", "accuracy"):
        if metric not in frame:
            continue
        plt.figure(figsize=(7, 5))
        plt.plot(frame[metric], label=f"Training {metric}")
        if f"val_{metric}" in frame:
            plt.plot(frame[f"val_{metric}"], label=f"Validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"training_{metric}.png", dpi=300)
        plt.close()


class EpochTimer(tf.keras.callbacks.Callback):
    """Persist exact wall-clock epoch durations for interrupted-run audits."""

    def __init__(self, output_csv: Path):
        super().__init__()
        self.output_csv = Path(output_csv)
        self.rows: List[dict] = []
        self._start = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        row = {
            "epoch": int(epoch + 1),
            "epoch_time_seconds": float(time.perf_counter() - self._start),
        }
        for key, value in (logs or {}).items():
            row[key] = float(value)
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.output_csv, index=False)


def smoke_test(cfg: Config) -> None:
    print("Running architecture smoke test with synthetic data...")
    feature_dims = {name: 8 for name in FAMILY_NAMES}
    num_classes = 6
    model = build_model(feature_dims, num_classes, cfg)
    model.compile(optimizer="adam", loss=EvidentialLoss(num_classes), metrics=[DirichletMeanAccuracy()])
    batch = 2
    x = {"image": np.random.rand(batch, cfg.image_size, cfg.image_size, 1).astype(np.float32),
         "quality": np.random.randn(batch, len(QUALITY_NAMES)).astype(np.float32)}
    x.update({name: np.random.randn(batch, 8).astype(np.float32) for name in FAMILY_NAMES})
    y = tf.keras.utils.to_categorical(np.asarray([0, 1]), num_classes)
    model.train_on_batch(x, y)
    alpha = model.predict(x, verbose=0)
    assert alpha.shape == (batch, num_classes) and np.all(alpha >= 1.0)
    model.summary()
    print("Smoke test passed.")


def compile_model(model: tf.keras.Model, y_train: np.ndarray, num_classes: int, cfg: Config) -> None:
    weights = compute_class_weights(y_train, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=EvidentialLoss(num_classes, cfg.kl_weight, weights),
        metrics=[DirichletMeanAccuracy()],
    )


def summarize_fold_metrics(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    metric_names = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc",
        "ece_15_bins", "brier_multiclass", "negative_log_likelihood",
        "uncertainty_error_auroc", "quality_uncertainty_spearman_rho",
    ]
    rows = []
    for name in metric_names:
        values = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy(float)
        n = len(values)
        mean = float(np.mean(values)) if n else float("nan")
        sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
        half_width = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
        rows.append({
            "metric": name, "folds": n, "mean": mean, "standard_deviation": sd,
            "ci95_low": mean - half_width, "ci95_high": mean + half_width,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False)
    return summary


def compare_models_on_locked_folds(
    model_metric_files: Dict[str, str | Path],
    metric: str,
    output_dir: str | Path,
) -> Tuple[pd.DataFrame, dict]:
    """Paired fold analysis for models trained on the same locked assignments.

    Use only CSV files generated from identical ``locked_patient_fold_assignments.csv``.
    Friedman is calculated when at least three models are supplied. Pairwise
    Wilcoxon p-values receive a family-wise Holm adjustment.
    """
    series = {}
    for model_name, file_path in model_metric_files.items():
        frame = pd.read_csv(file_path)
        required = {"fold", metric}
        if not required.issubset(frame.columns):
            raise ValueError(f"{file_path} lacks columns {sorted(required - set(frame.columns))}")
        values = frame[["fold", metric]].drop_duplicates("fold").set_index("fold")[metric]
        series[model_name] = values
    paired = pd.DataFrame(series).sort_index().dropna()
    if len(paired) != 10:
        raise ValueError(f"Expected 10 complete paired folds, obtained {len(paired)}.")

    omnibus = {"metric": metric, "models": len(paired.columns), "folds": len(paired),
               "friedman_statistic": float("nan"), "friedman_p": float("nan")}
    if len(paired.columns) >= 3:
        result = friedmanchisquare(*[paired[name].to_numpy() for name in paired.columns])
        omnibus.update(friedman_statistic=float(result.statistic), friedman_p=float(result.pvalue))

    comparisons = []
    for left, right in combinations(paired.columns, 2):
        differences = paired[left].to_numpy() - paired[right].to_numpy()
        try:
            result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
            statistic, raw_p = float(result.statistic), float(result.pvalue)
        except ValueError:
            statistic, raw_p = 0.0, 1.0
        comparisons.append({
            "model_a": left, "model_b": right, "metric": metric,
            "mean_difference_a_minus_b": float(np.mean(differences)),
            "median_difference_a_minus_b": float(np.median(differences)),
            "wilcoxon_statistic": statistic, "raw_p": raw_p,
        })
    comparison_frame = pd.DataFrame(comparisons)
    if len(comparison_frame):
        order = np.argsort(comparison_frame["raw_p"].to_numpy())
        adjusted_sorted, running = np.empty(len(order)), 0.0
        for rank, original_position in enumerate(order):
            adjusted = min(1.0, (len(order) - rank) * comparison_frame.iloc[original_position]["raw_p"])
            running = max(running, adjusted)
            adjusted_sorted[rank] = running
        adjusted_original = np.empty(len(order))
        for rank, original_position in enumerate(order):
            adjusted_original[original_position] = adjusted_sorted[rank]
        comparison_frame["holm_adjusted_p"] = adjusted_original
        comparison_frame["significant_at_0.05"] = comparison_frame["holm_adjusted_p"] < 0.05

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paired.to_csv(destination / f"paired_{metric}_by_fold.csv")
    comparison_frame.to_csv(destination / f"wilcoxon_holm_{metric}.csv", index=False)
    (destination / f"friedman_{metric}.json").write_text(
        json.dumps(omnibus, indent=2), encoding="utf-8"
    )
    return comparison_frame, omnibus


def _extract_raw_batch(
    batch_paths: Sequence[str], cfg: Config, scattering,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, float, float]:
    """Read raw images and calculate every handcrafted/quality descriptor."""
    load_start = time.perf_counter()
    images = [read_grayscale(Path(path), cfg.image_size) for path in batch_paths]
    loading_seconds = float(time.perf_counter() - load_start)
    descriptor_start = time.perf_counter()
    families = {
        "wst": np.stack([wavelet_scattering_features(image, scattering) for image in images]),
        "gabor": np.stack([gabor_features(image) for image in images]),
        "lbp": np.stack([lbp_features(image) for image in images]),
        "hog": np.stack([hog_features(image) for image in images]),
        "shape": np.stack([shape_features(image) for image in images]),
    }
    quality = np.stack([quality_features(image) for image in images])
    descriptor_seconds = float(time.perf_counter() - descriptor_start)
    image_tensor = np.stack(images).astype(np.float32)[..., None] / 255.0
    return image_tensor, families, quality, loading_seconds, descriptor_seconds


def _transform_raw_batch(
    image_tensor: np.ndarray,
    raw_families: Dict[str, np.ndarray],
    raw_quality: np.ndarray,
    processors: dict,
) -> Dict[str, np.ndarray]:
    inputs: Dict[str, np.ndarray] = {"image": image_tensor}
    for family in FAMILY_NAMES:
        family_processor = processors[family]
        values = raw_families[family][:, family_processor["raw_mask"]]
        values = family_processor["scaler"].transform(values)
        if family_processor["pca"] is not None:
            values = family_processor["pca"].transform(values)
        inputs[family] = values.astype(np.float32)
    inputs["quality"] = processors["quality_scaler"].transform(raw_quality).astype(np.float32)
    return inputs


def benchmark_raw_image_to_prediction(
    model: tf.keras.Model,
    processors: dict,
    test_paths: Sequence[str],
    cfg: Config,
    output_dir: Path,
) -> pd.DataFrame:
    """Benchmark the complete uncached raw-image-to-prediction pipeline."""
    if Scattering2D is None:
        raise RuntimeError("End-to-end publication timing requires Kymatio.")
    scattering = Scattering2D(J=2, shape=(cfg.image_size, cfg.image_size), L=8, max_order=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "timing_protocol.json").write_text(json.dumps({
        "scope": "uncached raw image file to predicted class",
        "included_stages": ["disk read and resize", "WST", "Gabor", "LBP", "HOG",
                            "shape descriptors", "quality attributes", "raw mask",
                            "StandardScaler", "PCA", "host-to-device transfer and network forward pass",
                            "Dirichlet mean and argmax postprocessing"],
        "excluded_stages": ["model construction", "weight loading", "Scattering2D construction"],
        "warmup_images": int(min(cfg.latency_warmup_images, len(test_paths))),
        "repetitions": int(cfg.latency_repetitions),
        "batch_sizes": list(map(int, cfg.latency_batch_sizes)),
        "timer": "time.perf_counter",
        "feature_cache_used": False,
    }, indent=2), encoding="utf-8")
    warmup_paths = list(test_paths[: min(cfg.latency_warmup_images, len(test_paths))])
    for start in range(0, len(warmup_paths), 16):
        image_tensor, raw_families, raw_quality, _, _ = _extract_raw_batch(
            warmup_paths[start:start + 16], cfg, scattering
        )
        inputs = _transform_raw_batch(image_tensor, raw_families, raw_quality, processors)
        _ = model(inputs, training=False).numpy()

    rows: List[dict] = []
    ordered_paths = list(map(str, test_paths))
    for batch_size in cfg.latency_batch_sizes:
        for repetition in range(1, cfg.latency_repetitions + 1):
            stages = {"image_loading_seconds": 0.0,
                      "descriptors_quality_seconds": 0.0,
                      "transformations_seconds": 0.0,
                      "model_forward_seconds": 0.0,
                      "postprocessing_seconds": 0.0}
            total_start = time.perf_counter()
            for start in range(0, len(ordered_paths), batch_size):
                batch_paths = ordered_paths[start:start + batch_size]
                image_tensor, raw_families, raw_quality, loading, descriptors = _extract_raw_batch(
                    batch_paths, cfg, scattering
                )
                stages["image_loading_seconds"] += loading
                stages["descriptors_quality_seconds"] += descriptors
                transform_start = time.perf_counter()
                inputs = _transform_raw_batch(image_tensor, raw_families, raw_quality, processors)
                stages["transformations_seconds"] += float(time.perf_counter() - transform_start)
                forward_start = time.perf_counter()
                alpha = model(inputs, training=False).numpy()
                stages["model_forward_seconds"] += float(time.perf_counter() - forward_start)
                post_start = time.perf_counter()
                _ = np.argmax(alpha / alpha.sum(axis=1, keepdims=True), axis=1)
                stages["postprocessing_seconds"] += float(time.perf_counter() - post_start)
            total_seconds = float(time.perf_counter() - total_start)
            rows.append({"batch_size": int(batch_size), "repetition": repetition,
                         "images": len(ordered_paths), **stages,
                         "raw_image_to_prediction_seconds": total_seconds,
                         "raw_image_to_prediction_ms_per_image": 1000.0 * total_seconds / len(ordered_paths),
                         "raw_image_to_prediction_fps": len(ordered_paths) / total_seconds})
            print(f"End-to-end latency: batch={batch_size}, repetition={repetition}/"
                  f"{cfg.latency_repetitions}, "
                  f"{rows[-1]['raw_image_to_prediction_ms_per_image']:.3f} ms/image")

    raw_frame = pd.DataFrame(rows)
    raw_frame.to_csv(output_dir / "end_to_end_latency_raw.csv", index=False)
    summary_rows = []
    for batch_size, group in raw_frame.groupby("batch_size"):
        values = group["raw_image_to_prediction_ms_per_image"].to_numpy(float)
        summary_rows.append({"batch_size": int(batch_size), "repetitions": len(values),
                             "mean_ms_per_image": float(np.mean(values)),
                             "sample_SD_ms_per_image": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                             "median_ms_per_image": float(np.median(values)),
                             "iqr_ms_per_image": float(np.percentile(values, 75) - np.percentile(values, 25)),
                             "p95_ms_per_image": float(np.percentile(values, 95)),
                             "mean_fps": float(np.mean(group["raw_image_to_prediction_fps"]))})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "end_to_end_latency_summary.csv", index=False)
    return summary


def calculate_model_complexity(
    model: tf.keras.Model,
    output_dir: Path,
) -> dict:
    """Profile one batch-1 TensorFlow forward pass and save exact counts.

    The FLOP estimate covers only operations in the frozen QGECAF TensorFlow
    graph. CPU-side image reading, resizing, handcrafted descriptors, quality
    calculation, scaler/PCA transformations and postprocessing are deliberately
    excluded and are measured by ``benchmark_raw_image_to_prediction``.
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    input_specs = []
    for model_input in model.inputs:
        input_shape = [
            1 if dimension is None else int(dimension)
            for dimension in model_input.shape
        ]
        input_specs.append(
            tf.TensorSpec(
                shape=input_shape,
                dtype=model_input.dtype,
                name=model_input.name.split(":")[0],
            )
        )

    @tf.function
    def forward_pass(*model_inputs):
        return model(list(model_inputs), training=False)

    try:
        concrete_function = forward_pass.get_concrete_function(*input_specs)
        frozen_function = convert_variables_to_constants_v2(concrete_function)
        profile_options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        profile_options["output"] = "none"
        profile = tf.compat.v1.profiler.profile(
            graph=frozen_function.graph,
            run_meta=tf.compat.v1.RunMetadata(),
            cmd="op",
            options=profile_options,
        )
        total_flops = int(profile.total_float_ops) if profile is not None else 0
        if total_flops <= 0:
            raise RuntimeError("TensorFlow profiler returned zero floating-point operations.")
        profiler_error = None
    except Exception as exc:
        total_flops = 0
        profiler_error = repr(exc)
        warnings.warn(
            "TensorFlow FLOP profiling failed. Parameters and memory are still "
            f"reported, but GFLOPs will be NaN. Details: {exc}",
            RuntimeWarning,
        )

    total_parameters = int(model.count_params())
    trainable_parameters = int(
        sum(np.prod(variable.shape) for variable in model.trainable_weights)
    )
    nontrainable_parameters = int(
        sum(np.prod(variable.shape) for variable in model.non_trainable_weights)
    )
    gflops = total_flops / 1e9 if total_flops > 0 else float("nan")
    # One multiply-accumulate is conventionally approximated as two FLOPs.
    gmacs = total_flops / 2e9 if total_flops > 0 else float("nan")
    weight_memory_mib = total_parameters * 4 / (1024 ** 2)
    estimated_adam_parameter_memory_mib = (
        (total_parameters + 2 * trainable_parameters) * 4 / (1024 ** 2)
    )

    results = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "nontrainable_parameters": nontrainable_parameters,
        "total_parameters_million": total_parameters / 1e6,
        "network_flops_per_image": total_flops,
        "network_gflops_per_image": gflops,
        "approximate_gmacs_per_image": gmacs,
        "batch_size_used_for_profiling": 1,
        "float_precision_assumed_for_memory": "float32",
        "weight_memory_mib": weight_memory_mib,
        "estimated_adam_parameter_memory_mib": estimated_adam_parameter_memory_mib,
        "scope": "QGECAF TensorFlow model forward pass only",
        "excluded_from_network_gflops": [
            "image loading",
            "image resizing",
            "WST extraction",
            "Gabor extraction",
            "LBP extraction",
            "HOG extraction",
            "shape descriptor extraction",
            "quality attribute calculation",
            "StandardScaler transformation",
            "PCA transformation",
            "probability and argmax postprocessing",
        ],
        "profiler_error": profiler_error,
    }
    (output_dir / "model_complexity.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    summary_lines: List[str] = []
    model.summary(print_fn=summary_lines.append)
    (output_dir / "final_model_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print("\n" + "=" * 72)
    print("FINAL QGECAF MODEL COMPLEXITY")
    print("=" * 72)
    print(f"Total parameters          : {total_parameters:,}")
    print(f"Trainable parameters      : {trainable_parameters:,}")
    print(f"Non-trainable parameters  : {nontrainable_parameters:,}")
    print(f"Weight memory             : {weight_memory_mib:.3f} MiB")
    if total_flops > 0:
        print(f"Network GFLOPs/image      : {gflops:.6f}")
        print(f"Approximate GMACs/image   : {gmacs:.6f}")
    else:
        print("Network GFLOPs/image      : unavailable; inspect model_complexity.json")
    return results


def choose_inner_split(
    outer_train_idx: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    cfg: Config,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.seed + fold
    )
    splits = list(splitter.split(
        np.zeros(len(outer_train_idx)), labels[outer_train_idx],
        groups=patient_ids[outer_train_idx]
    ))
    inner_train_rel, inner_val_rel = splits[(fold - 1) % len(splits)]
    inner_train_idx = outer_train_idx[inner_train_rel]
    inner_val_idx = outer_train_idx[inner_val_rel]
    if set(patient_ids[inner_train_idx]).intersection(set(patient_ids[inner_val_idx])):
        raise RuntimeError("Patient leakage in the inner early-stopping split.")
    return inner_train_idx, inner_val_idx


def main(cfg: Config | None = None) -> None:
    cfg = parse_args() if cfg is None else cfg
    set_reproducibility(cfg.seed)
    configure_gpu()
    if cfg.smoke_test:
        smoke_test(cfg)
        return

    output_dir = Path(cfg.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    expected_protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "locked_manifest": str(Path(cfg.locked_manifest)),
        "outer_folds": int(cfg.outer_folds),
        "inner_folds": int(cfg.inner_folds),
        "seed": int(cfg.seed),
        "bgwo_preprocessing": "raw_mask_then_fold_train_scaler_then_fold_train_pca",
    }
    if protocol_path.is_file():
        observed_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if observed_protocol != expected_protocol:
            raise RuntimeError(
                "The existing output directory belongs to a different protocol. "
                "Use a new empty output directory instead of mixing experiments."
            )
    protocol_path.write_text(json.dumps(expected_protocol, indent=2), encoding="utf-8")
    (output_dir / "configuration.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    write_environment_manifest(output_dir)
    write_method_manifest(output_dir, cfg)

    # STEP 1-2: discover the fixed split and cache deterministic image features.
    paths, labels, class_names, split_names = discover_presplit_dataset(Path(cfg.dataset))
    raw = extract_all_features(paths, labels, cfg, output_dir / "raw_features_presplit.npz")
    labels = raw["labels"]
    train_idx, val_idx, test_idx = recover_presplit_indices(raw["paths"], paths, split_names)
    patient_ids, _ = load_patient_ids(
        raw["paths"], Path(cfg.metadata_csv), cfg.patient_column, cfg.image_column
    )
    validate_patient_separation(patient_ids, train_idx, val_idx, test_idx)

    num_classes = len(class_names)
    development_idx = np.concatenate([train_idx, val_idx])
    development_patients = set(patient_ids[development_idx])
    test_patients = set(patient_ids[test_idx])
    if development_patients.intersection(test_patients):
        raise RuntimeError("Development/test patient leakage detected.")

    patient_counts = []
    for class_index, class_name in enumerate(class_names):
        class_dev = development_idx[labels[development_idx] == class_index]
        n_patients = len(set(patient_ids[class_dev]))
        patient_counts.append({"class_index": class_index, "class_name": class_name,
                               "development_patients": n_patients,
                               "development_images": len(class_dev)})
        if n_patients < cfg.outer_folds:
            raise ValueError(
                f"Class '{class_name}' has only {n_patients} development patients; "
                f"{cfg.outer_folds}-fold grouped stratification is not defensible."
            )
    pd.DataFrame(patient_counts).to_csv(output_dir / "development_class_patient_counts.csv", index=False)
    (output_dir / "dataset_fingerprint_sha256.txt").write_text(
        dataset_fingerprint(raw["paths"], labels, patient_ids), encoding="utf-8"
    )
    (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    # STEP 3-4: recover—not regenerate—the original ten locked outer folds.
    fold_assignment = load_locked_fold_assignments(
        Path(cfg.locked_manifest), raw["paths"], patient_ids, labels,
        development_idx, test_idx, cfg.outer_folds,
    )
    development_folds = fold_assignment[development_idx]
    outer_splits = []
    for fold in range(1, cfg.outer_folds + 1):
        outer_holdout_rel = np.flatnonzero(development_folds == fold)
        outer_train_rel = np.flatnonzero(development_folds != fold)
        outer_splits.append((outer_train_rel, outer_holdout_rel))
    audit_table = pd.DataFrame({
        "path": raw["paths"], "patient_id": patient_ids, "label": labels,
        "class_name": [class_names[i] for i in labels],
        "original_split": np.where(np.isin(np.arange(len(labels)), train_idx), "train",
                            np.where(np.isin(np.arange(len(labels)), val_idx), "validation", "test")),
        "outer_fold": fold_assignment,
    })
    audit_table.to_csv(output_dir / "locked_patient_fold_assignments.csv", index=False)

    # STEP 5-8: train one fresh, fully fold-fitted model per outer fold.
    # A completed fold is resumable only when both its metrics row and its
    # exactly-once holdout predictions exist and match the currently locked fold.
    existing_metrics_path = output_dir / "fold_metrics.csv"
    existing_by_fold: Dict[int, dict] = {}
    if cfg.resume and existing_metrics_path.is_file():
        existing_frame = pd.read_csv(existing_metrics_path)
        if "fold" in existing_frame.columns:
            for _, existing_row in existing_frame.drop_duplicates("fold", keep="last").iterrows():
                existing_by_fold[int(existing_row["fold"])] = existing_row.to_dict()
        print(f"Resume enabled: found {len(existing_by_fold)} completed fold metric row(s).")

    fold_rows, best_epochs, oof_frames = [], [], []
    for fold, (outer_train_rel, outer_holdout_rel) in enumerate(outer_splits, start=1):
        print("\n" + "=" * 78)
        print(f"PATIENT-LEVEL OUTER FOLD {fold}/{cfg.outer_folds}")
        print("=" * 78)
        fold_dir = output_dir / "cross_validation" / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        outer_train_idx = development_idx[outer_train_rel]
        outer_holdout_idx = development_idx[outer_holdout_rel]
        completed_predictions = fold_dir / "outer_holdout_predictions.csv"
        if cfg.resume and fold in existing_by_fold and completed_predictions.is_file():
            saved_predictions = pd.read_csv(completed_predictions)
            expected_paths = set(map(str, raw["paths"][outer_holdout_idx]))
            saved_paths = set(saved_predictions["path"].astype(str)) if "path" in saved_predictions else set()
            if saved_paths != expected_paths or len(saved_predictions) != len(outer_holdout_idx):
                raise RuntimeError(
                    f"Fold {fold} checkpoint does not match the locked patient assignment. "
                    "Do not resume with a changed seed, dataset or split."
                )
            saved_row = existing_by_fold[fold]
            fold_rows.append(saved_row)
            best_epochs.append(int(saved_row["best_epoch"]))
            oof_frames.append(saved_predictions)
            print(f"Fold {fold} is complete and verified; skipping retraining.")
            continue

        inner_train_idx, inner_val_idx = choose_inner_split(
            outer_train_idx, labels, patient_ids, cfg, fold
        )
        train_groups = set(patient_ids[inner_train_idx])
        val_groups = set(patient_ids[inner_val_idx])
        holdout_groups = set(patient_ids[outer_holdout_idx])
        if train_groups & val_groups or train_groups & holdout_groups or val_groups & holdout_groups:
            raise RuntimeError(f"Patient leakage detected in fold {fold}.")
        pd.concat(
            [
                pd.DataFrame({"partition": "inner_train", "image_index": inner_train_idx,
                              "patient_id": patient_ids[inner_train_idx],
                              "path": raw["paths"][inner_train_idx]}),
                pd.DataFrame({"partition": "inner_validation", "image_index": inner_val_idx,
                              "patient_id": patient_ids[inner_val_idx],
                              "path": raw["paths"][inner_val_idx]}),
                pd.DataFrame({"partition": "outer_holdout", "image_index": outer_holdout_idx,
                              "patient_id": patient_ids[outer_holdout_idx],
                              "path": raw["paths"][outer_holdout_idx]}),
            ], ignore_index=True,
        ).to_csv(fold_dir / "patient_partition_audit.csv", index=False)

        fold_cfg = replace(cfg, seed=cfg.seed + fold)
        set_reproducibility(fold_cfg.seed)
        processors, transformed, quality_z = fit_preprocessing(
            raw, inner_train_idx, labels, patient_ids, fold_cfg,
            artifact_dir=fold_dir / "nested_preprocessing",
        )
        joblib.dump(processors, fold_dir / "preprocessing.joblib")
        x_train = make_inputs(raw, transformed, quality_z, inner_train_idx)
        x_val = make_inputs(raw, transformed, quality_z, inner_val_idx)
        x_holdout = make_inputs(raw, transformed, quality_z, outer_holdout_idx)
        y_train = tf.keras.utils.to_categorical(labels[inner_train_idx], num_classes)
        y_val = tf.keras.utils.to_categorical(labels[inner_val_idx], num_classes)
        feature_dims = {name: transformed[name].shape[1] for name in FAMILY_NAMES}

        tf.keras.backend.clear_session()
        model = build_model(feature_dims, num_classes, fold_cfg)
        compile_model(model, labels[inner_train_idx], num_classes, fold_cfg)
        epoch_timer = EpochTimer(fold_dir / "epoch_timing.csv")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(fold_dir / "best_model.keras"), monitor="val_loss",
                save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.early_stopping_patience,
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
            ),
            tf.keras.callbacks.CSVLogger(str(fold_dir / "epoch_log.csv")),
            epoch_timer,
        ]
        training_start = time.perf_counter()
        history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val), epochs=cfg.epochs,
            batch_size=cfg.batch_size, callbacks=callbacks, verbose=1
        )
        training_seconds = float(time.perf_counter() - training_start)
        plot_history(history, fold_dir)
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        best_epochs.append(best_epoch)
        metrics = evaluate(
            model, x_holdout, labels[outer_holdout_idx], class_names,
            quality_z[outer_holdout_idx], raw["paths"][outer_holdout_idx], fold_dir
        )
        metrics.update({
            "fold": fold, "best_epoch": best_epoch,
            "epochs_run": len(history.history["loss"]),
            "training_time_seconds": training_seconds,
            "mean_epoch_time_seconds": float(np.mean(
                [row["epoch_time_seconds"] for row in epoch_timer.rows]
            )),
            "inner_train_images": len(inner_train_idx), "inner_validation_images": len(inner_val_idx),
            "outer_holdout_images": len(outer_holdout_idx),
            "inner_train_patients": len(train_groups), "inner_validation_patients": len(val_groups),
            "outer_holdout_patients": len(holdout_groups),
        })
        fold_rows.append(metrics)
        fold_frame = pd.read_csv(fold_dir / "test_predictions_uncertainty.csv")
        fold_frame.insert(0, "outer_fold", fold)
        fold_frame.insert(1, "patient_id", patient_ids[outer_holdout_idx])
        fold_frame.to_csv(fold_dir / "outer_holdout_predictions.csv", index=False)
        oof_frames.append(fold_frame)
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
        (fold_dir / "FOLD_COMPLETE.json").write_text(json.dumps({
            "fold": fold, "best_epoch": best_epoch,
            "outer_holdout_images": len(outer_holdout_idx),
            "outer_holdout_patients": len(holdout_groups),
        }, indent=2), encoding="utf-8")

        del model, processors, transformed, quality_z, x_train, x_val, x_holdout
        gc.collect()
        tf.keras.backend.clear_session()

    # STEP 9: aggregate fold results and combine exactly-once OOF predictions.
    fold_metrics = pd.DataFrame(fold_rows).sort_values("fold")
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary = summarize_fold_metrics(fold_metrics, output_dir / "cross_validation_summary.csv")
    oof = pd.concat(oof_frames, ignore_index=True)
    if len(oof) != len(development_idx) or oof["path"].duplicated().any():
        raise RuntimeError("OOF predictions are incomplete or duplicated.")
    oof.to_csv(output_dir / "development_out_of_fold_predictions.csv", index=False)

    # STEP 10-11: final locked training on every development patient.
    # The median CV-optimal epoch avoids using the untouched test set for early stopping.
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    final_dir = output_dir / "final_untouched_test"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_cfg = replace(cfg, seed=cfg.seed + 1000)
    set_reproducibility(final_cfg.seed)
    processors, transformed, quality_z = fit_preprocessing(
        raw, development_idx, labels, patient_ids, final_cfg,
        artifact_dir=final_dir / "nested_preprocessing",
    )
    joblib.dump(processors, final_dir / "preprocessing.joblib")
    x_development = make_inputs(raw, transformed, quality_z, development_idx)
    x_test = make_inputs(raw, transformed, quality_z, test_idx)
    y_development = tf.keras.utils.to_categorical(labels[development_idx], num_classes)
    feature_dims = {name: transformed[name].shape[1] for name in FAMILY_NAMES}
    tf.keras.backend.clear_session()
    final_model = build_model(feature_dims, num_classes, final_cfg)
    compile_model(final_model, labels[development_idx], num_classes, final_cfg)
    final_epoch_timer = EpochTimer(final_dir / "epoch_timing.csv")
    final_training_start = time.perf_counter()
    final_history = final_model.fit(
        x_development, y_development, epochs=final_epochs, batch_size=cfg.batch_size,
        callbacks=[tf.keras.callbacks.CSVLogger(str(final_dir / "epoch_log.csv")),
                   final_epoch_timer], verbose=1
    )
    final_training_seconds = float(time.perf_counter() - final_training_start)
    plot_history(final_history, final_dir)
    final_model.save(final_dir / "final_model.keras")
    trainable_ids = {id(value) for value in final_model.trainable_weights}
    parameter_rows = []
    for variable in final_model.weights:
        parameter_rows.append({"variable": variable.name,
                               "parameters": int(np.prod(variable.shape)),
                               "trainable": id(variable) in trainable_ids,
                               "shape": str(tuple(int(value) for value in variable.shape))})
    pd.DataFrame(parameter_rows).to_csv(
        final_dir / "parameter_accounting_by_variable.csv", index=False
    )
    (final_dir / "parameter_summary.json").write_text(json.dumps({
        "total_parameters": int(final_model.count_params()),
        "trainable_parameters": int(sum(np.prod(value.shape) for value in final_model.trainable_weights)),
        "nontrainable_parameters": int(sum(np.prod(value.shape) for value in final_model.non_trainable_weights)),
    }, indent=2), encoding="utf-8")
    complexity_results = calculate_model_complexity(final_model, final_dir)
    final_metrics = evaluate(
        final_model, x_test, labels[test_idx], class_names, quality_z[test_idx],
        raw["paths"][test_idx], final_dir
    )
    final_predictions = pd.read_csv(final_dir / "test_predictions_uncertainty.csv")
    final_predictions.insert(1, "patient_id", patient_ids[test_idx])
    final_predictions.to_csv(final_dir / "test_predictions_uncertainty.csv", index=False)
    (final_dir / "final_training_protocol.json").write_text(json.dumps({
        "selected_epochs": final_epochs,
        "selection_rule": "rounded median of the 10 fold-specific best validation epochs",
        "development_images": len(development_idx),
        "development_patients": len(development_patients),
        "test_images": len(test_idx), "test_patients": len(test_patients),
        "test_used_during_model_selection": False,
        "training_time_seconds": final_training_seconds,
        "mean_epoch_time_seconds": float(np.mean(
            [row["epoch_time_seconds"] for row in final_epoch_timer.rows]
        )),
    }, indent=2), encoding="utf-8")
    end_to_end_summary = None
    if cfg.run_end_to_end_latency:
        end_to_end_summary = benchmark_raw_image_to_prediction(
            final_model, processors, raw["paths"][test_idx], cfg,
            final_dir / "raw_image_to_prediction_benchmark",
        )

    # STEP 12: concise publication table and run-completion marker.
    final_row = {"evaluation": "untouched_test", **final_metrics}
    final_row.update({
        "total_parameters": complexity_results["total_parameters"],
        "trainable_parameters": complexity_results["trainable_parameters"],
        "nontrainable_parameters": complexity_results["nontrainable_parameters"],
        "network_gflops_per_image": complexity_results["network_gflops_per_image"],
        "approximate_gmacs_per_image": complexity_results["approximate_gmacs_per_image"],
        "final_training_epochs": final_epochs,
        "final_training_time_seconds": final_training_seconds,
        "mean_epoch_time_seconds": float(np.mean(
            [row["epoch_time_seconds"] for row in final_epoch_timer.rows]
        )),
        "model_only_throughput_images_per_second": (
            1000.0 / final_metrics["inference_time_ms_per_image"]
        ),
    })
    if end_to_end_summary is not None and len(end_to_end_summary):
        batch_one = end_to_end_summary.loc[
            end_to_end_summary["batch_size"] == 1
        ]
        if len(batch_one):
            batch_one = batch_one.iloc[0]
            final_row.update({
                "end_to_end_batch_1_mean_ms_per_image": float(
                    batch_one["mean_ms_per_image"]
                ),
                "end_to_end_batch_1_sample_SD_ms_per_image": float(
                    batch_one["sample_SD_ms_per_image"]
                ),
                "end_to_end_batch_1_mean_fps": float(batch_one["mean_fps"]),
            })
    pd.DataFrame([final_row]).to_csv(output_dir / "final_test_metrics.csv", index=False)
    (output_dir / "RUN_COMPLETE.txt").write_text(
        "All 10 patient-level folds and the untouched final test evaluation completed successfully.\n",
        encoding="utf-8"
    )
    print("\n10-fold summary")
    print(summary.to_string(index=False))
    print("\nUntouched final-test metrics")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

