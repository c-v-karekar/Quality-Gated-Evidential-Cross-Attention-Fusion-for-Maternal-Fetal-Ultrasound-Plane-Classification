"""Matched G2 rerun and gate-behaviour analysis for QGECAF.

The program reuses, without refitting, every fold's saved BGWO masks,
StandardScaler objects, PCA objects, quality scaler and patient partitions.
The MobileNetV2 stream, handcrafted tokens, bidirectional cross-attention,
evidential head, KL loss, optimizer, seeds and holdouts are held fixed.

Only G2_matched_rerun is trained. Existing G0, G1, original G2 and G3 results
are read without modification from the completed matched-control experiment.
The new G2 run uses the same patient folds, preprocessing objects, seed
schedule, backbone, handcrafted tokens, cross-attention, evidential head,
loss, optimizer, callbacks and stopping rules as the other controls. Unlike
G3, its quality rows retain the correct image-to-quality correspondence.

The program writes a separate output tree and never deletes, replaces or
selects between the original G2 and G2_matched_rerun. The rerun is treated as
verification. In addition to fold-wise performance, it produces quality-group
gate summaries, fold-wise quality--lambda correlations, paired gate-behaviour
tests, and gate-distribution figures.

The untouched final-test cohort (outer_fold=-1) is rejected. The program is
resumable and writes one independently auditable result tree per control.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Match the completed evidential QGECAF execution. Run this program in a fresh
# kernel; TensorFlow operation-determinism state is process-global.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, spearmanr, t, wilcoxon
from sklearn.metrics import accuracy_score, f1_score


PROTOCOL_VERSION = "qgecaf_g2_matched_rerun_saved_preprocessing_v1"
FAMILY_NAMES = ("wst", "gabor", "lbp", "hog", "shape")
QUALITY_NAMES = ("blur", "brightness", "contrast", "entropy", "noise", "background")
TRAIN_VARIANTS = ("G2_matched_rerun",)
EXISTING_VARIANTS = (
    "G0_fixed_0p5",
    "G1_learned_global",
    "G2_correct_quality",
    "G3_shuffled_quality",
)
CONTROL_VARIANTS = (
    "G0_fixed_0p5",
    "G1_learned_global",
    "G2_matched_rerun",
    "G3_shuffled_quality",
)
ALL_VARIANTS = (*EXISTING_VARIANTS, *TRAIN_VARIANTS)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "mcc",
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
    qgecaf_source: str = str(
        Path(__file__).with_name("QGECAF_Reviewer_Revision_Nested_BGWO.py")
    )
    controls_root: str = (
        r"D:\chaitali\qgacef"
        r"\QGECAF_MATCHED_GATE_CONTROLS"
    )
    output: str = (
        r"D:\chaitali\qgacef"
        r"\QGECAF_G2_MATCHED_RERUN"
    )
    image_size: int = 128
    num_classes: int = 6
    outer_folds: int = 10
    seed: int = 42
    embed_dim: int = 64
    attention_heads: int = 4
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    kl_weight: float = 1e-3
    dropout: float = 0.30
    fine_tune_last: int = 30
    early_stopping_patience: int = 8
    gate_hidden_units: int = 16
    fixed_gate_lambda: float = 0.5
    ece_bins: int = 15
    lower_quality_percentile: float = 33.0
    upper_quality_percentile: float = 67.0
    resume: bool = True
    smoke_test: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qgecaf-root", default=Config.qgecaf_root)
    parser.add_argument("--qgecaf-source", default=Config.qgecaf_source)
    parser.add_argument("--controls-root", default=Config.controls_root)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--smoke-test", action="store_true")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_qgecaf_source(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"QGECAF source file not found: {path}\n"
            "Keep QGECAF_Reviewer_Revision_Nested_BGWO.py beside this file."
        )
    module_name = "qgecaf_gate_control_reference"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import QGECAF source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
        raise RuntimeError(f"{description} lacks {sorted(missing)}")
    result = frame.copy()
    result["patient_id"] = result["patient_id"].map(normalize_patient)
    result["record_key"] = [
        record_key(patient, path)
        for patient, path in zip(result["patient_id"], result["path"])
    ]
    if result["record_key"].duplicated().any():
        raise RuntimeError(f"Duplicate records in {description}.")
    return result


def read_reference(cfg: Config) -> Tuple[pd.DataFrame, List[str], dict]:
    root = Path(cfg.qgecaf_root)
    required = [
        root / "locked_patient_fold_assignments.csv",
        root / "raw_features_presplit.npz",
        root / "fold_metrics.csv",
        root / "class_names.json",
    ]
    for fold in range(1, cfg.outer_folds + 1):
        fold_dir = root / "cross_validation" / f"fold_{fold:02d}"
        required.extend([
            fold_dir / "patient_partition_audit.csv",
            fold_dir / "preprocessing.joblib",
            fold_dir / "outer_holdout_predictions.csv",
        ])
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(map(str, missing)))

    manifest = add_record_key(
        pd.read_csv(root / "locked_patient_fold_assignments.csv"),
        "locked patient manifest",
    )
    manifest["outer_fold"] = manifest["outer_fold"].astype(int)
    manifest["label"] = manifest["label"].astype(int)
    class_names = json.loads((root / "class_names.json").read_text(encoding="utf-8"))
    if len(class_names) != cfg.num_classes:
        raise RuntimeError("class_names.json does not contain six classes.")
    with np.load(root / "raw_features_presplit.npz", allow_pickle=True) as data:
        raw = {name: data[name] for name in data.files}
    if len(raw["paths"]) != len(manifest):
        raise RuntimeError("Raw cache and locked manifest have different lengths.")
    if not np.array_equal(raw["paths"].astype(str), manifest["path"].astype(str).to_numpy()):
        raise RuntimeError("Raw-cache path order differs from the locked manifest.")
    if not np.array_equal(raw["labels"].astype(int), manifest["label"].to_numpy(int)):
        raise RuntimeError("Raw-cache labels differ from the locked manifest.")
    return manifest, [str(name) for name in class_names], raw


def validate_protocol(cfg: Config, output: Path) -> None:
    root = Path(cfg.qgecaf_root)
    controls_root = Path(cfg.controls_root)
    source = Path(cfg.qgecaf_source)
    output_preexisted = output.exists() and any(output.iterdir())
    output.mkdir(parents=True, exist_ok=True)
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "qgecaf_root": str(root),
        "controls_root": str(controls_root),
        "locked_manifest_sha256": sha256_file(root / "locked_patient_fold_assignments.csv"),
        "qgecaf_source_sha256": sha256_file(source),
        "existing_control_metrics_sha256": sha256_file(
            controls_root / "all_gate_control_fold_metrics.csv"
        ),
        "variants": list(ALL_VARIANTS),
        "G2_original_policy": "read completed QGECAF results; never overwrite",
        "G2_matched_rerun_policy": (
            "train once with the exact matched-control pipeline and correct quality rows"
        ),
        "preprocessing_policy": "load each saved fold preprocessing.joblib; never refit",
        "existing_controls_policy": "read G0, G1 and G3; never retrain or overwrite",
        "untouched_test_policy": "outer_fold=-1 excluded from all operations",
        "seed": cfg.seed,
        "fixed_gate_lambda": cfg.fixed_gate_lambda,
    }
    protocol_file = output / "protocol.json"
    if output_preexisted and not protocol_file.is_file():
        raise RuntimeError("Non-empty output has no protocol.json; use a new output folder.")
    if protocol_file.is_file():
        observed = json.loads(protocol_file.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError("Output belongs to an incompatible gate-control protocol.")
    protocol_file.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    (output / "configuration.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    (output / "environment_manifest.json").write_text(
        json.dumps({
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
        }, indent=2),
        encoding="utf-8",
    )


def load_fold(
    cfg: Config,
    fold: int,
    manifest: pd.DataFrame,
    raw: dict,
) -> Tuple[dict, dict, pd.DataFrame, str]:
    root = Path(cfg.qgecaf_root)
    fold_dir = root / "cross_validation" / f"fold_{fold:02d}"
    audit = pd.read_csv(fold_dir / "patient_partition_audit.csv")
    needed = {"partition", "image_index", "patient_id", "path"}
    if not needed.issubset(audit.columns):
        raise RuntimeError(f"Fold {fold} audit lacks {sorted(needed.difference(audit.columns))}")
    audit["patient_id"] = audit["patient_id"].map(normalize_patient)
    audit["image_index"] = audit["image_index"].astype(int)
    expected_partitions = {"inner_train", "inner_validation", "outer_holdout"}
    if set(audit["partition"]) != expected_partitions:
        raise RuntimeError(f"Fold {fold} has invalid audit partitions.")
    partitions = {
        name: audit.loc[audit["partition"] == name].copy()
        for name in expected_partitions
    }
    patient_sets = {name: set(frame["patient_id"]) for name, frame in partitions.items()}
    names = list(expected_partitions)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = patient_sets[left].intersection(patient_sets[right])
            if overlap:
                raise RuntimeError(f"Fold {fold}: patient leakage between {left} and {right}.")
    for name, frame in partitions.items():
        idx = frame["image_index"].to_numpy(int)
        if not np.array_equal(raw["paths"][idx].astype(str), frame["path"].astype(str).to_numpy()):
            raise RuntimeError(f"Fold {fold}: raw-cache paths do not match {name} audit.")
        manifest_fold = manifest.iloc[idx]["outer_fold"].to_numpy(int)
        if name == "outer_holdout" and not np.all(manifest_fold == fold):
            raise RuntimeError(f"Fold {fold}: incorrect outer holdout assignment.")
        if name != "outer_holdout" and np.any((manifest_fold == fold) | (manifest_fold == -1)):
            raise RuntimeError(f"Fold {fold}: invalid development row in {name}.")
    if any(np.any(manifest.iloc[f["image_index"].to_numpy(int)]["outer_fold"].to_numpy(int) == -1)
           for f in partitions.values()):
        raise RuntimeError(f"Fold {fold}: untouched final-test row entered the control.")

    preprocessing_file = fold_dir / "preprocessing.joblib"
    processors = joblib.load(preprocessing_file)
    protocol = processors.get("protocol", {})
    if int(protocol.get("fitted_images", -1)) != len(partitions["inner_train"]):
        raise RuntimeError(f"Fold {fold}: preprocessor fitted-image audit failed.")
    for family in FAMILY_NAMES:
        if family not in processors:
            raise RuntimeError(f"Fold {fold}: missing saved {family} preprocessing.")
    if "quality_scaler" not in processors:
        raise RuntimeError(f"Fold {fold}: missing saved quality scaler.")
    return partitions, processors, audit, sha256_file(preprocessing_file)


def transform_partition(
    raw: dict,
    processors: dict,
    frame: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int]]:
    idx = frame["image_index"].to_numpy(int)
    quality_z = processors["quality_scaler"].transform(raw["quality"][idx]).astype(np.float32)
    inputs: Dict[str, np.ndarray] = {
        "image": raw["images"][idx].astype(np.float32) / 255.0,
        "quality": quality_z,
    }
    dimensions = {}
    for family in FAMILY_NAMES:
        details = processors[family]
        mask = np.asarray(details["raw_mask"], dtype=bool)
        selected = np.asarray(raw[family][idx][:, mask], dtype=np.float32)
        transformed = details["scaler"].transform(selected)
        if details.get("pca") is not None:
            transformed = details["pca"].transform(transformed)
        inputs[family] = np.asarray(transformed, dtype=np.float32)
        dimensions[family] = int(inputs[family].shape[1])
    return inputs, quality_z, dimensions


for registered_name in (
    "GateControls>FixedGate",
    "GateControls>GlobalGate",
):
    tf.keras.utils.get_custom_objects().pop(registered_name, None)


@tf.keras.utils.register_keras_serializable(package="GateControls")
class FixedGate(tf.keras.layers.Layer):
    def __init__(self, value: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.value = float(value)

    def call(self, inputs):
        return tf.ones_like(inputs[:, :1]) * tf.cast(self.value, inputs.dtype)

    def get_config(self):
        return {**super().get_config(), "value": self.value}


@tf.keras.utils.register_keras_serializable(package="GateControls")
class GlobalGate(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.logit_lambda = self.add_weight(
            name="logit_lambda",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        value = tf.math.sigmoid(self.logit_lambda)
        return tf.ones_like(inputs[:, :1]) * tf.cast(value, inputs.dtype)


def build_gate_model(q, feature_dims: Dict[str, int], cfg: Config, variant: str) -> tf.keras.Model:
    image_input = tf.keras.layers.Input((cfg.image_size, cfg.image_size, 1), name="image")
    rgb = tf.keras.layers.Concatenate(name="grayscale_to_rgb")([image_input] * 3)
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
    deep_projected = tf.keras.layers.Conv2D(
        cfg.embed_dim, 1, padding="same", name="deep_projection"
    )(feature_map)
    spatial = deep_projected.shape[1:3]
    deep_tokens = tf.keras.layers.Reshape(
        (int(spatial[0] * spatial[1]), cfg.embed_dim), name="deep_spatial_tokens"
    )(deep_projected)
    deep_tokens = tf.keras.layers.LayerNormalization(name="deep_token_norm")(deep_tokens)

    family_inputs, family_tokens = [], []
    for family in FAMILY_NAMES:
        inp = tf.keras.layers.Input((feature_dims[family],), name=family)
        token = tf.keras.layers.Dense(
            cfg.embed_dim, activation="gelu", name=f"{family}_token_projection"
        )(inp)
        token = tf.keras.layers.LayerNormalization(name=f"{family}_token_norm")(token)
        token = tf.keras.layers.Reshape((1, cfg.embed_dim), name=f"{family}_token")(token)
        family_inputs.append(inp)
        family_tokens.append(token)
    handcrafted_tokens = tf.keras.layers.Concatenate(
        axis=1, name="handcrafted_family_tokens"
    )(family_tokens)
    key_dim = cfg.embed_dim // cfg.attention_heads
    deep_from_hand = tf.keras.layers.MultiHeadAttention(
        num_heads=cfg.attention_heads,
        key_dim=key_dim,
        dropout=0.1,
        name="deep_to_handcrafted_attention",
    )(query=deep_tokens, value=handcrafted_tokens, key=handcrafted_tokens)
    deep_enriched = tf.keras.layers.LayerNormalization(name="deep_attention_norm")(
        tf.keras.layers.Add(name="deep_attention_residual")([deep_tokens, deep_from_hand])
    )
    hand_from_deep = tf.keras.layers.MultiHeadAttention(
        num_heads=cfg.attention_heads,
        key_dim=key_dim,
        dropout=0.1,
        name="handcrafted_to_deep_attention",
    )(query=handcrafted_tokens, value=deep_tokens, key=deep_tokens)
    hand_enriched = tf.keras.layers.LayerNormalization(name="handcrafted_attention_norm")(
        tf.keras.layers.Add(name="handcrafted_attention_residual")(
            [handcrafted_tokens, hand_from_deep]
        )
    )
    deep_vector = tf.keras.layers.GlobalAveragePooling1D(
        name="deep_cross_attention_pool"
    )(deep_enriched)
    hand_vector = tf.keras.layers.GlobalAveragePooling1D(
        name="handcrafted_cross_attention_pool"
    )(hand_enriched)

    quality_input = tf.keras.layers.Input((len(QUALITY_NAMES),), name="quality")
    if variant == "G0_fixed_0p5":
        gate = FixedGate(cfg.fixed_gate_lambda, name="quality_gate_lambda")(quality_input)
    elif variant == "G1_learned_global":
        gate = GlobalGate(name="quality_gate_lambda")(quality_input)
    elif variant in ("G2_matched_rerun", "G3_shuffled_quality"):
        gate_hidden = tf.keras.layers.Dense(
            cfg.gate_hidden_units, activation="relu", name="quality_gate_hidden"
        )(quality_input)
        gate = tf.keras.layers.Dense(
            1, activation="sigmoid", name="quality_gate_lambda"
        )(gate_hidden)
    else:
        raise ValueError(f"Unsupported trained gate variant: {variant}")
    inverse_gate = q.OneMinus(name="inverse_quality_gate")(gate)
    gated_deep = tf.keras.layers.Multiply(name="gated_deep_cross_attention")(
        [deep_vector, gate]
    )
    gated_hand = tf.keras.layers.Multiply(name="gated_handcrafted_cross_attention")(
        [hand_vector, inverse_gate]
    )
    fused = tf.keras.layers.Add(name="quality_gated_fusion")([gated_deep, gated_hand])
    fused = tf.keras.layers.LayerNormalization(name="fusion_norm")(fused)
    fused = tf.keras.layers.Dense(128, activation="gelu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(cfg.dropout, name="fusion_dropout")(fused)
    evidence = tf.keras.layers.Dense(
        cfg.num_classes, activation="softplus", name="class_evidence"
    )(fused)
    alpha = q.EvidenceToDirichlet(name="dirichlet_alpha")(evidence)
    return tf.keras.Model(
        inputs=[image_input, *family_inputs, quality_input],
        outputs=alpha,
        name=f"QGECAF_{variant}",
    )


def validate_existing_controls(cfg: Config) -> pd.DataFrame:
    """Load the completed G0/G1/original-G2/G3 metrics without changing them."""
    controls_root = Path(cfg.controls_root)
    required = [
        controls_root / "all_gate_control_fold_metrics.csv",
        controls_root / "quality_stratified_gate_controls.csv",
        controls_root / "RUN_COMPLETE.json",
    ]
    for variant in ("G0_fixed_0p5", "G1_learned_global", "G3_shuffled_quality"):
        for fold in range(1, cfg.outer_folds + 1):
            required.append(
                controls_root / variant / "cross_validation" / f"fold_{fold:02d}"
                / "outer_holdout_predictions.csv"
            )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The completed matched-control run is incomplete:\n"
            + "\n".join(map(str, missing))
        )
    metrics = pd.read_csv(required[0])
    metrics["fold"] = metrics["fold"].astype(int)
    observed = set(metrics["variant"].astype(str))
    if not set(EXISTING_VARIANTS).issubset(observed):
        raise RuntimeError(
            "Existing controls do not contain all required variants: "
            f"{sorted(set(EXISTING_VARIANTS).difference(observed))}"
        )
    selected = metrics.loc[metrics["variant"].isin(EXISTING_VARIANTS)].copy()
    for variant in EXISTING_VARIANTS:
        folds = set(selected.loc[selected["variant"] == variant, "fold"])
        if folds != set(range(1, cfg.outer_folds + 1)):
            raise RuntimeError(f"Existing {variant} metrics do not contain all ten folds.")
    if selected.duplicated(["variant", "fold"]).any():
        raise RuntimeError("Existing control metrics contain duplicate variant-fold rows.")
    return selected


def existing_prediction_file(cfg: Config, variant: str, fold: int) -> Path:
    if variant == "G2_correct_quality":
        return (
            Path(cfg.qgecaf_root) / "cross_validation" / f"fold_{fold:02d}"
            / "outer_holdout_predictions.csv"
        )
    return (
        Path(cfg.controls_root) / variant / "cross_validation" / f"fold_{fold:02d}"
        / "outer_holdout_predictions.csv"
    )


def shuffled_quality_inputs(
    inputs: Dict[str, np.ndarray],
    frame: pd.DataFrame,
    seed: int,
    output_csv: Path,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(frame))
    result = {name: value.copy() for name, value in inputs.items()}
    result["quality"] = inputs["quality"][permutation].copy()
    mapping = pd.DataFrame({
        "target_record_key": [record_key(p, x) for p, x in zip(frame["patient_id"], frame["path"])],
        "quality_source_record_key": [
            record_key(p, x)
            for p, x in zip(
                frame["patient_id"].to_numpy()[permutation],
                frame["path"].to_numpy()[permutation],
            )
        ],
        "permutation_index": permutation,
    })
    mapping.to_csv(output_csv, index=False)
    return result


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (
            (confidence >= lower) & (confidence <= upper)
            if index == 0 else
            (confidence > lower) & (confidence <= upper)
        )
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def quality_stratified_rows(
    fold: int,
    variant: str,
    predictions: pd.DataFrame,
    q33: float,
    q67: float,
    class_names: Sequence[str],
    cfg: Config,
) -> List[dict]:
    frame = predictions.copy()
    frame["quality_group"] = np.where(
        frame["quality_score"] < q33,
        "Low",
        np.where(frame["quality_score"] <= q67, "Medium", "High"),
    )
    probability_columns = [f"probability_{name}" for name in class_names]
    rows = []
    for group_name in ("Low", "Medium", "High"):
        group = frame.loc[frame["quality_group"] == group_name]
        if group.empty:
            rows.append({
                "variant": variant,
                "fold": fold,
                "quality_group": group_name,
                "inner_train_q33": q33,
                "inner_train_q67": q67,
                "images": 0,
                "patients": 0,
                "classes_present": 0,
                "missing_classes": ";".join(class_names),
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "negative_log_likelihood": np.nan,
                "brier_multiclass": np.nan,
                "ece_15_bins": np.nan,
                "gate_lambda_mean": np.nan,
                "gate_lambda_sample_SD": np.nan,
                "gate_lambda_median": np.nan,
                "gate_lambda_IQR": np.nan,
            })
            continue
        y = group["true_index"].to_numpy(int)
        probabilities = group[probability_columns].to_numpy(float)
        prediction = probabilities.argmax(axis=1)
        one_hot = np.eye(cfg.num_classes)[y]
        gate = group["quality_gate_lambda"].to_numpy(float)
        present = sorted(set(y.tolist()))
        rows.append({
            "variant": variant,
            "fold": fold,
            "quality_group": group_name,
            "inner_train_q33": q33,
            "inner_train_q67": q67,
            "images": len(group),
            "patients": group["patient_id"].nunique(),
            "classes_present": len(present),
            "missing_classes": ";".join(
                class_names[i] for i in range(cfg.num_classes) if i not in present
            ),
            "accuracy": accuracy_score(y, prediction),
            "macro_f1": f1_score(
                y, prediction, labels=np.arange(cfg.num_classes),
                average="macro", zero_division=0,
            ),
            "negative_log_likelihood": -np.mean(
                np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-8, 1.0))
            ),
            "brier_multiclass": np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)),
            "ece_15_bins": expected_calibration_error(y, probabilities, cfg.ece_bins),
            "gate_lambda_mean": np.mean(gate),
            "gate_lambda_sample_SD": np.std(gate, ddof=1) if len(gate) > 1 else np.nan,
            "gate_lambda_median": np.median(gate),
            "gate_lambda_IQR": np.percentile(gate, 75) - np.percentile(gate, 25),
        })
    return rows


def gate_value_rows(
    fold: int,
    variant: str,
    predictions: pd.DataFrame,
    q33: float,
    q67: float,
) -> pd.DataFrame:
    """Return image-level gate values used only for gate-behaviour analysis."""
    needed = {
        "patient_id", "path", "quality_score", "quality_gate_lambda",
        "true_index", "predicted_index",
    }
    missing = needed.difference(predictions.columns)
    if missing:
        raise RuntimeError(
            f"{variant} fold {fold} predictions lack gate-analysis columns: "
            f"{sorted(missing)}"
        )
    frame = predictions.copy()
    frame["patient_id"] = frame["patient_id"].map(normalize_patient)
    frame["variant"] = variant
    frame["fold"] = fold
    frame["quality_group"] = np.where(
        frame["quality_score"] < q33,
        "Low",
        np.where(frame["quality_score"] <= q67, "Medium", "High"),
    )
    if "gate_input_quality_score" not in frame.columns:
        frame["gate_input_quality_score"] = frame["quality_score"]
    keep = [
        "variant", "fold", "patient_id", "path", "quality_group",
        "quality_score", "gate_input_quality_score", "quality_gate_lambda",
        "true_index", "predicted_index",
    ]
    return frame[keep]


def summarize_quality_strata(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "accuracy", "macro_f1", "negative_log_likelihood",
        "brier_multiclass", "ece_15_bins", "gate_lambda_mean",
        "gate_lambda_sample_SD", "gate_lambda_median", "gate_lambda_IQR",
    )
    rows = []
    for (variant, quality_group), group in frame.groupby(
        ["variant", "quality_group"], sort=False
    ):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            n = len(values)
            mean = np.mean(values) if n else np.nan
            sd = np.std(values, ddof=1) if n > 1 else np.nan
            half = t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else np.nan
            rows.append({
                "variant": variant,
                "quality_group": quality_group,
                "metric": metric,
                "folds": n,
                "mean": mean,
                "sample_SD": sd,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
            })
    return pd.DataFrame(rows)


def pooled_gate_summary(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, quality_group), group in values.groupby(
        ["variant", "quality_group"], sort=False
    ):
        gate = group["quality_gate_lambda"].to_numpy(float)
        rows.append({
            "variant": variant,
            "quality_group": quality_group,
            "images": len(group),
            "patients": group["patient_id"].nunique(),
            "gate_lambda_mean": np.mean(gate),
            "gate_lambda_sample_SD": np.std(gate, ddof=1) if len(gate) > 1 else np.nan,
            "gate_lambda_median": np.median(gate),
            "gate_lambda_q25": np.percentile(gate, 25),
            "gate_lambda_q75": np.percentile(gate, 75),
            "gate_lambda_IQR": np.percentile(gate, 75) - np.percentile(gate, 25),
            "gate_lambda_minimum": np.min(gate),
            "gate_lambda_maximum": np.max(gate),
        })
    return pd.DataFrame(rows)


def gate_quality_correlations(values: pd.DataFrame, output: Path) -> None:
    rows = []
    for (variant, fold), group in values.groupby(["variant", "fold"], sort=False):
        gate = group["quality_gate_lambda"].to_numpy(float)
        for quality_column, relation in (
            ("quality_score", "actual_quality_vs_lambda"),
            ("gate_input_quality_score", "gate_input_quality_vs_lambda"),
        ):
            quality = group[quality_column].to_numpy(float)
            if np.allclose(gate, gate[0]) or np.allclose(quality, quality[0]):
                rho, p_value = np.nan, np.nan
            else:
                result = spearmanr(quality, gate, nan_policy="omit")
                rho, p_value = float(result.statistic), float(result.pvalue)
            rows.append({
                "variant": variant,
                "fold": int(fold),
                "relationship": relation,
                "images": len(group),
                "patients": group["patient_id"].nunique(),
                "spearman_rho": rho,
                "raw_p": p_value,
            })
    correlation_frame = pd.DataFrame(rows)
    correlation_frame.to_csv(
        output / "gate_quality_spearman_by_fold.csv", index=False
    )
    summary_rows = []
    for (variant, relation), group in correlation_frame.groupby(
        ["variant", "relationship"], sort=False
    ):
        values_rho = group["spearman_rho"].dropna().to_numpy(float)
        n = len(values_rho)
        mean = np.mean(values_rho) if n else np.nan
        sd = np.std(values_rho, ddof=1) if n > 1 else np.nan
        half = t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else np.nan
        summary_rows.append({
            "variant": variant,
            "relationship": relation,
            "folds": n,
            "mean_spearman_rho": mean,
            "sample_SD": sd,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
        })
    pd.DataFrame(summary_rows).to_csv(
        output / "gate_quality_spearman_summary.csv", index=False
    )


def gate_group_tests(quality_frame: pd.DataFrame, output: Path) -> None:
    omnibus_rows, paired_rows = [], []
    group_order = ("Low", "Medium", "High")
    adaptive_variants = (
        "G1_learned_global", "G2_correct_quality",
        "G2_matched_rerun", "G3_shuffled_quality",
    )
    for variant in adaptive_variants:
        selected = quality_frame.loc[quality_frame["variant"] == variant]
        pivot = selected.pivot(
            index="fold", columns="quality_group", values="gate_lambda_mean"
        ).reindex(columns=group_order).dropna()
        if len(pivot) == 10:
            try:
                result = friedmanchisquare(
                    *(pivot[group].to_numpy(float) for group in group_order)
                )
                statistic, raw_p = float(result.statistic), float(result.pvalue)
            except ValueError:
                statistic, raw_p = 0.0, 1.0
            omnibus_rows.append({
                "variant": variant,
                "folds": len(pivot),
                "friedman_statistic": statistic,
                "raw_p": raw_p,
            })
        variant_rows = []
        for left, right in (("Low", "Medium"), ("Low", "High"), ("Medium", "High")):
            differences = pivot[left].to_numpy(float) - pivot[right].to_numpy(float)
            try:
                test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
                statistic, raw_p = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, raw_p = 0.0, 1.0
            variant_rows.append({
                "variant": variant,
                "comparison": f"{left}_minus_{right}",
                "folds": len(differences),
                "mean_gate_lambda_difference": np.mean(differences),
                "wilcoxon_statistic": statistic,
                "raw_p": raw_p,
                "rank_biserial": rank_biserial(differences),
            })
        adjusted = holm_adjust(np.asarray([row["raw_p"] for row in variant_rows]))
        for row, adjusted_p in zip(variant_rows, adjusted):
            row["holm_adjusted_p_within_variant"] = adjusted_p
            row["significant_at_0.05"] = adjusted_p < 0.05
            paired_rows.append(row)
    pd.DataFrame(omnibus_rows).to_csv(
        output / "gate_lambda_quality_group_friedman.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        output / "gate_lambda_quality_group_wilcoxon_holm.csv", index=False
    )


def plot_gate_distributions(values: pd.DataFrame, output: Path) -> None:
    variants = [
        "G1_learned_global", "G2_correct_quality",
        "G2_matched_rerun", "G3_shuffled_quality",
    ]
    group_order = ["Low", "Medium", "High"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, variant in zip(axes.ravel(), variants):
        selected = values.loc[values["variant"] == variant]
        arrays = [
            selected.loc[selected["quality_group"] == group, "quality_gate_lambda"]
            .dropna().to_numpy(float)
            for group in group_order
        ]
        axis.boxplot(arrays, labels=group_order, showfliers=False)
        axis.set_title(variant.replace("_", " "))
        axis.set_xlabel("Quality group")
        axis.set_ylabel(r"Gate weight $\lambda$")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Gate-weight distributions on the locked outer holdouts")
    fig.tight_layout()
    fig.savefig(output / "gate_lambda_by_quality_group.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in frame.groupby("variant", sort=False):
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            n = len(values)
            mean = np.mean(values) if n else np.nan
            sd = np.std(values, ddof=1) if n > 1 else np.nan
            half = t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else np.nan
            rows.append({
                "variant": variant,
                "metric": metric,
                "folds": n,
                "mean": mean,
                "standard_deviation": sd,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
            })
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def statistical_comparisons(frame: pd.DataFrame, output: Path) -> None:
    omnibus_rows, pairwise_rows = [], []
    reference = "G2_matched_rerun"
    comparisons = [
        "G0_fixed_0p5", "G1_learned_global", "G3_shuffled_quality"
    ]
    verification_rows = []
    for metric in METRICS:
        pivot = frame.pivot(index="fold", columns="variant", values=metric)
        control_pivot = pivot[list(CONTROL_VARIANTS)].dropna()
        if len(control_pivot) == 10:
            result = friedmanchisquare(
                *(control_pivot[name].to_numpy() for name in CONTROL_VARIANTS)
            )
            omnibus_rows.append({
                "metric": metric,
                "folds": len(control_pivot),
                "variants": ";".join(CONTROL_VARIANTS),
                "friedman_statistic": result.statistic,
                "raw_p": result.pvalue,
            })
        metric_rows = []
        for comparison in comparisons:
            paired = control_pivot[[reference, comparison]].dropna()
            differences = paired[reference].to_numpy() - paired[comparison].to_numpy()
            try:
                test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
                statistic, raw_p = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, raw_p = 0.0, 1.0
            metric_rows.append({
                "metric": metric,
                "reference": reference,
                "comparison": comparison,
                "folds": len(paired),
                "mean_difference_G2R_minus_control": np.mean(differences),
                "wilcoxon_statistic": statistic,
                "raw_p": raw_p,
                "rank_biserial_G2R_minus_control": rank_biserial(differences),
            })
        adjusted = holm_adjust(np.asarray([row["raw_p"] for row in metric_rows]))
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["holm_adjusted_p_within_metric"] = adjusted_p
            row["significant_at_0.05"] = adjusted_p < 0.05
            pairwise_rows.append(row)

        verification = pivot[[reference, "G2_correct_quality"]].dropna()
        verification_differences = (
            verification[reference].to_numpy(float)
            - verification["G2_correct_quality"].to_numpy(float)
        )
        try:
            test = wilcoxon(
                verification_differences, alternative="two-sided", zero_method="wilcox"
            )
            verification_statistic = float(test.statistic)
            verification_p = float(test.pvalue)
        except ValueError:
            verification_statistic, verification_p = 0.0, 1.0
        verification_rows.append({
            "metric": metric,
            "reference": reference,
            "comparison": "G2_correct_quality",
            "folds": len(verification),
            "original_G2_mean": verification["G2_correct_quality"].mean(),
            "G2_matched_rerun_mean": verification[reference].mean(),
            "mean_difference_G2R_minus_original_G2": np.mean(
                verification_differences
            ),
            "wilcoxon_statistic": verification_statistic,
            "raw_p": verification_p,
            "rank_biserial_G2R_minus_original_G2": rank_biserial(
                verification_differences
            ),
        })
    pd.DataFrame(omnibus_rows).to_csv(
        output / "gate_controls_with_G2R_friedman_tests.csv", index=False
    )
    pd.DataFrame(pairwise_rows).to_csv(
        output / "G2R_vs_controls_wilcoxon_holm.csv", index=False
    )
    pd.DataFrame(verification_rows).to_csv(
        output / "G2R_vs_original_G2_verification.csv", index=False
    )


def run(cfg: Config | None = None) -> None:
    cfg = Config() if cfg is None else cfg
    if cfg.smoke_test:
        cfg = replace(cfg, epochs=1)
    if not 0.0 < cfg.fixed_gate_lambda < 1.0:
        raise ValueError("fixed_gate_lambda must be between zero and one.")
    set_reproducibility(cfg.seed)
    configure_gpu()
    q = load_qgecaf_source(Path(cfg.qgecaf_source))
    manifest, class_names, raw = read_reference(cfg)
    existing_metrics = validate_existing_controls(cfg)
    output = Path(cfg.output)
    validate_protocol(cfg, output)
    root = Path(cfg.qgecaf_root)

    all_metric_rows: List[dict] = []
    quality_rows: List[dict] = []
    threshold_rows: List[dict] = []
    gate_value_frames: List[pd.DataFrame] = []

    # Read the completed controls and original G2. They are never overwritten.
    all_metric_rows.extend(existing_metrics.to_dict("records"))

    for fold in range(1, cfg.outer_folds + 1):
        print("\n" + "=" * 80)
        print(f"MATCHED GATE CONTROLS — OUTER FOLD {fold}/{cfg.outer_folds}")
        print("=" * 80)
        partitions, processors, _, preprocessing_sha = load_fold(
            cfg, fold, manifest, raw
        )
        transformed, quality_z, dimensions = {}, {}, None
        for partition_name, partition_frame in partitions.items():
            x, qz, dims = transform_partition(raw, processors, partition_frame)
            transformed[partition_name] = x
            quality_z[partition_name] = qz
            if dimensions is None:
                dimensions = dims
            elif dimensions != dims:
                raise RuntimeError(f"Fold {fold}: feature dimensions differ by partition.")
        assert dimensions is not None
        train_frame = partitions["inner_train"]
        validation_frame = partitions["inner_validation"]
        holdout_frame = partitions["outer_holdout"]
        train_idx = train_frame["image_index"].to_numpy(int)
        validation_idx = validation_frame["image_index"].to_numpy(int)
        holdout_idx = holdout_frame["image_index"].to_numpy(int)
        inner_scores = q.quality_score_from_standardized(quality_z["inner_train"])
        q33, q67 = np.percentile(
            inner_scores,
            [cfg.lower_quality_percentile, cfg.upper_quality_percentile],
        )
        threshold_rows.append({
            "fold": fold,
            "inner_train_images": len(inner_scores),
            "inner_train_q33": q33,
            "inner_train_q67": q67,
            "preprocessing_sha256": preprocessing_sha,
        })

        expected_keys = {
            record_key(patient, path)
            for patient, path in zip(holdout_frame["patient_id"], holdout_frame["path"])
        }

        # Reconstruct all existing quality-stratified rows from predictions so
        # the analysis does not depend on a truncated display or earlier table.
        for existing_variant in EXISTING_VARIANTS:
            prediction_path = existing_prediction_file(cfg, existing_variant, fold)
            existing_predictions = add_record_key(
                pd.read_csv(prediction_path),
                f"{existing_variant} fold {fold} predictions",
            )
            if set(existing_predictions["record_key"]) != expected_keys:
                raise RuntimeError(
                    f"Fold {fold}: {existing_variant} predictions do not match the holdout."
                )
            quality_rows.extend(quality_stratified_rows(
                fold, existing_variant, existing_predictions,
                q33, q67, class_names, cfg,
            ))
            gate_value_frames.append(gate_value_rows(
                fold, existing_variant, existing_predictions, q33, q67,
            ))

        for variant_index, variant in enumerate(TRAIN_VARIANTS):
            print(f"\n--- {variant} ---")
            variant_root = output / variant
            fold_dir = variant_root / "cross_validation" / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            completion_file = fold_dir / "FOLD_COMPLETE.json"
            prediction_file = fold_dir / "outer_holdout_predictions.csv"
            metrics_file = fold_dir / "fold_metrics.json"
            if cfg.resume and completion_file.is_file() and prediction_file.is_file() and metrics_file.is_file():
                completion = json.loads(completion_file.read_text(encoding="utf-8"))
                if completion.get("preprocessing_sha256") != preprocessing_sha:
                    raise RuntimeError(f"Fold {fold} {variant}: preprocessor changed.")
                saved = add_record_key(pd.read_csv(prediction_file), str(prediction_file))
                if set(saved["record_key"]) != expected_keys:
                    raise RuntimeError(f"Fold {fold} {variant}: saved holdout mismatch.")
                saved_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                all_metric_rows.append(saved_metrics)
                quality_rows.extend(quality_stratified_rows(
                    fold, variant, saved, q33, q67, class_names, cfg,
                ))
                gate_value_frames.append(gate_value_rows(
                    fold, variant, saved, q33, q67,
                ))
                print("Completed fold verified; skipping retraining.")
                continue

            fold_seed = cfg.seed + fold
            set_reproducibility(fold_seed)
            x_train = {name: value.copy() for name, value in transformed["inner_train"].items()}
            x_val = {name: value.copy() for name, value in transformed["inner_validation"].items()}
            x_holdout = {name: value.copy() for name, value in transformed["outer_holdout"].items()}
            # G2_matched_rerun deliberately retains the correct quality row for
            # every image in train, validation and outer holdout partitions.

            y_train = tf.keras.utils.to_categorical(raw["labels"][train_idx], cfg.num_classes)
            y_val = tf.keras.utils.to_categorical(raw["labels"][validation_idx], cfg.num_classes)
            tf.keras.backend.clear_session()
            model = build_gate_model(q, dimensions, cfg, variant)
            q.compile_model(model, raw["labels"][train_idx], cfg.num_classes, cfg)
            if fold == 1:
                q.calculate_model_complexity(model, variant_root / "model_complexity")
            timer = q.EpochTimer(fold_dir / "epoch_timing.csv")
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    str(fold_dir / "best_model.keras"), monitor="val_loss",
                    save_best_only=True, verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=cfg.early_stopping_patience,
                    restore_best_weights=True, verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3,
                    min_lr=1e-7, verbose=1,
                ),
                tf.keras.callbacks.CSVLogger(str(fold_dir / "epoch_log.csv")),
                timer,
            ]
            started = time.perf_counter()
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                callbacks=callbacks,
                verbose=1,
            )
            training_seconds = time.perf_counter() - started
            best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
            metrics = q.evaluate(
                model,
                x_holdout,
                raw["labels"][holdout_idx].astype(int),
                class_names,
                quality_z["outer_holdout"],
                raw["paths"][holdout_idx],
                fold_dir,
            )
            metrics.update({
                "variant": variant,
                "fold": fold,
                "seed": fold_seed,
                "best_epoch": best_epoch,
                "epochs_run": len(history.history["loss"]),
                "training_time_seconds": training_seconds,
                "mean_epoch_time_seconds": np.mean(
                    [row["epoch_time_seconds"] for row in timer.rows]
                ),
                "inner_train_images": len(train_idx),
                "inner_validation_images": len(validation_idx),
                "outer_holdout_images": len(holdout_idx),
                "inner_train_patients": train_frame["patient_id"].nunique(),
                "inner_validation_patients": validation_frame["patient_id"].nunique(),
                "outer_holdout_patients": holdout_frame["patient_id"].nunique(),
                "preprocessing_sha256": preprocessing_sha,
            })
            metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            predictions = pd.read_csv(fold_dir / "test_predictions_uncertainty.csv")
            predictions.insert(0, "outer_fold", fold)
            predictions.insert(1, "patient_id", holdout_frame["patient_id"].to_numpy())
            predictions["gate_input_quality_score"] = predictions["quality_score"]
            predictions.to_csv(prediction_file, index=False)
            completion_file.write_text(json.dumps({
                "variant": variant,
                "fold": fold,
                "preprocessing_sha256": preprocessing_sha,
                "outer_holdout_images": len(holdout_idx),
                "outer_holdout_patients": holdout_frame["patient_id"].nunique(),
            }, indent=2), encoding="utf-8")
            all_metric_rows.append(metrics)
            quality_rows.extend(quality_stratified_rows(
                fold, variant, predictions, q33, q67, class_names, cfg,
            ))
            gate_value_frames.append(gate_value_rows(
                fold, variant, predictions, q33, q67,
            ))
            del model, x_train, x_val, x_holdout
            gc.collect()
            tf.keras.backend.clear_session()

    metrics_frame = pd.DataFrame(all_metric_rows)
    metrics_frame["fold"] = metrics_frame["fold"].astype(int)
    metrics_frame = metrics_frame.sort_values(["variant", "fold"])
    for variant in ALL_VARIANTS:
        variant_rows = metrics_frame.loc[metrics_frame["variant"] == variant]
        if set(variant_rows["fold"]) != set(range(1, cfg.outer_folds + 1)):
            raise RuntimeError(f"Variant {variant} does not contain all ten folds.")
        variant_root = output / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        variant_rows.to_csv(variant_root / "fold_metrics.csv", index=False)
    metrics_frame.to_csv(output / "all_gate_controls_with_G2R_fold_metrics.csv", index=False)
    summary = summarize_metrics(metrics_frame)
    summary.to_csv(output / "gate_controls_with_G2R_mean_sd_ci95.csv", index=False)
    thresholds = pd.DataFrame(threshold_rows).sort_values("fold")
    thresholds.to_csv(output / "quality_thresholds_by_fold.csv", index=False)
    quality_frame = pd.DataFrame(quality_rows).sort_values(
        ["variant", "fold", "quality_group"]
    )
    quality_frame.to_csv(output / "quality_stratified_gate_controls.csv", index=False)
    summarize_quality_strata(quality_frame).to_csv(
        output / "quality_stratified_mean_sd_ci95.csv", index=False
    )
    gate_values = pd.concat(gate_value_frames, ignore_index=True)
    gate_values = gate_values.sort_values(["variant", "fold", "patient_id", "path"])
    gate_values.to_csv(output / "all_outer_holdout_gate_values.csv", index=False)
    pooled_gate_summary(gate_values).to_csv(
        output / "pooled_gate_distribution_summary.csv", index=False
    )
    gate_quality_correlations(gate_values, output)
    gate_group_tests(quality_frame, output)
    plot_gate_distributions(gate_values, output)
    statistical_comparisons(metrics_frame, output)
    (output / "RUN_COMPLETE.json").write_text(json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "folds": cfg.outer_folds,
        "variants": list(ALL_VARIANTS),
        "original_G2_replaced": False,
        "G2_matched_rerun_completed": True,
        "saved_preprocessing_refitted": False,
        "untouched_final_test_evaluated": False,
    }, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print("MATCHED G2 RERUN AND GATE-BEHAVIOUR ANALYSIS COMPLETE")
    print("=" * 80)
    print("Results:", output)
    print("Original G2 replaced: NO")
    print("G2_matched_rerun trained: YES (10 locked outer folds)")
    print("Saved preprocessing refitted: NO")
    print("Untouched final test evaluated: NO")
    print(summary)


if __name__ == "__main__":
    run(parse_args())
