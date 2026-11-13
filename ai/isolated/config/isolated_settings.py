"""
Configuration settings for Individual ISL Sign Recognition (10-sign MVP).
"""

from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 10 Target Classes
ISOLATED_CLASSES: List[str] = [
    "food",
    "hello",
    "help",
    "no",
    "please",
    "sorry",
    "stop",
    "thank you",
    "water",
    "yes",
]
NUM_CLASSES: int = len(ISOLATED_CLASSES)

CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(ISOLATED_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for idx, cls in enumerate(ISOLATED_CLASSES)}

# Landmark Dimensions (75 keypoints: 33 Pose, 21 Left Hand, 21 Right Hand)
LANDMARKS_PER_FRAME: int = 75
CHANNELS_PER_LANDMARK: int = 4  # [x, y, z, confidence]
SPATIAL_FEATURE_DIM: int = LANDMARKS_PER_FRAME * CHANNELS_PER_LANDMARK  # 300

# Sequence Dimensions
DEFAULT_SEQUENCE_LENGTH: int = 64  # Uniform resampled frame count for isolated classifier

# Model Storage Paths
ISOLATED_CHECKPOINT_DIR: Path = PROJECT_ROOT / "models" / "checkpoints" / "isolated"
ISOLATED_CHECKPOINT_PATH: Path = ISOLATED_CHECKPOINT_DIR / "isolated_10class_best.joblib"
ISOLATED_IMPROVED_CHECKPOINT_PATH: Path = ISOLATED_CHECKPOINT_DIR / "isolated_10class_improved.joblib"
ISOLATED_BASELINE_CHECKPOINT_PATH: Path = ISOLATED_CHECKPOINT_DIR / "f022_isolated_10class_baseline.joblib"
ISOLATED_LABELS_PATH: Path = PROJECT_ROOT / "models" / "labels" / "isolated_labels.json"

# 20-Class Side-by-Side Model and Labels Paths (Rollback-safe)
ISOLATED_20CLASS_CHECKPOINT_PATH: Path = ISOLATED_CHECKPOINT_DIR / "isolated_20class_best.joblib"
ISOLATED_20CLASS_LABELS_PATH: Path = PROJECT_ROOT / "models" / "labels" / "isolated_20class_labels.json"
ISOLATED_20CLASS_METADATA_PATH: Path = ISOLATED_CHECKPOINT_DIR / "model_metadata_20class.json"

# Dataset Paths
SPLITS_DIR: Path = PROJECT_ROOT / "datasets" / "splits"
LANDMARKS_DIR: Path = PROJECT_ROOT / "datasets" / "landmarks"
FEATURES_DIR: Path = PROJECT_ROOT / "datasets" / "features"
MANIFEST_PATH: Path = PROJECT_ROOT / "datasets" / "manifests" / "isl_10class_manifest.csv"
MANIFEST_20CLASS_PATH: Path = PROJECT_ROOT / "datasets" / "manifests" / "isl_20class_manifest.csv"
