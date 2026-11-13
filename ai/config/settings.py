"""
Signova AI - AI Pipeline Configuration and Constants.
Defines MediaPipe landmark specifications, tensor dimensions, and default model paths.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Set

AI_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_DIR.parent

# Video and Media Formats
SUPPORTED_VIDEO_FORMATS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Landmark Dimensions
NUM_KEYPOINTS: int = 94
KEYPOINT_COORDS: int = 4  # [x, y, z, visibility]
FEATURE_DIM: int = NUM_KEYPOINTS * 3  # 94 * 3 = 282
TARGET_FRAMES: int = 30

# MediaPipe Holistic Mapping
POSE_INDICES: List[int] = [11, 12, 13, 14, 23, 24]  # 6 points
HAND_LANDMARK_COUNT: int = 21  # 21 per hand

# iSign Continuous SLT Specifications (F-014, F-015)
ISIGN_POSE_COUNT: int = 33
ISIGN_HAND_COUNT: int = 21
ISIGN_TOTAL_LANDMARKS: int = 75  # 33 + 21 + 21
ISIGN_CHANNELS: int = 4          # [x, y, z, visibility/confidence]
ISIGN_SPATIAL_DIM: int = 300     # 75 * 4
ISIGN_MOTION_DIM: int = 600      # 300 + 300

FACE_LIPS: List[int] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 95, 88, 178, 87, 14, 78, 308,
]
FACE_EYES: List[int] = [
    33, 133, 160, 158, 153, 144, 145, 246,
    362, 263, 387, 385, 380, 373, 374, 249,
]
FACE_EYEBROWS: List[int] = [
    107, 66, 105, 63, 70,
    336, 296, 334, 293, 300,
]
FACE_INDICES: List[int] = FACE_LIPS + FACE_EYES + FACE_EYEBROWS

GROUP_SLICES: Dict[str, slice] = {
    "pose": slice(0, 6),
    "left_hand": slice(6, 27),
    "right_hand": slice(27, 48),
    "lips": slice(48, 68),
    "eyes": slice(68, 84),
    "eyebrows": slice(84, 94),
}

# Quality and Confidence Thresholds
MIN_HAND_PRESENCE_RATIO: float = 0.35
MIN_DYNAMIC_MOTION_ENERGY: float = 0.012
CONFIDENCE_RECOGNIZED_THRESHOLD: float = 0.65
CONFIDENCE_LOW_THRESHOLD: float = 0.45

# Default Model Paths
DEFAULT_MEDIAPIPE_TASK_PATH: Path = (
    Path(os.environ["MEDIAPIPE_MODEL_PATH"]).resolve()
    if "MEDIAPIPE_MODEL_PATH" in os.environ
    else PROJECT_ROOT / "models" / "assets" / "holistic_landmarker.task"
)
DEFAULT_POSE_TASK_PATH: Path = (
    Path(os.environ["MEDIAPIPE_POSE_TASK_PATH"]).resolve()
    if "MEDIAPIPE_POSE_TASK_PATH" in os.environ
    else PROJECT_ROOT / "models" / "assets" / "pose_landmarker_full.task"
)
DEFAULT_HAND_TASK_PATH: Path = (
    Path(os.environ["MEDIAPIPE_HAND_TASK_PATH"]).resolve()
    if "MEDIAPIPE_HAND_TASK_PATH" in os.environ
    else PROJECT_ROOT / "models" / "assets" / "hand_landmarker.task"
)

DEFAULT_ISOLATED_CHECKPOINT_PATH: Path = (
    PROJECT_ROOT / "models" / "checkpoints" / "isolated_model_best.pt"
)
DEFAULT_ISOLATED_LABELS_PATH: Path = (
    PROJECT_ROOT / "models" / "labels" / "isolated_labels.json"
)
DEFAULT_VSRS_MODEL_PATH: Path = (
    PROJECT_ROOT / "models" / "checkpoints" / "vsrs" / "model.joblib"
)
