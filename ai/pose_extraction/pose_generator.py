"""
Signova AI - Official iSign .pose File Generator & Validator (F-016).
Constructs, writes, and validates binary .pose files conforming strictly to the
official iSign dataset specification and pose-format library standard.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from pose_format import Pose, PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeaderComponent, PoseHeaderDimensions

from ai.config.settings import (
    ISIGN_CHANNELS,
    ISIGN_HAND_COUNT,
    ISIGN_POSE_COUNT,
    ISIGN_SPATIAL_DIM,
    ISIGN_TOTAL_LANDMARKS,
)

logger = logging.getLogger("signova.ai.pose_generator")

# ---------------------------------------------------------------------------
# Official MediaPipe & iSign Topology Specifications
# ---------------------------------------------------------------------------
POSE_LANDMARK_NAMES: List[str] = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

HAND_LANDMARK_NAMES: List[str] = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def build_isign_pose_header(
    width: int = 1920,
    height: int = 1080,
    depth: int = 1000,
) -> PoseHeader:
    """
    Construct the official PoseHeader containing the 3 iSign components:
    1. POSE_LANDMARKS (33 points)
    2. LEFT_HAND_LANDMARKS (21 points)
    3. RIGHT_HAND_LANDMARKS (21 points)
    Total points: 75. Point format: 'XYZC'.
    """
    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)
    components = [
        PoseHeaderComponent(
            name="POSE_LANDMARKS",
            points=POSE_LANDMARK_NAMES,
            limbs=POSE_CONNECTIONS,
            colors=[(255, 0, 0)],
            point_format="XYZC",
        ),
        PoseHeaderComponent(
            name="LEFT_HAND_LANDMARKS",
            points=HAND_LANDMARK_NAMES,
            limbs=HAND_CONNECTIONS,
            colors=[(0, 255, 0)],
            point_format="XYZC",
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND_LANDMARKS",
            points=HAND_LANDMARK_NAMES,
            limbs=HAND_CONNECTIONS,
            colors=[(0, 0, 255)],
            point_format="XYZC",
        ),
    ]
    return PoseHeader(version=0.1, dimensions=dimensions, components=components)


def landmarks_to_pose(
    landmarks_spatial: np.ndarray,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    depth: int = 1000,
) -> Pose:
    """
    Convert an extracted spatial landmark sequence into an official pose-format Pose object.
    Input:
        landmarks_spatial: np.ndarray of shape (frames, 300) or (frames, 75, 4)
    Returns:
        Pose object with official PoseHeader and NumPyPoseBody.
    """
    arr = np.nan_to_num(np.asarray(landmarks_spatial, dtype=np.float32), nan=0.0)
    if arr.ndim == 2:
        if arr.shape[1] != ISIGN_SPATIAL_DIM:
            raise ValueError(
                f"Expected 2D landmarks with shape (frames, {ISIGN_SPATIAL_DIM}), got {arr.shape}"
            )
        frames = arr.shape[0]
        kps_75_4 = arr.reshape(frames, ISIGN_TOTAL_LANDMARKS, ISIGN_CHANNELS)
    elif arr.ndim == 3:
        if arr.shape[1] != ISIGN_TOTAL_LANDMARKS or arr.shape[2] != ISIGN_CHANNELS:
            raise ValueError(
                f"Expected 3D landmarks with shape (frames, {ISIGN_TOTAL_LANDMARKS}, {ISIGN_CHANNELS}), got {arr.shape}"
            )
        frames = arr.shape[0]
        kps_75_4 = arr
    else:
        raise ValueError(f"Unsupported landmarks shape: {arr.shape}")

    # Extract coordinates (x, y, z) -> shape (frames, 1, 75, 3)
    coords = np.expand_dims(kps_75_4[:, :, :3], axis=1)
    # Extract confidence channel -> shape (frames, 1, 75)
    confidence = np.expand_dims(kps_75_4[:, :, 3], axis=1)

    header = build_isign_pose_header(width=width, height=height, depth=depth)
    body = NumPyPoseBody(fps=float(fps), data=coords, confidence=confidence)

    return Pose(header=header, body=body)


def save_pose_file(
    pose: Pose,
    output_path: Union[str, Path],
) -> Path:
    """
    Serialize a Pose object into a binary .pose file using official pose-format serializer.
    """
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pose.write(f)
    logger.info("Saved official .pose file (%d bytes) to: %s", target.stat().st_size, target)
    return target


def video_to_pose_file(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    extractor: Optional[Any] = None,
    max_frames: Optional[int] = None,
    hand_missing_strategy: str = "zero",
    max_forward_fill_frames: int = 15,
) -> Tuple[Path, Pose, Dict[str, Any]]:
    """
    Extract landmarks from a video file using MediaPipe and save directly as an official .pose file.
    Returns:
        (saved_pose_path, pose_object, video_metadata)
    """
    from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor

    video_p = Path(video_path).resolve()
    if not video_p.exists():
        raise FileNotFoundError(f"Video file not found: {video_p}")

    ext = extractor or MediaPipeLandmarkExtractor(engine="modern")
    should_close = extractor is None

    try:
        raw_kps, metadata = ext.extract_isign_video(
            video_p,
            max_frames=max_frames,
            hand_missing_strategy=hand_missing_strategy,
            max_forward_fill_frames=max_forward_fill_frames,
        )
    finally:
        if should_close:
            ext.close()

    # Extract resolution and fps from metadata
    res_str = metadata.get("resolution", "1920x1080")
    try:
        w_str, h_str = res_str.split("x")
        w, h = int(w_str), int(h_str)
    except Exception:
        w, h = 1920, 1080

    fps = float(metadata.get("fps", 30.0))
    if fps <= 0.0 or np.isnan(fps):
        fps = 30.0

    pose = landmarks_to_pose(
        landmarks_spatial=raw_kps,
        width=w,
        height=h,
        fps=fps,
    )

    if output_path is None:
        # Default destination: datasets/poses/<video_stem>.pose (kept outside Git)
        poses_dir = video_p.parent.parent / "poses"
        target_path = poses_dir / f"{video_p.stem}.pose"
    else:
        target_path = Path(output_path).resolve()

    saved_path = save_pose_file(pose, target_path)
    metadata["pose_file_path"] = str(saved_path)
    metadata["pose_file_bytes"] = saved_path.stat().st_size

    return saved_path, pose, metadata


def validate_pose_file(
    pose_path: Union[str, Path],
    expected_landmarks: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Reopen and validate a .pose file using the official pose-format parser.
    Verifies:
      - Header version and metadata
      - Frame count
      - Landmark topology and ordering
      - Coordinate values
      - Confidence values
      - Missing-landmark representation
      - Numerical equality against original extracted landmarks (if provided)
    """
    path_obj = Path(pose_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f".pose file not found: {path_obj}")

    file_bytes = path_obj.stat().st_size
    with open(path_obj, "rb") as f:
        raw = f.read()

    # 1. Parse via official pose_format.Pose.read
    pose = Pose.read(raw)

    header = pose.header
    body = pose.body
    data = np.asarray(body.data, dtype=np.float32)
    confidence = np.asarray(body.confidence, dtype=np.float32)

    # 2. Topology verification
    components = [c.name for c in header.components]
    expected_components = ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
    topology_valid = components == expected_components

    points_per_comp = {c.name: len(c.points) for c in header.components}
    expected_points = {"POSE_LANDMARKS": 33, "LEFT_HAND_LANDMARKS": 21, "RIGHT_HAND_LANDMARKS": 21}
    points_valid = points_per_comp == expected_points

    total_points = sum(len(c.points) for c in header.components)

    # 3. Shape verification
    frames = data.shape[0]
    people = data.shape[1] if data.ndim == 4 else 0
    dims = data.shape[3] if data.ndim == 4 else 0
    shape_valid = (data.ndim == 4 and people == 1 and total_points == 75 and dims == 3)

    # 4. Confidence shape verification
    conf_valid = (confidence.ndim == 3 and confidence.shape == (frames, people, total_points))

    # 5. Missing landmark representation check
    # In pose-format, missing landmarks have confidence 0.0
    zero_conf_count = int(np.sum(confidence == 0.0))
    zero_conf_pct = round(float(zero_conf_count / max(confidence.size, 1) * 100.0), 2)

    # 6. Official load_pose_file verification
    from ai.preprocessing.pose_preprocessor import load_pose_file
    loaded_features = load_pose_file(path_obj)
    official_load_success = loaded_features.ndim == 2 and loaded_features.shape == (frames, 300)

    # 7. Roundtrip numerical equality against expected landmarks
    exact_roundtrip_match = None
    max_roundtrip_diff = None
    if expected_landmarks is not None:
        exp_arr = np.nan_to_num(np.asarray(expected_landmarks, dtype=np.float32), nan=0.0).reshape(frames, 300)
        max_roundtrip_diff = float(np.max(np.abs(exp_arr - loaded_features)))
        exact_roundtrip_match = bool(max_roundtrip_diff == 0.0)

    is_valid = bool(
        topology_valid
        and points_valid
        and shape_valid
        and conf_valid
        and official_load_success
        and (exact_roundtrip_match is None or exact_roundtrip_match)
    )

    return {
        "is_valid": is_valid,
        "pose_path": str(path_obj),
        "file_size_bytes": file_bytes,
        "header_version": round(float(header.version), 3),
        "dimensions": {
            "width": header.dimensions.width,
            "height": header.dimensions.height,
            "depth": header.dimensions.depth,
        },
        "fps": float(body.fps),
        "frame_count": frames,
        "people_count": people,
        "total_landmarks": total_points,
        "components": components,
        "points_per_component": points_per_comp,
        "topology_valid": topology_valid,
        "points_valid": points_valid,
        "shape_valid": shape_valid,
        "confidence_valid": conf_valid,
        "zero_confidence_count": zero_conf_count,
        "zero_confidence_percentage": zero_conf_pct,
        "official_load_success": official_load_success,
        "official_loaded_shape": list(loaded_features.shape),
        "max_roundtrip_difference": max_roundtrip_diff,
        "exact_roundtrip_match": exact_roundtrip_match,
    }
