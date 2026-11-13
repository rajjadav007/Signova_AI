"""
Signova AI - Pose Input Preparation Module (F-014).
Preprocesses landmark sequences for the iSign-T5 pose-to-text model (t5-small-motion).
Source of truth: official implementation at D:/iSign-t5-pose-to-text/inference/pose_utils.py.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

def resolve_device(requested_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve device to torch.device; falls back to CPU if CUDA unavailable."""
    if isinstance(requested_device, torch.device):
        return requested_device
    if isinstance(requested_device, str) and requested_device.lower().startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested_device)
    return torch.device("cpu")

logger = logging.getLogger("signova.ai.preprocessing")

# Landmark component counts according to official iSign-T5 specification
POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
TOTAL_LANDMARKS = POSE_LANDMARK_COUNT + 2 * HAND_LANDMARK_COUNT  # 75
CHANNELS_PER_LANDMARK = 4  # (x, y, z, confidence)
BASE_SPATIAL_DIM = TOTAL_LANDMARKS * CHANNELS_PER_LANDMARK  # 300
MOTION_DIM = BASE_SPATIAL_DIM  # 300
TOTAL_FEATURE_DIM = BASE_SPATIAL_DIM + MOTION_DIM  # 600
DEFAULT_MAX_POSE_LENGTH = 500

DEFAULT_COMPONENTS = (
    "POSE_LANDMARKS",
    "LEFT_HAND_LANDMARKS",
    "RIGHT_HAND_LANDMARKS",
)


@dataclass(frozen=True)
class PreprocessedPose:
    """
    Standard preprocessed pose representation ready for iSign PoseT5 encoder.
    """
    pose_tensor: torch.Tensor       # Shape: (1, max_length, 300) for simple model, dtype: torch.float32
    pose_length: torch.Tensor       # Shape: (1,), dtype: torch.long
    features: np.ndarray            # Shape: (max_length, 300) for simple model, dtype: np.float32
    actual_length: int              # Integer frame count before padding (bounded to max_length)
    device: torch.device
    metadata: Optional[Dict[str, Any]] = None


def normalize_pose(pose: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Official temporal normalization across frames.
    Preserves exact formula from D:/iSign-t5-pose-to-text/inference/pose_utils.py:
        mean = np.nanmean(pose, axis=0, keepdims=True)
        std = np.nanstd(pose, axis=0, keepdims=True)
        (pose - mean) / np.where(std == 0, 1.0, std)
    """
    if pose.size == 0:
        return pose

    if method == "zscore":
        mean = np.nanmean(pose, axis=0, keepdims=True)
        std = np.nanstd(pose, axis=0, keepdims=True)
        return np.nan_to_num(
            (pose - mean) / np.where(std == 0, 1.0, std)
        )

    if method == "minmax":
        min_val = np.nanmin(pose, axis=0, keepdims=True)
        max_val = np.nanmax(pose, axis=0, keepdims=True)
        return np.nan_to_num(
            (pose - min_val) / np.where(max_val == min_val, 1.0, max_val - min_val)
        )

    raise ValueError(f"Unknown normalization method: {method}")


def add_motion_features(pose: np.ndarray) -> np.ndarray:
    """
    Official motion features calculation.
    velocity[t] = pose[t] - pose[t - 1]
    velocity[0] is zero.
    Returns:
        np.concatenate([pose, velocity], axis=1) -> shape (frames, 600)
    """
    if pose.size == 0:
        return pose

    velocity = np.zeros_like(pose, dtype=np.float32)
    velocity[1:] = pose[1:] - pose[:-1]

    return np.concatenate([pose, velocity], axis=1)


def pad_or_truncate_pose(
    pose: np.ndarray,
    max_length: int = DEFAULT_MAX_POSE_LENGTH,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Official padding or truncation to max_length frames.
    Returns:
        padded: (max_length, feature_dim)
        length: actual frame count bounded to max_length
    """
    if pose.size == 0:
        width = pose.shape[1] if pose.ndim > 1 else 0
        return np.full((max_length, width), pad_value, dtype=np.float32), 0

    length = min(pose.shape[0], max_length)
    padded = np.full((max_length, pose.shape[1]), pad_value, dtype=pose.dtype)
    padded[:length] = pose[:length]

    return padded, length


def load_pose_file(
    pose_path: Union[str, Path],
    components: Optional[Sequence[str]] = DEFAULT_COMPONENTS,
    include_confidence: bool = True,
) -> np.ndarray:
    """
    Official loader for .pose binary or JSON keypoint file.
    Returns array of shape (frames, 300).
    """
    path_obj = Path(pose_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Pose file not found at: {path_obj}")

    with open(path_obj, "rb") as f:
        raw = f.read()

    if not raw or len(raw.strip()) == 0:
        raise ValueError(f"Pose file at {path_obj} is empty.")

    # 1. JSON payload handling
    if raw.lstrip().startswith((b"{", b"[")):
        try:
            payload = json.loads(raw.decode("utf-8"))
            frames = (
                payload.get("data", [])
                if isinstance(payload, dict)
                else payload
            )
            rows = [
                np.asarray(frame.get("keypoints", []), dtype=np.float32).reshape(-1)
                for frame in frames
            ]
            if not rows:
                return np.array([])
            return np.asarray(rows, dtype=np.float32)
        except Exception as exc:
            logger.error("Failed to parse JSON pose file %s: %s", path_obj, exc)
            raise ValueError(f"Invalid JSON pose data in {path_obj}: {exc}") from exc

    # 2. pose-format binary parsing
    try:
        from pose_format import Pose
        pose = Pose.read(raw)
    except Exception as exc:
        logger.error("Failed to load binary pose file %s: %s", path_obj, exc)
        raise ValueError(f"Failed to parse binary .pose file {path_obj}: {exc}") from exc

    data = np.asarray(pose.body.data, dtype=np.float32)
    confidence = np.asarray(pose.body.confidence, dtype=np.float32)

    if data.ndim != 4 or data.shape[1] == 0:
        logger.warning("Unexpected pose shape in %s: %s", path_obj, data.shape)
        return np.array([])

    data = data[:, 0]
    confidence = confidence[:, 0] if confidence.ndim == 3 else None

    wanted = set(components) if components is not None else None
    selected = []
    offset = 0

    for component in pose.header.components:
        count = len(component.points)
        if wanted is None or component.name in wanted:
            comp_data = data[:, offset:offset + count, :]
            if include_confidence and confidence is not None:
                comp_data = np.concatenate(
                    [comp_data, confidence[:, offset:offset + count, None]],
                    axis=2,
                )
            selected.append(comp_data)
        offset += count

    if not selected:
        logger.warning("No requested components found in %s", path_obj)
        return np.array([])

    return np.concatenate(selected, axis=1).reshape(data.shape[0], -1)


def assemble_frame_landmarks(
    pose: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    left_hand: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    right_hand: Optional[Union[np.ndarray, Sequence[Any]]] = None,
) -> np.ndarray:
    """
    Assemble a single frame of landmarks into the official 300-dim vector:
    - 33 Pose landmarks (indices 0..32, 132 features)
    - 21 Left hand landmarks (indices 33..53, 84 features)
    - 21 Right hand landmarks (indices 54..74, 84 features)
    Total: 75 landmarks * 4 channels = 300 features.
    Missing components are filled with zeros.
    """
    frame_vec = np.zeros(BASE_SPATIAL_DIM, dtype=np.float32)

    def _convert_component(comp: Any, expected_count: int, name: str) -> np.ndarray:
        if comp is None:
            return np.zeros((expected_count, 4), dtype=np.float32)

        if isinstance(comp, np.ndarray):
            arr = comp.astype(np.float32)
            if arr.ndim == 1:
                if arr.size == expected_count * 4:
                    return arr.reshape(expected_count, 4)
                if arr.size == expected_count * 3:
                    # Missing confidence -> append 1.0 default
                    arr = arr.reshape(expected_count, 3)
                    conf = np.ones((expected_count, 1), dtype=np.float32)
                    return np.concatenate([arr, conf], axis=1)
                raise ValueError(
                    f"Component '{name}' has invalid 1D length {arr.size}; expected {expected_count * 4}."
                )
            if arr.ndim == 2:
                if arr.shape[0] != expected_count:
                    raise ValueError(
                        f"Component '{name}' has {arr.shape[0]} landmarks; expected {expected_count}."
                    )
                if arr.shape[1] == 4:
                    return arr
                if arr.shape[1] == 3:
                    conf = np.ones((expected_count, 1), dtype=np.float32)
                    return np.concatenate([arr, conf], axis=1)
                raise ValueError(
                    f"Component '{name}' has {arr.shape[1]} channels; expected 3 or 4."
                )

        # Sequence of landmark objects (e.g. MediaPipe NormalizedLandmark with x, y, z, visibility)
        if isinstance(comp, (list, tuple)):
            if len(comp) != expected_count and len(comp) != 0:
                raise ValueError(
                    f"Component '{name}' has {len(comp)} landmarks; expected {expected_count}."
                )
            if len(comp) == 0:
                return np.zeros((expected_count, 4), dtype=np.float32)

            res = np.zeros((expected_count, 4), dtype=np.float32)
            for idx, lm in enumerate(comp[:expected_count]):
                if hasattr(lm, "x"):
                    x = getattr(lm, "x", 0.0)
                    y = getattr(lm, "y", 0.0)
                    z = getattr(lm, "z", 0.0)
                    vis = getattr(lm, "visibility", getattr(lm, "presence", 1.0))
                    res[idx] = [x, y, z, vis]
                elif isinstance(lm, (list, tuple, np.ndarray)):
                    lm_arr = np.asarray(lm, dtype=np.float32)
                    if len(lm_arr) >= 4:
                        res[idx] = lm_arr[:4]
                    elif len(lm_arr) == 3:
                        res[idx] = [lm_arr[0], lm_arr[1], lm_arr[2], 1.0]
                    else:
                        raise ValueError(f"Landmark in '{name}' must have at least 3 coordinates.")
            return res

        raise TypeError(f"Unsupported data type for component '{name}': {type(comp)}")

    pose_arr = _convert_component(pose, POSE_LANDMARK_COUNT, "pose")
    left_arr = _convert_component(left_hand, HAND_LANDMARK_COUNT, "left_hand")
    right_arr = _convert_component(right_hand, HAND_LANDMARK_COUNT, "right_hand")

    # Combine in exact official order: [pose (33), left_hand (21), right_hand (21)]
    combined = np.concatenate([pose_arr, left_arr, right_arr], axis=0)  # (75, 4)
    return combined.reshape(-1)  # (300,)


def _get_component_from_dict(data: Dict[str, Any], *keys: str) -> Any:
    """Safely retrieve component from dict without boolean evaluation of numpy arrays."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def to_spatial_matrix(
    pose_input: Union[np.ndarray, Sequence[Any], Dict[str, Any], str, Path],
) -> np.ndarray:
    """
    Convert diverse input structures into raw spatial matrix of shape (frames, 300).
    Validates dimensions and applies official component ordering.
    """
    # 1. File path (.pose, .json, or video formats)
    if isinstance(pose_input, (str, Path)):
        p = Path(pose_input)
        if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            from ai.pose_extraction.pose_generator import video_to_pose_file
            pose_path, _, _ = video_to_pose_file(p)
            raw_mat = load_pose_file(pose_path)
            return np.nan_to_num(raw_mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        raw_mat = load_pose_file(p)
        if raw_mat.size == 0:
            raise ValueError(f"No valid pose frames extracted from file: {pose_input}")
        return np.nan_to_num(raw_mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 1b. Direct pose_format Pose object
    try:
        from pose_format import Pose
        if isinstance(pose_input, Pose):
            import io
            buf = io.BytesIO()
            pose_input.write(buf)
            buf.seek(0)
            data = np.asarray(pose_input.body.data, dtype=np.float32)[:, 0]
            confidence = np.asarray(pose_input.body.confidence, dtype=np.float32)[:, 0]
            wanted = set(DEFAULT_COMPONENTS)
            selected = []
            offset = 0
            for component in pose_input.header.components:
                count = len(component.points)
                if component.name in wanted:
                    comp_data = data[:, offset:offset + count, :]
                    comp_data = np.concatenate([comp_data, confidence[:, offset:offset + count, None]], axis=2)
                    selected.append(comp_data)
                offset += count
            raw_mat = np.concatenate(selected, axis=1).reshape(data.shape[0], -1)
            return np.nan_to_num(raw_mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except ImportError:
        pass

    # 2. NumPy array
    if isinstance(pose_input, np.ndarray):
        arr = np.nan_to_num(pose_input.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size == 0 or arr.shape[0] == 0:
            raise ValueError("Input pose array is empty (0 frames).")

        # Case 2a: (frames, 300)
        if arr.ndim == 2 and arr.shape[1] == BASE_SPATIAL_DIM:
            return arr

        # Case 2b: (frames, 75, 4)
        if arr.ndim == 3 and arr.shape[1] == TOTAL_LANDMARKS and arr.shape[2] == CHANNELS_PER_LANDMARK:
            return arr.reshape(arr.shape[0], BASE_SPATIAL_DIM)

        # Case 2c: (frames, 75, 3) -> append confidence 1.0
        if arr.ndim == 3 and arr.shape[1] == TOTAL_LANDMARKS and arr.shape[2] == 3:
            conf = np.ones((arr.shape[0], TOTAL_LANDMARKS, 1), dtype=np.float32)
            with_conf = np.concatenate([arr, conf], axis=2)
            return with_conf.reshape(arr.shape[0], BASE_SPATIAL_DIM)

        # Case 2d: Single frame 1D (300,)
        if arr.ndim == 1 and arr.size == BASE_SPATIAL_DIM:
            return arr.reshape(1, BASE_SPATIAL_DIM)

        # Case 2e: Single frame 2D (75, 4)
        if arr.ndim == 2 and arr.shape[0] == TOTAL_LANDMARKS and arr.shape[1] == CHANNELS_PER_LANDMARK:
            return arr.reshape(1, BASE_SPATIAL_DIM)

        raise ValueError(
            f"Unsupported pose array shape: {arr.shape}. "
            f"Expected (frames, 300) or (frames, 75, 4)."
        )

    # 3. Dict with sequence keys: {"pose": (F, 33, 4), "left_hand": (F, 21, 4), "right_hand": ...}
    if isinstance(pose_input, dict):
        pose_comp = _get_component_from_dict(
            pose_input, "pose", "POSE_LANDMARKS", "pose_landmarks"
        )
        left_comp = _get_component_from_dict(
            pose_input, "left_hand", "LEFT_HAND_LANDMARKS", "left_hand_landmarks"
        )
        right_comp = _get_component_from_dict(
            pose_input, "right_hand", "RIGHT_HAND_LANDMARKS", "right_hand_landmarks"
        )

        # Determine number of frames
        def _get_len(c: Any) -> int:
            if c is None:
                return 0
            if isinstance(c, (np.ndarray, list, tuple)):
                return len(c)
            return 0

        lengths = [_get_len(pose_comp), _get_len(left_comp), _get_len(right_comp)]
        num_frames = max(lengths)
        if num_frames == 0:
            raise ValueError("Structured pose dictionary contains no frames.")

        frames = []
        for f_idx in range(num_frames):
            p = pose_comp[f_idx] if pose_comp is not None and f_idx < len(pose_comp) else None
            lh = left_comp[f_idx] if left_comp is not None and f_idx < len(left_comp) else None
            rh = right_comp[f_idx] if right_comp is not None and f_idx < len(right_comp) else None
            frame_vec = assemble_frame_landmarks(pose=p, left_hand=lh, right_hand=rh)
            frames.append(frame_vec)
        return np.asarray(frames, dtype=np.float32)

    # 4. List of frame dicts: [{"pose": ..., "left_hand": ..., "right_hand": ...}, ...]
    if isinstance(pose_input, (list, tuple)):
        if len(pose_input) == 0:
            raise ValueError("Input pose sequence list is empty (0 frames).")

        frames = []
        for f_idx, item in enumerate(pose_input):
            if isinstance(item, dict):
                p = _get_component_from_dict(item, "pose", "POSE_LANDMARKS", "pose_landmarks")
                lh = _get_component_from_dict(item, "left_hand", "LEFT_HAND_LANDMARKS", "left_hand_landmarks")
                rh = _get_component_from_dict(item, "right_hand", "RIGHT_HAND_LANDMARKS", "right_hand_landmarks")
                frame_vec = assemble_frame_landmarks(pose=p, left_hand=lh, right_hand=rh)
                frames.append(frame_vec)
            elif isinstance(item, np.ndarray):
                if item.size == BASE_SPATIAL_DIM:
                    frames.append(item.reshape(-1))
                elif item.size == TOTAL_LANDMARKS * CHANNELS_PER_LANDMARK:
                    frames.append(item.reshape(-1))
                else:
                    raise ValueError(
                        f"Frame {f_idx} has invalid array size {item.size}; expected {BASE_SPATIAL_DIM}."
                    )
            elif isinstance(item, (list, tuple)):
                if len(item) == BASE_SPATIAL_DIM:
                    frames.append(np.asarray(item, dtype=np.float32))
                else:
                    raise ValueError(
                        f"Frame {f_idx} has invalid list length {len(item)}; expected {BASE_SPATIAL_DIM}."
                    )
            else:
                raise TypeError(f"Frame {f_idx} has unsupported type: {type(item)}")

        return np.asarray(frames, dtype=np.float32)

    raise TypeError(f"Unsupported pose input type: {type(pose_input)}")


def preprocess_pose(
    pose_input: Union[np.ndarray, Sequence[Any], Dict[str, Any], str, Path],
    max_length: int = DEFAULT_MAX_POSE_LENGTH,
    normalization: str = "zscore",
    use_motion: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_pose_file: bool = True,
) -> PreprocessedPose:
    """
    Main entrypoint for F-014 / F-016: Pose Input Preparation.

    Pipeline:
    1. Extract and assemble spatial features -> (frames, 300).
       For video files: extracts landmarks and serializes to official .pose file when
       use_pose_file is True, then loads via official load_pose_file().
    2. Validate non-empty sequence and clean NaNs/Infs.
    3. Apply official temporal Z-score normalization across frames.
    4. Compute motion features (velocity) and append -> (frames, 600) only when use_motion is True.
       For simple model (use_motion=False), preserves strictly 300 features.
    5. Pad or truncate to max_length (500) -> (500, 300).
    6. Convert to PyTorch tensors placed on the specified device:
       - pose_tensor: (1, 500, 300), torch.float32
       - pose_length: (1,), torch.long
    """
    # 1. Spatial assembly
    extracted_metadata = None
    if isinstance(pose_input, (str, Path)):
        p = Path(pose_input)
        if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            if use_pose_file:
                from ai.pose_extraction.pose_generator import video_to_pose_file
                pose_file_path, _, extracted_metadata = video_to_pose_file(p)
                spatial_mat = load_pose_file(pose_file_path)
            else:
                from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
                extractor = MediaPipeLandmarkExtractor()
                try:
                    spatial_mat, extracted_metadata = extractor.extract_isign_video(p)
                finally:
                    extractor.close()
        else:
            spatial_mat = to_spatial_matrix(pose_input)
    else:
        spatial_mat = to_spatial_matrix(pose_input)

    final_metadata = metadata if metadata is not None else extracted_metadata

    # 2. Validation
    if spatial_mat.size == 0 or spatial_mat.shape[0] == 0:
        raise ValueError("Pose sequence contains no valid frames.")

    # Clean NaNs and infinite values
    cleaned = np.nan_to_num(spatial_mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 3. Temporal normalization
    normalized = normalize_pose(cleaned, method=normalization)

    # 4. Motion features
    if use_motion:
        features = add_motion_features(normalized)
    else:
        features = normalized

    # 5. Temporal padding or truncation
    padded, actual_len = pad_or_truncate_pose(features, max_length=max_length)

    # 6. Target device resolution
    target_device = resolve_device(device if isinstance(device, str) else None) if not isinstance(device, torch.device) else device

    # 7. Convert to tensors
    pose_tensor = torch.from_numpy(padded).unsqueeze(0).float().to(target_device)
    pose_length_tensor = torch.tensor([actual_len], dtype=torch.long, device=target_device)

    return PreprocessedPose(
        pose_tensor=pose_tensor,
        pose_length=pose_length_tensor,
        features=padded,
        actual_length=actual_len,
        device=target_device,
        metadata=final_metadata,
    )
