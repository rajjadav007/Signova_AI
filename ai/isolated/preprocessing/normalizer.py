"""
Preprocessing and normalization module for Individual ISL Sign Recognition.
Provides spatial coordinate normalization and temporal resampling for variable-length sequences.
"""

from typing import Optional, Tuple, Union
import numpy as np


def spatial_normalize(landmarks_75x4: np.ndarray) -> np.ndarray:
    """
    Spatially center coordinates on upper-torso mid-shoulder anchor and scale
    by clip-level median shoulder width, preserving relative hand-body geometry.
    Strictly preserves zeros for undetected landmarks.
    
    Input: (T, 75, 4) where channels are [x, y, z, confidence].
    Output: (T, 75, 4) normalized coordinates.
    """
    T = landmarks_75x4.shape[0]
    if T == 0:
        return np.zeros_like(landmarks_75x4, dtype=np.float32)

    normed = np.zeros_like(landmarks_75x4, dtype=np.float32)

    # 1. Compute robust clip-level median shoulder width from confident frames
    shoulder_widths = []
    for t in range(T):
        ls = landmarks_75x4[t, 11]  # Left shoulder
        rs = landmarks_75x4[t, 12]  # Right shoulder
        if ls[3] >= 0.5 and rs[3] >= 0.5:
            dist = float(np.linalg.norm(ls[:2] - rs[:2]))
            if dist > 0.05:
                shoulder_widths.append(dist)

    if shoulder_widths:
        clip_scale = max(float(np.median(shoulder_widths)), 0.10)
    else:
        clip_scale = 0.20  # Default human anatomy ratio fallback

    # 2. Normalize frame-by-frame
    for t in range(T):
        frame = landmarks_75x4[t]
        ls = frame[11]
        rs = frame[12]

        # Determine anchor
        if ls[3] > 0.3 and rs[3] > 0.3:
            mid_shoulder = (ls[:2] + rs[:2]) / 2.0
            mid_z = (ls[2] + rs[2]) / 2.0
        elif ls[3] > 0.3:
            mid_shoulder = ls[:2]
            mid_z = ls[2]
        elif rs[3] > 0.3:
            mid_shoulder = rs[:2]
            mid_z = rs[2]
        else:
            mid_shoulder = np.array([0.5, 0.5], dtype=np.float32)
            mid_z = 0.0

        # Center and scale ONLY detected landmarks; missing points remain exactly 0.0
        for k in range(75):
            conf = frame[k, 3]
            if conf > 0.05:
                normed[t, k, 0] = (frame[k, 0] - mid_shoulder[0]) / clip_scale
                normed[t, k, 1] = (frame[k, 1] - mid_shoulder[1]) / clip_scale
                normed[t, k, 2] = (frame[k, 2] - mid_z) / clip_scale
                normed[t, k, 3] = conf
            else:
                normed[t, k, :] = [0.0, 0.0, 0.0, 0.0]

    return normed.astype(np.float32)


def temporal_resample(
    landmarks_75x4: np.ndarray,
    target_length: int = 64,
    detection_mask: Optional[np.ndarray] = None,
    original_fps: float = 30.0,
    return_metadata: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Uniformly resample variable-length sequence (T, 75, 4) to fixed target_length
    using linear interpolation along the temporal axis.
    Re-applies zero-masking where resampled confidence drops below 0.05.
    
    Returns:
      If return_metadata=False: resampled_landmarks (target_length, 75, 4) float32
      If return_metadata=True: (resampled_landmarks, resampled_mask, resampled_timestamps_ms)
    """
    orig_length = landmarks_75x4.shape[0]
    if orig_length == 0:
        raise ValueError("Cannot resample empty landmark sequence.")

    duration_sec = orig_length / max(original_fps, 1.0)
    resampled_timestamps_ms = np.linspace(0, int(round(duration_sec * 1000.0)), num=target_length, dtype=np.int64)

    if orig_length == target_length:
        resampled_landmarks = landmarks_75x4.copy()
        if detection_mask is not None:
            resampled_mask = detection_mask.copy()
        else:
            det_pose = np.any(resampled_landmarks[:, :33, 3] > 0.05, axis=1)
            det_lh = np.any(resampled_landmarks[:, 33:54, 3] > 0.05, axis=1)
            det_rh = np.any(resampled_landmarks[:, 54:75, 3] > 0.05, axis=1)
            resampled_mask = np.column_stack([det_pose, det_lh, det_rh]).astype(bool)
        if return_metadata:
            return resampled_landmarks, resampled_mask, resampled_timestamps_ms
        return resampled_landmarks

    old_indices = np.linspace(0, orig_length - 1, num=orig_length)
    new_indices = np.linspace(0, orig_length - 1, num=target_length)

    resampled_landmarks = np.zeros((target_length, 75, 4), dtype=np.float32)

    # Resample coordinate channels with 1D linear interpolation
    for k in range(75):
        for c in range(4):
            resampled_landmarks[:, k, c] = np.interp(
                new_indices, old_indices, landmarks_75x4[:, k, c]
            )

        # Clamp confidence to [0, 1]
        resampled_landmarks[:, k, 3] = np.clip(resampled_landmarks[:, k, 3], 0.0, 1.0)

        # Strict zero-masking: if resampled confidence < 0.05, zero out coordinates
        invalid_mask = resampled_landmarks[:, k, 3] < 0.05
        resampled_landmarks[invalid_mask, k, :] = [0.0, 0.0, 0.0, 0.0]

    # Resample detection mask via nearest-neighbor to preserve detection transitions
    if detection_mask is not None and detection_mask.shape[0] == orig_length:
        nn_indices = np.clip(np.round(new_indices).astype(int), 0, orig_length - 1)
        resampled_mask = detection_mask[nn_indices].astype(bool)
    else:
        det_pose = np.any(resampled_landmarks[:, :33, 3] > 0.05, axis=1)
        det_lh = np.any(resampled_landmarks[:, 33:54, 3] > 0.05, axis=1)
        det_rh = np.any(resampled_landmarks[:, 54:75, 3] > 0.05, axis=1)
        resampled_mask = np.column_stack([det_pose, det_lh, det_rh]).astype(bool)

    if return_metadata:
        return resampled_landmarks, resampled_mask, resampled_timestamps_ms
    return resampled_landmarks


def compute_hand_motion_energy(landmarks_75x4: np.ndarray) -> np.ndarray:
    """
    Calculate hand motion energy per frame using valid hand landmark coordinates (33..74).
    Input: (T, 75, 4) where channels are [x, y, z, confidence].
    Output: (T,) array of hand motion energy per frame.
    """
    T = landmarks_75x4.shape[0]
    if T < 2:
        return np.zeros(T, dtype=np.float32)

    hand_pts = landmarks_75x4[:, 33:75, :3]
    hand_conf = landmarks_75x4[:, 33:75, 3]
    valid = hand_conf > 0.05

    energy = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        pv = valid[t] & valid[t - 1]
        if np.any(pv):
            d = np.linalg.norm(hand_pts[t, pv] - hand_pts[t - 1, pv], axis=-1)
            energy[t] = float(np.mean(d))
        else:
            energy[t] = 0.0

    if T >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        energy = np.convolve(energy, kernel, mode="same")
    return energy.astype(np.float32)


def detect_active_motion_window(
    landmarks_75x4: np.ndarray,
    min_motion_threshold: float = 0.015,
    peak_ratio: float = 0.10,
    context_padding: int = 20,
    min_active_frames: int = 32,
    min_resting_frames_to_trim: int = 10,
) -> Tuple[int, int]:
    """
    Identify the active motion window [start, end) of meaningful hand movement.
    Preserves context before and after gesture.
    Handles low-motion signs and missing landmarks safely.
    Returns (0, T) fallback if motion detection fails or clip has no significant resting frames.
    """
    try:
        T = landmarks_75x4.shape[0]
        if T <= min_active_frames:
            return 0, T

        energy = compute_hand_motion_energy(landmarks_75x4)
        peak = float(np.max(energy))

        # Handle low-motion signs or missing landmarks safely
        if peak < min_motion_threshold:
            return 0, T

        thresh = max(min_motion_threshold, min(0.04, peak * peak_ratio))
        active_indices = np.where(energy >= thresh)[0]

        if len(active_indices) == 0:
            return 0, T

        first_active = int(active_indices[0])
        last_active = int(active_indices[-1])

        start_frame = max(0, first_active - context_padding)
        end_frame = min(T, last_active + 1 + context_padding)

        # Avoid micro-trimming if resting frames at edges are negligible
        if start_frame < min_resting_frames_to_trim:
            start_frame = 0
        if (T - end_frame) < min_resting_frames_to_trim:
            end_frame = T

        if (end_frame - start_frame) < min_active_frames:
            return 0, T

        return start_frame, end_frame
    except Exception:
        # Fall back to existing processing path if anything goes wrong
        return 0, landmarks_75x4.shape[0]
