"""
Temporal Feature Engineering for Continuous Indian Sign Language Recognition.

Module: ai.continuous.temporal_features
Design:
- 75 landmarks: Pose (0..32), Left Hand (33..53), Right Hand (54..74)
- Torso-centered on shoulder midpoint (landmarks 11 & 12)
- Scaled by inter-shoulder Euclidean distance (clamped >= 0.01, fallback 0.2)
- Strict zero-imputation: missing landmarks/components remain 0.0
- Velocity: dx/dt, dy/dt, dz/dt between consecutive frames with safe missing landmark handling
  (If landmark was missing [detection_mask False] in frame t or t-1, velocity is 0.0)
- Acceleration: dv/dt between consecutive frames with missing-landmark safety
  (If velocity is zero due to missing landmark in frame t or t-1, acceleration is 0.0)
- Detection mask: (T, 3) boolean flags converted to float32
- Total output dimension per frame: 225 spatial + 225 velocity + 225 acceleration + 3 masks = 678 features
"""

from __future__ import annotations

from typing import Any, List, Optional, Union
import numpy as np


def extract_temporal_features(
    landmarks: Union[np.ndarray, List[Any]],
    detection_mask: Optional[Union[np.ndarray, List[Any]]] = None,
    timestamps_ms: Optional[Union[np.ndarray, List[int]]] = None,
    fps: float = 25.0,
) -> np.ndarray:
    """
    Extract 678-dimensional temporal feature matrix from landmark sequence.

    Args:
        landmarks: Array of shape (T, 75, 4), (T, 300), or (T, 678).
                   Channels are [x, y, z, visibility/confidence].
        detection_mask: Optional boolean/binary array of shape (T, 3) indicating
                        presence of [pose, left_hand, right_hand].
        timestamps_ms: Optional millisecond timestamps of shape (T,).
        fps: Sampling rate fallback in Hz (default: 25.0).

    Returns:
        np.ndarray: Feature matrix of shape (T, 678) with dtype float32:
                    - [0:225]: Torso-normalized spatial coordinates (x, y, z for 75 landmarks)
                    - [225:450]: Safe velocities (dx/dt, dy/dt, dz/dt)
                    - [450:675]: Safe accelerations (dv/dt)
                    - [675:678]: Detection mask flags [pose, left_hand, right_hand]
    """
    arr = np.asarray(landmarks, dtype=np.float32)

    # Handle 1D input
    if arr.ndim == 1:
        if arr.shape[0] == 678:
            return arr.reshape(1, 678)
        if arr.shape[0] == 300:
            arr = arr.reshape(1, 75, 4)
        else:
            raise ValueError(f"Unexpected 1D landmark size: {arr.shape[0]}")

    # Handle 2D input
    if arr.ndim == 2:
        if arr.shape[1] == 678:
            return arr.astype(np.float32)
        if arr.shape[1] == 300 or arr.shape[1] == 75 * 4:
            arr = arr.reshape(-1, 75, 4)
        else:
            raise ValueError(f"Unexpected 2D landmark shape: {arr.shape}")

    if arr.ndim != 3 or arr.shape[1] != 75 or arr.shape[2] < 3:
        raise ValueError(f"Expected landmarks with shape (T, 75, 4), got {arr.shape}")

    T = arr.shape[0]
    if T == 0:
        return np.zeros((0, 678), dtype=np.float32)

    coords = arr[:, :, :3].copy()
    confs = arr[:, :, 3].copy() if arr.shape[2] >= 4 else np.ones((T, 75), dtype=np.float32)

    # 1. Parse or infer detection mask: shape (T, 3) -> [Pose, Left Hand, Right Hand]
    if detection_mask is not None:
        mask = np.asarray(detection_mask, dtype=bool)
        if mask.ndim == 1 and mask.shape[0] == 3:
            mask = np.repeat(mask[None, :], T, axis=0)
        elif mask.shape[0] != T or mask.shape[1] != 3:
            raise ValueError(f"detection_mask must be shape ({T}, 3), got {mask.shape}")
    else:
        pose_det = (np.sum(np.abs(coords[:, :33, :]), axis=(1, 2)) > 1e-4) | (np.sum(confs[:, :33] > 0.05, axis=1) > 0)
        lh_det = (np.sum(np.abs(coords[:, 33:54, :]), axis=(1, 2)) > 1e-4) | (np.sum(confs[:, 33:54] > 0.05, axis=1) > 0)
        rh_det = (np.sum(np.abs(coords[:, 54:75, :]), axis=(1, 2)) > 1e-4) | (np.sum(confs[:, 54:75] > 0.05, axis=1) > 0)
        mask = np.stack([pose_det, lh_det, rh_det], axis=-1)

    # 2. Torso-center on shoulder midpoint (landmarks 11, 12) & scale by inter-shoulder distance
    left_sh = coords[:, 11, :2]
    right_sh = coords[:, 12, :2]
    sh_mid = (left_sh + right_sh) / 2.0  # (T, 2)
    sh_dist = np.linalg.norm(left_sh - right_sh, axis=-1, keepdims=True)  # (T, 1)

    # Clamped >= 0.01, fallback 0.2
    valid_dist = np.where((sh_dist >= 0.01) & np.isfinite(sh_dist), sh_dist, 0.2)
    sh_mid = np.nan_to_num(sh_mid, nan=0.0)

    norm_coords = np.zeros((T, 75, 3), dtype=np.float32)

    # Pose (0..32)
    pose_present = mask[:, 0:1]
    norm_coords[:, 0:33, :2] = np.where(
        pose_present[:, :, None],
        (coords[:, 0:33, :2] - sh_mid[:, None, :]) / valid_dist[:, :, None],
        0.0,
    )
    norm_coords[:, 0:33, 2] = np.where(
        pose_present,
        coords[:, 0:33, 2] / valid_dist,
        0.0,
    )

    # Left Hand (33..53)
    lh_present = mask[:, 1:2]
    norm_coords[:, 33:54, :2] = np.where(
        lh_present[:, :, None],
        (coords[:, 33:54, :2] - sh_mid[:, None, :]) / valid_dist[:, :, None],
        0.0,
    )
    norm_coords[:, 33:54, 2] = np.where(
        lh_present,
        coords[:, 33:54, 2] / valid_dist,
        0.0,
    )

    # Right Hand (54..74)
    rh_present = mask[:, 2:3]
    norm_coords[:, 54:75, :2] = np.where(
        rh_present[:, :, None],
        (coords[:, 54:75, :2] - sh_mid[:, None, :]) / valid_dist[:, :, None],
        0.0,
    )
    norm_coords[:, 54:75, 2] = np.where(
        rh_present,
        coords[:, 54:75, 2] / valid_dist,
        0.0,
    )

    # Ensure clean finite numbers
    norm_coords = np.nan_to_num(norm_coords, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Calculate dt per frame
    fallback_dt = 1.0 / max(fps, 1.0)
    if timestamps_ms is not None and len(timestamps_ms) == T and T > 1:
        ts = np.asarray(timestamps_ms, dtype=np.float64)
        delta_ms = np.diff(ts).astype(np.float32)
        delta_sec = delta_ms / 1000.0
        dt = np.where((delta_sec > 0.0) & np.isfinite(delta_sec), delta_sec, fallback_dt)[:, None, None]
    else:
        dt = np.full((max(0, T - 1), 1, 1), fallback_dt, dtype=np.float32)

    # Track component presence for all 75 landmarks
    comp_present = np.zeros((T, 75), dtype=bool)
    comp_present[:, 0:33] = mask[:, 0:1]
    comp_present[:, 33:54] = mask[:, 1:2]
    comp_present[:, 54:75] = mask[:, 2:3]

    # 4. Safe Velocity: dx/dt, dy/dt, dz/dt
    # If landmark was missing in frame t or frame t-1, velocity must be 0.0!
    velocity = np.zeros((T, 75, 3), dtype=np.float32)
    if T > 1:
        valid_vel = comp_present[1:] & comp_present[:-1]  # (T-1, 75)
        diff_coords = norm_coords[1:] - norm_coords[:-1]  # (T-1, 75, 3)
        v = diff_coords / dt
        velocity[1:] = np.where(valid_vel[:, :, None], v, 0.0)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Safe Acceleration: dv/dt
    # If velocity is zero due to missing landmark in frame t or t-1, acceleration must be 0.0!
    # Requires landmark present in t, t-1, and t-2
    acceleration = np.zeros((T, 75, 3), dtype=np.float32)
    if T > 2:
        valid_acc = comp_present[2:] & comp_present[1:-1] & comp_present[:-2]  # (T-2, 75)
        diff_vel = velocity[2:] - velocity[1:-1]  # (T-2, 75, 3)
        dt_acc = dt[1:]  # (T-2, 1, 1)
        a = diff_vel / dt_acc
        acceleration[2:] = np.where(valid_acc[:, :, None], a, 0.0)
        acceleration = np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0)

    # 6. Concatenate: (225 spatial + 225 velocity + 225 acceleration + 3 masks) = 678
    spatial_flat = norm_coords.reshape(T, 225)
    vel_flat = velocity.reshape(T, 225)
    acc_flat = acceleration.reshape(T, 225)
    mask_float = mask.astype(np.float32)

    features = np.concatenate(
        [spatial_flat, vel_flat, acc_flat, mask_float],
        axis=-1,
    ).astype(np.float32)

    return features
