"""
Feature engineering module for Individual ISL Sign Recognition.
Extracts masked first-order temporal velocity features (600 dimensions total: 300 spatial + 300 velocity).
"""

from typing import Tuple, Union
import numpy as np


def compute_masked_velocity(landmarks_75x4: np.ndarray) -> np.ndarray:
    """
    Computes first-order temporal velocity: v_t = s_t - s_{t-1}.
    Masked: velocity is calculated ONLY when landmark k is detected in BOTH frame t and frame t-1.
    If undetected in either frame, v_t = 0.0 to prevent artificial motion spikes on hand entry/exit.
    
    Input: (T, 75, 4)
    Output: (T, 300) velocity features (75 landmarks x 4 channels [vx, vy, vz, v_valid]).
    """
    T = landmarks_75x4.shape[0]
    velocity = np.zeros((T, 75, 4), dtype=np.float32)

    if T <= 1:
        return velocity.reshape(T, -1)

    for t in range(1, T):
        curr_frame = landmarks_75x4[t]
        prev_frame = landmarks_75x4[t - 1]

        for k in range(75):
            curr_conf = curr_frame[k, 3]
            prev_conf = prev_frame[k, 3]

            # Valid in both frames
            if curr_conf > 0.05 and prev_conf > 0.05:
                velocity[t, k, 0] = curr_frame[k, 0] - prev_frame[k, 0]
                velocity[t, k, 1] = curr_frame[k, 1] - prev_frame[k, 1]
                velocity[t, k, 2] = curr_frame[k, 2] - prev_frame[k, 2]
                velocity[t, k, 3] = min(curr_conf, prev_conf)
            else:
                velocity[t, k, :] = [0.0, 0.0, 0.0, 0.0]

    return velocity.reshape(T, -1).astype(np.float32)


# Backward-compatible alias
compute_velocity = compute_masked_velocity


def extract_isolated_features(
    landmarks_75x4: np.ndarray,
    return_components: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extracts spatial (300) and masked velocity (300) features for isolated sign recognition.
    
    Input: (T, 75, 4)
    Returns:
      If return_components=False: combined_features: (T, 600) float32
      If return_components=True: (combined_600, spatial_300, velocity_300)
    """
    T = landmarks_75x4.shape[0]
    spatial_300 = landmarks_75x4.reshape(T, -1).astype(np.float32)
    velocity_300 = compute_masked_velocity(landmarks_75x4)
    combined_600 = np.concatenate([spatial_300, velocity_300], axis=1).astype(np.float32)

    if return_components:
        return combined_600, spatial_300, velocity_300
    return combined_600


def pool_temporal_features(seq_features: np.ndarray) -> np.ndarray:
    """
    Statistical and temporal summary pooling over time axis T.
    Computes:
      - Mean over T (600)
      - Standard deviation over T (600)
      - Max over T (600)
      - Displacement delta: last frame - first frame (600)
    Result: 2,400 dimensions.
    Input: (T, 600) or (N, T, 600)
    Output: (2400,) or (N, 2400) float32
    """
    is_batched = seq_features.ndim == 3
    if not is_batched:
        seq = seq_features[np.newaxis, ...]
    else:
        seq = seq_features

    f_mean = np.mean(seq, axis=1)
    f_std = np.std(seq, axis=1)
    f_max = np.max(seq, axis=1)
    f_delta = seq[:, -1, :] - seq[:, 0, :]

    pooled = np.concatenate([f_mean, f_std, f_max, f_delta], axis=-1).astype(np.float32)

    return pooled if is_batched else pooled[0]
