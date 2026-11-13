"""
Data augmentation utilities for Individual ISL Sign Recognition.
IMPORTANT: Augmentations are STRICTLY designed for training split DataLoader transforms.
NEVER apply to validation or test splits.
"""

from typing import Optional, Tuple
import numpy as np


POSE_SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
]


class IsolatedAugmenter:
    """
    Data augmenter for isolated sign landmark sequences.
    
    Transforms:
      1. Temporal scaling: resamples to random speed factor in [scale_min, scale_max] (default 0.9x to 1.1x)
         and resamples back to fixed length T=64.
      2. Spatial Gaussian jitter: adds N(0, jitter_std^2) (default std <= 0.005) strictly to x, y coordinates
         of DETECTED landmarks (confidence > 0.05). Undetected landmarks remain exactly 0.0.
      3. Horizontal bilateral flipping: mirrors pose and swaps left/right hands to simulate left-handed signers.
    """

    def __init__(
        self,
        jitter_std: float = 0.003,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        apply_prob: float = 0.5,
        flip_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if jitter_std > 0.005:
            raise ValueError(f"jitter_std must be <= 0.005 to preserve fine finger gestures, got {jitter_std}")
        self.jitter_std = jitter_std
        self.scale_min, self.scale_max = scale_range
        self.apply_prob = apply_prob
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed)

    def apply_horizontal_flip(self, landmarks_75x4: np.ndarray) -> np.ndarray:
        """
        Horizontally mirrors signer pose and swaps left/right hands to simulate left-handed signers.
        Strictly preserves zeros for undetected landmarks.
        """
        flipped = landmarks_75x4.copy()
        # Invert X coordinate for detected landmarks
        det = flipped[:, :, 3] > 0.05
        flipped[det, 0] = -flipped[det, 0]

        # Swap paired pose landmarks (left <-> right)
        for l, r in POSE_SWAP_PAIRS:
            flipped[:, [l, r], :] = flipped[:, [r, l], :].copy()

        # Swap Left Hand (33:54) and Right Hand (54:75)
        lh = flipped[:, 33:54, :].copy()
        rh = flipped[:, 54:75, :].copy()
        flipped[:, 33:54, :] = rh
        flipped[:, 54:75, :] = lh

        return flipped

    def apply_spatial_jitter(self, landmarks_75x4: np.ndarray) -> np.ndarray:
        """
        Adds zero-mean Gaussian jitter to x, y coordinates of detected landmarks only.
        Preserves z and confidence channels, and keeps undetected points at 0.0.
        """
        augmented = landmarks_75x4.copy()
        detected = augmented[:, :, 3] > 0.05
        noise = self.rng.normal(0.0, self.jitter_std, size=(*landmarks_75x4.shape[:2], 2)).astype(np.float32)
        augmented[detected, 0] += noise[detected, 0]
        augmented[detected, 1] += noise[detected, 1]
        return augmented

    def apply_temporal_scaling(self, landmarks_75x4: np.ndarray) -> np.ndarray:
        """
        Applies subtle speed scaling (0.9x to 1.1x) while preserving output length T.
        """
        T = landmarks_75x4.shape[0]
        factor = float(self.rng.uniform(self.scale_min, self.scale_max))
        intermediate_len = max(int(round(T * factor)), 8)

        old_idx = np.linspace(0, T - 1, num=T)
        inter_idx = np.linspace(0, T - 1, num=intermediate_len)
        inter_landmarks = np.zeros((intermediate_len, 75, 4), dtype=np.float32)
        for k in range(75):
            for c in range(4):
                inter_landmarks[:, k, c] = np.interp(inter_idx, old_idx, landmarks_75x4[:, k, c])

        final_idx = np.linspace(0, intermediate_len - 1, num=T)
        back_idx = np.linspace(0, intermediate_len - 1, num=intermediate_len)
        rescaled = np.zeros((T, 75, 4), dtype=np.float32)
        for k in range(75):
            for c in range(4):
                rescaled[:, k, c] = np.interp(final_idx, back_idx, inter_landmarks[:, k, c])
            rescaled[:, k, 3] = np.clip(rescaled[:, k, 3], 0.0, 1.0)
            invalid = rescaled[:, k, 3] < 0.05
            rescaled[invalid, k, :] = [0.0, 0.0, 0.0, 0.0]

        return rescaled

    def augment(self, landmarks_75x4: np.ndarray, split: str = "train") -> np.ndarray:
        """
        Executes stochastic augmentation on a single sample.
        Raises an error if invoked on validation or test splits.
        """
        if split.lower() in ("val", "validation", "test"):
            raise ValueError(f"Augmentation is strictly forbidden on {split} split!")

        out = landmarks_75x4.copy()
        if self.rng.random() < self.flip_prob:
            out = self.apply_horizontal_flip(out)
        if self.rng.random() < self.apply_prob:
            out = self.apply_temporal_scaling(out)
        if self.rng.random() < self.apply_prob:
            out = self.apply_spatial_jitter(out)
        return out
