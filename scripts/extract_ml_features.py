"""
Feature Preparation Pipeline for ISL Sentence Recognition ML Algorithms.

Extracts and caches:
1. Temporal Pyramid features (5,424-dim): 4 segments (Global, Early, Middle, Late) x (mean + std)
2. Train-fitted StandardScaler and PCA features
3. Frame-level PCA (48-dim) with 2x temporal stride for DTW+KNN
4. Variable-length ordered sequence tensors for Temporal CNN

Strict data hygiene:
- 499 canonical train samples, 85 validation samples, 98 test samples.
- Preprocessors (StandardScaler, PCA) fitted strictly on train split only.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.sentence_dataset import ISLCSLTRSentenceDataset


def extract_temporal_pyramid(seq: np.ndarray) -> np.ndarray:
    """
    Extract 7-segment temporal pyramid features from sequence of shape (T, 678).
    Structure:
      - Level 0 (Whole):    [0 : T]
      - Level 1 (Halves):   [0 : T//2], [T//2 : T]
      - Level 2 (Quarters): [0 : T//4], [T//4 : 2*(T//4)], [2*(T//4) : 3*(T//4)], [3*(T//4) : T]
    Statistics per segment: Mean (678,) + Std (678,)
    Total output dimension: 7 segments * 2 statistics * 678 features = 9,492 dimensions.
    """
    T, D = seq.shape
    assert D == 678, f"Expected 678 dimensions, got {D}"

    # Partition indices
    h1 = max(1, T // 2)

    q1 = max(1, T // 4)
    q2 = max(q1 + 1, 2 * (T // 4))
    q3 = max(q2 + 1, 3 * (T // 4))

    segments = [
        seq,                 # 1. Whole sequence (0:T)
        seq[:h1],            # 2. First half (0:T//2)
        seq[h1:],            # 3. Second half (T//2:T)
        seq[:q1],            # 4. First quarter (0:T//4)
        seq[q1:q2],          # 5. Second quarter (T//4:2*T//4)
        seq[q2:q3],          # 6. Third quarter (2*T//4:3*T//4)
        seq[q3:],            # 7. Fourth quarter (3*T//4:T)
    ]

    pyramid_parts = []
    for seg in segments:
        if len(seg) == 0:
            seg = seq  # fallback if sequence extremely short (<4 frames)
        mean_stat = np.mean(seg, axis=0)  # (678,)
        std_stat = np.std(seg, axis=0)    # (678,)
        pyramid_parts.extend([mean_stat, std_stat])

    return np.concatenate(pyramid_parts).astype(np.float32)


def main() -> None:
    cache_dir = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "manifests"

    print("=== EXTRACTING ML FEATURES ===")
    t0 = time.time()

    splits = ["train", "validation", "test"]
    datasets = {}
    for split in splits:
        manifest_file = manifests_dir / f"{split}_manifest.json"
        ds = ISLCSLTRSentenceDataset(
            manifest_path=manifest_file,
            canonical_only=True,
            cache_features=True,
            augment=False,
        )
        datasets[split] = ds
        print(f"Loaded {split} split: {len(ds)} canonical samples")

    # 1. Extract raw sequences and pyramid features
    split_data = {}
    for split in splits:
        ds = datasets[split]
        raw_seqs: List[np.ndarray] = []
        pyramids: List[np.ndarray] = []
        labels: List[int] = []
        signers: List[int] = []
        video_ids: List[str] = []
        masks: List[np.ndarray] = []

        for i in range(len(ds)):
            item = ds[i]
            # features is torch.Tensor (T, 678)
            seq_np = item["features"].cpu().numpy()
            pyr_np = extract_temporal_pyramid(seq_np)

            raw_seqs.append(seq_np)
            pyramids.append(pyr_np)
            labels.append(int(item["label"]))
            signers.append(int(item["signer_id"]))
            video_ids.append(str(item["video_id"]))
            # Detection mask is stored in last 3 dimensions of 678 features (indices 675:678)
            masks.append(seq_np[:, 675:678])

        split_data[split] = {
            "raw_seqs": raw_seqs,
            "pyramids": np.stack(pyramids, axis=0),  # (N, 5424)
            "labels": np.array(labels, dtype=np.int64),
            "signers": np.array(signers, dtype=np.int64),
            "video_ids": video_ids,
            "masks": masks,
        }
        print(f"Processed {split}: pyramids shape {split_data[split]['pyramids'].shape}")

    # 2. Fit Scaler strictly on train pyramid features
    print("\nFitting StandardScaler strictly on train pyramid features...")
    scaler = StandardScaler()
    scaler.fit(split_data["train"]["pyramids"])

    # 3. Fit PCA strictly on train scaled pyramid features (for SVM testing)
    print("Fitting PCA(n_components=128) on train scaled pyramid features...")
    pca_pyramid_128 = PCA(n_components=128, random_state=42)
    X_train_scaled = scaler.transform(split_data["train"]["pyramids"])
    pca_pyramid_128.fit(X_train_scaled)

    pca_pyramid_64 = PCA(n_components=64, random_state=42)
    pca_pyramid_64.fit(X_train_scaled)

    # 4. Fit Frame-level PCA(n_components=48) strictly on train frames for DTW
    print("Fitting Frame-level PCA(n_components=48) strictly on train frames for DTW...")
    all_train_frames = np.concatenate(split_data["train"]["raw_seqs"], axis=0)  # (total_train_frames, 678)
    frame_scaler = StandardScaler()
    all_train_frames_scaled = frame_scaler.fit_transform(all_train_frames)

    frame_pca_48 = PCA(n_components=48, random_state=42)
    frame_pca_48.fit(all_train_frames_scaled)
    print(f"Frame PCA explained variance ratio (48 components): {frame_pca_48.explained_variance_ratio_.sum():.4f}")

    # Transform all splits for DTW (stride 2 + frame PCA)
    dtw_data = {}
    for split in splits:
        dtw_seqs = []
        for seq, mask in zip(split_data[split]["raw_seqs"], split_data[split]["masks"]):
            # Stride 2 downsampling
            sub_seq = seq[::2]
            sub_mask = mask[::2]
            # Standardize and project to 48 PCA dims
            sub_seq_scaled = frame_scaler.transform(sub_seq)
            sub_seq_pca = frame_pca_48.transform(sub_seq_scaled).astype(np.float32)
            dtw_seqs.append({
                "features_pca": sub_seq_pca,  # (T//2, 48)
                "mask": sub_mask,             # (T//2, 3)
            })
        dtw_data[split] = {
            "seqs": dtw_seqs,
            "labels": split_data[split]["labels"],
            "signers": split_data[split]["signers"],
            "video_ids": split_data[split]["video_ids"],
        }

    # Save to disk
    print("\nSaving prepared feature caches to disk...")
    # 1. Pyramid NPZ
    for split in splits:
        X_raw = split_data[split]["pyramids"]
        X_sc = scaler.transform(X_raw).astype(np.float32)
        X_pca128 = pca_pyramid_128.transform(X_sc).astype(np.float32)
        X_pca64 = pca_pyramid_64.transform(X_sc).astype(np.float32)

        np.savez_compressed(
            cache_dir / f"pyramid_{split}.npz",
            X_raw=X_raw,
            X_scaled=X_sc,
            X_pca128=X_pca128,
            X_pca64=X_pca64,
            y=split_data[split]["labels"],
            signers=split_data[split]["signers"],
            video_ids=split_data[split]["video_ids"],
        )

    # 2. DTW PKL
    with open(cache_dir / "dtw_dataset.pkl", "wb") as f:
        pickle.dump(dtw_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. Raw sequences for CNN / PyTorch
    raw_seq_cache = {
        split: {
            "seqs": split_data[split]["raw_seqs"],
            "masks": split_data[split]["masks"],
            "labels": split_data[split]["labels"],
            "signers": split_data[split]["signers"],
            "video_ids": split_data[split]["video_ids"],
        }
        for split in splits
    }
    with open(cache_dir / "raw_sequences.pkl", "wb") as f:
        pickle.dump(raw_seq_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 4. Save preprocessors
    with open(cache_dir / "preprocessors.pkl", "wb") as f:
        pickle.dump({
            "pyramid_scaler": scaler,
            "pyramid_pca128": pca_pyramid_128,
            "pyramid_pca64": pca_pyramid_64,
            "frame_scaler": frame_scaler,
            "frame_pca_48": frame_pca_48,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Feature preparation complete in {time.time() - t0:.2f}s! Artifacts stored in {cache_dir}")


if __name__ == "__main__":
    main()
