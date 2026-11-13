# F-021: Feature Engineering and Normalization Report

**Feature ID:** F-021  
**Module:** Isolated ISL Sign Recognition (10-Class MVP)  
**Date:** 2026-11-12  
**Status:** Completed & Validated  
**Artifact Directory:** `datasets/features/`  
**Test Suite:** `tests/test_isolated_recognition_pipeline.py` (10 passed)  

---

## 1. Executive Summary

This report documents the design, implementation, and empirical validation of the feature engineering and normalization pipeline for Signova AI's 10-class individual Indian Sign Language (ISL) recognition system.

The pipeline converts raw 75-landmark MediaPipe coordinate extractions (produced in F-020) into standardized, leakage-free, fixed-length feature sequences ready for classification model training.

### Key Metrics Summary
| Metric | Specification / Result | Validation Status |
| :--- | :--- | :--- |
| **Approved Input Clips** | 187 clips (Train: 129, Val: 28, Test: 30) | Verified (100% processed) |
| **Target Sign Classes** | 10 classes (`food`, `hello`, `help`, `no`, `please`, `sorry`, `stop`, `thank you`, `water`, `yes`) | 100% covered across all splits |
| **Output Feature Directory** | `datasets/features/{train,validation,test}/*.npz` | 187 `.npz` files generated |
| **Total Storage Size** | 29.86 MB (compressed `.npz`) | Verified (159 KB/clip) |
| **Sequence Resampling** | Uniform linear interpolation to $T = 64$ frames | Zero temporal distortion |
| **Feature Dimension per Frame** | 600 features (300 spatial coordinates + 300 masked velocities) | Verified `(64, 600)` |
| **Landmark Zero Preservation** | Undetected points strictly `[0.0, 0.0, 0.0, 0.0]` | Zero ghost landmarks |
| **Velocity Spikes on Entry/Exit** | Prevented via strict dual-frame detection masking | Zero false motion spikes |
| **Missing / NaN / Inf Values** | 0 NaNs, 0 Infs across all 187 feature arrays | Verified |
| **Augmentation Policy** | Training DataLoader only; strictly forbidden on Val/Test | Unit test validated |
| **iSign T5 Translation Impact** | Zero modification to continuous translation pipeline | Decoupling preserved |

---

## 2. Pipeline Architecture & Feature Extraction Flow

```
Raw Landmark Extraction (.npz)
   Shape: (T, 75, 4) where T in [56, 229]
   Channels: [x, y, z, confidence]
                 │
                 ▼
    ┌─────────────────────────┐
    │  Spatial Normalization  │  Anchor: Mid-shoulder [(LS + RS) / 2]
    │   (spatial_normalize)   │  Scale: Clip-level median shoulder width
    └────────────┬────────────┘  Undetected points strictly zero-masked
                 │
                 ▼
    ┌─────────────────────────┐
    │   Temporal Resampling   │  Uniform 1D linear interpolation
    │   (temporal_resample)   │  Fixed sequence length: T = 64
    └────────────┬────────────┘  Strict confidence threshold (< 0.05 -> 0.0)
                 │
                 ▼
    ┌─────────────────────────┐
    │ Masked Velocity Extract │  v_t = s_t - s_{t-1} iff detected in both frames
    │ (compute_masked_vel)    │  v_0 = 0.0; Missing in either -> 0.0
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Combined Feature Tensor │  Concatenate [spatial_300, velocity_300]
    │       Shape: (64, 600)  │  Type: float32, Contiguous, Leakage-Free
    └─────────────────────────┘
```

---

## 3. Spatial Normalization Design & Validation

### 3.1 Normalization Formulation
For each video clip:
1. **Shoulder Anchor Point**: For every frame $t$ where both left shoulder ($k=11$) and right shoulder ($k=12$) have confidence $\ge 0.5$, compute the midpoint:
   $$\mathbf{m}_t = \frac{\mathbf{s}_{t, 11}^{xy} + \mathbf{s}_{t, 12}^{xy}}{2}$$
   If shoulders are undetected in frame $t$, fall back to clip-level median mid-shoulder position.
2. **Body Scale Normalization**: Compute Euclidean distance between left and right shoulders across all high-confidence frames:
   $$d_t = \|\mathbf{s}_{t, 11}^{xy} - \mathbf{s}_{t, 12}^{xy}\|_2$$
   Use the **clip-level median** shoulder width $d_{\text{scale}} = \text{median}(\{d_t\})$, clamped to a minimum of $0.10$ to prevent division by zero or extreme scaling:
   $$\text{scale} = \max(d_{\text{scale}}, 0.10)$$
3. **Normalized Coordinates**:
   For detected landmarks ($c_{t, k} > 0.05$):
   $$\hat{x}_{t, k} = \frac{x_{t, k} - m_{t, x}}{\text{scale}}, \quad \hat{y}_{t, k} = \frac{y_{t, k} - m_{t, y}}{\text{scale}}, \quad \hat{z}_{t, k} = \frac{z_{t, k}}{\text{scale}}$$
4. **Strict Zero-Masking of Undetected Points**:
   If $c_{t, k} \le 0.05$:
   $$\hat{\mathbf{s}}_{t, k} = [0.0, 0.0, 0.0, 0.0]$$
   Undetected landmarks are **never** shifted by the mid-shoulder anchor, guaranteeing zero artificial ghost landmarks.

---

## 4. Temporal Resampling to $T=64$

The original video clips in `datasets/landmarks/` vary from 56 frames to 229 frames (mean: 114.2 frames). A uniform fixed sequence length $T=64$ frames was established based on gesture duration distribution.

### Resampling Strategy
- **Coordinate Trajectories**: Linear 1D interpolation along the temporal axis for each of the $75 \times 4$ channels.
- **Strict Post-Resampling Thresholding**: Because interpolation between a detected coordinate and an undetected point ($0.0$) could produce intermediate confidence values, any resampled frame with confidence $< 0.05$ is clamped to $[0.0, 0.0, 0.0, 0.0]$.
- **Detection Mask Resampling**: Nearest-neighbor interpolation preserves discrete presence/absence state boundaries for `[pose, left_hand, right_hand]`.
- **Timestamps**: Linearly scaled to maintain millisecond progression from $0$ to total clip duration.

**Visual Verification:**
See `docs/reports/figures/feature_temporal_resampling.png` demonstrating preservation of right wrist velocity profile between original ($T=112$) and resampled ($T=64$) sequences.

---

## 5. Masked Velocity Features

Raw spatial coordinates alone describe hand shape and position, but sign recognition requires temporal dynamics (e.g. distinguishing the upward motion of `water` from the downward sweep of `thank you`).

### Masked Velocity Formulation
For frame $t \in [1, T-1]$ and landmark $k \in [0, 74]$:
$$\mathbf{v}_{t, k} = \begin{cases}
[\hat{x}_{t, k} - \hat{x}_{t-1, k}, \; \hat{y}_{t, k} - \hat{y}_{t-1, k}, \; \hat{z}_{t, k} - \hat{z}_{t-1, k}, \; \min(c_{t, k}, c_{t-1, k})], & \text{if } c_{t, k} > 0.05 \text{ and } c_{t-1, k} > 0.05 \\
[0.0, 0.0, 0.0, 0.0], & \text{otherwise}
\end{cases}$$

At $t = 0$:
$$\mathbf{v}_{0, k} = [0.0, 0.0, 0.0, 0.0]$$

### Prevention of False Spikes
In standard naive velocity computation $v_t = s_t - s_{t-1}$, when a hand first enters the frame, $s_{t-1} = 0.0$ and $s_t \approx 0.5$, creating a massive artificial velocity spike of $+0.50$ (and $-0.50$ on exit).
The dual-frame detection condition ensures that hand entry/exit creates zero artificial velocity, preserving authentic gesture kinetics.

**Visual Verification:**
- `docs/reports/figures/feature_spatial_vs_velocity.png`: Demonstrates feature matrix heatmaps for spatial vs masked velocity channels.
- `docs/reports/figures/feature_class_velocity_profile.png`: Shows class-specific mean hand velocity profiles across the 64 frames (e.g., dynamic multi-cycle wave in `hello` vs stationary hold in `stop`).

---

## 6. Dataset Split Statistical Summary

The feature dataset is stored in `datasets/features/{train,validation,test}/` and indexed in `datasets/features/feature_index.csv`.

### Aggregate Split Statistics
| Split | Approved Samples | Total File Size | Spatial Mean $\pm$ Std | Velocity Mean $\pm$ Std | Velocity Non-Zero Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 129 | 20.41 MB | $0.1651 \pm 0.9756$ | $0.1050 \pm 0.3152$ | 49.62% |
| **Validation** | 28 | 4.68 MB | $0.1395 \pm 0.9302$ | $0.1145 \pm 0.3247$ | 53.05% |
| **Test** | 30 | 4.77 MB | $0.1443 \pm 0.9083$ | $0.1080 \pm 0.3157$ | 49.97% |
| **Total / Overall** | **187** | **29.86 MB** | — | — | **50.19%** |

### Class Balance Across Splits
| Class Label | Train (N=129) | Validation (N=28) | Test (N=30) | Total Approved |
| :--- | :---: | :---: | :---: | :---: |
| `food` | 12 | 3 | 2 | 17 |
| `hello` | 26 | 5 | 5 | 36 |
| `help` | 12 | 2 | 2 | 16 |
| `no` | 12 | 3 | 2 | 17 |
| `please` | 12 | 3 | 3 | 18 |
| `sorry` | 3 | 1 | 2 | 6 |
| `stop` | 1 | 1 | 1 | 3 |
| `thank you` | 27 | 5 | 7 | 39 |
| `water` | 12 | 3 | 3 | 18 |
| `yes` | 12 | 2 | 3 | 17 |
| **Total** | **129** | **28** | **30** | **187** |

---

## 7. Training-Only Data Augmentation

Implemented in `ai/isolated/training/augmentation.py` via `IsolatedAugmenter`.

### Augmentation Rules:
1. **Target Split Restriction**: Augmentation is **strictly prohibited** on validation and test sets. Invoking `augment(..., split="validation")` or `augment(..., split="test")` immediately raises a `ValueError`.
2. **Temporal Scaling**: Uniform scaling by factor $\alpha \in [0.90, 1.10]$, simulating signers speaking slightly slower or faster, with linear re-interpolation back to $T=64$.
3. **Spatial Gaussian Jitter**: Zero-mean additive Gaussian noise $\mathcal{N}(0, \sigma^2)$ where $\sigma \le 0.005$ applied strictly to $(x, y)$ of detected landmarks ($c > 0.05$). Undetected landmarks and confidence values remain strictly uncorrupted.

---

## 8. Verification and Test Results

Automated test suite `tests/test_isolated_recognition_pipeline.py` executes 10 comprehensive tests:

| Test Name | Focus | Result |
| :--- | :--- | :---: |
| `test_1_temporal_resampling` | Arbitrary frame lengths resample to exactly 64 frames | PASSED |
| `test_2_spatial_normalization` | Mid-shoulder centering and scale normalization | PASSED |
| `test_3_feature_extraction` | 600-dim feature concatenation & first-frame velocity = 0.0 | PASSED |
| `test_4_isolated_recognizer_prediction` | Inference wrapper returns top-k predictions with latencies | PASSED |
| `test_5_load_split_dataset` | Direct loading of 28 validation split clips | PASSED |
| `test_6_evaluation_metrics` | Multi-class accuracy, macro F1, and confusion matrix | PASSED |
| `test_7_module_separation_from_isign_t5` | Isolated code executes with zero dependency on iSign T5 | PASSED |
| `test_8_isolated_augmenter` | Boundary enforcement and zero-mask preservation during jitter | PASSED |
| `test_9_precomputed_features_loading` | Fast loading from `datasets/features/` produces identical shape | PASSED |
| `test_10_feature_dataset_files_and_manifest` | Manifest row count (187) and summary JSON consistency | PASSED |

---

## 9. Next Steps for F-022

Feature engineering and dataset artifact preparation are complete.
The pipeline is fully ready for **F-022: Isolated Sign Classifier Model Exploration & Training**:
- Baseline models: Random Forest, SVM, LightGBM, and 1D-CNN / Bi-LSTM / GRU.
- Stratified evaluation on `datasets/features/validation/`.
- Final testing on `datasets/features/test/`.
- Exporting best model checkpoint to `models/checkpoints/isolated/isolated_10class_best.joblib`.
