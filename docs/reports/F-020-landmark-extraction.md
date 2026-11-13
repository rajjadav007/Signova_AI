# F-020: MediaPipe Landmark Extraction Pipeline Report

**Date:** 2026-11-12  
**Dataset Source:** `D:\Signova AI\dataset\isl-isolated-40words`  
**Input Splits:** `datasets/splits/{train,validation,test}.csv`  
**Output Location:** `datasets/landmarks/{train,validation,test}/`  
**Prior Reports:**  
- [F-018.1-dataset-inspection.md](file:///D:/Signova%20AI/docs/reports/F-018.1-dataset-inspection.md)  
- [F-018.2-class-validation.md](file:///D:/Signova%20AI/docs/reports/F-018.2-class-validation.md)  
- [F-019-dataset-splitting.md](file:///D:/Signova%20AI/docs/reports/F-019-dataset-splitting.md)  
**Pipeline Script:** `scripts/extract_split_landmarks.py`  
**Status:** Complete & 100% Validated  

---

## 1. Executive Summary

This report documents the implementation and execution of the video-to-landmark feature extraction pipeline for Signova AI's 10-class individual Indian Sign Language (ISL) recognition dataset.

Reusing the established `ModernMediaPipeExtractor` (Google MediaPipe Tasks API: `PoseLandmarker` + `HandLandmarker` in continuous VIDEO mode), all 187 approved video clips across Train, Validation, and Test partitions were successfully processed without failure or corruption.

- **Extraction Success Rate:** 100% (187 attempted, 187 successful, 0 failed).
- **Total Frames Processed:** 21,357 video frames.
- **Landmark Specification:** 75 keypoints per frame (33 pose + 21 left hand + 21 right hand) × 4 channels `[x, y, z, confidence]` = 300 spatial features.
- **Serialization:** Compressed `.npz` files storing landmarks `(T, 75, 4)`, spatial features `(T, 300)`, timestamps `(T,)`, frame indices `(T,)`, and detection masks `(T, 3)`.
- **Total Execution Time:** 1,016.2 seconds (~16.9 minutes).
- **Average Extraction Throughput:** ~21.0 FPS.

---

## 2. Pipeline Architecture & Implementation

### 2.1 Component Reuse
The extraction engine reuses the project's native `ModernMediaPipeExtractor` from `ai/pose_extraction/mediapipe_extractor.py`:
- `pose_landmarker_full.task` (MediaPipe Tasks API)
- `hand_landmarker.task` (MediaPipe Tasks API, configured for up to 2 hands)
- Windows C++ LiteRT double-free exception suppression (`STATUS_INVALID_HANDLE` 0xc0000008).
- Anatomical handedness correction: Disambiguates left vs right hands using wrist proximity to pose landmarks #15 and #16 (overriding selfie-mirror inversions in third-person observer footage).

### 2.2 Extraction Outputs Structure
Outputs are saved in a clean, split-isolated directory structure:
```text
datasets/landmarks/
├── train/                     # 129 compressed .npz archives
├── validation/                # 28 compressed .npz archives
├── test/                      # 30 compressed .npz archives
├── extraction_index.csv       # Complete metadata manifest (187 rows)
└── extraction_summary.json    # Aggregated run metrics and quality logs
```

### 2.3 Per-Video `.npz` File Schema
Each video yields a single compressed `{video_id}.npz` archive containing:
1. `landmarks`: `np.ndarray` of shape `(T, 75, 4)`, `float32` (Pose 0..32, Left Hand 33..53, Right Hand 54..74).
2. `spatial_features`: `np.ndarray` of shape `(T, 300)`, `float32` (flattened $75 \times 4$).
3. `timestamps_ms`: `np.ndarray` of shape `(T,)`, `int64` (strictly monotonic capture timestamps).
4. `frame_indices`: `np.ndarray` of shape `(T,)`, `int32` ($0, 1, \dots, T-1$).
5. `detection_mask`: `np.ndarray` of shape `(T, 3)`, `bool` indicating detection status for `[pose, left_hand, right_hand]` per frame.

---

## 3. Split-by-Split Extraction Results

| Split | Attempted | Succeeded | Failed | Total Frames | Mean Pose Rate | Mean LH Rate | Mean RH Rate |
|---|---|---|---|---|---|---|---|
| **Train** | 129 | 129 (100%) | 0 | 15,075 | 1.0000 (100%) | 0.2387 (23.9%) | 0.4264 (42.6%) |
| **Validation** | 28 | 28 (100%) | 0 | 3,042 | 1.0000 (100%) | 0.2561 (25.6%) | 0.5134 (51.3%) |
| **Test** | 30 | 30 (100%) | 0 | 3,240 | 1.0000 (100%) | 0.3326 (33.3%) | 0.3557 (35.6%) |
| **Total / Avg** | **187** | **187 (100%)** | **0** | **21,357** | **1.0000 (100%)** | **0.2564 (25.6%)** | **0.4281 (42.8%)** |

---

## 4. Class-by-Class Landmark Characteristics

The table below breaks down the frame counts and anatomical detection rates across all 10 classes:

| Class Label | Total Clips | Frame Range [Min-Max] | Average Frames | Pose Rate | Left Hand Rate | Right Hand Rate | Dominance / Movement Pattern |
|---|---|---|---|---|---|---|---|
| **food** | 17 | 76 – 177 | 132.0 | 100% | 2.0% | 32.3% | Primarily right-handed (feeding motion to mouth) |
| **hello** | 36 | 56 – 229 | 96.4 | 100% | 60.1% | 52.8% | Mixed dominance (both left and right hand wave) |
| **help** | 16 | 80 – 175 | 128.2 | 100% | 20.3% | 32.8% | Two-handed lifting support sign |
| **no** | 17 | 62 – 200 | 124.5 | 100% | 3.1% | 33.6% | Single-handed right index finger side-shake |
| **please** | 18 | 75 – 194 | 133.9 | 100% | 3.4% | 38.2% | Single flat hand outward/rotational gesture |
| **sorry** | 6 | 88 – 106 | 98.5 | 100% | 40.2% | 55.2% | Mixed: fist circular motion & two-handed ear touch |
| **stop** | 3 | 76 – 130 | 111.3 | 100% | 12.1% | 40.6% | Forward-facing open palm |
| **thank you** | 39 | 62 – 168 | 98.2 | 100% | 45.3% | 56.2% | Two-handed base tap or chin-to-outward salute |
| **water** | 18 | 56 – 182 | 121.2 | 100% | 3.2% | 31.8% | Single-handed cupped palm to lips |
| **yes** | 17 | 79 – 175 | 125.2 | 100% | 4.2% | 32.5% | Single-handed right fist nod |

> **Note on Detection Rates:** Body pose detection achieved 100% across all 21,357 frames. For single-handed signs (`water`, `food`, `no`, `yes`, `please`), the non-dominant hand rests naturally at the waist/lap, producing expected low detection rates (2–4%) for the idle hand. For two-handed signs (`thank you`, `help`, `sorry`), detection rates reach 40–60% during active gesture phases.

---

## 5. Sample Landmark Visualizations

Visual inspection plots were generated for representative clips across different gesture topologies and upstream sources.

### 5.1 Visualization Assets Generated
Static 2D keyframe projections and hand spatial trajectories were saved in `docs/reports/figures/`:
1. `landmark_vis_hello.png`: [landmark_vis_hello.png](file:///D:/Signova%20AI/docs/reports/figures/landmark_vis_hello.png)
   - Demonstrates single-hand wave trajectory with clear wrist oscillation in upper torso quadrant.
2. `landmark_vis_thank_you.png`: [landmark_vis_thank_you.png](file:///D:/Signova%20AI/docs/reports/figures/landmark_vis_thank_you.png)
   - Illustrates coordinated two-hand trajectories converging from chin/chest towards the observer.
3. `landmark_vis_sorry.png`: [landmark_vis_sorry.png](file:///D:/Signova%20AI/docs/reports/figures/landmark_vis_sorry.png)
   - Visualizes circular fist trajectory over central chest region.
4. `landmark_vis_stop.png`: [landmark_vis_stop.png](file:///D:/Signova%20AI/docs/reports/figures/landmark_vis_stop.png)
   - Visualizes abrupt deceleration and spatial fixation of forward open palm.

---

## 6. Dataset Quality Checks and Integrity Sign-Off

The following automated and manual quality checks were executed on the extracted assets:

1. **Completeness Check:** All 187 approved clips from F-019 were attempted, and exactly 187 `.npz` files exist in their respective split directories.
2. **Dimension Uniformity:** 100% of `.npz` files verified to contain `landmarks` with shape $(T, 75, 4)$ and `spatial_features` with shape $(T, 300)$.
3. **No Empty Sequences:** Sequence lengths range from 56 to 229 frames (mean: 114.2 frames). Zero empty or zero-frame extractions.
4. **Data Leakage Check:**
   - 129 train files in `datasets/landmarks/train/`
   - 28 validation files in `datasets/landmarks/validation/`
   - 30 test files in `datasets/landmarks/test/`
   - $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
5. **No Data Augmentation Applied:** Extracted landmark arrays contain strictly unaugmented, authentic video observations.
6. **Git Hygiene:** `datasets/landmarks/` and all binary `.npz` files are excluded from Git version control via `.gitignore` (`datasets/*`).

---

## 7. Recommendations and Next Steps for F-021 (Model Training)

1. **Temporal Resampling / Padding:**
   - Sequence lengths vary between 56 and 229 frames. Preprocessing in F-021 should standardize sequences to a uniform frame count (e.g., $T = 64$ or $T = 96$) via uniform temporal interpolation.
2. **Feature Input Vector:**
   - Machine learning classifiers (e.g., BiLSTM, 1D-CNN, or Transformer encoder) can consume `spatial_features` `(T, 300)` directly or append velocity motion vectors $\Delta x_t = x_t - x_{t-1}$ to yield 600-dimensional representations using `ai.preprocessing.pose_preprocessor.add_motion_features`.
3. **Training-Time Augmentation:**
   - For classes `stop` (1 train clip) and `sorry` (3 train clips), apply the F-019 augmentation protocol during DataLoader batching: temporal scaling ($0.9\times$ to $1.1\times$) and Gaussian landmark jitter ($\sigma = 0.003$).
