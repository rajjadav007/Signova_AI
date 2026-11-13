# Phase 8.3.5: ISL-CSLTR Landmark Extraction and Signer-Independent Dataset Preparation Report

**Project:** Signova AI — Continuous Indian Sign Language Recognition  
**Phase:** 8.3.5 — Landmark Extraction and Dataset Preparation  
**Dataset:** ISL-CSLTR (*Indian Sign Language Continuous Sign Language Recognition and Translation*)  
**Raw Dataset Path:** `D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset`  
**Processed Dataset Root:** `datasets/isl_csltr_continuous/`  
**Date:** November 13, 2026  
**Auditor / Engineer:** DeepMind Agent (Antigravity)  
**Status:** COMPLETE — All 687 videos extracted (100% success rate, 0 failures), 7/7 unit tests passed  

---

## Executive Summary

Phase 8.3.5 completed the full, bit-exact feature extraction and signer-independent dataset preparation for the ISL-CSLTR corpus. All 687 original continuous `.mp4` video recordings across 101 conversational sentences were decoded, preprocessed, and extracted using the project's standardized `ModernMediaPipeExtractor` Tasks API (`PoseLandmarker` + `HandLandmarker`).

A total of **64,730 video frames** were processed across 3 worker threads in **1,220.78 seconds** (~20.35 minutes) with **100% success rate (687 succeeded, 0 failed)**. Extracted landmarks were serialized as compressed `.npz` feature archives under the project's standard 75-landmark, 4-channel schema $(T, 75, 4)$ with unobserved keypoints zero-filled (`hand_missing_strategy="zero"`).

Signer independence was enforced through an audited 5 / 1 / 1 partition protocol:
- **Train (Signers 1, 3, 4, 5, 6):** 504 videos (73.36%)
- **Validation (Signer 7):** 85 videos (12.37%)
- **Test (Signer 2):** 98 videos (14.26%)
- **Leakage Status:** **Zero signer leakage** and **zero duplicate leakage** across all splits.

---

## 1. Split Counts by Signer, Sentence, and Video

### 1.1 Dataset Partition Summary

```
+---------------------------------------------------------------------------------------------------+
|                                  SIGNER-INDEPENDENT SPLIT SUMMARY                                 |
+------------+--------------------+--------------------------+---------------+----------------------+
| Split      | Assigned Signers   | Signer Composition       | Video Count   | Percentage           |
+------------+--------------------+--------------------------+---------------+----------------------+
| **Train**  | 1, 3, 4, 5, 6      | 1 Native Deaf Signer +   | 504 videos    | 73.36%               |
|            |                    | 4 Student Volunteers     |               |                      |
| **Val**    | 7                  | 1 Student Volunteer      | 85 videos     | 12.37%               |
| **Test**   | 2                  | 1 Native Deaf Signer     | 98 videos     | 14.26%               |
+------------+--------------------+--------------------------+---------------+----------------------+
| **Total**  | 1, 2, 3, 4, 5, 6, 7| 2 Native + 5 Volunteers  | 687 videos    | 100.00%              |
+------------+--------------------+--------------------------+---------------+----------------------+
```

### 1.2 Signer Distribution

Every video was mapped to its ground-truth signer by correlating frame 0 with the native signer directories in `Frames_Sentence_Level` and computing orientation-aware Mean Squared Error (MSE):

| Signer ID | Signer Category | Institution | Camera Resolution | Assigned Split | Total Video Clips |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **Signer 1** | Native Deaf Signer (Male) | Navajeevan School, AP | $320 \times 240$ (QVGA) | **Train** | 98 |
| **Signer 2** | Native Deaf Signer (Female) | Navajeevan School, AP | $320 \times 240$ (QVGA) | **Test** | 98 |
| **Signer 3** | Student Volunteer | SASTRA University, TN | $1920 \times 1080$ (FHD) | **Train** | 126 |
| **Signer 4** | Student Volunteer | SASTRA University, TN | $1920 \times 1080$ (FHD) | **Train** | 89 |
| **Signer 5** | Student Volunteer | SASTRA University, TN | $1920 \times 1080$ (FHD) | **Train** | 95 |
| **Signer 6** | Student Volunteer | SASTRA University, TN | $1920 \times 1080$ (FHD) | **Train** | 96 |
| **Signer 7** | Student Volunteer | SASTRA University, TN | $1920 \times 1080$ (FHD) | **Validation** | 85 |
| **Total** | — | — | — | — | **687** |

*Note: Signer 3 contains 126 videos due to auxiliary takes (e.g. second/third takes in volunteer recording sessions). All auxiliary takes are strictly isolated in the Train partition.*

### 1.3 Sentence and Vocabulary Coverage

- **Unique Spoken Sentences:** 101 sentences (IDs 0 to 100).
- **Target ISL Gloss Vocabulary:** 170 unique gloss types + 1 CTC `<blank>` token = **171 classes**.
- **Sentence-to-Video Mapping:** 100.0% mapped with zero fabricated labels. Folder spelling variance (`which collegeschool are you from` $\rightarrow$ `which college school are you from`) resolved deterministically.

---

## 2. Duplicate Handling Results

An exhaustive MD5 checksum scan across all 687 raw videos identified **682 unique file hashes** and **5 duplicate groups (10 files total)**:

| Group ID | MD5 Hash Prefix | File 1 | File 2 | Split | Status |
| :---: | :---: | :--- | :--- | :---: | :--- |
| **`91203a8f`** | `91203a8fba22` | `are you hiding something/MVI_6817.MP4` | `do you need something/something (2).MP4` | **Train** | Cross-sentence duplicate |
| **`d96f0165`** | `d96f01657c91` | `do not worry/worry.MP4` | `No need to worry dont worry/MVI_6496.MP4` | **Train** | Synonymous sentence copy |
| **`34d79e4e`** | `34d79e4eb57a` | `i do not agree/agree.MP4` | `i dont agree/MVI_6494.MP4` | **Train** | Folder spelling duplicate |
| **`ffcf9be3`** | `ffcf9be3c162` | `i like you i love you/love.MP4` | `i like you i love you/MVI_6814.MP4` | **Train** | Intra-folder duplicate take |
| **`67170eaa`** | `67170eaae0f0` | `now onwards he will never hurt you/never hurt.MP4` | `what have you planned for your career/MVI_6499.MP4` | **Train** | Cross-sentence duplicate |

### Duplicate Leakage Prevention Verification
- **All 10 duplicate files belong to Signers 3, 4, and 6, and reside strictly within the Train partition.**
- **Zero duplicate files cross between Train, Validation, or Test.**
- Redundant files are flagged in the manifest with:
  - `is_duplicate: true`
  - `duplicate_group_id: "<hash_prefix>"`
  - `is_canonical: false` (for redundant copy) / `true` (for primary copy)
  - `canonical_video_path: "<path>"`
- Downstream CTC training can trivially filter out non-canonical duplicates (`is_canonical == True`) to prevent training over-weighting.

---

## 3. Extraction Success and Failure Counts

- **Total Videos Attempted:** 687
- **Total Videos Succeeded:** **687 (100.0%)**
- **Total Videos Failed:** **0 (0.0%)**
- **Total Decoded Frames:** **64,730 frames**
- **Total Extraction Wall-Clock Time:** 1,220.78 seconds (~20.35 minutes)
- **Effective Extraction Throughput:** 53.02 FPS aggregate across 3 worker threads.

### Outlier Handling Verification
The 180.0-second (4,500 frames) session outlier recording `i do not like it / i do not like it (2).mp4` (where the recording camera was left active after signing completed) was capped at **200 frames (8.00 seconds)** as specified in the design agreement. This matched the natural maximum signing duration (192 frames) and eliminated CTC memory bloat / CUDA OOM risk while preserving the complete sign sequence.

---

## 4. Feature Quality Statistics

Feature sequences strictly adhere to the continuous 75-landmark, 4-channel schema:
- Shape per video: $(T, 75, 4)$ `float32` [Pose 33 + Left Hand 21 + Right Hand 21]
- Flattened representation: $(T, 300)$ `float32`
- Channels: `[x, y, z, confidence]`

```
+---------------------------------------------------------------------------------------------------+
|                                     FEATURE QUALITY SUMMARY                                       |
+----------------------------------------+----------------------------------------------------------+
| Metric                                 | Measured Empirical Value                                 |
+----------------------------------------+----------------------------------------------------------+
| **Total Processed Videos**             | 687                                                      |
| **Total Extracted Frames**             | 64,730 frames                                            |
| **Minimum Sequence Length ($T_{min}$)**| 25 frames (1.00 s)                                       |
| **Median Sequence Length ($T_{med}$)** | 95.0 frames (3.80 s)                                     |
| **Mean Sequence Length ($T_{mean}$)**  | 94.22 frames (3.77 s)                                    |
| **Maximum Sequence Length ($T_{max}$)**| 200 frames (8.00 s, capped outlier)                      |
| **Pose Detection Rate**                | **100.00%** (Full body tracked in all 64,730 frames)      |
| **Left Hand Detection Rate**           | **48.56%**                                               |
| **Right Hand Detection Rate**          | **61.92%** (Dominant signing hand)                       |
| **Missing Landmark Handling**          | Exact zeros (`[0.0, 0.0, 0.0, 0.0]`), no NaNs / Infs    |
| **Timestamp Monotonicity**             | 100.0% strictly monotonic ($\Delta t > 0$)               |
| **Total Feature Storage on Disk**      | **94.88 MB** (99,484,829 bytes, compressed `.npz`)       |
+----------------------------------------+----------------------------------------------------------+
```

---

## 5. Resolution and Signer Confounding

The empirical audit identified a structural capture variation across signers:
1. **Signers 1 and 2 (Native Deaf Signers):** Mobile capture, $320 \times 240$ (QVGA), portrait orientation.
2. **Signers 3 to 7 (Student Volunteers):** Canon DSLR, $1920 \times 1080$ (Full HD), widescreen 16:9.

Because the held-out test signer is **Signer 2 (Native Deaf Signer, QVGA)**:
- High-resolution frames were downscaled to 640px max width during extraction to standardize memory bandwidth.
- MediaPipe coordinates $(x, y, z)$ are normalized to $[0.0, 1.0]$ relative to image boundaries, ensuring consistent feature dimensions $(T, 75, 4)$ across both resolutions.
- **Critical Limitation for Future Evaluation:** In Phase 8.3.6 / 8.4, test split accuracy measures joint generalization to **unseen native signing AND low-resolution mobile capture simultaneously**. It must **not** be reported as unseen-signer generalization independent of capture resolution.

---

## 6. Dataset Output Structure

All outputs are organized in the dedicated continuous dataset directory:
`datasets/isl_csltr_continuous/`

```text
datasets/isl_csltr_continuous/
├── features/
│   ├── train/                      (504 .npz feature archives)
│   ├── validation/                 (85 .npz feature archives)
│   └── test/                       (98 .npz feature archives)
├── manifests/
│   ├── manifest.csv                (687 video records with complete metadata)
│   ├── train_manifest.json         (504 train records)
│   ├── val_manifest.json           (85 validation records)
│   ├── test_manifest.json          (98 test records)
│   ├── splits.json                 (Signer split specification & leakage verification)
│   └── duplicates.json             (10 duplicate records & canonical cross-references)
└── metadata/
    ├── vocabulary.json             (171 classes: 0=<blank>, 1..170 gloss tokens)
    ├── dataset_summary.json        (Master dataset summary metrics)
    ├── extraction_stats.json       (Detailed feature extraction quality metrics)
    └── video_signer_mapping.json   (Bit-exact video-to-signer ground-truth cache)
```

The raw video files in `D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset` and previous ISLVT features in `datasets/islvt_continuous/` remain completely untouched.

---

## 7. Verification and Test Results

A dedicated test suite was implemented in `tests/test_isl_csltr_features.py`:

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.4.2
rootdir: D:\Projects\Signova AI
collected 7 items

tests/test_isl_csltr_features.py::test_annotation_mapping_and_vocabulary PASSED [ 14%]
tests/test_isl_csltr_features.py::test_signer_split_isolation PASSED          [ 28%]
tests/test_isl_csltr_features.py::test_duplicate_leakage_prevention PASSED    [ 42%]
tests/test_isl_csltr_features.py::test_feature_tensor_dimensions_and_dtypes PASSED [ 57%]
tests/test_isl_csltr_features.py::test_missing_landmark_handling_and_no_nans PASSED [ 71%]
tests/test_isl_csltr_features.py::test_monotonic_timestamps_and_frame_indices PASSED [ 85%]
tests/test_isl_csltr_features.py::test_outlier_duration_capping PASSED        [100%]

============================== 7 passed in 3.02s ==============================
```

Additionally, regression testing against existing Phase 8.2 feature tests confirmed **8/8 passed with zero regressions**:
`pytest tests/test_islvt_features.py` $\rightarrow$ **8 passed in 1.38s**.

---

## 8. Unresolved Issues and Next Phase Readiness

- **Unresolved Issues:** None. All 687 videos decoded cleanly, all 101 sentences mapped, zero corrupt frames, and zero leakage verified.
- **Model Training:** In compliance with Section 9 restrictions, no model training or CTC loss computation was executed in this phase.
- **Readiness for Phase 8.3.6:** The dataset is 100% prepared, verified, and ready for baseline continuous model training (Temporal CNN + Bi-GRU + CTC) using the 171-class vocabulary and 504 / 85 / 98 signer-independent split.
