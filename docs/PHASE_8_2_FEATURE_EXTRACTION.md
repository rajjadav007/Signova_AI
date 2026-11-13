# Phase 8.2: ISLVT Feature Extraction Pipeline

**Project:** Signova AI — Continuous Indian Sign Language Recognition  
**Phase:** 8.2 — Feature Extraction and Dataset Preparation  
**Target Vocabulary:** `I`, `GO`, `EAT`, `FOOD` (4-sign approved experimental subset)  
**Date:** November 13, 2026  
**Auditor / Engineer:** DeepMind Agent (Antigravity)  
**Status:** COMPLETE — All 32 recordings extracted, structured, and validated (8/8 tests passed)  

---

## Executive Summary

Phase 8.2 implemented an end-to-end MediaPipe feature extraction pipeline for the selected 4-sign experimental vocabulary (`I`, `GO`, `EAT`, `FOOD`) from the ISLVT dataset. 

Using the project's native `ModernMediaPipeExtractor` (Google MediaPipe Tasks API: `PoseLandmarker` + `HandLandmarker`), **32 video recordings** across **16 target sentences** (total **6,237 frames**) were processed in **271.24 seconds** (~23 FPS) with **100% success rate (0 failures)**.

Extracted features were serialized into structured `.npz` archives and placed in a dedicated, isolated directory (`datasets/islvt_continuous/`), leaving production checkpoints and isolated-sign datasets completely untouched. An automated test suite (`tests/test_islvt_features.py`) verified 100% data integrity, tensor geometry, absence of NaNs, and leakage-safe camera splits.

---

## 1. Videos Successfully Processed

All **32 video files** associated with the 16 target sentences were successfully decoded, preprocessed, and extracted. Zero videos failed or raised unhandled exceptions.

- **Total Videos Attempted:** 32
- **Total Videos Succeeded:** **32 (100.0%)**
- **Total Frames Extracted:** **6,237 frames**
- **Total Extraction Wall-Clock Time:** 271.24 seconds
- **Mean Extraction Speed:** 22.99 FPS
- **Video Format:** QuickTime `.MOV` ($1920 \times 1080$ @ 25.0 FPS)

### Inventory of Processed Videos

| Sentence ID | Primary Gloss Sequence | Target Signs | Video Filename | Camera View | Split | Frames | Duration | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **4** | `mother food cook` | `food` | `4 mother food cook  - Copy-CokUY0.MOV` | Side | Train | 180 | 7.20 s | OK |
| **4** | `mother food cook` | `food` | `4 mother food cook  - Copy.MOV` | Side | Train | 180 | 7.20 s | OK |
| **11** | `grand mother market go` | `go` | `11 grand mother market go  (2).MOV` | Side | Train | 207 | 8.28 s | OK |
| **11** | `grand mother market go` | `go` | `11 grand mother market go .MOV` | Front | Train | 219 | 8.76 s | OK |
| **12** | `grandmother market go will` | `go` | `12 grandmother market go will  (2).MOV` | Side | Train | 291 | 11.64 s | OK |
| **12** | `grandmother market go will` | `go` | `12 grandmother market go will .MOV` | Front | Train | 189 | 7.56 s | OK |
| **13** | `before grand mother market go finish` | `go` | `13 before grand mother market go finish  (2).MOV` | Side | Train | 243 | 9.72 s | OK |
| **13** | `before grand mother market go finish` | `go` | `13 before grand mother market go finish .MOV` | Front | Train | 210 | 8.40 s | OK |
| **17** | `we place see go  which` | `go` | `17 we place see go which  (2).MOV` | Side | Train | 225 | 9.00 s | OK |
| **17** | `we place see go  which` | `go` | `17 we place see go which .MOV` | Front | Train | 201 | 8.04 s | OK |
| **18** | `i home reach very late` | `i` | `18 i home reach very late  (2).MOV` | Side | Train | 192 | 7.68 s | OK |
| **18** | `i home reach very late` | `i` | `18 i home reach very late .MOV` | Front | Train | 186 | 7.44 s | OK |
| **20** | `this day i inform you` | `i` | `20 this day i inform you  (2).MOV` | Side | Train | 150 | 6.00 s | OK |
| **20** | `this day i inform you` | `i` | `20 this day i inform you .MOV` | Front | Train | 144 | 5.76 s | OK |
| **21** | `we night go out` | `go` | `21 we night go out  (2).MOV` | Side | Test | 162 | 6.48 s | OK |
| **21** | `we night go out` | `go` | `21 we night go out .MOV` | Front | Test | 159 | 6.36 s | OK |
| **26** | `first him eat after i eat` | `eat`, `i` | `26 first him eat after i eat  (2).MOV` | Side | Train | 228 | 9.12 s | OK |
| **26** | `first him eat after i eat` | `eat`, `i` | `26 first him eat after i eat .MOV` | Front | Train | 243 | 9.72 s | OK |
| **31** | `food taste good` | `food` | `31 food taste good  (2).MOV` | Side | Train | 141 | 5.64 s | OK |
| **31** | `food taste good` | `food` | `31 food taste good .MOV` | Front | Train | 138 | 5.52 s | OK |
| **34** | `pizza food healthy not` | `food` | `34 pizza food healthy not  (2).MOV` | Side | Test | 213 | 8.52 s | OK |
| **34** | `pizza food healthy not` | `food` | `34 pizza food healthy not .MOV` | Front | Test | 213 | 8.52 s | OK |
| **44** | `i black coffe like` | `i` | `44 i black coffe like  (2).MOV` | Side | Train | 147 | 5.88 s | OK |
| **44** | `i black coffe like` | `i` | `44 i black coffe like .MOV` | Front | Train | 171 | 6.84 s | OK |
| **50** | `i fruit fork eat` | `eat`, `i` | `50 i fruit fork eat  (2).MOV` | Side | Test | 189 | 7.56 s | OK |
| **50** | `i fruit fork eat` | `eat`, `i` | `50 i fruit fork eat .MOV` | Front | Test | 219 | 8.76 s | OK |
| **63** | `i her give ten rupees have` | `i` | `63 i her give ten rupees have   (2).MOV` | Side | Train | 246 | 9.84 s | OK |
| **63** | `i her give ten rupees have` | `i` | `63 i her give ten rupees have  .MOV` | Front | Train | 243 | 9.72 s | OK |
| **67** | `i paisa five have` | `i` | `67  i paisa five have   (2).MOV` | Side | Train | 186 | 7.44 s | OK |
| **67** | `i paisa five have` | `i` | `67  i paisa five have  .MOV` | Front | Train | 183 | 7.32 s | OK |
| **72** | `i  bathroom go` | `go`, `i` | `72 i  bathroom go   (2).MOV` | Side | Test | 177 | 7.08 s | OK |
| **72** | `i  bathroom go` | `go`, `i` | `72 i  bathroom go  .MOV` | Front | Test | 213 | 8.52 s | OK |

---

## 2. Videos That Failed and Why

- **Total Extraction Failures:** **0 (Zero)**
- **Corrupted Video Streams:** **0**
- **Decoder Exceptions:** **0**

### Notable Dataset Defect Handled
- **Sentence ID 4:** The raw dataset directory contains two binary-identical files (`4 mother food cook  - Copy.MOV` and `4 mother food cook  - Copy-CokUY0.MOV`). Both recordings capture the **side profile view**; the front-view take is missing from the original ISLVT distribution. The pipeline extracted both takes safely and logged the pairing anomaly in the manifest under `pair_status: "duplicate_side_take_no_front"`.

---

## 3. Annotation Mapping Results

The 16 sentences map directly to the official ISLVT Excel workbook (`marathi-English_ISLgloss1.xlsx`):

| Sentence ID | English Translation | Marathi Reference Sentence | ISL Gloss Sequence | Target Tokens Found |
| :---: | :--- | :--- | :--- | :--- |
| **4** | mother cooks food | आई अन्न शिजवते | `mother food cook` | `food` |
| **11** | Grand mother is going to market | आजी बाजारात जाणार आहेत | `grand mother market go` | `go` |
| **12** | Grandmother will go the market | आजी बाजारात जाईल | `grandmother market go will` | `go` |
| **13** | grandmother had gone to market | आजी बाजारात गेली होती | `before grand mother market go finish` | `go` |
| **17** | which places can we go to see | आपण कोणत्या ठिकाणी जाऊन पाहू शकतो | `we place see go  which` | `go` |
| **18** | i came home very late | मी खूप उशीरा घरी आलो | `i home reach very late` | `i` |
| **20** | i will let you know in a day | मी तुम्हाला एका दिवसात कळवेल | `this day i inform you` | `i` |
| **21** | we are going out in night | आम्ही रात्री बाहेर जात आहोत | `we night go out` | `go` |
| **26** | i will eat after him | मी त्याच्या मागे खाईन | `first him eat after i eat` | `eat`, `i` |
| **31** | food is tastty | अन्न चवदार आहे | `food taste good` | `food` |
| **34** | pizza is not a healthy food | पिझ्झा हे निरोगी अन्न नाही | `pizza food healthy not` | `food` |
| **44** | i like black coffee | मला ब्लॅक कॉफी आवडते | `i black coffe like` | `i` |
| **50** | i eat fruits with fork | मी काट्याने फ्युट्स खातो | `i fruit fork eat` | `eat`, `i` |
| **63** | i have to give her ten rupees | मला तिला दहा रुपये द्यावे लागतील. | `i her give ten rupees have` | `i` |
| **67** | i have five paisa | नाणी खूप आहेत *(corrupted row)* | `i paisa five have` | `i` |
| **72** | i go to bathroom | मी बाथरूमला जातो | `i  bathroom go` | `go`, `i` |

### Target Token Frequency
- **`I`**: 8 sentences (10 token occurrences: ID 26 contains 2 occurrences)
- **`GO`**: 6 sentences (6 token occurrences)
- **`FOOD`**: 3 sentences (3 token occurrences)
- **`EAT`**: 2 sentences (3 token occurrences: ID 26 contains 2 occurrences)
- **Timestamp Status:** No sign-level timestamps exist in the dataset. Feature sequences represent complete unsegmented utterances.

---

## 4. Extracted Feature Dimensions

Features are extracted strictly matching the project's continuous iSign specification (`F-014`, `F-015`):

```text
Each frame t contains:
  Pose Landmarks:       33 points x 4 channels [x, y, z, visibility]  = 132 values
  Left Hand Landmarks:  21 points x 4 channels [x, y, z, confidence]  =  84 values
  Right Hand Landmarks: 21 points x 4 channels [x, y, z, confidence]  =  84 values
  --------------------------------------------------------------------------------
  Total per Frame:      75 points x 4 channels                         = 300 values
```

### Serialized Tensor Specifications

| Array Key | Tensor Shape | Data Type | Description |
| :--- | :--- | :---: | :--- |
| **`landmarks`** | $(T, 75, 4)$ | `float32` | Explicit 3D spatial coordinates + visibility/confidence channel |
| **`spatial_features`** | $(T, 300)$ | `float32` | Flattened spatial representation ready for dense/recurrent encoders |
| **`timestamps_ms`** | $(T,)$ | `int64` | Strictly monotonic timestamp sequence (step: $40\text{ ms}$ at 25 FPS) |
| **`frame_indices`** | $(T,)$ | `int32` | Monotonic 0-indexed integer frame indices $[0, 1, \dots, T-1]$ |
| **`detection_mask`** | $(T, 3)$ | `bool` | Boolean flags indicating $[pose\_present, left\_hand\_present, right\_hand\_present]$ |
| **`metadata_json`** | Scalar string | `UTF-8` | Comprehensive JSON payload describing labels, video attributes, and rates |

---

## 5. Missing Landmark Statistics

MediaPipe Tasks API was executed in `VIDEO` mode with monotonic timestamps. To satisfy the prompt's restriction (*"Do not silently substitute missing values with fabricated landmarks"*), `hand_missing_strategy="zero"` was enforced: unobserved hands are encoded as exact zeros (`0.0`).

| Metric | Minimum | Mean | Maximum | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Pose Detection Rate** | 100.0% | **100.0%** | 100.0% | Full body detected across all 6,237 frames |
| **Mean Pose Visibility** | 0.651 | **0.712** | 0.768 | High visibility confidence under green backdrop |
| **Left Hand Detection Rate** | 41.2% | **74.20%** | 100.0% | Hands occasionally drop when resting at waist |
| **Right Hand Detection Rate**| 68.3% | **86.18%** | 100.0% | Dominant signing hand; higher detection consistency |
| **Missing Landmark Rate** | 3.5% | **11.10%** | 20.3% | Fraction of landmark channels missing per sequence |
| **Corrupted Coordinates** | 0 | **0** | 0 | Verified 0 `NaN`, 0 `Inf`, no corrupted floats |

---

## 6. Camera-Pair Validation and Leakage Prevention

### Validation Protocol
Because the dataset features **a single signer**, splitting camera angles of the same performance across Train and Test constitutes severe visual data leakage.

1. **Camera View Assignment:**
   - **Front-View Takes:** 15 videos
   - **Side-View Takes:** 17 videos (includes the 2 side duplicate takes of Sentence 4)
2. **Leakage Gate:**
   - Both camera takes of any given `sentence_id` are strictly grouped together into the same split.
3. **Partition Stratification:**
   - **Train Split (75% / 12 Sentences / 24 Videos):** IDs 4, 11, 12, 13, 17, 18, 20, 26, 31, 44, 63, 67
   - **Test Split (25% / 4 Sentences / 8 Videos):** IDs 21, 34, 50, 72

### Target Vocabulary Split Balance

| Target Sign | Train Sentences | Train Video Takes | Test Sentences | Test Video Takes | Both Splits Covered? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`I`** | 6 (IDs 18, 20, 26, 44, 63, 67) | 12 | 2 (IDs 50, 72) | 4 | **YES** |
| **`GO`** | 4 (IDs 11, 12, 13, 17) | 8 | 2 (IDs 21, 72) | 4 | **YES** |
| **`EAT`** | 1 (ID 26, 2 occurrences) | 2 | 1 (ID 50) | 2 | **YES** |
| **`FOOD`** | 2 (IDs 4, 31) | 4 | 1 (ID 34) | 2 | **YES** |

All 4 target classes are present in both the training set and the hold-out test set.

---

## 7. Dataset Storage Structure

Extracted features and metadata are stored in an isolated, dedicated directory:

```text
D:\Projects\Signova AI\datasets\islvt_continuous\
├── features\
│   ├── train\                    (24 compressed .npz archives)
│   │   ├── 4_mother_food_cook____Copy.npz
│   │   ├── 4_mother_food_cook____Copy_CokUY0.npz
│   │   ├── 11_grand_mother_market_go___2_.npz
│   │   ├── 11_grand_mother_market_go_.npz
│   │   └── ...
│   └── test\                     (8 compressed .npz archives)
│       ├── 21_we_night_go_out___2_.npz
│       ├── 21_we_night_go_out_.npz
│       ├── 34_pizza_food_healthy_not___2_.npz
│       ├── 34_pizza_food_healthy_not_.npz
│       ├── 50_i_fruit_fork_eat___2_.npz
│       ├── 50_i_fruit_fork_eat_.npz
│       ├── 72_i__bathroom_go____2_.npz
│       └── 72_i__bathroom_go__.npz
├── manifest.csv                  (32-row CSV containing complete video & label metadata)
├── manifest.json                 (Full JSON records with detection rates and metrics)
├── splits.json                   (Formal train/test partition specification)
└── feature_summary.json          (Aggregate statistics, extraction time, and detector parameters)
```

---

## 8. Remaining Limitations

1. **Single-Signer Bias:**
   All 32 recordings were performed by the same individual. Models trained on these features will learn signer-dependent characteristics; performance on unseen signers cannot be evaluated on this dataset alone.
2. **Absence of Sign-Level Timestamps:**
   The dataset provides only sentence-level gloss strings (`first him eat after i eat`), with no start/stop frame boundaries for individual signs. Sequence training must rely on Connectionist Temporal Classification (CTC) or sequence-to-sequence attention to infer alignments.
3. **Small Sample Size:**
   With 24 training sequences and 8 test sequences, deep autoregressive models (e.g., standard Transformers) will quickly overfit. Architectures must use heavy regularization, lightweight recurrent layers (e.g., Bi-GRU), or pre-trained sign representations.

---

## 9. Suitability for Phase 8.3 Model Development

| Criterion | Evaluation | Assessment |
| :--- | :---: | :--- |
| **Feature Extraction Quality** | **PASSED** | 100% pose detection, clean hand coordinates, strictly zero NaNs/Infs. |
| **Target Vocabulary Coverage** | **PASSED** | All 4 signs (`I`, `GO`, `EAT`, `FOOD`) are represented in both Train and Test splits. |
| **Tensor Geometry & Formatting** | **PASSED** | Uniform $(T, 75, 4)$ and $(T, 300)$ representations ready for PyTorch / TensorFlow dataloaders. |
| **Pipeline Verification** | **PASSED** | 8 out of 8 unit and integration tests passed in `tests/test_islvt_features.py`. |
| **Verdict for Phase 8.3** | **SUITABLE** | **Approved for Phase 8.3 model development** (Bi-GRU + CTC or lightweight sequence classifier on the 4-sign vocabulary). |

---
*Phase 8.2 feature extraction pipeline complete and verified.*
