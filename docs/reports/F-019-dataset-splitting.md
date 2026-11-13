# F-019: Dataset Quality Checks and Signer-Independent Splitting

**Date:** 2026-11-12  
**Dataset Source:** [vidit031/isl-isolated-40words](https://huggingface.co/datasets/vidit031/isl-isolated-40words)  
**Manifest Used:** `datasets/manifests/isl_10class_manifest.csv`  
**Generated Splits:**  
- `datasets/splits/train.csv`  
- `datasets/splits/validation.csv`  
- `datasets/splits/test.csv`  
- `datasets/splits/split_summary.json`  
**Prior Reports:**  
- [F-018.1-dataset-inspection.md](file:///D:/Signova%20AI/docs/reports/F-018.1-dataset-inspection.md)  
- [F-018.2-class-validation.md](file:///D:/Signova%20AI/docs/reports/F-018.2-class-validation.md)  
**Status:** Complete & Integrity Verified  

---

## 1. Executive Summary

This report establishes the official, leakage-free 3-way split (Train / Validation / Test) for Signova AI's 10-class isolated Indian Sign Language (ISL) MVP model. 

- **Total Approved Clips:** 187 (1 flagged `stop` sample excluded).
- **Split Distribution:**
  - **Train:** 129 clips (68.98%)
  - **Validation:** 28 clips (14.97%)
  - **Test:** 30 clips (16.04%)
- **Signer Independence:** Strict partitioning across human signers. Zero signer identity leakage between splits.
- **Class Coverage:** 100% of the 10 target classes are represented in all three partitions (including low-volume classes `sorry` and `stop`).

---

## 2. Dataset and Signer Distribution

### 2.1 Upstream Dataset Contributions
| Dataset | Total Clips | Signer Architecture | Approved Clips in Split |
|---|---|---|---|
| **ISL500** | 120 | 15 distinct human users (`User001` - `User015`), 8 target words each | 120 |
| **INCLUDE** | 42 | 41 distinct capture session IDs (21 `hello`, 21 `thank you`) | 42 |
| **CISLR** | 22 | 9 semantic domain category labels over distinct YouTube clips | 22 |
| **ISLRTC Dictionary** | 3 | 1 standard reference signer (`islrtc_dict`) | 3 |
| **Total** | **187** | **66 recorded signer identifier tags** | **187** |

### 2.2 Investigation of "Signer 16"
The prompt requested accounting for "signer 16" in relation to ISL500 signers 1–12 and 13–15:
- **Investigation Finding:** In the ISL500 dataset, exactly 15 distinct signers exist (`User001` through `User015`). There is no `User016`.
- **Origin of "16":** In F-018.2, classes such as `water`, `food`, `please`, `yes`, and `no` were reported as having "16 signers". This was the aggregate count of the 15 ISL500 human signers plus 1 distinct CISLR category tag (e.g., `foods and drinks`, `substance`, `gesture`). 
- **Accounting:** `User001`–`User011` are assigned to Train, `User012`–`User013` to Validation, `User014`–`User015` to Test. CISLR clips are partitioned by distinct source video IDs.

---

## 3. Signer-Independent Split Strategy

### 3.1 ISL500 Allocation (120 clips)
- **Train (88 clips):** `User001`, `User002`, `User003`, `User004`, `User005`, `User006`, `User007`, `User008`, `User009`, `User010`, `User011` (11 users × 8 classes = 88 clips; 73.3% of ISL500).
- **Validation (16 clips):** `User012`, `User013` (2 users × 8 classes = 16 clips; 13.3% of ISL500).
- **Test (16 clips):** `User014`, `User015` (2 users × 8 classes = 16 clips; 13.3% of ISL500).

### 3.2 INCLUDE Allocation (42 clips)
- Signer `include_MVI_9984` (has 1 `hello` and 1 `thank you`) is isolated in **Train**.
- Remaining 40 single-clip signers are distributed:
  - **Train:** 28 signers (14 `hello`, 14 `thank you`)
  - **Validation:** 6 signers (3 `hello`, 3 `thank you`)
  - **Test:** 6 signers (3 `hello`, 3 `thank you`)

### 3.3 ISLRTC Dictionary Allocation (3 clips)
- Reference signer `islrtc_dict` (2 `sorry` clips, 1 `stop` clip) is assigned exclusively to **Test**.
- This ensures test evaluation contains standardized national dictionary benchmarks for both low-volume classes.

### 3.4 CISLR Allocation (22 clips)
- In CISLR, the metadata `signer` field contains domain category strings (`stage`, `greeting`, etc.). Each clip originates from an independent YouTube video.
- Clips are partitioned by source YouTube video IDs:
  - `stop` (2 clips): 1 to Train (`FEi50jcTxxI`), 1 to Validation (`FHwjyQeLNuY`).
  - `sorry` (4 clips): 3 to Train (`0sE5EMDNsDo`, `KXnSguJGdN8`, `v7LOMsrEiVI`), 1 to Validation (`TiPPYhh323E`).
  - Remaining 16 clips distributed to preserve balance across splits.

---

## 4. Class Coverage and Split Matrix

| Target Sign Class | Train (clips) | Validation (clips) | Test (clips) | Total Approved | Split Coverage Status |
|---|---|---|---|---|---|
| **hello** | 26 | 5 | 5 | 36 | Full 3-way coverage |
| **thank you** | 27 | 5 | 7 | 39 | Full 3-way coverage |
| **sorry** | 3 | 1 | 2 | 6 | Full 3-way coverage (ISLRTC in Test) |
| **please** | 12 | 3 | 3 | 18 | Full 3-way coverage |
| **yes** | 12 | 2 | 3 | 17 | Full 3-way coverage |
| **no** | 12 | 3 | 2 | 17 | Full 3-way coverage |
| **help** | 12 | 2 | 2 | 16 | Full 3-way coverage |
| **stop** | 1 | 1 | 1 | 3 | Full 3-way coverage (1 per split) |
| **water** | 12 | 3 | 3 | 18 | Full 3-way coverage |
| **food** | 12 | 3 | 2 | 17 | Full 3-way coverage |
| **Total Clips** | **129 (68.98%)** | **28 (14.97%)** | **30 (16.04%)** | **187 (100%)** | **All 10 classes covered in all splits** |

---

## 5. Dataset Integrity and Leakage Verification

All automated verification checks succeeded prior to sign-off:

1. **Zero Video Overlap:**
   - $\text{Train} \cap \text{Validation} = \emptyset$
   - $\text{Train} \cap \text{Test} = \emptyset$
   - $\text{Validation} \cap \text{Test} = \emptyset$
   - Verified across all 187 distinct `video_id` keys.
2. **Zero Signer Identity Leakage:**
   - ISL500 users strictly partitioned: Train `{User001-User011}`, Val `{User012-User013}`, Test `{User014-User015}`. No intersection.
   - ISLRTC dictionary signer (`islrtc_dict`) strictly isolated to Test.
   - INCLUDE multi-clip signer (`include_MVI_9984`) kept intact in Train.
3. **Exclusion of Flagged Clip:**
   - Flagged sample `stop__ISLRTC_dictionary__00631__Cease_Discontinue_Halt_Stop.mp4` verified excluded from all 3 split CSVs.
4. **Physical File Verification:**
   - 100% of the 187 referenced video paths physically exist on disk at `D:\Signova AI\dataset\isl-isolated-40words`.
5. **Class Label Consistency:**
   - All records retain normalized class labels matching the target 10-class vocabulary (`thank you` preserved, folder `thank_you` mapped).

---

## 6. Training Data Augmentation Policy (F-020 Guidance)

Data augmentation must only be executed during training. Validation and test sets must remain strictly pristine to guarantee realistic generalisation metrics.

### 6.1 Permissible Training Augmentations
1. **Temporal Scaling (Speed Perturbation):**
   - Resample video frames or landmarks by $\pm 10\%$ ($0.9\times$ to $1.1\times$).
   - Simulates natural signer cadence differences without distorting gesture morphology.
2. **Spatial Landmark Scaling and Translation:**
   - Bounded coordinate jitter: Add Gaussian noise with standard deviation $\sigma \le 0.005$ to normalized coordinates.
   - Minor global scale variations ($\pm 5\%$) to simulate distance variations from the webcam.
3. **Synthetic Landmark Dropout:**
   - Randomly mask $1$–$2\%$ of non-critical face landmarks to improve robustness against partial occlusion.

### 6.2 Prohibited Augmentations
- **No Horizontal Flipping (Mirroring):** In Indian Sign Language, lateral hand dominance and directional sign semantics can be corrupted by blind left-right mirroring.
- **No Validation or Test Augmentation:** Validation and testing metrics must evaluate pure recorded data.
- **Integrity Notice:** Augmentation mitigates low-sample variance for `stop` and `sorry`, but does not substitute for real native signer diversity.

---

## 7. Known Limitations and Bottlenecks

1. **Severe Low Volume for `stop`:**
   - `stop` contains only 1 sample in Train, 1 in Validation, and 1 in Test.
   - While mathematically sufficient for multi-class classification and pipeline testing, a single validation error causes an apparent 100% drop in class recall.
2. **Limited Variety for `sorry`:**
   - 3 clips in Train, 1 in Validation, 2 in Test.
3. **Signer Representation in CISLR:**
   - CISLR metadata uses broad semantic categories rather than individual signer IDs. While YouTube source separation prevents clip leakage, true anatomical independence cannot be mathematically guaranteed for CISLR.

---

## 8. Recommendations for Next Phase (F-020)

1. **Input Manifests:**
   - Direct MediaPipe feature extraction to consume `datasets/splits/train.csv`, `validation.csv`, and `test.csv`.
2. **Evaluation Metric Protocol:**
   - Because `stop` and `sorry` have small sample counts, report both:
     - Macro-averaged F1-score across all 10 classes.
     - Per-class precision, recall, and confusion matrix.
     - 8-class core accuracy (excluding `stop` and `sorry`) alongside full 10-class accuracy.
3. **Priority Recording Action:**
   - Before model finalization, record 10 fresh isolated clips for `stop` and `sorry` from new signers, placing them into train/val splits following this architecture.
