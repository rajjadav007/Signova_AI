# Phase 8.1: ISLVT Dataset Audit and Daily Vocabulary Selection

**Project:** Signova AI — Continuous Indian Sign Language Recognition  
**Audit Date:** November 13, 2026  
**Auditor:** Automated Diagnostic Suite (DeepMind Agent / Antigravity)  
**Dataset Path:** `D:\Signova AI\Indian Sign Language Video and Text dataset for sentences (ISLVT)`  
**Status:** Complete Audit  

---

## Executive Summary

An exhaustive technical audit of the downloaded **Indian Sign Language Video and Text dataset for sentences (ISLVT)** was performed. The dataset consists of **153 total files** (1 Excel annotation file, 152 Apple QuickTime `.MOV` video recordings, total size **2.67 GB / 2,733.33 MB**), created by Prachi P. Waghmare (Mendeley Data, DOI: `10.17632/98mzk82wbb.1`).

### Primary Verdict
- **Target 10-Sign Daily Conversation Demonstration (HELLO, YOU, HOW, WATER, FOOD, HELP, HOME, GO, WHERE, THANK YOU):** **NO-GO**
  - Three candidate concepts (**HELP**, **WHERE**, **THANK YOU**) **do not exist** anywhere in the dataset (0 occurrences).
  - Two candidate concepts (**HELLO**, **HOW**) occur in **exactly one sentence** (`hello you how?`, Sentence ID 1), providing zero test samples if placed in training.
  - Two candidate concepts (**HOME**, **WATER**) occur in only two sentences, with corrupted video pairing (only 1 front-view video available for WATER).
- **Initial 4-Sign Technical Experiment:** **CONDITIONAL GO**
  - Viable **only** if selecting high-frequency glosses actually present in the dataset (e.g., `I`, `GO`, `EAT`, `FOOD` / `NOT`), recognizing that total sequence volume is limited to 78 sentences performed by a single signer.
- **Signer Independence:** **UNSUPPORTED**
  - The dataset contains **exactly one signer** (single South Asian male in a black polo shirt on a green chroma backdrop). Signer-independent generalization cannot be evaluated on this dataset alone.

---

## 1. Dataset Directory Structure

The directory contains no subfolders. All files reside in the root directory:

```text
D:\Signova AI\Indian Sign Language Video and Text dataset for sentences (ISLVT)\
├── marathi-English_ISLgloss1.xlsx  (15,330 bytes, Excel Spreadsheet)
├── 1 hello you how .MOV            (10,387,750 bytes)
├── 1 hello you how 2 .MOV          (10,433,738 bytes)
...
└── 81 this comb  wooden.MOV        (17,443,132 bytes)
```

### File Inventory Summary

| Category | Metric | Verified Value | Notes |
| :--- | :--- | :--- | :--- |
| **Total Entries** | Count | **153 files** | 0 subdirectories |
| **Annotation Files** | Count | **1 file** | `marathi-English_ISLgloss1.xlsx` (.xlsx) |
| **Video Files** | Count | **152 files** | All `.MOV` (QuickTime format) |
| **Total Video Size** | Bytes / GB | **2,866,117,046 bytes (2.67 GB)** | Mean video size: 17.98 MB |
| **Corrupted Files** | Count | **0 files** | All 152 videos decoded cleanly via OpenCV |
| **Identical Duplicate Files** | Count | **5 files** | Accidental binary copies replacing opposite angle |
| **Sentence ID Span** | Integer range | **1 to 81** | Total valid sentence IDs = 78 |
| **Missing Sentence IDs** | Missing values | **27, 42, 78** | Absent in both Excel and video files |

---

## 2. Annotation Schema

The annotation workbook `marathi-English_ISLgloss1.xlsx` contains a single sheet named `Sheet1`. The first three rows (index 0, 1, 2) contain empty headers; the tabular header begins at row index 3 (Row 4 in 1-based indexing).

### Table Schema

| Column Name | Excel Header Text | Data Type | Non-Null Count | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`sr.no`** | `sr.no` | `Float64` / `Int` | 78 | Unique sentence identifier (values 1–81, skipping 27, 42, 78) |
| **`english`** | `English sentence` | `String` | 78 | English reference translation |
| **`marathi`** | `Marathi sentence ` | `String` | 78 | Marathi reference sentence (Devanagari script) |
| **`isl_gloss`** | `Indian sign language gloss`| `String` | 78 | ISL gloss transcription in Latin script |

### Notable Structural Findings
1. **Row 75 is Completely Empty:** In Excel row index 75 (between sr.no 77 and sr.no 79), all columns contain `NaN`.
2. **Missing Sentence IDs:**
   - `sr.no 27`: Missing between 26 (`first him eat after i eat`) and 28 (`you work finish`).
   - `sr.no 42`: Missing between 41 (`milk colour white`) and 43 (`eyes colour black`).
   - `sr.no 78`: Missing between 77 (`room in sun light`) and 79 (`your Teeth clean`).
3. **No Timestamps:** The annotations contain **zero sign-level timestamps** (start frame/time, end frame/time) and **zero sentence-level timestamps**. The dataset provides only utterance-level text strings.

---

## 3. Unique Gloss Vocabulary

Tokenization was performed across all 78 ISL gloss annotations by stripping punctuation (`?`, `!`, `.`, `,`) and normalizing whitespace. Hyphenated compound glosses (e.g., `vitamin-mineral`) were preserved as single semantic units.

- **Total Gloss Tokens:** 254 tokens across 78 sentences.
- **Average Sentence Length:** 3.26 glosses per sentence (min 2, max 7).
- **Unique Gloss Lexicon Size:** **177 unique glosses**.
- **Lexical Sparsity:**
  - **Singleton Glosses (Count = 1):** 137 glosses (**77.4%** of the entire vocabulary occurs only once).
  - **Doubleton Glosses (Count = 2):** 27 glosses (**15.3%** occurs twice).
  - **Repeated Glosses (Count >= 3):** 13 glosses (**7.3%** occurs three or more times).

---

## 4. Gloss Frequency Table

The table below lists all glosses appearing with a frequency $\ge 2$ across the entire 78-sentence corpus:

| Rank | Gloss Label | Frequency | Sentence IDs Where Gloss Appears | Example Sentence Context |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `i` | 8 | 18, 20, 26, 44, 50, 63, 67, 72 | `i bathroom go`, `i home reach very late` |
| 2 | `go` | 6 | 11, 12, 13, 17, 21, 72 | `grand mother market go`, `i bathroom go` |
| 3 | `this` | 5 | 19, 20, 22, 24, 81 | `this work my not`, `this comb wooden` |
| 4 | `not` | 4 | 19, 24, 34, 45 | `she money not`, `pizza food healthy not` |
| 5 | `have` | 4 | 46, 52, 63, 67 | `cereals protein have`, `i paisa five have` |
| 6 | `you` | 3 | 1, 20, 28 | `hello you how?`, `you work finish` |
| 7 | `mother` | 3 | 4, 11, 13 | `mother food cook`, `grand mother market go` |
| 8 | `food` | 3 | 4, 31, 34 | `mother food cook`, `food taste good` |
| 9 | `market` | 3 | 11, 12, 13 | `grandmother market go will` |
| 10 | `we` | 3 | 17, 21, 22 | `we night go out`, `we this week holiday` |
| 11 | `my` | 3 | 19, 39, 73 | `my teeth brush`, `this work my not` |
| 12 | `eat` | 3 | 26, 33, 50 | `first him eat after i eat`, `i fruit fork eat` |
| 13 | `hot` | 3 | 29, 30, 37 | `furnac machine near temperature hot` |
| 14 | `good` | 3 | 31, 35, 47 | `food taste good`, `milk health good` |
| 15 | `cold` | 3 | 32, 38, 53 | `water cold`, `it cold coffee` |
| 16 | `like` | 3 | 44, 48, 49 | `she live life like queen`, `i black coffe like` |
| 17 | `white` | 3 | 41, 51, 57 | `shirt white`, `cup color white light` |
| 18 | `color` | 3 | 41, 43, 51 | `eyes color black`, `cup color white light` |
| 19 | `in` | 3 | 46, 68, 77 | `apple in vitamin-mineral have` |
| 20 | `teeth` | 3 | 73, 74, 79 | `your teeth clean`, `her teeth broke` |
| 21 | `home` | 2 | 2, 18 | `Home welcome`, `i home reach very late` |
| 22 | `water` | 2 | 8, 53 | `aunty plants water`, `water cold` |
| 23 | `clean` | 2 | 79, 80 | `your Teeth clean`, `floor clean` |
| 24 | `comb` | 2 | 76, 81 | `his hair comb will`, `this comb wooden` |
| 25 | `coffee` | 2 | 32, 37 | `it cold coffee`, `coffee hot` |
| 26 | `cookies` | 2 | 54, 55 | `cookies fresh`, `cookies stale` |
| 27 | `work` | 2 | 19, 28 | `you work finish`, `this work my not` |
| 28 | `after` | 2 | 25, 26 | `first him eat after i eat` |
| 29 | `first` | 2 | 25, 26 | `raju first come tanu after` |
| 30 | `finish` | 2 | 13, 28 | `you work finish`, `...market go finish` |

*(All remaining 137 glosses, including `hello`, `how`, `baby`, `marraige`, `animals`, `sleep`, appear exactly 1 time).*

---

## 5. Sentence-to-Video Mapping

Every sentence in the dataset was cross-referenced against all 152 video files:

- **Total Annotated Sentences:** 78
- **Sentences with 2 Video Recordings:** 74 sentences
- **Sentences with Only 1 Video Recording:** 4 sentences
  1. `sr.no 25`: `25 raju first come tanu after.MOV` (Front view only)
  2. `sr.no 43`: `43 eyescolor black .MOV` (Front view only)
  3. `sr.no 74`: `74 her  teeth  broke .MOV` (Front view only)
  4. `sr.no 79`: `79 your Teeth clean 2.MOV` (Side view only; front view missing)
- **Unmatched Videos:** 0 (all 152 videos map to one of the 78 sentence IDs).
- **Missing Videos:** 0 (all 78 annotated sentences have at least 1 video).

---

## 6. Front/Side Camera Pairing and Video Duplicates

### Camera Angle Findings
Visual frame inspection and OpenCV facial orientation analysis revealed the recording configuration:
1. **Camera Angles:**
   - **Front-View (Frontal Angle):** The camera directly faces the signer.
   - **Side-View (Profile Angle):** The camera captures the signer at a $90^\circ$ right-profile angle.
2. **Consecutive Takes (Not Simultaneous Genlocked Streams):**
   - Frame-by-frame synchronization testing on paired files (e.g., `1 hello you how .MOV` vs `1 hello you how 2 .MOV`) confirms that the front and side views **were not recorded simultaneously by two synchronized cameras**.
   - Instead, the signer performed the sentence once facing the front camera, and signed a second take facing the side camera. This is proven by disparate frame counts (93 vs 102 frames) and distinct gesture phases at identical elapsed timestamps.

### Accidental Binary Duplicate Files
Five sentence IDs contain **exact binary duplicate files** (identical MD5 checksums). In each case, a file was duplicated with a `- Copy` or hash suffix instead of capturing the opposite angle:

| Sentence ID | Primary Video File | Duplicate Video File | MD5 Hash | Verified View | Defect Description |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **2** | `2 Home welcome.MOV` | `2 Home welcome - Copy.MOV` | `0de60bc973e379b6ad15df2d8e76765f` | **Front** | Side view missing; front file duplicated |
| **4** | `4 mother food cook  - Copy.MOV` | `4 mother food cook  - Copy-CokUY0.MOV` | `95b6943f70cd08692a550dcc07118404` | **Side** | Front view missing; side file duplicated |
| **8** | `8 aunty plants water  (2).MOV` | `8 aunty plants water  (2) - Copy.MOV` | `aa0d94a0f364dd953e24f90133262789` | **Side** | Front view missing; side file duplicated |
| **53** | `53 water cold .MOV` | `53 water cold  - Copy.MOV` | `9408712430723e14c6edb161b91b642a` | **Front** | Side view missing; front file duplicated |
| **64** | `64 Money american use dollars  .MOV` | `64 Money american use dollars  -GZoEyI.MOV` | `bd5ed8138b14ea1a20cabe4a9881adc6` | **Front** | Side view missing; front file duplicated |

### Aggregate Angle Breakdown
- **Front-View Videos:** 79 files
- **Side-View Videos:** 73 files
- **True Paired Sentences (Both Front + Side Available):** 69 sentences
- **Front-Only Sentences:** 7 sentences (IDs 2, 25, 43, 53, 64, 74, plus duplicate front takes)
- **Side-Only Sentences:** 2 sentences (IDs 4, 8, 79)

---

## 7. Video Quality Summary

Every video file was processed using OpenCV to extract encoding and geometric metadata:

| Metric | Minimum | Mean | Maximum | Uniformity |
| :--- | :---: | :---: | :---: | :---: |
| **Resolution** | $1920 \times 1080$ | $1920 \times 1080$ | $1920 \times 1080$ | **100% (152 / 152 videos)** |
| **Frame Rate (FPS)** | 25.0 fps | 25.0 fps | 25.0 fps | **100% (152 / 152 videos)** |
| **Duration (seconds)** | 3.72 s | 7.26 s | 12.96 s | Variable by sentence length |
| **Frame Count** | 93 frames | 181.38 frames | 324 frames | Proportional to duration |
| **Color Space / Backdrop**| Solid Green | Solid Green | Solid Green | Controlled chroma backdrop |
| **Signer Apparel** | Black polo | Black polo | Black polo | High contrast against background |
| **Lighting & Blur** | Good | Good | Minor hand motion blur | Stable indoor lighting |
| **Integrity / Readability**| Clean | Clean | Clean | 0 unreadable or corrupted files |

---

## 8. Candidate Vocabulary Feasibility Table

The candidate daily-conversation vocabulary specified in the project brief was evaluated against verified annotations:

| Candidate Concept | Exists in Annotations? | Exact Original Gloss Label | Frequency (Sentence Count) | Matching Sentence IDs | Front Videos | Side Videos | Visually Inspectable? | Verified Meaning / Convention | Sufficient for Initial Experiment? | Feasibility Verdict |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HELLO** | **Yes** | `hello` | 1 | 1 | 1 | 1 | Yes | Two-handed wave / greeting | No (1 sample only) | **INSUFFICIENT** |
| **YOU** | **Yes** | `you` / `your` | 5 | 1, 20, 28, 62, 79 | 4 | 5 | Yes | Index pointing forward | Marginal (5 samples) | **MARGINAL** |
| **HOW** | **Yes** | `how` | 1 | 1 | 1 | 1 | Yes | Two hands open twisting upward | No (1 sample only) | **INSUFFICIENT** |
| **WATER** | **Yes** | `water` | 2 | 8, 53 | 1 | 2 (1 dup) | Yes | 'W' / cupped hand at chin | No (1 front sample only!) | **INSUFFICIENT** |
| **FOOD** | **Yes** | `food` | 3 | 4, 31, 34 | 2 | 2 (1 dup) | Yes | Hand to mouth eating motion | Marginal (2 front samples) | **MARGINAL** |
| **HELP** | **NO** | *None* | **0** | *None* | 0 | 0 | N/A | Absent from corpus | No | **ABSENT** |
| **HOME** | **Yes** | `Home` / `home` | 2 | 2, 18 | 2 (1 dup) | 1 | Yes | Roof-tent handshape | No (2 samples only) | **INSUFFICIENT** |
| **GO** | **Yes** | `go` | 6 | 11, 12, 13, 17, 21, 72 | 6 | 6 | Yes | Forward directional hand sweep | Acceptable (6 samples) | **VIABLE** |
| **WHERE** | **NO** | *None* | **0** | *None* | 0 | 0 | N/A | Absent from corpus | No | **ABSENT** |
| **THANK YOU**| **NO** | *None* | **0** | *None* | 0 | 0 | N/A | Absent from corpus | No | **ABSENT** |

### Critical Findings on Candidate Concepts
1. **Non-Existent Signs (30%):** `HELP`, `WHERE`, and `THANK YOU` (or `THANKS`) do not appear anywhere in the dataset.
2. **Singleton Signs (20%):** `HELLO` and `HOW` exist only in Sentence 1 (`hello you how?`). If sentence 1 is in the training split, zero evaluation samples exist in the test split; if in test, zero training samples exist.
3. **Severe Data Asymmetry in `WATER` and `FOOD`:**
   - `WATER` in Sentence 8 (`aunty plants water`) is only available in side view (two duplicate side files). In Sentence 53 (`water cold`), it is only available in front view (two duplicate front files). Thus, there is **only 1 front-view recording** of the word `WATER` in the entire dataset.
   - `FOOD` in Sentence 4 (`mother food cook`) is only available in side view. Only Sentences 31 and 34 contain front-view recordings of `FOOD`.

---

## 9. Annotation and Language Limitations

1. **Regional Marathi Linguistic Influence:**
   - The dataset was created in Maharashtra, India. The syntax follows Marathi **Subject-Object-Verb (SOV)** structure rather than standard English SVO or international sign syntax:
     - Sentence 9: English *"uncle gets fruits"* $\rightarrow$ Marathi *"काकांना फळे मिळतात"* $\rightarrow$ ISL Gloss: `uncle fruits get`.
     - Sentence 4: English *"mother cooks food"* $\rightarrow$ Marathi *"आई अन्न शिजवते"* $\rightarrow$ ISL Gloss: `mother food cook`.
     - Sentence 50: English *"i eat fruits with fork"* $\rightarrow$ Marathi *"मी काट्याने फ्युट्स खातो"* $\rightarrow$ ISL Gloss: `i fruit fork eat`.
2. **Row Misalignment and Annotation Corruption in Excel:**
   - Inspection of rows 61–68 revealed copy-paste offset errors in the Marathi column:
     - Row 63: English *"i have to give her ten rupees"* $\rightarrow$ Marathi *"मला तिला दहा रुपये द्यावे लागतील."*
     - Row 65: English *"coins are too many"* $\rightarrow$ Marathi *"मला तिला दहा रुपये द्यावे लागतील"* (Erroneously duplicated from Row 63).
     - Row 67: English *"i have five paisa"* $\rightarrow$ Marathi *"नाणी खूप आहेत"* (Actually translates *"coins are many"*, belonging to Row 65).
     - Row 68: English *"there are many animals in zoo"* $\rightarrow$ Marathi *"तिकिटांची किंमत काय आहे"* (Actually translates *"what is the cost of tickets"*, belonging to Row 66).
3. **Single Signer Constraint:**
   - All 152 recordings feature the **same individual signer**.
   - No demographic, dialectal, handedness, or anatomical diversity exists in the dataset.
   - Any evaluation will reflect signer-dependent memorization rather than true generalized sign recognition.

---

## 10. Data Split Recommendation

Because the dataset features a single signer, an authentic **signer-independent split is impossible**. To prevent catastrophic data leakage, the following split protocol is mandatory:

### Split Protocol Rules
1. **Sentence-Level Splitting:** All takes (front and side) of a given Sentence ID must remain in the **same split**. Splitting front into Train and side into Test causes severe visual memorization leakage.
2. **Target Ratio:** 80% Train (62 sentences, ~120 video takes) / 20% Test (16 sentences, ~32 video takes).
3. **Stratification Constraint:** Rare and singleton classes cannot be evaluated across both splits. If a model must be trained on continuous sequences, sentences containing target vocabulary must be strategically partitioned.

### Recommended 80/20 Leakage-Safe Split (Sentence IDs)

```text
TRAIN SPLIT (62 Sentences):
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40,
41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
60, 61, 62, 63, 64, 65

TEST / EVALUATION SPLIT (16 Sentences):
66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 34
```

---

## 11. Recommended Initial 4 Signs

To evaluate continuous sequence recognition (e.g., Bi-GRU + CTC) without creating an immediate zero-shot failure, candidate signs must have adequate frequency and clean front-view coverage:

| Selected Gloss | Total Sentences | Train Instances | Test Instances | Syntactic Context | Justification |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **`I`** | 8 | 6 | 2 (IDs 67, 72) | Subject pronoun | Highest frequency gloss in corpus; clean front view |
| **`GO`** | 6 | 5 | 1 (ID 72) | Action verb | High frequency; clear directional motion |
| **`FOOD`** | 3 | 2 | 1 (ID 34) | Core noun | Daily conversation relevance; co-occurs with verbs |
| **`EAT`** | 3 | 2 | 1 (ID 26/50) | Action verb | Co-occurs with `FOOD` and `I`; natural continuous transitions |

*Alternative 4th sign:* **`NOT`** (Frequency = 4; sentences 19, 24, 34, 45; useful for negative expressions like `food healthy not`).

---

## 12. Recommended Final 10 Signs (Empirically Supported)

The project's target candidate set cannot be fulfilled because `HELP`, `WHERE`, and `THANK YOU` do not exist. If a 10-sign continuous recognition demonstration is required from this dataset, the following data-backed vocabulary is recommended:

| Rank | Gloss Label | Total Occurrences | Example Sentences | Pragmatic Meaning |
| :---: | :--- | :---: | :--- | :--- |
| 1 | **`I`** | 8 | `i bathroom go`, `i home reach very late` | First person pronoun |
| 2 | **`GO`** | 6 | `grand mother market go`, `i bathroom go` | Motion / travel |
| 3 | **`THIS`** | 5 | `this work my not`, `this day i inform you` | Demonstrative |
| 4 | **`YOU`** | 5 | `hello you how?`, `you work finish`, `your hands wash` | Second person pronoun |
| 5 | **`NOT`** | 4 | `she money not`, `pizza food healthy not` | Negation |
| 6 | **`HAVE`** | 4 | `cereals protein have`, `i paisa five have` | Possession / state |
| 7 | **`FOOD`** | 3 | `food taste good`, `pizza food healthy not` | Nutrition / daily need |
| 8 | **`EAT`** | 3 | `first him eat after i eat`, `i fruit fork eat` | Daily action |
| 9 | **`GOOD`** | 3 | `food taste good`, `milk health good` | Quality / state |
| 10 | **`COLD`** | 3 | `water cold`, `it cold coffee` | Temperature / physical attribute |

*Note:* Even this top-10 vocabulary has only 3 to 8 samples per sign.

---

## 13. Sequence-Model Training Feasibility

### Architecture Compatibility: Bi-GRU + CTC / Transformer + CTC
1. **CTC Loss Mechanism:** Connectionist Temporal Classification (CTC) aligns variable-length frame sequences to target gloss sequences without explicit frame-level segmentations. CTC relies on learning when to emit tokens and when to emit blank (`ε`) tokens.
2. **Minimum Data Requirement for CTC:**
   - In literature (e.g., PHOENIX-Weather, CSL-Continuous), CTC requires **hundreds to thousands of sequence pairs** to learn alignment without collapsing into all-blank predictions.
   - Training a Bi-GRU or Transformer with CTC on **only 62 training sentences** from **1 signer** is mathematically prone to:
     - **Catastrophic blank token collapse:** The model emits only blanks because blank predictions yield lower entropy on small data.
     - **Severe sequence memorization:** The model memorizes exact frame counts for the 62 sentences rather than learning invariant sign boundaries.
3. **Absence of Timestamps:**
   - Because no sign-level timestamps exist, supervised segment training (e.g., frame-level classification) is impossible without manual annotation.
   - CTC is the only viable sequence loss, but the sample size ($N=62$) is critically undersized.

---

## 14. Final Decision

| Scope | Decision | Justification |
| :--- | :---: | :--- |
| **Requested 10-Sign Daily Conversation Demo** *(HELLO, YOU, HOW, WATER, FOOD, HELP, HOME, GO, WHERE, THANK YOU)* | **NO-GO** | **Fatal Lexical Deficit:** 3 signs do not exist (`HELP`, `WHERE`, `THANK YOU`), 2 signs appear only once (`HELLO`, `HOW`), and 1 sign has only 1 front-view take (`WATER`). |
| **Alternative 4-Sign Continuous Experiment** *(`I`, `GO`, `EAT`, `FOOD`)* | **CONDITIONAL GO** | **Feasible for Pipeline Validation Only:** Can validate feature extraction, sequence batching, and CTC loss pipeline integration, provided expectations account for single-signer memorization. |
| **Standalone Production Continuous ISL Model** | **NO-GO** | **Sample Size & Diversity Deficit:** 78 sentences by 1 signer cannot yield a generalized, robust continuous ISL model. Multi-signer data collection or pre-trained sign representations are strictly required. |

---
*Report compiled autonomously following Phase 8.1 specification.*
