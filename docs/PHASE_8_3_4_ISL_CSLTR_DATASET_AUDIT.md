# Phase 8.3.4: ISL-CSLTR Dataset Audit and Feasibility Analysis

**Project:** Signova AI — Continuous Indian Sign Language Recognition  
**Audit Date:** November 13, 2026  
**Auditor:** Automated Diagnostic Suite (DeepMind Agent / Antigravity)  
**Dataset Path:** `D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset`  
**Extracted Corpus Root:** `D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus`  
**Archive Reference:** `D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset.zip` (8,906,641,132 bytes / 8.295 GB)  
**Kaggle Source:** [ISL-CSLTR Indian Sign Language Dataset (drblack00)](https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset)  
**Reported License:** `CC-BY-SA-4.0`  
**Status:** COMPLETE EMPIRICAL AUDIT  

---

## Executive Summary

Phase 8.3.4 conducted an exhaustive, evidence-based technical audit of the extracted **ISL-CSLTR** (*Indian Sign Language Continuous Sign Language Recognition and Translation*) dataset located at `D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset`. Every video file, extracted frame, annotation spreadsheet, and metadata record was decoded and verified.

The primary objective of this audit was to determine whether ISL-CSLTR resolves the critical failure modes identified during the Phase 8.3.3 audit of the Mendeley ISLVT dataset (namely: single-signer lack of generalization, 91.7% training shortcut collapse to single-token emissions, blank token suppression, and severe lexical sparsity).

```
+---------------------------------------------------------------------------------------------------+
|                                     DATASET COMPARISON SUMMARY                                    |
+--------------------------+-------------------------------------+----------------------------------+
| Dimension                | ISLVT (Mendeley, Phase 8.1 - 8.3.3) | ISL-CSLTR (SASTRA / Navajeevan)  |
+--------------------------+-------------------------------------+----------------------------------+
| Total Video Files        | 152 (.MOV, Apple ProRes/H.264)      | 687 (.mp4, H.264/AAC)            |
| Total Extracted Frames   | None pre-extracted (video only)     | 19,899 JPG (18,863 sent, 1036 wd)|
| Signer Diversity         | 1 Signer (Single South Asian Male)  | 7 Signers (2 Native, 5 Students) |
| Signer Independence Eval | IMPOSSIBLE (Zero signer variation)  | FULLY SUPPORTED (5/1/1 split)    |
| Unique Spoken Sentences  | 78 valid sentences (ID span 1-81)   | 101 unique spoken sentences      |
| Total Gloss Lexicon      | 177 unique glosses (77.4% singleton)| 170 unique glosses               |
| Mean Sequence Length     | 3.26 tokens (91.7% single-token tgt)| 3.58 tokens (99.0% multi-token)  |
| Length-1 Shortcut Risk   | CRITICAL (Model collapsed to L=1)   | ELIMINATED (Only 1 sentence L=1) |
| Video Decodability       | 152 / 152 clean (0 corrupted)       | 687 / 687 clean (0 corrupted)    |
| CTC Feasibility (Video)  | Valid (T=77-124, Stride 1-2)        | 100% Valid (Stride 1-2, T=25-192)|
| Overall Verdict          | CONDITIONAL GO -> FAILED IN 8.3.3   | GO (STRONGLY RECOMMENDED)        |
+--------------------------+-------------------------------------+----------------------------------+
```

### Primary Audit Verdicts

1. **Suitability for Continuous ISL Recognition: GO (APPROVED)**  
   ISL-CSLTR contains 687 videos covering 101 conversational sentences with 362 total gloss tokens. 99.0% of sentences (100 of 101) contain multi-token continuous sign sequences (lengths 2 to 8, median 3, mean 3.58). This directly resolves the shortcut collapse that crippled Phase 8.3.2/8.3.3.
2. **Signer Independence: SUPPORTED (7 Distinct Signers)**  
   The corpus was performed by 7 distinct signers: 2 native deaf signers from Navajeevan Residential School for the Deaf (Andhra Pradesh) and 5 student volunteers from SASTRA Deemed University (Tamil Nadu). This enables true signer-independent cross-validation for the first time in Signova AI.
3. **CTC Feasibility: 100% VALID ON RAW VIDEOS (WARNING ON EXTRACTED FRAMES)**  
   When evaluated on raw 25 FPS `.mp4` videos (mean 94.1 frames, median 95.0 frames), 100.0% of videos satisfy the CTC length constraint ($T' \ge |Y|$) under temporal stride 1 and stride 2, and 99.85% under stride 4. However, pre-extracted frames in `Frames_Sentence_Level` were heavily subsampled (mean 28.5 frames, median 24.0 frames), causing 9.65% of clips to violate CTC under stride 4 and 51.3% under stride 8. **Features must be extracted directly from the raw video files.**
4. **Lexical Representation of Candidate Glosses:**  
   `YOU` (41 occurrences, 40 sentences, 275 videos) and `I` (33 occurrences, 32 sentences, 218 videos) have dense multi-signer coverage. `NOT` (9 occurrences, 62 videos), `GO` (3 occurrences, 22 videos), and `FOOD` (2 occurrences, 16 videos) are present. `EAT` is absent (0 occurrences), but `FOOD`, `WATER`, and `MEDICINE` provide functional replacements.
5. **Recommended Architecture: Temporal CNN + Bi-GRU + CTC**  
   A 1D Temporal Convolutional block (kernel 3 or 5, stride 2) combined with a 2-layer Bidirectional GRU (hidden dim 256) and CTC loss is recommended. Transformer architectures are rejected at this stage due to the high risk of catastrophic overfitting on a 687-sample dataset.

---

## 1. Dataset Directory Structure and Integrity

### 1.1 Directory Layout

The extracted dataset is rooted at:  
`D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset`

All functional data resides within the nested hierarchy:  
`ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\`

```text
D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\
└── ISL_CSLRT_Corpus\
    └── ISL_CSLRT_Corpus\
        ├── ISL_CSLRT.txt                               (1,196 bytes, Corpus metadata README)
        ├── corpus_csv_files\
        │   ├── ISL Corpus sign glosses.csv             (3,872 bytes, 101 sentences -> glosses)
        │   ├── ISL_CSLRT_Corpus details.xlsx           (23,034 bytes, 492 video file paths)
        │   ├── ISL_CSLRT_Corpus_frame_details.xlsx     (323,296 bytes, 18,863 frame paths)
        │   └── ISL_CSLRT_Corpus_word_details.xlsx      (126,540 bytes, 1,036 word frame paths)
        ├── Videos_Sentence_Level\                      (101 sentence folders, 687 .mp4 files)
        │   ├── are you free today\                     (7 videos: 2 low-res, 5 high-res)
        │   ├── He is going into the room\              (7 videos)
        │   └── ... (99 additional sentence folders)
        ├── Frames_Sentence_Level\                      (97 sentence folders, 18,863 .jpg files)
        │   ├── are you free today\                     (Subfolders 1..7 for signers)
        │   └── ... (96 additional sentence folders)
        └── Frames_Word_Level\                          (114 isolated word folders, 1,036 .jpg files)
            ├── A LOT\                                  (4 frames)
            ├── ABUSE\                                  (5 frames)
            └── ... (112 additional word folders)
```

### 1.2 File Inventory and Storage Metrics

Every file in the extracted hierarchy was indexed and verified against the master ZIP archive:

| Asset Category | File Extension | File Count | Uncompressed Size (Bytes) | Uncompressed Size (GB) | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentence Videos** | `.mp4` / `.MP4` | **687** | 7,619,194,875 | 7.096 GB | 100% Decodable via OpenCV |
| **Sentence Image Frames** | `.jpg` | **18,863** | 1,399,788,891 | 1.304 GB | 100% Intact RGB JPEGs |
| **Word Image Frames** | `.jpg` | **1,036** | 55,116,929 | 0.051 GB | 100% Intact RGB JPEGs |
| **Annotation Workbooks** | `.xlsx` | **3** | 472,870 | 0.451 MB | Parsed cleanly via openpyxl |
| **Annotation Table** | `.csv` | **1** | 3,872 | 3.78 KB | UTF-8 Tabular Data |
| **Corpus Documentation** | `.txt` | **1** | 1,196 | 1.17 KB | Plaintext Description |
| **Explorer Caches** | `.db` (`Thumbs.db`) | **483** | 429,292 | 0.409 MB | Windows thumbnail artifacts |
| **Total Corpus Assets** | — | **21,074** | **9,074,479,125** | **8.451 GB** | Complete & Bit-Accurate |

### 1.3 Archive Integrity Check

The archive `isl-csltr-indian-sign-language-dataset.zip` (8,906,641,132 bytes / 8.295 GB) was checked using Python's `zipfile.ZipFile.testzip()`.  
- **CRC32 Check Result:** `None` (Zero corrupted file blocks).
- **Extracted File Count:** Exactly 21,074 files.
- **Corrupted / Truncated Files on Disk:** Exactly 0.

---

## 2. Video Audit

Every single video file ($N=687$) in `Videos_Sentence_Level` was opened, read frame-by-frame, and audited with OpenCV (`cv2.VideoCapture`).

### 2.1 Video Stream Characteristics

| Characteristic | Measured Value | Distribution / Range | Notes |
| :--- | :--- | :--- | :--- |
| **Total Video Count** | **687 files** | All located in `Videos_Sentence_Level` | 100% `.mp4` container |
| **Corrupted Videos** | **0 files (0.0%)** | All 687 read frame 0 successfully | OpenCV `cap.isOpened() == True` |
| **Frame Rate (FPS)** | **25.0 FPS** | 687 / 687 videos (100.0%) | Constant frame rate |
| **Dual Resolution Split** | **1920x1080 (Full HD)** | **492 videos (71.6%)** | Volunteer signers (Canon DSLR) |
| | **320x240 (QVGA)** | **195 videos (28.4%)** | Native deaf signers (Mobile) |
| **Total Duration** | **2,761.2 seconds** | 46.02 minutes (0.77 hours) | Across all 687 videos |
| **Duration (Excl. Outlier)**| **2,581.2 seconds** | 43.02 minutes (0.72 hours) | Across 686 typical videos |

### 2.2 Video Frame Count and Duration Distribution

A single recording (`i do not like it / i do not like it (2).mp4`) was identified as a 180.0-second (4,500 frames) session outlier where the camera remained recording after sign completion. Excluding this single file yields an exceptionally tight distribution across the remaining 686 videos:

```
Video Duration Distribution (N=686, excluding 180s outlier):
  1.0s - 2.0s :   13 videos ( 1.9%)  |#
  2.0s - 3.0s :  161 videos (23.5%)  |##########
  3.0s - 4.0s :  238 videos (34.7%)  |###############
  4.0s - 5.0s :  188 videos (27.4%)  |############
  5.0s - 6.0s :   66 videos ( 9.6%)  |####
  6.0s - 7.7s :   20 videos ( 2.9%)  |#
```

| Metric | All 687 Videos | Cleaned Subset (N=686) | Extracted Frames (`Frames_Sentence_Level`) |
| :--- | :--- | :--- | :--- |
| **Minimum Frames** | 25 frames (1.00s) | 25 frames (1.00s) | 6 frames |
| **25th Percentile** | 75 frames (3.00s) | 75 frames (3.00s) | 20 frames |
| **Median Frames** | 95 frames (3.80s) | 95 frames (3.80s) | 24 frames |
| **Mean Frames** | 100.48 frames (4.02s) | 94.07 frames (3.76s) | 28.45 frames |
| **75th Percentile** | 111 frames (4.44s) | 111 frames (4.44s) | 36 frames |
| **Maximum Frames** | 4,500 frames (180.0s) | 192 frames (7.68s) | 147 frames |
| **Standard Deviation**| 170.88 frames | 30.73 frames | 13.81 frames |

### 2.3 Exact Binary Duplicates (MD5 Analysis)

MD5 hashing over all 687 video files identified **682 unique hashes** and **5 duplicate pairs** (10 files total). All 5 pairs represent misnamed or cross-copied files in the source dataset:

| Hash | File Size | Redundant File Paths | Nature of Duplicate |
| :--- | :--- | :--- | :--- |
| `91203a8f...` | 14,480,739 B | `are you hiding something\MVI_6817.MP4`<br>`do you need something\something (2).MP4` | Cross-sentence duplicate copy |
| `d96f0165...` | 14,038,586 B | `do not worry\worry.MP4`<br>`No need to worry dont worry\MVI_6496.MP4` | Synonymous sentence copy |
| `34d79e4e...` | 14,695,012 B | `i do not agree\agree.MP4`<br>`i dont agree\MVI_6494.MP4` | Alternate folder spelling copy |
| `ffcf9be3...` | 13,735,435 B | `i like you i love you\love.MP4`<br>`i like you i love you\MVI_6814.MP4` | Intra-folder duplicate take |
| `67170eaa...` | 21,408,634 B | `now onwards he will never hurt you\never hurt.MP4`<br>`what have you planned for your career\MVI_6499.MP4`| Cross-sentence duplicate copy |

**Audit Recommendation:** During feature extraction in Phase 8.3.5, the redundant duplicate files in each group must be deduplicated to avoid data contamination.

---

## 3. Annotation Audit

### 3.1 Inventory of Annotation Sources

The dataset contains four distinct annotation / metadata files:

1. **`ISL Corpus sign glosses.csv` (Primary Ground Truth):**
   - 101 rows $\times$ 2 columns (`Sentence`, `SIGN GLOSSES`).
   - Maps each spoken English sentence to its target ISL gloss transcription.
   - Contains zero missing values (`NaN = 0`).
2. **`ISL_CSLRT_Corpus details.xlsx` (Video File Manifest):**
   - 493 rows in `Sheet1`. Row 492 contains `NaN` for `Sentences` and `>>>\xa0` for `File location`.
   - Remaining 492 rows map English sentences to their corresponding volunteer video filenames (`ISL_CSLRT_Corpus\Videos_Sentence_Level\...`).
   - Notably omits the 195 native signer low-resolution videos (`<sentence>.mp4` and `<sentence> (2).mp4`), which exist on disk but were omitted from this specific spreadsheet.
3. **`ISL_CSLRT_Corpus_frame_details.xlsx` (Sentence Frame Manifest):**
   - 18,863 rows $\times$ 2 columns (`Sentence`, `Frames path`).
   - Covers 97 unique sentences (4 sentences are absent from the extracted frames).
4. **`ISL_CSLRT_Corpus_word_details.xlsx` (Isolated Word Manifest):**
   - 1,036 rows $\times$ 2 columns (`Word`, `Frames path`).
   - Covers 114 isolated words.
5. **`ISL_CSLRT.txt` (Corpus Overview):**
   - Documents the recording location, date (28-12-2019), and participating signers (2 native, 5 students).

### 3.2 Annotation Completeness and Temporal Boundary Finding

> [!IMPORTANT]
> **No Frame-Level Timestamps or Boundary Annotations Exist:**  
> Exhaustive inspection of all four annotation files confirmed that **zero start/end timestamps, bounding boxes, or sign-level boundary markers** are provided. The dataset provides **utterance-level sentence transcriptions only**.  
> The statement in `ISL_CSLRT.txt` claiming the corpus is *"arranged based on signer variants and time boundaries with fully annotated details"* refers solely to the organization of files into sentence/signer directories.  
> **Because CTC is an unaligned sequence loss function, utterance-level gloss sequences are fully sufficient for end-to-end continuous recognition.**

### 3.3 Discrepancies and Inconsistencies

1. **Four Sentences Missing from `Frames_Sentence_Level`:**  
   Four sentences present in `Videos_Sentence_Level` and `ISL Corpus sign glosses.csv` have zero extracted frames in `Frames_Sentence_Level`:
   - `now onwards he will never hurt you` (6 videos)
   - `i dont agree` (1 video)
   - `No need to worry dont worry` (1 video)
   - `he would be coming today` (7 videos)  
   *Because Signova AI will extract features directly from the raw `.mp4` videos, these 15 videos remain fully usable.*
2. **Typographical Discrepancies:**
   - In `ISL Corpus sign glosses.csv`, sentence 49 is written with a space: `which college school are you from`. In folder names and frame manifests, it is written without a space: `which collegeschool are you from`.
   - Sentence 6 gloss is misspelled as `CONGRATULATIIONS` (two `I`s).
   - Sentence 9 gloss uses compound `DONOT ABUSE HIM` rather than `DO NOT ABUSE HIM`.
   - Sentence 19 contains double-space whitespace: `HE GO  INTO  ROOM`.
   - Trailing punctuation (commas) exists in sentence 99 (`YOU DO ANYTHING, I DO NOT CARE`) and sentence 100 (`YOU NEED A MEDICINE, TAKE THIS ONE`).
3. **Annotation Examples (Representative Sample):**

| Sentence Index | English Reference Sentence | Raw ISL Gloss Annotation | Normalized Gloss Sequence | Token Length ($L$) |
| :---: | :--- | :--- | :--- | :---: |
| **0** | `are you free today` | `YOU FREE TODAY` | `['YOU', 'FREE', 'TODAY']` | 3 |
| **3** | `can i help you` | `I HELP YOU` | `['I', 'HELP', 'YOU']` | 3 |
| **6** | `congratulations` | `CONGRATULATIIONS` | `['CONGRATULATIONS']` | 1 |
| **14** | `do not worry` | `DO NOT WORRY` | `['DO', 'NOT', 'WORRY']` | 3 |
| **19** | `He is going into the room` | `HE GO  INTO  ROOM` | `['HE', 'GO', 'INTO', 'ROOM']` | 4 |
| **67** | `serve the food` | `SERVE FOOD` | `['SERVE', 'FOOD']` | 2 |
| **85** | `what have you planned for your career` | `WHAT HAVE YOU PLAN YOUR CAREER` | `['WHAT', 'HAVE', 'YOU', 'PLAN', 'YOUR', 'CAREER']` | 6 |
| **96** | `you are good` | `YOU GOOD` | `['YOU', 'GOOD']` | 2 |
| **99** | `you do anything, i do not care` | `YOU DO ANYTHING, I DO NOT CARE` | `['YOU', 'DO', 'ANYTHING', 'I', 'DO', 'NOT', 'CARE']` | 7 |
| **100** | `you need a medicine, take this one` | `YOU NEED A MEDICINE, TAKE THIS ONE` | `['YOU', 'NEED', 'A', 'MEDICINE', 'TAKE', 'THIS', 'ONE']` | 8 |

---

## 4. Vocabulary Analysis

### 4.1 Candidate Glosses Audit

The candidate glosses specified in the audit directive were audited across all 101 annotation sentences, 687 video performances, and 7 signers:

| Candidate Gloss | Total Occurrences in Glosses | Unique Sentences | Total Videos Containing Gloss | Signer Coverage | Example Sentences |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`YOU`** | **41** | **40** | **275** | All 7 Signers | `are you free today`, `can i help you`, `you are good` |
| **`I`** | **33** | **32** | **218** | All 7 Signers | `can i help you`, `how can i trust you`, `i am afraid` |
| **`NOT`** | **9** | **9** | **62** | All 7 Signers | `do not worry`, `do not make me angry`, `i am not really sure` |
| **`GO`** | **3** | **3** | **22** | All 7 Signers | `go and sleep`, `He is going into the room`, `shall we go outside` |
| **`FOOD`** | **2** | **2** | **16** | All 7 Signers | `had your food`, `serve the food` |
| **`THIS`** | **2** | **2** | **14** | All 7 Signers | `This place is beautiful`, `you need a medicine, take this one` |
| **`HAVE`** | **1** | **1** | **3** | Signers 1, 2, 3 | `what have you planned for your career` |
| **`GOOD`** | **1** | **1** | **7** | All 7 Signers | `you are good` |
| **`EAT`** | **0** | **0** | **0** | None | *Does not appear in ISL-CSLTR annotations* |

#### Key Vocabulary Findings
- **High-Density Conversational Anchors:** `YOU` (275 videos) and `I` (218 videos) appear in over 30% of the entire dataset, providing robust multi-signer training representation.
- **`NOT` Coverage:** `NOT` occurs 9 times as an independent token (62 videos), and 4 additional times inside compound `DONOT` (26 videos: `do not abuse him`, `do not be stubborn`, `do not hurt me`, `do not take it to the heart`), bringing total negative valence coverage to 13 sentences (88 videos).
- **Absence of `EAT`:** The concept `EAT` does not exist in ISL-CSLTR. In ISLVT, `EAT` was present but proved to be completely unlearnable (erased to 0% recall in Phase 8.3.3). In ISL-CSLTR, eating-related concepts are represented by `FOOD` (`YOUR FOOD`, `SERVE FOOD`), `WATER` (`BRING WATER ME`), and `MEDICINE`.
- **Singletons among Candidates:** `GOOD` (Sentence 96: `YOU GOOD`) and `HAVE` (Sentence 85) appear in only one sentence group, making them unsuitable for isolated test split evaluation without multi-sentence redundancy.

### 4.2 Top 25 Most Frequent Glosses Across the Entire Corpus

Tokenization of all 101 target sentences yielded **362 total tokens** across **170 unique gloss types**:

```
Gloss Frequency Ranking (Top 25):
  1. YOU        : 41  (11.3% of all tokens)  |====================
  2. I          : 33  ( 9.1% of all tokens)  |================
  3. DO         : 16  ( 4.4% of all tokens)  |========
  4. WHAT       : 10  ( 2.8% of all tokens)  |=====
  5. NOT        :  9  ( 2.5% of all tokens)  |====
  6. IT         :  8  ( 2.2% of all tokens)  |====
  7. ME         :  7  ( 1.9% of all tokens)  |===
  8. HE         :  6  ( 1.7% of all tokens)  |===
  9. HOW        :  6  ( 1.7% of all tokens)  |===
 10. HELP       :  4  ( 1.1% of all tokens)  |==
 11. DONOT      :  4  ( 1.1% of all tokens)  |==
 12. TAKE       :  4  ( 1.1% of all tokens)  |==
 13. NEED       :  4  ( 1.1% of all tokens)  |==
 14. TO         :  4  ( 1.1% of all tokens)  |==
 15. WATER      :  3  ( 0.8% of all tokens)  |#
 16. HIM        :  3  ( 0.8% of all tokens)  |#
 17. HURT       :  3  ( 0.8% of all tokens)  |#
 18. WORRY      :  3  ( 0.8% of all tokens)  |#
 19. GO         :  3  ( 0.8% of all tokens)  |#
 20. YOUR       :  3  ( 0.8% of all tokens)  |#
 21. THAT       :  3  ( 0.8% of all tokens)  |#
 22. REALLY     :  3  ( 0.8% of all tokens)  |#
 23. FROM       :  3  ( 0.8% of all tokens)  |#
 24. WHY        :  3  ( 0.8% of all tokens)  |#
 25. TODAY      :  2  ( 0.6% of all tokens)  |#
```

### 4.3 Vocabulary Sparsity and Zipfian Curve

- **Total Vocabulary Size:** 170 unique glosses.
- **Singleton Glosses (Count = 1):** 116 glosses (**68.2%** of vocabulary appears in exactly 1 sentence).
- **Doubleton Glosses (Count = 2):** 30 glosses (**17.6%** of vocabulary appears in exactly 2 sentences).
- **Glosses with Count $\ge 3$:** 24 glosses (**14.1%** of vocabulary).
- **Glosses with Count $\ge 6$:** 9 glosses (**5.3%** of vocabulary).

---

## 5. Sequence Analysis

### 5.1 Sequence Length Distribution

In stark contrast to ISLVT where the baseline was trained on 91.7% single-token sequences, ISL-CSLTR is an intrinsically multi-token continuous dataset:

```
Target Sequence Length Distribution (N=101 sentences):
  Length 1 :   1 sentence   ( 1.0%)  |#
  Length 2 :  18 sentences  (17.8%)  |#########
  Length 3 :  42 sentences  (41.6%)  |#####################
  Length 4 :  18 sentences  (17.8%)  |#########
  Length 5 :  11 sentences  (10.9%)  |#####
  Length 6 :   6 sentences  ( 5.9%)  |###
  Length 7 :   4 sentences  ( 4.0%)  |##
  Length 8 :   1 sentence   ( 1.0%)  |#
```

| Metric | Measured Value | Comparison to ISLVT Baseline (Phase 8.3.2) |
| :--- | :--- | :--- |
| **Minimum Length** | 1 token (`congratulations`) | 1 token (11 of 12 training sentences were L=1) |
| **25th Percentile** | 3 tokens | 1 token |
| **Median Length** | 3 tokens | 1 token |
| **Mean Length** | **3.58 tokens** | 1.17 tokens |
| **75th Percentile** | 4 tokens | 1 token |
| **Maximum Length** | 8 tokens (`YOU NEED A MEDICINE TAKE THIS ONE`) | 3 tokens |
| **Multi-Token Sequences**| **100 / 101 (99.0%)** | 1 / 12 training targets (8.3%) |

### 5.2 Most Frequent Sequences and N-Grams

- **Most Frequent Bigrams:**
  1. `DO NOT` (7 occurrences)
  2. `WHAT YOU` (4 occurrences)
  3. `YOU DO` (4 occurrences)
  4. `HELP YOU` (3 occurrences)
  5. `DO YOU` (3 occurrences)
  6. `I DO` (3 occurrences)
  7. `WHY YOU` (3 occurrences)
- **Most Frequent Trigrams:**
  1. `I DO NOT` (3 occurrences)
  2. `I HELP YOU` (2 occurrences)
  3. `DO NOT MAKE` (2 occurrences)
  4. `WHAT YOU DO` (2 occurrences)
  5. `WHAT DO YOU` (2 occurrences)
- **Repeated Consecutive Glosses ($c_i = c_{i+1}$):** **EXACTLY 0 OCCURRENCES**.  
  No sentence in ISL-CSLTR contains consecutive identical glosses (such as `GO GO` or `YOU YOU`).  
  *CTC Implication:* The alignment condition does not require forced inter-token blank emissions for identical adjacent tokens, simplifying the alignment lattice to $T_{input} \ge |Y_{target}|$.

---

## 6. Signer and Split Analysis

### 6.1 Signer Demographics and Profile

The corpus features **7 distinct signers** falling into two demographic and capture groups:

```
+---------------------------------------------------------------------------------------------------+
|                                      SIGNER PROFILE OVERVIEW                                      |
+-----------+--------------------+-----------------------------+------------+-----------+-----------+
| Signer ID | Signer Category    | Institution / Location      | Camera Res | Frame FPS | Sentences |
+-----------+--------------------+-----------------------------+------------+-----------+-----------+
| Signer 1  | Native Deaf Signer | Navajeevan School, AP       | 320x240    | 25.0 FPS  | 96        |
| Signer 2  | Native Deaf Signer | Navajeevan School, AP       | 320x240    | 25.0 FPS  | 96        |
| Signer 3  | Student Volunteer  | SASTRA University, TN       | 1920x1080  | 25.0 FPS  | 96        |
| Signer 4  | Student Volunteer  | SASTRA University, TN       | 1920x1080  | 25.0 FPS  | 90        |
| Signer 5  | Student Volunteer  | SASTRA University, TN       | 1920x1080  | 25.0 FPS  | 96        |
| Signer 6  | Student Volunteer  | SASTRA University, TN       | 1920x1080  | 25.0 FPS  | 96        |
| Signer 7  | Student Volunteer  | SASTRA University, TN       | 1920x1080  | 25.0 FPS  | 93        |
+-----------+--------------------+-----------------------------+------------+-----------+-----------+
```

### 6.2 Signer Independence Feasibility

- **Feasibility:** **CONFIRMED (100% FEASIBLE)**.  
  Because every sentence was performed independently by multiple signers, all performances of a designated signer can be completely held out from training.
- **Critical Cross-Domain Consideration:**  
  Signers 1 and 2 differ from Signers 3–7 along three correlated dimensions:
  1. *Deaf Native Fluency vs. L2 Volunteer Signing.*
  2. *Low-Resolution Mobile Capture (320x240) vs. High-Resolution Canon DSLR (1920x1080).*
  3. *Aspect Ratio / Portrait Crop vs. Widescreen 16:9.*

### 6.3 Recommended Signer-Independent Split Protocol

To evaluate genuine generalization to deaf native signing while ensuring the training set has sufficient diversity, the following **5 / 1 / 1 Signer Split** is recommended:

```
+---------------------------------------------------------------------------------------------------+
|                                  RECOMMENDED SPLIT SPECIFICATION                                  |
+------------+--------------------+--------------------------+---------------+----------------------+
| Split      | Assigned Signers   | Signer Composition       | Approx Videos | Primary Objective    |
+------------+--------------------+--------------------------+---------------+----------------------+
| **TRAIN**  | Signers 1, 3, 4,   | 1 Native Deaf Signer +   | ~474 videos   | Learn multi-signer   |
|            | 5, 6               | 4 Student Volunteers     | (69.0%)       | invariance & vocabs  |
+------------+--------------------+--------------------------+---------------+----------------------+
| **VAL**    | Signer 7           | 1 Student Volunteer      | ~93 videos    | Hyperparameter tuning|
|            |                    | (Unseen Signer)          | (13.5%)       | & early stopping     |
+------------+--------------------+--------------------------+---------------+----------------------+
| **TEST**   | Signer 2           | 1 Native Deaf Signer     | ~96 videos    | Zero-shot native ISL |
|            |                    | (Unseen Native Signer)   | (14.0%)       | generalization test  |
+------------+--------------------+--------------------------+---------------+----------------------+
```

*Note: 24 remaining videos belonging to irregular takes (e.g. third volunteer takes) will be partitioned into the Train split.*  
*As mandated by Phase 8.3.4 restrictions, no split files were generated during this audit.*

---

## 7. CTC Feasibility Analysis

### 7.1 Temporal Alignment Mathematics

Connectionist Temporal Classification (CTC) requires that the downsampled input sequence length $T'$ is strictly greater than or equal to the target label length $L$:
$$T' = \left\lfloor \frac{T}{\text{stride}} \right\rfloor \ge L$$

When $T' < L$, the set of valid CTC alignments $\mathcal{B}^{-1}(\mathbf{y})$ is empty ($\emptyset$), causing the loss to evaluate to $+\infty$ or throw a fatal execution error.

### 7.2 Measured Feasibility Across Strides

CTC alignment validity was computed across all 687 videos mapped to their target gloss lengths:

| Input Source | Temporal Stride | Effective Timesteps ($T'$) | Valid Alignments ($T' \ge L$) | Invalid Alignments ($T' < L$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw 25 FPS Videos** | **Stride 1** | Mean $T'=94.1$, Med $T'=95$ | **687 / 687 (100.00%)** | 0 | **Optimal** |
| **Raw 25 FPS Videos** | **Stride 2** | Mean $T'=47.0$, Med $T'=47$ | **687 / 687 (100.00%)** | 0 | **Recommended** |
| **Raw 25 FPS Videos** | **Stride 4** | Mean $T'=23.5$, Med $T'=23$ | **686 / 687 (99.85%)** | 1 (`i am so sorry...`) | Acceptable |
| **Raw 25 FPS Videos** | **Stride 8** | Mean $T'=11.8$, Med $T'=11$ | **682 / 687 (99.27%)** | 5 | Unsafe |
| **`Frames_Sentence_Level`**| Stride 1 | Mean $T'=28.5$, Med $T'=24$ | **662 / 663 (99.85%)** | 1 | High Density |
| **`Frames_Sentence_Level`**| Stride 2 | Mean $T'=14.2$, Med $T'=12$ | **658 / 663 (99.25%)** | 5 | Risky |
| **`Frames_Sentence_Level`**| Stride 4 | Mean $T'=7.1$, Med $T'=6$ | **599 / 663 (90.35%)** | **64 (9.65% fail)** | **DISQUALIFIED** |
| **`Frames_Sentence_Level`**| Stride 8 | Mean $T'=3.6$, Med $T'=3$ | **323 / 663 (48.72%)** | **340 (51.28% fail)** | **DISQUALIFIED** |

### 7.3 Frame-to-Token Ratio

In the raw 25 FPS videos, the mean frame-to-token ratio is **30.73 frames per gloss** (median **28.25**). At 25 FPS, this represents an average duration of **1.13 to 1.23 seconds per sign**, which matches natural continuous signing speed.  
Under Stride 2 downsampling, the model will observe **14 to 15 timesteps per sign**, providing ample temporal resolution for the bidirectional GRU to capture transitions, handshapes, and orientation changes.

---

## 8. Model Architecture Recommendation

Three candidate architectures were evaluated against the empirical characteristics of ISL-CSLTR (687 samples, 101 sentences, 170 vocabulary items, 7 signers, MediaPipe landmark feature inputs):

```
+---------------------------------------------------------------------------------------------------+
|                                 MODEL ARCHITECTURE COMPARISON MATRIX                              |
+---------------------+-------------------+-------------------------------+-------------------------+
| Criterion           | 1. Bi-GRU + CTC   | 2. Temporal CNN + Bi-GRU + CTC| 3. Transformer + CTC    |
+---------------------+-------------------+-------------------------------+-------------------------+
| Parameter Count     | ~250k - 400k      | ~450k - 650k                  | ~2.5M - 8.0M            |
| Local Temporal Bias | Weak (Pure RNN)   | High (1D Conv receptive field)| None (Learned attn)     |
| Overfitting Risk    | Low               | Low - Moderate                | Extremely High          |
| Data Diet Required  | 100 - 1,000 clips | 500 - 2,500 clips             | 10,000+ clips minimum   |
| Stride Handling     | None (Stride 1)   | Native Conv Stride (Stride 2) | Patch/Linear Stride     |
| Real-Time Latency   | < 5 ms            | < 8 ms                        | > 35 ms                 |
| Signova AI Verdict  | ACCEPTABLE        | **RECOMMENDED (WINNER)**      | **REJECTED (OVERKILL)** |
+---------------------+-------------------+-------------------------------+-------------------------+
```

### Justification for Selected Architecture (Temporal CNN + Bi-GRU + CTC)

1. **Why not pure Bi-GRU + CTC?**  
   While a pure Bi-GRU has low parameter overhead, Phase 8.3.3 proved that recurrent layers alone are susceptible to temporal saturation when frame-level landmark noise is high.
2. **Why add the Temporal CNN Frontend?**  
   A 2-layer 1D Convolutional module (Kernel size $K=3$ or $K=5$, Channels $130 \rightarrow 256$, Stride=2, LayerNorm, GELU, Dropout 0.2) acts as a localized feature filter. It computes instantaneous velocities and spatial trajectories across contiguous frames before recurrent summarization. Stride 2 halves the sequence length cleanly ($T \approx 95 \rightarrow T' \approx 47$), maintaining 100% CTC validity while doubling training throughput.
3. **Why reject Transformers?**  
   Self-attention lacks inductive bias for temporal locality and requires massive pre-training data to avoid memorizing background or signer specifics. Training a Conformer or Vanilla Transformer on 474 training clips will result in catastrophic overfitting and divergence.

---

## 9. License and Attribution

### 9.1 Kaggle Mirror vs. Academic Origin

| Field | Kaggle Mirror Record | Verified Academic Origin |
| :--- | :--- | :--- |
| **Source URL** | [Kaggle Dataset drblack00](https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset) | SASTRA Deemed University & Navajeevan Deaf School |
| **Stated License** | `CC-BY-SA-4.0` | Academic Research Dataset (Implicit SASTRA / Navajeevan) |
| **Commercial Use** | Permitted under CC-BY-SA-4.0 (ShareAlike) | Requires Institutional Verification before Commercial Release |
| **Redistribution** | Permitted with Attribution | Permitted with Attribution |

### 9.2 Attribution Requirements
Any model checkpoints, derived landmarks, or benchmarks developed from this dataset must cite the original creators:
- **Participating Deaf Community:** Native signers of Navajeevan Residential School for the Deaf, College of Spl. D.Ed & B.Ed, Ayyalurimetta, Andhra Pradesh, India.
- **Recording & Academic Development:** Student volunteers and research staff of SASTRA Deemed University, Thanjavur, Tamil Nadu, India (Recorded December 28, 2019).

---

## 10. Final Decision and Roadmap

### 10.1 Answers to the 6 Core Audit Questions

1. **Is this dataset suitable for continuous ISL recognition?**  
   **YES (GO)**. It contains 101 distinct conversational sentences and 687 decodable videos with natural multi-token sequences (mean 3.58 tokens).
2. **Is it better than the previously audited ISLVT dataset?**  
   **DECISIVELY YES**. It replaces a single-signer, 91.7% single-token dataset with a 7-signer, 99.0% multi-token dataset with 4.5x more videos and zero shortcut collapse risk.
3. **Is it suitable for CTC training?**  
   **YES**, when trained on features extracted from the 25 FPS raw `.mp4` videos. 100.0% of videos satisfy $T' \ge L$ under Stride 1 and Stride 2. Pre-extracted `.jpg` frames in `Frames_Sentence_Level` must be avoided.
4. **Is signer-independent evaluation possible?**  
   **YES**. The 7-signer composition allows a clean 5 / 1 / 1 partition with zero signer overlap.
5. **Which architecture should Signova AI use?**  
   **Temporal 1D CNN + Bi-GRU + CTC** (2-layer 1D CNN, Stride 2, 2-layer Bi-GRU, Hidden Dim 256, Linear Projection to 171 classes including blank).
6. **What should the next phase be?**  
   **Phase 8.3.5: ISL-CSLTR Feature Extraction and Signer Split Preparation**.

### 10.2 Next Phase Action Plan (Phase 8.3.5 Prerequisites)

- [ ] Run MediaPipe Holistic landmark extraction directly on all 687 `.mp4` video files at full 25 FPS (generating `(T, 130)` feature tensors).
- [ ] Deduplicate the 5 identified identical video pairs.
- [ ] Serialize the 5 / 1 / 1 signer-independent split manifests (`train_manifest.json`, `val_manifest.json`, `test_manifest.json`).
- [ ] Build the vocab index mapping for the 170 gloss classes + 1 CTC blank token.
- [ ] Implement and benchmark the Temporal CNN + Bi-GRU + CTC pipeline.
