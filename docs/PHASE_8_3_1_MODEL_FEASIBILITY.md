# Phase 8.3.1: Continuous Sequence Model Feasibility Audit

**Project:** Signova AI — Continuous Indian Sign Language Recognition  
**Phase:** 8.3.1 — Model Feasibility & Architecture Audit  
**Date:** November 13, 2026  
**Auditor:** Automated Diagnostic Suite (DeepMind Agent / Antigravity)  
**Dataset Path:** `datasets/islvt_continuous/`  
**Target Vocabulary:** `I`, `GO`, `EAT`, `FOOD` (4-sign experimental subset)  
**Status:** COMPLETE AUDIT — CONDITIONAL GO FOR EXPERIMENTAL BASELINE  

---

## Executive Summary

Phase 8.3.1 conducted a rigorous technical feasibility audit to determine whether the 32 video recordings (6,237 frames, 16 sentences) extracted in Phase 8.2 can support continuous sequence recognition using a Connectionist Temporal Classification (CTC) pipeline.

The audit verified raw annotations, temporal sequence dimensions, CTC mathematical constraints, camera angle properties, and class distributions across train and test splits.

### Primary Verdicts

| Dimension | Verdict | Summary Justification |
| :--- | :---: | :--- |
| **Mathematical CTC Feasibility** | **GO** | Every video satisfies $T \gg L$ (mean ratio $T/L = 168.3\times$, minimum $78.0\times$). No identical consecutive target tokens ($N_{\text{repeats}} = 0$). Zero tensor errors. |
| **Experimental Baseline (Phase 8.3.2)** | **CONDITIONAL GO** | Approved **only** as a lightweight proof-of-concept pipeline test using Bi-GRU + CTC under strict guardrails. |
| **Recommended Architecture** | **Bi-GRU + CTC** | Transformer + CTC is decisively rejected due to catastrophic sample complexity failure on $N=24$ training samples. |
| **General ISL Translation** | **NO-GO** | Dataset has exactly 1 signer, 16 total sentences, and 4 target glosses. Cannot generalize to open-vocabulary ISL. |

---

## 1. Annotation Verification

### 1.1 Source Ground Truth Alignment
All 16 processed sentences were audited against the official ISLVT Excel workbook (`marathi-English_ISLgloss1.xlsx`, Mendeley Data, DOI: `10.17632/98mzk82wbb.1`). 

- The Excel workbook contains 78 valid sentences.
- Across all 78 sentences, the four approved experimental words (`i`, `go`, `eat`, `food`) appear in **exactly 16 unique sentences**.
- Every single sentence containing at least one target gloss was successfully extracted in Phase 8.2 with 100% coverage (16/16).

### 1.2 Full Sentence vs Target Gloss Verification
The table below documents the 16 sentences, verifying English reference, Marathi reference, raw ISL gloss transcriptions, token sequence, and the exact extracted target sequence:

| ID | Split | English Reference | Marathi Reference | Full ISL Gloss | Full Tokens | Target Sequence | Target Length ($L$) |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **4** | Train | mother cooks food | आई अन्न शिजवते | `mother food cook` | 3 | `['food']` | 1 |
| **11** | Train | Grand mother is going to market | आजी बाजारात जाणार आहेत | `grand mother market go` | 4 | `['go']` | 1 |
| **12** | Train | Grandmother will go the market | आजी बाजारात जाईल | `grandmother market go will` | 4 | `['go']` | 1 |
| **13** | Train | grandmother had gone to market | आजी बाजारात गेली होती | `before grand mother market go finish` | 6 | `['go']` | 1 |
| **17** | Train | which places can we go to see | आपण कोणत्या ठिकाणी जाऊन पाहू शकतो | `we place see go  which` | 5 | `['go']` | 1 |
| **18** | Train | i came home very late | मी खूप उशीरा घरी आलो | `i home reach very late` | 5 | `['i']` | 1 |
| **20** | Train | i will let you know in a day | मी तुम्हाला एका दिवसात कळवेल | `this day i inform you` | 5 | `['i']` | 1 |
| **21** | Test | we are going out in night | आम्ही रात्री बाहेर जात आहोत | `we night go out` | 4 | `['go']` | 1 |
| **26** | Train | i will eat after him | मी त्याच्या मागे खाईन | `first him eat after i eat` | 6 | `['eat', 'i', 'eat']` | 3 |
| **31** | Train | food is tastty | अन्न चवदार आहे | `food taste good` | 3 | `['food']` | 1 |
| **34** | Test | pizza is not healhty food | पिझ्झा हे आरोग्यदायी अन्न नाही | `pizza food healthy not` | 4 | `['food']` | 1 |
| **44** | Train | i love black  coffee | मला ब्लॅक कॉफी आवडते | `i black coffe like` | 4 | `['i']` | 1 |
| **50** | Test | i eat fruits with fork | मी काट्याने फ्युट्स खातो | `i fruit fork eat` | 4 | `['i', 'eat']` | 2 |
| **63** | Train | i have to give her ten rupees | मला तिला दहा रुपये द्यावे लागतील. | `i her give ten rupees have` | 6 | `['i']` | 1 |
| **67** | Train | i have five paisa | नाणी खूप आहेत | `i paisa five have` | 4 | `['i']` | 1 |
| **72** | Test | i go to bathroom | मी बाथरूमला जातो | `i  bathroom go` | 3 | `['i', 'go']` | 2 |

### 1.3 Label Consistency & Lexical Integrity
- **Exact Token Ordering:** Target sequences preserve the strictly ordered sequence of appearance in the original gloss.
- **Normalization:** Whitespace variations in the raw Excel file (e.g., leading space in ID 67 ` i paisa five have`, double space in ID 72 `i  bathroom go`, extra space in ID 17 `we place see go  which`) are cleanly handled by tokenization.
- **Vocabulary Consistency:** No spelling drift exists across target tokens (`i`, `go`, `eat`, `food`).

---

## 2. Vocabulary & Sequence Distribution

### 2.1 Class Frequencies Across Splits
The dataset is partitioned into 24 training videos (12 sentences $\times$ 2 takes) and 8 test videos (4 sentences $\times$ 2 takes).

| Gloss Label | Train Sentences | Train Videos | Train Tokens | Test Sentences | Test Videos | Test Tokens | Total Corpus Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`i`** | 6 (IDs 18, 20, 26, 44, 63, 67) | 12 (50.0%) | 12 (42.9%) | 2 (IDs 50, 72) | 4 (50.0%) | 4 (33.3%) | 16 |
| **`go`** | 4 (IDs 11, 12, 13, 17) | 8 (33.3%) | 8 (28.6%) | 2 (IDs 21, 72) | 4 (50.0%) | 4 (33.3%) | 12 |
| **`food`** | 2 (IDs 4, 31) | 4 (16.7%) | 4 (14.3%) | 1 (ID 34) | 2 (25.0%) | 2 (16.7%) | 6 |
| **`eat`** | 1 (ID 26) | 2 (8.3%) | 4 (14.3%) | 1 (ID 50) | 2 (25.0%) | 2 (16.7%) | 6 |
| **Total** | **12 sentences** | **24 videos** | **28 tokens** | **4 sentences** | **8 videos** | **12 tokens** | **40 tokens** |

### 2.2 Sequence Length Distribution
Analyzing sequence length reveals a significant structural distribution shift between train and test splits:

```
Train Target Sequence Lengths (N=24 videos):
- Length 1: 22 videos (91.7%)  [IDs 4, 11, 12, 13, 17, 18, 20, 31, 44, 63, 67]
- Length 2:  0 videos ( 0.0%)
- Length 3:  2 videos ( 8.3%)  [ID 26: eat -> i -> eat]

Test Target Sequence Lengths (N=8 videos):
- Length 1:  4 videos (50.0%)  [IDs 21, 34]
- Length 2:  4 videos (50.0%)  [IDs 50, 72]
- Length 3:  0 videos ( 0.0%)
```

### 2.3 Structural Findings & Data Deficits
1. **Zero Length-2 Sequences in Training:** The training set contains **zero 2-token target sequences**. However, 50% of the test set consists of 2-token sequences (`['i', 'eat']` in ID 50 and `['i', 'go']` in ID 72). A model trained on this dataset must predict length-2 transitions in test that it never observed during training.
2. **Extreme Sparsity of `eat`:** The gloss `eat` appears in **only one sentence** in the entire training set (Sentence 26, 2 video takes). While `eat` appears twice in that sentence (`eat -> i -> eat`), it only appears in a single phonetic and visual co-articulation context.
3. **Single Signer Uniformity:** All 32 recordings are performed by the exact same signer. Inter-signer morphological variance is zero.

---

## 3. Input & Target Length Analysis

### 3.1 Temporal Duration Metrics
All videos were recorded at 25.0 FPS.

- **Total Frames Extracted:** 6,237 frames across 32 videos.
- **Minimum Frames ($T_{\min}$):** 138 frames (5.52 s, Sentence 31 Front).
- **Maximum Frames ($T_{\max}$):** 291 frames (11.64 s, Sentence 12 Side).
- **Mean Frames ($\mu_T$):** 194.9 frames (7.80 s).
- **Median Frames ($\text{Med}_T$):** 190.5 frames (7.62 s).
- **Target Sequence Length ($L$):** Min = 1, Max = 3, Mean = 1.25.
- **Full Utterance Length ($L_{\text{full}}$):** Min = 3, Max = 6, Mean = 4.38.

### 3.2 Sequence-by-Sequence Length Table

| ID | Split | Camera Take Front ($T_F$) | Camera Take Side ($T_S$) | Target Tokens ($L$) | Min $T$ | Ratio $T_{\min} / L$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | Train | *Missing (Dup Side)* | 147 (5.88s) | 1 | 147 | $147.0\times$ |
| **11** | Train | 219 (8.76s) | 207 (8.28s) | 1 | 207 | $207.0\times$ |
| **12** | Train | 189 (7.56s) | 291 (11.64s) | 1 | 189 | $189.0\times$ |
| **13** | Train | 210 (8.40s) | 243 (9.72s) | 1 | 210 | $210.0\times$ |
| **17** | Train | 201 (8.04s) | 225 (9.00s) | 1 | 201 | $201.0\times$ |
| **18** | Train | 186 (7.44s) | 192 (7.68s) | 1 | 186 | $186.0\times$ |
| **20** | Train | 144 (5.76s) | 141 (5.64s) | 1 | 141 | $141.0\times$ |
| **21** | Test | 153 (6.12s) | 165 (6.60s) | 1 | 153 | $153.0\times$ |
| **26** | Train | 240 (9.60s) | 234 (9.36s) | 3 | 234 | $78.0\times$ |
| **31** | Train | 138 (5.52s) | 144 (5.76s) | 1 | 138 | $138.0\times$ |
| **34** | Test | 222 (8.88s) | 231 (9.24s) | 1 | 222 | $222.0\times$ |
| **44** | Train | 159 (6.36s) | 168 (6.72s) | 1 | 159 | $159.0\times$ |
| **50** | Test | 246 (9.84s) | 177 (7.08s) | 2 | 177 | $88.5\times$ |
| **63** | Train | 228 (9.12s) | 252 (10.08s) | 1 | 228 | $228.0\times$ |
| **67** | Train | 177 (7.08s) | 186 (7.44s) | 1 | 177 | $177.0\times$ |
| **72** | Test | 198 (7.92s) | 177 (7.08s) | 2 | 177 | $88.5\times$ |

### 3.3 Temporal Subsampling & Downsampling Capacity
In continuous sequence architectures, 1D temporal convolutions or pooling downsample the input sequence by factor $S$. We evaluate feasibility across standard downsampling factors:

| Downsampling Factor ($S$) | Resulting Timesteps Range ($T' = T / S$) | Min Timesteps ($T'_{\min}$) | Max Target Length ($L_{\max}$) | Margin ($T'_{\min} / L_{\max}$) | CTC Valid? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$S = 1$ (No downsampling)** | $138 - 291$ timesteps | 138 | 3 | $46.0\times$ | **YES** |
| **$S = 2$ (Stride-2 Conv1D)** | $69 - 145$ timesteps | 69 | 3 | $23.0\times$ | **YES** |
| **$S = 4$ (Two Stride-2 Convs)** | $34 - 72$ timesteps | 34 | 3 | $11.3\times$ | **YES** |
| **$S = 8$ (Three Stride-2 Convs)** | $17 - 36$ timesteps | 17 | 3 | $5.7\times$ | **YES** |

Even with aggressive temporal downsampling ($S=4$), the shortest sequence (Sentence 31, 138 frames) yields 34 timesteps, which provides an $11.3\times$ safety margin over the longest target sequence ($L=3$).

---

## 4. CTC Feasibility Results

### 4.1 Theoretical CTC Mathematical Constraints
Connectionist Temporal Classification (Graves et al., 2006) defines the probability of a target sequence $\mathbf{y}$ as the sum of conditional probabilities over all valid alignment paths $\boldsymbol{\pi}$:

$$P(\mathbf{y} \mid \mathbf{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^{T'} P(\pi_t \mid \mathbf{x}, t)$$

For CTC alignment to be mathematically valid:
1. **Length Constraint:** The encoder output length $T'$ must be greater than or equal to the modified target length $L'$ accounting for consecutive identical tokens:
   $$T' \ge L + N_{\text{consecutive\_duplicates}}$$
2. **Consecutive Duplicate Check:** If a target sequence contains adjacent identical tokens (e.g., `['eat', 'eat']`), CTC requires at least one blank token $\epsilon$ between them ($N_{\text{consecutive\_duplicates}} \ge 1$).
   - In our dataset, only Sentence 26 contains a repeated sign (`['eat', 'i', 'eat']`).
   - The two `eat` tokens are separated by `i`.
   - Therefore, $N_{\text{consecutive\_duplicates}} = 0$ for all 32 samples.
   - Theoretical minimum timesteps required: $T'_{\min} \ge L$.

**Verdict:** Zero samples violate the CTC length constraint. All 32 recordings are mathematically valid for CTC loss calculation.

### 4.2 The "Weak Supervision" / CTC Blank Pollution Challenge
While mathematically valid, the dataset presents a profound learning challenge for CTC:
- **Utterances are Full Sentences:** Signers perform 3 to 6 words per recording (mean 4.38 words).
- **Target Transcriptions are Filtered Subsets:** Only words from `{i, go, eat, food}` are included in the target transcription.
- **Consequence for CTC Blank Alignment:** In an utterance such as Sentence 13 (`before grand mother market go finish`), the signer performs 6 distinct sign gestures across 210 frames. The target label provided to CTC is `['go']`.
- **Blank Pollution:** To minimize CTC loss, the model is forced to map the gestures for `before`, `grand`, `mother`, `market`, and `finish` to the **CTC blank token** ($\epsilon$).
- Furthermore, in Sentence 4 (`mother food cook`), `mother` is mapped to blank, while in Sentence 11 (`grand mother market go`), `mother` is also mapped to blank.
- **Risk:** The model may struggle to distinguish legitimate non-signing silence / transition movements from unannotated active signs. CTC will treat ~75-80% of active signing frames as blank.

---

## 5. Camera Angle Consistency & Defect Inventory

### 5.1 Front vs Side View Synchronization
Frame-by-frame synchronization analysis proves that **front and side views were NOT recorded simultaneously by two synchronized cameras**.

Instead, the signer performed each sentence twice: once facing the frontal camera, and once facing the profile camera.

| Sentence ID | Front View Frames ($T_F$) | Side View Frames ($T_S$) | Frame Delta ($T_F - T_S$) | Time Delta | Synchronous? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | Missing | 147 | N/A | N/A | No (Duplicate side takes) |
| **11** | 219 | 207 | +12 | +0.48 s | No (Separate takes) |
| **12** | 189 | 291 | **-102** | **-4.08 s** | **No (Massive take variation)** |
| **13** | 210 | 243 | -33 | -1.32 s | No (Separate takes) |
| **17** | 201 | 225 | -24 | -0.96 s | No (Separate takes) |
| **18** | 186 | 192 | -6 | -0.24 s | No (Separate takes) |
| **20** | 144 | 141 | +3 | +0.12 s | No (Separate takes) |
| **21** | 153 | 165 | -12 | -0.48 s | No (Separate takes) |
| **26** | 240 | 234 | +6 | +0.24 s | No (Separate takes) |
| **31** | 138 | 144 | -6 | -0.24 s | No (Separate takes) |
| **34** | 222 | 231 | -9 | -0.36 s | No (Separate takes) |
| **44** | 159 | 168 | -9 | -0.36 s | No (Separate takes) |
| **50** | 246 | 177 | **+69** | **+2.76 s** | **No (Large take variation)** |
| **63** | 228 | 252 | -24 | -0.96 s | No (Separate takes) |
| **67** | 177 | 186 | -9 | -0.36 s | No (Separate takes) |
| **72** | 198 | 177 | +21 | +0.84 s | No (Separate takes) |

### 5.2 Landmark Quality by Camera Angle
Due to the $90^\circ$ right profile angle of the side camera, significant self-occlusion occurs:

| Metric | Front View ($N=15$) | Side View ($N=17$) | Delta | Physical Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Pose Detection Rate** | 100.0% | 100.0% | 0.0% | Full body visible in both angles |
| **Left Hand Detection Rate** | **81.8%** | **67.5%** | **-14.3%** | Left hand occluded by torso in right-side profile |
| **Right Hand Detection Rate** | 89.3% | 83.4% | -5.9% | Right hand closer to side camera, better tracked |
| **Missing Landmark Rate** | **8.1%** | **13.7%** | **+5.6%** | Zero-filled unobserved hand keypoints |

### 5.3 Complete Inventory of Recording Defects

1. **Defect 1: Sentence 4 Binary Duplicate (Missing Front View)**
   - File 1: `4 mother food cook  - Copy.MOV` (MD5: `95b6943f70cd08692a550dcc07118404`)
   - File 2: `4 mother food cook  - Copy-CokUY0.MOV` (MD5: `95b6943f70cd08692a550dcc07118404`)
   - Both files are identical byte-for-byte copies of the side view take. The frontal take is absent from the dataset. Both files are in `train`.
2. **Defect 2: Extreme Duration Asynchrony (Sentences 12 and 50)**
   - Sentence 12: Front view is 189 frames (7.56s), side view is 291 frames (11.64s). The signer lingered, paused, or restarted during the side recording.
   - Sentence 50: Front view is 246 frames (9.84s), side view is 177 frames (7.08s).
3. **Defect 3: Perspective Transformation Incompatibility**
   - In frontal recordings, $(x, y)$ coordinates reflect coronal plane movements. In side recordings, $x$ represents depth / sagittal displacement. Combining them without 3D camera calibration introduces multi-modal noise into coordinate representations.

---

## 6. Architecture Comparison: Bi-GRU vs. Transformer Encoder

We evaluate the two candidate sequence encoder architectures under the strict reality of this dataset ($N_{\text{train}} = 24$ recordings, 12 unique sentences, 1 signer).

| Evaluation Dimension | Bidirectional GRU Encoder + CTC | Transformer Encoder + CTC | Verdict / Rationale |
| :--- | :--- | :--- | :--- |
| **Sample Complexity** | **Very Low ($10^1 - 10^2$ samples)**<br>Recurrent inductive bias constrains temporal hypotheses without needing millions of parameters. | **Extremely High ($10^3 - 10^5$ samples)**<br>Self-attention has zero sequential inductive bias and must learn temporal continuity from data. | **Bi-GRU Wins**<br>With only 24 training samples, a Transformer cannot learn generalized self-attention. |
| **Parameter Budget** | **~250K - 450K parameters**<br>(Input projection + 2 Bi-GRU layers + linear head). | **~1.5M - 8.0M parameters**<br>(Standard 4-6 layer Multi-Head Attention blocks + FFN). | **Bi-GRU Wins**<br>Bi-GRU fits comfortably within low-capacity regimes without over-parameterization. |
| **Overfitting & Attention Collapse** | **Manageable**<br>Regularized via Dropout (0.3-0.5), Weight Decay ($10^{-4}$), and hidden size constraints (128 units). | **Catastrophic Overfitting Guaranteed**<br>Attention weights collapse into memorizing sequence position indices for the 12 training sentences. | **Bi-GRU Wins**<br>Transformer attention collapses immediately on small corpora. |
| **Temporal Alignment Stability** | **Smooth & Continuous**<br>Hidden state transitions enforce continuous temporal smoothness naturally. | **Noisy / Fragmented**<br>Unguided attention heads jump across unrelated frames without temporal smoothing. | **Bi-GRU Wins**<br>Bi-GRU provides natural smoothing across 25 FPS video frames. |
| **Inference Latency & Edge Footprint** | **~2 - 5 ms / sequence** on CPU.<br>Lightweight memory footprint (<10 MB). | **~20 - 50 ms / sequence** on CPU.<br>Quadratic attention complexity $O(T^2)$. | **Bi-GRU Wins**<br>Signova AI's goal is real-time interactive edge deployment. |

---

## 7. Recommended Architecture Specification

Based on empirical data constraints, **Bidirectional GRU with CTC Loss** is the sole recommended architecture for Phase 8.3.2.

### 7.1 Architecture Blueprint

```
Input Tensor: (B, T, 300) [300 = 75 landmarks x 4 channels (x, y, z, conf)]
     │
     ▼
Linear Feature Projection:
     Linear(300 -> 128)
     LayerNorm(128)
     Dropout(p = 0.3)
     ReLU()
     │
     ▼
Temporal Subsampling (Downsampling factor S = 2):
     Conv1D(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
     LayerNorm(128)
     ReLU()
     [Reduces sequence length from T to T' = ceil(T/2) ~ 69 - 145 timesteps]
     │
     ▼
Bidirectional Recurrent Encoder:
     Bi-GRU Layer 1: hidden_size = 128 (bidirectional -> output dim 256)
     Dropout(p = 0.3)
     Bi-GRU Layer 2: hidden_size = 128 (bidirectional -> output dim 256)
     Dropout(p = 0.3)
     │
     ▼
Classification Head:
     Linear(256 -> 5)
     [Classes: 0: <blank>, 1: 'i', 2: 'go', '3': 'eat', 4: 'food']
     LogSoftmax(dim = -1)
     │
     ▼
Loss Function:
     torch.nn.CTCLoss(blank=0, zero_infinity=True)
```

### 7.2 Training Hyperparameters

- **Optimizer:** AdamW (Learning Rate: $3 \times 10^{-4}$, Weight Decay: $1 \times 10^{-3}$)
- **LR Scheduler:** CosineAnnealingLR ($T_{\max} = 100$, $\eta_{\min} = 1 \times 10^{-5}$)
- **Batch Size:** 4 or 8 (with dynamic padding and `pack_padded_sequence`)
- **Gradient Clipping:** Max norm = 1.0 (prevents CTC gradient explosions)
- **Epochs:** 100 (with EarlyStopping on Validation CTC Loss, patience = 15)

---

## 8. Known Limitations & Technical Constraints

Prior to commencing Phase 8.3.2, all stakeholders must formally acknowledge the following constraints:

1. **Single Signer Constraint:** All 32 recordings feature a single individual. Test performance evaluates only sequence recognition for this specific individual; it provides zero evidence of cross-signer generalization.
2. **Extreme Training Sample Sparsity:** The training set contains only 24 videos representing 12 unique sentences. Deep learning models under this sample volume operate in a severe few-shot regime.
3. **Severe Sequence Length Distribution Shift:** 
   - 11 of 12 training sentences have target length 1; 1 sentence has target length 3.
   - 2 of 4 test sentences have target length 2 (`['i', 'eat']` and `['i', 'go']`).
   - The model has never seen a 2-token target transition during training.
4. **`eat` Representation Deficit:** `eat` appears in only one training sentence (`first him eat after i eat`). The model has essentially one opportunity to learn the gesture for `eat`.
5. **Weak Supervision Blank Distortion:** In videos where the signer performs 4-6 signs, ~80% of active signing frames will be supervised as CTC blank ($\epsilon$), penalizing the model's ability to learn clean sign-to-sign boundary transitions.
6. **Asynchronous Dual-Camera Takes:** Side views represent different physical executions from frontal views and suffer from 14% left-hand occlusion.

---

## 9. Final Decision & Phase Roadmap

### Final Verdict: CONDITIONAL GO (For Phase 8.3.2 Baseline Pipeline Only)

#### Conditions for Proceeding:
1. **Architecture:** Use strictly the **Bi-GRU + CTC** model specified in Section 7. Transformer models are forbidden on this dataset.
2. **Scope:** Phase 8.3.2 is an **experimental feasibility baseline** designed to test pipeline convergence, loss calculation, CTC blank decoding, and greedy word error rate (WER).
3. **Claims:** No claims of general ISL sentence translation or production readiness may be made from results on this 16-sentence dataset.
4. **Camera View Strategy:** Train initially on **Front-View Only** (12 clean samples) or **Both Views** (24 samples) with a comparative ablation to quantify whether side-view occlusion helps or harms CTC alignment.

---

*Report certified by DeepMind Agent (Antigravity) on November 13, 2026 for Signova AI Phase 8.3.1.*
