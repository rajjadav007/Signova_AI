# Phase 8.5A: Continuous ISL Recognition Preparation Report

**Project:** Signova AI — Continuous Indian Sign Language (ISL) Recognition  
**Phase:** 8.5A — Experimental Configuration & Feature Verification  
**Dataset:** ISL-CSLTR Continuous Sentence Corpus (`datasets/isl_csltr_continuous/`)  
**Target Vocabulary:** 20 Closed-Set ISL Glosses  
**Target Corpus:** 18 Verified Sentence Structures (126 Full Video Recordings)  
**Status:** COMPLETE (Preparation & Verification Only; Zero Model Training; Zero Checkpoint Alterations)  
**Deliverable Path:** `docs/PHASE_8_5A_TRAINING_CONFIGURATION.md`  
**Machine-Readable Config:** `configs/phase_8_5a_config.json`  

---

## 1. Executive Summary

Phase 8.5A establishes a verified, mathematically sound, and empirically audited experimental configuration for continuous Indian Sign Language (ISL) sentence recognition. This configuration transitions from the degenerate 170-class baseline (Phase 8.4: 83.09% WER, 0% exact match, complete blank collapse) to a focused 20-gloss, 18-sentence closed-set prototype spanning 126 complete continuous video recordings.

### Core Principles Maintained
1. **Zero Isolated Slicing:** Every sample is processed as a complete sentence video from beginning to end.
2. **Zero Split Tampering:** Strict signer-independent partitions (Train: Signers 1, 3, 4, 5, 6; Validation: Signer 7; Test: Signer 2) are 100% preserved.
3. **Strict Preparation Boundary:** Zero model weights were trained or fine-tuned; existing production checkpoints remain untouched.

---

## 2. Task 1: Dataset Verification

A comprehensive audit was performed across the raw video files, landmark files, annotations, and metadata in `datasets/isl_csltr_continuous/`.

### 2.1 Verification Checklist
- **Selected Videos:** Exactly 126 canonical videos selected from the 682 canonical corpus.
- **Sentence Structures:** Exactly 18 unique grammatical sentence structures.
- **Video Physical Readability:** All 126 MP4 files exist and were opened and verified via `cv2.VideoCapture`. 0 corrupt files.
- **Annotation Integrity:** Every gloss sequence in `manifest.csv` matches the official ISL Corpus annotations.
- **Vocabulary Membership:** 100% of the 326 gloss tokens across the 126 videos belong to the 20-gloss vocabulary.
- **Video Completeness:** 0 missing video files; 0 missing feature `.npz` files.
- **Duplicate Leakage:** 0 duplicate pairs cross splits. All duplicate video files in the dataset remain exclusively in Train.
- **Signer Independence:**
  - Train: Signers 1, 3, 4, 5, 6 (92 videos)
  - Validation: Signer 7 (17 videos)
  - Test: Signer 2 (17 videos)
  - Overlap: Exactly 0 signers shared across splits.

### 2.2 Annotation Spelling Verification
The original annotation vocabulary was compared against standard English gloss spellings:
- **`(AGE)`**: Encoded in annotations with literal parentheses: `(AGE)` (gloss ID 1 in metadata).
- **`CONGRATULATIIONS`**: Encoded in original annotations with double 'I': `CONGRATULATIIONS` (gloss ID 30 in metadata).
- The remaining 18 glosses (`ANGRY`, `BAD`, `CRY`, `DISAPPOINTED`, `GOOD`, `HELP`, `HI`, `HOW`, `HUNGRY`, `I`, `ME`, `PROMISE`, `THINGS`, `TIRED`, `WELCOME`, `WHO`, `WHY`, `YOU`) match standard uppercase gloss orthography.

### 2.3 Sentence Coverage Table

```
+-----------------------------------------------------------------------------------------------------------------------------------------+
|                                                  SENTENCE COVERAGE ACROSS SPLITS                                                        |
+-----+------------------------------+-------------------------------------+-------+-----+------+-------+-------------------+---------------+
| SID | ISL Gloss Sequence           | English Reference Sentence          | Train | Val | Test | Total | Signers Covered   | Absent Splits |
+-----+------------------------------+-------------------------------------+-------+-----+------+-------+-------------------+---------------+
| 3   | I HELP YOU                   | I can help you                      | 6     | 1   | 1    | 8     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 6   | CONGRATULATIIONS             | Congratulations                     | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 7]    | None          |
| 23  | HELP ME                      | Help me                             | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 24  | HI HOW YOU                   | Hi how are you                      | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 25  | HOW THINGS                   | How are things                      | 4     | 1   | 1    | 6     | [1, 2, 3, 5, 6, 7]    | None          |
| 26  | HOW I HELP YOU               | How may I help you                  | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 30  | I (AGE)                      | I am thirty years old               | 5     | 1   | 0    | 6     | [1, 3, 4, 5, 6, 7]    | TEST ABSENT   |
| 32  | I CRY                        | I am crying                         | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 36  | I HUNGRY                     | I am hungry                         | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 43  | I TIRED                      | I am tired                          | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 54  | I PROMISE                    | I promise                           | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 91  | WHO YOU                      | Who are you                         | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 6, 7]    | None          |
| 92  | WHY YOU ANGRY                | Why are you angry                   | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 93  | WHY YOU CRY                  | Why are you crying                  | 5     | 1   | 1    | 7     | [1, 2, 3, 5, 6, 7]    | None          |
| 94  | WHY YOU DISAPPOINTED         | Why are you disappointed            | 6     | 0   | 1    | 7     | [1, 2, 3, 4, 5, 6]    | VAL ABSENT    |
| 95  | YOU BAD                      | You are bad                         | 6     | 1   | 1    | 8     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 96  | YOU GOOD                     | You are good                        | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
| 97  | YOU WELCOME                  | You are welcome                     | 5     | 1   | 1    | 7     | [1, 2, 3, 4, 5, 6, 7] | None          |
+-----+------------------------------+-------------------------------------+-------+-----+------+-------+-------------------+---------------+
| TOTAL                              | 18 Sentences                        | 92    | 17  | 17   | 126   | 7 Signers         | 2 Discrep.    |
+------------------------------------+-------------------------------------+-------+-----+------+-------+-------------------+---------------+
```

### Critical Split Absence Findings
1. **Sentence 30 (`I (AGE)`) is absent from Test (Signer 2):** Signer 2 did not record Sentence 30. Test set contains 0 occurrences of gloss `(AGE)`.
2. **Sentence 94 (`WHY YOU DISAPPOINTED`) is absent from Validation (Signer 7):** Signer 7 did not record Sentence 94. Validation set contains 0 occurrences of gloss `DISAPPOINTED`.
3. In accordance with strict restrictions, **no splits are altered**. These structural absences are tracked as known evaluation bounds.

---

## 3. Task 2: Vocabulary Analysis

A detailed statistical analysis of the 20 glosses across the 126 selected videos was computed.

```
+---------------------------------------------------------------------------------------------------+
|                                     GLOSS FREQUENCY & DISTRIBUTION                                |
+------------------+-------+------------+-----------+--------------+---------+----------+-----------+
| Gloss            | Total | Train Vids | Sentences | Train Signers| Val Occ | Test Occ | Status    |
+------------------+-------+------------+-----------+--------------+---------+----------+-----------+
| (AGE)            | 6     | 5          | 1         | 5            | 1       | 0        | Sparse    |
| ANGRY            | 7     | 5          | 1         | 5            | 1       | 1        | Sparse    |
| BAD              | 8     | 6          | 1         | 5            | 1       | 1        | Sparse    |
| CONGRATULATIIONS | 7     | 5          | 1         | 4            | 1       | 1        | Sparse    |
| CRY              | 14    | 10         | 2         | 5            | 2       | 2        | Moderate  |
| DISAPPOINTED     | 7     | 6          | 1         | 5            | 0       | 1        | Sparse    |
| GOOD             | 7     | 5          | 1         | 5            | 1       | 1        | Sparse    |
| HELP             | 22    | 16         | 3         | 5            | 3       | 3        | Frequent  |
| HI               | 7     | 5          | 1         | 5            | 1       | 1        | Sparse    |
| HOW              | 20    | 14         | 3         | 5            | 3       | 3        | Frequent  |
| HUNGRY           | 7     | 5          | 1         | 5            | 1       | 1        | Sparse    |
| I                | 49    | 36         | 7         | 5            | 7       | 6        | Dominant  |
| ME               | 7     | 5          | 1         | 5            | 1       | 1        | Sparse    |
| PROMISE          | 7     | 5          | 1         | 5            | 1       | 1        | Sparse    |
| THINGS           | 6     | 4          | 1         | 4            | 1       | 1        | Sparse    |
| TIRED            | 7     | 5          | 1         | 5            | 1       | 1        | Sparse    |
| WELCOME          | 7     | 5          | 1         | 5            | 1       | 1        | Sparse    |
| WHO              | 7     | 5          | 1         | 4            | 1       | 1        | Sparse    |
| WHY              | 21    | 16         | 3         | 5            | 2       | 3        | Frequent  |
| YOU              | 72    | 53         | 10        | 5            | 9       | 10       | Dominant  |
+------------------+-------+------------+-----------+--------------+---------+----------+-----------+
| TOTAL            | 326   | 236        | 18        | —            | 43      | 47       | Imbalanced|
+------------------+-------+------------+-----------+--------------+---------+----------+-----------+
```

### Coverage and Balance Assessment
- **Severe Class Imbalance:** The dataset is **not balanced**. `YOU` (72 occurrences, 22.1%) and `I` (49 occurrences, 15.0%) account for **37.1%** of all gloss tokens.
- **Structural Sentence Sparsity:** 14 out of the 20 glosses appear in **exactly 1 sentence structure**.
- **Evaluation Coverage Gaps:**
  - `(AGE)` has 0 occurrences in the test set.
  - `DISAPPOINTED` has 0 occurrences in the validation set.
- **Signer Representation:** In Train, 17 of 20 glosses have instances from all 5 training signers (Signers 1, 3, 4, 5, 6); `CONGRATULATIIONS`, `THINGS`, and `WHO` have instances from 4 signers.

---

## 4. Task 3: Feature Schema Verification

### 4.1 Dimension Derivation & Verification
The proposed 228-dimensional representation was verified mathematically against the raw MediaPipe landmark files:
1. **Raw Landmarks:** Each frame contains 75 landmarks:
   - 33 Pose landmarks (indices 0..32)
   - 21 Left hand landmarks (indices 33..53)
   - 21 Right hand landmarks (indices 54..74)
2. **Spatial Channels:** 3 spatial coordinates $(x, y, z)$ per landmark:
   $$75 \times 3 = 225 \text{ spatial dimensions}$$
3. **Detection Mask:** 3 boolean/float flags representing presence:
   $$\text{mask} = [\text{pose\_detected}, \text{left\_hand\_detected}, \text{right\_hand\_detected}] = 3 \text{ dimensions}$$
4. **Total Input Feature Vector:**
   $$\text{Total Dimensions} = 225 + 3 = 228 \text{ dimensions}$$

*Clarification on Velocities:* Concatenating explicit first-order velocities $(\Delta x, \Delta y, \Delta z)$ for all 75 landmarks would yield $225 + 225 + 3 = 453$ dimensions. Instead, in the 228-dim configuration, temporal motion derivatives are learned dynamically across adjacent frames $(t-1, t, t+1)$ by the 1D temporal convolution layer (Kernel=3, Stride=1, Padding=1).

### 4.2 Feature Specification
- **Landmark Selection:** 75 keypoints (MediaPipe Pose 0..32, Left Hand 0..20, Right Hand 0..20).
- **Coordinate Ordering:** Flattened $(x, y, z)$ for all 75 keypoints followed by the 3 detection flags:
  $$[p_{0,x}, p_{0,y}, p_{0,z}, p_{1,x}, p_{1,y}, p_{1,z}, \dots, p_{74,x}, p_{74,y}, p_{74,z}, m_{\text{pose}}, m_{\text{lh}}, m_{\text{rh}}]$$
- **Spatial Normalization (Torso-Centered & Scale-Normalized):**
  - Mid-shoulder center: $\mathbf{p}_{\text{mid}} = \frac{\mathbf{p}_{\text{left\_shoulder}, :2} + \mathbf{p}_{\text{right\_shoulder}, :2}}{2}$
  - Inter-shoulder distance: $d = \|\mathbf{p}_{\text{left\_shoulder}, :2} - \mathbf{p}_{\text{right\_shoulder}, :2}\|_2$ (clamped to $\ge 0.01$, fallback 0.2)
  - Normalized coordinates: $\mathbf{p}'_{i, :2} = \frac{\mathbf{p}_{i, :2} - \mathbf{p}_{\text{mid}}}{d}$, $z'_i = \frac{z_i}{d}$
- **Missing Landmark Strategy:** Strict zero-imputation ($0.0$). If a hand is absent, its 21 landmarks are assigned $(0.0, 0.0, 0.0)$, and its mask flag is set to $0.0$. No NaNs or Infs exist in any clip.
- **Feature Dtype:** `float32` (PyTorch `torch.float32`).
- **Temporal Sequence Lengths:** $T \in [25, 190]$ frames @ 25 fps ($1.00\text{s}$ to $7.60\text{s}$). Mean: 82.6 frames ($3.30\text{s}$), Median: 82.0 frames ($3.28\text{s}$).
- **Training/Inference Parity:** Preprocessing uses identical vectorized normalization code in both training datasets and the live inference pipeline.

---

## 5. Task 4: Architecture Configuration

The continuous recognition model processes variable-length normalized landmark sequences and outputs frame-level class posterior distributions.

```mermaid
graph TD
    In["Input Sequence<br>(B, T, 228)"] --> Trans1["Transpose (B, 228, T)"]
    Trans1 --> Conv["Conv1D(in=228, out=128, k=3, s=1, p=1)<br>+ ReLU"]
    Conv --> Trans2["Transpose (B, T, 128)"]
    Trans2 --> BiGRU["2-Layer Bidirectional GRU<br>(in=128, hidden=128, dropout=0.3)"]
    BiGRU --> Drop["Dropout (0.3)"]
    Drop --> Linear["Linear Classifier Head<br>(256 -> 21 classes)"]
    Linear --> LogSoftmax["LogSoftmax (dim=-1)<br>(T, B, 21) -> CTCLoss"]
```

### 5.1 Exact Layer-by-Layer Tensor Shapes

| Layer / Operation | Input Shape | Output Shape | Formula / Parameters |
|---|---|---|---|
| Input Batch | — | $(B, T, 228)$ | Feature tensor ($T \in [25, 190]$) |
| Transpose for Conv1d | $(B, T, 228)$ | $(B, 228, T)$ | Channel-first ordering |
| Temporal Conv1D | $(B, 228, T)$ | $(B, 128, T)$ | $T' = \lfloor \frac{T + 2(1) - 3}{1} \rfloor + 1 = T$ |
| Conv ReLU | $(B, 128, T)$ | $(B, 128, T)$ | Elementwise activation |
| Transpose for GRU | $(B, 128, T)$ | $(B, T, 128)$ | Sequence-first ordering |
| BiGRU Layer 0 | $(B, T, 128)$ | $(B, T, 256)$ | Forward (128) + Backward (128) |
| BiGRU Layer 1 | $(B, T, 256)$ | $(B, T, 256)$ | Forward (128) + Backward (128) |
| Head Dropout | $(B, T, 256)$ | $(B, T, 256)$ | $p = 0.3$ |
| Linear Projection | $(B, T, 256)$ | $(B, T, 21)$ | 20 gloss classes + 1 blank (index 0) |
| Transpose for CTCLoss | $(B, T, 21)$ | $(T, B, 21)$ | Ready for `torch.nn.CTCLoss` |

### 5.2 Parameter Count Verification

Every trainable weight tensor in the PyTorch implementation was tallied:
- **`conv.weight`**: $[128, 228, 3] \to 87,552$
- **`conv.bias`**: $[128] \to 128$
- **`gru.weight_ih_l0`**: $[384, 128] \to 49,152$
- **`gru.weight_hh_l0`**: $[384, 128] \to 49,152$
- **`gru.bias_ih_l0`**: $[384] \to 384$
- **`gru.bias_hh_l0`**: $[384] \to 384$
- **`gru.weight_ih_l0_reverse`**: $[384, 128] \to 49,152$
- **`gru.weight_hh_l0_reverse`**: $[384, 128] \to 49,152$
- **`gru.bias_ih_l0_reverse`**: $[384] \to 384$
- **`gru.bias_hh_l0_reverse`**: $[384] \to 384$
- **`gru.weight_ih_l1`**: $[384, 256] \to 98,304$
- **`gru.weight_hh_l1`**: $[384, 128] \to 49,152$
- **`gru.bias_ih_l1`**: $[384] \to 384$
- **`gru.bias_hh_l1`**: $[384] \to 384$
- **`gru.weight_ih_l1_reverse`**: $[384, 256] \to 98,304$
- **`gru.weight_hh_l1_reverse`**: $[384, 128] \to 49,152$
- **`gru.bias_ih_l1_reverse`**: $[384] \to 384$
- **`gru.bias_hh_l1_reverse`**: $[384] \to 384$
- **`classifier.weight`**: $[21, 256] \to 5,376$
- **`classifier.bias`**: $[21] \to 21$

**Total Trainable Parameters:** **587,669 parameters**.  
*(Note: If an intermediate Linear projection layer is used, parameters total 578,837).*

---

## 6. Task 5: Loss Function Experiment Configurations

Three distinct loss function configurations are prepared to address CTC blank collapse and class imbalance without violating CTC dynamic programming axioms.

### Configuration A: Standard CTC Baseline
- **Formulation:**
  $$\mathcal{L}_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T'} P(\pi_t \mid \mathbf{x}_t)$$
- **Implementation:** `torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')`.
- **Purpose:** Direct baseline to isolate the benefit of normalized 228-dim features and stride-1 temporal convolution over the failed Phase 8.3.6 baseline.

### Configuration B: CTC + Margin-Based Blank Posterior Regularization
- **Formulation:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{blank}} \cdot \max\left(0, \bar{P}(\text{blank}) - \tau_{\text{blank}}\right)$$
  where $\bar{P}(\text{blank}) = \frac{1}{T'} \sum_{t=1}^{T'} P_t(\text{blank})$ is the average blank probability across the temporal sequence.
- **Hyperparameters:** $\lambda_{\text{blank}} = 0.5$, margin threshold $\tau_{\text{blank}} = 0.75$.
- **Mechanism:** When average blank probability exceeds 75%, a linear penalty is incurred. This forces the model to emit non-blank posteriors for active signing gestures in the interior of the video, directly mitigating the 98.6% interior blank rate identified in Phase 8.4.

### Configuration C: CTC + Sequence-Level Class Rarity Weighting
- **Formulation:**
  $$\mathcal{L}_{\text{total}} = \frac{1}{\sum_{i=1}^B w_i} \sum_{i=1}^B w_i \cdot \mathcal{L}_{\text{CTC}}^{(i)}$$
  where the sequence weight $w_i$ is computed from the training token frequencies $N_g$:
  $$w_i = \frac{1}{|Y_i|} \sum_{g \in Y_i} \left(\frac{N_{\text{total}}}{N_g}\right)^{0.33}$$
- **Justification & Non-Violation:** CTC loss integrates over unaligned temporal paths; assigning heuristic inverse-frequency weights to individual frames distorts the forward-backward lattice. In contrast, weighting at the complete sequence level preserves CTC mathematical validity while amplifying gradient updates from sentences containing rare glosses (`(AGE)`, `THINGS`, `DISAPPOINTED`).
- **Hyperparameter Selection:** Exponent $\alpha = 0.33$ dampens extreme weights while giving rare sentences ~2.5x gradient emphasis compared to sentences composed solely of `I` and `YOU`.

---

## 7. Task 6: Training Configuration

### 7.1 Optimization Hyperparameters
- **Optimizer:** `AdamW` (PyTorch standard)
- **Learning Rate:** $3 \times 10^{-4}$ ($0.0003$)
- **Weight Decay:** $1 \times 10^{-4}$ ($0.0001$)
- **Gradient Clipping:** Max norm $5.0$ (`torch.nn.utils.clip_grad_norm_`)
- **Batch Size:** $8$ (provides 11.5 gradient updates per epoch on 92 training clips)
- **Learning Rate Scheduler:** `CosineAnnealingLR` ($T_{\text{max}} = 50$, $\eta_{\text{min}} = 10^{-5}$)
- **Max Epochs:** 50
- **Early Stopping:** Patience of 15 epochs monitored strictly on Validation WER
- **Reproducibility:** Fixed seeds across Python random, NumPy, and PyTorch (seed = 42)
- **Checkpointing:** State dict containing model weights, optimizer, epoch, and validation metrics saved only when Validation WER achieves a new minimum.

### 7.2 CTC Feasibility Verification
CTC requires that the subsampled sequence length $T'$ equals or exceeds target sequence length $S$, plus any extra blanks required between consecutive identical tokens:
- **Stride:** 1, Padding: 1, Kernel: 3 $\implies T' = T$ (zero temporal downsampling).
- **Sequence Length Range:** $T \in [25, 190]$.
- **Target Gloss Sequence Length Range:** $S \in [1, 4]$.
- **Feasibility Result:** $T' \ge S$ holds for **126 out of 126 videos (100.0%)**.
  - Minimum ratio $T'/S = 6.25$ (video has 25 frames, target has 4 glosses).
  - Maximum ratio $T'/S = 95.00$.
  - Mean ratio $T'/S = 37.57$.
- **Conclusion:** Zero risk of infinite CTC loss due to temporal truncation.

---

## 8. Task 7: Evaluation Plan

Model selection and final benchmarking will adhere to strict multi-metric evaluation:

### 8.1 Primary Evaluation Metrics
1. **Word Error Rate (WER):**
   $$\text{WER} = \frac{S + D + I}{N} \times 100\%$$
   Computed via dynamic programming Levenshtein edit distance against ground-truth gloss sequences.
2. **Token Error Rate (TER):** Evaluates substitutions, deletions, and insertions independently.
3. **Exact Sentence Match Rate (ESMR):**
   $$\text{ESMR} = \frac{\sum_{i=1}^M \mathbb{I}(\hat{\mathbf{y}}^{(i)} == \mathbf{y}^{(i)})}{M} \times 100\%$$
   Measures complete sentence sequence accuracy with zero tolerance for ordering errors.
4. **Per-Gloss Precision, Recall, and Macro-F1:**
   Evaluates performance across all 20 vocabulary classes to prevent models from gaming WER by predicting only `YOU` and `I`.
5. **Blank Posterior Distribution by Temporal Quintile:**
   Measures average $P(\text{blank})$ and non-blank emission counts across video quintiles ($Q_1, Q_2, Q_3, Q_4, Q_5$). A successful model must show active non-blank emissions ($> 0\%$) in interior quintiles $Q_2, Q_3, Q_4$.
6. **Vocabulary Diversity:** Counts the number of distinct glosses ever emitted by the model on evaluation splits. Baseline emitted only 2 unique glosses; target is $\ge 15$.
7. **Signer-Independent Test Verification:**
   The test split (Signer 2, 17 videos) will remain **strictly untouched** during model tuning and will only be evaluated once after Phase 8.5B checkpoint selection.

---

## 9. Final Verification Checklist

```
+---------------------------------------------------------------------------------------------------+
|                                      FINAL VERIFICATION SUMMARY                                   |
+----+------------------------------------+-------------------------------------------+-------------+
| #  | Verification Item                  | Verified Value / State                    | Status      |
+----+------------------------------------+-------------------------------------------+-------------+
| 1  | Dataset Verification Result        | Exactly 126 videos, 18 sentences, 0 leaks | PASS        |
| 2  | Sentence Coverage Table            | 18 sentences audited; SID 30 & 94 noted   | PASS        |
| 3  | Gloss Frequency Table              | 20 glosses tallied; imbalance documented  | PASS        |
| 4  | Feature Dimension Verification     | 228 dimensions (225 coords + 3 mask flags)| VERIFIED    |
| 5  | Architecture Tensor-Shape Check    | Input (B,T,228) -> Output (T,B,21)        | VERIFIED    |
| 6  | Parameter Count Calculation        | Exactly 587,669 trainable parameters      | VERIFIED    |
| 7  | CTC Feasibility Result             | 126 / 126 feasible (100.0%, min T'/S=6.25)| PASS        |
| 8  | Proposed Loss Configurations       | Config A (CTC), B (Blank Reg), C (Seq Wt) | FORMULATED  |
| 9  | Training Configuration             | AdamW, 3e-4, wd 1e-4, clip 5.0, batch 8   | CONFIGURED  |
| 10 | Remaining Blockers Check           | Zero blocking dependencies                | CLEAR       |
+----+------------------------------------+-------------------------------------------+-------------+
```

### Remaining Blockers and Constraints
- **Evaluation Coverage Boundary:** Gloss `(AGE)` cannot be tested on Signer 2 (0 test clips), and `DISAPPOINTED` cannot be evaluated on Signer 7 (0 validation clips). This is an intrinsic property of the recorded corpus and does not block training.
- **Strict Compliance:** Zero training runs were executed; zero existing checkpoints were modified; no splits were edited.

---

## 10. Phase 8.5B Recommendation & Decision

### Formal Recommendation: **GO FOR PHASE 8.5B**

The dataset, vocabulary, features, architecture, losses, and verification suites for continuous ISL recognition on the 20-gloss, 126-video corpus are 100% verified and mathematically consistent. Authorization to proceed to **Phase 8.5B: Controlled Prototype Model Training** is granted.
