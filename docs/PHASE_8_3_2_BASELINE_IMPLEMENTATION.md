# Phase 8.3.2: Bi-GRU + CTC Baseline Implementation

**Project:** Signova AI — Continuous Indian Sign Language Recognition  
**Phase:** 8.3.2 — Baseline Model Implementation & Empirical Evaluation  
**Date:** November 13, 2026  
**Auditor / Engineer:** DeepMind Agent (Antigravity)  
**Dataset:** `datasets/islvt_continuous/` (Single Signer, Mendeley ISLVT)  
**Status:** COMPLETE EXPERIMENT — BASELINE VALIDATED & BENCHMARKED (19/19 Tests Passed)  

---

## Executive Summary

Phase 8.3.2 implemented, trained, and evaluated an experimental **Bi-GRU + CTC** baseline for continuous Indian Sign Language (ISL) recognition on the approved 4-sign vocabulary (`I`, `GO`, `EAT`, `FOOD`).

The implementation adheres strictly to the architectural constraints established in Phase 8.3.1. Training was conducted on **24 recordings across 12 sentences**, and evaluated on **8 hold-out recordings across 4 sentences**. All paired camera views (front and side) were strictly kept together in their respective splits to eliminate visual data leakage.

### Key Empirical Findings

1. **Pipeline Convergence:** The Bi-GRU + CTC architecture successfully converges from initial training loss of **120.6857** to **7.2141** in 100 epochs using AdamW without numerical instability or gradient explosion.
2. **Severe Overfitting Gap:** Final test CTC loss reached **159.3718** (vs. 7.2141 train loss), directly reflecting the severe sample-complexity deficit of training 584,197 parameters on only 12 unique sentences from a single signer.
3. **Exact Sequence Match:**
   - **Recording-Level Accuracy:** **25.0%** (2 / 8 test recordings matched exactly).
   - **Sentence-Level Strict Accuracy:** **25.0%** (1 / 4 sentences: Sentence 21 matched on both camera takes).
   - **Sentence-Level Relaxed Accuracy:** **25.0%** (1 / 4 sentences).
4. **Token Error Rate (TER):** **50.0%** (Token accuracy: **50.0%**).
5. **No Blank-Only Collapse:** Blank prediction percentage was **0.0%** across test timesteps; the model did not collapse into emitting all-blank tokens.
6. **Failure on Length-2 Transitions:** The model predicted single-token outputs (`['i']`) for all length-2 test sentences (`['i', 'eat']` and `['i', 'go']`), directly confirming the distribution shift identified in Phase 8.3.1 (zero length-2 sequences existed in training).
7. **Production Boundary:** This is an isolated research experiment on a single-signer dataset. It is **not** suitable for production continuous ISL translation.

---

## 1. Model Architecture & Tensor Dimensions

The model was implemented in PyTorch (`ai/continuous/bigru_ctc.py`) strictly conforming to the Phase 8.3.1 architectural specification.

```
Input Spatial Features: (B, T, 300)
    │
    ▼
1. Linear Feature Projection:
    Linear(in_features=300, out_features=128)
    LayerNorm(normalized_shape=128)
    Dropout(p=0.3)
    ReLU()
    │
    ▼
2. Temporal Subsampling (Stride-2 Conv1D):
    Permute to (B, 128, T)
    Conv1D(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
    Permute to (B, T', 128)
    LayerNorm(normalized_shape=128)
    ReLU()
    [Subsampled length: T' = (T + 1) // 2]
    │
    ▼
3. Bidirectional Recurrent Encoder:
    Bi-GRU Layer 1: hidden_size=128, bidirectional=True (output 256)
    Bi-GRU Layer 2: hidden_size=128, bidirectional=True (output 256)
    Dropout(p=0.3)
    │
    ▼
4. Classification Head:
    Dropout(p=0.3)
    Linear(in_features=256, out_features=5)
    LogSoftmax(dim=-1)
    │
    ▼
Output Tensor: (T', B, 5) -> torch.nn.CTCLoss(blank=0, zero_infinity=True)
```

### Verified Tensor Geometry Throughout Forward Pass

| Module / Operation | Input Shape | Output Shape | Parameters | Verified via Test |
| :--- | :---: | :---: | :---: | :---: |
| **Input Features** | $(B, T, 300)$ | $(B, T, 300)$ | 0 | `test_model_forward_dimensions` |
| **Linear Projection** | $(B, T, 300)$ | $(B, T, 128)$ | 38,528 | `test_model_forward_dimensions` |
| **Projection LayerNorm** | $(B, T, 128)$ | $(B, T, 128)$ | 256 | `test_model_forward_dimensions` |
| **Conv1D (stride=2)** | $(B, 128, T)$ | $(B, 128, T')$ | 49,280 | `test_subsampled_length_formula` |
| **Conv LayerNorm** | $(B, T', 128)$ | $(B, T', 128)$ | 256 | `test_model_forward_dimensions` |
| **2-Layer Bi-GRU** | $(B, T', 128)$ | $(B, T', 256)$ | 494,592 | `test_gru_bidirectional_dimension` |
| **Linear Classifier** | $(B, T', 256)$ | $(B, T', 5)$ | 1,285 | `test_gru_bidirectional_dimension` |
| **Total Model** | $(B, T, 300)$ | $(T', B, 5)$ | **584,197** | `test_ctc_loss_and_gradient_flow` |

### Subsampled Length Calculation

For Conv1D with $K=3, S=2, P=1$:
$$T' = \left\lfloor \frac{T + 2(1) - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{T - 1}{2} \right\rfloor + 1 = \left\lfloor \frac{T + 1}{2} \right\rfloor$$

This integer formula was verified across all sequence lengths ($T \in [1, 291]$) against native PyTorch convolution output in `test_subsampled_length_formula`.

---

## 2. Target Vocabulary & Mapping

The vocabulary preserves the 5-class index convention specified in the audit:

| Index | Label | Token Meaning | Category | Train Support (Videos) | Test Support (Videos) |
| :---: | :---: | :--- | :--- | :---: | :---: |
| **0** | `<blank>` | CTC Alignment Blank Token ($\epsilon$) | CTC Special | N/A | N/A |
| **1** | `i` | First-person pronoun | Lexical | 12 (6 sentences) | 4 (2 sentences) |
| **2** | `go` | Motion / directional verb | Lexical | 8 (4 sentences) | 4 (2 sentences) |
| **3** | `eat` | Ingestion action verb | Lexical | 2 (1 sentence: ID 26) | 2 (1 sentence: ID 50) |
| **4** | `food` | Nourishment / meal noun | Lexical | 4 (2 sentences: IDs 4, 31) | 2 (1 sentence: ID 34) |

---

## 3. Dataset Handling & Experimental Safeguards

### 3.1 Strict Data Partitioning
- **Train Set:** 24 recordings, 12 unique sentences (`IDs: 4, 11, 12, 13, 17, 18, 20, 26, 31, 44, 63, 67`).
- **Test Set:** 8 recordings, 4 unique sentences (`IDs: 21, 34, 50, 72`).
- **Camera Pairing Integrity:** Both front and side takes for every sentence were kept strictly within the same partition.
- **No Random Re-Splitting:** The frozen partition from Phase 8.1 and Phase 8.2 (`splits.json`) was used without alteration.

### 3.2 Absence of Independent Validation Set
As established in Phase 8.1 and Phase 8.3.1, the total dataset comprises only 16 sentences. Carving out an independent validation split would reduce training data to 10 sentences and starve classes with minimal support (`eat` and `food`).
- **Safeguard Enforced:** No validation split was fabricated.
- **Zero Test Leakage:** The hold-out test set was **strictly never used** for validation loss monitoring, early stopping, hyperparameter tuning, or model selection.
- **Fixed Epoch Training:** Training was run for a fixed schedule of 100 epochs, and the final checkpoint was evaluated once on the test set.

### 3.3 Recording vs Sentence Tracking
Sentences and recordings were audited separately:
- 24 training recordings $\neq$ 24 independent samples; they represent 12 sentence performances executed in two camera orientations.
- 8 test recordings $\neq$ 8 independent samples; they represent 4 sentences executed in two camera orientations.
- Metrics are reported per recording take and aggregated by sentence ID.

---

## 4. Training Configuration & Hyperparameters

Deterministic training was executed via `scripts/train_islvt_bigru_ctc.py`:

| Parameter | Configuration Value | Rationale |
| :--- | :---: | :--- |
| **Random Seed** | `42` | Complete reproducibility across PyTorch, NumPy, and random |
| **Optimizer** | `AdamW` | Standard adaptive gradient with decoupled weight decay |
| **Learning Rate** | $3 \times 10^{-4}$ | Recommended CTC learning rate avoiding alignment divergence |
| **Weight Decay** | $1 \times 10^{-3}$ | L2 regularization against parameter over-capacity |
| **LR Scheduler** | `CosineAnnealingLR` | Anneals from $3 \times 10^{-4}$ to $1 \times 10^{-5}$ across 100 epochs |
| **Batch Size** | `4` | Dynamic padding with `pack_padded_sequence` |
| **Gradient Clipping** | `1.0` max norm | Prevents CTC gradient spikes during early alignment |
| **Epochs** | `100` | Full convergence on the 24-sample training set |
| **Device** | `CPU` | Intel Core architecture (deterministic CPU backend) |
| **Checkpoint Path** | `models/islvt_continuous/bigru_ctc_baseline.pt` | Isolated model persistence |

### Training Loss Trajectory

```
Epoch   1/100 | CTC Train Loss: 120.6857
Epoch  10/100 | CTC Train Loss:  85.1709
Epoch  20/100 | CTC Train Loss:  59.6989
Epoch  30/100 | CTC Train Loss:  45.4506
Epoch  40/100 | CTC Train Loss:  31.9339
Epoch  50/100 | CTC Train Loss:  31.3579
Epoch  60/100 | CTC Train Loss:  26.3084
Epoch  70/100 | CTC Train Loss:  16.8487
Epoch  80/100 | CTC Train Loss:  11.5425
Epoch  90/100 | CTC Train Loss:   8.1035
Epoch 100/100 | CTC Train Loss:   7.2141
```

---

## 5. Empirical Evaluation Results

### 5.1 Primary Test Metrics (Both Views Baseline)

| Metric | Result | Description / Diagnostic Meaning |
| :--- | :---: | :--- |
| **Final Train Loss** | **7.2141** | Loss converged cleanly over 100 epochs |
| **Validation Loss** | **N/A** | *None — Documented dataset limitation; no test data used* |
| **Test CTC Loss (All 8 Takes)** | **159.3718** | Severe generalization gap (+2,109% above train loss) |
| **Exact Match Acc (Recording)** | **25.0% (2 / 8)** | Sentence 21 Front and Sentence 21 Side matched reference |
| **Exact Match Acc (Sentence Strict)** | **25.0% (1 / 4)** | Sentence 21 correct across both takes |
| **Exact Match Acc (Sentence Relaxed)**| **25.0% (1 / 4)** | Sentence 21 correct |
| **Token-Level Accuracy** | **50.0% (6 / 12)** | Fraction of reference tokens correctly detected |
| **Token Error Rate (TER)** | **50.0%** | Total Levenshtein edit distance / Total reference tokens |
| **CTC Blank Percentage** | **0.0%** | Fraction of collapsed timesteps emitting blank token |
| **Blank-Only Collapse?** | **FALSE** | Model successfully predicted lexical tokens throughout |

---

### 5.2 Per-Recording Test Predictions (Every Sample Audited)

The table below documents the exact reference sequence, decoded prediction, raw subsampled timesteps, edit distance, and exact match status for all 8 test recordings:

| Sentence ID | Camera Take | Subsampled Frames ($T'$) | Reference Sequence | Decoded Output | Raw Frame Prediction | Edit Dist | Match Status |
| :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **21** | Side | 83 | `['go']` | `['go']` | All 2s (`go`) | 0 | **EXACT MATCH** |
| **21** | Front | 77 | `['go']` | `['go']` | All 2s (`go`) | 0 | **EXACT MATCH** |
| **34** | Side | 116 | `['food']` | `['i']` | All 1s (`i`) | 1 | **MISMATCH** |
| **34** | Front | 111 | `['food']` | `['i']` | All 1s (`i`) | 1 | **MISMATCH** |
| **50** | Side | 89 | `['i', 'eat']` | `['i']` | All 1s (`i`) | 1 | **MISMATCH** |
| **50** | Front | 124 | `['i', 'eat']` | `['i']` | All 1s (`i`) | 1 | **MISMATCH** |
| **72** | Side | 89 | `['i', 'go']` | `['i']` | All 1s (`i`) | 1 | **MISMATCH** |
| **72** | Front | 100 | `['i', 'go']` | `['i']` | All 1s (`i`) | 1 | **MISMATCH** |

---

### 5.3 Per-Class Performance Breakdown

| Class Label | Support (Test Tokens) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score | Diagnostic Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`i`** | 4 | 4 | 2 | 0 | **66.67%** | **100.0%** | **80.0%** | Dominant training class (50% of train samples). Model defaults to `i`. |
| **`go`** | 4 | 2 | 0 | 2 | **100.0%** | **50.0%** | **66.67%** | Perfect on Sentence 21 (`go`); dropped on Sentence 72 (`i go`). |
| **`eat`** | 2 | 0 | 0 | 2 | **0.0%** | **0.0%** | **0.0%** | **Complete Failure:** Only 1 training sentence (ID 26). Dropped in test. |
| **`food`** | 2 | 0 | 0 | 2 | **0.0%** | **0.0%** | **0.0%** | **Complete Failure:** Replaced by `i` on Sentence 34. |

---

## 6. Camera View Analysis & Front-Only Ablation

### 6.1 Front vs. Side Test View Performance (Primary Model)

| Evaluation Subset | Sample Count | Mean CTC Test Loss | Exact Match (Recording) | Token Accuracy | Token Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Test** | 8 | 159.3718 | 25.0% | 50.0% | 50.0% |
| **Front-View Takes** | 4 | **133.3538** | 25.0% | 50.0% | 50.0% |
| **Side-View Takes** | 4 | **185.3898** | 25.0% | 50.0% | 50.0% |

**Key Finding:** Side-view recordings exhibit **39.0% higher CTC test loss** than frontal recordings (185.39 vs 133.35). This confirms the Phase 8.3.1 finding that the $90^\circ$ profile view introduces ~14% left-hand occlusion and depth distortion that degrades temporal feature alignment.

### 6.2 Ablation Study: Training on Front-View Only vs. Both Views

To determine whether side-view training data helps or harms continuous sequence learning, an ablation model was trained strictly on the 11 frontal training recordings (Sentence 4 lacks a front take):

| Model Variant | Train Takes | Train CTC Loss | Test CTC Loss (All) | Test Match (Overall) | Test Match (Front) | Test Match (Side) | TER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary (Both Views)** | 24 | **7.21** | 159.37 | **25.0%** | 25.0% | **25.0%** | **50.0%** |
| **Ablation (Front-Only)**| 11 | 28.31 | **91.81** | **12.5%** | 25.0% | **0.0%** | **83.33%** |

#### Ablation Insights:
1. When trained only on front takes, the model completely collapsed on side-view test takes (**0.0% exact match** on side view vs. 25.0% in primary model).
2. The front-only model produced spurious repetitive predictions on Sentence 21 Side (`['i', 'go', 'i']`) and Sentence 34 Front (`['i', 'go', 'i']`), driving the Token Error Rate up to **83.33%**.
3. **Conclusion:** Training on both camera views provides valuable angle regularization that stabilizes predictions across viewpoints, even though side views suffer from partial occlusion.

---

## 7. Deep Root-Cause Analysis of Baseline Failures

The empirical results reveal four decisive failure modes inherent to continuous sequence modeling on this small dataset:

### 1. Structural Distribution Shift (The "Zero Length-2" Deficit)
- **Train Set Distribution:** 11 of 12 sentences had target length $L=1$; exactly 1 sentence had target length $L=3$ (Sentence 26: `['eat', 'i', 'eat']`). **Zero** sentences had target length $L=2$.
- **Test Set Distribution:** 2 of 4 sentences had target length $L=2$ (`['i', 'eat']` in ID 50 and `['i', 'go']` in ID 72).
- **Observed Behavior:** The model had never observed a two-token transition in training. In both Sentence 50 and Sentence 72, the model emitted only the first token `['i']` and completely dropped the trailing verb.

### 2. Extreme `EAT` Under-Representation
- The sign `EAT` appears in only **one sentence** in the entire training set (Sentence 26, 2 video takes).
- While `EAT` appears twice within that sentence, both instances occur within the single compound context `first him eat after i eat`.
- Consequently, the model could not decouple the gesture for `EAT` from its immediate co-articulation neighbors, resulting in **0.0% recall** on test.

### 3. Majority-Class Prior Collapse
- `I` represents 42.9% of all target tokens in training and appears in 6 of 12 training sentences.
- When faced with ambiguous gesture sequences (e.g. Sentence 34 `food`, Sentence 50 `eat`), the recurrent hidden states default to the dominant token `I` (Precision = 66.7%, Recall = 100.0%).

### 4. Weak Supervision Blank Pollution
- In sentences with 4 to 6 natural words (e.g., Sentence 13: `before grand mother market go finish`), the signer performs continuous arm, body, and hand movements.
- Because our vocabulary is restricted to 4 target words, CTC is forced to map active signing movements (such as `mother`, `market`, `finish`) to the **blank token** ($\epsilon$).
- Under this contradictory supervision, the model learns that complex hand gestures are often "blank", severely impeding boundary demarcation.

---

## 8. Automated Verification Suite

The implementation was validated against a 19-test automated test suite across two verification suites:

```text
tests/test_islvt_features.py (8 tests):
  [x] test_manifest_and_summary_files_exist                 PASSED
  [x] test_video_to_annotation_mapping                     PASSED
  [x] test_landmark_tensor_dimensions                      PASSED
  [x] test_missing_landmark_handling_and_no_nans            PASSED
  [x] test_sequence_length_and_validity                    PASSED
  [x] test_duplicate_recordings_and_single_views           PASSED
  [x] test_camera_pair_consistency_and_splits             PASSED
  [x] test_metadata_schema_integrity                       PASSED

tests/test_islvt_bigru_ctc.py (11 tests):
  [x] test_vocab_definition                                PASSED
  [x] test_subsampled_length_formula                       PASSED
  [x] test_model_forward_dimensions                        PASSED
  [x] test_gru_bidirectional_dimension                     PASSED
  [x] test_ctc_loss_and_gradient_flow                     PASSED
  [x] test_ctc_repeated_label_handling                     PASSED
  [x] test_dataset_loading                                 PASSED
  [x] test_collate_function                                PASSED
  [x] test_greedy_decoder                                  PASSED
  [x] test_blank_only_collapse_detection                   PASSED
  [x] test_levenshtein_edit_distance                       PASSED
----------------------------------------------------------------------
Result: 19 passed in 5.46s (100% test pass rate)
```

---

## 9. Restrictions & Boundaries Reaffirmed

1. **Production System Untouched:** The production 10-class isolated-sign model (`models/checkpoints/isl_isolated_classifier.pt`) was completely unaltered.
2. **Feature Integrity Preserved:** All 32 `.npz` feature archives extracted in Phase 8.2 remain bit-for-bit identical.
3. **Frontend & Backend Unchanged:** No changes were made to the FastAPI backend or React frontend.
4. **No Claim of General Continuous ISL Recognition:** This baseline operates strictly within a 4-sign single-signer research experimental subset. It cannot translate open-vocabulary continuous ISL sentences.
5. **No Production Deployment:** This experiment is retained as a research baseline and benchmark for Phase 8.

---

## 10. Summary File Artifacts Created

| File | Purpose |
| :--- | :--- |
| `ai/continuous/bigru_ctc.py` | PyTorch Bi-GRU + CTC model, dataset loader, greedy decoder, and edit distance metrics |
| `scripts/train_islvt_bigru_ctc.py` | Deterministic training runner, evaluation pipeline, checkpoint serializer, and ablation suite |
| `tests/test_islvt_bigru_ctc.py` | 11 automated unit and integration tests verifying tensor dimensions, loss, and decoding |
| `models/islvt_continuous/bigru_ctc_baseline.pt` | Saved model weights and metadata for the primary 24-sample baseline |
| `models/islvt_continuous/bigru_ctc_baseline_front.pt` | Saved model weights for the 11-sample front-only ablation |
| `datasets/islvt_continuous/baseline_results.json` | Complete machine-readable results, loss history, and per-sample predictions |
| `docs/PHASE_8_3_2_BASELINE_IMPLEMENTATION.md` | Comprehensive experimental audit report (this document) |

---
*Report compiled autonomously by DeepMind Agent (Antigravity) on November 13, 2026 for Signova AI Phase 8.3.2.*
