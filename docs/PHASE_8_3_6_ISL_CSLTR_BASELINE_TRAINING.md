# Phase 8.3.6: Accuracy-First Continuous ISL Recognition Baseline Training Report

**Project:** Signova AI — Continuous Indian Sign Language Recognition  
**Phase:** 8.3.6 — Accuracy-First Continuous Baseline Model & Training Evaluation  
**Dataset:** ISL-CSLTR Continuous Feature Corpus (`datasets/isl_csltr_continuous/`)  
**Date:** November 13, 2026  
**Status:** COMPLETE (Reproducible Baseline Established; Honest Failure-Mode Analysis Documented)  
**Primary Outcome:** Baseline model successfully converged (Val WER: 82.57%, Test WER: 83.09%) exhibiting characteristic CTC mode collapse onto dominant class priors (`YOU`, `I`) and blank dominance (96.3% blanks). 14/14 unit & regression tests passing.

---

## Executive Summary

Phase 8.3.6 establishes the first end-to-end continuous Indian Sign Language (ISL) recognition baseline trained on the standardized ISL-CSLTR feature corpus. Using a **Temporal Conv1D (stride=2) + 2-layer Bidirectional GRU (hidden=128) + CTC** architecture (626,859 parameters), the model was trained across 499 canonical samples from 5 signers (Signers 1, 3, 4, 5, 6), validated on Signer 7 (85 videos), and evaluated on an unseen native Deaf signer (Signer 2, 98 videos).

Per the strict mandate of Phase 8.3.6, **success is not claimed merely because training completed without error**. While training converged smoothly over 35 epochs (286.9s wall-clock time) and lowered validation CTC loss from 22.44 to 4.3026, the sequence-level recognition accuracy demonstrates that **naive CTC training on unbalanced continuous sign language features collapses into emitting only high-prior glosses (`['I', 'YOU']`)**.

An ablation experiment (Experiment 1: Shoulder-Centered Spatial Normalization) was conducted, reducing validation WER marginally from 82.57% to 82.24% while leaving test WER unchanged at 83.09%. This demonstrates that spatial coordinate scale is secondary to the true bottlenecks: **severe gloss frequency imbalance** and **CTC blank dominance**.

---

## 1. Pre-Training Audit Results

A bit-exact pre-training audit of all manifests, feature files, sequence lengths, and duplicate groups was conducted prior to model compilation:

```
+---------------------------------------------------------------------------------------------------+
|                                 PRE-TRAINING AUDIT SPECIFICATION                                  |
+----------------------+-------------------+-------------------+-------------------+----------------+
| Metric               | Train Partition   | Validation Split  | Test Split        | Whole Corpus   |
+----------------------+-------------------+-------------------+-------------------+----------------+
| Total Video Clips    | 504               | 85                | 98                | 687            |
| Canonical Clips      | 499               | 85                | 98                | 682            |
| Non-Canonical Dups   | 5                 | 0                 | 0                 | 5 (10 files)   |
| Assigned Signers     | 1, 3, 4, 5, 6     | 7                 | 2                 | 1, 2, 3, 4, 5, 6, 7|
| Resolution Profile   | 98 QVGA, 406 FHD  | 85 FHD (1080p)    | 97 QVGA, 1 FHD    | 195 QVGA, 492 FHD|
| Min Frame Length     | 25 frames         | 58 frames         | 25 frames         | 25 frames      |
| Median Frame Length  | 97.0 frames       | 106.0 frames      | 50.0 frames       | 95.0 frames    |
| Max Frame Length     | 192 frames        | 186 frames        | 200 frames        | 200 frames     |
| Target Tokens / Seq  | 1 - 8 (mean 3.56) | 1 - 8 (mean 3.58) | 1 - 8 (mean 3.56) | 1 - 8 (mean 3.56)|
| Missing NPZ Files    | 0                 | 0                 | 0                 | 0 (100% on disk)|
| CTC Invalid Samples  | 0                 | 0                 | 0                 | 0 (all T' >= S)|
+----------------------+-------------------+-------------------+-------------------+----------------+
```

### CTC Length Constraint Verification
Under the temporal Conv1D stride of 2, subsampled output lengths are calculated as:
$$T' = \left\lfloor \frac{T + 1}{2} \right\rfloor$$
- Shortest clip: $T = 25 \rightarrow T' = 13$ frames.
- Maximum target gloss sequence length: $S = 8$ tokens.
- Across all 687 clips: **$T' \ge S$ holds true for 100.0% of samples**.
- Exactly zero samples were dropped due to CTC sequence length invalidity.

---

## 2. Final Vocabulary and Class Count

- **Target ISL Glosses:** Exactly 170 distinct glosses.
- **CTC Blank Token:** Index `0` (`<blank>`).
- **Total Output Classes:** **171** (`nn.Linear(256, 171)`).
- **Index Assignment:**
  - Index `0`: `<blank>`
  - Indices `1` to `170`: Alphabetical ISL glosses (`(AGE)`, `A`, `ABOUT`, ..., `YOU`, `YOUR`)
- **Unseen Gloss Audit:**
  - Glosses in Validation not in Train: **0** (100% coverage).
  - Glosses in Test not in Train: **0** (100% coverage).
  - *Note:* In the canonical training split (499 videos), two glosses (`DONT` and `NO`) only occurred in duplicate copies of synonymous sentences (`MVI_6494.MP4` and `MVI_6496.MP4`). Neither `DONT` nor `NO` appears in the validation or test splits.

---

## 3. Effective Training Sample Count and Duplicate Handling

The audit confirmed 5 duplicate groups (10 files total). All 10 redundant files originated from student volunteer recording sessions and resided strictly in the training partition:

```
+--------------------------------------------------------------------------------------------------------------------+
|                                         DUPLICATE REMOVAL AUDIT IN TRAIN                                           |
+------------+----------------+-------------------------------+--------------------------+---------------------------+
| Group Hash | Duplicate ID   | Video Filename                | Sentence Annotation      | Primary Canonical File    |
+------------+----------------+-------------------------------+--------------------------+---------------------------+
| 91203a8f   | s015_signer3   | something (2).MP4             | do you need something    | are you hiding something  |
| 34d79e4e   | s049_signer3   | MVI_6494.MP4                  | i dont agree             | i do not agree            |
| ffcf9be3   | s052_signer5   | MVI_6814.MP4                  | i like you i love you    | i like you i love you     |
| d96f0165   | s063_signer3   | MVI_6496.MP4                  | No need to worry dont... | do not worry              |
| 67170eaa   | s064_signer3   | never hurt.MP4                | now onwards he will...   | what have you planned...  |
+------------+----------------+-------------------------------+--------------------------+---------------------------+
```

- **Manifest Total:** 504 videos.
- **Excluded Redundant Copies:** 5 videos (`is_canonical == False`).
- **Effective Training Sample Count:** **499 videos** (1,768 total token occurrences).

---

## 4. Model Architecture and Parameters

The architecture follows a minimal, standard PyTorch implementation without unrequested framework dependencies:

```
                                [ Input Landmarks: (B, T, 300) ]
                                                │
                                    Linear(300, 128) + LayerNorm
                                         Dropout(0.3) + ReLU
                                                │
                                     (B, T, 128) Feature Map
                                                │
                          Conv1d(128, 128, kernel=3, stride=2, pad=1)
                                         LayerNorm + ReLU
                                                │
                                   (B, T', 128) Subsampled Map
                                                │
                             2-Layer Bidirectional GRU (hidden=128)
                                    [packed_padded_sequence]
                                                │
                                     (B, T', 256) Context Map
                                                │
                                    Dropout(0.3) + Linear(256, 171)
                                                │
                                       LogSoftmax(dim=-1)
                                                │
                            [ CTC Loss Log-Probs: (T', B, 171) ]
```

### Parameter Breakdown

| Component | Layer Specification | Input Dim | Output Dim | Trainable Parameters |
| :--- | :--- | :---: | :---: | :---: |
| **Feature Projection** | `nn.Linear` + `LayerNorm` | 300 | 128 | 38,784 |
| **Temporal Subsampling** | `nn.Conv1d` (stride=2) + `LayerNorm` | 128 | 128 | 49,536 |
| **Bidirectional GRU Layer 1** | `nn.GRU` (2 directions) | 128 | 256 | 198,144 |
| **Bidirectional GRU Layer 2** | `nn.GRU` (2 directions) | 256 | 256 | 296,448 |
| **Classifier Head** | `nn.Linear` | 256 | 171 | 43,947 |
| **Total Model Parameters** | — | — | — | **626,859** |

---

## 5. Training Configuration

- **Dataset Split:** Train (Signers 1, 3, 4, 5, 6; 499 canonical samples)
- **Validation Split:** Signer 7 (85 samples)
- **Batch Size:** 16 (variable-length zero-padding with length tracking)
- **Optimizer:** `AdamW` (learning rate: $3 \times 10^{-4}$, weight decay: $1 \times 10^{-4}$)
- **Learning Rate Scheduler:** `CosineAnnealingLR` ($T_{\max} = 80$, $\eta_{\min} = 1 \times 10^{-5}$)
- **Gradient Clipping:** Max norm = 5.0
- **Loss Function:** `torch.nn.CTCLoss(blank=0, zero_infinity=True)`
- **Early Stopping:** Patience = 15 epochs monitored on Validation WER
- **Execution Device:** Intel Core 12-core CPU (`torch.device("cpu")`, 8 worker threads)
- **Seed:** 42 (deterministic `random`, `np.random`, `torch.manual_seed`)

---

## 6. Training and Validation Progression

Training ran for **35 epochs** before the 15-epoch early stopping trigger terminated the run:

```
+-------+--------------+------------+-----------+--------------+-----------+--------------+------------+
| Epoch | Train Loss   | Val Loss   | Val WER   | Exact Match  | Blank %   | Learning Rate| Epoch Time |
+-------+--------------+------------+-----------+--------------+-----------+--------------+------------+
| 1     | 22.4399      | 5.3829     | 100.00%   | 0.00%        | 100.0%    | 0.000300     | 10.7s *    |
| 2     | 5.0819       | 4.9645     | 100.00%   | 0.00%        | 100.0%    | 0.000300     | 10.9s      |
| 5     | 4.8734       | 4.8010     | 100.00%   | 0.00%        | 100.0%    | 0.000297     | 9.6s       |
| 10    | 4.6226       | 4.6158     | 96.71%    | 0.00%        | 99.5%     | 0.000289     | 6.4s *     |
| 12    | 4.5464       | 4.5052     | 91.45%    | 0.00%        | 98.6%     | 0.000284     | 6.5s *     |
| 13    | 4.5210       | 4.4729     | 89.14%    | 0.00%        | 98.1%     | 0.000282     | 6.8s *     |
| 17    | 4.4006       | 4.3480     | 86.84%    | 0.00%        | 97.7%     | 0.000269     | 6.5s *     |
| 19    | 4.3550       | 4.3123     | 84.54%    | 0.00%        | 97.0%     | 0.000261     | 9.9s *     |
| 20    | 4.3464       | 4.3026     | 82.57%    | 0.00%        | 96.3%     | 0.000258     | 10.4s *    |
| 25    | 4.2657       | 4.2314     | 82.89%    | 0.00%        | 96.2%     | 0.000236     | 7.5s       |
| 30    | 4.1740       | 4.1746     | 82.89%    | 0.00%        | 96.2%     | 0.000210     | 6.4s       |
| 35    | 4.1063       | 4.1569     | 82.89%    | 0.00%        | 96.2%     | 0.000183     | 6.4s       |
+-------+--------------+------------+-----------+--------------+-----------+--------------+------------+
```
*\* Denotes new best model checkpoint based on Validation WER.*

**Total Training Duration:** 286.93 seconds (~4.78 minutes).

---

## 7. Best Checkpoint Details

- **Path:** `models/isl_csltr_continuous/best_checkpoint.pt`
- **Checkpoint Epoch:** **Epoch 20**
- **Validation Metrics at Best Checkpoint:**
  - CTC Loss: `4.3026`
  - Word Error Rate (WER): `82.57%`
  - Token Error Rate (TER): `82.57%`
  - Sentence Exact Match: `0.00%`
  - Macro F1: `0.0053`
  - Mean Blank Percentage: `96.3%`
- **Reproducibility Artifacts Saved:**
  - `model_config.json`: Bit-exact architecture and parameter dimensions.
  - `training_config.json`: Complete hyperparameter and seed manifest.
  - `vocabulary.json`: Complete 171-class index-to-gloss dictionary.
  - `epoch_history.json`: Per-epoch metrics for all 35 epochs.
  - `evaluation_results.json`: Full evaluation results and sample transcriptions.

---

## 8. Primary Sequence-Level Evaluation Metrics

Evaluation was conducted using greedy CTC decoding (consecutive repeat collapse + blank elimination):

```
+------------------------------------+--------------------------+--------------------------+
| Evaluation Metric                  | Validation (Signer 7)    | Test (Signer 2)          |
+------------------------------------+--------------------------+--------------------------+
| Video Clip Count                   | 85                       | 98                       |
| Signer Profile                     | Student Volunteer (FHD)  | Native Deaf Signer (QVGA)|
| Total Reference Tokens             | 304                      | 349                      |
| CTC Loss                           | 4.3026                   | 4.3602                   |
| Word Error Rate (WER)              | **82.57%**               | **83.09%**               |
| Token Error Rate (TER)             | **82.57%**               | **83.09%**               |
| Sentence Exact-Match Accuracy      | **0.00%** (0 / 85)       | **0.00%** (0 / 98)       |
| Edit Distance Operations:          |                          |                          |
| - Substitutions                    | 114                      | 123                      |
| - Deletions                        | 136                      | 166                      |
| - Insertions                       | 1                        | 1                        |
| - Total Distance                   | 251                      | 290                      |
| Macro F1 Score                     | 0.0053                   | 0.0046                   |
| Mean Blank Frame Percentage        | 96.3%                    | 92.0%                    |
+------------------------------------+--------------------------+--------------------------+
```

---

## 9. Per-Gloss Performance and Imbalance Analysis

### Gloss Frequency in Canonical Train
The training distribution exhibits extreme class imbalance:
- Total token instances: 1,768
- `YOU`: 205 instances (11.6% of entire corpus)
- `I`: 161 instances (9.1% of entire corpus)
- Top 2 glosses (`YOU`, `I`): **20.7% of all tokens**
- Remaining 166 glosses: Median frequency = **5 instances** (many having only 2 or 3 occurrences)

### Output Hypothesis Distribution
Because greedy CTC decoding with standard loss heavily penalizes false insertions, the model collapsed into emitting only the highest-prior tokens:
- **Validation Hypotheses (85 clips):**
  - Predicted gloss `YOU`: 85 times
  - Predicted gloss `I`: 84 times
  - Predicted other glosses: **0 times**
- **Test Hypotheses (98 clips):**
  - Predicted gloss `YOU`: 98 times
  - Predicted gloss `I`: 86 times
  - Predicted other glosses: **0 times**

### Gloss-Level Precision, Recall, and F1

```
+------------+-------------+--------------+------------+-----------+--------+
| Gloss      | Ref Occurrences | Hyp Predictions | Precision  | Recall    | F1     |
+------------+-------------+--------------+------------+-----------+--------+
| **YOU**    | 37 (Val)    | 85           | 25.88%     | 59.46%    | 0.3607 |
| **I**      | 24 (Val)    | 84           | 20.24%     | 70.83%    | 0.3148 |
| **ALL OTHER 168 GLOSSES** | 243 (Val) | 0 | 0.00% | 0.00% | 0.0000 |
+------------+-------------+--------------+------------+-----------+--------+
```

---

## 10. Confusion and Error Pattern Analysis

Because the baseline collapsed into emitting `['I', 'YOU']` or `['YOU']` for every sentence, the edit distance is dominated by **deletions of all intermediate vocabulary** and **substitutions of non-dominant tokens**:

### Top 10 Confusion Pairs in Validation

```
+--------------------+----------------------+---------------+------------------------------------------+
| Reference Gloss    | Predicted Hypothesis | Count in Val  | Error Type                               |
+--------------------+----------------------+---------------+------------------------------------------+
| YOU                | <deletion>           | 15            | Deleted when sentence had repeated YOU   |
| WHAT               | I                    | 7             | Substituted by dominant prior            |
| DO                 | I                    | 6             | Substituted by dominant prior            |
| IT                 | <deletion>           | 6             | Unrecognized rare gloss                  |
| HE                 | I                    | 5             | Pronoun confusion                        |
| DO                 | YOU                  | 5             | Substituted by dominant prior            |
| NOT                | YOU                  | 4             | Substituted by dominant prior            |
| HOW                | I                    | 4             | Substituted by dominant prior            |
| NOT                | <deletion>           | 4             | Unrecognized rare gloss                  |
| ME                 | YOU                  | 3             | Pronoun confusion                        |
+--------------------+----------------------+---------------+------------------------------------------+
```

### Top 10 Confusion Pairs in Test (Signer 2)

```
+--------------------+----------------------+---------------+------------------------------------------+
| Reference Gloss    | Predicted Hypothesis | Count in Test | Error Type                               |
+--------------------+----------------------+---------------+------------------------------------------+
| YOU                | <deletion>           | 19            | Deleted due to native signer cadence     |
| WHAT               | I                    | 8             | Substituted by dominant prior            |
| DO                 | I                    | 6             | Substituted by dominant prior            |
| IT                 | <deletion>           | 6             | Unrecognized rare gloss                  |
| ME                 | <deletion>           | 5             | Unrecognized rare gloss                  |
| DO                 | YOU                  | 5             | Substituted by dominant prior            |
| NOT                | <deletion>           | 5             | Unrecognized rare gloss                  |
| HE                 | I                    | 5             | Pronoun confusion                        |
| DO                 | <deletion>           | 5             | Unrecognized rare gloss                  |
| NOT                | YOU                  | 4             | Substituted by dominant prior            |
+--------------------+----------------------+---------------+------------------------------------------+
```

---

## 11. Test Evaluation & Resolution Confounding

The test set evaluation was performed **strictly once** on the best checkpoint without hyperparameter tuning:

```
+----------------------------------------------------------------------------------------------------+
|                                    SPLIT PROFILE COMPARISON                                        |
+--------------------+-----------------------+-------------------------+-----------------------------+
| Feature            | Train Split           | Validation Split        | Test Split                  |
+--------------------+-----------------------+-------------------------+-----------------------------+
| Signers            | Signers 1, 3, 4, 5, 6 | Signer 7                | Signer 2                    |
| Signer Category    | 1 Native + 4 Students | 1 Student Volunteer     | 1 Native Deaf Signer        |
| Camera Resolution  | 98 QVGA, 406 FHD      | 85 Full HD ($1920\times 1080$) | 97 QVGA ($320\times 240$), 1 FHD |
| Mean Frame Count   | 98.5 frames           | 111.2 frames            | 57.5 frames                 |
| Cadence            | Mixed / Moderate      | Slow, deliberate        | Fast, natural native signing|
| WER                | —                     | 82.57%                  | 83.09%                      |
+--------------------+-----------------------+-------------------------+-----------------------------+
```

### Critical Confounding Finding
As mandated by Section 8:
> **The test set does NOT measure unseen-signer generalization independently of resolution.**

Signer 2 is recorded at $320 \times 240$ (QVGA) with an average sentence duration of only 57.5 frames (~2.3s), whereas Signer 7 (Val) is recorded at $1920 \times 1080$ (FHD) with an average duration of 111.2 frames (~4.4s). The lower frame count reflects a native Deaf signer communicating at natural speed. The baseline model trained primarily on slow student volunteers (Signers 3, 4, 5, 6) suffered equal WER on both test and validation, proving that the model did not learn motion phonemes, only static prior probabilities.

---

## 12. Controlled Accuracy Experiment (Section 9)

To test whether the mode collapse was caused by spatial coordinate variation across camera resolutions and distances, **Experiment 1 (Shoulder-Centered Spatial Normalization)** was executed:

- **Hypothesis:** Normalizing all 75 landmarks relative to the shoulder midpoint and inter-shoulder distance would align coordinate scales between QVGA and FHD signers and improve recognition.
- **Ablation Setup:**
  - Standardized normalization:
    $$\mathbf{p}' = \frac{\mathbf{p} - \mathbf{p}_{\text{mid\_shoulder}}}{\|\mathbf{p}_{\text{left\_shoulder}} - \mathbf{p}_{\text{right\_shoulder}}\|}$$
  - Exact same model architecture, optimizer, and training schedule.

### Controlled Experiment Results

```
+-----------------------------------+--------------------+--------------------+----------------------+
| Configuration                     | Best Val WER       | Best Epoch         | Final Test WER       |
+-----------------------------------+--------------------+--------------------+----------------------+
| **Baseline (Raw 300D Features)**  | 82.57%             | Epoch 20           | 83.09%               |
| **Experiment 1 (Shoulder Norm)**  | **82.24%**         | Epoch 28           | **83.09%**           |
| Delta                             | -0.33% (marginal)  | +8 epochs          | 0.00% (identical)    |
+-----------------------------------+--------------------+--------------------+----------------------+
```

**Conclusion from Experiment 1:** Spatial coordinate variance is not the primary bottleneck. The model still collapsed to `YOU` and `I`. The failure is rooted in sequence-level learning dynamics under CTC.

---

## 13. Known Limitations and Failure Modes

1. **CTC Mode Collapse & Blank Dominance:** Standard CTC loss allows the model to achieve ~4.1-4.3 loss by predicting >96% blanks and only firing on the highest-probability priors (`YOU`, `I`).
2. **Severe Class Imbalance:** With 170 gloss classes and only 499 training videos (mean 10 samples per gloss, median 5), the model lacks sufficient gradient pressure to distinguish rare glosses.
3. **Absence of Language Model / Transition Priors:** Without an N-gram or neural language model decoder, greedy decoding cannot infer syntactic structure.
4. **Coarticulation and Movement Dynamics:** A 1D Conv with stride 2 downsamples too quickly (reducing a 50-frame clip to 25 timesteps), blurring fast finger and hand transitions.
5. **No Signer Adaptation:** Native signers move hands at twice the speed of student volunteers. Fixed temporal pooling cannot adapt to this temporal discrepancy.

---

## 14. Recommended Next Improvements

1. **Focal CTC Loss or Blank Penalty:** Penalize the `<blank>` token logit by $\alpha \in [0.1, 0.5]$ during loss computation or decoding to force the model to emit non-blank lexical tokens.
2. **Class-Balanced Loss Weighting:** Weight CTC gradients inversely proportional to class frequency ($\frac{1}{\sqrt{f_c}}$) so rare glosses produce meaningful gradient updates.
3. **Temporal Data Augmentation:** Apply speed perturbation ($0.8\times, 1.0\times, 1.2\times$) and temporal jitter to volunteer recordings to match native signing cadences.
4. **Hand Crop Feature Fusion:** Concatenate MediaPipe landmark vectors with cropped hand ROI embeddings (e.g. MobileNetV3 or lightweight 3D CNN) to provide texture and shape details that landmarks miss.
5. **Pre-training on Isolated Glosses:** Initialize the BiGRU encoder weights by pre-training on the 170 isolated sign classes before fine-tuning on continuous sentences.
6. **Beam Search with ISL Language Model:** Replace greedy search with prefix beam search constrained by a 3-gram language model trained on ISL sentence grammar.

---

## Verification & Deliverable Sign-off

- [x] Pre-training audit executed and documented (all 687 files verified, 0 invalid CTC lengths).
- [x] Vocabulary mapping verified bit-exact (170 glosses + blank 0 = 171 classes).
- [x] Duplicate handling implemented (499 canonical training samples, 5 non-canonical copies excluded).
- [x] Model implemented: Conv1D (stride=2) + 2-layer BiGRU (hidden=128) + Linear head.
- [x] Checkpoint saved separately in `models/isl_csltr_continuous/best_checkpoint.pt`.
- [x] Full training history recorded in `models/isl_csltr_continuous/epoch_history.json`.
- [x] Validation (Signer 7) and Test (Signer 2) evaluations recorded separately.
- [x] 14/14 automated tests passed (`tests/test_isl_csltr_continuous_baseline.py` and `tests/test_isl_csltr_features.py`).
- [x] Phase 8.3.6 stopped before real-time integration.
