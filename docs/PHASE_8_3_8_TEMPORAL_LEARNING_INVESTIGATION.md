# Phase 8.3.8: Fast Temporal Learning Investigation and Optimization Report

**Project:** Signova AI — Continuous Indian Sign Language Recognition  
**Phase:** 8.3.8 — Fast Temporal Learning Investigation and Optimization  
**Dataset:** ISL-CSLTR Continuous Feature Corpus (`datasets/isl_csltr_continuous/`)  
**Date:** November 13, 2026  
**Status:** COMPLETE (Audits Executed; Controlled Stride-1 Training Completed; Honest Negative Result Documented; Baseline Checkpoint Retained)  
**Primary Outcome:** Step A Feature Audit proved that middle 60% frames possess 1.306x (train) and 1.801x (val) higher active hand motion than endpoints, with low missing hand rates (< 1.5%), confirming that gestures are present and well-captured. Step B CTC Alignment Audit proved the baseline model collapsed into a static degenerate shortcut (`['I', 'YOU']` in 98.82% of validation sentences). In Step C/D, removing temporal downsampling (stride 2 $\rightarrow$ 1, full $T'=T$) did not restore middle-gesture learning: the model over-smoothed temporal sequence representations, predicting 99.1% blank frames and collapsing even further to a single unigram prior (`['YOU']` in 100% of sentences), producing Val WER of **89.14%** (vs 82.57% baseline) and Val Macro F1 of **0.0011** (vs 0.0053 baseline). In accordance with Section 10 & 12 Decision Rules, Phase 8.3.6 baseline checkpoint is retained as authoritative reference model.

---

## 1. Objective

The objective of Phase 8.3.8 is to investigate and optimize temporal learning in the continuous ISL recognition pipeline on the ISL-CSLTR dataset. The target is continuous sentence-level recognition ($X_{1:T} \rightarrow Y_{1:S}$), outputting ordered gloss sequences under CTC loss without frame-level supervision.

Specific goals:
1. Conduct an empirical temporal feature audit to test whether middle frames contain gesture motion or are uninformative.
2. Formally investigate CTC sequence lengths, output length arithmetic, and alignment degeneracy.
3. Implement and evaluate one controlled architectural hypothesis: reducing temporal downsampling (stride 2 $\rightarrow$ 1) to preserve full temporal granularity ($T' = T$) for the BiGRU.
4. Execute validation-gated model selection and final test evaluation.

---

## 2. Existing Baseline Failure Analysis

The Phase 8.3.6 baseline model architecture comprises:
- Input: MediaPipe spatial landmarks (T, 300)
- Projection: `Linear(300 -> 128)` + LayerNorm + Dropout(0.3) + ReLU
- Temporal Conv1D: `Conv1d(128, 128, kernel=3, stride=2, padding=1)` + LayerNorm + ReLU ($T' = (T+1)//2$)
- Recurrent Encoder: 2-layer Bidirectional GRU (hidden=128, bidirectional=256, dropout=0.3)
- Classifier Head: `Linear(256 -> 171)` with CTC Blank at index 0.

### Baseline Failure Mode Summary
- **Validation WER:** 82.57%
- **Test WER:** 83.09%
- **Exact Match:** 0.00%
- **Validation Macro F1:** 0.0053
- **Emitted Glosses:** Only 2 distinct glosses (`YOU`: 85, `I`: 84) out of 170 ISL classes.
- **Deletions:** 44.7% deletion rate (136 / 304 tokens).
- **Temporal Quintile Collapse:** In Phase 8.3.7 diagnostics, the middle 60% of all frames (quintiles Q2, Q3, Q4) had 100.0% blank argmax predictions ($P(\text{blank}) > 0.985$).

---

## 3. Step A: Fast Temporal Feature Audit Findings

All 499 canonical training feature files and all 85 validation feature files were audited directly from disk (`scripts/diagnose_temporal_features_phase_8_3_8.py`).

### 3.1 Frame Duration and Quality Statistics
- **Training Duration ($T$):** Min = 25, Mean = 97.9, Median = 97, Max = 192 frames.
- **Validation Duration ($T$):** Min = 58, Mean = 111.7, Median = 111, Max = 186 frames.
- **Pose Detection Confidence:** Mean = 0.9926 (Train), 0.9912 (Val).
- **Left Hand Confidence:** Mean = 0.8174 (Train), 0.8385 (Val).
- **Right Hand Confidence:** Mean = 0.8228 (Train), 0.8406 (Val).

### 3.2 Temporal Quintile Motion and Missing Hands Analysis
Each video sequence was divided into 5 normalized temporal segments (Q1: 0-20%, Q2: 20-40%, Q3: 40-60%, Q4: 60-80%, Q5: 80-100%):

| Temporal Quintile | Video Region | Train Active Hand Motion | Val Active Hand Motion | Train Both Hands Missing | Val Both Hands Missing |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Q1 (0-20%)** | Start Boundary | 0.02431 | 0.02521 | 6.8% | 2.5% |
| **Q2 (20-40%)** | Early Interior | 0.03348 | 0.04680 | 1.1% | 0.2% |
| **Q3 (40-60%)** | Core Signing | 0.03395 | 0.04944 | 0.7% | 0.0% |
| **Q4 (60-80%)** | Late Interior | 0.03212 | 0.04634 | 0.8% | 0.0% |
| **Q5 (80-100%)** | End Boundary | 0.02641 | 0.02758 | 4.9% | 2.2% |

### 3.3 Core Audit Conclusions
1. **Middle frames contain the peak signing gesture motion:**
   - In Train, middle 60% active hand motion is **1.306x** higher than endpoints ($0.03318$ vs $0.02536$).
   - In Val, middle 60% active hand motion is **1.801x** higher than endpoints ($0.04753$ vs $0.02640$).
2. **Missing hands do not explain middle blank collapse:** Both hands are missing in less than 1.1% of middle frames (0.0% in Q3 of validation).
3. **Verdict:** Landmark extraction preserves rich, dynamic, high-amplitude signing motion throughout the entire middle 60% of videos. The 100% blank prediction in middle frames is entirely attributable to model sequence learning failure and CTC loss optimization pathology.

---

## 4. Step B: CTC Alignment Investigation Findings

The baseline model's alignment dynamics were formally tested on all 85 validation videos (`scripts/diagnose_ctc_alignment_phase_8_3_8.py`).

### 4.1 Convolution Arithmetic Verification
The PyTorch Conv1d output length formula:
$$T_{\text{out}} = \left\lfloor \frac{T_{\text{in}} + 2P - D(K - 1) - 1}{S} \right\rfloor + 1$$
- For baseline ($K=3, S=2, P=1, D=1$): Formula yields $\lfloor (T-1)/2 \rfloor + 1 = (T+1)//2$. Verified across 191 lengths in $[10, 200]$: **0 mismatches**.
- For alternative ($K=3, S=1, P=1, D=1$): Formula yields $\lfloor (T-1)/1 \rfloor + 1 = T$. Verified across 191 lengths: **0 mismatches**.

### 4.2 Sequence Length Ratios and CTC Feasibility
- Target sequence length ($S$): Min = 1, Mean = 3.58, Max = 8 tokens.
- Subsampled sequence length ($T'$ with stride 2): Min = 29, Mean = 55.85, Max = 93 frames.
- Ratio $T' / S$: Min = **6.33**, Mean = **17.61**, Max = 46.5.
- Feasibility violations ($T' < S$): **0 samples** (100% valid).

### 4.3 Baseline Model Temporal Emission Localization
Across 4,747 validation frames:
- **Q1 (0-20%):** 84 non-blank emissions (8.56% rate), all predicting `I`. Mean blank posterior = 0.9151.
- **Q2 (20-40%):** 0 non-blank emissions (0.00% rate). Mean blank posterior = 0.9862.
- **Q3 (40-60%):** 0 non-blank emissions (0.00% rate). Mean blank posterior = 0.9859.
- **Q4 (60-80%):** 0 non-blank emissions (0.00% rate). Mean blank posterior = 0.9857.
- **Q5 (80-100%):** 85 non-blank emissions (9.26% rate), all predicting `YOU`. Mean blank posterior = 0.8595.

### 4.4 Static Shortcut Proof
- **Predicted Sequence:** Exactly **84 out of 85 sentences (98.82%)** predict the identical sequence `['I', 'YOU']`. Exactly 1 sentence predicts `['YOU']`.
- **Verdict:** The baseline model learned zero temporal gesture alignment. It converged to a trivial boundary-conditioned unigram memorizer.

---

## 5. Step C: Controlled Alternative Experiment — Stride-1 Temporal Conv1D

### Hypothesis
Reducing the temporal Conv1D stride from 2 to 1 maintains the full temporal sampling rate ($T' = T \approx 100$ steps instead of 50). This supplies the 2-layer BiGRU with uncompressed frame-by-frame temporal trajectory dynamics, preventing the temporal boundary collapse.

### Architecture Specifications
- **Input:** $x \in \mathbb{R}^{B \times T \times 300}$
- **Projection:** `Linear(300 -> 128)` + `LayerNorm(128)` + `Dropout(0.3)` + `ReLU()`
- **Temporal Conv1D:** `Conv1d(128, 128, kernel_size=3, stride=1, padding=1)` + `LayerNorm(128)` + `ReLU()` $\implies T' = T$
- **Recurrence:** `nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)`
- **Head:** `Dropout(0.3)` + `Linear(256 -> 171)`
- **Loss:** `CTCLoss(blank=0, zero_infinity=True)`
- **Parameter Count:** 755,321 parameters (identical to baseline; zero parameter bloat).

---

## 6. Step D: Training Execution

### Smoke Test
Executed on CPU before main training:
- Input shape $(2, 30, 300)$, lengths $[30, 25]$, targets $[1, 2, 3, 4, 5]$, target lengths $[3, 2]$.
- Forward pass verified: $T'=30, B=2, C=171$, $T' = T$.
- CTC Loss: 53.5517 (finite).
- Backward pass: gradient norm = 27.0451 (finite).
- Checkpoint save and load verified successfully.

### Main Training Run
- **Hardware:** CPU
- **Optimizer:** AdamW ($\text{lr} = 3 \times 10^{-4}, \text{weight decay} = 1 \times 10^{-4}$)
- **Scheduler:** `CosineAnnealingLR(T_max=50, eta_min=1e-5)`
- **Gradient Clipping:** Max norm = 5.0
- **Batch Size:** 16
- **Training Epochs:** 27 epochs (Early stopping triggered; patience = 15).
- **Total Training Duration:** 398.81 seconds (6.65 minutes; ~14.7 seconds/epoch).
- **Best Validation Epoch:** Epoch 12.

---

## 7. Step E & F: Validation Results and Comparison

All 85 canonical validation sentences (Signer 7, 304 reference tokens) were evaluated using identical evaluation code:

| Metric | Phase 8.3.6 Baseline (Stride 2) | Phase 8.3.8 Alternative (Stride 1) | Delta | Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **Validation WER** | **82.57%** | 89.14% | $+6.57\%$ | Worse |
| **Validation TER** | **82.57%** | 89.14% | $+6.57\%$ | Worse |
| **Validation Exact Match** | **0.00%** | 0.00% | $\pm 0.00\%$ | Equal |
| **Validation Macro F1** | **0.0053** | 0.0011 | $-0.0042$ | Worse |
| **Substitutions** | 115 | 52 | $-63$ | - |
| **Deletions** | 136 (44.7%) | 219 (72.0%) | $+83$ | Severe Collapse |
| **Insertions** | 0 | 0 | 0 | Equal |
| **Mean Blank Frame Rate** | 96.3% | 99.1% | $+2.8\%$ | Higher Blank Rate |
| **Mean Blank Posterior** | 0.9468 | 0.9749 | $+0.0281$ | Higher Blank Certainty |
| **Distinct Glosses Emitted** | 2 (`YOU`: 85, `I`: 84) | 1 (`YOU`: 85) | $-1$ gloss | Single-Class Degeneracy |
| **Mean Predicted Sequence Length**| 1.99 tokens | 1.00 tokens | $-0.99$ tokens | Emits only 1 token/video |

### Failure Analysis of Stride-1
Counter-intuitively, retaining full temporal resolution ($T'=T$) worsened CTC sequence learning:
1. **Dilation of Blank Paths in Dynamic Programming:** With $T \approx 100$ and $S \approx 3.5$, the CTC trellis has an even larger ratio of valid alignment paths dominated by blanks ($T'/S \approx 30$). Gradient updates to the blank token accumulate over twice as many timesteps per sequence, further reinforcing blank dominance.
2. **Single-Token Collapse:** In Phase 8.3.6 (stride 2), the network emitted 2 tokens (`['I', 'YOU']`). With stride 1, the network collapsed to emitting only a single token per video (`['YOU']`), causing the deletion rate to spike from 44.7% to 72.0% and increasing WER from 82.57% to 89.14%.

---

## 8. Final Test Evaluation (Signer 2 - Unseen Native Deaf Signer)

Per Section 10 rules, the best Stride-1 checkpoint (Epoch 12) was evaluated once on the untouched test partition (98 videos, 349 reference tokens):

| Metric | Phase 8.3.6 Baseline Test | Phase 8.3.8 Stride-1 Test | Delta |
| :--- | :---: | :---: | :---: |
| **Test WER** | **83.09%** | 88.83% | $+5.74\%$ (Worse) |
| **Test TER** | **83.09%** | 88.83% | $+5.74\%$ (Worse) |
| **Test Exact Match** | **0.00%** | 0.00% | $\pm 0.00\%$ |
| **Test Macro F1** | **0.0046** | 0.0009 | $-0.0037$ (Worse) |
| **Substitutions** | 141 | 59 | $-82$ |
| **Deletions** | 148 (42.4%) | 251 (71.9%) | $+103$ |
| **Insertions** | 1 | 0 | $-1$ |
| **Mean Blank Frame Rate** | 92.9% | 98.26% | $+5.36\%$ |
| **Mean Blank Posterior** | 0.9087 | 0.9579 | $+0.0492$ |
| **Distinct Glosses Emitted** | 2 (`YOU`: 98, `I`: 86) | 1 (`YOU`: 98) | $-1$ gloss |
| **Mean Predicted Length** | 1.88 tokens | 1.00 tokens | $-0.88$ tokens |

---

## 9. Best Checkpoint Selection and Rationale

Per Section 10 & 12 of the Project Specification:
- **Selected Reference Checkpoint:** `models/isl_csltr_continuous/best_checkpoint.pt` (Phase 8.3.6 Baseline).
- **Selection Reason:** The Phase 8.3.8 Stride-1 model yielded Val WER of 89.14% (vs 82.57% baseline) and Val Macro F1 of 0.0011 (vs 0.0053 baseline). Because performance degraded, Phase 8.3.6 is retained as the authoritative reference model to prevent regression.
- **Outcome Classification:** **Outcome B: No Improvement**.

---

## 10. Remaining Limitations

1. **CTC Blank Path Attraction:** Standard CTC loss without strong acoustic/gesture pre-training exhibits a severe mathematical preference for blank emissions when the temporal sequence length $T$ substantially exceeds the token length $S$.
2. **Lack of Pre-Trained Gesture Representations:** Training a BiGRU from scratch on 499 continuous video sequences across 170 gloss classes does not provide sufficient gradient guidance to discover gesture boundaries and classify subtle handshapes simultaneously.
3. **Absence of Temporal Alignment Constraints:** Without auxiliary boundary supervision (e.g. CTC with peak regularization or alignment prior), gradient descent reliably falls into the global minimum of predicting blanks for gesture frames and emitting high-frequency unigram priors at boundaries.

---

## 11. Recommended Next Steps

1. **Do not run further speculative CTC architecture tweaks from scratch.** Stride reduction, class-balanced sampling, and blank penalty have all failed to escape the CTC shortcut attractor.
2. **Move to Application Demonstration:** In accordance with Phase 8.3.8 Section 12 Outcome B instructions, preserve the reference checkpoint and transition to the core Signova AI application demonstration.
3. **Future Continuous ISL Work (Post-Hackathon):**
   - Initialize the temporal encoder with feature weights from the production isolated sign classifier.
   - Use Connectionist Temporal Classification with alignment-aware decoding or an Autoregressive Attention-based Encoder-Decoder (AED / Conformer).

---

## 12. Deliverable Verification

All artifacts required by the specification have been created, verified, and saved to `models/isl_csltr_continuous/phase_8_3_8/`:

- Temporal feature audit: [temporal_diagnostics.json](file:///D:/Projects/Signova%20AI/models/isl_csltr_continuous/phase_8_3_8/temporal_diagnostics.json)
- CTC alignment diagnostics: [ctc_alignment_diagnostics.json](file:///D:/Projects/Signova%20AI/models/isl_csltr_continuous/phase_8_3_8/ctc_alignment_diagnostics.json)
- Experiment configuration: [experiment_config.json](file:///D:/Projects/Signova%20AI/models/isl_csltr_continuous/phase_8_3_8/experiment_config.json)
- Training epoch history: [training_history.json](file:///D:/Projects/Signova%20AI/models/isl_csltr_continuous/phase_8_3_8/training_history.json)
- Validation results: [validation_results.json](file:///D:/Projects/Signova%20AI/models/isl_csltr_continuous/phase_8_3_8/validation_results.json)
- Final test results: [final_test_results.json](file:///D:/Projects/Signova%20AI/models/isl_csltr_continuous/phase_8_3_8/final_test_results.json)
- Best Phase 8.3.8 checkpoint: [best_checkpoint.pt](file:///D:/Projects/Signova%20AI/models/isl_csltr_continuous/phase_8_3_8/best_checkpoint.pt)
- Comprehensive Phase Report: [PHASE_8_3_8_TEMPORAL_LEARNING_INVESTIGATION.md](file:///D:/Projects/Signova%20AI/docs/PHASE_8_3_8_TEMPORAL_LEARNING_INVESTIGATION.md)
