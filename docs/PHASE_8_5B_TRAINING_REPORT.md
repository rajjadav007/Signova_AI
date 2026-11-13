# Phase 8.5B: Controlled Prototype Model Training & Ablation Report

**Project:** Signova AI — Continuous Indian Sign Language (ISL) Recognition  
**Phase:** 8.5B — Controlled Prototype Model Training & Ablation  
**Dataset:** ISL-CSLTR Continuous Sentence Corpus (`datasets/isl_csltr_continuous/`)  
**Target Vocabulary:** 20 Closed-Set ISL Glosses + Blank (`<blank>`: 0)  
**Target Corpus:** 18 Verified Sentence Structures (126 Full Canonical Video Recordings)  
**Splits:** Train: 92 videos (Signers 1, 3, 4, 5, 6); Validation: 17 videos (Signer 7); Test: 17 videos (Signer 2)  
**Status:** COMPLETE (Training & Evaluation Finished; Production Integration Halted For Review)  
**Execution Script:** `scripts/run_phase_8_5b_experiments.py`  
**Checkpoint Path:** `models/isl_csltr_continuous/phase_8_5b/`  

---

## 1. Pre-Training Verification Summary

Prior to executing any training runs, all 6 pre-training confirmation requirements were verified against the codebase and PyTorch backend:

| # | Requirement | Verified Configuration | Result |
|---|---|---|---|
| 1 | Feature Dimensions | Exactly 228 dimensions (225 normalized torso-centered $(x, y, z)$ coordinates + 3 detection mask flags). Zero explicit velocity channels (learned dynamically via Conv1d kernel 3). | **PASS** |
| 2 | Preprocessing Parity | Single vectorized function `preprocess_landmarks_228()` used identically across train, val, test loaders and inference. | **PASS** |
| 3 | CTC Loss Reduction & Scaling | PyTorch `CTCLoss(reduction='mean')` divides per-sample loss by target sequence length $S_i$. Equivalence to per-sample normalized reduction verified via Context7 C++ source inspection. | **PASS** |
| 4 | Sequence Rarity Weighting | Exponent $\alpha = 0.33$ computed over 216 training tokens: $w_i = \frac{1}{\|Y_i\|} \sum_{g \in Y_i} (N_{\text{total}} / N_g)^{0.33}$. Minimum weight: 1.58 (`YOU`), maximum weight: 3.51 (`THINGS`). | **PASS** |
| 5 | Vocabulary Mapping | 20 glosses mapped to IDs 1..20; blank index 0; total output classes = 21. | **PASS** |
| 6 | Dataset & Split Integrity | Exactly 126 canonical videos: Train 92 (Signers 1, 3, 4, 5, 6), Val 17 (Signer 7), Test 17 (Signer 2). 0 duplicate leakage, 0 signer overlap. | **PASS** |

---

## 2. Multi-Experiment Performance Comparison

Three controlled experiments were executed with fixed seed 42, AdamW optimizer, learning rate $3 \times 10^{-4}$, weight decay $1 \times 10^{-4}$, gradient clipping 5.0, batch size 8, CosineAnnealingLR, and early stopping patience 15 monitored on Validation WER:

- **Experiment A:** Standard CTC Baseline (`torch.nn.CTCLoss(reduction='mean')`).
- **Experiment B:** CTC + Margin Blank Posterior Regularization ($\lambda_{\text{blank}} = 0.5, \tau_{\text{blank}} = 0.75$).
- **Experiment C:** CTC + Sequence-Level Class Rarity Weighting ($w_i = \frac{1}{|Y_i|} \sum_{g} (N_{\text{total}} / N_g)^{0.33}$).

### Summary Comparison Table

| Metric | Exp A (Standard CTC) | Exp B (Blank Reg) | Exp C (Rarity Weighting) |
|---|---|---|---|
| **Best Epoch** | 24 | 24 | **27** |
| **Total Epochs Trained** | 39 (Early Stop) | 39 (Early Stop) | 42 (Early Stop) |
| **Training Duration** | 139.77s | 164.65s | 150.69s |
| **Train Loss (Best Epoch)** | 2.6185 | 2.7318 | **2.5527** |
| **Validation Loss** | 2.7780 | 2.7776 | **2.7566** |
| **Validation WER (%)** | 76.92% | 76.92% | 76.92% |
| **Validation TER (%)** | 76.92% | 76.92% | 76.92% |
| **Validation Exact Match (%)** | 0.0% | 0.0% | 0.0% |
| **Validation Vocab Diversity** | 1 / 20 (`YOU`) | 1 / 20 (`YOU`) | 1 / 20 (`YOU`) |
| **Validation Macro F1 (%)** | 3.64% | 3.64% | 3.64% |
| **Validation Blank Posterior Mean** | 0.97 | 0.97 | 0.98 |
| **Validation Interior Nonblank (%)** | 0.0% | 0.0% | 0.0% |
| **Validation Q5 Nonblank (%)** | 5.50% | 5.50% | 5.50% |
| **Untouched Test Loss** | 2.7104 | 2.7087 | **2.6765** |
| **Untouched Test WER (%)** | 75.00% | 75.00% | 75.00% |
| **Untouched Test TER (%)** | 75.00% | 75.00% | 75.00% |
| **Untouched Test Exact Match (%)** | 0.0% | 0.0% | 0.0% |
| **Untouched Test Vocab Diversity**| 1 / 20 (`YOU`) | 1 / 20 (`YOU`) | 1 / 20 (`YOU`) |
| **Untouched Test Macro F1 (%)** | 3.90% | 3.90% | 3.90% |
| **Untouched Test Blank Posterior Mean** | 0.96 | 0.96 | 0.96 |
| **Untouched Test Interior Nonblank (%)** | 0.0% | 0.0% | 0.0% |
| **Untouched Test Q5 Nonblank (%)** | 11.53% | 11.53% | 11.53% |

---

## 3. Temporal Quintile Nonblank Emission Profile

Across all three experiments, the temporal distribution of nonblank emissions reveals identical degenerate behavior:

```
+-------------------------------------------------------------------------------+
|                       NONBLANK EMISSION RATE BY QUINTILE                      |
+---------------------+---------+---------+---------+---------+---------+-------+
| Split / Experiment  | Q1 (%)  | Q2 (%)  | Q3 (%)  | Q4 (%)  | Q5 (%)  | Mean  |
+---------------------+---------+---------+---------+---------+---------+-------+
| Validation (Exp A)  | 0.00%   | 0.00%   | 0.00%   | 0.00%   | 5.50%   | 1.10% |
| Validation (Exp B)  | 0.00%   | 0.00%   | 0.00%   | 0.00%   | 5.50%   | 1.10% |
| Validation (Exp C)  | 0.00%   | 0.00%   | 0.00%   | 0.00%   | 5.50%   | 1.10% |
+---------------------+---------+---------+---------+---------+---------+-------+
| Test (Exp A)        | 0.00%   | 0.00%   | 0.00%   | 0.00%   | 11.53%  | 2.31% |
| Test (Exp B)        | 0.00%   | 0.00%   | 0.00%   | 0.00%   | 11.53%  | 2.31% |
| Test (Exp C)        | 0.00%   | 0.00%   | 0.00%   | 0.00%   | 11.53%  | 2.31% |
+---------------------+---------+---------+---------+---------+---------+-------+
```

**Observation:**
Interior quintiles $Q_1, Q_2, Q_3, Q_4$ have **exactly 0.0% nonblank emissions**. The models emit blank probabilities $> 96\%$ across the entirety of the signing video, outputting a single nonblank emission spike for gloss `YOU` solely in $Q_5$ (the trailing video frames).

---

## 4. Per-Gloss Breakdown (Validation & Test Sets)

### Validation Set (Signer 7, 17 Videos, 39 Reference Tokens)

| Gloss | Precision (%) | Recall (%) | F1 (%) | Support | Predicted | True Positives |
|---|---|---|---|---|---|---|
| `YOU` | 52.94% | 100.00% | 69.23% | 9 | 17 | 9 |
| `(AGE)` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `ANGRY` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `BAD` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `CONGRATULATIIONS` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `CRY` | 0.00% | 0.00% | 0.00% | 2 | 0 | 0 |
| `DISAPPOINTED` | 0.00% | 0.00% | 0.00% | 0 | 0 | 0 |
| `GOOD` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `HELP` | 0.00% | 0.00% | 0.00% | 3 | 0 | 0 |
| `HI` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `HOW` | 0.00% | 0.00% | 0.00% | 3 | 0 | 0 |
| `HUNGRY` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `I` | 0.00% | 0.00% | 0.00% | 7 | 0 | 0 |
| `ME` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `PROMISE` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `THINGS` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `TIRED` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `WELCOME` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `WHO` | 0.00% | 0.00% | 0.00% | 1 | 0 | 0 |
| `WHY` | 0.00% | 0.00% | 0.00% | 2 | 0 | 0 |
| **Macro Average** | **2.65%** | **5.00%** | **3.64%** | **39** | **17** | **9** |

---

## 5. Sample Inference Output Examples

| Video ID | Sentence Text | Reference Gloss Sequence | Hypothesis (All Exps) | Match |
|---|---|---|---|---|
| `s003_signer7_help__5_` | can i help you | `I HELP YOU` | `YOU` | FAIL |
| `s006_signer7_congratulations__5_` | congratulations | `CONGRATULATIIONS` | `YOU` | FAIL |
| `s023_signer7_help_me__8_` | help me | `HELP ME` | `YOU` | FAIL |
| `s024_signer7_hi_how_are_you__3_` | hi how are you | `HI HOW YOU` | `YOU` | FAIL |
| `s025_signer7_things__4_` | how are things | `HOW THINGS` | `YOU` | FAIL |

---

## 6. Mathematical Diagnosis of CTC Mode Collapse

### 6.1 The Majority-Class Floor Proof
In continuous CTC training with class imbalance, let $N_{\text{ref}}$ be total reference tokens in an evaluation set, and let $N_{\text{dom}}$ be the frequency of the single dominant token $g^* = \text{YOU}$.
If the model collapses into emitting the constant single-token sequence $\hat{Y} = [g^*]$ for every sample:
1. Every sample containing $g^*$ incurs 0 substitution errors for $g^*$, but $L_i - 1$ deletions for the remaining tokens.
2. Every sample not containing $g^*$ incurs 1 substitution error and $L_i - 1$ deletions.
3. Total Levenshtein edit distance:
   $$\text{Total Errors} = \sum_{i=1}^M \left[ \mathbb{I}(g^* \in Y_i) \cdot (L_i - 1) + \mathbb{I}(g^* \notin Y_i) \cdot L_i \right] = N_{\text{ref}} - N_{\text{dom}}$$
4. The resulting Word Error Rate is:
   $$\text{WER}_{\text{collapse}} = \frac{N_{\text{ref}} - N_{\text{dom}}}{N_{\text{ref}}} \times 100\%$$
- **Validation Split:** $N_{\text{ref}} = 39, N_{\text{dom}} = 9 \implies \text{WER} = \frac{39 - 9}{39} = \frac{30}{39} = \mathbf{76.92\%}$.
- **Test Split:** $N_{\text{ref}} = 40, N_{\text{dom}} = 10 \implies \text{WER} = \frac{40 - 10}{40} = \frac{30}{40} = \mathbf{75.00\%}$.

The observed validation and test WERs of exactly 76.92% and 75.00% match this theoretical degenerate minimum to four decimal places.

### 6.2 Why Early Stopping Selected the Degenerate Checkpoint
As training progresses into epochs 28–35 (in Experiment C):
- The model attempted to emit additional glosses (diversity increased to 3: `YOU`, `I`, `WHY`).
- However, without precise temporal boundary alignment or segment-level supervision, emitting multiple glosses created spurious insertions and substitutions, causing Validation WER to degrade from 76.92% to 84.62%.
- Because early stopping is governed strictly by minimum `val_wer`, the checkpoint selector rejected the multi-gloss epochs and preserved the collapsed single-token checkpoint.

### 6.3 Why Regularization and Rarity Weighting Were Insufficient
1. **Experiment B ($\lambda_{\text{blank}} = 0.5$):** The average blank probability was 0.97. The linear penalty incurred was $0.5 \times (0.97 - 0.75) = 0.11$. Against a CTC loss of $\sim 2.6 - 3.0$, an auxiliary penalty of $0.11$ ($< 4\%$ of the total loss) was insufficient to overcome the gradient pull of the blank attractor.
2. **Experiment C ($\alpha = 0.33$):** While sequence weighting scaled gradients by up to $2.5\times$ for rare sentences, all sentences still shared the dominant tokens (`YOU` appears in 10/18 sentences; `I` in 7/18). The model still found the lowest total CTC loss by outputting `YOU` rather than attempting unconstrained temporal alignment.

---

## 7. Model Selection & Untouched Test Performance

### Best Model Selection
Based strictly on validation performance with Validation Loss as tiebreaker (all three tied at 76.92% Val WER):
- **Selected Best Model:** **Experiment C (CTC with Sequence-Level Rarity Weighting)**
  - Best Epoch: 27
  - Validation Loss: **2.7566** (vs 2.7780 for A, 2.7776 for B)
  - Validation WER: 76.92%
  - Checkpoint Path: `models/isl_csltr_continuous/phase_8_5b/exp_c/best_checkpoint.pt`

### Untouched Test Performance of Selected Best Model (Signer 2)
- **Test Loss:** **2.6765** (lowest across all three experiments)
- **Test WER:** **75.00%**
- **Test TER:** **75.00%**
- **Test Exact Sentence Match Rate:** **0.0%**
- **Test Vocabulary Diversity:** 1 / 20 (`YOU`)
- **Test Macro F1:** 3.90%
- **Test Blank Posterior Mean:** 0.96
- **Test Interior Nonblank Rate:** 0.0% (Q5 = 11.53%)

---

## 8. Checkpoint Isolation & Production Safety

- **Production Isolated-Sign Model (`models/checkpoints/`):** Completely untouched.
- **Production 170-Gloss Checkpoint (`models/isl_csltr_continuous/best_checkpoint.pt`):** Completely untouched (Size: 7,552,514 bytes; MD5 preserved).
- **Phase 8.5B Checkpoints:** Saved separately in `models/isl_csltr_continuous/phase_8_5b/`:
  - `exp_a/best_checkpoint.pt` (7,078,470 bytes)
  - `exp_b/best_checkpoint.pt` (7,078,470 bytes)
  - `exp_c/best_checkpoint.pt` (7,078,470 bytes)
- **Zero Automatic Integration:** As instructed, none of these prototype models have been promoted to production inference. All changes await human review and approval.
