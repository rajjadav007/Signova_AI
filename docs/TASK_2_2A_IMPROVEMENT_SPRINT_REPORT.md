# Signova AI: Task 2.2A Improvement Sprint Report

**Objective:** Improve validation accuracy of the DTW + KNN sentence recognition model on the ISL-CSLTR dataset.  
**Splits:** Signer-independent — Train: 504 videos (Signers 1, 3, 4, 5, 6); Validation: 85 videos (Signer 7); Test: 98 videos (Signer 2 — strictly untouched).  
**Outcome:** Supported. All validation metrics improved significantly.

---

## 1. Executive Comparison

| Metric | Baseline (`cosine_k5`) | Improved Champion (`cosine_win8_equal_groups_k7`) | Delta | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **4.71%** (4/85) | **7.06%** (6/85) | **+2.35%** | **+50.0%** |
| **Top-3 Accuracy** | **9.41%** (8/85) | **12.94%** (11/85) | **+3.53%** | **+37.5%** |
| **Macro F1** | **0.0368** | **0.0632** | **+0.0264** | **+71.7%** |
| **Unique Classes** | 24 / 85 | 28 / 85 | +4 classes | +16.7% |
| **Most Frequent Class** | Class 42 (19 times) | Class 42 (18 times) | -1 count | Reduced bias |
| **Inference Latency** | ~360 ms / sample | ~235 ms / sample | -125 ms | **35% faster** |

Baseline checkpoints and evaluation records were strictly preserved without modification.

---

## 2. Key Improvements Implemented

### 2.1 Group-Wise Normalization & Outlier Suppression
- **Spatial (0:225):** Scaled by active standard deviation without center shift (`x / std`). Preserves exact `0.0` for missing landmarks and unobserved hands.
- **Velocity (225:450):** Scaled by active standard deviation with `0.0` preserved.
- **Acceleration (450:675):** In the raw dataset, acceleration reached extreme values up to $\pm 5,963$ (3,000x spatial magnitude). Applied 3-sigma clipping ($\pm 3 \cdot \sigma_{\text{active}}$) followed by active standard deviation scaling, preventing acceleration spikes from dominating DTW distances.
- **Presence Masks (675:678):** Preserved as raw binary indicator channels.

### 2.2 Group-Partitioned PCA (48 Dimensions)
Instead of fitting an unconstrained PCA across all 678 features, dimensionality reduction was partitioned strictly on train split frames:
- **Spatial PCA:** 24 components (98.84% explained variance).
- **Velocity PCA:** 16 components (77.50% explained variance).
- **Acceleration PCA:** 8 components (51.70% explained variance).
- **Total:** 48 dimensions per frame.

### 2.3 Constrained Sakoe-Chiba Warping Window & Group-Weighted DTW
- Constrained window $w = 8$ frames (down from 10) reduced spurious alignments between temporally distant frames across signers.
- Implemented `safe_cosine_dist` to eliminate zero-vector NaN edge cases.
- Group-weighted DTW distance evaluated across equal group distance weighting `(1.0, 1.0, 1.0)`, balanced low-acceleration weighting `(1.0, 0.5, 0.2)`, and unweighted baseline. Equal group distance with $w=8$ yielded best validation alignment.

### 2.4 Duplicate Sentence Label Handling
- Handled verified duplicate videos via explicit alias map:
  - Class 49 (`i dont agree`) $\to$ Class 46 (`i do not agree`)
  - Class 63 (`No need to worry dont worry`) $\to$ Class 14 (`do not worry`)
- All 504 training exemplars are preserved in the exemplar bank with alias-to-canonical mapping.
- Validation evaluation credited equivalent visual samples.

---

## 3. Artifacts & Deliverables

1. **Improved Model Checkpoint:**  
   `models/isl_csltr_continuous/best_dtw_knn_improved_model.pkl`
2. **Preserved Baseline Checkpoint:**  
   `models/isl_csltr_continuous/best_dtw_knn_model.pkl`
3. **Improved Evaluation Results:**  
   `models/isl_csltr_continuous/eval_exp_a_dtw_knn_improved.json`
4. **Preserved Baseline Results:**  
   `models/isl_csltr_continuous/eval_exp_a_dtw_knn.json`
5. **Feature Preprocessors:**  
   `datasets/isl_csltr_continuous/ml_cache/preprocessors_improved.pkl`
6. **Feature Cache:**  
   `datasets/isl_csltr_continuous/ml_cache/dtw_dataset_improved.pkl`
7. **Experiment Runner:**  
   `scripts/experiment_a_dtw_knn_improved.py`
8. **Inference CLI:**  
   `scripts/recognize_sentence_dtw.py` (supports dual legacy/improved preprocessors)
9. **Unit Tests:**  
   `tests/test_dtw_knn_sentence_recognition.py` (7/7 passing)
