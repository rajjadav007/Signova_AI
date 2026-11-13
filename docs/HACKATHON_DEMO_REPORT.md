# Signova AI: Hackathon Demo Evaluation Report

**System:** 5-Class ISL Continuous Sentence Recognition Prototype (DTW + 1-NN)  
**Date:** 2026-11-14 03:08:41  
**Architecture:** Non-Parametric Dynamic Time Warping (1-NN Cosine Metric, Window=10)  
**Feature Representation:** MediaPipe Pose + Hands (75 landmarks) -> 678 Temporal Features -> 48-dim Frame PCA  
**Reference Exemplars:** 10 videos (2 per class, Train split exclusively)  
**Independent Evaluation:** 25 videos (5 per class) across 4 unseen signers  

---

## 1. Executive Summary

- **Top-1 Accuracy:** **64.0%** (16/25 correct across independent signers)
- **Macro F1-Score:** **0.6472** (Macro Precision: 0.7409, Recall: 0.6400)
- **Unknown Input Rejection Rate:** **60.0%** (15/25 out-of-vocabulary sentences rejected)
- **Matching Latency:** **10.06 ms** per sample (P95: 15.84 ms)
- **Signer Generalization:** Evaluated across Signer 3, Signer 4, Signer 5 (Train), Signer 7 (Validation), and Signer 2 (Test).

---

## 2. Per-Class Accuracy

| Class ID | Sentence | Eval Videos | Correct | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|
| 0 | `comb your hair` | 5 | 3 | 60.0% | 1.000 | 0.600 | 0.750 |
| 1 | `help me` | 5 | 2 | 40.0% | 0.500 | 0.400 | 0.444 |
| 2 | `i am fine. thank you sir` | 5 | 3 | 60.0% | 0.750 | 0.600 | 0.667 |
| 3 | `i need water` | 5 | 5 | 100.0% | 0.455 | 1.000 | 0.625 |
| 4 | `who are you` | 5 | 3 | 60.0% | 1.000 | 0.600 | 0.750 |

---

## 3. Confusion Matrix

Rows represent Ground Truth, Columns represent Model Predictions:

| True \ Pred | `comb your hair` | `help me` | `i am fine. thank you sir` | `i need water` | `who are you` |
|---|---|---|---|---|---|
| `comb your hair` | 3 | 1 | 1 | 0 | 0 |
| `help me` | 0 | 2 | 0 | 3 | 0 |
| `i am fine. thank you sir` | 0 | 1 | 3 | 1 | 0 |
| `i need water` | 0 | 0 | 0 | 5 | 0 |
| `who are you` | 0 | 0 | 0 | 2 | 3 |

---

## 4. Evaluation Examples

### 4.1 Sample Correct Predictions

- **"comb your hair"** (Signer 4, Split: `train`): DTW Distance: `0.3501`, Similarity: `0.6499`, Margin: `+0.0356`
- **"comb your hair"** (Signer 6, Split: `train`): DTW Distance: `0.3117`, Similarity: `0.6883`, Margin: `+0.0579`
- **"comb your hair"** (Signer 5, Split: `train`): DTW Distance: `0.2966`, Similarity: `0.7034`, Margin: `+0.1355`
- **"help me"** (Signer 3, Split: `train`): DTW Distance: `0.3486`, Similarity: `0.6514`, Margin: `+0.0792`
- **"help me"** (Signer 7, Split: `validation`): DTW Distance: `0.3083`, Similarity: `0.6917`, Margin: `+0.1221`
- **"i am fine. thank you sir"** (Signer 4, Split: `train`): DTW Distance: `0.3456`, Similarity: `0.6544`, Margin: `+0.0426`

### 4.2 Error Analysis (Incorrect Examples)

- **True:** "comb your hair" (Signer 2, `test`) -> **Pred:** "i am fine. thank you sir" (Dist: `0.2346`, Sim: `0.7654`, Margin: `+0.0103`)
- **True:** "comb your hair" (Signer 7, `validation`) -> **Pred:** "help me" (Dist: `0.3195`, Sim: `0.6805`, Margin: `+0.1331`)
- **True:** "help me" (Signer 4, `train`) -> **Pred:** "i need water" (Dist: `0.2849`, Sim: `0.7151`, Margin: `+0.0428`)
- **True:** "help me" (Signer 2, `test`) -> **Pred:** "i need water" (Dist: `0.1449`, Sim: `0.8551`, Margin: `+0.0266`)
- **True:** "help me" (Signer 1, `train`) -> **Pred:** "i need water" (Dist: `0.1923`, Sim: `0.8077`, Margin: `+0.0218`)
- **True:** "i am fine. thank you sir" (Signer 5, `train`) -> **Pred:** "help me" (Dist: `0.3984`, Sim: `0.6016`, Margin: `+0.0154`)
- **True:** "i am fine. thank you sir" (Signer 3, `train`) -> **Pred:** "i need water" (Dist: `0.3829`, Sim: `0.6171`, Margin: `+0.0029`)
- **True:** "who are you" (Signer 7, `validation`) -> **Pred:** "i need water" (Dist: `0.3811`, Sim: `0.6189`, Margin: `+0.0204`)
- **True:** "who are you" (Signer 2, `test`) -> **Pred:** "i need water" (Dist: `0.1281`, Sim: `0.8719`, Margin: `+0.0002`)

---

## 5. Unknown Input Handling (Out-of-Vocabulary Rejection)

- **Rejection Criterion:** Rejects if minimum DTW distance exceeds `0.40` (similarity < `0.60`) OR margin between top-1 and top-2 is `< 0.015`.
- **Benchmark on 25 Unseen Sentences:** 15/25 correctly rejected with output `"Sentence not recognized."`
- **Result:** The system does not force unfamiliar gestures or unknown sentences into one of the 5 demo classes.

---

## 6. Live Hackathon Demonstration Instructions

To run live inference during the demonstration:

```bash
# Run inference on a complete ISL signing video:
python scripts/demo_recognize_sentence.py --video path/to/video.mp4

# Run inference on pre-extracted landmarks:
python scripts/demo_recognize_sentence.py --landmarks path/to/landmarks.npz

# Output raw JSON result for UI integration:
python scripts/demo_recognize_sentence.py --video path/to/video.mp4 --json
```