# Signova AI: Hackathon Demo Quickstart Guide

## Overview
Working 5-class continuous Indian Sign Language (ISL) sentence recognition prototype for live demonstration.
Powered by MediaPipe landmark extraction and 1-NN Dynamic Time Warping (DTW) with Cosine metric over 48-dim Frame PCA.

---

## 5 Selected Sentence Classes
1. `comb your hair` (ISL Gloss: `COMB YOU HAIR`, Class ID: 0, Original SID: 5)
2. `help me` (ISL Gloss: `HELP ME`, Class ID: 1, Original SID: 23)
3. `i am fine. thank you sir` (ISL Gloss: `I FINE. THANK YOU SIR`, Class ID: 2, Original SID: 35)
4. `i need water` (ISL Gloss: `I NEED WATER`, Class ID: 3, Original SID: 53)
5. `who are you` (ISL Gloss: `WHO YOU`, Class ID: 4, Original SID: 91)

Reference Pool: Exactly 2 training videos per class (10 reference videos total). Test and validation splits remain completely untouched.

---

## Live Demo Commands

### 1. Run Recognition on Video File (.mp4)
```bash
python scripts/demo_recognize_sentence.py --video "D:/Datasets/ISL-CSLTR/isl-csltr-indian-sign-language-dataset/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level/comb your hair/comb (4).MP4"
```

### 2. Run Recognition on Pre-extracted Landmarks (.npz)
```bash
python scripts/demo_recognize_sentence.py --landmarks "datasets/isl_csltr_continuous/features/train/s005_signer6_comb__4_.npz"
```

### 3. Output JSON for UI / Backend Integration
```bash
python scripts/demo_recognize_sentence.py --landmarks "datasets/isl_csltr_continuous/features/train/s005_signer6_comb__4_.npz" --json
```

### 4. Unknown Input Rejection Demo
Pass an out-of-vocabulary signing video or gesture to demonstrate rejection:
```bash
python scripts/demo_recognize_sentence.py --landmarks "datasets/isl_csltr_continuous/features/train/s009_signer6_abuse__3_.npz"
```
Output:
```text
STATUS: REJECTED
RESULT: "Sentence not recognized."
```

---

## Artifact Locations
- Reference Dataset & Classifier: `models/isl_csltr_continuous/hackathon_demo/`
  - `demo_classes.json`: Class mappings and glosses
  - `reference_metadata.json`: Signer IDs, video paths, landmark stats
  - `reference_dataset.pkl`: 10 preprocessed reference sequences
  - `demo_dtw_knn_model.pkl`: Fitted DTWKNNClassifier checkpoint
  - `eval_results.json`: Full benchmark evaluation results
- Inference Script: `scripts/demo_recognize_sentence.py`
- Evaluation Suite: `scripts/evaluate_hackathon_demo.py`
- Detailed Benchmark Report: `docs/HACKATHON_DEMO_REPORT.md`
- Integrity Unit Test: `tests/test_hackathon_demo.py`
