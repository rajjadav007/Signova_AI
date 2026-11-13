# AI/ML Model Strategy

## Signova AI

**Project:** Signova AI  
**Domain:** Indian Sign Language (ISL) Recognition  
**AI Framework:** PyTorch  
**Backend:** Python 3.11 + FastAPI  
**Development Approach:** Pretrained Models First, Local Development First

---

# 1. Purpose

This document defines the AI/ML architecture, model selection, preprocessing, inference, evaluation, and integration rules for Signova AI.

The goal is to build a reliable ISL recognition system using verified models and reproducible processing pipelines.

Signova AI currently follows a separate two-engine architecture:

1. **Individual Words Engine:** A Bi-GRU-based model for recognizing a defined vocabulary of 10 isolated ISL signs.
2. **Continuous Translation Engine:** A separate experimental iSign T5 pose-to-text pipeline.

These engines must remain independent.

Do not treat isolated sign classification as continuous sentence translation.

---

# 2. Core AI/ML Principles

All AI/ML development must follow these principles:

1. Prefer suitable pretrained models over training from scratch.
2. Verify that a model supports Indian Sign Language before integration.
3. Keep model loading, preprocessing, inference, and postprocessing separate.
4. Preserve the original model architecture and expected input format.
5. Validate model compatibility before integration.
6. Never generate fake predictions or hardcode recognition results.
7. Keep AI modules independent from frontend and API implementation.
8. Document known model limitations.
9. Prefer reproducible experiments.
10. Do not claim model accuracy without evaluation evidence.
11. Do not modify working checkpoints without approval.
12. Keep isolated recognition and continuous translation separate.

---

# 3. Sign Language Rules

Signova AI focuses on **Indian Sign Language (ISL)**.

Antigravity must follow these rules:

- Do not treat American Sign Language (ASL) as ISL.
- Do not present ASL alphabet recognition as ISL translation.
- Do not assume signs are identical across different sign languages.
- Do not claim support for all ISL vocabulary.
- Do not claim continuous sentence translation when only isolated signs are recognized.
- Distinguish isolated sign recognition from continuous sign language recognition.
- Consider hand movements, body pose, and facial expressions when the selected model requires them.
- Verify the supported vocabulary and sign language of every model.

---

# 4. Signova AI Model Architecture

Signova AI uses two separate AI engines.

## 4.1 Individual Words Engine

**Status:** Current implemented recognition engine.

**Task:** Isolated ISL sign classification.

**Model:** Bi-GRU.

**Vocabulary:** 10 selected signs.

| Sign | Class |
|---|---|
| Food | food |
| Hello | hello |
| Help | help |
| No | no |
| Please | please |
| Sorry | sorry |
| Stop | stop |
| Thank you | thank you |
| Water | water |
| Yes | yes |

### Model Pipeline

```text
Webcam / Video
      |
      v
Frame Processing
      |
      v
MediaPipe Tasks API
      |
      v
Pose + Left Hand + Right Hand
      |
      v
Landmark Extraction
      |
      v
Spatial Normalization
      |
      v
Temporal Resampling (T=64)
      |
      v
Velocity Feature Extraction
      |
      v
600-Dimensional Feature Sequence
      |
      v
Bi-GRU Model
      |
      v
10-Class Prediction
      |
      v
Confidence Validation
      |
      v
Text + Speech Output
```

### Feature Extraction

The current isolated recognition pipeline uses:

- 33 body pose landmarks.
- 21 left-hand landmarks.
- 21 right-hand landmarks.
- 75 total landmarks.
- 4 channels per landmark: x, y, z, and confidence.

The raw landmark sequence has the shape:

```text
(T, 75, 4)
```

The feature extraction pipeline performs:

1. Shoulder-centered spatial normalization.
2. Temporal resampling to 64 frames.
3. Masked first-order velocity calculation.
4. Spatial and velocity feature combination.

Feature dimensions:

| Feature | Dimensions |
|---|---:|
| Spatial features | 300 |
| Velocity features | 300 |
| Combined features | 600 |
| Final sequence | 64 × 600 |

### Bi-GRU Configuration

The current model configuration is:

- Model: 2-layer Bidirectional GRU.
- Hidden dimension: 128.
- Input dimension: 600.
- Sequence length: 64.
- Output classes: 10.

Do not change the architecture, feature dimensions, sequence length, or checkpoint without inspecting the existing implementation and obtaining approval.

### Current Evaluation Results

The reported held-out test results are:

| Metric | Result |
|---|---:|
| Test samples | 30 |
| Accuracy | 76.67% |
| Top-3 Accuracy | 86.67% |
| Macro F1 | 0.6700 |
| Weighted F1 | 0.7511 |

These results apply only to the reported test split.

They must not be presented as universal ISL recognition accuracy.

Some vocabulary classes have limited training samples. Low-data classes must be clearly documented.

### Prediction Validation

The model must support confidence-based rejection.

- Return a prediction only when it passes the configured acceptance rule.
- Return an uncertain result when confidence is insufficient.
- Do not invent a fallback prediction.
- Read the actual threshold from the current implementation.
- Do not silently change the configured threshold.

---

## 4.2 Continuous Translation Engine

**Status:** Separate experimental pipeline. Verify its current integration status before claiming it is operational.

**Candidate model:** `manavdhamecha77/iSign-t5-pose-to-text`

**Task:** Pose-to-text sequence generation.

This engine is separate from the 10-class Bi-GRU model.

### Important Requirements

- Treat iSign T5 as a pose-to-text model.
- Do not assume it accepts raw webcam frames.
- Inspect its original inference implementation.
- Verify the exact checkpoint configuration.
- Verify expected pose dimensions and landmark ordering.
- Verify normalization and motion processing.
- Verify tokenizer and generation settings.
- Use its matching preprocessing pipeline.
- Test standalone inference before application integration.

### Checkpoint Selection

The repository may contain multiple checkpoint variants.

Do not assume that their configurations or input dimensions are interchangeable.

Before selecting a checkpoint, verify:

- Model architecture.
- Input dimensions.
- Feature representation.
- Pose and hand landmark requirements.
- Motion feature requirements.
- Tokenizer files.
- Model configuration.
- Hardware requirements.

### Validation Requirements

Record the following for every standalone test:

| Field | Required Information |
|---|---|
| Checkpoint | Exact checkpoint path |
| Input sample | Dataset or sample identifier |
| Expected output | Ground-truth text |
| Predicted output | Actual model output |
| Input shape | Actual tensor dimensions |
| Preprocessing | Configuration used |
| Inference time | Measured latency |
| Errors | Any observed failures |

A prediction on an unrelated dataset is not sufficient to establish model accuracy or failure.

Do not describe the experimental engine as a reliable continuous ISL translator until its compatibility and performance have been validated.

---

# 5. Model Separation Rules

The two engines must remain independent.

| Area | Individual Words Engine | Continuous Translation Engine |
|---|---|---|
| Task | Isolated classification | Sequence-to-text |
| Model | Bi-GRU | iSign T5 |
| Input | 64 × 600 feature sequence | Its verified pose sequence format |
| Output | One of 10 sign classes | Generated text |
| Preprocessing | Isolated pipeline | Separate T5 pipeline |
| Checkpoint | Bi-GRU checkpoint | Selected T5 checkpoint |
| Evaluation | Classification metrics | Sequence-level metrics |

Strict rules:

1. Do not replace the Bi-GRU model with iSign T5.
2. Do not use Bi-GRU preprocessing for iSign T5 unless independently verified.
3. Do not use T5 preprocessing for the Bi-GRU model.
4. Do not share incompatible model configurations.
5. Do not combine isolated predictions into a sentence and call it continuous translation.
6. Keep API services and inference modules separate.
7. Keep each model's status independently visible.

---

# 6. Pretrained Model Selection Rules

Before selecting a model, verify:

- Supported sign language.
- Supported vocabulary.
- Input type.
- Output format.
- Model architecture.
- Required preprocessing.
- Dataset used.
- Training and evaluation details.
- Model licence.
- Hardware requirements.
- Inference dependencies.

## Model Input Categories

| Model Type | Expected Input |
|---|---|
| Image Classification | Individual images |
| Video Classification | Video frames or clips |
| Landmark Classification | Extracted landmarks |
| Pose-to-Text | Pose or landmark sequences |
| Continuous Sign Translation | Temporal sign sequences |

Do not pass raw webcam frames into a model that expects pose landmarks.

Do not pass landmarks into a model that expects images.

Never select a model based only on its name or model card title.

---

# 7. Model Integration Rules

Before integrating a new model:

1. Inspect the model repository.
2. Read its original documentation.
3. Inspect the inference implementation.
4. Identify expected input dimensions.
5. Identify normalization requirements.
6. Identify checkpoint and tokenizer requirements.
7. Run standalone inference.
8. Record the test result.
9. Verify compatibility with the intended dataset.
10. Integrate only after compatibility is established.

Never modify the original model architecture just to make incompatible input work without a documented technical reason.

Reuse existing working modules wherever possible.

---

# 8. Preprocessing Rules

Preprocessing is critical for model compatibility.

Antigravity must:

1. Reproduce the preprocessing used by the original model.
2. Preserve landmark ordering.
3. Preserve expected coordinate formats.
4. Apply the correct normalization.
5. Maintain expected input dimensions.
6. Handle missing or invalid landmarks.
7. Validate input shape before inference.
8. Avoid silently changing input values.
9. Keep preprocessing deterministic where possible.

## Landmark Processing

When a model uses MediaPipe landmarks:

- Verify the required MediaPipe solution.
- Verify which landmarks are used.
- Verify landmark ordering.
- Verify coordinate dimensions.
- Verify whether body pose, left hand, right hand, or facial landmarks are required.
- Verify how missing landmarks are represented.
- Verify whether temporal information is required.

Do not assume that all landmark-based models use the same format.

---

# 9. Model Loading Rules

Model loading must happen separately from individual prediction requests.

Requirements:

- Load the model during application startup or through controlled lazy loading.
- Avoid loading model weights for every request.
- Check whether model files exist.
- Validate checkpoint compatibility.
- Handle missing dependencies.
- Handle CPU and CUDA devices.
- Provide clear model status.
- Release resources safely during shutdown where applicable.

The application must not crash silently when model loading fails.

Each engine must have independent loading and status handling.

---

# 10. Hardware and Device Rules

Primary development environment:

- Windows.
- Python 3.11.
- NVIDIA GPU with 4 GB VRAM.
- Local development first.

## Device Selection

Use automatic device selection:

1. Check whether CUDA is available.
2. Use a compatible GPU when available.
3. Fall back to CPU when necessary.
4. Log the selected device.

Example:

```python
import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")
```

## GPU Rules

- Do not assume CUDA is available.
- Verify the installed PyTorch build.
- Do not assume the system CUDA driver version equals the PyTorch CUDA runtime version.
- Monitor GPU memory.
- Avoid loading multiple large models unnecessarily.
- Use smaller checkpoints when appropriate.
- Handle CUDA out-of-memory errors.

If local execution is not practical, Kaggle can be used as a fallback for experiments.

Do not move the entire development workflow to Kaggle without a clear reason.

---

# 11. Inference Rules

Every prediction request must follow a controlled pipeline:

```text
Input
  |
  v
Input Validation
  |
  v
Preprocessing
  |
  v
Model Availability Check
  |
  v
Model Inference
  |
  v
Postprocessing
  |
  v
Output Validation
  |
  v
Structured Prediction Result
```

Requirements:

- Validate input before inference.
- Check model availability.
- Use the correct preprocessing pipeline.
- Run inference using the selected device.
- Use evaluation mode for inference.
- Disable gradient calculation when appropriate.
- Handle inference exceptions.
- Validate model output.
- Return structured results.

Never return a successful prediction when inference has failed.

---

# 12. Dataset Rules

Before using a dataset, verify:

- Dataset source.
- Sign language.
- Vocabulary.
- Annotation format.
- Number of samples.
- Video or image format.
- Landmark availability.
- Train, validation, and test splits.
- Licence and usage restrictions.

## Dataset Handling

- Do not modify original datasets.
- Keep raw datasets separate from processed data.
- Do not commit large datasets to Git.
- Avoid downloading huge archives without checking available storage.
- Preserve original annotations.
- Record preprocessing steps.
- Keep train and test data separate.
- Prevent data leakage between splits.

Respect the dataset licence.

Do not assume that a publicly downloadable dataset is unrestricted for commercial use.

---

# 13. Model Evaluation Rules

Every model must be evaluated before being considered suitable for the application.

## Classification Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Correctness of positive predictions |
| Recall | Coverage of actual positive samples |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Class-level prediction errors |
| Top-K Accuracy | Whether the correct class appears in the top K |
| Inference Time | Prediction speed |
| Memory Usage | Hardware requirements |

## Sequence Translation Metrics

For sequence-to-text models, use suitable sequence-level evaluation methods.

Possible metrics include:

- BLEU.
- ROUGE.
- Word Error Rate where appropriate.
- Character Error Rate where appropriate.
- Sequence-level accuracy where clearly defined.
- Inference latency.

Not every metric applies to every model.

## Evaluation Requirements

- Use compatible test data.
- Avoid evaluating only on training samples.
- Keep test samples separate.
- Record the model checkpoint.
- Record preprocessing configuration.
- Save evaluation results.
- Report limitations.
- Do not invent evaluation metrics.
- Do not report mock-model tests as real model validation.

---

# 14. Backend Integration Rules

AI modules must remain independent from FastAPI routes.

Recommended flow:

```text
FastAPI Route
      |
      v
AI Service
      |
      v
Model Adapter
      |
      v
Preprocessing
      |
      v
Model Inference
      |
      v
Postprocessing
      |
      v
Structured Response
```

Backend requirements:

- Use separate AI service modules.
- Validate API inputs.
- Return structured JSON.
- Handle model loading errors.
- Handle inference errors.
- Avoid blocking unrelated API routes.
- Add suitable request limits.
- Keep model configuration outside route files.
- Report whether each AI engine is available.

Do not create duplicate routes or services without inspecting the existing backend.

---

# 15. API and Engine Separation

The current application has separate API paths for isolated and continuous recognition.

Reported route structure:

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/recognize/isolated/video` | Isolated sign recognition from video |
| `POST /api/v1/recognize/isolated/landmarks` | Isolated recognition from landmarks |
| `POST /api/v1/predict/video` | Continuous pipeline video input |
| `POST /api/v1/predict/landmarks` | Continuous pipeline landmark input |

Verify actual route definitions and request schemas before changing or documenting them.

Do not assume that the continuous endpoints mean the iSign T5 engine is fully integrated or validated.

Health endpoints must report the actual status of the relevant model.

---

# 16. Error Handling Rules

Handle the following conditions:

- Model file missing.
- Invalid checkpoint.
- Unsupported input format.
- Incorrect input dimensions.
- Missing landmarks.
- Invalid landmark values.
- Camera input unavailable.
- CUDA unavailable.
- GPU memory exhausted.
- Inference timeout.
- Model output invalid.
- Model dependency missing.
- Unsupported vocabulary.
- Insufficient recognition confidence.

Errors must be understandable and must not expose sensitive internal details.

Do not convert an error into a fake prediction.

---

# 17. Performance Rules

Optimise only after measuring actual performance.

Priorities:

1. Avoid repeated model loading.
2. Avoid unnecessary frame processing.
3. Use suitable image or video sizes.
4. Avoid unnecessary memory copies.
5. Use GPU acceleration when supported.
6. Process frames at a controlled rate.
7. Keep inference and API response handling efficient.
8. Monitor memory usage.

Do not sacrifice prediction correctness for speed without measuring the impact.

Record actual latency separately for each engine.

---

# 18. Testing Rules

Every AI module must have tests where practical.

Required testing areas:

- Model loading.
- Input validation.
- Preprocessing.
- Landmark extraction.
- Input shape validation.
- Inference.
- Postprocessing.
- API integration.
- Error handling.
- CPU execution.
- CUDA execution when available.

Use mock models for API tests when actual model weights are unavailable.

Clearly distinguish:

- Unit tests.
- Integration tests.
- Mock-model tests.
- Real checkpoint inference tests.
- Held-out model evaluation.

Do not report mock-model tests as real AI model validation.

---

# 19. Model and File Management

Keep large model files outside Git.

Suggested organization:

```text
models/
├── README.md
├── checkpoints/
│   ├── isolated/
│   └── continuous/
└── configs/
    ├── isolated/
    └── continuous/
```

This is a logical organization guide. Reuse the existing project structure where possible.

Rules:

- Do not commit large checkpoints.
- Do not commit downloaded model caches.
- Do not delete existing weights without inspecting them.
- Keep download instructions documented.
- Store model paths in configuration.
- Never hardcode private tokens.
- Do not upload restricted datasets or weights without checking their licences.
- Do not move files belonging to the separate iSign T5 project into Signova AI without approval.

Use `.gitignore` for local model files and temporary experiment outputs.

---

# 20. AI/ML Development Workflow

Follow this implementation order when introducing a new model.

## Phase 1: Research

- Identify candidate pretrained models.
- Verify ISL support.
- Inspect model architecture.
- Inspect input requirements.
- Check model licence.

## Phase 2: Standalone Testing

- Download a suitable checkpoint.
- Set up the environment.
- Test model loading.
- Test preprocessing.
- Run standalone inference.
- Record results.

## Phase 3: Evaluation

- Use compatible samples.
- Compare predictions with ground truth.
- Record relevant metrics.
- Identify limitations.

## Phase 4: Modular Integration

- Create or reuse a model adapter.
- Implement model loading.
- Implement preprocessing.
- Implement inference.
- Implement postprocessing.

## Phase 5: Backend Integration

- Connect the AI service to FastAPI.
- Add validation.
- Add error handling.
- Test API responses.

## Phase 6: Frontend Integration

- Connect the frontend to the backend.
- Send supported inputs.
- Display actual predictions.
- Show processing and error states.

Do not repeat completed work without a technical reason.

---

# 21. Strict Rules for Antigravity

Before changing AI/ML code:

1. Inspect the existing project structure.
2. Read the relevant model documentation.
3. Check existing preprocessing and inference code.
4. Reuse working implementations.
5. Avoid duplicate modules.
6. Do not delete datasets, checkpoints, or experiments without approval.
7. Do not change model architecture without a clear reason and approval.
8. Do not invent model capabilities.
9. Do not generate fake predictions.
10. Do not claim successful testing without running it.
11. Do not install unnecessary dependencies.
12. Keep code modular and maintainable.
13. Keep project files and comments in English.
14. Update relevant documentation when implementation changes.
15. Preserve the separation between the two AI engines.
16. Do not change model thresholds without approval.
17. Do not replace a working checkpoint without approval.
18. Do not retrain the current model without approval.
19. Inspect actual configuration before reporting model settings.
20. Report uncertainty instead of implementing an unsupported assumption.

If model compatibility is uncertain, stop and report the uncertainty.

---

# 22. Final AI/ML Goal

The AI system must provide genuine, testable ISL recognition capabilities.

The priorities are:

1. Correct model selection.
2. Verified input compatibility.
3. Correct preprocessing.
4. Reliable inference.
5. Honest evaluation.
6. Modular backend integration.
7. Practical performance.
8. Clear separation between isolated and continuous recognition.

**Correctness and verified ISL support are more important than adding unsupported features or making inaccurate claims.**