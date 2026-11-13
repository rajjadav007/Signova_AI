# F-025: Signova AI Real-World Validation

## Objective

Implement real-world validation of the existing Signova AI application, following the Hackathon Demonstration & Evaluation Guide.

Read:
- `docs/FEATURES.md`
- `docs/PRD.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/SYSTEM_ARCHITECTURE.md`
- `docs/AI_MODEL_STRATEGY.md`
- `docs/PROGRESS_TRACKER.md`
- The Hackathon Demonstration & Evaluation Guide

## 1. Preserve the Dual-Engine Architecture

Signova AI has two independent engines.

Engine 1: Individual Words (10-Sign MVP)
- Bi-GRU model
- MediaPipe landmark extraction
- 64 × 600 input features
- 10 individual ISL words
- Text and speech output

Engine 2: Continuous Translation (iSign T5)
- Pretrained T5 encoder-decoder
- Pose sequence processing
- Continuous sign-to-text translation

For F-025, work ONLY on Engine 1.

Do not modify Engine 2, its model weights, configuration, preprocessing, or API routes.

Do not remove the existing mode switcher.

## 2. Inspect Existing Implementation

Inspect the existing webcam component, API routes, model loader, inference pipeline and testing history.

The documented webcam workflow includes:
- Enable Camera
- Start Recording
- Stop & Recognize
- Display prediction
- Display confidence and latency
- Display top-3 candidates
- Text-to-speech

Retain working functionality. Do not create duplicate components.

## 3. Validate All 10 Signs

Test:
hello, thank you, sorry, please, yes, no, help, stop, water and food.

Allow repeated attempts and compare genuine predictions against manually selected expected signs.

The expected label must never influence model inference.

## 4. Evaluation Dashboard

Show:
- Total attempts
- Correct predictions
- Incorrect predictions
- Uncertain predictions
- Overall accuracy
- Accepted prediction accuracy
- Prediction coverage
- Average confidence
- Average inference latency

Provide per-sign metrics and identify weak classes.

## 5. Model Configuration Verification

Inspect the actual confidence threshold in the model configuration.

The demonstration guide mentions 0.75, while previous development reports mention 0.80.

Do not silently change the threshold. Report the actual configured value and retain it during baseline testing.

Also verify that confidence values and top-3 probabilities come from actual model inference.

## 6. Webcam and Landmark Diagnostics

Where practical, show:
- Pose detection status
- Left-hand detection
- Right-hand detection
- Missing landmark warnings
- Invalid input warnings
- Model loading status

Do not change existing preprocessing or landmark ordering.

## 7. Demo Sample Verification

Inspect `datasets/demo_samples/`.

Verify all 10 sample videos using the actual model.

Run:

`.venv\Scripts\python.exe -m pytest tests/test_demo_samples.py -v`

Do not assume the documented confidence values are current until they have been reproduced.

## 8. Privacy and Storage

- No automatic webcam video storage.
- No mandatory PostgreSQL.
- Keep test results in session state.
- Provide CSV or JSON export if practical.

## 9. Regression Testing

Run the relevant existing frontend and backend tests.

Verify both application modes remain functional.

Specifically verify:
- Individual Words mode.
- Continuous Translation mode.
- Mode switching.
- Isolated Bi-GRU inference.
- Existing iSign T5 routes and configuration remain untouched.

## 10. Documentation

Create:

`docs/reports/F-025-personal-webcam-validation.md`

Update:

`docs/PROGRESS_TRACKER.md`

Report actual results, limitations, and recommended F-026 improvements.

## Final Rules

- Extend the existing application.
- Do not create a second application.
- Do not retrain the model automatically.
- Do not change the model architecture.
- Do not modify the T5 engine.
- Do not claim perfect recognition.
- Do not claim continuous translation accuracy without evidence.

Inspect first, implement only necessary changes, run tests, and report verified results.