# F-025: Real-World Webcam Validation of Signova AI

## Objective

Implement the next development phase of the existing Signova AI project, following:

- `docs/FEATURES.md`
- `docs/PRD.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/SYSTEM_ARCHITECTURE.md`
- `docs/AI_MODEL_STRATEGY.md`
- `docs/PROGRESS_TRACKER.md`

First inspect the current repository and implementation.

Do not recreate existing features.

## 1. Current Project Architecture

Signova AI currently contains:

- React frontend
- FastAPI backend
- MediaPipe landmark extraction
- Isolated ISL recognition using a trained Bi-GRU model
- 10 individual sign classes
- Webcam testing interface
- Separate continuous iSign T5 experimentation

For this task, focus ONLY on the isolated Bi-GRU recognition application.

Do not modify, move, retrain, or integrate the separate iSign T5 project.

## 2. Existing Model

The current model supports:

1. Hello
2. Thank you
3. Sorry
4. Please
5. Yes
6. No
7. Help
8. Stop
9. Water
10. Food

Existing architecture:

- Bi-GRU
- Input: 64 frames × 600 features
- MediaPipe-based landmark extraction
- Existing confidence rejection threshold: 0.80

Preserve the existing model and preprocessing implementation.

## 3. Webcam Validation

Inspect and extend the existing:

`frontend/src/components/recognition/IsolatedWebcamTester.jsx`

Requirements:

- Test using the user's actual laptop webcam.
- Display live mirrored camera preview.
- Capture a 3-second signing clip.
- Send the clip to the existing isolated recognition API.
- Display the genuine model prediction.
- Display top-3 predictions, confidence and latency.
- Compare the prediction with the manually selected expected sign.
- Show CORRECT, INCORRECT or UNCERTAIN.

The expected sign must never influence model inference.

## 4. Testing Session

Support repeated testing of all 10 signs.

Recommended target: 10 attempts per sign.

Allow users to:

- Start and stop testing.
- Select the expected sign.
- Repeat individual attempts.
- Reset the current testing session.
- View results immediately.

Do not force users to complete all 100 attempts.

## 5. Recognition Feedback and Analytics

Show:

- Total attempts.
- Correct predictions.
- Incorrect predictions.
- Uncertain predictions.
- Overall accuracy.
- Accepted prediction accuracy.
- Prediction coverage.
- Average confidence.
- Average inference latency.

Provide per-sign performance.

Pay particular attention to:

- Stop: 1 training sample.
- Sorry: 3 training samples.

Do not invent confidence scores or predictions.

## 6. Model and Landmark Diagnostics

When possible, expose useful diagnostic information:

- Pose landmark detection.
- Left-hand detection.
- Right-hand detection.
- Missing landmark warnings.
- Invalid feature shape.
- Preprocessing errors.
- Model loading status.

Do not change the model's landmark format or preprocessing logic just to improve the UI.

## 7. Project Scope

Follow the existing product roadmap.

This phase is for testing and validation.

Do not implement prematurely:

- Continuous ISL sentence translation.
- New model architectures.
- Automatic model retraining.
- New vocabulary classes.
- User authentication.
- PostgreSQL integration.
- Cloud GPU infrastructure.

Keep the existing frontend and backend architecture.

Do not add unnecessary dependencies.

## 8. Privacy and Storage

Follow `docs/DATABASE_SCHEMA.md`.

- Do not save raw webcam footage automatically.
- Do not introduce PostgreSQL.
- Keep testing results in session state.
- Provide CSV or JSON export only if practical.
- Do not store unnecessary personal information.

## 9. Testing

Run relevant existing tests.

Verify:

- Camera access and permissions.
- Video recording.
- Backend video endpoint.
- Actual Bi-GRU inference.
- Expected-sign comparison.
- Confidence rejection.
- Prediction metrics.
- Frontend build.
- Existing backend and frontend tests.

Report actual results. Do not claim untested functionality works.

## 10. Documentation

Create:

`docs/reports/F-025-personal-webcam-validation.md`

Update:

`docs/PROGRESS_TRACKER.md`

Include:

- Testing methodology.
- Number of attempts.
- Overall results.
- Per-sign results.
- Confidence and latency.
- Landmark detection issues.
- Incorrect predictions.
- Weak signs.
- Recommendations for F-026.

## Final Requirement

F-025 must extend the existing Signova AI application.

The objective is to determine how reliably the existing isolated Bi-GRU model recognizes signs performed by a real person.

Do not claim production-level ISL translation.

Do not replace the current Bi-GRU model with iSign T5.

Inspect first, implement only necessary changes, test, and report.