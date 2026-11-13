# F-025: Personal Webcam Testing and Model Validation

## Project: Signova AI

### Objective

Implement the first real-world testing phase for the existing Signova AI isolated sign recognition model.

I want to test the trained Bi-GRU model using my own hand through my laptop webcam.

This is an extension of the existing project, not a new application.

## 1. Inspect the Existing Project First

Before making changes, inspect:

- Existing React frontend.
- Existing FastAPI backend.
- `IsolatedWebcamTester.jsx`.
- Isolated recognition API routes.
- `IsolatedSignRecognizer`.
- Trained Bi-GRU checkpoint.
- Existing confidence rejection logic.
- Existing test history implementation.
- Existing frontend and backend tests.

Understand the existing implementation before modifying anything.

Do not recreate features that already work.

## 2. Model Information

The current isolated recognition model supports these 10 signs:

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

Model details:

- Architecture: Bi-GRU
- Input: 64 temporal frames × 600 features
- Landmark extraction: MediaPipe
- Confidence rejection threshold: 0.80
- Current held-out test accuracy: 76.67%

These are existing reported results. Do not change or misrepresent them.

## 3. Personal Webcam Testing

Extend the existing webcam interface to support systematic testing.

Requirements:

- Use the laptop webcam.
- Display a mirrored webcam preview.
- Keep the existing upper-body framing guide.
- Allow selecting an expected sign.
- Capture a 3-second video.
- Send the video to the existing isolated recognition API.
- Display the model's actual prediction.
- Display confidence and top-3 predictions.
- Display inference latency.
- Show CORRECT, INCORRECT, or UNCERTAIN status.

Important: The expected sign is only ground truth for evaluation. It must never influence the model prediction.

## 4. Testing Session

Allow the user to test each sign repeatedly.

Recommended initial test:

- 10 attempts per sign.
- 100 total attempts for all 10 signs.

The user should be able to:

- Start a testing session.
- Select the expected sign.
- Perform the sign.
- Record the prediction.
- Repeat the test.
- Reset the current session.

Do not force the user to complete all 100 attempts in one session.

## 5. Testing Analytics

Show a testing dashboard with:

- Total attempts.
- Correct predictions.
- Incorrect predictions.
- Uncertain predictions.
- Overall accuracy.
- Accepted prediction accuracy.
- Prediction coverage.
- Average confidence.
- Average inference latency.

Calculate:

Overall accuracy = Correct predictions / Total attempts × 100.

Accepted prediction accuracy = Correct predictions / (Correct + Incorrect) × 100.

Prediction coverage = (Correct + Incorrect) / Total attempts × 100.

Do not count uncertain predictions as correct.

## 6. Per-Sign Performance

Display a table:

| Sign | Attempts | Correct | Incorrect | Uncertain | Accuracy |
|---|---:|---:|---:|---:|---:|
| Hello | | | | | |
| Thank you | | | | | |
| Sorry | | | | | |
| Please | | | | | |
| Yes | | | | | |
| No | | | | | |
| Help | | | | | |
| Stop | | | | | |
| Water | | | | | |
| Food | | | | | |

Identify weak-performing signs.

Pay special attention to:

- Stop: only 1 training sample.
- Sorry: only 3 training samples.

Show warnings for these low-data classes.

## 7. Database Decision

Follow `docs/DATABASE_SCHEMA.md`.

PostgreSQL is optional and is not required for this MVP.

Do not introduce:

- PostgreSQL.
- SQLAlchemy.
- Alembic.
- User authentication.
- Database tables.
- Database dependencies.

Keep testing results in frontend session state.

If useful, allow the user to export the test report as CSV or JSON.

Do not upload or permanently store raw webcam footage.

## 8. Model Safety

Do not:

- Retrain the model automatically.
- Change the Bi-GRU architecture.
- Change the confidence threshold.
- Modify the separate iSign T5 project.
- Mix isolated recognition with continuous translation.
- Modify model weights during testing.

The purpose is to evaluate the current model, not alter its performance.

## 9. Testing and Verification

After implementation:

- Run existing frontend tests.
- Run existing backend tests.
- Test webcam permissions and recording.
- Test the actual video recognition endpoint.
- Verify expected-sign evaluation.
- Verify metric calculations.
- Verify uncertain prediction handling.
- Verify CSV/JSON export if implemented.

Do not claim successful tests unless they actually pass.

## 10. Documentation

Create or update:

`docs/reports/F-025-personal-webcam-testing.md`

Include:

- Objective.
- Existing model information.
- Testing methodology.
- Test results.
- Per-sign accuracy.
- Incorrect predictions.
- Uncertain predictions.
- Confidence statistics.
- Latency statistics.
- Known limitations.
- Recommended next steps.

Update `docs/PROGRESS_TRACKER.md`.

## Final Requirement

Implement F-025 inside the existing Signova AI application.

Do not create another application.

Do not introduce PostgreSQL for this phase.

Do not claim that the model is perfect.

The goal is to measure how well the existing trained Bi-GRU model recognizes signs performed by a real person through a webcam.