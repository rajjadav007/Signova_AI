# Phase Roadmap

## Signova AI

**Project:** Signova AI  
**Version:** 2.0  
**Status:** Active Development  
**Development Approach:** Backend First, Incremental Integration  
**Primary Environment:** Local Windows  
**Fallback Environment:** Kaggle GPU  
**Frontend:** React + JavaScript + Vite  
**Backend:** Python 3.11 + FastAPI  
**Primary Recognition Model:** Bi-GRU  
**Experimental Translation Model:** iSign T5

---

# 1. Roadmap Overview

This roadmap defines the development phases of Signova AI, from environment setup to a functional Indian Sign Language recognition application.

Signova AI currently has two independent AI engines.

### Engine 1: Individual Words Recognition

**Status:** Implemented MVP.

- Uses a 10-class Bi-GRU recognition model.
- Processes pose and hand landmarks.
- Uses 600-dimensional features.
- Uses temporal resampling with T=64.
- Supports 10 isolated ISL signs.
- Provides prediction text, confidence, top-3 results, and recognition status.

### Engine 2: Continuous Translation

**Status:** Experimental, separate pipeline.

- Uses the iSign T5 pose-to-text model.
- Has separate model files and preprocessing requirements.
- Must not reuse the Bi-GRU inference pipeline.
- Must not be presented as a working continuous translator until its integration and evaluation are verified.

### Immediate Development Goal

Prepare a reliable hackathon demonstration using the existing Individual Words recognition engine.

Do not rebuild completed functionality or train another model from scratch without approval.

---

# 2. Development Status Summary

The following statuses reflect previously reported implementation progress. Antigravity must inspect the current repository, execute relevant tests, and verify each completion criterion before confirming the status.

| Phase | Area | Reported status |
|---|---|---|
| 0 | Repository and Environment Setup | Completed |
| 1 | Backend Foundation | Completed |
| 2 | AI Model Research and Experimentation | Completed for isolated model |
| 3 | Model Evaluation and Selection | Completed for isolated model |
| 4 | Model Adapter and Inference Integration | Completed for isolated model |
| 5 | Visual Input and Processing Pipeline | Completed for isolated model |
| 6 | AI Prediction API | Completed for isolated model |
| 7 | Frontend Development and Integration | Implemented |
| 8 | Text-to-Speech Integration | Implemented, verify |
| 9 | Testing and Optimisation | In Progress |
| 10 | Demo and Hackathon Preparation | In Progress |

**Important:** These are reported statuses, not an independent audit. Do not mark a phase complete solely because files or functions exist.

---

# 3. Development Phases

## Phase 0: Repository and Environment Setup

**Priority:** Critical  
**Reported Status:** Completed

### Objectives

Prepare the project structure and development environments.

### Tasks

- [x] Inspect the existing repository.
- [x] Review project documentation.
- [x] Establish frontend and backend directories.
- [x] Configure Python 3.11.
- [x] Configure Node.js and npm.
- [x] Configure the Python environment.
- [x] Verify PyTorch installation.
- [x] Check NVIDIA GPU availability.
- [x] Configure project dependencies.
- [x] Configure environment variables.
- [x] Keep model checkpoints and datasets outside Git.

### Completion Criteria

Python, Node.js, backend dependencies, and frontend dependencies work in the development environment.

### Verification

Check the current environment and dependency files before confirming completion.

---

## Phase 1: Backend Foundation

**Priority:** Critical  
**Reported Status:** Completed

### Objectives

Establish the FastAPI backend and its core application services.

### Tasks

- [x] Create the FastAPI application.
- [x] Configure application settings.
- [x] Configure CORS.
- [x] Establish API route structure.
- [x] Implement health and service status functionality.
- [x] Add request validation.
- [x] Add error handling.
- [x] Configure API documentation.
- [x] Add backend tests.

### Completion Criteria

The backend starts successfully, exposes implemented endpoints, and returns valid responses.

### Verification

- Start the backend.
- Check health endpoints.
- Execute backend tests.
- Confirm error handling.

---

## Phase 2: AI Model Research and Experimentation

**Priority:** Critical  
**Reported Status:** Completed for the isolated recognition model

### Objectives

Identify and implement a suitable pretrained or previously trained model for the isolated ISL recognition task.

### Actual Isolated Recognition Model

**Model:** Bi-GRU  
**Dataset:** `vidit031/isl-isolated-40words`

Supported vocabulary:

1. Food
2. Hello
3. Help
4. No
5. Please
6. Sorry
7. Stop
8. Thank you
9. Water
10. Yes

### Implemented Model Pipeline

- MediaPipe Tasks API.
- Pose and hand landmark extraction.
- 75 landmarks with 4 channels.
- Shoulder-centred spatial normalisation.
- Temporal resampling to T=64.
- First-order velocity features.
- Combined 600-dimensional features.
- Two-layer Bi-GRU with hidden dimension 128.
- Bilateral mirror augmentation.
- Confidence-based prediction rejection.

### Separate Experimental Candidate

The iSign T5 model remains a separate continuous translation candidate.

It must not replace the current Bi-GRU model or be treated as the isolated recognition engine.

### Completion Criteria

The isolated recognition model has a working inference pipeline and produces genuine predictions on compatible samples.

### Verification

- Inspect the actual checkpoint.
- Verify model configuration.
- Verify preprocessing compatibility.
- Confirm class mapping.
- Run genuine inference.
- Check model loading and resource usage.

---

## Phase 3: Model Evaluation and Selection

**Priority:** Critical  
**Reported Status:** Completed for the isolated recognition model

### Objectives

Evaluate the isolated recognition model using held-out test data.

### Reported Evaluation Results

| Metric | Reported result |
|---|---:|
| Test samples | 30 |
| Accuracy | 76.67% |
| Top-3 accuracy | 86.67% |
| Macro F1 | 0.6700 |
| Weighted F1 | 0.7511 |
| Reported CPU inference | Approximately 18 ms |

### Additional Evaluation Work

- [x] Dataset inspection.
- [x] Class validation.
- [x] Train, validation, and test splitting.
- [x] Landmark extraction.
- [x] Feature engineering.
- [x] Model benchmarking.
- [x] Generalisation analysis.
- [x] Bilateral mirror augmentation.
- [x] Held-out evaluation.

### Important Limitations

- The held-out test set contains only 30 samples.
- Some classes have very limited training data.
- Results do not establish universal ISL recognition accuracy.
- Performance may vary across signers, lighting, backgrounds, and camera conditions.

### Completion Criteria

The model has reproducible evaluation results and documented limitations.

### Verification

Re-run evaluation only when required. Do not retrain or replace the current checkpoint without approval.

---

## Phase 4: Model Adapter and Inference Integration

**Priority:** Critical  
**Reported Status:** Implemented for isolated recognition

### Objectives

Integrate the Bi-GRU model into the backend inference architecture.

### Tasks

- [x] Establish isolated recognition modules.
- [x] Implement model loading.
- [x] Implement model configuration.
- [x] Implement model-specific preprocessing.
- [x] Implement inference execution.
- [x] Implement prediction validation.
- [x] Implement output formatting.
- [x] Implement confidence-based rejection.
- [x] Add inference tests.

### Architecture Separation

Keep the following independent:

**Individual Words Engine**
- Bi-GRU model.
- Isolated preprocessing.
- Isolated feature extraction.
- Isolated inference.

**Continuous Translation Engine**
- iSign T5 model.
- T5-specific preprocessing.
- T5-specific inference.
- Separate model status.

### Completion Criteria

The Bi-GRU model can be loaded and executed through the backend without depending on frontend code.

---

## Phase 5: Visual Input and Processing Pipeline

**Priority:** Critical  
**Reported Status:** Implemented for isolated recognition

### Objectives

Convert supported video input into the exact feature format required by the Bi-GRU model.

### Tasks

- [x] Implement video input processing.
- [x] Implement frame extraction.
- [x] Integrate MediaPipe Tasks API.
- [x] Extract pose landmarks.
- [x] Extract left and right hand landmarks.
- [x] Implement coordinate normalisation.
- [x] Implement temporal resampling.
- [x] Implement velocity features.
- [x] Validate model input dimensions.
- [x] Test compatible samples.

### Actual Isolated Input Specification

| Property | Value |
|---|---|
| Pose landmarks | 33 |
| Left hand landmarks | 21 |
| Right hand landmarks | 21 |
| Total landmarks | 75 |
| Channels | 4 |
| Temporal length | 64 |
| Spatial features | 300 |
| Velocity features | 300 |
| Combined features | 600 |
| Final input shape | `(64, 600)` |

### Completion Criteria

Supported video input is converted into the model-compatible feature format.

### Verification

Inspect the actual preprocessing implementation and test outputs.

Do not assume that generic MediaPipe landmarks are automatically compatible with iSign T5.

---

## Phase 6: AI Prediction API

**Priority:** Critical  
**Reported Status:** Implemented for isolated recognition

### Objectives

Expose supported inference functionality through FastAPI.

### Reported API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/recognize/isolated/video` | Individual sign recognition from video |
| POST | `/api/v1/recognize/isolated/landmarks` | Individual sign recognition from landmarks |
| POST | `/api/v1/predict/video` | Separate continuous translation pipeline |
| POST | `/api/v1/predict/landmarks` | Separate continuous translation pipeline |

### Tasks

- [x] Create AI service modules.
- [x] Connect the isolated model loader.
- [x] Implement isolated recognition endpoints.
- [x] Add request validation.
- [x] Add prediction response formatting.
- [x] Add inference error handling.
- [x] Add processing time measurement.
- [x] Add API tests.

### Completion Criteria

Supported isolated recognition input reaches the Bi-GRU model and returns genuine predictions through FastAPI.

### Important

The existence of a continuous translation route does not prove that the T5 model is fully integrated or operational.

Verify its model loading, preprocessing, inference, and output before presenting it as available.

---

## Phase 7: Frontend Development and Integration

**Priority:** Critical  
**Reported Status:** Implemented MVP

### Objectives

Connect the React frontend to the backend and provide a usable recognition interface.

### Technology

- React.
- JavaScript ES6+.
- Vite.
- CSS.
- Fetch API or Axios.

Do not introduce TypeScript.

### Individual Words Interface

Reported implemented functionality:

- [x] Webcam preview.
- [x] Camera start and stop controls.
- [x] Camera permission handling.
- [x] Three-second countdown.
- [x] Capture and Recognize.
- [x] Expected-sign selector.
- [x] Video upload fallback.
- [x] Prediction display.
- [x] Confidence display.
- [x] Top-3 predictions.
- [x] Recognition rejection reason.
- [x] Inference latency.
- [x] Speech controls.
- [x] Recognition evaluation history.

### Evaluation History

Display:

- Total attempts.
- Correct predictions.
- Incorrect predictions.
- Uncertain predictions.
- Reset history control.

### Continuous Translation Interface

Keep the separate continuous mode independent.

Show its real operational status. Do not display fabricated sentence translations.

### Completion Criteria

Users can provide supported signing input and receive actual backend predictions through the frontend.

### Verification

- Test camera permissions.
- Test capture and recognition.
- Test video upload.
- Test API failures.
- Test actual prediction display.
- Test responsive layout.

---

## Phase 8: Text-to-Speech Integration

**Priority:** High  
**Reported Status:** Implemented, verification required

### Objectives

Convert available recognition text into spoken output.

### Tasks

- [x] Integrate or provide browser SpeechSynthesis support.
- [x] Provide speech playback controls.
- [x] Provide speech stop control.
- [ ] Verify speech behaviour across supported browsers.
- [ ] Verify repeated speech cancellation.
- [ ] Verify speech cleanup.
- [ ] Verify unsupported browser handling.

### Completion Criteria

Actual recognized text can be converted into speech using a supported browser voice.

Speech must not alter the original AI prediction.

---

## Phase 9: Testing and Optimisation

**Priority:** High  
**Reported Status:** In Progress

### Objectives

Validate the complete supported application workflow.

### Reported Test Results

| Area | Previously reported result |
|---|---|
| Frontend Vitest | 4 passed |
| Frontend build | Successful |
| Isolated pipeline tests | 15 passed, 1 skipped |
| Isolated validation tests | 6 passed |

These are historical reported results. Re-run the current tests before using them as final submission evidence.

### Tasks

- [x] Implement backend tests.
- [x] Implement frontend tests.
- [x] Test model input validation.
- [x] Test prediction handling.
- [ ] Re-run all current backend tests.
- [ ] Re-run all current frontend tests.
- [ ] Test actual webcam recognition.
- [ ] Test different signers.
- [ ] Test different lighting conditions.
- [ ] Test incorrect and uncertain predictions.
- [ ] Test API failure handling.
- [ ] Test camera cleanup.
- [ ] Test speech output.
- [ ] Measure end-to-end latency.
- [ ] Review resource usage.
- [ ] Fix critical bugs.

### Completion Criteria

The supported recognition workflow operates reliably and known limitations are documented.

---

## Phase 10: Demo and Hackathon Preparation

**Priority:** Critical  
**Reported Status:** In Progress

### Objectives

Prepare Signova AI for a clear, reliable, and honest hackathon demonstration.

### Tasks

- [x] Prepare the hackathon demonstration guide.
- [x] Document the two-engine architecture.
- [x] Document isolated recognition evaluation results.
- [x] Document known model limitations.
- [ ] Verify the current repository state.
- [ ] Run the backend from a clean startup.
- [ ] Run the frontend from a clean startup.
- [ ] Test the complete webcam workflow.
- [ ] Test all 10 supported signs.
- [ ] Prepare compatible demonstration videos.
- [ ] Verify actual prediction responses.
- [ ] Verify confidence and rejection behaviour.
- [ ] Verify speech output.
- [ ] Prepare fallback video samples.
- [ ] Prepare the project presentation.
- [ ] Prepare the architecture explanation.
- [ ] Prepare model evaluation evidence.
- [ ] Prepare answers about model limitations.
- [ ] Verify the final demo environment.

### Completion Criteria

The demonstration works with genuine model predictions and does not depend on fabricated outputs.

---

# 4. Future Development Phases

These phases are outside the current 10-sign MVP.

## Phase 11: Expanded ISL Vocabulary

**Priority:** High for future development  
**Status:** Pending

### Objectives

Expand the isolated recognition vocabulary beyond the current 10 signs.

### Potential Tasks

- Add additional verified ISL signs.
- Inspect suitable datasets.
- Validate class labels and samples.
- Evaluate additional pretrained models.
- Collect more signer variations where permitted.
- Improve recognition across signers.
- Test different backgrounds and lighting.
- Address low-data classes.
- Evaluate class-level performance.
- Validate the expanded vocabulary before release.

### Completion Criteria

Additional signs are supported by validated data and measured recognition performance.

Do not add a sign to the supported vocabulary simply because it appears in a dataset.

---

## Phase 12: Dynamic ISL Recognition

**Priority:** High for future development  
**Status:** Pending

### Objectives

Move beyond isolated sign classification towards recognition of longer movement-based signing sequences.

### Potential Tasks

- Investigate suitable temporal recognition models.
- Process longer video sequences.
- Recognize movement-based signs.
- Improve sign transition detection.
- Investigate signer-independent recognition.
- Evaluate dynamic recognition datasets.
- Measure sequence-level performance.
- Validate real-time processing.

### Completion Criteria

A validated dynamic recognition pipeline can recognize supported signing sequences under documented conditions.

---

## Phase 13: Continuous ISL Translation

**Priority:** High for future development  
**Status:** Experimental research

### Objectives

Develop a validated pipeline that converts continuous ISL signing into meaningful text.

### Potential Tasks

- Investigate continuous ISL datasets.
- Verify iSign T5 model compatibility.
- Validate pose preprocessing.
- Implement sign segmentation.
- Recognize connected signs.
- Investigate context-aware sentence generation.
- Evaluate sentence-level translation.
- Investigate ISL grammar.
- Measure translation quality.
- Test across different signers.

### Important Requirements

- Keep iSign T5 separate from Bi-GRU.
- Verify model inputs and outputs.
- Do not assume isolated recognition results form a sentence.
- Do not claim complete continuous translation without suitable evaluation.

### Completion Criteria

A continuous ISL translation pipeline produces measurable, validated sentence-level results on suitable data.

---

## Phase 14: Advanced Application Features

**Priority:** Medium  
**Status:** Pending

### Potential Tasks

- User authentication.
- Recognition history persistence.
- PostgreSQL integration.
- User preferences.
- Improved speech options.
- Additional accessibility features.
- Expanded supported language options for speech.
- User feedback and evaluation tools.

### Requirements

- Add authentication only when required.
- Keep PostgreSQL optional until persistent storage is needed.
- Protect user data.
- Do not store raw webcam footage by default.

---

# 5. Current Hackathon MVP Scope

The hackathon MVP should focus on the smallest complete and demonstrable recognition workflow.

| Feature | Current MVP scope |
|---|---|
| Backend | FastAPI |
| Primary Model | 10-class Bi-GRU |
| Model Input | 75 landmarks, 600D features |
| Inference | Genuine model output |
| Frontend | React + JavaScript + Vite |
| Camera | Permission and live preview |
| Recognition | Capture and recognize |
| Result | Prediction, confidence, top-3 where available |
| Evaluation | Expected-sign comparison and history |
| Speech | Browser text-to-speech |
| Testing | Backend and frontend validation |
| Documentation | Architecture, evaluation, limitations |

### MVP Boundaries

The MVP does not claim:

- Universal ISL recognition.
- Recognition of every ISL sign.
- Complete continuous sentence translation.
- Support for all regional sign variations.
- Perfect recognition across all users and environments.

Do not attempt to complete all future phases during the hackathon.

---

# 6. Development Dependencies

The original development phases establish the following dependency order:

```text
Phase 0: Environment Setup
          |
          v
Phase 1: Backend Foundation
          |
          v
Phase 2: Model Research
          |
          v
Phase 3: Model Evaluation
          |
          v
Phase 4: Model Integration
          |
          v
Phase 5: Visual Processing
          |
          v
Phase 6: Prediction API
          |
          v
Phase 7: Frontend Integration
          |
          v
Phase 8: Speech Output
          |
          v
Phase 9: Testing
          |
          v
Phase 10: Hackathon Demo
```

Future development proceeds independently where appropriate:

- Phase 11 expands isolated vocabulary.
- Phase 12 investigates dynamic recognition.
- Phase 13 develops continuous translation.
- Phase 14 adds advanced application features.

If model evaluation fails, revisit model research before proceeding with model-dependent implementation.

Do not repeat completed work without a technical reason.

---

# 7. Milestone Summary

| Milestone | Expected outcome |
|---|---|
| M0 | Development environment ready |
| M1 | Backend running |
| M2 | Isolated recognition model validated |
| M3 | Evaluation results documented |
| M4 | Bi-GRU inference integrated |
| M5 | Landmark processing working |
| M6 | Isolated prediction API working |
| M7 | Frontend connected |
| M8 | Speech output verified |
| M9 | Complete supported workflow tested |
| M10 | Hackathon demonstration ready |
| M11 | Expanded vocabulary |
| M12 | Dynamic recognition |
| M13 | Continuous ISL translation |
| M14 | Advanced application features |

---

# 8. Progress Tracking Rules

Update this roadmap whenever a phase starts, changes, or finishes.

Use these statuses:

- `Pending`
- `In Progress`
- `Blocked`
- `Completed`
- `Cancelled`

### Completion Rules

A phase should be marked completed only when:

1. Its implementation exists.
2. Its acceptance criteria are satisfied.
3. Relevant tests have been executed.
4. Actual results are recorded.
5. Known limitations are documented.

Do not mark a feature completed merely because its files or functions have been created.

Record actual test results, blockers, and remaining work.

Do not replace historical test results with newer values without preserving the distinction.

---

# 9. Instructions for Antigravity

Before starting a development phase:

1. Read `AGENTS.md` if present.
2. Read `docs/PRD.md`.
3. Read `docs/PROJECT_OVERVIEW.md`.
4. Read `docs/FEATURES.md`.
5. Read `docs/TECH_STACK.md`.
6. Read `docs/SYSTEM_ARCHITECTURE.md`.
7. Read `docs/AI_MODEL_STRATEGY.md`.
8. Read `docs/API_DOCUMENTATION.md`.
9. Read `docs/FRONTEND_DEVELOPMENT_RULES.md`.
10. Read `docs/BACKEND_DEVELOPMENT_RULES.md`.
11. Read this roadmap.
12. Inspect the current repository state.

### Implementation Rules

- Follow the roadmap without rebuilding completed features.
- Work backend-first when implementing new backend-dependent functionality.
- Do not skip model evaluation.
- Do not train a model from scratch without approval.
- Do not treat ASL as ISL.
- Do not claim continuous ISL translation without validation.
- Do not create unnecessary files.
- Preserve existing useful code and data.
- Keep model weights outside Git.
- Keep Bi-GRU and iSign T5 pipelines independent.
- Do not change model weights, checkpoints, or thresholds without approval.
- Do not introduce TypeScript.
- Do not introduce mandatory PostgreSQL for the MVP.
- Test each new phase before moving forward.
- Update relevant documentation.

### After Each Phase, Report

- Phase completed or current status.
- Files created or modified.
- Features implemented.
- Tests executed.
- Actual test results.
- Blockers.
- Remaining tasks.
- Recommended next phase.

Do not claim completion without evidence.

---

# 10. Final Roadmap Goal

The immediate goal is to deliver a working Signova AI prototype that:

1. Accepts supported signing input.
2. Processes it using the correct model-specific pipeline.
3. Runs the validated 10-class Bi-GRU model.
4. Displays genuine recognition results.
5. Shows uncertainty and error states.
6. Provides speech output.
7. Demonstrates measurable recognition performance.
8. Documents current limitations.

The long-term goal is to expand Signova AI into a reliable dynamic and continuous Indian Sign Language understanding system.

**Build on the working MVP, validate every capability, and expand only when the required data and models are ready.**