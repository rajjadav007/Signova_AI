# Product Requirements Document (PRD)

## Signova AI

**Product Name:** Signova AI  
**Product Type:** AI-powered Indian Sign Language Recognition and Translation Web Application  
**Version:** 2.0  
**Status:** Development Planning  
**Platform:** Web Application

---

# 1. Product Overview

Signova AI is an AI-powered web application designed to reduce communication barriers between Indian Sign Language (ISL) users and people who do not understand sign language.

The application captures signing through a webcam, extracts relevant visual or landmark information, processes it using a suitable ISL recognition model, and presents the recognised output as text.

The recognised text can also be converted into speech.

The system will use a modular architecture so that different pretrained ISL models can be evaluated and integrated without redesigning the entire application.

## 1.1 Product Vision

To develop an accessible, practical, and extensible ISL recognition application that supports communication through visual sign recognition and spoken output.

## 1.2 Core Workflow

```text
User performs ISL signs
        |
        v
Webcam captures video
        |
        v
Frame processing
        |
        v
Visual / Landmark extraction
        |
        v
Preprocessing
        |
        v
ISL recognition model
        |
        v
Prediction processing
        |
        v
Recognised text
        |
        v
Text-to-speech
```

---

# 2. Problem Statement

People who communicate using Indian Sign Language can face communication barriers when interacting with individuals who do not understand ISL.

Existing communication methods may require a human interpreter or manual assistance.

Signova AI aims to provide a technology-assisted communication interface that recognises supported ISL signs and converts the results into readable text and speech.

The project also addresses the technical challenge of integrating pretrained sign recognition models into a real-time web application.

---

# 3. Product Goals

The primary goals are:

1. Build a functional ISL recognition web application.
2. Capture live video through the user's webcam.
3. Extract visual or landmark features required by the selected model.
4. Integrate a suitable pretrained ISL recognition model.
5. Convert recognised signs into text.
6. Provide speech output for recognised text.
7. Develop a modular FastAPI backend.
8. Support local model inference using available hardware.
9. Evaluate recognition quality using appropriate ISL test samples.
10. Keep the system extensible for future recognition improvements.

---

# 4. Target Users

## 4.1 Primary Users

- People who communicate using Indian Sign Language.
- Individuals who want to communicate with ISL users.
- People interested in learning and understanding supported ISL signs.

## 4.2 Secondary Users

- Researchers exploring sign language recognition.
- Students and developers working on accessibility technologies.
- Organisations interested in assistive communication systems.

---

# 5. Product Scope

## 5.1 Initial MVP Scope

The first working version will focus on the following features:

| Feature | Description | Priority |
|---|---|---|
| Webcam access | Capture live video from the browser | Critical |
| Camera controls | Start and stop camera | Critical |
| Frame processing | Prepare frames for AI processing | Critical |
| Landmark extraction | Extract required hand and body landmarks when supported by the selected model | Critical |
| Preprocessing | Convert input into the model's expected format | Critical |
| Model integration | Load and execute a verified pretrained ISL model | Critical |
| Sign recognition | Recognise supported signs | Critical |
| Text output | Display recognised predictions | Critical |
| Text-to-speech | Speak recognised text | High |
| Error handling | Handle unsupported or uncertain input | High |
| Recognition status | Display processing and prediction status | High |

The initial MVP will support only the recognition capability validated by the selected model.

## 5.2 Future Scope

The following features are planned for later development:

- Expanded ISL vocabulary.
- Dynamic sign recognition.
- Recognition of longer sign sequences.
- Continuous ISL sentence understanding.
- Improved sentence construction.
- Recognition of facial and non-manual sign features.
- Recognition history.
- User accounts and personal settings.
- Additional accessibility improvements.

These features are not considered complete MVP functionality.

---

# 6. Functional Requirements

## FR-01: Webcam Access

The application must:

- Request permission before accessing the camera.
- Display a live webcam preview.
- Provide start and stop controls.
- Handle unavailable cameras.
- Display permission and camera errors.

## FR-02: Video Processing

The application must:

- Capture frames from the webcam.
- Process frames in a controlled sequence.
- Support the input format required by the selected model.
- Avoid unnecessary frame processing.
- Handle missing or invalid frames.

## FR-03: Visual Feature Extraction

The AI pipeline must:

- Extract the visual information required by the selected model.
- Support hand and body landmarks when required.
- Maintain consistent feature ordering.
- Handle missing landmarks.
- Preserve temporal ordering for sequence-based models.

MediaPipe will be used only when its output is compatible with the selected model's input requirements.

## FR-04: Preprocessing

The AI pipeline must:

- Apply the model's original preprocessing requirements.
- Validate input dimensions.
- Handle missing values.
- Normalize data when required.
- Prepare image or landmark sequences in the expected format.

## FR-05: Model Inference

The backend must:

- Load the selected model and its checkpoint.
- Reuse the loaded model for subsequent requests.
- Process compatible input.
- Return the model's actual prediction.
- Handle inference errors.
- Identify unsupported inputs.

The application must not generate fabricated predictions when model inference fails.

## FR-06: Recognition Output

The application must:

- Display the recognised sign or text.
- Handle uncertain predictions.
- Display an appropriate message for unsupported signs.
- Show confidence only when the model provides a meaningful confidence measure.
- Identify the recognition type where necessary.

## FR-07: Speech Output

The application must:

- Convert recognised text into speech.
- Provide speech playback controls.
- Support replay and stop actions.
- Handle empty text and unavailable speech voices.

Browser SpeechSynthesis will be considered for the initial implementation. A backend speech service may be added later.

## FR-08: Backend API

The backend must:

- Expose health-check endpoints.
- Validate recognition requests.
- Connect API routes to application services.
- Call the AI inference pipeline.
- Return structured responses.
- Handle errors consistently.

## FR-09: Model Management

The system must:

- Separate model weights from source code.
- Keep model-specific configurations.
- Support replacing model adapters.
- Record model versions and licences.
- Avoid loading multiple unnecessary models.

## FR-10: Performance and Reliability

The application must:

- Measure inference latency.
- Handle model loading failures.
- Avoid blocking the user interface.
- Handle invalid video or landmark sequences.
- Provide clear error messages.
- Recover from failed recognition requests where possible.

---

# 7. Non-Functional Requirements

## 7.1 Performance

- The application should provide responsive camera controls.
- Model inference latency must be measured.
- Frame processing should avoid unnecessary resource consumption.
- GPU acceleration should be used when available and compatible.
- CPU inference should remain possible where practical.

No fixed recognition speed or accuracy is guaranteed before testing.

## 7.2 Reliability

- Invalid requests must be rejected safely.
- Model failures must not crash the entire application.
- Prediction outputs must follow a documented response format.
- Backend and frontend errors must be handled independently.

## 7.3 Maintainability

- Keep frontend, backend, and AI responsibilities separate.
- Use reusable service modules.
- Keep model-specific logic inside model adapters.
- Maintain clear configuration files.
- Avoid duplicate implementations.

## 7.4 Security and Privacy

- Request webcam permission explicitly.
- Do not store raw camera recordings by default.
- Do not upload video without user awareness and consent.
- Keep API keys in backend environment variables.
- Exclude local datasets and model checkpoints from Git.
- Validate uploaded files and API requests.

## 7.5 Accessibility

- Use readable text and clear controls.
- Provide visible camera and recognition status.
- Avoid communicating recognition success when a prediction is uncertain.
- Provide clear error messages.
- Keep important interactions simple.

---

# 8. Technical Requirements

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React | User interface |
| Frontend language | TypeScript | Type-safe application development |
| Build tool | Vite | Frontend development and build |
| Backend | FastAPI | API and application services |
| Backend language | Python 3.11 | Backend and AI integration |
| Deep learning | PyTorch | Compatible model inference |
| Computer vision | OpenCV | Image and video processing |
| Landmark extraction | MediaPipe | Landmark extraction where required |
| Model discovery | Hugging Face Hub | Checkpoint discovery and download |
| Speech | Browser SpeechSynthesis | Initial speech output |
| Testing | Pytest and Vitest | Backend, AI and frontend testing |

Dependencies must be selected based on the actual model's requirements.

---

# 9. AI Model Requirements

The project will prioritise pretrained models over training a model from scratch.

Each candidate model must be checked for:

- ISL language support.
- Recognition task.
- Architecture.
- Checkpoint availability.
- Input format.
- Preprocessing requirements.
- Dataset and label definitions.
- Licence.
- Hardware requirements.
- Independently measured performance.

## 9.1 Initial Model Experiment

The iSign T5 pose-to-text model is an experimental candidate.

Initial testing will use the `t5-small` checkpoint where compatible with the local hardware.

This model expects processed pose sequences. It does not directly accept webcam images.

Before integrating it into the application, the project must verify:

1. The expected pose input format.
2. Landmark ordering and dimensions.
3. Normalization and preprocessing.
4. Compatibility with the available ISL data.
5. Prediction quality on correctly labeled test samples.
6. Local CPU and GPU inference performance.

It must not be considered an approved production model until these checks are completed.

## 9.2 Model Execution Environments

**Primary environment:** Local Windows development with an NVIDIA GPU.

**Fallback environment:** Kaggle GPU notebooks when local memory, dependency, or hardware limitations prevent testing.

The same model adapter and preprocessing implementation should be reusable across environments.

---

# 10. System Architecture Requirements

The application will use a modular monorepo architecture.

Main modules:

1. Frontend application.
2. Backend API.
3. AI preprocessing.
4. Visual feature extraction.
5. Model adapters.
6. Model inference.
7. Prediction postprocessing.
8. Speech output.
9. Testing and evaluation.

The backend must not depend on a specific model implementation. It should communicate through a stable model adapter interface.

---

# 11. API Requirements

Initial backend endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check backend health |
| GET | `/api/v1/ai/health` | Check model service status |
| GET | `/api/v1/ai/models` | List configured models |
| POST | `/api/v1/ai/predict/image` | Static image recognition, if supported |
| POST | `/api/v1/ai/predict/video` | Video sequence recognition, if supported |

A WebSocket endpoint for live recognition may be introduced after the basic inference API works.

Only implement endpoints supported by the selected model and actual pipeline.

---

# 12. Development Phases

| Phase | Description | Expected result |
|---|---|---|
| Phase 0 | Repository and environment setup | Clean working development environment |
| Phase 1 | Backend foundation | FastAPI starts and health endpoint works |
| Phase 2 | AI model experimentation | Standalone local inference works |
| Phase 3 | Model adapter integration | Model is accessible through backend services |
| Phase 4 | AI preprocessing and visual extraction | Compatible inputs can be prepared |
| Phase 5 | Recognition API | Backend returns actual model predictions |
| Phase 6 | Frontend integration | Webcam and prediction interface work together |
| Phase 7 | Speech output | Recognised text can be spoken |
| Phase 8 | Testing and optimisation | Evaluation and performance results |
| Phase 9 | Demonstration preparation | Reliable hackathon prototype |

Phases must be completed and validated progressively.

---

# 13. Acceptance Criteria

The MVP is considered ready for demonstration when:

1. The backend starts successfully.
2. The application can access the webcam.
3. The AI pipeline extracts the required model input.
4. The selected model loads successfully.
5. The model returns actual predictions on supported inputs.
6. Predictions are displayed in the frontend.
7. Uncertain or unsupported inputs are handled.
8. Recognised text can be converted into speech.
9. The application has been tested under realistic conditions.
10. Model limitations are clearly documented.

A successful model import or checkpoint load alone does not satisfy recognition acceptance criteria.

---

# 14. Out of Scope for Initial MVP

The following are excluded from the initial release unless a validated implementation becomes available:

- Complete continuous ISL sentence translation.
- Universal recognition of every ISL sign.
- Recognition of all regional sign variations.
- Automatic ISL grammar correction.
- Guaranteed recognition across every signer and environment.
- Training a large recognition model from scratch.
- Automatic sign-language animation generation.

---

# 15. Project Success Metrics

The project will measure:

| Metric | Measurement |
|---|---|
| Recognition accuracy | Correct predictions on labeled test samples |
| Macro F1-score | Performance across recognition classes |
| Inference latency | Time required for prediction |
| Camera reliability | Successful camera startup and frame capture |
| Pipeline reliability | Successful processing of valid inputs |
| Speech functionality | Successful playback of recognised text |
| Resource usage | GPU memory, RAM and CPU consumption |

All reported results must come from actual project tests.

---

# 16. Future Improvements

After the MVP is validated:

- Expand the supported ISL vocabulary.
- Improve dynamic sign recognition.
- Investigate continuous ISL translation.
- Include facial and non-manual features where supported by suitable data.
- Improve sentence construction.
- Evaluate additional pretrained models.
- Improve performance and deployment capabilities.

---

# 17. Final Product Direction

Signova AI will first establish a working, measurable ISL recognition baseline.

The development priority is:

1. Build the backend foundation.
2. Test the pretrained model locally.
3. Validate its predictions.
4. Integrate it through a reusable model adapter.
5. Connect visual feature extraction.
6. Connect the frontend.
7. Add speech output.
8. Evaluate and optimise the complete system.

**Core principle:** Signova AI must report only the ISL recognition capabilities that have been implemented and verified. It must never present an experimental model as a reliable continuous ISL translator.