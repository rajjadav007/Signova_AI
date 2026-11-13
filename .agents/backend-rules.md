# Backend Development Rules

## Signova AI

**Project:** Signova AI  
**Domain:** Indian Sign Language (ISL) Recognition  
**Backend Framework:** FastAPI  
**Language:** Python 3.11  
**AI Framework:** PyTorch  
**API Style:** REST  
**Development Approach:** Backend First, Modular, Local Development First

---

# 1. Project Backend Overview

Signova AI is an ISL recognition web application with two separate AI engines.

### Engine 1: Individual Words Recognition

- Model: Bi-GRU
- Task: Isolated ISL sign classification
- Vocabulary: 10 selected signs
- Input: Video or compatible landmark data
- Processing: MediaPipe Tasks API and custom feature extraction
- Output: Predicted sign, confidence, top predictions, and recognition status

### Engine 2: Continuous Translation

- Candidate model: iSign T5
- Task: Pose-to-text sequence generation
- Input: Compatible pose or landmark sequences
- Output: Generated text
- Status: Experimental. Verify actual integration and model availability before claiming it is operational.

**Important:** These engines must remain independent. Do not combine their preprocessing, model loading, inference, or output handling.

---

# 2. Core Backend Principles

All backend development must follow these rules:

1. Use Python 3.11.
2. Use FastAPI for backend APIs.
3. Keep API routes separate from business logic.
4. Keep AI model code separate from API code.
5. Use Pydantic for request and response validation.
6. Use environment variables for configuration.
7. Handle errors consistently.
8. Write testable and maintainable code.
9. Avoid unnecessary dependencies.
10. Do not implement features outside the agreed project scope.
11. Preserve the existing working recognition pipeline.
12. Keep the Bi-GRU and iSign T5 services separate.
13. Never return fabricated recognition results.
14. Do not change model thresholds or checkpoints without approval.

---

# 3. Backend Architecture

Follow a modular architecture.

The following is a logical structure, not an instruction to recreate the existing repository.

```text
backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── isolated_recognition.py
│   │   │   └── continuous_prediction.py
│   │   └── router.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   │
│   ├── schemas/
│   │   ├── common.py
│   │   ├── isolated_recognition.py
│   │   └── continuous_prediction.py
│   │
│   ├── services/
│   │   ├── isolated_recognition_service.py
│   │   └── continuous_translation_service.py
│   │
│   └── main.py
│
├── tests/
├── requirements.txt
└── .env.example
```

### Architecture Requirements

- Inspect the existing repository before creating folders or files.
- Reuse existing modules wherever possible.
- Keep route handlers small.
- Keep model inference outside API route handlers.
- Keep preprocessing separate from inference.
- Keep each model's service independent.
- Avoid duplicate services, routes, and configurations.
- Do not move or delete existing modules without approval.

---

# 4. FastAPI Rules

- Use FastAPI for backend HTTP endpoints.
- Use `APIRouter` to organise routes.
- Add request and response schemas.
- Use dependency injection where appropriate.
- Add API documentation through FastAPI OpenAPI support.
- Use suitable HTTP status codes.
- Add clear endpoint descriptions.
- Avoid placing model inference logic directly inside route handlers.
- Keep the application startup and shutdown lifecycle controlled.

The backend must start through the existing application entry point.

---

# 5. API Design Rules

Use the existing API version prefix:

```text
/api/v1
```

### Existing Recognition API Structure

The following endpoints are the reported current API structure. Verify their actual route definitions and schemas before changing them.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/recognize/isolated/video` | Recognize one of the supported isolated signs from video |
| POST | `/api/v1/recognize/isolated/landmarks` | Recognize an isolated sign from compatible landmark data |
| POST | `/api/v1/predict/video` | Send video to the separate continuous pipeline |
| POST | `/api/v1/predict/landmarks` | Send landmark sequences to the continuous pipeline |

Health endpoints must report the actual backend and AI engine status.

### API Requirements

- Preserve existing working API contracts.
- Validate every request.
- Return predictable JSON.
- Keep response formats consistent.
- Document required fields.
- Set reasonable request size limits.
- Handle unsupported input types.
- Avoid exposing internal file paths.
- Do not report a model as available when it has failed to load.
- Do not expose experimental functionality as production-ready.

Do not create new endpoints simply because they appear in an old planning document.

---

# 6. Request and Response Validation

Use Pydantic models for API validation.

Example:

```python
from pydantic import BaseModel, Field

class RecognitionRequest(BaseModel):
    input_type: str = Field(min_length=1)
```

This example is illustrative. Reuse the actual existing request schema where available.

### Validation Rules

- Validate required fields.
- Validate data types.
- Validate supported input formats.
- Validate input dimensions where relevant.
- Validate landmark structures.
- Reject invalid requests with useful errors.
- Never silently accept malformed input.
- Do not trust frontend validation alone.
- Match validation to the actual model requirements.

### Isolated Recognition Validation

For the Bi-GRU engine, validate:

- Video or landmark input.
- Landmark structure.
- Expected landmark ordering.
- Feature dimensions.
- Temporal sequence requirements.
- Valid numeric values.
- Model availability.

The current feature sequence is expected to use 64 time steps and 600 features per time step.

Do not change these dimensions without checking the actual preprocessing and model configuration.

### Continuous Translation Validation

For iSign T5:

- Validate its own expected input format.
- Validate pose sequence dimensions.
- Validate required motion information.
- Validate tokenizer and model requirements where applicable.
- Do not reuse Bi-GRU input validation blindly.

---

# 7. AI Integration Rules

The backend must communicate with AI modules through service and adapter layers.

Recommended flow:

```text
API Route
    |
    v
AI Service
    |
    v
Model Service / Adapter
    |
    v
Model-Specific Preprocessing
    |
    v
Model Inference
    |
    v
Model-Specific Postprocessing
    |
    v
API Response
```

### Individual Words Engine

The isolated recognition service must use the existing Bi-GRU pipeline:

```text
Video / Landmark Input
        |
        v
Input Validation
        |
        v
MediaPipe Landmark Extraction
        |
        v
Spatial Normalization
        |
        v
Temporal Resampling
        |
        v
Velocity Feature Extraction
        |
        v
600D Feature Sequence
        |
        v
Bi-GRU Inference
        |
        v
Confidence Validation
        |
        v
Structured Recognition Response
```

### Continuous Translation Engine

The continuous service must remain independent:

```text
Video / Compatible Pose Input
        |
        v
Input Validation
        |
        v
Model-Specific Pose Processing
        |
        v
iSign T5 Preprocessing
        |
        v
T5 Inference
        |
        v
Text Postprocessing
        |
        v
Structured Translation Response
```

The continuous service must not be marked operational unless its model, preprocessing, checkpoint, and inference pipeline are verified.

### Strict Requirements

- Do not load models inside individual API requests.
- Load models during startup or through controlled lazy loading.
- Reuse loaded models.
- Keep preprocessing outside route handlers.
- Keep inference outside route handlers.
- Validate model availability before prediction.
- Handle model failures.
- Return real model output only.
- Never hardcode example predictions as production results.
- Keep each engine's model loading and configuration separate.

---

# 8. Model Loading and Device Management

The backend must load and manage the AI models independently.

### Bi-GRU Model

- Load the existing isolated recognition checkpoint.
- Use its matching architecture and configuration.
- Validate checkpoint compatibility.
- Preserve the current feature dimensions.
- Preserve the current prediction threshold.
- Reuse the loaded model across requests.

### iSign T5 Model

- Load only the selected and verified checkpoint.
- Verify checkpoint configuration.
- Verify tokenizer availability.
- Verify input compatibility.
- Do not assume it is ready for production use.

### Device Selection

Use automatic device selection:

1. Check whether CUDA is available.
2. Use a compatible GPU when available.
3. Fall back to CPU when necessary.
4. Log the selected device.
5. Handle GPU memory errors.

Example:

```python
import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The development environment includes an NVIDIA GPU with 4 GB VRAM. Do not assume every model can fit into GPU memory.

Avoid loading both engines unnecessarily when they are not being used.

---

# 9. Error Handling

Use consistent error responses.

Recommended structure:

```json
{
  "success": false,
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "The recognition model is currently unavailable."
  }
}
```

Handle these conditions:

- Invalid request.
- Unsupported input format.
- Missing input.
- Model unavailable.
- Model loading failure.
- Preprocessing failure.
- Landmark extraction failure.
- Incorrect feature dimensions.
- Insufficient landmarks.
- Inference failure.
- Invalid model output.
- Request too large.
- CUDA unavailable.
- GPU memory exhausted.
- Internal server error.

### Rules

- Use appropriate HTTP status codes.
- Do not expose stack traces to frontend users.
- Log technical details on the backend.
- Return clear and safe error messages.
- Do not convert errors into successful responses.
- Return an uncertain recognition result when the model rejects a prediction.
- Do not invent a sign when confidence is insufficient.

---

# 10. Configuration and Environment Variables

Keep configuration centralised.

Use environment variables for:

- Application environment.
- API host and port.
- CORS origins.
- Isolated model directory.
- Isolated model checkpoint.
- Continuous model directory.
- Continuous model checkpoint.
- Device selection.
- Logging level.
- Optional database connection.

Example `.env.example`:

```env
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000

CORS_ORIGINS=http://localhost:5173

MODEL_DEVICE=auto

LOG_LEVEL=INFO
```

The exact model paths and configuration names must match the existing backend.

### Configuration Rules

- Never commit `.env`.
- Keep `.env.example` free of secrets.
- Do not hardcode machine-specific paths.
- Validate required configuration at startup.
- Use sensible development defaults.
- Do not expose secrets through API responses or logs.
- Keep each engine's model configuration separate.

---

# 11. CORS Rules

Configure CORS for the frontend origin.

Development frontend:

```text
http://localhost:5173
```

Requirements:

- Allow only required origins.
- Configure required HTTP methods.
- Configure required headers.
- Do not use unrestricted CORS in production.
- Keep production origins in environment configuration.

---

# 12. File and Input Handling

When accepting images or videos:

- Validate file extensions.
- Validate MIME types.
- Check file size.
- Avoid trusting filenames.
- Use safe temporary file handling.
- Reject unsupported formats.
- Clean up temporary files.
- Avoid storing user media unless explicitly required.

For landmark input:

- Validate the structure.
- Validate dimensions.
- Validate numeric values.
- Validate compatibility with the selected engine.

Do not record or persist webcam footage by default.

---

# 13. Camera and Privacy Rules

The browser should handle camera permission and camera capture.

The backend should receive only the input required for recognition.

Requirements:

- Do not request camera access from the backend.
- Do not store camera footage by default.
- Do not log image or video contents.
- Do not expose uploaded media publicly.
- Process temporary files safely.
- Clearly document any future data storage feature.

The frontend should control camera start, stop, capture, and permission handling.

---

# 14. Speech Output Rules

The current application uses frontend speech playback.

Backend responsibilities:

- Return the actual recognized sign or generated text.
- Return a structured response suitable for frontend display.
- Do not add a separate backend TTS service unless explicitly required.

Frontend responsibilities:

- Display recognized text.
- Provide speech playback controls.
- Handle browser speech support.
- Allow users to control playback.

Do not duplicate speech functionality in the backend unnecessarily.

---

# 15. Logging Rules

Use Python's logging module.

Log important backend events:

- Application startup.
- Application shutdown.
- Model loading.
- Model availability.
- Selected inference device.
- Prediction requests.
- Processing failures.
- Unexpected exceptions.

Do not log:

- Secrets.
- Authentication tokens.
- Full user media.
- Sensitive personal information.
- Unnecessary request payloads.

Use appropriate logging levels:

| Level | Usage |
|---|---|
| DEBUG | Development details |
| INFO | Normal application events |
| WARNING | Recoverable problems |
| ERROR | Failed operations |
| CRITICAL | Serious application failures |

---

# 16. Performance Rules

- Load models once and reuse them.
- Avoid unnecessary file operations.
- Validate inputs before expensive processing.
- Avoid unnecessary frame processing.
- Use suitable request limits.
- Monitor CPU and GPU memory.
- Avoid blocking unrelated requests with long-running tasks.
- Use asynchronous endpoints when they provide a real benefit.
- Measure inference latency separately for each engine.

Do not add background workers or queues unless the actual workload requires them.

Do not sacrifice recognition correctness for speed without measuring the impact.

---

# 17. Database Rules

PostgreSQL is optional for the initial MVP.

Basic recognition must not depend on a database.

Do not add a database dependency unless a required feature needs persistent storage.

If a database is introduced:

- Keep database logic separate from API routes.
- Use environment-based connection settings.
- Use parameterised queries or a suitable ORM.
- Validate stored data.
- Handle connection failures.
- Do not store raw camera footage by default.
- Use migrations for schema changes.
- Store recognition history only when the feature is approved and appropriate consent is provided.

---

# 18. Security Rules

- Validate all incoming requests.
- Limit upload sizes.
- Restrict CORS.
- Keep secrets outside source code.
- Avoid exposing internal paths.
- Avoid exposing stack traces.
- Keep dependencies updated.
- Restrict file access.
- Handle temporary files safely.
- Use HTTPS in production.
- Add rate limiting when deployment requirements justify it.

Do not add authentication unless it is part of the approved product scope.

---

# 19. Testing Rules

Use Pytest for backend tests.

Test these areas:

- Application startup.
- Health endpoints.
- API request validation.
- Response schemas.
- Error responses.
- CORS configuration.
- Isolated recognition service.
- Continuous translation service.
- Model loading failures.
- Input preprocessing.
- Landmark extraction.
- Feature dimensions.
- Inference integration.
- File upload validation.
- CPU execution.
- CUDA execution when available.

### Testing Rules

- Use mocked AI services for API tests when actual model weights are unavailable.
- Clearly distinguish mocked tests from real model inference tests.
- Test the Bi-GRU pipeline independently.
- Test the iSign T5 pipeline independently.
- Do not report mock-model tests as real AI validation.
- Do not claim an endpoint works without running it.
- Do not assume historical test results represent the current repository state.

---

# 20. Dependency Management

Keep backend dependencies in the existing requirements file.

Rules:

- Install only required packages.
- Avoid duplicate dependencies.
- Use compatible package versions.
- Do not install packages without a clear reason.
- Keep development-only dependencies separate when practical.
- Verify imports after dependency changes.
- Document important version requirements.
- Do not upgrade PyTorch or MediaPipe without checking compatibility with the existing pipeline.

Do not replace working dependencies just to follow a newer package recommendation.

---

# 21. Development Workflow

Follow this order when making backend changes.

## Phase 1: Existing Backend Inspection

- Inspect the current backend structure.
- Identify existing FastAPI routes.
- Identify existing services.
- Identify model-loading modules.
- Identify preprocessing modules.
- Identify current dependencies.
- Read relevant project documentation.

## Phase 2: API Verification

- Verify current endpoints.
- Verify request schemas.
- Verify response schemas.
- Verify error handling.
- Verify CORS.
- Verify health endpoints.

## Phase 3: AI Service Verification

- Verify the isolated Bi-GRU service.
- Verify its checkpoint loading.
- Verify its preprocessing pipeline.
- Verify its inference response.
- Inspect the continuous iSign T5 service separately.
- Report any incomplete or experimental integration.

## Phase 4: Backend Testing

- Start the FastAPI application.
- Test health endpoints.
- Test input validation.
- Test isolated recognition.
- Test continuous endpoints only with compatible inputs.
- Test error handling.
- Run the relevant Pytest suite.

## Phase 5: Frontend Connection

- Connect the React frontend.
- Use Fetch API or Axios.
- Handle loading states.
- Handle API errors.
- Display actual prediction results.
- Preserve frontend speech playback.
- Keep the two recognition modes separate.

---

# 22. Strict Rules for Antigravity

Before making backend changes:

1. Inspect the existing backend.
2. Read the relevant project documentation.
3. Reuse existing working code.
4. Do not create duplicate services or routes.
5. Do not delete existing files without approval.
6. Do not change AI model logic unnecessarily.
7. Do not put all backend code in `main.py`.
8. Do not place model inference inside API routes.
9. Do not invent API responses.
10. Do not claim an endpoint works without testing it.
11. Do not install unnecessary dependencies.
12. Do not commit secrets or model weights.
13. Keep code, comments, and tests in English.
14. Update relevant documentation when API contracts change.
15. Preserve the separation between Bi-GRU and iSign T5.
16. Do not change model thresholds without approval.
17. Do not replace existing checkpoints without approval.
18. Do not retrain the current model without approval.
19. Do not add PostgreSQL as a mandatory dependency.
20. Do not store raw webcam footage by default.
21. Do not claim continuous translation is operational without verification.
22. Report incomplete functionality honestly.

If a requirement is unclear, inspect the existing implementation and report the uncertainty before making a risky change.

---

# 23. Final Backend Goal

The Signova AI backend must be:

- Modular.
- Secure.
- Testable.
- Maintainable.
- Efficient.
- Compatible with verified AI models.
- Independent from frontend implementation.
- Capable of supporting the isolated Bi-GRU recognition engine.
- Structured to keep continuous iSign T5 translation separate.
- Honest about model availability and limitations.

**The backend must return genuine model results and clearly report failures, uncertainty, or unsupported capabilities.**