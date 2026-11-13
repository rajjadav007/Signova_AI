# API Documentation

## Signova AI

**Project:** Signova AI  
**API Framework:** FastAPI  
**Language:** Python 3.11  
**API Version:** v1  
**Status:** Development Planning

---

# 1. Overview

The Signova AI backend provides APIs for communication between the frontend application and the AI recognition system.

The backend is responsible for:

- Receiving image, video, and landmark inputs.
- Validating incoming requests.
- Preparing model-compatible inputs.
- Running pretrained ISL recognition models.
- Returning recognised text and prediction information.
- Reporting model and service status.
- Handling errors safely.

The backend must remain independent of frontend implementation and individual model architectures.

## Base URL

Local development:

```text
http://localhost:8000
```

API prefix:

```text
/api/v1
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

# 2. Technology Stack

| Component | Technology |
|---|---|
| Backend framework | FastAPI |
| Programming language | Python 3.11 |
| Validation | Pydantic |
| AI framework | PyTorch |
| Image processing | OpenCV |
| Landmark extraction | MediaPipe, if required |
| Model source | Hugging Face or local checkpoint |
| Testing | Pytest |
| API format | JSON |
| File uploads | Multipart form data |

---

# 3. API Architecture

```text
React Frontend
      |
      v
FastAPI Backend
      |
      v
API Routes
      |
      v
Request Validation
      |
      v
AI Service Layer
      |
      v
Model Adapter
      |
      v
Preprocessing
      |
      v
ISL Model Inference
      |
      v
Postprocessing
      |
      v
JSON Response
      |
      v
React Frontend
```

The API routes must not contain model-specific inference logic directly.

---

# 4. API Endpoints

## 4.1 Endpoint Summary

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/health` | GET | Backend health |
| `/api/v1/ai/health` | GET | AI service health |
| `/api/v1/ai/models` | GET | Available model information |
| `/api/v1/ai/predict/image` | POST | Image-based prediction |
| `/api/v1/ai/predict/video` | POST | Video-based prediction |
| `/api/v1/ai/predict/landmarks` | POST | Landmark-based prediction |

Only implement prediction endpoints supported by the selected model. Do not expose an endpoint as functional if its recognition pipeline is not ready.

---

# 5. Health APIs

## 5.1 Backend Health

**Endpoint**

```http
GET /api/v1/health
```

**Purpose**

Checks whether the FastAPI application is running.

**Example response**

```json
{
  "success": true,
  "status": "healthy",
  "service": "signova-backend"
}
```

## 5.2 AI Service Health

**Endpoint**

```http
GET /api/v1/ai/health
```

**Purpose**

Checks the AI service and model loading status.

**Example response**

```json
{
  "success": true,
  "status": "ready",
  "model_loaded": true,
  "model_name": "isign_t5",
  "device": "cuda"
}
```

Possible statuses:

- `ready`: AI service is ready.
- `loading`: Model is loading.
- `unavailable`: AI service is unavailable.
- `error`: An internal error occurred.

The response must reflect the actual runtime state. Do not return `ready` if the model has not loaded successfully.

---

# 6. Model Information API

## 6.1 List Available Models

**Endpoint**

```http
GET /api/v1/ai/models
```

**Purpose**

Returns configured models and their capabilities.

**Example response**

```json
{
  "success": true,
  "models": [
    {
      "id": "isign_t5_small",
      "name": "iSign T5 Small",
      "task": "pose_to_text",
      "status": "experimental",
      "input_type": "pose_sequence"
    }
  ]
}
```

**Model status values**

| Status | Meaning |
|---|---|
| `research` | Under investigation |
| `experimental` | Testing and evaluation |
| `available` | Ready for supported inference |
| `unavailable` | Cannot currently be loaded |
| `rejected` | Not selected for integration |

The iSign T5 candidate must remain experimental until compatible evaluation is completed.

---

# 7. Image Prediction API

## 7.1 Predict from Image

**Endpoint**

```http
POST /api/v1/ai/predict/image
```

**Content-Type**

```text
multipart/form-data
```

**Purpose**

Accepts an image for static sign recognition.

**Request fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Image containing a sign |
| `model_id` | String | No | Selected model identifier |

**Supported formats**

- JPG
- JPEG
- PNG

The backend must validate file type, file size, image dimensions, and image readability.

**Example response**

```json
{
  "success": true,
  "prediction": {
    "text": "HELLO",
    "confidence": 0.94
  },
  "recognition_type": "static_sign",
  "model": {
    "id": "example_static_model",
    "status": "available"
  },
  "processing_time_ms": 85
}
```

The example prediction and confidence are illustrative. They must never be hardcoded.

**Important:** This endpoint should only be activated when an image-compatible ISL model has been verified and integrated. The iSign T5 pose-to-text model cannot accept a raw image directly.

---

# 8. Video Prediction API

## 8.1 Predict from Video

**Endpoint**

```http
POST /api/v1/ai/predict/video
```

**Content-Type**

```text
multipart/form-data
```

**Purpose**

Accepts a video containing one or more signs for sequence-based recognition.

**Request fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Video containing signs |
| `model_id` | String | No | Selected model identifier |

**Supported formats**

The backend should explicitly configure accepted video formats based on its decoding implementation, such as MP4.

**Example response**

```json
{
  "success": true,
  "prediction": {
    "text": "HELLO",
    "segments": [
      {
        "text": "HELLO",
        "start_time": 0.0,
        "end_time": 1.5
      }
    ]
  },
  "recognition_type": "dynamic_sign",
  "model": {
    "id": "example_dynamic_model",
    "status": "available"
  },
  "processing_time_ms": 1250
}
```

Segment timestamps are optional and should only be returned when the model or processing pipeline can provide them.

**Important:** A video upload endpoint does not automatically provide continuous ISL translation. That requires a suitable sequence recognition and language-processing pipeline.

---

# 9. Landmark Prediction API

## 9.1 Predict from Landmarks

**Endpoint**

```http
POST /api/v1/ai/predict/landmarks
```

**Content-Type**

```text
application/json
```

**Purpose**

Accepts preprocessed landmark sequences for models that use pose or landmark inputs.

This endpoint is particularly relevant to the experimental iSign T5 pose-to-text candidate.

**Example request**

```json
{
  "model_id": "isign_t5_small",
  "sequence": {
    "pose_landmarks": [],
    "left_hand_landmarks": [],
    "right_hand_landmarks": []
  }
}
```

The empty arrays above are placeholders, not valid model inputs.

**Input validation**

The backend must validate:

- Required landmark groups.
- Landmark dimensions.
- Sequence length.
- Numeric values.
- Missing or invalid landmarks.
- Expected coordinate representation.
- Model-specific input format.

**Example response**

```json
{
  "success": true,
  "prediction": {
    "text": "HELLO"
  },
  "recognition_type": "pose_to_text",
  "model": {
    "id": "isign_t5_small",
    "status": "experimental"
  },
  "processing_time_ms": 450
}
```

The output is illustrative. The actual response must contain genuine model inference results.

---

# 10. Standard Response Format

All API responses should follow a consistent structure.

## 10.1 Success Response

```json
{
  "success": true,
  "data": {},
  "message": "Request completed successfully"
}
```

Prediction endpoints may use a structured `prediction` field instead of a generic `data` field.

## 10.2 Error Response

```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "The uploaded input is invalid"
  }
}
```

Do not expose internal file paths, API keys, stack traces, or sensitive implementation details.

---

# 11. HTTP Status Codes

| Status code | Meaning |
|---|---|
| `200` | Request completed |
| `400` | Invalid request |
| `401` | Authentication required, if authentication is added |
| `404` | Endpoint or resource not found |
| `413` | Uploaded file too large |
| `415` | Unsupported media type |
| `422` | Request validation failed |
| `429` | Rate limit exceeded, if configured |
| `500` | Internal server error |
| `503` | Model or AI service unavailable |

---

# 12. Error Codes

| Error code | Description |
|---|---|
| `INVALID_INPUT` | Invalid request data |
| `INVALID_IMAGE` | Image cannot be processed |
| `INVALID_VIDEO` | Video cannot be processed |
| `INVALID_LANDMARKS` | Landmark data does not match expected format |
| `UNSUPPORTED_FORMAT` | Unsupported file format |
| `FILE_TOO_LARGE` | File exceeds configured size |
| `MODEL_NOT_FOUND` | Requested model does not exist |
| `MODEL_NOT_LOADED` | Model is not ready |
| `MODEL_LOAD_FAILED` | Model loading failed |
| `INFERENCE_FAILED` | Model inference failed |
| `GPU_MEMORY_ERROR` | GPU memory is insufficient |
| `PREDICTION_UNCERTAIN` | Prediction is not reliable enough |
| `INTERNAL_ERROR` | Unexpected backend error |

---

# 13. Frontend Integration

The React frontend communicates with FastAPI using HTTP requests.

## 13.1 API Base URL

Use an environment variable:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Do not hardcode the backend URL across frontend components.

## 13.2 Example API Service

```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export async function checkBackendHealth() {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/health`
  );

  if (!response.ok) {
    throw new Error("Backend health check failed");
  }

  return response.json();
}
```

## 13.3 Example Image Request

```typescript
export async function predictImage(
  file: File,
  modelId?: string
) {
  const formData = new FormData();

  formData.append("file", file);

  if (modelId) {
    formData.append("model_id", modelId);
  }

  const response = await fetch(
    `${API_BASE_URL}/api/v1/ai/predict/image`,
    {
      method: "POST",
      body: formData
    }
  );

  if (!response.ok) {
    throw new Error("Image prediction failed");
  }

  return response.json();
}
```

Do not manually set the multipart `Content-Type` header. The browser generates the required boundary.

---

# 14. CORS Configuration

During local development, FastAPI must allow requests from the configured frontend origin.

Example:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For deployment, replace local origins with the actual frontend domain. Avoid unrestricted origins in production.

---

# 15. FastAPI Project Structure

Suggested backend structure:

```text
backend/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── health.py
│   │       └── ai.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── model_manager.py
│   │
│   ├── schemas/
│   │   ├── common.py
│   │   └── prediction.py
│   │
│   ├── services/
│   │   └── ai_service.py
│   │
│   └── main.py
│
├── tests/
│   ├── test_health.py
│   └── test_ai_api.py
│
└── requirements.txt
```

Create files only when implementing the corresponding functionality.

---

# 16. API Performance Requirements

The backend should:

- Load models once during startup.
- Avoid loading checkpoints for every request.
- Validate input before inference.
- Use GPU when available and compatible.
- Support CPU fallback where practical.
- Measure inference latency.
- Limit uploaded file size.
- Avoid unnecessary frame processing.
- Return clear errors when inference fails.

Performance targets must be measured on the actual development hardware before being documented as achieved.

---

# 17. Security and Privacy

- Validate uploaded files.
- Restrict file size and accepted formats.
- Keep API keys in environment variables.
- Never expose model checkpoints through public API routes.
- Avoid storing webcam footage by default.
- Do not log raw video or private image content.
- Restrict CORS to approved frontend origins.
- Handle unexpected exceptions safely.

---

# 18. Testing Strategy

Use Pytest for backend API testing.

Required tests:

| Test | Expected result |
|---|---|
| Backend health | Returns healthy status |
| AI health | Reports actual model state |
| Model listing | Returns configured model metadata |
| Invalid image | Returns validation error |
| Invalid video | Returns validation error |
| Invalid landmarks | Returns validation error |
| Missing model | Returns model error |
| Model unavailable | Returns service unavailable |
| Prediction success | Returns actual model output |
| CORS | Allows configured frontend origin |

Use mocked model adapters for API unit tests. Test real model inference separately with compatible sample data.

Do not treat mocked predictions as model evaluation results.

---

# 19. Implementation Order

Implement the API in this order:

1. Create FastAPI application.
2. Add backend health endpoint.
3. Add AI service health endpoint.
4. Create model manager.
5. Create model adapter interface.
6. Implement standalone model inference.
7. Add landmark validation schemas.
8. Add landmark prediction endpoint.
9. Add image and video endpoints only when supported models are available.
10. Add API tests.
11. Connect frontend services.
12. Test the complete application.

The first working API milestone is the backend health endpoint, followed by standalone model inference and the landmark prediction endpoint if the selected model input is compatible.

---

# 20. Development Rules

1. Keep API routes separate from AI inference code.
2. Use Pydantic for request validation.
3. Return consistent JSON responses.
4. Never fabricate predictions.
5. Do not expose unsupported model capabilities.
6. Do not claim continuous ISL translation without evaluation.
7. Keep model weights outside Git.
8. Load models once and reuse them.
9. Keep frontend and backend URLs configurable.
10. Test endpoints before frontend integration.
11. Document actual endpoint behaviour.
12. Update this document when API contracts change.

---

# 21. Final API Direction

The first backend milestone for Signova AI is a stable FastAPI application with health checks and a modular AI service.

After the selected pretrained model has passed compatible input and output testing, expose its supported prediction task through the API.

The API must represent the real capabilities of the integrated model, not planned or assumed functionality.

---

# 22. FE-08: Sarvam AI Translation & Bulbul Speech Endpoints

Implemented for milestone FE-08 to complete the forward communication direction:
**Deaf Person (ISL Recognition) &rarr; Editable Message &rarr; Sarvam Translation &rarr; Bulbul Spoken Audio**.

### Environment Configuration

| Variable | Type | Default | Description |
|---|---|---|---|
| `SARVAM_API_KEY` | string | `""` | Primary subscription key for Sarvam AI cloud services. Secret: kept strictly on backend. |
| `SARVAM_BASE_URL` | string | `https://api.sarvam.ai` | Base API gateway URL. |
| `SARVAM_TIMEOUT_SECONDS` | float | `15.0` | Upstream HTTP request timeout. |

---

### Endpoint: `POST /api/v1/translate`

Translates an edited message into the hearing person's selected Indian language.

**Request Schema (`TranslationRequest`):**
```json
{
  "input": "Hello water please",
  "target_language_code": "hi-IN",
  "source_language_code": "en-IN",
  "model": "sarvam-translate:v1"
}
```

**Response Schema (`TranslationResponse`):**
```json
{
  "success": true,
  "translated_text": "नमस्ते, कृपया पानी दीजिए",
  "source_language_code": "en-IN",
  "target_language_code": "hi-IN",
  "processing_time_ms": 112.5
}
```

**Error Responses:**
- `400 Bad Request`: Empty input or unsupported target language.
- `503 Service Unavailable`: `SARVAM_API_KEY` not configured in `.env`.
- `504 Gateway Timeout`: Sarvam AI upstream timeout.
- `502 Bad Gateway`: Upstream provider failure (sanitized; no credentials leaked).

---

### Endpoint: `GET /api/v1/translate/languages`

Returns the supported target language dictionary (code to display name).

**Response (`200 OK`):**
```json
{
  "en-IN": "English",
  "hi-IN": "Hindi",
  "gu-IN": "Gujarati",
  "kn-IN": "Kannada",
  "ta-IN": "Tamil",
  "bn-IN": "Bengali",
  "te-IN": "Telugu",
  "ml-IN": "Malayalam",
  "mr-IN": "Marathi",
  "pa-IN": "Punjabi",
  "od-IN": "Odia"
}
```

---

### Endpoint: `POST /api/v1/speech/synthesize`

Converts translated or edited text into spoken audio via Sarvam Bulbul TTS.

**Request Schema (`TTSRequest`):**
```json
{
  "text": "नमस्ते, कृपया पानी दीजिए",
  "target_language_code": "hi-IN",
  "speaker": "meera",
  "pace": 1.0,
  "model": "bulbul:v1"
}
```

**Response Schema (`TTSResponse`):**
```json
{
  "success": true,
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=",
  "audio_format": "wav",
  "target_language_code": "hi-IN",
  "speaker": "meera",
  "processing_time_ms": 240.2
}
```

---

### Endpoint: `GET /api/v1/speech/speakers`

Returns the catalog of available voice speakers for Bulbul TTS.

**Response (`200 OK`):**
```json
["amol", "amartya", "arvind", "maitreyi", "meera", "pavithra"]
```

---

### Supported Language & Voice Matrix

| Language | Code | Translation Supported | Bulbul TTS Supported | Default Speaker |
|---|---|---|---|---|
| English | `en-IN` | Yes | Yes | `meera` |
| Hindi | `hi-IN` | Yes | Yes | `meera` |
| Gujarati | `gu-IN` | Yes | Yes | `meera` |
| Kannada | `kn-IN` | Yes | Yes | `meera` |
| Tamil | `ta-IN` | Yes | Yes | `pavithra` / `meera` |
| Bengali | `bn-IN` | Yes | Yes | `meera` |
| Telugu | `te-IN` | Yes | Yes | `meera` |
| Malayalam | `ml-IN` | Yes | Yes | `meera` |
| Marathi | `mr-IN` | Yes | Yes | `meera` |
| Punjabi | `pa-IN` | Yes | Yes | `meera` |
| Odia | `od-IN` | Yes | Yes | `meera` |

---

### Known Limitations

1. **Isolated Word Grammar:** Recognition outputs individual gloss words (e.g. "Food Water"). The message builder permits manual grammar refinement before translation so that Sarvam produces coherent natural-language sentences.
2. **API Key Dependency:** Cloud translation and Bulbul speech synthesis require an active Sarvam subscription key. When absent, the UI displays an actionable setup reminder.
3. **Audio Playback Autoplay:** Web browsers enforce user-activation policies for audio playback. Audio playback is triggered by explicit user clicks ("Generate Speech" or "Play Audio").
