# Testing Strategy

## Signova AI

**Project:** Signova AI  
**Version:** 1.0  
**Frontend:** React + JavaScript + Vite  
**Backend:** FastAPI + Python 3.11  
**AI Framework:** PyTorch  
**Testing Approach:** Unit, Integration, Model Evaluation, and End-to-End Testing

---

# 1. Purpose

This document defines the testing strategy for Signova AI.

The goal is to verify that each application component works correctly and that the complete sign recognition workflow produces genuine and reliable results.

Testing must cover:

- Frontend components.
- Backend APIs.
- AI model loading.
- Model preprocessing.
- Model inference.
- Prediction output.
- Camera input.
- Text-to-speech.
- Complete application workflow.

Do not mark a feature as completed until its required tests pass.

---

# 2. Testing Objectives

1. Verify that the backend starts and responds correctly.
2. Verify that frontend components behave as expected.
3. Validate input before model inference.
4. Verify pretrained model compatibility.
5. Evaluate recognition quality using compatible ISL samples.
6. Test frontend-backend communication.
7. Measure inference performance.
8. Handle invalid inputs and failures safely.
9. Verify text-to-speech functionality.
10. Confirm that the complete MVP works.

---

# 3. Testing Levels

| Testing Level | Purpose |
|---|---|
| Unit Testing | Test individual functions and modules |
| Integration Testing | Test communication between components |
| API Testing | Verify backend endpoints |
| Model Testing | Verify model loading and inference |
| Evaluation Testing | Measure recognition performance |
| UI Testing | Verify frontend interactions |
| End-to-End Testing | Test the complete application workflow |
| Performance Testing | Measure latency and resource usage |

---

# 4. Frontend Testing

**Technology:** Vitest and React Testing Library

### Components to Test

| Component | Test Cases |
|---|---|
| Camera | Permission, preview, start and stop |
| Recognition | Start and stop recognition |
| Results | Display prediction text |
| Speech | Play and stop speech |
| API Service | Request and response handling |
| Error UI | Display useful error messages |
| Loading UI | Show and hide loading states |

### Frontend Test Cases

- [ ] Application renders successfully.
- [ ] Camera permission is requested.
- [ ] Permission denial is handled.
- [ ] Camera preview starts when supported.
- [ ] Camera can be stopped.
- [ ] API requests use the correct endpoint.
- [ ] API failures display error messages.
- [ ] Prediction text is displayed.
- [ ] Empty predictions are handled.
- [ ] Speech playback works where supported.
- [ ] Speech can be stopped.
- [ ] UI works at different screen sizes.

### Important Rule

Mock browser APIs such as camera access and speech synthesis in unit tests.

Use a real browser for final camera and speech integration tests.

---

# 5. Backend Testing

**Technology:** Pytest and FastAPI TestClient

### Backend Modules to Test

- Application initialization.
- API routes.
- Request schemas.
- Response schemas.
- Configuration.
- AI service layer.
- Model manager.
- Error handling.

### Backend Test Cases

- [ ] FastAPI application starts.
- [ ] Health endpoint returns a valid response.
- [ ] AI health endpoint reports model status.
- [ ] Model information endpoint returns expected data.
- [ ] Invalid requests return validation errors.
- [ ] Unsupported file types are rejected.
- [ ] Oversized uploads are rejected.
- [ ] Missing model checkpoints are handled.
- [ ] Model loading errors are handled.
- [ ] Inference errors are handled.
- [ ] Response structures match API documentation.

### Example Test

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

The expected response must match the actual implemented health endpoint contract.

---

# 6. AI Model Testing

**Framework:** PyTorch

The selected pretrained model must be tested independently before integration with FastAPI.

### Initial Experimental Candidate

`manavdhamecha77/iSign-t5-pose-to-text`

Initial checkpoint: `t5-small`

### Model Loading Tests

- [ ] Model repository is accessible.
- [ ] Required checkpoint files are available.
- [ ] Tokenizer loads successfully.
- [ ] Model architecture loads successfully.
- [ ] Model configuration is correct.
- [ ] CPU inference is tested.
- [ ] CUDA inference is tested if available.
- [ ] Missing checkpoint errors are handled.

### Input Validation Tests

- [ ] Expected input keys are present.
- [ ] Input dimensions are correct.
- [ ] Landmark ordering is correct.
- [ ] Data types are valid.
- [ ] Missing landmarks are handled.
- [ ] Invalid numerical values are handled.
- [ ] Sequence length restrictions are respected.
- [ ] Preprocessing matches the original model implementation.

### Inference Tests

- [ ] Compatible sample produces an output.
- [ ] Output can be decoded.
- [ ] Empty output is handled.
- [ ] Repeated inference works.
- [ ] Invalid input is rejected.
- [ ] Inference exceptions are handled.

### Important Rule

Successful model loading does not prove recognition accuracy.

The model must be evaluated on suitable labeled ISL samples before being selected for the application.

---

# 7. Model Evaluation Strategy

Model evaluation measures the quality of predictions against ground-truth annotations.

### Dataset Requirements

Use compatible labeled ISL samples.

Verify:

- Dataset language.
- Sign vocabulary.
- Annotation format.
- Model input compatibility.
- Preprocessing compatibility.
- Dataset licence.

Do not treat ASL samples as ISL evaluation data.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct positive predictions |
| Recall | Coverage of actual positive samples |
| F1-score | Balance between precision and recall |
| Macro F1 | Average performance across classes |
| Confusion Matrix | Identify classification errors |
| BLEU | Sequence-generation evaluation, where appropriate |
| ROUGE | Text-generation comparison, where appropriate |

Use metrics appropriate to the model's task.

For sequence translation, also inspect actual predictions and references. A single metric is not enough to establish translation quality.

### Evaluation Record

| Field | Value |
|---|---|
| Model | Not selected |
| Checkpoint | Not selected |
| Dataset | Not selected |
| Test Samples | Not evaluated |
| Accuracy | Not evaluated |
| F1-score | Not evaluated |
| Latency | Not measured |
| Final Decision | Pending |

---

# 8. Preprocessing Testing

Preprocessing is critical because the model expects a specific input format.

### Test Cases

- [ ] Frame extraction produces valid frames.
- [ ] Landmark extraction works when required.
- [ ] Landmark order matches model requirements.
- [ ] Coordinate normalisation is correct.
- [ ] Expected dimensions are produced.
- [ ] Sequence preparation is correct.
- [ ] Invalid input is rejected.
- [ ] Preprocessing output matches reference examples.

For the iSign T5 candidate, test pose, left-hand, and right-hand landmark preparation against its original inference implementation.

Do not assume that generic MediaPipe landmarks are directly compatible.

---

# 9. API Integration Testing

Test communication between the frontend and FastAPI backend.

### Planned Endpoints

| Endpoint | Method | Testing |
|---|---|---|
| `/api/v1/health` | GET | Health response |
| `/api/v1/ai/health` | GET | AI service status |
| `/api/v1/ai/models` | GET | Model information |
| `/api/v1/ai/predict/image` | POST | If image inference is supported |
| `/api/v1/ai/predict/video` | POST | If video inference is supported |
| `/api/v1/ai/predict/landmarks` | POST | If landmark inference is supported |

### Integration Test Cases

- [ ] Frontend can reach backend.
- [ ] CORS configuration works.
- [ ] Valid requests are accepted.
- [ ] Invalid requests are rejected.
- [ ] Model predictions reach the frontend.
- [ ] API errors are displayed correctly.
- [ ] Request timeouts are handled.
- [ ] Model unavailable responses are handled.

Only test prediction endpoints after their model pipelines are implemented.

---

# 10. Camera Testing

Camera testing must use a real browser and supported camera device.

### Test Cases

- [ ] Camera permission prompt appears.
- [ ] Camera starts after permission is granted.
- [ ] Live preview displays correctly.
- [ ] Permission denial is handled.
- [ ] Camera unavailable errors are handled.
- [ ] Camera can be stopped.
- [ ] Repeated start and stop operations work.
- [ ] Camera resources are released when no longer needed.

Test in supported browsers over localhost or HTTPS.

---

# 11. Text-to-Speech Testing

**Technology:** Browser SpeechSynthesis API

### Test Cases

- [ ] Speech synthesis is available.
- [ ] Recognised text can be spoken.
- [ ] Speech can be stopped.
- [ ] Empty text is handled.
- [ ] Unsupported browser functionality is handled.
- [ ] Available voice selection works where supported.
- [ ] Repeated playback works.

Speech output must use the actual recognised text.

---

# 12. End-to-End Testing

End-to-end testing verifies the complete supported application workflow.

### Main Workflow

```text
Open Application
       |
       v
Grant Camera Permission
       |
       v
Start Camera
       |
       v
Capture Supported Input
       |
       v
Send Input to Backend
       |
       v
Validate and Preprocess
       |
       v
Run Pretrained Model
       |
       v
Receive Prediction
       |
       v
Display Recognised Text
       |
       v
Play Speech Output
```

### End-to-End Checklist

- [ ] Application loads successfully.
- [ ] Camera permission works.
- [ ] Camera preview works.
- [ ] Supported input is captured.
- [ ] Backend receives the request.
- [ ] Preprocessing succeeds.
- [ ] Model inference succeeds.
- [ ] Genuine prediction is returned.
- [ ] Recognised text appears.
- [ ] Speech output works.
- [ ] Errors are handled correctly.

---

# 13. Performance Testing

Performance testing measures the application's resource usage and response time.

### Metrics

| Metric | Measurement |
|---|---|
| Backend startup time | Seconds |
| Model loading time | Seconds |
| Preprocessing time | Milliseconds |
| Inference latency | Milliseconds |
| API response time | Milliseconds |
| End-to-end latency | Milliseconds |
| CPU usage | Percentage |
| GPU memory usage | MB or GB |
| GPU utilisation | Percentage |

### Performance Test Cases

- [ ] Measure cold model loading time.
- [ ] Measure repeated inference latency.
- [ ] Compare CPU and GPU inference.
- [ ] Monitor GPU memory.
- [ ] Test multiple input sizes.
- [ ] Identify preprocessing bottlenecks.
- [ ] Measure complete frontend-to-backend latency.

Record the hardware and test conditions with every result.

Do not claim real-time performance without measuring the complete recognition workflow.

---

# 14. Security and Privacy Testing

- [ ] Invalid file uploads are rejected.
- [ ] Upload size limits are enforced.
- [ ] API inputs are validated.
- [ ] CORS allows only configured origins.
- [ ] Secrets are not exposed in frontend code.
- [ ] Debug traces are not returned in production.
- [ ] Camera access requires permission.
- [ ] Temporary media is not retained unnecessarily.
- [ ] Sensitive information is not written to logs.

See `SECURITY.md` for the complete security policy.

---

# 15. Test Data Management

Use separate test data for different purposes.

| Data Type | Purpose |
|---|---|
| Unit test fixtures | Test individual functions |
| Compatible pose samples | Test pose preprocessing |
| Labeled ISL samples | Evaluate recognition quality |
| Invalid inputs | Test validation |
| Video samples | Test frame processing |
| Browser mocks | Test frontend behavior |

Rules:

- Use legally available datasets.
- Check dataset licences.
- Keep large datasets outside Git.
- Avoid committing private recordings.
- Keep test samples reproducible.
- Record the dataset version used for evaluation.

---

# 16. Test Directory Structure

Suggested structure:

```text
backend/
└── tests/
    ├── test_health.py
    ├── test_api.py
    ├── test_schemas.py
    └── test_ai_service.py

frontend/
└── src/
    └── __tests__/
        ├── camera.test.jsx
        ├── recognition.test.jsx
        ├── speech.test.jsx
        └── api.test.js

ai/
└── tests/
    ├── test_preprocessing.py
    ├── test_model_loading.py
    ├── test_inference.py
    └── test_postprocessing.py
```

Create only the test files required by the actual implementation.

---

# 17. Test Execution

### Backend

```bash
pytest
```

### Frontend

```bash
npm run test
```

The frontend test script must be configured in `package.json`.

### Frontend Production Build

```bash
npm run build
```

### Backend API

Run the FastAPI server and test the available endpoints through the API documentation or an HTTP client.

### AI Model

Run the model's standalone inference tests before testing it through FastAPI.

---

# 18. Bug Tracking

Record discovered issues with:

| Field | Description |
|---|---|
| Bug ID | Unique issue identifier |
| Title | Short issue description |
| Component | Frontend, backend, or AI |
| Priority | Critical, High, Medium, Low |
| Steps to Reproduce | How to reproduce |
| Expected Result | Expected behavior |
| Actual Result | Observed behavior |
| Status | Open, In Progress, Fixed, Verified |

A bug is fixed only after the relevant test passes.

---

# 19. Testing Completion Criteria

The MVP testing process is complete when:

- [ ] Backend tests pass.
- [ ] Frontend tests pass.
- [ ] AI model loading is verified.
- [ ] Preprocessing tests pass.
- [ ] Model evaluation is documented.
- [ ] API integration tests pass.
- [ ] Camera tests pass.
- [ ] Speech tests pass.
- [ ] End-to-end workflow passes.
- [ ] Critical bugs are resolved.
- [ ] Performance results are recorded.
- [ ] Known limitations are documented.

---

# 20. Testing Principles

1. Test each module independently.
2. Test model compatibility before integration.
3. Use compatible ISL samples for model evaluation.
4. Never fabricate prediction results.
5. Keep failed experiments documented.
6. Test error cases, not only successful cases.
7. Measure performance instead of assuming it.
8. Keep test data and model weights outside Git.
9. Update tests when application behavior changes.
10. Mark features complete only after verification.

**Final Goal:** Ensure that Signova AI produces genuine, tested recognition results through a reliable and accessible application workflow.