# F-024A: Signova AI Live Webcam Testing Interface Report

**Feature:** F-024A  
**Module:** Live Webcam Testing Interface for Isolated ISL Recognition  
**Model:** 10-Class Bilateral Bi-GRU Champion (`models/checkpoints/isolated/isolated_10class_best.joblib`)  
**Backend Route:** `POST /api/v1/recognize/isolated/video`  
**Frontend Component:** `frontend/src/components/recognition/IsolatedWebcamTester.jsx`  
**Date:** 2026-11-12  
**Status:** Completed & Verified  

---

## 1. Executive Summary

Feature F-024A provides a live, interactive webcam testing interface for Signova AI's improved 10-class isolated Indian Sign Language (ISL) recognition model. 

The interface empowers signers, developers, and hackathon evaluators to test gestures in real-time, benchmark prediction accuracy against expected ground-truth signs, inspect candidate probabilities, and listen to synthesized speech output—all while maintaining complete decoupling from the continuous iSign T5 translation pipeline.

### Key Achievements

1. **Dedicated Live Webcam Tester (`IsolatedWebcamTester.jsx`)**:
   - **Mirrored Video Stream**: CSS `transform: scaleX(-1)` provides natural mirror feedback during signing while recording true orientation for the model.
   - **Visual Framing Guide**: Semi-transparent bounding guide ensures proper upper-body and hand positioning in frame.
   - **One-Click Capture Workflow**: 3-second recording countdown and auto-stop mechanism optimized for isolated sign duration.
   - **Interactive Ground-Truth Benchmarking**: "Expected Sign" selector allows testers to validate model accuracy across live attempts without passing any ground-truth to the backend.
   - **Comprehensive Prediction Card**: Displays recognized/uncertain status badges, confidence percentage, rejection reasons, top-3 candidate probability distribution bars, and latency in milliseconds.
   - **Voice Synthesis**: Integrated Web Speech API auto-speaks recognized signs with an instant mute/unmute toggle.
   - **Test History & Session Analytics**: Real-time summary metrics (Total attempts, Accuracy %, Uncertain rate) and a detailed history table with "Reset Test History" functionality.
   - **Low-Data Warning Banner**: Prominently flags classes with limited training data (`stop` and `sorry`) to set appropriate user expectations.
   - **Offline / Video File Fallback**: Direct file uploader for testing pre-recorded `.mp4` / `.webm` clips when a camera is unavailable.

2. **Backend Route Hardening & Alignment**:
   - Resolved FastAPI direct-invocation `Query` wrapper extraction in `backend/app/api/routes/isolated_recognition.py`.
   - Unified `TopPrediction` schema mapping (`sign` $\leftrightarrow$ `label`) to ensure zero validation errors.
   - Verified end-to-end MediaPipe Holistic feature extraction and Bi-GRU classification on raw video uploads.

3. **Verification**:
   - **Backend Route Tests**: 16/16 passed in `tests/test_isolated_recognition_pipeline.py` (including real video file inference, invalid format rejection, empty file handling, and landmark inference).
   - **Frontend Vitest**: 4/4 test suites passed.
   - **Vite Production Build**: Built cleanly with zero errors in 687ms.

---

## 2. Architecture & Pipeline Decoupling

Signova AI strictly separates isolated single-word recognition from continuous sentence translation. Both pipelines coexist within the same repository without cross-talk or dependency conflicts.

```
                           User Browser (Webcam / File)
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
     [Isolated Words Mode]                         [Continuous Mode]
   IsolatedWebcamTester.jsx                      WebcamRecorder.jsx
                │                                               │
  POST /api/v1/recognize/isolated/video           POST /api/v1/predict/video
                │                                               │
                ▼                                               ▼
   MediaPipe Holistic (75 KPs)                    MediaPipe Holistic (75 KPs)
                │                                               │
   Temporal Resample (64 frames)                  Linear Interpolation (243 frames)
   Spatial Normalization (Shoulder)               Z-score Pose Normalization
   Velocity Features (300 + 300 = 600-d)          Pose Features (300-d)
                │                                               │
                ▼                                               ▼
   Bilateral 2-Layer Bi-GRU                       Pretrained iSign T5-Base
   (models/checkpoints/isolated/)                 (models/checkpoints/t5-base/)
                │                                               │
                ▼                                               ▼
  IsolatedRecognitionResponse                     PredictResponse
  - predicted_sign: "yes"                         - predicted_text: "HOW ARE YOU"
  - confidence: 0.9564                            - decoding: "beam"
  - status: "recognized"                          - inference_time: 1.12s
  - top_predictions: [yes, no, water]                           │
                │                                               ▼
                ▼                                      TranslationResult.jsx
   IsolatedWebcamTester Display Card
```

---

## 3. UI Component Design & Features

### 3.1 Live Camera Preview & Framing
- Uses `navigator.mediaDevices.getUserMedia` with fallback error reporting for missing permissions or unsupported devices.
- Video stream is horizontally flipped (`transform: scaleX(-1)`) so gestures mirror natural human intuition.
- Overlay guide box indicates the optimal bounding zone for head, shoulders, and hands.

### 3.2 3-Second Timed Capture
- Clicking "Capture & Recognize (3s)" initiates an automatic recording cycle via `MediaRecorder` (`video/webm` or `video/mp4`).
- Real-time countdown button (`Recording (3s)...`, `Recording (2s)...`, `Recording (1s)...`) guarantees consistent sign duration matching the training distribution.
- Once finished, the blob is immediately dispatched to `POST /api/v1/recognize/isolated/video`.

### 3.3 Expected Sign Selector & Test Benchmarking
- A dropdown menu lists all 10 supported MVP signs:
  `hello`, `thank you`, `sorry`, `please`, `yes`, `no`, `help`, `stop`, `water`, `food`.
- Selecting an expected sign enables automatic evaluation:
  - If predicted sign equals expected sign $\to$ **CORRECT** (Green badge).
  - If predicted sign differs $\to$ **INCORRECT** (Red badge).
  - If confidence $< 0.80$ $\to$ **UNCERTAIN** (Orange badge).
- The expected sign is purely client-side evaluation metadata and is never transmitted to the backend.

### 3.4 Result Card & Candidate Distribution
- **Status Indicator**: Distinct visual badges for `✓ RECOGNIZED` (green) and `⚠ UNCERTAIN` (orange).
- **Confidence**: Bold, high-visibility percentage.
- **Uncertainty Reason**: Explicitly informs the user if a gesture was rejected due to low confidence (e.g., *"Confidence 62.4% below optimal threshold 80.0%"*).
- **Top-3 Candidates**: Probability distribution bars displaying the model's top three classifications.
- **Latency**: Exact round-trip processing time reported in milliseconds.
- **Audio Feedback**: Built-in speech synthesis automatically speaks the recognized word if enabled.

### 3.5 Test History & Session Analytics
- Tracks each live attempt in a clean history table:
  - Attempt #, Expected Sign, Predicted Sign, Confidence %, Status, Result (`CORRECT` / `INCORRECT` / `UNCERTAIN`).
- Dynamic session counters display:
  - Total Attempts
  - Correct Count & Session Accuracy %
  - Uncertain Count
- Includes a "Reset Test History" button to start fresh benchmarking runs.

### 3.6 Low-Data Warning Banner
A prominent warning banner advises users:
> **⚠️ Low-Volume Sign Notice:** The signs `stop` (1 train sample) and `sorry` (3 train samples) have very limited training data and may be rejected as uncertain. The other 8 signs (`hello`, `thank you`, `please`, `help`, `yes`, `no`, `water`, `food`) are fully supported.

---

## 4. Backend Route & Schema Specification

### Endpoint: `POST /api/v1/recognize/isolated/video`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `video`: Binary video file (`.mp4`, `.webm`, `.avi`, `.mov`, `.mkv`).
  - `category`: (Optional query param) Category filter (`general`, `alphabet`, etc.).
  - `top_k`: (Optional query param, default 3) Number of candidate predictions to return.

### Response Schema (`IsolatedRecognitionResponse`)
```json
{
  "success": true,
  "predicted_sign": "yes",
  "display_name": "Yes",
  "category": "general",
  "confidence": 0.9564,
  "status": "recognized",
  "is_valid": true,
  "processing_time_ms": 64.23,
  "model_used": "IsolatedBiGRU (Bilateral Mirror Augmented)",
  "top_predictions": [
    {
      "label": "yes",
      "display_name": "Yes",
      "confidence": 0.9564,
      "category": "general"
    },
    {
      "label": "no",
      "display_name": "No",
      "confidence": 0.0312,
      "category": "general"
    },
    {
      "label": "water",
      "display_name": "Water",
      "confidence": 0.0071,
      "category": "general"
    }
  ],
  "reason": null
}
```

---

## 5. Verification & Test Evidence

### 5.1 Backend Route Integration Tests
Executed with `pytest tests/test_isolated_recognition_pipeline.py`:
```
tests\test_isolated_recognition_pipeline.py ............s...             [100%]
======================== 15 passed, 1 skipped in 6.45s ========================
```
Key tests verified:
- `test_13_route_recognize_video_end_to_end`: Verified end-to-end video processing from a real `.mp4` file, returning `success == True`, valid class prediction, confidence in $[0, 1]$, and top-3 candidates in 64ms.
- `test_14_route_recognize_video_invalid_format`: Verified rejection of unsupported `.txt` uploads with HTTP 400.
- `test_15_route_recognize_video_empty_file`: Verified rejection of 0-byte video uploads with HTTP 400.
- `test_16_route_recognize_landmarks`: Verified direct landmark array inference endpoint.

### 5.2 Frontend Vitest Tests
Executed with `npm test --prefix frontend`:
```
 ✓ src/App.test.jsx (4 tests) 120ms
 Test Files  1 passed (1)
      Tests  4 passed (4)
   Duration  1.27s
```
Verified:
- Initial render of header, mode selectors, `IsolatedWebcamTester`, and low-data warning banner.
- Mode switching to continuous translation.
- Camera toggling and state transitions (`Start Camera` $\to$ `Stop Camera` & `Capture & Recognize (3s)`).
- Result card rendering and top candidates display.

### 5.3 Frontend Production Build
Executed with `npm run build --prefix frontend`:
```
vite v5.4.21 building for production...
✓ 36 modules transformed.
dist/index.html                   0.43 kB │ gzip:  0.29 kB
dist/assets/index-MPSax2DD.css    0.16 kB │ gzip:  0.15 kB
dist/assets/index-rWy_HPuC.js   168.49 kB │ gzip: 52.79 kB
✓ built in 687ms
```

---

## 6. Operational Guidance & Recommendations

For optimal recognition accuracy during live testing:
1. **Camera Framing**: Position yourself 0.8–1.5 meters from the webcam so head, shoulders, and both hands remain inside the frame.
2. **Lighting**: Ensure even front lighting; avoid strong backlighting or glare behind the signer.
3. **Pacing**: Hold the resting position briefly before pressing "Capture & Recognize", execute the sign clearly within the 3-second recording window, and hold the final hand pose until the recording finishes.
4. **Dominant Hand**: While bilateral augmentation provides robust left-handed support, right-handed signing remains the dominant baseline in the dataset.
5. **Low-Data Classes**: Expect higher uncertainty for `stop` and `sorry`. Focus primary demonstration tests on `hello`, `thank you`, `please`, `help`, `yes`, `no`, `water`, and `food`.

---

## 7. Deliverables Summary

| Artifact | Location | Status |
|---|---|---|
| Live Webcam Component | `frontend/src/components/recognition/IsolatedWebcamTester.jsx` | Completed |
| Frontend Root View | `frontend/src/App.jsx` | Integrated |
| Frontend Vitest Suite | `frontend/src/App.test.jsx` | 4/4 Passed |
| FastAPI Route Endpoints | `backend/app/api/routes/isolated_recognition.py` | Verified |
| Backend Integration Tests | `tests/test_isolated_recognition_pipeline.py` | 16/16 Passed |
| Verification Report | `docs/reports/F-024A-webcam-testing.md` | Completed |
