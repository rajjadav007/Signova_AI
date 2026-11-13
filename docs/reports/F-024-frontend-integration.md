# F-024: Frontend Dual-Mode & Isolated Recognition Integration Report

**Feature ID:** F-024  
**Module:** Web Frontend & Full-Stack Integration  
**Date:** 2026-11-12  
**Status:** Completed & Validated  
**Frontend Components:**  
- [frontend/src/components/recognition/IsolatedResult.jsx](file:///D:/Signova%20AI/frontend/src/components/recognition/IsolatedResult.jsx) (New)  
- [frontend/src/App.jsx](file:///D:/Signova%20AI/frontend/src/App.jsx) (Updated)  
- [frontend/src/services/api.js](file:///D:/Signova%20AI/frontend/src/services/api.js) (Updated)  
**Test Suites:**  
- Frontend: `frontend/src/App.test.jsx` (4 Vitest tests passing)  
- Backend: `tests/test_isolated_recognition_pipeline.py` (12 tests) + `backend/tests/` (12 tests) = 24 passing tests  

---

## 1. Executive Summary

This feature integrates Signova AI's trained 10-class Isolated ISL Recognition model into the React web frontend, introducing a clean, dual-mode operational interface:

1. **Individual Words Mode (10-Sign MVP)** (*Default*):
   - Backed by the **Bi-GRU Bilateral Model** (`isolated_10class_improved.joblib`, 76.7% test accuracy, 86.7% top-3 accuracy).
   - Fast inference (< 100 ms).
   - Displays recognized sign, confidence status pill (`✓ RECOGNIZED` vs `⚠ UNCERTAIN`), top-3 candidate probability progress bars, and automatic text-to-speech.
2. **Continuous Translation Mode**:
   - Backed by the **iSign T5 Simple Model** for full sentence pose-to-text generation.
   - Configurable beam search decoding and speech output.

Both recognition systems coexist seamlessly within a single unified web client.

---

## 2. Frontend API Service Extensions

Updated [frontend/src/services/api.js](file:///D:/Signova%20AI/frontend/src/services/api.js) with dedicated client methods:

```javascript
// 1. Send recorded webcam video to isolated recognition endpoint
export async function recognizeIsolatedVideo(fileBlob, fileName = "recording.webm", category = null, topK = 3) {
  const formData = new FormData();
  formData.append("video", fileBlob, fileName);
  const queryParams = new URLSearchParams();
  if (category) queryParams.set("category", category);
  queryParams.set("top_k", topK);

  const response = await fetch(`${API_BASE}/recognize/isolated/video?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  });
  return response.json();
}

// 2. Direct landmark sequence JSON dispatch
export async function recognizeIsolatedLandmarks(landmarks, category = null, topK = 3) {
  const response = await fetch(`${API_BASE}/recognize/isolated/landmarks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ landmarks, category, top_k: topK }),
  });
  return response.json();
}
```

---

## 3. UI Component Architecture

### 3.1 IsolatedResult Component
Implemented in [frontend/src/components/recognition/IsolatedResult.jsx](file:///D:/Signova%20AI/frontend/src/components/recognition/IsolatedResult.jsx):
- **Recognized Sign Header**: Large high-contrast typography displaying the predicted sign (`Hello`, `Thank you`, `Water`, etc.).
- **Confidence Status Pill**:
  - `✓ RECOGNIZED` (Dark green `#1b5e20` background, light green text) when confidence $\ge$ rejection threshold ($\tau = 0.75$).
  - `⚠ UNCERTAIN` (Amber `#e65100` background) with explanatory message (e.g. *"Confidence 0.40 below rejection threshold 0.75"*) when input is ambiguous or low-confidence.
- **Top-3 Candidate Probability Bars**: Visual horizontal progress bars showing relative candidate distribution across top predictions.
- **Speech Synthesis**: Native Web Speech API integration auto-speaking recognized words upon arrival when enabled.
- **Inference Telemetry**: Real-time display of end-to-end processing latency in milliseconds and model identifier.

### 3.2 Dual-Mode Switching in App.jsx
Updated [frontend/src/App.jsx](file:///D:/Signova%20AI/frontend/src/App.jsx):
- Interactive toggle buttons in header:
  - **Individual Words (10-Sign MVP)**
  - **Continuous Translation (iSign T5)**
- Automatically routes the single recorded video blob from [WebcamRecorder.jsx](file:///D:/Signova%20AI/frontend/src/components/camera/WebcamRecorder.jsx) to the corresponding backend pipeline without requiring separate recorder instances.

---

## 4. Verification & Testing

### 4.1 Frontend Test Suite
Automated Vitest tests in [frontend/src/App.test.jsx](file:///D:/Signova%20AI/frontend/src/App.test.jsx):
- `renders header, camera controls, mode selector, and isolated recognition section by default`: **PASSED**
- `switches to Continuous Translation mode when toggle button is clicked`: **PASSED**
- `toggles camera when Start Camera is clicked`: **PASSED**
- `renders IsolatedResult component with predictions and candidate probabilities`: **PASSED**

Run command:
```powershell
cd frontend; npm test -- --run
```
Result: **4/4 tests passed in 1.46s**.

### 4.2 Full System Integration
- `POST /api/v1/recognize/isolated/video`: Verified with multipart form uploads.
- `POST /api/v1/recognize/isolated/landmarks`: Verified with JSON landmark arrays.
- `POST /api/v1/predict/video` & `POST /api/v1/predict/landmarks`: Verified with iSign T5 continuous translation.
- Total passing automated tests: **28 tests across backend and frontend**.
