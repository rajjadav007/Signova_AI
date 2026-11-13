# Frontend Development Rules

## Signova AI

**Project:** Signova AI  
**Domain:** Indian Sign Language (ISL) Recognition  
**Frontend Framework:** React  
**Language:** JavaScript ES6+  
**Build Tool:** Vite  
**Styling:** CSS  
**Backend:** Python 3.11 + FastAPI  
**API Communication:** Fetch API or Axios  
**Development Approach:** Component-Based, Responsive, Backend-Connected

---

# 1. Project Frontend Overview

Signova AI is an Indian Sign Language recognition web application.

The frontend provides an interface for capturing signing input, sending compatible data to the backend, displaying actual AI predictions, and converting recognized text into speech.

The application has two separate recognition modes.

## 1.1 Individual Words Mode

**Status:** Primary implemented MVP.

This mode uses the existing 10-class Bi-GRU recognition engine.

Supported signs:

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

The interface must support:

- Webcam preview.
- Camera start and stop.
- Three-second recording countdown.
- Capture and Recognize.
- Expected-sign selection for evaluation.
- Video upload fallback.
- Prediction result.
- Confidence score.
- Top-3 predictions.
- Recognition status and rejection reason.
- Inference latency.
- Speech playback.
- Recognition test history.
- Correct, incorrect, and uncertain evaluation counts.

The frontend must display actual backend responses.

## 1.2 Continuous Translation Mode

**Status:** Separate experimental mode.

This mode is intended for the iSign T5 pose-to-text pipeline.

Requirements:

- Keep it separate from Individual Words mode.
- Do not reuse Bi-GRU-specific input processing.
- Do not display fabricated sentence translations.
- Check the actual backend model status.
- Show an unavailable or experimental state if the model is not operational.
- Enable translation controls only when the required backend functionality is verified.

Do not claim that isolated sign recognition provides continuous ISL sentence translation.

---

# 2. Core Frontend Principles

All frontend development must follow these rules:

1. Use React with JavaScript.
2. Use Vite as the build tool.
3. Use `.jsx` for React components.
4. Use `.js` for JavaScript utility modules.
5. Do not introduce TypeScript.
6. Keep components reusable and focused.
7. Separate UI, business logic, and API communication.
8. Keep the interface responsive and accessible.
9. Handle loading, success, empty, unavailable, and error states.
10. Do not create fake AI predictions or hardcoded recognition results.
11. Preserve existing working frontend functionality.
12. Keep Individual Words and Continuous Translation separate.
13. Do not redesign unrelated sections without a clear requirement.

---

# 3. Technology Rules

| Area | Technology |
|---|---|
| UI Framework | React |
| Language | JavaScript ES6+ |
| Build Tool | Vite |
| Styling | CSS |
| API Communication | Fetch API or Axios |
| Camera | Browser MediaDevices API |
| Speech Output | Browser SpeechSynthesis API |
| AI Processing | FastAPI backend |
| Testing | Vitest + React Testing Library |

Do not add another frontend framework or state-management library without a clear requirement.

Do not introduce TypeScript.

---

# 4. Frontend Architecture

Use a modular component-based structure.

The following is a logical reference structure. Inspect the existing repository and reuse existing files before creating new ones.

```text
frontend/
├── public/
├── src/
│   ├── assets/
│   ├── components/
│   │   ├── camera/
│   │   │   ├── CameraPreview.jsx
│   │   │   └── CameraControls.jsx
│   │   │
│   │   ├── recognition/
│   │   │   ├── IsolatedWebcamTester.jsx
│   │   │   ├── RecognitionResult.jsx
│   │   │   ├── PredictionCard.jsx
│   │   │   └── RecognitionHistory.jsx
│   │   │
│   │   ├── speech/
│   │   │   └── SpeechControls.jsx
│   │   │
│   │   ├── translation/
│   │   │   └── ContinuousTranslation.jsx
│   │   │
│   │   ├── layout/
│   │   │   ├── Header.jsx
│   │   │   └── Footer.jsx
│   │   │
│   │   └── common/
│   │       ├── Button.jsx
│   │       ├── Card.jsx
│   │       └── StatusIndicator.jsx
│   │
│   ├── services/
│   │   ├── api.js
│   │   └── recognitionService.js
│   │
│   ├── hooks/
│   ├── context/
│   ├── utils/
│   ├── styles/
│   ├── App.jsx
│   └── main.jsx
│
├── index.html
└── package.json
```

### Architecture Requirements

- Reuse existing components wherever possible.
- Keep camera logic separate from recognition logic.
- Keep speech logic separate from recognition logic.
- Keep API communication in service modules.
- Avoid putting the entire application in `App.jsx`.
- Avoid unnecessary component nesting.
- Do not create duplicate components or services.
- Keep the existing project structure unless a change is necessary.

---

# 5. React Component Rules

- Use functional components.
- Use React Hooks where appropriate.
- Keep components small and focused.
- Reuse common UI components.
- Keep camera logic separate from recognition logic.
- Keep speech logic separate from recognition logic.
- Keep recognition history separate from prediction display.
- Avoid unnecessary component nesting.
- Use meaningful component and variable names.
- Clean up event listeners and browser resources.

Do not create a separate component for every small HTML element unless reuse or clarity justifies it.

---

# 6. JavaScript Rules

- Use modern JavaScript ES6+.
- Use `const` by default.
- Use `let` only when reassignment is required.
- Use clear variable and function names.
- Use `async/await` for asynchronous operations.
- Handle rejected promises.
- Avoid unnecessary global variables.
- Avoid deeply nested callbacks.
- Keep utility functions reusable.
- Do not use TypeScript syntax.

All frontend source code must use `.js` and `.jsx` files.

---

# 7. API Integration Rules

The frontend must communicate with FastAPI through service modules.

### Existing API Structure

The reported API endpoints are:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/recognize/isolated/video` | Individual sign recognition from video |
| POST | `/api/v1/recognize/isolated/landmarks` | Individual sign recognition from landmarks |
| POST | `/api/v1/predict/video` | Separate continuous pipeline |
| POST | `/api/v1/predict/landmarks` | Separate continuous pipeline |

Verify actual endpoint contracts before modifying frontend API calls.

### Requirements

- Keep API calls outside presentation components where practical.
- Use a central API base URL.
- Use environment variables for backend URLs.
- Handle network failures.
- Handle non-success HTTP responses.
- Validate response data before using it.
- Avoid duplicate API logic.
- Do not hardcode prediction responses.
- Keep isolated and continuous API services separate.

Example environment variable:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Example API service:

```javascript
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

The example is illustrative. Use the actual health endpoint implemented in the backend.

Do not expose secrets in frontend environment variables.

Only variables intended to be public should use the `VITE_` prefix.

---

# 8. Camera Integration Rules

Use the browser MediaDevices API for camera access.

### Requirements

- Request camera access only after user interaction.
- Handle permission denial.
- Handle missing cameras.
- Handle camera startup errors.
- Display the live camera stream.
- Stop camera tracks when the camera is stopped.
- Clean up streams when the component unmounts.
- Avoid creating multiple unnecessary camera streams.
- Provide clear camera status.
- Keep the camera preview mirrored where required by the existing interface.

### Camera Behaviour

The user should be able to:

1. Start the camera.
2. View the live preview.
3. Stop the camera.
4. Capture a signing sample.
5. See the recording countdown.
6. Submit the captured sample for recognition.

Do not request camera permission automatically when the application opens.

### Important

The frontend is responsible for browser camera access.

The backend receives only the input required for recognition.

Do not assume that every AI model accepts raw camera frames. Check the selected model's input requirements before implementing frame submission.

---

# 9. Individual Words Recognition Interface

The Individual Words mode is the primary MVP interface.

It must support the existing 10-class Bi-GRU recognition workflow.

### Recognition Workflow

```text
User Opens Individual Words Mode
             |
             v
       Start Camera
             |
             v
       Live Preview
             |
             v
     Capture & Recognize
             |
             v
    Three-Second Countdown
             |
             v
       Record Sample
             |
             v
      Send Video to API
             |
             v
    Backend Bi-GRU Inference
             |
             v
      Receive Prediction
             |
             v
    Validate Recognition Status
             |
             v
       Display Result
             |
             v
       Optional Speech
```

### Expected-Sign Selector

Provide a selector containing the supported 10 signs.

Purpose:

- Allow the user to identify the sign they are attempting.
- Compare the actual prediction with the selected expected sign.
- Support evaluation history.

The selected expected sign is ground truth supplied by the user.

It must not influence or override the AI prediction.

### Prediction Result

Display the actual backend response.

Where returned by the backend, show:

- Predicted sign.
- Confidence.
- Top-3 predictions.
- Recognition status.
- Rejection reason.
- Inference latency.

Do not fabricate missing fields.

### Recognition Status

Support meaningful states:

- Recognized.
- Uncertain.
- Processing.
- Failed.
- Unsupported.
- Model unavailable.

An uncertain result must not be displayed as a successful recognition.

---

# 10. Recognition History and Evaluation

The existing tester includes an evaluation history.

Preserve its behaviour.

Display:

- Total attempts.
- Correct predictions.
- Incorrect predictions.
- Uncertain predictions.
- Recognition history.
- Reset history control.

### Evaluation Rules

- Compare the returned prediction with the selected expected sign.
- Count uncertain predictions separately.
- Do not count failed API requests as correct predictions.
- Do not change the actual prediction to match the expected sign.
- Do not fabricate evaluation records.
- Make clear that the history is a testing feature, not a validated universal accuracy score.

Do not introduce persistent database storage for this feature unless explicitly required.

---

# 11. Video Upload Fallback

Support video upload where the existing backend accepts compatible video files.

Requirements:

- Provide an upload control.
- Accept only supported formats.
- Validate file size where applicable.
- Show the selected file.
- Provide clear upload and processing states.
- Handle invalid files.
- Display actual backend results.
- Handle upload failures.
- Avoid storing uploaded videos unnecessarily.

The upload feature must use the same recognition pipeline as the supported backend endpoint.

---

# 12. Continuous Translation Interface

Keep Continuous Translation separate from Individual Words.

### Requirements

- Use a separate mode or section.
- Connect only to verified continuous translation functionality.
- Show actual model availability.
- Handle experimental or unavailable model status.
- Use the correct iSign T5 input format.
- Display actual generated text only.
- Handle generation errors.
- Do not reuse isolated recognition predictions as sentence output.

Do not display a successful continuous translation when the T5 model is unavailable or has failed.

---

# 13. Speech Output Rules

Use the browser SpeechSynthesis API for basic text-to-speech.

### Requirements

- Speak only available recognition text.
- Provide a stop action.
- Handle unsupported browser functionality.
- Prevent repeated speech requests from overlapping unexpectedly.
- Show the current speech status.
- Cancel speech when appropriate during component cleanup.

Speech output is separate from sign language recognition.

The speech feature must not change or fabricate the AI prediction.

---

# 14. State Management Rules

Use React state and hooks for local UI state.

Examples:

- Camera active or inactive.
- Recording countdown.
- Recognition loading.
- Current prediction.
- Confidence.
- Top-3 predictions.
- Recognition status.
- Expected sign.
- Recognition history.
- API connection status.
- Speech playing or stopped.
- Error messages.

### Requirements

- Keep state close to the component that uses it.
- Lift state only when multiple components need it.
- Avoid unnecessary global state.
- Avoid duplicated state.
- Do not store values that can be derived from existing state.
- Clean up event listeners and media resources.
- Prevent stale prediction data from appearing as a new result.

Do not introduce Redux or another state library unless the application genuinely requires it.

---

# 15. UI/UX Rules

Follow `UI_UX_GUIDELINES.md`.

The interface must be:

- Clean.
- Modern.
- Minimal.
- Responsive.
- Accessible.
- Consistent.

### Visual Identity

Use the established white-and-teal visual identity.

Reference palette:

| Purpose | Colour |
|---|---|
| Primary | `#0F766E` |
| Primary Dark | `#115E59` |
| Light Teal | `#CCFBF1` |
| Background | `#F8FAFC` |
| Main Text | `#0F172A` |

### Interface Priorities

1. Recognition mode selection.
2. Camera preview.
3. Recognition controls.
4. Expected-sign selector.
5. Prediction result.
6. Speech controls.
7. Evaluation history.
8. System and model status.

Avoid unnecessary dashboard elements or decorative sections that distract from recognition.

The Individual Words mode should be the default mode unless the product requirements are explicitly changed.

---

# 16. Responsive Design Rules

Support:

- Desktop.
- Tablet.
- Mobile.

Requirements:

- Use flexible layouts.
- Use CSS Grid and Flexbox where appropriate.
- Avoid fixed widths that break smaller screens.
- Prevent horizontal overflow.
- Keep camera controls accessible.
- Make result text readable.
- Keep expected-sign selection usable.
- Test important screen sizes.
- Do not hide essential recognition controls on mobile.

---

# 17. Accessibility Rules

- Use semantic HTML.
- Add labels to form controls.
- Add accessible names to icon buttons.
- Support keyboard navigation.
- Maintain visible focus states.
- Use sufficient colour contrast.
- Provide meaningful error messages.
- Avoid colour-only status indicators.
- Respect reduced-motion preferences.
- Ensure interactive elements have suitable touch targets.
- Communicate camera activity clearly.

Accessibility must be considered during development, not added only at the end.

---

# 18. Error and Loading States

Every asynchronous action must provide feedback.

### Loading

Show a clear loading indicator during API requests, video processing, and recognition.

### Success

Display the actual returned result.

### Uncertain

Explain that the model could not confidently recognize the sign.

### Empty

Explain when no recognition result is available.

### Error

Show a short, useful message.

### Offline

Indicate when the backend cannot be reached.

### Model Unavailable

Explain that the recognition service is currently unavailable.

Do not expose internal server errors or stack traces to users.

Do not show stale results as successful new predictions.

---

# 19. Performance Rules

- Avoid unnecessary React re-renders.
- Avoid unnecessary API requests.
- Process camera frames at a controlled rate.
- Release media streams when no longer needed.
- Avoid loading unnecessary libraries.
- Optimise images and static assets.
- Use memoisation only when it provides a real benefit.
- Avoid blocking the main UI thread with expensive processing.
- Prevent duplicate recognition requests.
- Provide feedback during long-running operations.

Do not send every camera frame to the backend without considering the model's processing requirements and system performance.

---

# 20. Security and Privacy Rules

- Do not store camera footage by default.
- Do not expose backend secrets.
- Do not trust frontend validation alone.
- Avoid unsafe HTML rendering.
- Use HTTPS in production.
- Do not log sensitive user media.
- Handle API errors safely.
- Request only necessary browser permissions.
- Stop camera tracks when no longer required.

The application must clearly communicate when the camera is active.

---

# 21. Testing Rules

Use Vitest and React Testing Library.

Test:

- Component rendering.
- Button interactions.
- Camera permission states.
- Camera cleanup.
- Recording countdown.
- Video upload validation.
- API success responses.
- API failure responses.
- Recognition loading states.
- Recognition result display.
- Confidence and top-3 display.
- Uncertain recognition state.
- Expected-sign selection.
- Evaluation history.
- Speech controls.
- Responsive behaviour where practical.
- Accessibility basics.

Mock browser APIs and backend requests where needed.

Do not report mocked API tests as real backend or AI model validation.

Run the frontend production build after relevant changes.

---

# 22. Dependency Rules

- Keep dependencies minimal.
- Check existing packages before installing new ones.
- Avoid duplicate libraries.
- Use compatible package versions.
- Do not install a UI library without a clear need.
- Keep package scripts organised.
- Verify the build after dependency changes.
- Do not replace existing working tools without a technical reason.

---

# 23. Development Workflow

Follow this order when implementing new frontend functionality.

## Phase 1: Existing Frontend Inspection

- Inspect the current frontend structure.
- Identify existing React components.
- Identify the default application mode.
- Identify existing camera implementation.
- Identify API service modules.
- Read `UI_UX_GUIDELINES.md`.
- Read `TECH_STACK.md`.

## Phase 2: Camera Interface

- Verify camera permission handling.
- Verify live preview.
- Verify start and stop controls.
- Verify recording countdown.
- Verify camera cleanup.
- Verify upload fallback.

## Phase 3: Backend Connection

- Verify API base URL.
- Verify service modules.
- Verify backend health connection.
- Verify request and response contracts.
- Handle loading and network errors.

## Phase 4: Recognition Interface

- Verify the isolated recognition workflow.
- Verify expected-sign selection.
- Display actual predictions.
- Display available confidence and top-3 results.
- Display uncertain and error states.
- Verify evaluation history.

## Phase 5: Speech Output

- Verify SpeechSynthesis support.
- Add or preserve speak and stop controls.
- Handle browser compatibility.
- Verify speech cleanup.

## Phase 6: Continuous Translation

- Inspect actual backend readiness.
- Connect only verified functionality.
- Keep the continuous mode separate.
- Show an honest unavailable state when necessary.

## Phase 7: Testing and Polish

- Test important interactions.
- Test responsive layouts.
- Check accessibility.
- Run frontend tests.
- Verify production build.

Do not repeat completed work without a technical reason.

---

# 24. Strict Rules for Antigravity

Before making frontend changes:

1. Inspect the existing frontend structure.
2. Read `UI_UX_GUIDELINES.md`.
3. Read `TECH_STACK.md`.
4. Read the API documentation.
5. Reuse existing components and services.
6. Do not create duplicate files.
7. Do not introduce TypeScript.
8. Do not hardcode fake recognition results.
9. Do not assume unsupported AI capabilities.
10. Do not change backend contracts without agreement.
11. Do not delete existing files without approval.
12. Do not install unnecessary dependencies.
13. Do not claim features work without testing them.
14. Keep code, comments, and tests in English.
15. Update relevant documentation when implementation changes.
16. Preserve the existing Individual Words recognition workflow.
17. Keep Continuous Translation separate from Individual Words.
18. Do not change model thresholds through frontend code.
19. Do not change the expected-sign selector into a mechanism that controls AI predictions.
20. Do not display mock results as real recognition.
21. Do not store webcam recordings without approval.
22. Preserve existing working UI functionality unless a change is required.

If a backend endpoint or AI model is not ready, build an honest loading, unavailable, or error state instead of simulating a successful result.

---

# 25. Final Frontend Goal

The Signova AI frontend must provide a smooth and accessible experience for:

- Capturing signing input.
- Recognizing supported isolated ISL signs.
- Displaying genuine model predictions.
- Showing confidence and recognition status.
- Evaluating recognition attempts.
- Converting available recognized text into speech.
- Providing a separate interface for future continuous translation.

**The frontend must reflect the real backend and AI capabilities. It must never hide model limitations behind fake UI behaviour.**