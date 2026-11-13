# Technology Stack

## Signova AI

**Project:** Signova AI  
**Architecture:** React Frontend + FastAPI Backend + Modular AI Pipeline  
**Development Approach:** Backend First  
**Primary Environment:** Local Windows

---

# 1. Technology Overview

Signova AI uses a modular technology stack for frontend development, backend services, computer vision, and pretrained AI model integration.

| Layer | Technology | Purpose |
|---|---|---|
| Frontend Framework | React | Build the user interface |
| Frontend Language | JavaScript (ES6+) | Frontend application logic |
| Build Tool | Vite | Frontend development and bundling |
| Styling | CSS | Application styling |
| Backend Framework | FastAPI | REST API development |
| Backend Language | Python 3.11 | Backend and AI integration |
| AI Framework | PyTorch | Pretrained model inference |
| Computer Vision | OpenCV | Image and video processing |
| Landmark Extraction | MediaPipe | Hand and body landmarks, if required |
| Model Repository | Hugging Face | Pretrained model access |
| Speech Output | Web Speech API | Text-to-speech |
| Database | PostgreSQL | Optional future data storage |
| Backend Testing | Pytest | Python testing |
| Frontend Testing | Vitest | JavaScript testing |

Final dependencies must be confirmed against the selected pretrained model.

---

# 2. Frontend Technology

## React

**Purpose:** Build a component-based user interface.

Responsibilities:

- Application layout.
- Camera interface.
- Recognition controls.
- Prediction display.
- Speech controls.
- Reusable UI components.

## JavaScript (ES6+)

**Purpose:** Implement all frontend application logic.

JavaScript will be used for:

- React components.
- State management.
- API communication.
- Camera handling.
- Event handling.
- Recognition workflow.
- Speech controls.

**File extensions:**

- `.js` for JavaScript modules.
- `.jsx` for React components.

TypeScript is not part of the selected frontend stack.

## Vite

**Purpose:** Frontend development server and production build tool.

Responsibilities:

- Fast development environment.
- Module handling.
- Production bundling.
- Environment variable configuration.

## CSS

**Purpose:** Style the application.

Responsibilities:

- Layout and responsive design.
- Camera preview styling.
- Recognition result styling.
- Accessibility-focused UI.
- Application themes.

---

# 3. Backend Technology

## Python 3.11

**Purpose:** Main backend programming language.

Responsibilities:

- API logic.
- AI service integration.
- Model loading.
- Preprocessing coordination.
- Prediction processing.
- Error handling.

## FastAPI

**Purpose:** Provide REST APIs for frontend and AI communication.

Responsibilities:

- HTTP endpoints.
- Request validation.
- Response formatting.
- API documentation.
- CORS configuration.
- Error handling.

## Pydantic

**Purpose:** Validate backend request and response data.

Responsibilities:

- Request schemas.
- Response schemas.
- Data validation.
- Configuration validation.

---

# 4. AI and Machine Learning Technology

## PyTorch

**Purpose:** Load and execute pretrained machine learning models.

Responsibilities:

- Model loading.
- Inference execution.
- CPU and GPU execution.
- Tensor processing.
- Model evaluation.

## Hugging Face

**Purpose:** Access pretrained models and model configurations.

Initial experimental candidate:

`manavdhamecha77/iSign-t5-pose-to-text`

Initial checkpoint:

`t5-small`

The candidate uses structured pose and hand landmark inputs.

It requires model-specific preprocessing before inference.

## Transformers

**Purpose:** Load and execute compatible transformer-based models.

Use the version and configuration supported by the selected model repository.

## OpenCV

**Purpose:** Image and video processing.

Potential responsibilities:

- Video frame extraction.
- Image resizing.
- Frame validation.
- Image format conversion.

## MediaPipe

**Purpose:** Extract visual landmarks when required by the selected model.

Potential responsibilities:

- Hand landmark extraction.
- Pose landmark extraction.
- Landmark formatting.

MediaPipe output must be converted into the exact format expected by the selected model.

It must not be added as a mandatory dependency if the selected model uses a different input pipeline.

---

# 5. Speech Technology

## Web Speech API

**Purpose:** Convert recognised text into spoken audio in the browser.

Main interface:

`SpeechSynthesis`

Responsibilities:

- Speak recognised text.
- Stop speech playback.
- Select available browser voices.
- Handle unsupported speech features.

The initial MVP does not require a separate speech generation backend.

---

# 6. Database Technology

## PostgreSQL

**Status:** Optional future integration.

Potential responsibilities:

- User accounts.
- Recognition history.
- User preferences.
- Model metadata.

PostgreSQL is not required for the initial recognition MVP.

The basic recognition workflow must work without a database connection.

---

# 7. Testing Technology

| Tool | Purpose |
|---|---|
| Pytest | Backend and AI module testing |
| Vitest | Frontend JavaScript testing |
| React Testing Library | React component testing |
| FastAPI TestClient | API endpoint testing |

Testing should cover individual modules and the complete recognition workflow.

---

# 8. Development Tools

| Tool | Purpose |
|---|---|
| Visual Studio Code / Antigravity | Development environment |
| Git | Version control |
| GitHub | Source code repository |
| npm | JavaScript dependency management |
| pip | Python dependency management |
| Virtual Environment | Python dependency isolation |
| Kaggle | GPU fallback for experiments |
| Hugging Face Hub | Pretrained model downloads |

Primary development should happen locally on Windows.

Kaggle may be used if local hardware or memory is insufficient.

---

# 9. Frontend Project Structure

```text
frontend/
├── public/
└── src/
    ├── assets/
    ├── components/
    │   ├── camera/
    │   ├── recognition/
    │   ├── translation/
    │   ├── speech/
    │   ├── layout/
    │   └── common/
    ├── pages/
    ├── hooks/
    ├── services/
    ├── context/
    ├── utils/
    ├── styles/
    ├── App.jsx
    └── main.jsx
```

Use JavaScript and JSX throughout the frontend.

---

# 10. Backend and AI Project Structure

```text
backend/
├── app/
│   ├── api/
│   │   └── routes/
│   ├── core/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── tests/
└── requirements.txt

ai/
├── preprocessing/
├── pose_extraction/
├── inference/
├── model_adapters/
├── postprocessing/
└── config/
```

Keep AI processing separate from FastAPI route handlers.

---

# 11. Environment Requirements

## Frontend

- Node.js.
- npm.
- React.
- Vite.
- JavaScript ES6+.

## Backend

- Python 3.11.
- FastAPI.
- Uvicorn.
- Pydantic.

## AI

- PyTorch.
- Transformers, if required.
- OpenCV, if required.
- MediaPipe, if required.
- Model-specific dependencies.

Use the official PyTorch installation selector to choose a compatible CPU or CUDA-enabled build.

Verify GPU availability before running local model inference.

---

# 12. Dependency Management Rules

- Keep frontend dependencies in `package.json`.
- Keep backend dependencies in `requirements.txt` or an existing Python dependency manager.
- Pin or constrain important dependency versions.
- Avoid unnecessary packages.
- Keep model-specific dependencies isolated where practical.
- Review dependency compatibility before upgrades.
- Do not commit virtual environments.
- Do not commit large model checkpoints.

---

# 13. Technology Selection Principles

The technology stack follows these principles:

1. Use JavaScript for all frontend application code.
2. Use Python for backend and AI integration.
3. Prefer pretrained models over training from scratch for the hackathon MVP.
4. Keep model-specific preprocessing separate.
5. Use GPU acceleration when compatible and available.
6. Keep database integration optional.
7. Select dependencies based on actual implementation needs.
8. Prefer maintainable and testable modules.

---

# 14. Final Technology Stack

| Component | Selected Technology |
|---|---|
| Frontend | React |
| Frontend Language | JavaScript |
| Build Tool | Vite |
| Styling | CSS |
| Backend | FastAPI |
| Backend Language | Python 3.11 |
| AI Framework | PyTorch |
| Model Access | Hugging Face |
| Computer Vision | OpenCV |
| Landmark Extraction | MediaPipe, if required |
| Speech | Browser SpeechSynthesis |
| Database | PostgreSQL, optional |
| Backend Testing | Pytest |
| Frontend Testing | Vitest |
| Version Control | Git and GitHub |

**Final decision:** React + JavaScript + Vite for the frontend, and Python + FastAPI + PyTorch for the backend and AI pipeline.