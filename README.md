# Signova AI 🤟

### Real-Time Two-Way Indian Sign Language (ISL) Recognition & Speech Platform

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%7C%20Vite-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/AI%2FML-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/Computer%20Vision-MediaPipe-0078D4?style=flat&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Sarvam AI](https://img.shields.io/badge/Cloud%20AI-Sarvam%20AI%20(Translation%20%26%20Bulbul%20TTS)-7928CA?style=flat)](https://sarvam.ai/)
[![Tests](https://img.shields.io/badge/Tests-143%20Vitest%20%7C%2063%2B%20Pytest%20Passing-success?style=flat)](file:///d:/Projects/Signova%20AI/docs/PROGRESS_TRACKER.md)
[![Team](https://img.shields.io/badge/Team-The%20Nexus-8A2BE2?style=flat)](#-the-team--hackathon-credentials)
[![Pitch Deck](https://img.shields.io/badge/Pitch%20Deck-PPTX%20%26%20HTML5-06B6D4?style=flat)](#-the-team--hackathon-credentials)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📖 Table of Contents

- [The Team & Pitch Deck](#-the-team--hackathon-credentials)
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Two-Way Communication Paradigm](#-two-way-communication-paradigm)
- [System Architecture & Flow](#-system-architecture--flow)
- [Supported ISL Vocabulary (20 Signs)](#-supported-isl-vocabulary-20-signs)
- [Supported Indian Languages & Voices](#-supported-indian-languages--voices)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Environment Configuration](#2-environment-configuration)
  - [3. Backend Setup](#3-backend-setup)
  - [4. Frontend Setup](#4-frontend-setup)
- [API Reference](#-api-reference)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Model Architecture & Benchmarks](#-model-architecture--benchmarks)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 👥 The Team • Hackathon Credentials

| Role | Name | Focus Areas |
| :--- | :--- | :--- |
| **Team Name** | **The Nexus** | National Level Hackathon Project |
| **Team Leader** | **Raj Jadav** | Full-Stack Architecture, AI Pipeline & System Integration |
| **Team Member** | **Harsh Sanchaniya** | AI/ML Modeling, Bi-GRU Optimization & Feature Engineering |
| **Team Member** | **Jaineel Khatri** | Frontend React Development, Two-Way UI & Accessibility |
| **Team Member** | **Om Panday** | Cloud APIs, Sarvam AI Integration, Testing & Validation |

### 📽️ Hackathon Presentation Decks
- **PowerPoint Presentation (.pptx):** [Signova_AI_Hackathon_Pitch_The_Nexus.pptx](file:///d:/Projects/Signova%20AI/Signova_AI_Hackathon_Pitch_The_Nexus.pptx) *(11 slides, 16:9 widescreen, dark theme)*
- **Interactive Browser Slide Deck (.html):** [docs/HACKATHON_PITCH_DECK.html](file:///d:/Projects/Signova%20AI/docs/HACKATHON_PITCH_DECK.html) *(Zero dependency, presenter arrows `←`/`→`/`Space`)*
- **PPT Generator Script:** [scripts/generate_hackathon_ppt.py](file:///d:/Projects/Signova%20AI/scripts/generate_hackathon_ppt.py)

---

## 🌟 Overview

**Signova AI** is a real-time, bidirectional communication assistant designed to bridge the everyday conversation gap between Deaf individuals who communicate through **Indian Sign Language (ISL)** and hearing individuals who communicate through spoken regional languages.

Unlike traditional unidirectional sign recognition tools, Signova AI operates as a unified **Two-Way Assistant**:
1. **Deaf &rarr; Hearing:** Translates live ISL hand and body gestures into editable sentence sequences, translates them into 11 Indian languages via Sarvam AI, and synthesizes natural regional voice output using Sarvam Bulbul v3 neural text-to-speech.
2. **Hearing &rarr; Deaf:** Listens to the hearing participant's spoken response through speech recognition, translates it instantly into the Deaf user's preferred Indian regional script, and displays visual sign token references and high-contrast captions.

---

## ✨ Key Features

- **Real-Time Visual Landmark Extraction:** Powered by Google MediaPipe Holistic to track 75 upper-body landmarks (hands and pose) with zero wearable hardware required.
- **Dual AI Engine Architecture:**
  - **Isolated Sign Recognition (Production Champion):** A lightweight 2-Layer Bidirectional GRU (Bi-GRU) trained on 20 daily vocabulary classes with bilateral mirror augmentation, achieving **<20ms inference latency** and robust generalization for both right- and left-handed signers.
  - **Continuous Translation Engine (Experimental):** An integrated pipeline supporting the pretrained `iSign-t5-pose-to-text` transformer for sequence-to-sequence continuous sign translation.
- **Interactive ISL Message Builder:** Allows signers to assemble multi-sign sentences, reorder or remove sign tokens, insert suggested words, and fine-tune grammar before sending.
- **Multilingual Indian Language Translation:** Seamlessly integrated with **Sarvam AI Translation API** supporting 11 regional languages (Hindi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Punjabi, Odia, English).
- **Natural Voice Synthesis:** Powered by **Sarvam Bulbul v3** neural TTS for rich Indian voice personas (e.g. `priya`, `aditya`) with adjustable speech tempo and Web Speech API fallback.
- **Confidence Scoring & Uncertainty Detection:** Rejects low-confidence predictions ($\tau = 0.50$–$0.75$) with helpful suggestion banners, ensuring high conversational trust.
- **Webcam Validation & Benchmarking Suite:** Built-in testing suite (`IsolatedWebcamTester`) with real-time video preview, capture countdown, top-3 candidates, latency monitor, and session accuracy history.

---

## 🔄 Two-Way Communication Paradigm

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               SIGNOVA AI ASSISTANT                                     │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│         PANEL A: DEAF ➔ HEARING           │          PANEL B: HEARING ➔ DEAF           │
│             ("Your Sign")                 │               ("Other Voice")              │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ 1. Laptop Webcam / Video Stream           │ 1. Hearing Person Speaks into Microphone   │
│       │                                   │       │                                    │
│ 2. MediaPipe Landmark Extraction          │ 2. Speech-to-Text / Audio Capture          │
│       │                                   │       │                                    │
│ 3. Bilateral Bi-GRU Inference (<20ms)     │ 3. Real-Time Regional Translation          │
│       │                                   │       │                                    │
│ 4. ISL Message Builder & Token Ordering   │ 4. High-Contrast Text Display & Visual     │
│       │                                   │    ISL Sign Reference Tokens               │
│ 5. Sarvam AI Regional Translation         │       │                                    │
│       │                                   │ 5. Read by Deaf User Instantly             │
│ 6. Sarvam Bulbul v3 Spoken Audio          │                                            │
│       │                                   │                                            │
│ 7. Heard by Hearing Person Instantly      │                                            │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

---

## 🏗️ System Architecture & Flow

```text
   +------------------+             +-------------------+
   |  React Frontend  | < - - - - > |  FastAPI Backend  |
   | (Vite + Tailwind)|             |   (Python 3.11)   |
   +--------+---------+             +---------+---------+
            |                                 |
            v                                 v
   +------------------+             +-------------------+
   |  Webcam Stream   |             |   API Routers:    |
   |   & Audio Mic    |             | - /recognize      |
   +--------+---------+             | - /predict (T5)   |
            |                       | - /translate      |
            v                       | - /speech         |
   +------------------+             +---------+---------+
   | MediaPipe Hands  |                       |
   | & Upper Body     |                       v
   +--------+---------+             +-------------------+
            | (64 frames x 600-d)   | AI Service Layer  |
            +---------------------> | - Bi-GRU Champion |
                                    | - Sarvam Client   |
                                    +-------------------+
```

---

## 🤟 Supported ISL Vocabulary (20 Signs)

The primary isolated sign engine (`isolated_20class_best.joblib`) recognizes 20 high-frequency Indian Sign Language gestures:

| Category | Supported Signs |
| :--- | :--- |
| **Greetings & Polite** | `hello`, `thank you`, `please`, `sorry`, `welcome` |
| **Responses & Controls** | `yes`, `no`, `stop`, `help` |
| **Needs & Sustenance** | `food`, `water` |
| **Family & People** | `father`, `mother`, `sister`, `friend`, `you` |
| **Education & Society** | `teacher`, `student`, `school`, `market`, `hospital` |

*Input format:* 64 normalized temporal frames $\times$ 600-dimensional masked spatial-velocity features extracted from MediaPipe joints.

---

## 🌐 Supported Indian Languages & Voices

Signova AI connects with **Sarvam AI** cloud services to translate and synthesize speech across India's major linguistic regions:

| Language Code | Language | Sarvam Translate | Bulbul v3 Neural Voice |
| :--- | :--- | :---: | :---: |
| `hi-IN` | Hindi | ✅ | ✅ (`priya`, `aditya`, etc.) |
| `gu-IN` | Gujarati | ✅ | ✅ (`pooja`, `bhavik`) |
| `ta-IN` | Tamil | ✅ | ✅ (`gokul`, `aravind`) |
| `te-IN` | Telugu | ✅ | ✅ (`kavitha`, `tarun`) |
| `kn-IN` | Kannada | ✅ | ✅ (`chaitra`, `chetan`) |
| `ml-IN` | Malayalam | ✅ | ✅ |
| `bn-IN` | Bengali | ✅ | ✅ (`roopa`, `arnab`) |
| `mr-IN` | Marathi | ✅ | ✅ (`ishita`, `mrunal`) |
| `pa-IN` | Punjabi | ✅ | ✅ (`anand`, `harpreet`) |
| `od-IN` | Odia | ✅ | ✅ |
| `en-IN` | English (Indian) | ✅ | ✅ (`aditi`, `ashwin`) |

---

## 🛠️ Technology Stack

| Layer | Technologies Used | Description |
| :--- | :--- | :--- |
| **Frontend** | React 18, Vite, React Router 7, Tailwind CSS, Lucide React | Modular single-page application with responsive dark/light styling |
| **Backend** | Python 3.11, FastAPI, Uvicorn, Pydantic v2 | High-throughput asynchronous REST API with Swagger documentation |
| **Deep Learning** | PyTorch, Scikit-Learn, Joblib, NumPy | 2-Layer Bi-GRU isolated sign classifier with bilateral mirror data augmentation |
| **Computer Vision** | Google MediaPipe, OpenCV | 75 landmarks tracking upper body, face contours, and 21 finger joints per hand |
| **Cloud AI Services** | Sarvam AI (`sarvam-translate:v1`, `bulbul:v3`) | Multilingual Indian translation & neural text-to-speech synthesis |
| **Speech Fallback** | Web Speech API (`SpeechSynthesis`, `webkitSpeechRecognition`) | Zero-latency local browser speech synthesis and recognition fallback |
| **Quality & Tests** | Vitest, Testing Library, Pytest, Pytest-AsyncIO | 143 Vitest frontend tests + 63+ Pytest backend tests |

---

## 📁 Repository Structure

```text
Signova AI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── health.py                 # System health check routes
│   │   │   │   ├── isolated_recognition.py   # Bi-GRU 20-sign classification routes
│   │   │   │   ├── predict.py                # Continuous iSign T5 translation routes
│   │   │   │   ├── speech.py                 # Sarvam Bulbul TTS & speaker routes
│   │   │   │   └── translation.py            # Sarvam multilingual translation routes
│   │   │   └── router.py                     # Centralized API v1 route aggregator
│   │   ├── core/
│   │   │   ├── config.py                     # Environment & application settings
│   │   │   └── exceptions.py                 # Sanitized custom API exception handlers
│   │   ├── schemas/                          # Pydantic v2 validation schemas
│   │   ├── services/
│   │   │   ├── isolated_inference_service.py # Bi-GRU inference runtime
│   │   │   ├── inference_service.py          # iSign T5 inference runtime
│   │   │   └── sarvam_service.py             # Sarvam AI translation & TTS client
│   │   └── main.py                           # FastAPI application entrypoint
│   ├── tests/                                # Backend pytest test suite (63+ tests)
│   └── requirements.txt                      # Backend dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── camera/                       # LiveWebcamPreview, WebcamRecorder
│   │   │   ├── communication/                # PanelYourSign, PanelOtherVoice, LanguageSelectorBar
│   │   │   ├── recognition/                  # IsolatedWebcamTester, IsolatedResult
│   │   │   ├── translation/                  # ISLMessageBuilder, SuggestionBanner
│   │   │   ├── speech/                       # SpeechSynthesizer, AudioPlayer
│   │   │   ├── dashboard/                    # Metrics, QuickAccess, Vocabulary cards
│   │   │   └── layout/                       # Header, Sidebar, MainLayout
│   │   ├── pages/                            # DashboardPage, LiveRecognitionPage
│   │   ├── services/                         # api.js client (endpoints, speech, translation)
│   │   └── App.jsx                           # React application router
│   ├── package.json                          # Frontend dependencies & scripts
│   └── vite.config.js                        # Vite bundler configuration
│
├── ai/
│   ├── isolated/                             # Feature extractor, preprocessor & Bi-GRU classifier
│   └── pose_extraction/                      # MediaPipe holistic landmark extractor
│
├── models/
│   ├── checkpoints/isolated/                 # Trained models (isolated_20class_best.joblib)
│   ├── labels/                               # isolated_20class_labels.json
│   └── assets/                               # MediaPipe model bundles
│
├── scripts/                                  # Data extraction, feature generation & training scripts
├── docs/                                     # Comprehensive architecture, API docs & pitch deck
│   └── HACKATHON_PITCH_DECK.html             # Interactive HTML5 presentation slide deck
├── Signova_AI_Hackathon_Pitch_The_Nexus.pptx # 11-Slide widescreen hackathon presentation
├── .env.example                              # Environment variable configuration template
├── pytest.ini                                # Pytest runner configuration
└── README.md                                 # Project documentation
```

---

## ⚙️ Getting Started

### Prerequisites

- **Python 3.11+** installed and available in your PATH.
- **Node.js 18+** and **npm** installed.
- Modern Web Browser with Webcam and Microphone access (Chrome, Edge, or Firefox).
- *(Optional)* Free **Sarvam AI API Key** for neural translation and Bulbul TTS ([Get an API Key](https://www.sarvam.ai/)).

---

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Signova-AI.git
cd "Signova AI"
```

---

### 2. Environment Configuration

Create a local `.env` file in the project root:

```bash
# Windows PowerShell
Copy-Item .env.example .env
```

Open `.env` and verify the settings:

```dotenv
# Application Configuration
APP_NAME="Signova AI"
APP_ENV=development
DEBUG=true

# Backend Server Configuration
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
API_V1_PREFIX=/api/v1
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Frontend Configuration
FRONTEND_URL=http://localhost:5173
VITE_API_URL=http://127.0.0.1:8000

# AI/ML Execution & Model Paths
MODEL_DEVICE=cpu
MODEL_PATH=models/checkpoints/

# Sarvam AI Cloud Services (FE-08)
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_BASE_URL=https://api.sarvam.ai
SARVAM_TIMEOUT_SECONDS=15.0
```

> **Note:** If `SARVAM_API_KEY` is omitted, Signova AI operates normally and automatically falls back to client-side browser speech synthesis.

---

### 3. Backend Setup

Open a terminal in the project directory:

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # On Linux/macOS: source .venv/bin/activate

# 2. Install backend dependencies
pip install -r backend/requirements.txt

# 3. Start the FastAPI development server
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Verify the backend is live:
- **Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

---

### 4. Frontend Setup

In a second terminal window:

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the Vite development server
npm run dev
```

Open your browser at **`http://localhost:5173`**.

---

## 📡 API Reference

Interactive OpenAPI documentation is live at `/docs`. Primary endpoints include:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Comprehensive system and AI model status |
| `POST` | `/api/v1/recognize/isolated/video` | Upload signing video clip for 20-class Bi-GRU inference |
| `POST` | `/api/v1/recognize/isolated/landmarks` | Send raw 64-frame landmark sequence for immediate classification |
| `GET` | `/api/v1/recognize/isolated/labels` | List all supported ISL vocabulary tokens |
| `POST` | `/api/v1/translate` | Translate text into any of the 11 supported Indian languages |
| `GET` | `/api/v1/translate/languages` | List supported translation language codes |
| `POST` | `/api/v1/speech/synthesize` | Convert text to spoken audio via Sarvam Bulbul v3 neural voice |
| `GET` | `/api/v1/speech/speakers` | List supported voice personas (`priya`, `aditya`, etc.) |
| `POST` | `/api/v1/predict/landmarks` | Experimental continuous iSign T5 pose-to-text inference |

### Sample Translation Request & Response

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Water please",
    "source_language_code": "en-IN",
    "target_language_code": "hi-IN"
  }'
```

```json
{
  "success": true,
  "translated_text": "कृपया पानी",
  "source_language_code": "en-IN",
  "target_language_code": "hi-IN",
  "processing_time_ms": 942.3
}
```

---

## 🧪 Testing & Quality Assurance

Signova AI includes full unit, integration, and regression test suites for both frontend and backend.

### Run Frontend Vitest Suite (143 Tests)

```bash
cd frontend
npm test
```

*Results:* 143 passed across 12 test files covering components, message builders, two-way panels, and reliability handlers.

### Run Backend Pytest Suite (63+ Tests)

```bash
# From repository root
pytest backend/tests/test_sarvam_integration.py backend/tests/test_health.py backend/tests/test_isolated_recognition_endpoints.py
```

*Results:* All health, Sarvam translation/TTS, and isolated classification tests pass cleanly without leaking credentials.

---

## 📊 Model Architecture & Benchmarks

The isolated sign classifier was developed to overcome key limitations in real-world signer interaction:

### Evaluation Highlights

- **Bilateral Mirror Data Augmentation:** Solved the left-hand signer asymmetry gap, boosting accuracy on left-dominant test users (`User014`) from **12.5% to 75.0%** and raising overall test accuracy to **76.7%**.
- **Ultra-Low Latency:** Inference completes in **~18ms** on a standard multi-core laptop CPU, enabling seamless real-time interaction without expensive GPU hardware.
- **Top-3 Accuracy:** Achieves **>86% Top-3 accuracy**, ensuring alternative candidate suggestions displayed in the UI contain the intended sign even in ambiguous lighting.

| Model Candidate | Target Classes | Validation Acc | Test Acc | Weighted F1 | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest (100 Trees) | 20 Classes | 62.5% | 52.0% | 0.597 | ~12ms |
| Linear SGD Baseline | 20 Classes | 23.4% | 20.1% | 0.223 | ~5ms |
| **Bi-GRU + Bilateral Mirroring (Champion)** | **20 Classes** | **73.4%** | **76.7%** | **0.726** | **~18ms** |

---

## 🗺️ Roadmap

- [x] **Phase 1–5:** Core backend, visual landmark pipeline, MediaPipe feature normalization.
- [x] **Phase 6–7:** Dual AI engine routes, React two-way communication UI, live webcam preview.
- [x] **Phase 8:** Multilingual Indian language translation & Sarvam Bulbul v3 neural voice synthesis.
- [x] **Phase 9:** 20-word vocabulary expansion (`father`, `mother`, `school`, `hospital`, etc.).
- [ ] **Phase 10:** Real-world webcam field testing and hackathon demo clip suite.
- [ ] **Continuous Translation Scaling:** Training custom transformer pose-to-text models on expanded Indian Sign Language datasets.
- [ ] **Mobile Progressive Web App (PWA):** Touch-optimized view for tablets and smartphones.

---

## 🤝 Contributing

Contributions, bug reports, and pull requests are warmly welcomed!

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m "Add some AmazingFeature"`).
4. Run tests to ensure all tests pass (`npm test` & `pytest`).
5. Push to the Branch (`git push origin feature/AmazingFeature`).
6. Open a Pull Request.

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more details.

---

<p align="center">
  <b>Signova AI</b> &bull; Bridging Communication, One Sign at a Time. 🤟
</p>
