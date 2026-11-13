# Signova AI 🤟

### Bridging Communication Through Indian Sign Language

Signova AI is an AI-powered Indian Sign Language (ISL) recognition and translation platform designed to help bridge communication between the Deaf and hearing communities.

The project aims to recognize ISL gestures through a camera, convert recognized signs into readable text, and provide speech output to make everyday communication more accessible.

> **Project status:** Under active development.

---

## 🎯 Vision

Communication should not be limited by language or hearing barriers.

Signova AI aims to make Indian Sign Language communication more accessible through computer vision, deep learning, and real-time AI-powered recognition.

## ✨ Planned Features

- **Live Camera Recognition:** Capture sign language gestures using a webcam.
- **ISL Gesture Recognition:** Identify supported Indian Sign Language signs.
- **Text Translation:** Convert recognized signs into readable text.
- **Text-to-Speech:** Convert recognized text into spoken audio.
- **Real-Time Processing:** Process camera frames and display predictions.
- **Modular AI Pipeline:** Support future expansion of the ISL vocabulary and recognition capabilities.

*Feature availability depends on successful model integration and validation.*

## 🔄 How It Works

The planned recognition pipeline:

```text
Camera Input
     |
     v
Video Frame Capture
     |
     v
Frame Processing
     |
     v
Landmark Extraction
(if required by the model)
     |
     v
Model Preprocessing
     |
     v
Pretrained ISL Model
     |
     v
Prediction Decoding
     |
     v
Recognized Text
     |
     v
Text-to-Speech
     |
     v
Spoken Output
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Frontend | React.js, JavaScript, Vite |
| Backend | Python, FastAPI |
| AI / ML | PyTorch |
| Computer Vision | OpenCV |
| Landmark Extraction | MediaPipe, if required |
| Model Integration | Hugging Face / Pretrained Models |
| API Documentation | FastAPI Swagger UI |
| Speech Output | Web Speech API |

## 📁 Project Structure

The intended modular architecture:

```text
Signova-AI/
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── assets/
│       ├── components/
│       │   ├── camera/
│       │   ├── recognition/
│       │   ├── translation/
│       │   ├── speech/
│       │   ├── layout/
│       │   └── common/
│       ├── pages/
│       ├── hooks/
│       ├── services/
│       ├── context/
│       ├── utils/
│       └── styles/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
│
├── ai/
│   ├── preprocessing/
│   ├── pose_extraction/
│   ├── inference/
│   ├── model_adapters/
│   ├── postprocessing/
│   └── config/
│
├── models/
│   └── README.md
│
├── datasets/
│   └── README.md
│
├── scripts/
├── tests/
├── docs/
│
├── .env.example
├── .gitignore
└── README.md
```

*The structure above represents the intended architecture. Update it to match the actual repository as development progresses.*

## ⚙️ Getting Started

### Prerequisites

- Python 3.11+
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Signova-AI
```

Replace the repository URL with your actual GitHub URL.

### 2. Backend Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Start the backend:

```bash
uvicorn app.main:app --app-dir backend --reload
```

### 3. Access API Documentation

Once the backend is running:

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI: http://127.0.0.1:8000/openapi.json
- Health Check: http://127.0.0.1:8000/api/v1/health

### 4. Frontend Setup

Once the frontend has been configured:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL shown in your terminal.

## 🧠 AI Model Strategy

Signova AI is designed to integrate pretrained Indian Sign Language recognition models rather than train a model from scratch for the initial prototype.

The model integration process includes:

1. Inspecting pretrained model architecture and compatibility.
2. Understanding the original dataset and preprocessing pipeline.
3. Preparing compatible model inputs.
4. Loading pretrained checkpoints.
5. Running standalone inference.
6. Evaluating prediction quality before integrating the model into the application.

**Important:** Isolated sign recognition and continuous sign language translation are different tasks. Continuous sentence-level translation is a future capability and should not be claimed until implemented and evaluated.

## 🚀 Development Roadmap

| Phase | Focus | Status |
|---|---|---|
| Phase 0 | Project Setup | In progress |
| Phase 1 | Backend Foundation | In progress |
| Phase 2 | Pretrained Model Integration | Planned |
| Phase 3 | AI Pipeline | Planned |
| Phase 4 | Backend-AI Integration | Planned |
| Phase 5 | Frontend Development | Planned |
| Phase 6 | Live Recognition | Planned |
| Phase 7 | Speech Output | Planned |
| Phase 8 | Testing and Demo | Planned |

## 🔮 Future Improvements

- Expanded ISL vocabulary.
- Dynamic sign recognition.
- Continuous ISL sentence translation.
- Improved prediction accuracy.
- Better real-time performance.
- Recognition history and user feedback.

## 🤝 Contribution

Contributions, ideas, and feedback are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Test your changes.
5. Submit a pull request.

## ⚠️ Disclaimer

Signova AI is an experimental development project.

Its recognition capabilities will depend on the pretrained models, datasets, and evaluation results used during development. It should not yet be considered a replacement for qualified human interpreters.

## 📄 License

Add the appropriate license after reviewing the licenses of the pretrained models, datasets, and other dependencies used in this project.

---

<p align="center">
  **Signova AI**
  <br/>
  Making communication more accessible, one sign at a time. 🤟
</p>