# Deployment Guide

**Project:** Signova AI  
**Document:** Deployment Guide  
**File:** `docs/DEPLOYMENT.md`  
**Status:** Development Guidelines  
**Version:** 1.0

---

## 1. Purpose

This document defines how to run, configure, and deploy Signova AI in development and production environments.

Signova AI uses a React frontend, Python FastAPI backend, and pretrained machine learning models for Indian Sign Language recognition.

The deployment architecture must support:

- React frontend hosting.
- FastAPI backend hosting.
- AI model loading and inference.
- Camera-based sign recognition.
- Optional real-time WebSocket communication.
- Optional Sarvam AI speech integration.
- Optional PostgreSQL database.
- Secure environment configuration.

The deployment process should remain simple and cost-conscious, especially during the hackathon prototype.

## 2. Deployment Architecture

### 2.1 Recommended Architecture

```text
                  USER
                   |
                   v
          React + Vite Frontend
                   |
             HTTPS / WSS
                   |
                   v
            FastAPI Backend
                   |
          +--------+---------+
          |                  |
          v                  v
     AI Inference       Sarvam AI API
          |
          v
    Pretrained ISL Model
          |
          v
    Recognition Result
          |
          v
       Frontend
```

Optional services:

- PostgreSQL for persistent application data.
- Authentication provider for account management.
- Cloud storage only if a future feature requires persistent file storage.

### 2.2 Deployment Components

| Component | Technology | Purpose |
|---|---|---|
| Frontend | React + Vite | User interface |
| Frontend hosting | Vercel | Hosting frontend application |
| Backend | Python + FastAPI | API and application logic |
| Backend hosting | Render or compatible host | Running Python backend |
| ML framework | PyTorch | Model inference |
| Computer vision | OpenCV / MediaPipe | Input processing when required by the selected model |
| Speech services | Sarvam AI | Supported speech and language features |
| Database | PostgreSQL | Optional persistent storage |
| Model storage | Backend filesystem or approved model storage | Pretrained model files |

The frontend and backend are deployed separately.

## 3. Deployment Environments

Signova AI should support three environments.

| Environment | Purpose | Configuration |
|---|---|---|
| Development | Local development | Local frontend and backend |
| Testing | Automated and manual testing | Test configuration and test database |
| Production | Public application | Secure hosting and production credentials |

Never use production credentials in local development or automated tests.

## 4. Prerequisites

Install the required tools before starting deployment.

### 4.1 Required Software

| Software | Purpose |
|---|---|
| Node.js | Frontend development and build |
| npm | JavaScript package management |
| Python 3.11 | Backend and ML development |
| pip | Python package management |
| Git | Source control |
| GitHub account | Repository hosting |
| Vercel account | Frontend deployment |
| Backend hosting account | FastAPI deployment |

Optional:

- PostgreSQL.
- NVIDIA CUDA environment for GPU inference.
- Docker, only if the deployment platform or workflow requires it.

### 4.2 Verify Installation

```bash
node --version
npm --version
python --version
git --version
```

Confirm that the installed versions satisfy the project's dependency requirements.

The frontend and backend should use consistent versions across local development and deployment.

## 5. Project Structure

Recommended deployment-related structure:

```text
signova-ai/
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── websocket/
│   │   └── main.py
│   │
│   ├── requirements.txt
│   └── tests/
│
├── ml/
│   ├── checkpoints/
│   ├── configs/
│   ├── inference/
│   └── preprocessing/
│
├── docs/
├── .env.example
├── .gitignore
├── AGENTS.md
└── README.md
```

This is a recommended structure. Keep the actual deployment configuration aligned with the existing repository.

## 6. Local Development Setup

Before deploying, verify that the application runs correctly on a local machine.

### 6.1 Clone Repository

```bash
git clone YOUR_REPOSITORY_URL
cd signova-ai
```

Replace the placeholder with the actual repository URL.

### 6.2 Frontend Setup

Open a terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server normally runs at:

```text
http://localhost:5173
```

Verify:

- Frontend loads successfully.
- Navigation works.
- UI components render correctly.
- Camera permission works.
- API requests use the correct backend URL.

### 6.3 Backend Setup

Open another terminal:

```bash
cd backend
python -m venv .venv
```

Activate the environment.

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

API documentation during development:

```text
http://127.0.0.1:8000/docs
```

Do not expose interactive API documentation publicly unless it is intentionally required.

### 6.4 Verify Backend Health

If the backend implements a health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected example:

```json
{
  "status": "healthy"
}
```

This is an illustrative response. Match it to the actual backend implementation.

### 6.5 Verify Frontend and Backend Communication

Confirm that:

- Frontend API requests reach FastAPI.
- CORS permits the local frontend origin.
- API errors are handled properly.
- The selected model loads successfully.
- Inference returns the expected response format.

Do not proceed to production deployment until the core application works locally.

## 7. Environment Configuration

Environment variables store configuration separately from source code.

### 7.1 Backend Environment Variables

Create `backend/.env` for local development.

Example:

```env
APP_ENV=development
DEBUG=true

HOST=127.0.0.1
PORT=8000

FRONTEND_URL=http://localhost:5173
ALLOWED_ORIGINS=http://localhost:5173

MODEL_PATH=../ml/checkpoints/model.pt
MODEL_DEVICE=cpu

SARVAM_API_KEY=your_private_sarvam_api_key

DATABASE_URL=
```

Only configure variables that the backend actually reads.

### 7.2 Frontend Environment Variables

Create `frontend/.env.local`.

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

If the application uses WebSockets, configure the corresponding WebSocket URL or derive it safely from the backend configuration.

### 7.3 Environment Security

- Never commit `.env` files.
- Keep `.env.example` in the repository.
- Do not place private keys in `VITE_` variables.
- Do not expose database credentials in frontend code.
- Use the hosting provider's secret configuration for production.
- Rotate credentials if they are accidentally exposed.

## 8. Frontend Deployment on Vercel

Vercel can host the React + Vite frontend.

### 8.1 Prepare Frontend

From the repository root:

```bash
cd frontend
npm install
npm run build
```

Vite normally creates:

```text
frontend/dist/
```

Verify that the production build completes without errors.

### 8.2 Push Code to GitHub

```bash
git add .
git commit -m "Prepare frontend for deployment"
git push origin main
```

Use your actual branch name if it is different.

### 8.3 Create Vercel Project

1. Open https://vercel.com/
2. Sign in with GitHub.
3. Select **Add New Project**.
4. Import the Signova AI repository.
5. Configure the frontend project.

Recommended settings:

| Setting | Value |
|---|---|
| Framework | Vite |
| Root Directory | `frontend` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Install Command | `npm install` |

Check the actual project configuration before deployment.

### 8.4 Configure Frontend Environment Variables

In Vercel project settings, add:

```env
VITE_API_BASE_URL=https://your-backend-domain.com
```

Replace the example with the deployed FastAPI URL.

Important:

- Vite environment variables are included in the frontend build.
- Only public configuration belongs in frontend variables.
- Never add Sarvam API keys or database passwords here.

Redeploy after changing production environment variables.

### 8.5 Deploy

Select **Deploy**.

After deployment, verify:

- Homepage loads.
- React routes work.
- Assets load correctly.
- API requests use the production backend.
- Camera permission works on the HTTPS domain.
- No local development URLs remain in production code.

### 8.6 React Router Configuration

If the application uses React Router, configure Vercel rewrites so that client-side routes resolve to the frontend entry point.

Example `frontend/vercel.json`:

```json
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

Test direct navigation and page refresh on nested routes.

## 9. Backend Deployment

The FastAPI backend can be hosted on Render or another platform that supports Python applications.

### 9.1 Backend Requirements

Before deployment:

- Ensure `requirements.txt` contains required production packages.
- Ensure the application entry point is correct.
- Confirm the model path is configurable.
- Remove development-only settings.
- Verify API and WebSocket functionality.
- Confirm the model can load within the host's available resources.

### 9.2 Render Deployment

Open:

https://render.com/

Steps:

1. Sign in using GitHub.
2. Select **New**.
3. Choose **Web Service**.
4. Connect the Signova AI repository.
5. Configure the backend service.

Recommended configuration:

| Setting | Value |
|---|---|
| Runtime | Python |
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

The start command assumes that `backend/app/main.py` defines the FastAPI application as `app`.

If the application uses another module path, update the command accordingly.

### 9.3 Backend Environment Variables

Configure these in the hosting provider's environment settings:

```env
APP_ENV=production
DEBUG=false

SARVAM_API_KEY=your_private_sarvam_api_key

FRONTEND_URL=https://your-frontend-domain.vercel.app
ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app

MODEL_PATH=/app/ml/checkpoints/model.pt
MODEL_DEVICE=cpu
```

These are illustrative values. Configure actual paths and variable names according to the backend implementation.

Add `DATABASE_URL` only when PostgreSQL is enabled.

### 9.4 Port Configuration

Production hosting platforms generally provide a dynamic port.

The application must listen on:

```text
0.0.0.0
```

And use the port supplied by the hosting platform.

Do not hardcode port `8000` for production if the platform requires a dynamic port.

### 9.5 Verify Backend Deployment

After deployment:

1. Open the backend URL.
2. Check the health endpoint.
3. Verify API documentation if enabled.
4. Test a simple API request.
5. Test model loading.
6. Test a recognition request.
7. Test CORS from the deployed frontend.
8. Test WebSocket communication if implemented.

Do not treat a successful health check as proof that model inference works.

## 10. AI Model Deployment

The pretrained ISL model is one of the most important deployment dependencies.

A backend can start successfully while the model still fails to load.

### 10.1 Model Deployment Options

| Option | Description | Suitable Use |
|---|---|---|
| Included with deployment | Model files are available in the backend build | Small checkpoints |
| Model storage download | Backend downloads a verified model at startup | Larger checkpoints |
| Persistent disk | Model files remain on mounted storage | Hosts supporting persistent storage |
| Dedicated inference service | Model runs separately from the main API | Larger workloads |

Choose the simplest option that supports the selected model and hosting environment.

### 10.2 Model File Management

Recommended layout:

```text
ml/
├── checkpoints/
│   ├── .gitkeep
│   └── model.pt
│
├── configs/
├── inference/
└── preprocessing/
```

Do not assume that an empty Git-tracked folder will contain the actual checkpoint during deployment.

If the checkpoint is stored outside Git:

- Download it from a trusted source.
- Verify its checksum when available.
- Validate the file before loading.
- Store it in the configured model directory.
- Handle missing or corrupted files safely.

Do not expose private model-storage credentials.

### 10.3 Model Loading

Load the model once during application startup or through a controlled model lifecycle.

Avoid loading the model for every request.

Example:

```python
from pathlib import Path
import torch

def load_model(model_path: str, device: str):
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path.name}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    return checkpoint
```

This is only an illustrative checkpoint-loading example.

Actual loading must follow the selected model's architecture, checkpoint structure, and preprocessing requirements. Do not assume that every `.pt` file can be loaded this way.

### 10.4 CPU and GPU Deployment

Model deployment depends on the hosting hardware.

| Deployment | Advantages | Limitations |
|---|---|---|
| CPU | Easier and often cheaper to host | Inference may be slower |
| GPU | Better performance for suitable models | Higher cost and limited availability |
| Local GPU | Useful for development and demos | Requires the user's machine |
| Managed inference | Less infrastructure management | Provider costs and external dependencies |

For the initial prototype, test CPU inference first if the model supports it.

Use GPU hosting only when measured performance requires it.

### 10.5 Model Resource Requirements

Measure:

- Checkpoint size.
- Model loading time.
- Peak RAM usage.
- CPU/GPU utilisation.
- Inference latency.
- Concurrent inference capacity.
- Video frame-processing cost.

Do not select a hosting plan based only on the checkpoint file size.

### 10.6 Important Model Limitation

A model trained for static alphabet recognition must not be deployed or presented as a continuous ISL sentence translation model.

The production interface must clearly communicate the actual recognition capability of the deployed model.

## 11. Camera and HTTPS Configuration

Browser camera access generally requires a secure context.

Production requirements:

- Deploy the frontend using HTTPS.
- Request camera access after a user action.
- Handle permission denial.
- Stop the camera stream when recognition ends.
- Avoid unnecessary camera uploads.
- Ensure backend endpoints accept only the required media types.

For local development, localhost is generally treated as a secure context by modern browsers.

## 12. WebSocket Deployment

If real-time recognition uses WebSockets, verify that the backend hosting provider supports them.

### 12.1 Production Connection

Example:

```text
wss://your-backend-domain.com/api/v1/ai/predict/stream
```

### 12.2 Requirements

- Use `wss://` in production.
- Validate allowed origins.
- Authenticate protected connections.
- Limit message sizes.
- Validate incoming data.
- Handle disconnections.
- Release processing resources.
- Apply connection timeouts.
- Test reverse-proxy and hosting timeout settings.

Do not assume WebSocket support or long-running connection support without checking the selected hosting provider's current plan and documentation.

## 13. Sarvam AI Deployment

Sarvam AI is an optional external integration for supported speech and language functionality.

### 13.1 Configuration

Store the API key on the backend:

```env
SARVAM_API_KEY=your_private_sarvam_api_key
```

The frontend must call the Signova AI backend instead of calling Sarvam with a private key.

### 13.2 Production Requirements

- Verify the API key is valid.
- Handle authentication failures.
- Handle provider rate limits.
- Configure request timeouts.
- Avoid unlimited retries.
- Handle provider downtime.
- Explain external audio processing to users.
- Review current provider usage limits and pricing.

### 13.3 Optional Feature Behaviour

If Sarvam AI is unavailable:

- Sign recognition should continue working if it does not depend on Sarvam.
- Speech functionality should display a clear error.
- The frontend should not crash.
- Users should be able to retry when the service becomes available.

## 14. Database Deployment

PostgreSQL is optional.

Do not add a production database simply because the architecture includes one.

Use PostgreSQL when persistent features such as accounts, conversation history, preferences, or saved learning progress are implemented.

### 14.1 Production Database

Before deployment:

- Create a production database.
- Create a dedicated application user.
- Configure a secure database URL.
- Restrict network access.
- Use encrypted connections.
- Apply migrations.
- Configure backups.

### 14.2 Database Migrations

If SQLAlchemy and Alembic are used:

```bash
alembic upgrade head
```

Run migrations through a controlled deployment step.

Do not use automatic table creation as a replacement for proper production migrations.

### 14.3 Database Backups

- Enable backups where available.
- Restrict backup access.
- Test restoration.
- Define retention periods.
- Avoid storing raw media in the database.

## 15. Production API Configuration

Example API configuration:

```text
Production API

GET    /health

POST   /api/v1/ai/predict/image
POST   /api/v1/ai/predict/video

WS     /api/v1/ai/predict/stream
```

These endpoints are illustrative. Use the routes actually implemented in `docs/API_DOCUMENTATION.md`.

Before deployment, verify:

- Request and response schemas.
- Authentication requirements.
- File size limits.
- Error responses.
- Processing timeouts.
- Rate limits.
- CORS settings.

## 16. Production Build and Testing

### 16.1 Frontend Build

```bash
cd frontend
npm ci
npm run lint
npm run test
npm run build
```

Only run commands that are defined in the project's `package.json`.

If a test script has not yet been configured, add it before relying on it in CI.

### 16.2 Backend Testing

```bash
cd backend
python -m pytest
```

Run linting if configured:

```bash
ruff check .
```

### 16.3 AI Model Testing

Verify:

- Model loads successfully.
- Correct preprocessing is applied.
- Input dimensions match the model.
- Class labels match the expected output.
- Inference returns valid results.
- Unknown or low-confidence signs are handled.
- CPU and GPU behaviour is tested where supported.

### 16.4 End-to-End Testing

Test the deployed application:

1. Open the production frontend.
2. Navigate to the recognition page.
3. Grant camera permission.
4. Start recognition.
5. Confirm the camera stream is active.
6. Perform a supported sign.
7. Verify the recognition result.
8. Stop the camera.
9. Verify camera tracks are released.
10. Test speech functionality if enabled.
11. Test error handling when the backend is unavailable.

Do not report a feature as production-ready unless it has been tested against the deployed services.

## 17. GitHub Actions Deployment Workflow

GitHub Actions can automate quality checks before deployment.

Example workflow:

`.github/workflows/ci.yml`

```yaml
name: Signova AI CI

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  frontend:
    runs-on: ubuntu-latest

    defaults:
      run:
        working-directory: frontend

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 24
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Run lint
        run: npm run lint

      - name: Run tests
        run: npm run test -- --run

      - name: Build frontend
        run: npm run build

  backend:
    runs-on: ubuntu-latest

    defaults:
      run:
        working-directory: backend

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest ruff

      - name: Run lint
        run: ruff check .

      - name: Run tests
        run: python -m pytest
```

Important:

- Confirm that the referenced npm scripts exist.
- Confirm that the test command matches the configured test runner.
- Pin GitHub Actions to trusted versions or commit SHAs as appropriate.
- Add secrets only when a workflow genuinely needs them.
- Never print secret values in logs.
- This example runs CI checks only. It does not automatically deploy the application.

## 18. Deployment Checklist

### Before Deployment

- [ ] Frontend works locally.
- [ ] Backend works locally.
- [ ] Model loads successfully.
- [ ] Inference has been tested.
- [ ] API routes have been verified.
- [ ] Environment variables are documented.
- [ ] No private credentials are committed.
- [ ] `.gitignore` is configured.
- [ ] Production frontend API URL is ready.
- [ ] CORS configuration is ready.
- [ ] Dependencies are installed correctly.
- [ ] Required tests pass.

### Frontend Deployment

- [ ] Repository is connected to Vercel.
- [ ] Root directory is set to `frontend`.
- [ ] Build command is correct.
- [ ] Output directory is `dist`.
- [ ] Production API URL is configured.
- [ ] HTTPS works.
- [ ] Client-side routes work.
- [ ] Camera permission works.

### Backend Deployment

- [ ] Repository is connected to backend hosting.
- [ ] Python version is supported.
- [ ] Build command is correct.
- [ ] Start command is correct.
- [ ] Port configuration is correct.
- [ ] Production secrets are configured.
- [ ] Model files are available.
- [ ] Health endpoint works.
- [ ] Inference works.
- [ ] CORS is restricted.
- [ ] WebSocket support is verified if required.

### After Deployment

- [ ] Frontend connects to the backend.
- [ ] API requests work.
- [ ] Camera processing works.
- [ ] Recognition results are correct for supported inputs.
- [ ] WebSocket works if enabled.
- [ ] Sarvam integration works if enabled.
- [ ] Errors are handled correctly.
- [ ] Logs do not expose sensitive information.
- [ ] Resource usage is acceptable.
- [ ] Production application has been manually tested.

## 19. Troubleshooting

| Problem | Possible Cause | Solution |
|---|---|---|
| Frontend build fails | Dependency or configuration issue | Check npm output and package versions |
| Backend fails to start | Incorrect module path or missing dependency | Verify the start command and requirements |
| Frontend cannot reach backend | Incorrect API URL | Check `VITE_API_BASE_URL` |
| CORS error | Frontend origin not allowed | Update backend CORS configuration |
| Model not found | Incorrect checkpoint path | Verify model storage and `MODEL_PATH` |
| Model loading fails | Incompatible checkpoint or dependency | Check model documentation and loading code |
| Out of memory | Model exceeds hosting capacity | Measure memory usage and optimise or change hosting |
| Inference is slow | CPU or resource limitations | Profile the model and consider suitable hardware |
| Camera unavailable | Permission or insecure context | Use HTTPS and check browser permissions |
| WebSocket disconnects | Hosting timeout or proxy configuration | Check provider WebSocket support and timeout settings |
| Sarvam API fails | Invalid key or provider limits | Check backend credentials and provider response |
| Database connection fails | Incorrect URL or network access | Verify credentials and database connectivity |
| Nested route returns 404 | Missing SPA rewrite | Configure Vercel rewrites |

## 20. Cost and Infrastructure Guidelines

Keep the first deployment simple.

Recommended initial approach:

| Component | Initial Approach |
|---|---|
| Frontend | Vercel |
| Backend | Compatible Python hosting service |
| Model | Verified pretrained checkpoint |
| Inference | CPU initially, if performance is acceptable |
| Database | Local or optional managed PostgreSQL |
| Speech | Sarvam AI only when needed |
| File storage | Temporary processing only |
| Monitoring | Hosting provider logs and basic health checks |

Before selecting a free or paid hosting plan, verify its current:

- Usage limits.
- RAM and CPU.
- Storage limits.
- Sleep and cold-start behaviour.
- GPU availability.
- Request duration.
- WebSocket support.
- Bandwidth limits.
- Pricing.

Free plans and provider features can change. Do not assume that a free plan can reliably host a large ML model or support a live demo.

## 21. Rollback and Recovery

Maintain a way to recover from unsuccessful deployments.

Requirements:

- Keep stable source-control commits.
- Record working model versions.
- Keep deployment configuration documented.
- Preserve database backups when persistent data is used.
- Avoid destructive database migrations.
- Test important changes before production release.
- Revert to a known working deployment if a release breaks core functionality.

For model updates, keep track of the previous verified model version and its configuration.

## 22. Antigravity Deployment Instructions

When using Antigravity to configure or implement deployment:

1. Read `AGENTS.md`.
2. Read `docs/DEPLOYMENT.md`.
3. Read `docs/TECH_STACK.md`.
4. Read `docs/SYSTEM_ARCHITECTURE.md`.
5. Read `docs/API_DOCUMENTATION.md`.
6. Inspect the existing project structure.
7. Identify actual frontend and backend entry points.
8. Verify existing environment variable names.
9. Implement only the requested deployment configuration.
10. Do not invent existing endpoints, scripts, or file paths.
11. Do not expose private credentials.
12. Add or update deployment documentation.
13. Run available build and test commands.
14. Report actual deployment limitations.

### Required Deployment Report

After completing deployment work, report:

- Hosting provider.
- Frontend deployment URL, if deployed.
- Backend deployment URL, if deployed.
- Build status.
- Test results.
- Model loading status.
- API connectivity status.
- WebSocket status, if implemented.
- Environment variables configured, without revealing their values.
- Known issues and limitations.

Do not claim that the application is deployed unless the deployment has actually succeeded.

## 23. Final Deployment Policy

Signova AI should be deployed in small, verifiable steps.

Recommended deployment order:

1. Verify local frontend and backend.
2. Verify pretrained model loading and inference.
3. Deploy the FastAPI backend.
4. Test backend APIs.
5. Deploy the React frontend.
6. Connect frontend and backend.
7. Test camera recognition on HTTPS.
8. Configure WebSockets if required.
9. Integrate Sarvam AI if enabled.
10. Add PostgreSQL when persistent features are ready.
11. Perform production security and performance checks.

**Final principle:** A successful deployment means the complete application works in its target environment, not simply that the frontend website opens.