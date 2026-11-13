# AI Model Strategy

## 1. Overview

Signova AI uses computer vision and deep learning to recognise Indian Sign Language (ISL) gestures and convert recognised signs into meaningful text.

The AI system is designed to support a modular approach so that different recognition models can be tested, evaluated, and integrated without changing the entire application.

The initial objective is to establish a reliable ISL recognition pipeline using verified pretrained models. Future development will focus on dynamic sign recognition, sequence understanding, and continuous sign language interpretation.

**Important:** Static alphabet recognition, dynamic word recognition, and continuous sign language translation are different tasks. The project must identify its actual capabilities based on model evaluation rather than making unsupported claims.

---

## 2. AI System Objectives

The AI system should:

- Recognise supported Indian Sign Language gestures.
- Process camera images or video frames.
- Detect and classify signs using an appropriate pretrained model.
- Support dynamic sign recognition where a suitable model is available.
- Convert recognised signs into text.
- Provide confidence scores for predictions.
- Identify uncertain or unsupported gestures.
- Support future sentence construction from recognised sign sequences.
- Maintain a modular architecture for replacing or improving models.
- Run efficiently on available hardware.

---

## 3. Recognition Types

Signova AI will distinguish between three recognition levels.

### 3.1 Static Sign Recognition

Static recognition identifies a sign from a single image or video frame.

Examples:

- ISL alphabet recognition.
- Static hand gestures.
- Individual signs represented by a fixed hand position.

**Initial implementation goal:** Verify and integrate a suitable ISL alphabet or static gesture model.

**Limitation:** Static recognition cannot independently understand continuous sign language or sentence structure.

### 3.2 Dynamic Sign Recognition

Dynamic recognition analyses multiple frames to identify signs involving movement.

It may use:

- Video frame sequences.
- Hand landmark sequences.
- Spatial and temporal features.
- CNN, LSTM, Transformer, or 3D CNN architectures.

**Development goal:** Evaluate pretrained dynamic ISL recognition models and integrate a suitable model if its performance and compatibility are acceptable.

### 3.3 Continuous Sign Language Understanding

Continuous sign language understanding involves interpreting connected signs and converting them into meaningful sentences.

It may require:

- Continuous video processing.
- Sign boundary detection.
- Temporal sequence modelling.
- Context-aware recognition.
- Non-manual features such as facial expressions.
- ISL grammar and sentence structure.

**Future goal:** Research and develop a more complete continuous ISL interpretation pipeline.

This feature must not be represented as complete until it has been implemented and evaluated on appropriate data.

---

## 4. Model Selection Strategy

Signova AI will prioritise verified pretrained models before considering training a new model from scratch.

### 4.1 Model Selection Requirements

Every candidate model must be checked for:

| Requirement | What to verify |
|---|---|
| Language | Must support Indian Sign Language for the intended task |
| Recognition type | Static, dynamic, or continuous |
| Architecture | CNN, LSTM, Transformer, 3D CNN, or other |
| Checkpoint | Availability of trained model weights |
| Class labels | Complete and understandable label mapping |
| Input format | Image, video, or landmark sequence |
| Preprocessing | Image size, normalisation, frame count, landmark format |
| Dataset | Dataset source and collection details |
| Licence | Permission to use and modify the model |
| Dependencies | Compatible Python and ML libraries |
| Hardware | CPU, GPU, RAM, and VRAM requirements |
| Evaluation | Available metrics and testing methodology |

### 4.2 Approved Primary Model

Signova AI has selected a single pretrained model for Indian Sign Language translation:

- **Model Repository:** `manavdhamecha77/iSign-t5-pose-to-text`
- **Hugging Face URL:** https://huggingface.co/manavdhamecha77/iSign-t5-pose-to-text
- **Local Model Directory:** `D:\iSign-t5-pose-to-text` (Total disk footprint: ~15.23 GB)

**Important:** No alternative models will be searched, recommended, or downloaded. Signova AI will use this model exclusively. Earlier research candidates (Dhawal-17 ISL, signBridge, isl-swin3d-model) are superseded.

### 4.4 Selected Pretrained Model Technical Specification (F-011 Inspection)

The local directory `D:\iSign-t5-pose-to-text` has been completely inspected and verified.

#### 4.4.1 Actual Downloaded Checkpoint Variants & File Integrity

All six expected checkpoint variants exist in `D:\iSign-t5-pose-to-text`. All weight files and configs were validated as uncorrupted:

| Checkpoint Variant | Architecture | Feature Type | Pose Dimension | Base T5 | `model.safetensors` | `pose_encoder.pt` | Status |
|---|---|---|---|---|---|---|---|
| `t5-small` | T5-Small | Spatial | 300 | `t5-small` | 242,041,896 B (~242 MB) | 837,263 B (256x300 -> 512x256) | Verified Valid |
| `t5-small-motion` | T5-Small | Spatial + Motion | 600 | `t5-small` | 242,041,896 B (~242 MB) | 1,144,463 B (256x600 -> 512x256) | Verified Valid |
| `t5-base` | T5-Base | Spatial | 300 | `t5-base` | 891,644,712 B (~891 MB) | 1,100,431 B (256x300 -> 768x256) | Verified Valid |
| `t5-base-motion` | T5-Base | Spatial + Motion | 600 | `t5-base` | 891,644,712 B (~891 MB) | 1,407,631 B (256x600 -> 768x256) | Verified Valid (Folder named `t5-base-motion`, not `-v2`) |
| `t5-large` | T5-Large | Spatial | 300 | `t5-large` | 2,950,734,544 B (~2.95 GB) | 1,363,599 B (256x300 -> 1024x256) | Verified Valid |
| `t5-large-motion` | T5-Large | Spatial + Motion | 600 | `t5-large` | 2,950,734,544 B (~2.95 GB) | 1,670,799 B (256x600 -> 1024x256) | Verified Valid |

*Key Directory Observations:*
- Each variant subfolder contains `pose_encoder.pt` and a `t5/` subfolder with `config.json`, `generation_config.json`, and `model.safetensors`.
- The directory on disk is named `t5-base-motion`. The name `t5-base-motion-v2` appears in the author's README but does not exist on disk.
- Tokenizer files (`tokenizer.json`, `spiece.model`) are NOT stored in the checkpoint directories; official inference downloads the base T5 tokenizer from Hugging Face.
- Official `inference/` folder contains `inference.py`, `model.py`, `pose_utils.py`, and `requirements.txt`.
- Defect identified in official `inference/inference.py`: `MODEL_CONFIGS` dictionary omits `t5-base-motion` / `t5-base-motion-v2`.

#### 4.4.2 Model Architecture & Inference Pipeline

The system uses a 2-stage sequence-to-sequence translation architecture:

```text
Pose Sequence: (batch, seq_len, pose_dim)
       |
       v
PoseEncoder (MLP): Linear(pose_dim, 256) -> ReLU -> Dropout(0.1) -> Linear(256, d_model)
       |
       v
Pose Embeddings: (batch, seq_len, d_model)
       |
       v
T5 Encoder (Bidirectional Self-Attention across sequence)
       |
       v
T5 Decoder (Autoregressive Generation with Beam Search, num_beams=4, max_length=128)
       |
       v
English Sentence Text
```

- **Pose Dimension (`pose_dim`):**
  - 300 for spatial models (75 landmarks * 4 channels: x, y, z, confidence).
  - 600 for motion models (300 spatial + 300 temporal velocity deltas).
- **Official Preprocessing Workflow (`pose_utils.py`):**
  1. Landmark extraction: 33 pose landmarks + 21 left hand + 21 right hand = 75 points.
  2. Coordinate packing: `[x, y, z, confidence]` per landmark -> 300 values per frame.
  3. Feature dimension padding/truncation: padded/truncated to 300 dimensions.
  4. Sequence-level Z-Score Normalization: `(pose - mean) / std` computed across all frames in the segment.
  5. Motion feature addition (for motion checkpoints): `velocity[t] = pose[t] - pose[t-1]`, concatenated with pose -> 600 values per frame.
  6. Temporal padding/truncation: padded with zeros or truncated to `max_length = 500` frames (~16.6s at 30 fps).

#### 4.4.3 Webcam Compatibility Requirements

The model does NOT accept raw video frames or single static images:
- **MediaPipe Holistic Integration:** A live capture pipeline must run MediaPipe Holistic to extract `POSE_LANDMARKS` (33), `LEFT_HAND_LANDMARKS` (21), and `RIGHT_HAND_LANDMARKS` (21).
- **Format Mapping:**
  - Coordinates: Normalized (x, y) [0.0, 1.0], relative depth z, and confidence/visibility scores.
  - When hands are not detected, coordinates and confidence are set to 0.0.
  - Flattening order must strictly match: Pose (indices 0..32), Left Hand (0..20), Right Hand (0..20).
- **Normalization Constraint:** Z-score normalization cannot be performed on single frames in real time. Frames must be accumulated in a temporal buffer over a complete signing duration before normalization.

#### 4.4.4 Checkpoint Recommendation

- **Recommended Checkpoint:** `t5-small-motion` (with `t5-small` as fallback).
- **Rationale:**
  - **Size & Memory:** ~243 MB total footprint, fits comfortably within ~500 MB - 1 GB RAM.
  - **Motion Awareness:** Motion features (600 dims) are essential for distinguishing dynamic ISL signs where hand trajectory differentiates meaning.
  - **Latency:** Fast CPU execution (~150-300 ms per segment translation).
  - **Hardware Fit:** Runs reliably without CUDA on CPU, completely avoiding the 4 GB GPU VRAM constraint.

#### 4.4.5 Hardware Feasibility (CPU vs GPU)

- **Target Dev Environment:** Windows, Python 3.11, NVIDIA GPU (~4 GB VRAM), CPU inference fallback.
- **Current PyTorch Environment:** CPU-only (`torch 2.2.2+cpu`, `torch.cuda.is_available() = False`).
- **CPU Verification:** Confirmed via standalone execution on `t5-small` and `t5-small-motion`. Both checkpoints instantiate, load state dicts, and execute `generate()` on CPU cleanly.
- **GPU Feasibility:** `t5-small` and `t5-base` will fit in 4 GB VRAM once CUDA PyTorch is available; `t5-large` variants (~2.95 GB weights + activations) would risk OOM on a 4 GB VRAM GPU.

#### 4.4.6 Continuous Translation Scope & Limitations

- **What the Model Supports:** Bounded segment-to-sentence translation. Given a completed sequence of signs (up to 500 frames), it produces a grammatical English sentence.
- **What the Model Does NOT Support:**
  - Streaming word-by-word instantaneous translation.
  - Infinite streaming buffers.
  - Unsegmented continuous video without pause detection.
- **Buffer & Segment Requirement:** Webcam video must be captured in segments (e.g. 1 to 16 seconds). Signing start and stop boundaries must be detected (via hand activity threshold or user push-to-translate action) to avoid corrupting sequence z-score normalization with idle frames.
- **Translation Hallucination Risk:** Like all generative T5 models, inputs outside the iSign training distribution may produce plausible-sounding but factually incorrect English sentences.

### 4.3 Model Approval Process

A model can be approved for integration only after:

1. Repository source code has been inspected.
2. Model architecture has been identified.
3. Pretrained checkpoint availability has been confirmed.
4. Class labels have been inspected.
5. Preprocessing requirements have been documented.
6. Licence and dataset provenance have been reviewed.
7. Dependencies have been checked.
8. Local inference has been tested.
9. Predictions have been evaluated on suitable test samples.
10. Hardware and inference performance have been measured.

Do not select a model based only on its GitHub description or reported accuracy.

---

## 5. Recommended ML Technology

| Technology | Purpose |
|---|---|
| Python 3.11 | Main ML development language |
| PyTorch | Model loading and inference |
| OpenCV | Image and video processing |
| MediaPipe | Hand landmark extraction when required |
| NumPy | Numerical operations |
| Pandas | Dataset analysis |
| Scikit-learn | Evaluation and ML utilities |
| Hugging Face Hub | Model and checkpoint discovery |
| FastAPI | Model inference API |

Use the preprocessing method expected by the selected model. Do not introduce MediaPipe into a model pipeline unless the model requires or supports it.

---

## 6. AI Pipeline Architecture

The initial recognition pipeline follows this structure:

```text
Camera Input
     |
     v
Image / Video Capture
     |
     v
Frame Preprocessing
     |
     v
Model Input Preparation
     |
     v
Pretrained ISL Recognition Model
     |
     v
Prediction and Confidence Score
     |
     v
Confidence Validation
     |
     v
Prediction Smoothing
     |
     v
Recognised Sign Label
     |
     v
Text Output
     |
     v
Frontend Display
```

For dynamic recognition, the pipeline will process a sequence of frames instead of classifying a single frame.

```text
Video Stream
     |
     v
Frame Sequence Collection
     |
     v
Temporal Preprocessing
     |
     v
Dynamic Recognition Model
     |
     v
Predicted Word / Sign
     |
     v
Sequence Validation
     |
     v
Text Output
```

---

## 7. Model Loading and Management

Models must be managed separately from API endpoints and frontend components.

### 7.1 Model Manager

Create a model management service responsible for:

- Loading approved checkpoints.
- Initialising the correct model architecture.
- Loading class labels.
- Applying model-specific preprocessing.
- Running inference.
- Returning structured predictions.
- Handling model loading and inference errors.

### 7.2 Model Configuration

Each model must have its own configuration.

Example:

```yaml
model:
  name: isl_static_recognition
  version: "1.0"
  task: static_gesture_recognition

  architecture: null
  checkpoint_path: null
  labels_path: null

  input:
    type: image
    image_size: null
    normalization: null

  inference:
    confidence_threshold: 0.70
    device: auto
```

The values must be filled from the actual model documentation and source code. Do not assume that all models use the same input size or normalisation.

### 7.3 Model Registry

Maintain a record of each evaluated model.

| Field | Description |
|---|---|
| Model ID | Unique identifier |
| Model name | Model name |
| Version | Model version |
| Task | Static or dynamic recognition |
| Architecture | Model architecture |
| Checkpoint | Checkpoint location |
| Labels | Class mapping |
| Licence | Model licence |
| Evaluation | Measured test results |
| Status | Research, tested, approved, or rejected |

---

## 8. Preprocessing Strategy

Preprocessing must match the selected model's original training pipeline.

### 8.1 Image Preprocessing

Possible operations include:

- Read image from camera.
- Validate image dimensions.
- Convert colour channels when required.
- Resize to the expected input dimensions.
- Normalise pixel values.
- Convert image to the required tensor format.
- Add batch dimension.

Do not apply arbitrary transformations that change the model's expected input distribution.

### 8.2 Video Preprocessing

For dynamic recognition:

- Capture video frames.
- Sample frames at the required rate.
- Maintain the expected sequence length.
- Resize frames if required.
- Apply consistent normalisation.
- Prepare tensors in the expected temporal order.
- Handle incomplete or invalid sequences.

### 8.3 Landmark Preprocessing

If the selected model requires hand landmarks:

- Detect hand landmarks.
- Extract required coordinates.
- Apply the model's expected coordinate normalisation.
- Handle missing or partially detected hands.
- Maintain consistent landmark ordering.
- Prepare landmark sequences when required.

The same preprocessing rules must be used during both model evaluation and production inference.

---

## 9. Prediction and Confidence Handling

Each prediction should include:

- Predicted class.
- Confidence score.
- Model identifier.
- Processing time.
- Recognition status.

Example response:

```json
{
  "success": true,
  "prediction": {
    "label": "A",
    "confidence": 0.94
  },
  "model": {
    "name": "isl_static_recognition",
    "version": "1.0"
  },
  "processing_time_ms": 85
}
```

The example values are illustrative, not actual model results.

### 9.1 Confidence Threshold

A confidence threshold should be selected using validation data.

For example:

- High confidence: show the recognised sign.
- Low confidence: show an uncertain prediction.
- Unsupported or invalid input: return an appropriate error or unknown result.

A fixed threshold such as 0.70 is only an initial configuration example. It must be tuned for the selected model.

### 9.2 Unknown Sign Handling

The system should not force every input into a known class.

Unknown or uncertain predictions should be handled explicitly.

Possible reasons include:

- Unsupported sign.
- Poor lighting.
- Hand outside the camera frame.
- Motion blur.
- Incorrect camera angle.
- Low model confidence.
- Invalid input sequence.

### 9.3 Prediction Smoothing

For live recognition, repeated predictions can be stabilised using:

- Sliding window voting.
- Consecutive prediction checks.
- Temporal smoothing.
- Duplicate prediction suppression.

Smoothing must not hide meaningful changes in a sign sequence.

---

## 10. Dataset Strategy

Dataset quality is critical for reliable ISL recognition.

### 10.1 Dataset Requirements

A suitable dataset should contain:

- Correct ISL labels.
- Clear class definitions.
- Sufficient samples per class.
- Multiple signers.
- Different lighting conditions.
- Different backgrounds.
- Different camera angles.
- Appropriate image or video samples.
- Clear usage and distribution terms.

### 10.2 Dataset Sources to Investigate

Potential research sources include:

- Indian Sign Language datasets.
- AI4Bharat research and datasets.
- Hugging Face datasets.
- Academic ISL recognition datasets.
- Properly licensed and consented project-collected samples.

Every dataset must be checked for language, labels, quality, licence, consent, and suitability for the target recognition task.

### 10.3 Dataset Preparation

The dataset pipeline should include:

1. Dataset inspection.
2. Label verification.
3. Missing or corrupted sample detection.
4. Duplicate detection.
5. Class distribution analysis.
6. Data quality checks.
7. Train, validation, and test split.
8. Preprocessing.
9. Dataset documentation.

### 10.4 Data Leakage Prevention

Samples from the same signer or recording session should not be unintentionally distributed across training and testing sets when evaluating generalisation to unseen signers.

Where possible, use signer-independent splits.

Do not use test data to tune thresholds, preprocessing, or model parameters.

---

## 11. Model Evaluation Strategy

Every model must be evaluated before it is used in the application.

### 11.1 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correctness of positive predictions |
| Recall | Ability to identify actual classes |
| F1-score | Balance between precision and recall |
| Macro F1 | Average class performance without weighting by class size |
| Confusion Matrix | Shows class-level prediction errors |
| Inference Latency | Measures prediction speed |
| Model Size | Measures storage requirements |

### 11.2 Evaluation Requirements

The evaluation process should:

- Use a separate test dataset.
- Report overall and per-class performance.
- Identify weakly recognised classes.
- Check class imbalance.
- Measure inference time.
- Test different lighting and backgrounds.
- Evaluate different signers.
- Record hardware details.
- Compare models using the same test conditions.

### 11.3 Model Comparison

Maintain a model comparison report.

| Model | Task | Accuracy | Macro F1 | Latency | Decision |
|---|---|---|---|---|---|
| Candidate A | Static ISL | Pending | Pending | Pending | Not evaluated |
| Candidate B | Dynamic ISL | Pending | Pending | Pending | Not evaluated |

Do not enter estimated or repository-reported results as independently measured project results.

---

## 12. FastAPI Integration

The ML inference pipeline will be exposed through the FastAPI backend.

Suggested endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/ai/health` | GET | Check model service health |
| `/api/v1/ai/models` | GET | List approved models |
| `/api/v1/ai/predict/image` | POST | Static image prediction |
| `/api/v1/ai/predict/video` | POST | Video sequence prediction |
| `/api/v1/ai/predict/stream` | WebSocket | Live recognition, when implemented |

### 12.1 Backend Responsibilities

- Validate incoming data.
- Load approved models.
- Run preprocessing.
- Execute model inference.
- Validate confidence.
- Return structured predictions.
- Handle errors.
- Measure processing time.

### 12.2 Error Handling

The backend should handle:

- Missing image or video.
- Invalid file format.
- Unsupported input dimensions.
- Model loading failure.
- Inference failure.
- Invalid frame sequence.
- Missing camera frames.
- Request timeout.

Never return a fabricated prediction when inference fails.

---

## 13. Hardware and Performance Strategy

The development environment may have limited GPU memory. Models must be tested before selecting deployment hardware.

### 13.1 Local Development

- Start with a small verified pretrained model.
- Test CPU inference.
- Test GPU inference where supported.
- Measure RAM and VRAM usage.
- Avoid loading unnecessary models simultaneously.
- Avoid downloading large checkpoints without inspecting their size.

### 13.2 Optimisation

Potential optimisation methods:

- Batch processing where appropriate.
- Efficient frame sampling.
- Model evaluation in inference mode.
- Appropriate input resolution.
- CPU or GPU selection based on measured performance.
- Model quantisation where supported.
- ONNX export where compatible.

Optimisation must preserve acceptable recognition quality.

### 13.3 Deployment

Before deployment, verify:

- Model checkpoint size.
- Available RAM.
- GPU availability.
- CPU inference speed.
- Request timeout limits.
- Concurrent inference capacity.
- WebSocket support if required.
- Model storage and download limits.

Do not assume that a free hosting service can run a large PyTorch model reliably.

---

## 14. Sarvam AI Integration

Sarvam AI will support speech and language features around the sign recognition system.

### Planned uses

- Text-to-speech for recognised text.
- Speech-to-text for reverse communication.
- Translation between supported languages.
- Spoken output in supported Indian languages.

### Responsibility Separation

| Component | Responsibility |
|---|---|
| ISL recognition model | Recognise visual signs |
| Computer vision pipeline | Process images and video |
| Sentence construction | Organise recognised signs into text |
| Sarvam STT | Convert supported speech input into text |
| Sarvam translation | Translate supported language text |
| Sarvam TTS | Convert text into speech |

Sarvam AI is not the ISL recognition model. Speech and language APIs must not be presented as automatically understanding sign gestures or generating sign animations.

API keys must remain on the backend and must never be exposed in frontend code.

---

## 15. Model Security and Privacy

Camera and video data may contain sensitive personal information.

Signova AI should:

- Request camera permission before accessing the camera.
- Process camera input only for the intended feature.
- Avoid storing raw video unless explicitly required.
- Avoid uploading camera footage without clear user consent.
- Protect stored datasets and model files.
- Keep API keys in environment variables.
- Validate uploaded files.
- Restrict file size and accepted formats.
- Avoid logging personal video or image data.
- Document dataset consent and licence information.

---

## 16. Future AI Improvements

Potential future improvements include:

### 16.1 Expanded ISL Vocabulary

- Add more verified ISL signs.
- Improve class balance.
- Include more signers.
- Improve robustness across environments.

### 16.2 Dynamic Recognition

- Recognise movement-based signs.
- Improve temporal modelling.
- Reduce repeated and incorrect predictions.
- Evaluate longer video sequences.

### 16.3 Sentence Construction

- Combine recognised signs into readable text.
- Reduce duplicate words.
- Handle uncertain predictions.
- Investigate ISL grammar and context.

### 16.4 Non-Manual Features

Investigate facial expressions and other non-manual cues where they are linguistically relevant and suitable datasets are available.

### 16.5 Continuous ISL Interpretation

Research models and datasets that support continuous signing, segmentation, and context-aware interpretation.

This remains a future research goal until validated with suitable ISL data and evaluation.

---

## 17. Implementation Phases

| Phase | Work | Expected Output |
|---|---|---|
| 1. Model Research | Find and inspect candidate ISL models | Model research report |
| 2. Model Verification | Verify checkpoints, labels, licences, and preprocessing | Approved candidate |
| 3. Local Testing | Run sample inference on local hardware | Working inference prototype |
| 4. Evaluation | Test on suitable unseen samples | Evaluation report |
| 5. ML Pipeline | Build reusable preprocessing and prediction services | ML inference module |
| 6. Backend Integration | Connect ML inference to FastAPI | Working prediction API |
| 7. Frontend Integration | Connect camera input and prediction output | Recognition interface |
| 8. Real-Time Processing | Add frame streaming and prediction smoothing | Live recognition prototype |
| 9. Dynamic Recognition | Integrate and evaluate a suitable sequence model | Dynamic recognition prototype |
| 10. Optimisation | Improve latency, reliability, and deployment readiness | Optimised system |

Each phase must be tested before moving to the next dependent phase.

---

## 18. Suggested ML Folder Structure

```text
ml/
├── configs/
│   ├── static_model.yaml
│   └── dynamic_model.yaml
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── preprocessing/
│   ├── image_preprocessing.py
│   ├── video_preprocessing.py
│   └── landmark_preprocessing.py
│
├── models/
│   ├── model_loader.py
│   ├── model_registry.py
│   └── model_manager.py
│
├── inference/
│   ├── predictor.py
│   ├── static_predictor.py
│   ├── dynamic_predictor.py
│   └── prediction_validator.py
│
├── evaluation/
│   ├── evaluate_model.py
│   ├── metrics.py
│   └── confusion_matrix.py
│
├── training/
│   └── README.md
│
├── checkpoints/
│   └── .gitkeep
│
└── notebooks/
    └── README.md
```

Only create modules when they are needed. Do not create empty implementations that appear functional.

---

## 19. Development Rules for AI Models

All AI development must follow these rules:

1. Verify a pretrained model before integrating it.
2. Use ISL models for the primary recognition task.
3. Never treat ASL recognition as ISL recognition.
4. Inspect the original preprocessing pipeline.
5. Keep model-specific preprocessing separate.
6. Do not invent class labels or predictions.
7. Do not report unmeasured accuracy.
8. Do not train a new model before establishing a baseline.
9. Keep checkpoints out of Git unless there is a clear reason.
10. Record licences and dataset sources.
11. Evaluate using unseen samples and signer-independent splits where possible.
12. Measure latency and resource usage.
13. Keep inference independent from frontend code.
14. Handle uncertain and unsupported signs.
15. Update the model documentation whenever the model or checkpoint changes.

---

## 20. Instructions for Antigravity

Before performing any AI or ML development task:

1. Read `AGENTS.md`.
2. Read `docs/PRD.md`.
3. Read `docs/PROJECT_OVERVIEW.md`.
4. Read `docs/FEATURES.md`.
5. Read `docs/TECH_STACK.md`.
6. Read `docs/SYSTEM_ARCHITECTURE.md`.
7. Read this `docs/AI_MODEL_STRATEGY.md`.
8. Inspect the existing ML code before making changes.

### Implementation Requirements

- Use Python 3.11.
- Use PyTorch for compatible model inference.
- Use OpenCV for image and video processing.
- Use MediaPipe only when required by the selected model.
- Keep preprocessing and inference modular.
- Verify checkpoints before downloading large files.
- Do not train a model without approval.
- Do not replace an approved model without evaluation.
- Write tests for preprocessing and inference.
- Record actual evaluation results.
- Update relevant documentation.
- Do not create fake predictions or placeholder metrics.

### Completion Report

After every ML task, report:

- What was implemented.
- Files created or modified.
- Model and checkpoint used.
- Dependencies added.
- Tests executed and their actual results.
- Performance measurements, if available.
- Known limitations.
- Recommended next step.

---

## 21. Final Strategy

The immediate priority for Signova AI is to establish a working and verified ISL recognition baseline.

The development order is:

1. Research and verify suitable ISL models.
2. Test pretrained model inference locally.
3. Evaluate recognition quality.
4. Build a reusable ML inference pipeline.
5. Integrate the pipeline with FastAPI.
6. Connect camera input to the frontend.
7. Add speech and language features using Sarvam AI.
8. Expand towards dynamic recognition and continuous ISL understanding.

**Success criterion:** Signova AI must demonstrate reliable recognition for its explicitly supported ISL classes, report measured performance honestly, and clearly communicate when a sign cannot be recognised.