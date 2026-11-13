1. Product overview
- Product: Signova AI
- Document: Product Requirements Document (PRD)
- Version: 2.0
- Purpose: Define the product vision, scope, features, requirements, architecture direction, and acceptance criteria.
Vision: Build an accessible, practical, and extensible ISL recognition application that supports visual sign recognition and spoken output.
2. Problem statement
Signova AI aims to reduce communication barriers between:
- Indian Sign Language users.
- People who do not understand sign language.
The application should recognize supported signs and convert them into readable text and spoken output.
3. Target users
1. ISL users.
2. Hearing individuals.
3. Sign language learners.
4. Researchers and students.
5. Developers.
6. Organisations.
4. Core application workflow
Webcam / Video Input

Frame Processing

Landmark Extraction

Preprocessing

AI Model Inference

Recognized Text

Text-to-Speech Output


5. Functional requirements
ID	Requirement	Description
FR-01	Webcam access	Access the user's camera
FR-02	Video processing	Process video frames
FR-03	Visual features	Extract visual features and landmarks
FR-04	Preprocessing	Prepare features for the model
FR-05	AI inference	Generate genuine model predictions
FR-06	Text output	Show recognized signs and handle uncertainty
FR-07	Speech output	Convert recognized text into speech
FR-08	Backend	Provide FastAPI services
FR-09	Model management	Load and manage AI models
FR-10	Performance	Maintain reliability and responsiveness

6. MVP and future scope
MVP requirements
- Webcam access and controls.
- Frame processing.
- Landmark extraction where supported.
- Model-compatible preprocessing.
- Verified pretrained model.
- Recognition of supported signs.
- Text output.
- Speech output.
- Error handling and recognition status.
Future features
- Expanded vocabulary.
- Dynamic sign recognition.
- Longer sign sequences.
- Continuous sentence translation.
- Facial expressions and non-manual features.
- Recognition history and user accounts.
- Accessibility improvements.
7. Technology stack
Layer	Technology
Frontend	React, TypeScript, Vite
Backend	FastAPI
Language	Python 3.11
AI	PyTorch
Computer vision	OpenCV, MediaPipe where compatible
Model resources	Hugging Face Hub
Speech	Browser SpeechSynthesis
Testing	Pytest, Vitest

8. AI model requirements
The PRD specifies that model selection should consider:
- ISL compatibility.
- Recognition task and architecture.
- Pretrained checkpoint availability.
- Input format and preprocessing.
- Supported labels.
- License.
- Hardware requirements.
- Measured performance.
The PRD originally identifies iSign T5 as an experimental candidate, not an approved production model.

Important distinction in the current implementation:
Engine	Current project direction
Individual Words	10-sign Bi-GRU model
Continuous Translation	Separate pretrained iSign T5 pipeline

The current Bi-GRU implementation should be documented as the isolated recognition engine. Do not describe iSign T5 as its replacement.
9. Non-functional requirements
The PRD also covers:
- Performance.
- Reliability.
- Maintainability.
- Security and privacy.
- Accessibility.
10. Out of scope
The PRD does not promise:
- Complete continuous ISL sentence translation in the MVP.
- Universal sign recognition.
- Every regional ISL variation.
- Guaranteed recognition for every signer and environment.
- Large-scale model training from scratch.
- Grammar correction.
- Sign animation.
11. Acceptance criteria
The product should demonstrate:
1. Backend starts successfully.
2. Webcam works.
3. Video processing works.
4. Feature extraction and preprocessing work.
5. Model loads successfully.
6. Genuine model predictions are returned.
7. UI displays predictions.
8. Uncertain signs are handled correctly.
9. Text-to-speech works.
10. Tests reflect realistic application behaviour.