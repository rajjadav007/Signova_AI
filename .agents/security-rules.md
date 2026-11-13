# Security Development Rules

## Signova AI

**Project:** Signova AI  
**Security Scope:** Frontend, Backend, AI/ML, Data, and Deployment  
**Backend:** Python 3.11 + FastAPI  
**Frontend:** React + JavaScript + Vite  
**AI Framework:** PyTorch  
**Primary AI Engine:** Bi-GRU Individual Words Recognition  
**Experimental AI Engine:** iSign T5 Continuous Translation

---

# 1. Core Security Principles

All development must follow these principles:

1. Protect user privacy.
2. Never expose secrets or credentials.
3. Validate all external input.
4. Use secure file handling.
5. Protect camera and media data.
6. Restrict API access appropriately.
7. Keep dependencies updated.
8. Avoid unnecessary data collection.
9. Handle errors without exposing internal details.
10. Protect pretrained model files and checkpoints.
11. Prevent unrestricted AI inference resource usage.
12. Never compromise security for convenience.

Security must apply to both recognition engines without mixing their implementation or model requirements.

---

# 2. User Privacy and Camera Security

Camera access is a sensitive feature of Signova AI.

The application uses webcam input for Individual Words recognition and may support video upload.

### Requirements

- Request camera permission only after user interaction.
- Clearly indicate when the camera is active.
- Provide an option to stop the camera.
- Stop all camera tracks when the camera is stopped.
- Stop camera tracks when the relevant component unmounts.
- Do not record or store video by default.
- Do not send camera footage to external services without explicit approval.
- Send only the data required for the recognition task.
- Do not log raw images, videos, or landmarks containing user data.
- Do not introduce background recording or hidden media collection.

### Privacy Rule

Camera data must be processed only for the requested recognition task.

Do not introduce persistent media storage or user tracking without an approved product requirement.

---

# 3. API Security

All FastAPI endpoints must validate incoming requests.

### Requirements

- Validate request bodies using Pydantic.
- Validate uploaded files.
- Restrict supported file formats.
- Apply reasonable upload size limits.
- Validate landmark arrays and their dimensions.
- Validate sequence lengths before AI inference.
- Handle malformed requests safely.
- Use suitable HTTP status codes.
- Avoid exposing internal server details.
- Add rate limiting when required by deployment conditions.
- Apply suitable request timeouts.
- Prevent invalid requests from reaching expensive model inference unnecessarily.

### Existing Recognition API

The reported application endpoints include:

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/recognize/isolated/video` | Individual Words video recognition |
| `POST /api/v1/recognize/isolated/landmarks` | Individual Words landmark recognition |
| `POST /api/v1/predict/video` | Separate continuous translation pipeline |
| `POST /api/v1/predict/landmarks` | Separate continuous translation pipeline |

Inspect actual route implementations before applying security changes.

Do not create duplicate endpoints or change existing request and response contracts without approval.

Never trust frontend validation as the only security layer.

---

# 4. CORS Security

Configure Cross-Origin Resource Sharing carefully.

### Development

Allow only the required local frontend origin.

```text
http://localhost:5173
```

If another development origin is required, add it explicitly.

### Production

- Allow only approved production domains.
- Avoid wildcard origins.
- Restrict HTTP methods.
- Restrict request headers.
- Do not enable credential sharing unnecessarily.
- Keep allowed origins in backend configuration.
- Do not allow arbitrary origins in production.

CORS configuration must match the actual frontend deployment environment.

---

# 5. Secrets and Environment Variables

Never hardcode secrets in source code.

Sensitive configuration includes:

- API keys.
- Database passwords.
- Authentication secrets.
- Private service tokens.
- External provider credentials.

### Requirements

- Use environment variables.
- Keep `.env` out of Git.
- Provide `.env.example` without real credentials.
- Never expose backend secrets through frontend code.
- Never commit credentials in documentation or logs.
- Rotate credentials if they are accidentally exposed.
- Review Git history if a secret has been committed.

### Frontend Rule

Vite variables beginning with `VITE_` are exposed to browser code.

Never store private credentials in `VITE_` variables.

The frontend may contain public configuration such as the backend API base URL, but not private tokens or passwords.

---

# 6. File Upload Security

Signova AI supports video input for recognition where the backend accepts compatible files.

### Requirements

- Validate file size.
- Validate file extension.
- Validate MIME type.
- Do not trust the original filename.
- Generate safe temporary filenames.
- Store temporary files outside public directories.
- Prevent path traversal.
- Delete temporary files after processing.
- Reject unsupported or malformed files.
- Avoid executing uploaded content.
- Apply suitable processing timeouts.
- Restrict video duration and frame count where appropriate.

### Supported Input

Inspect the existing backend implementation to identify the actual supported video formats.

Do not claim support for additional formats without testing them.

### Storage Rule

Do not permanently store uploaded videos unless a clearly approved feature requires it.

---

# 7. AI Model Security

Signova AI uses two separate AI engines.

Their model files, preprocessing requirements, and inference pipelines must remain independent.

## 7.1 Individual Words Bi-GRU Engine

The existing isolated recognition engine uses:

- MediaPipe Tasks API.
- Pose and hand landmarks.
- 75 landmarks with 4 channels.
- Temporal resampling to T=64.
- 600-dimensional combined features.
- A 2-layer Bi-GRU model.
- A 10-class recognition vocabulary.

### Security Requirements

- Load only the approved model checkpoint.
- Verify checkpoint paths.
- Verify model configuration compatibility.
- Validate feature dimensions.
- Validate sequence lengths.
- Reject malformed or incomplete input.
- Restrict access to model files.
- Do not expose local model paths through APIs.
- Do not allow users to upload arbitrary model checkpoints.
- Do not execute user-provided model code.

## 7.2 Continuous Translation iSign T5 Engine

The iSign T5 pipeline is separate and experimental.

### Security Requirements

- Verify the model repository and checkpoint source.
- Review the model licence.
- Verify checkpoint compatibility.
- Validate pose input dimensions and sequence lengths.
- Keep T5 preprocessing separate from Bi-GRU preprocessing.
- Restrict model file access.
- Avoid unrestricted inference requests.
- Show model unavailability safely when required.

Do not assume that the T5 model is operational simply because its files exist.

## 7.3 PyTorch Security

- Download models from trusted sources.
- Verify model repository identity.
- Review model licences.
- Avoid loading untrusted pickle-based checkpoints.
- Prefer safer model formats where supported.
- Use `weights_only=True` when compatible with the checkpoint and installed PyTorch version.
- Do not disable model-loading safety checks without a documented reason.
- Review custom model-loading code before execution.
- Restrict write access to model directories.

If an existing trusted checkpoint requires a particular loading method, inspect its format and loading implementation before changing it.

Do not alter or replace existing approved checkpoints as part of routine security changes.

## 7.4 Inference Security

- Validate all model inputs.
- Restrict input dimensions and sequence lengths.
- Apply reasonable processing limits.
- Handle malformed landmarks.
- Avoid unrestricted resource consumption.
- Prevent repeated model loading.
- Handle CPU and GPU resource errors.
- Never execute user-provided code or model instructions.

---

# 8. Dataset Security and Licensing

Before using any dataset:

- Verify its source.
- Review its licence.
- Check usage restrictions.
- Avoid downloading unknown executable files.
- Preserve original data.
- Restrict access to local datasets.
- Do not upload restricted datasets to public repositories.
- Keep private user recordings separate from public datasets.
- Avoid exposing dataset files through public frontend directories.

### Signova AI Dataset Rule

The current isolated recognition engine uses the ISL isolated-sign dataset and its selected 10-class subset.

Do not assume that public availability means unrestricted commercial or redistribution rights.

Respect attribution, non-commercial, and share-alike requirements where applicable.

Do not replace the current dataset or modify its manifests without approval.

---

# 9. Frontend Security

The React frontend must follow these rules:

- Avoid unsafe HTML rendering.
- Do not use `dangerouslySetInnerHTML` unless strictly necessary and safely sanitised.
- Validate data received from the backend.
- Do not expose private configuration.
- Avoid storing sensitive information in browser storage.
- Handle API errors safely.
- Use HTTPS in production.
- Do not include secrets in client-side source code.
- Avoid untrusted external scripts.
- Do not interpret prediction text as executable content.
- Request camera permission only after user interaction.
- Stop media streams when no longer needed.

Use normal React rendering for recognition output.

Do not execute or interpret model predictions as HTML or JavaScript.

The frontend must not be treated as the only security layer.

---

# 10. Authentication and Authorisation

Authentication is not required for the initial MVP unless explicitly approved.

If authentication is added later:

- Use a trusted authentication mechanism.
- Never store plaintext passwords.
- Validate authentication tokens on the backend.
- Protect private endpoints.
- Apply suitable session expiration.
- Restrict access to user-specific records.
- Do not rely on frontend-only access checks.
- Protect sensitive user data.

Do not add authentication or user accounts without a product requirement.

---

# 11. Database Security

PostgreSQL is optional for the initial MVP.

Basic sign recognition must not depend on a database.

If PostgreSQL is introduced:

- Keep credentials in environment variables.
- Use parameterised queries or a suitable ORM.
- Restrict database permissions.
- Validate stored data.
- Avoid storing raw camera footage.
- Protect personal information.
- Use migrations for schema changes.
- Do not expose database errors to users.
- Back up data securely when persistent storage is required.

Do not introduce database dependencies only for temporary frontend recognition history.

---

# 12. Error Handling and Logging

## Error Handling

- Return safe and understandable error messages.
- Do not expose stack traces.
- Do not reveal internal paths.
- Do not expose environment variables.
- Do not reveal database credentials.
- Handle unexpected exceptions safely.
- Return appropriate HTTP status codes.
- Distinguish invalid input from model inference failure.
- Avoid exposing checkpoint locations or internal model configuration.

## Logging

Log useful technical events:

- Application startup.
- Model loading status.
- API failures.
- Processing errors.
- Unexpected exceptions.
- Resource or timeout failures.

Never log:

- Passwords.
- API keys.
- Authentication tokens.
- Raw camera footage.
- Sensitive user data.
- Unnecessary request payloads.

Do not log complete landmark sequences unless required for an approved debugging task.

---

# 13. Dependency Security

All project dependencies must be reviewed.

### Requirements

- Install packages from trusted sources.
- Avoid unnecessary dependencies.
- Check package maintenance and compatibility.
- Review security advisories.
- Avoid installing unknown scripts.
- Pin important dependency versions where practical.
- Review major version upgrades before applying them.
- Check Python and npm dependencies.
- Do not replace working packages without a technical reason.

Do not run downloaded scripts without inspecting their source.

---

# 14. Network and Deployment Security

For production deployment:

- Use HTTPS.
- Configure secure CORS policies.
- Keep secrets on the backend.
- Restrict public API access where appropriate.
- Apply request size limits.
- Configure suitable timeouts.
- Avoid exposing internal development ports.
- Keep debug mode disabled.
- Use secure environment configuration.
- Monitor application errors.
- Protect model download and loading operations.

Do not expose development tools or debug endpoints publicly.

Do not deploy local development credentials.

---

# 15. Resource Protection

AI inference can consume significant CPU, GPU, and memory resources.

### Requirements

- Limit upload sizes.
- Limit video duration where applicable.
- Restrict frame counts.
- Validate sequence lengths.
- Avoid unlimited concurrent inference.
- Handle GPU memory errors.
- Set suitable request timeouts.
- Prevent repeated model loading.
- Avoid unrestricted background processing.
- Release temporary files and resources.
- Avoid allowing one request to block the entire application.

### Signova AI Specific Rules

For Individual Words recognition:

- Validate video duration before processing.
- Validate the extracted sequence.
- Ensure the expected feature dimensions.
- Reuse the loaded Bi-GRU model.
- Reject malformed inputs before inference.

For Continuous Translation:

- Apply separate resource limits.
- Do not assume its resource requirements match Bi-GRU.
- Keep its model loading and inference lifecycle independent.

Do not allow one request to consume unlimited system resources.

---

# 16. Security Testing

Test the following areas:

| Area | Security Test |
|---|---|
| API | Invalid request validation |
| Uploads | Unsupported file rejection |
| Uploads | Oversized file rejection |
| CORS | Unapproved origin rejection |
| Secrets | No credentials in frontend or Git |
| Camera | Permission and cleanup behaviour |
| AI Models | Untrusted checkpoint handling |
| AI Inputs | Invalid dimensions and sequence rejection |
| Database | Query safety, if database is used |
| Errors | No internal details exposed |
| Dependencies | Known vulnerability checks |
| Deployment | HTTPS and debug configuration |
| Resource Limits | Oversized and excessive requests |

Security testing must be performed before production deployment.

Do not report a security test as passed unless it was actually executed.

Distinguish mocked tests from real backend and deployment security tests.

---

# 17. Security Incident Handling

If a security issue is discovered:

1. Stop the affected operation if necessary.
2. Identify the affected component.
3. Avoid exposing additional user data.
4. Preserve relevant technical logs without collecting sensitive content.
5. Fix the root cause.
6. Rotate exposed credentials.
7. Test the fix.
8. Document the issue and resolution.

Do not hide security failures or claim they are resolved without verification.

---

# 18. Strict Rules for Antigravity

Before making security-related changes:

1. Inspect the existing implementation.
2. Read `SECURITY.md` if it exists.
3. Read the backend development rules.
4. Read the frontend development rules.
5. Read the AI model strategy.
6. Reuse existing security controls.
7. Do not weaken existing protections.
8. Do not expose credentials.
9. Do not delete security configuration without approval.
10. Do not install untrusted dependencies.
11. Do not add unnecessary authentication.
12. Do not store camera data without approval.
13. Do not load untrusted model checkpoints.
14. Do not claim security testing passed without running it.
15. Keep code, comments, and tests in English.
16. Update relevant documentation when security behaviour changes.
17. Keep the Bi-GRU and iSign T5 engines independent.
18. Do not modify model weights, checkpoints, or thresholds as part of security work without approval.
19. Do not introduce database requirements into the MVP without approval.
20. Do not remove existing working recognition functionality.

Before changing a security-sensitive implementation, explain the risk and the proposed change.

If a security requirement is unclear, report the risk before implementing a potentially unsafe change.

---

# 19. Final Security Goal

Signova AI must protect:

- User privacy.
- Webcam and uploaded media.
- Application credentials.
- Backend APIs.
- AI model files and checkpoints.
- Dataset files.
- System resources.
- Any future persistent user data.

Security must be part of every development phase.

**Never sacrifice user privacy or application security to make a feature appear functional.**