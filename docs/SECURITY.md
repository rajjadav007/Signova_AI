# Security Policy

## Signova AI

This document describes the security practices and vulnerability reporting process for the Signova AI project.

## 1. Supported Versions

During prototype development, security updates are focused on the latest development version.

| Version | Security Support |
|---|---|
| Current development version | Active |
| Older versions | Not guaranteed |

Security support may change as the project evolves.

---

## 2. Reporting a Vulnerability

If you discover a security vulnerability, report it responsibly.

Please do not publish sensitive vulnerability details in public issues before the project maintainers have had an opportunity to investigate.

### Reporting Process

1. Check whether the issue has already been reported.
2. Use GitHub's private vulnerability reporting feature or Security Advisories if enabled for the repository.
3. If private reporting is unavailable, contact the repository maintainer through an official private channel.
4. Include enough technical detail to reproduce the issue safely.

### Include the Following Information

- Vulnerability description.
- Affected component or file.
- Steps to reproduce.
- Potential security impact.
- Relevant logs or screenshots, with secrets removed.
- Suggested mitigation, if available.

Do not include passwords, API keys, access tokens, private user data, or other secrets in vulnerability reports.

---

## 3. Security Principles

Signova AI follows these security principles:

- Validate all incoming API requests.
- Keep secrets outside source code.
- Avoid exposing internal server details.
- Use secure dependency versions.
- Limit access to sensitive resources.
- Handle uploaded files safely.
- Protect user privacy.
- Avoid unnecessary collection or storage of personal data.

---

## 4. API Security

The FastAPI backend must follow these practices:

- Validate request bodies and parameters.
- Restrict accepted file types.
- Enforce upload size limits.
- Validate image and video content before processing.
- Handle malformed requests safely.
- Return appropriate HTTP status codes.
- Avoid exposing stack traces in production responses.
- Configure CORS for trusted frontend origins.
- Apply rate limiting where appropriate.
- Avoid logging sensitive request data.

API endpoints that accept uploaded media must not trust file extensions alone.

---

## 5. Camera and User Privacy

Signova AI may use a device camera for sign recognition.

The application should:

- Request camera permission before accessing the camera.
- Explain why camera access is needed.
- Allow users to stop camera access.
- Avoid recording or storing video without a clear requirement and user consent.
- Avoid sending camera data to unrelated services.
- Process data only as required for the recognition workflow.
- Avoid retaining temporary frames longer than necessary.

The initial prototype should not introduce automatic recording or unnecessary user tracking.

---

## 6. Model and Dataset Security

Signova AI may download pretrained models and use external datasets.

Follow these practices:

- Download models from trusted repositories.
- Verify repository ownership and source information.
- Review model licences before use.
- Inspect model loading code.
- Avoid loading untrusted executable model formats.
- Prefer safer model formats such as `safetensors` when supported.
- Do not execute unknown scripts from model repositories without inspection.
- Keep large model checkpoints outside Git.
- Do not commit private datasets or sensitive user recordings.
- Verify dataset licences and usage restrictions.

Model predictions must not be treated as trusted application instructions or executable content.

---

## 7. Environment Variables and Secrets

Sensitive configuration must be stored in environment variables.

Examples include:

- API keys.
- Database credentials.
- Authentication secrets.
- External service tokens.
- Deployment credentials.

Security rules:

- Keep `.env` files out of Git.
- Provide `.env.example` with placeholder values only.
- Never hardcode real credentials.
- Never expose backend secrets in frontend code.
- Rotate credentials if they are accidentally exposed.
- Use separate credentials for development and production.

---

## 8. Database Security

PostgreSQL is an optional future component.

If database functionality is introduced:

- Use parameterised queries or a safe ORM.
- Validate data before insertion.
- Restrict database permissions.
- Never expose database credentials to the frontend.
- Avoid storing unnecessary personal information.
- Protect sensitive records.
- Configure backups securely.
- Use encrypted connections in production where supported.

---

## 9. Frontend Security

The React frontend should:

- Avoid inserting untrusted content as raw HTML.
- Validate data received from APIs.
- Avoid storing sensitive credentials in browser storage.
- Use HTTPS in production.
- Handle API errors without exposing internal details.
- Request only necessary browser permissions.
- Avoid loading untrusted external scripts.

---

## 10. Dependency Security

The project uses Python and JavaScript dependencies.

Security practices:

- Use trusted package registries.
- Review dependency updates.
- Remove unused dependencies.
- Monitor known vulnerabilities.
- Keep dependency versions controlled.
- Review changes before upgrading important AI libraries.
- Avoid installing unknown packages from untrusted sources.

---

## 11. Deployment Security

Before production deployment:

- Enable HTTPS.
- Configure trusted CORS origins.
- Protect environment variables.
- Disable unnecessary debug features.
- Restrict server access.
- Configure request size limits.
- Protect model and dataset storage.
- Review exposed API endpoints.
- Monitor application errors.
- Keep backend and system dependencies updated.

---

## 12. Security Limitations

Signova AI is currently a prototype under development.

Not all security controls described in this document may be implemented yet.

The project must not be considered production-secure until the relevant controls have been implemented, tested, and reviewed.

---

## 13. Responsible Disclosure

Security researchers and users are encouraged to report vulnerabilities responsibly.

Please allow maintainers reasonable time to investigate and address reported issues before publicly disclosing sensitive details.

Thank you for helping improve the security and privacy of Signova AI.