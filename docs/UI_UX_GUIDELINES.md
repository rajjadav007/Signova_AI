# UI/UX Guidelines

## Signova AI

**Project:** Signova AI  
**Application Type:** Web Application  
**Frontend:** React + JavaScript + Vite  
**Styling:** CSS  
**Design Approach:** Modern, Minimal, Professional, Accessible  
**Primary Focus:** Indian Sign Language Recognition

---

# 1. Design Vision

Signova AI should provide a modern, accessible, and easy-to-use interface for Indian Sign Language recognition.

The application must help users understand:

- What the camera is capturing.
- Whether the recognition system is ready.
- Which AI engine is being used.
- What sign has been recognised.
- Whether the prediction is uncertain.
- When speech output is available.

The interface should feel like a professional AI accessibility product, not a generic admin dashboard.

The recognition workspace must remain the primary focus.

---

# 2. Project-Specific UI Requirements

Signova AI has two separate AI engines.

## 2.1 Individual Words Recognition

**Model:** Bi-GRU  
**Scope:** 10 supported ISL signs.

Supported vocabulary:

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

The interface should support:

- Webcam preview.
- Camera controls.
- Capture and Recognize action.
- Recording countdown.
- Expected-sign selector for evaluation.
- Prediction result.
- Confidence information when provided by the model.
- Top-3 predictions when available.
- Recognition latency.
- Uncertain prediction handling.
- Text-to-speech.
- Test history and evaluation summary.

Do not present the 10-sign classifier as a complete continuous ISL sentence translator.

## 2.2 Continuous Translation

**Model:** Separate pretrained iSign T5 pipeline.

This engine must remain independent of the Bi-GRU engine.

If the continuous translation feature is available in the application, its interface should provide:

- Separate translation workspace.
- Input capture or supported video upload.
- Processing status.
- Translation output.
- Model availability status.
- Error and unsupported-input handling.

Do not show fabricated translations.

Do not show the continuous translation engine as operational unless its actual backend integration and inference have been verified.

---

# 3. Design Principles

## 3.1 Simplicity

- Keep the interface clean.
- Avoid unnecessary sections.
- Use clear labels.
- Make important actions easy to find.
- Keep recognition results prominent.

## 3.2 Accessibility

- Use readable font sizes.
- Maintain sufficient colour contrast.
- Provide clear button labels.
- Support keyboard navigation.
- Provide visible focus indicators.
- Do not communicate status using colour alone.
- Support screen readers.

## 3.3 Consistency

- Use consistent spacing.
- Maintain a clear typography system.
- Reuse existing components.
- Keep buttons, cards, and status indicators consistent.

## 3.4 Responsiveness

- Support desktop, tablet, and mobile.
- Prevent horizontal overflow.
- Keep camera controls accessible on small screens.
- Maintain readable recognition results.

---

# 4. Visual Identity

## Design Direction

Use a professional AI accessibility product style:

- White and teal visual identity.
- Soft backgrounds.
- Clean cards.
- Rounded corners.
- Subtle shadows.
- Clear typography.
- Minimal distractions.
- Consistent spacing.

## Colour Palette

| Colour | Value | Usage |
|---|---|---|
| Primary Teal | #0F766E | Main actions |
| Dark Teal | #115E59 | Hover and emphasis |
| Light Teal | #CCFBF1 | Highlight backgrounds |
| Background | #F8FAFC | Main background |
| Surface | #FFFFFF | Cards and panels |
| Primary Text | #0F172A | Headings |
| Secondary Text | #64748B | Supporting text |
| Border | #E2E8F0 | Borders |
| Success | #16A34A | Successful operations |
| Warning | #D97706 | Warnings |
| Error | #DC2626 | Errors |

Use colours consistently. Avoid unnecessary accent colours.

---

# 5. Typography

Recommended fonts:

- Inter
- Geist Sans
- DM Sans

## Typography Scale

| Element | Suggested Size |
|---|---|
| Main Heading | 32px to 40px |
| Section Heading | 22px to 28px |
| Card Heading | 18px to 20px |
| Body Text | 14px to 16px |
| Supporting Text | 12px to 14px |
| Button Text | 14px to 16px |

Use readable font weights and line heights.

---

# 6. Main Application Layout

The application should organise its interface around the recognition workflow.

Suggested structure:

```text
Signova AI
│
├── Header
│   ├── Logo
│   ├── Product Name
│   └── Navigation
│
├── Main Workspace
│   ├── Welcome Section
│   ├── Recognition Engine Selection
│   │
│   ├── Individual Words Recognition
│   │   ├── Camera Preview
│   │   ├── Camera Controls
│   │   ├── Capture & Recognize
│   │   ├── Expected Sign Selector
│   │   └── Recognition Result
│   │
│   └── Continuous Translation
│       ├── Input Panel
│       ├── Processing Status
│       └── Translation Result
│
├── Speech Controls
│
├── Model Status
│
└── Footer
```

Important:

- Individual Words Recognition should be the default workspace.
- Keep the continuous translation engine separate.
- Do not display unavailable features as functional.
- Reuse existing application components.
- Do not create duplicate workspaces or unnecessary navigation.

---

# 7. Header and Navigation

Required elements:

- Signova AI logo.
- Product name.
- Simple navigation.
- Optional accessibility settings.

Design rules:

- Keep the header simple.
- Use consistent height.
- Avoid excessive navigation items.
- Maintain responsive behaviour.
- Make the current recognition workspace clear.

---

# 8. Individual Words Recognition Interface

This is the primary MVP interface.

## Required Elements

- Webcam preview.
- Camera permission status.
- Start Camera button.
- Stop Camera button.
- Capture & Recognize button.
- Recording countdown.
- Expected Sign selector.
- Recognition status.
- Prediction result.
- Confidence when provided.
- Top-3 predictions when available.
- Inference latency when available.
- Speech controls.

## Camera States

| State | UI Behaviour |
|---|---|
| Initial | Show camera start action |
| Permission Required | Explain camera permission |
| Camera Active | Display live preview |
| Camera Stopped | Show inactive state |
| Permission Denied | Display helpful instructions |
| Camera Unavailable | Show an error |
| Recording | Display countdown |
| Processing | Show processing indicator |

Do not start camera access automatically without user interaction.

Keep the user's hands and signing area visible. Do not place controls over important camera content.

---

# 9. Recognition Interface

The recognition interface must clearly communicate processing status.

## Recognition States

| State | Display |
|---|---|
| Ready | Ready to recognise |
| Recording | Recording signing input |
| Processing | Processing signing input |
| Success | Prediction available |
| Uncertain | Could not confidently recognise |
| Error | Recognition failed |
| Model Unavailable | Model is not available |

## Recognition Rules

- Display actual model output.
- Never hardcode fake predictions.
- Display confidence only when supplied by the model.
- Display top-3 results only when returned by the backend.
- Distinguish uncertain predictions from technical errors.
- Do not show an old prediction as a new result.
- Explain when a sign is outside the supported vocabulary.

---

# 10. Recognition Result Card

The result card should be visually prominent.

Suggested structure:

```text
Recognition Result

Recognised Sign
------------------------------
[ Actual model prediction ]

Confidence: [ Actual confidence ]

Top Predictions:
1. [ Actual prediction ]
2. [ Actual prediction ]
3. [ Actual prediction ]

Inference Time: [ Actual latency ]

[ Copy Text ] [ Speak ]
```

Requirements:

- Display actual backend responses.
- Support longer text.
- Provide copy functionality if implemented.
- Provide speech playback controls.
- Show a useful empty state.
- Clear or update old results when a new recognition attempt starts.

Do not display missing values as invented data.

---

# 11. Expected Sign and Evaluation Interface

The expected-sign selector is intended for testing and evaluation.

Requirements:

- Display only the supported 10-sign vocabulary.
- Allow the tester to select the expected sign.
- Keep this control visually separate from the model prediction.
- Never use the expected sign as the model prediction.
- Update evaluation history only after a valid evaluation.
- Clearly distinguish correct, incorrect, and uncertain predictions.

The expected sign is ground truth for evaluation. It must never influence the AI prediction.

---

# 12. Recognition Test History

If the existing evaluation history feature is available, display:

- Total attempts.
- Correct predictions.
- Incorrect predictions.
- Uncertain predictions.
- Accuracy for the recorded evaluation attempts.

Requirements:

- Use actual evaluation data.
- Provide a reset action if implemented.
- Do not fabricate evaluation results.
- Make it clear that the displayed accuracy belongs to the recorded test session, not universal ISL recognition accuracy.

---

# 13. Continuous Translation Interface

This interface belongs to the separate iSign T5 engine.

Keep it independent of Individual Words Recognition.

## Required Elements When Integrated

- Input capture or supported video upload.
- Processing status.
- Translation result.
- Model availability.
- Error handling.
- Speech output if supported.

## Design Rules

- Do not reuse Bi-GRU predictions as continuous translations.
- Do not mix model checkpoints.
- Do not imply that isolated word recognition produces complete sentences.
- Show an unavailable or experimental state when the T5 pipeline is not operational.
- Display only verified translations from the actual inference pipeline.

---

# 14. Text-to-Speech Interface

Speech controls should be simple and accessible.

Required controls:

- Speak.
- Stop.
- Speech status.
- Optional voice selection.

Speech states:

| State | Display |
|---|---|
| Idle | Ready to speak |
| Speaking | Speech playing |
| Stopped | Speech stopped |
| Unsupported | Speech unavailable |

Requirements:

- Disable speech controls when there is no result.
- Clearly indicate when speech is playing.
- Handle unsupported browser functionality.
- Do not speak fabricated or stale results.

---

# 15. Buttons and Interactive Elements

Use consistent button styles.

## Primary Buttons

Examples:

- Start Camera.
- Capture & Recognize.
- Start Translation, only if supported.

## Secondary Buttons

Examples:

- Stop Camera.
- Copy Text.
- Reset History.
- Stop Speech.

Requirements:

- Clear action labels.
- Visible hover and focus states.
- Proper disabled states.
- Prevent accidental repeated actions.
- Minimum recommended touch target: approximately 44 × 44 CSS pixels.

---

# 16. Cards and Containers

Recommended cards:

- Camera Preview Card.
- Recognition Result Card.
- Model Status Card.
- Evaluation History Card.
- Speech Control Card.
- Continuous Translation Card, when integrated.

Card styling:

- White background.
- Subtle border.
- Rounded corners.
- Light shadow where useful.
- Consistent internal spacing.

Avoid excessive card nesting.

---

# 17. Model Status and System Feedback

Display model status using actual backend health information.

Possible states:

| State | Display |
|---|---|
| Available | Model Ready |
| Loading | Loading Model |
| Unavailable | Model Unavailable |
| Error | Model Error |

Keep the status of the Bi-GRU and iSign T5 engines separate.

Do not display one engine as available based on the health status of the other.

Avoid exposing internal stack traces or server details.

---

# 18. Icons and Visual Elements

Use a consistent icon library such as Lucide.

Recommended icons:

| Purpose | Icon |
|---|---|
| Camera | Camera |
| Start | Play |
| Stop | Square |
| Recognition | Scan |
| Speech | Volume 2 |
| Copy | Copy |
| Success | Check Circle |
| Error | Alert Circle |
| Settings | Settings |
| Loading | Loader |

Icons must have accessible labels when they communicate an action.

Do not use icons alone when their meaning may be unclear.

---

# 19. Responsive Design

## Desktop

- Main workspace can use two columns.
- Camera and results can appear side by side.
- Keep recognition controls close to the camera.
- Preserve readable result information.

## Tablet

- Reduce spacing.
- Maintain readable content.
- Adjust camera and result panel widths.

## Mobile

- Use a single-column layout.
- Keep camera preview prominent.
- Place controls within easy reach.
- Avoid horizontal scrolling.
- Keep results readable.
- Ensure buttons have sufficient touch targets.

---

# 20. Accessibility Requirements

Follow these requirements:

- Use semantic HTML.
- Provide accessible names for controls.
- Support keyboard navigation.
- Maintain visible focus states.
- Use sufficient text contrast.
- Avoid colour-only status indicators.
- Provide clear error messages.
- Use appropriate ARIA attributes when needed.
- Support screen readers.
- Respect reduced-motion preferences.

Do not make the interface dependent on animations to communicate important information.

---

# 21. Loading, Empty, and Error States

Every important interaction should provide feedback.

## Loading

Show a spinner or progress indicator while processing.

## Empty

Explain what the user needs to do when no recognition result exists.

## Error

Display a short and helpful message.

## Uncertain Prediction

Explain that the model could not confidently recognise the signing input.

## Model Unavailable

Clearly explain which AI engine is unavailable.

Do not display fabricated predictions or translations.

---

# 22. Animation and Microinteractions

Use subtle animations only when they improve usability.

Suitable animations:

- Button hover transitions.
- Card appearance.
- Camera activation feedback.
- Loading indicators.
- Recognition result updates.

Avoid:

- Excessive motion.
- Distracting background effects.
- Long transitions.
- Animations that delay important actions.

Respect reduced-motion settings.

---

# 23. Frontend Component Structure

The following is a suggested structure, not a requirement to recreate existing components.

```text
frontend/src/
├── components/
│   ├── camera/
│   ├── recognition/
│   ├── speech/
│   ├── evaluation/
│   ├── translation/
│   ├── layout/
│   └── common/
│
├── services/
│   └── api.js
│
├── hooks/
├── context/
├── styles/
├── App.jsx
└── main.jsx
```

Existing implementation must be inspected before creating or moving components.

Reuse the existing webcam tester and recognition components wherever possible.

Do not create duplicate components just to match this suggested folder structure.

---

# 24. UI Development Rules for Antigravity

Antigravity must follow these rules:

1. Use React with JavaScript.
2. Use `.jsx` for React components.
3. Use `.js` for JavaScript utility modules.
4. Do not introduce TypeScript.
5. Reuse existing components.
6. Keep API logic inside service modules.
7. Keep camera logic separate from presentation components.
8. Keep speech logic separate from recognition logic.
9. Keep Individual Words and Continuous Translation engines separate.
10. Use consistent CSS variables for design tokens.
11. Avoid unnecessary dependencies.
12. Never hardcode fake AI predictions.
13. Display actual backend responses.
14. Handle loading, empty, uncertain, and error states.
15. Test responsive layouts.
16. Follow accessibility requirements.
17. Do not change AI models or inference logic as part of UI work.
18. Do not remove existing working functionality.
19. Do not change recognition thresholds without approval.
20. Do not claim unsupported signs are recognised.

---

# 25. Final UI/UX Goal

The final Signova AI interface should be:

- Modern.
- Minimal.
- Professional.
- Accessible.
- Responsive.
- Easy to understand.
- Focused on ISL recognition.

The Individual Words Recognition experience must remain the primary MVP focus.

The Continuous Translation engine must remain a separate feature with its own model status and verified output.

The interface should help users understand what the system is doing, what it has recognised, whether the result is reliable, and when speech output is available.