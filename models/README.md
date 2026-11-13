# Signova AI - Model Weights & Checkpoints Directory

This directory stores model configs, tokenizer assets, and references for Indian Sign Language recognition.

> **Note:** Binary model weights (`*.pt`, `*.ckpt`, `*.joblib`, `*.safetensors`, `*.task`) are excluded from Git version control via `.gitignore`.
> The primary 15.2 GB iSign T5 model directory is kept in its original external location (`D:\iSign-t5-pose-to-text`) and referenced via environment variables to avoid repository bloat.

## Directory Structure

```text
models/
├── checkpoints/             # Local checkpoint directory (.gitkeep)
├── config/                  # Model architecture and configuration specs
│   └── model_config.json    # iSign-T5 configuration and checkpoint registry
├── tokenizer/               # Offline tokenizer assets
│   └── t5-small/            # Google T5-Small tokenizer (vocab, configs)
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── spiece.model
│       └── special_tokens_map.json
└── README.md
```

## Primary Model: iSign-T5 Pose-to-Text

- **Repository:** `manavdhamecha77/iSign-t5-pose-to-text`
- **External Model Directory:** `D:\iSign-t5-pose-to-text` (Configured via `ISIGN_MODEL_DIR`)
- **Selected Checkpoint:** `t5-small-motion` (Configured via `ISIGN_CHECKPOINT_NAME`)
- **Features:** 600 dimensions (300 spatial landmarks + 300 temporal velocity deltas)
- **Offline Tokenizer:** `models/tokenizer/t5-small` (Configured via `ISIGN_TOKENIZER_DIR`)
- **Offline Mode:** `ISIGN_OFFLINE=true` (Guarantees zero remote network calls at startup)
