# Signova AI - Datasets Directory

This directory stores Indian Sign Language (ISL) datasets, annotations, and sample benchmark media.

> **Note:** Raw dataset videos, high-resolution media, and numpy landmark arrays (`*.mp4`, `*.avi`, `*.npy`) are excluded from Git version control via `.gitignore`.

## Expected Directory Structure

```text
datasets/
├── annotations/       # Ground truth label CSVs and metadata
├── samples/           # Evaluation and benchmark video clips
│   ├── isolated/      # Isolated sign test clips (alphabets, numbers, words)
│   └── continuous/    # Continuous sign language evaluation samples
└── README.md
```

## Recommended ISL Datasets

1. **INCLUDE Dataset:**
   - Isolated Indian Sign Language signs across multiple signers and categories.
2. **ISL-CSLTR:**
   - Continuous Sign Language Translation and Recognition dataset.
3. **Custom Benchmark Sets:**
   - Standardized evaluation videos for webcam testing and latency profiling.
