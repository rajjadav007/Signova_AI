"""
Video-to-Sentence Inference CLI for Continuous Indian Sign Language (ISL) Recognition.

Script: scripts/recognize_sentence.py
Purpose:
- Closed-vocabulary continuous ISL recognition for 101 known sentence classes.
- Accepts raw video (.mp4/.avi) OR pre-extracted .npz landmarks.
- Extracts MediaPipe landmarks -> 678 temporal features -> TemporalSentenceClassifier.
- Returns top-k predicted sentence classes with English text, ISL gloss, and confidence scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
# Prevent Matplotlib font cache hangs on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "C:/temp"), "mpl"))
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent MediaPipe C++ LiteRT double-free on Windows
if sys.platform.startswith("win"):
    try:
        from mediapipe.tasks.python import vision
        vision.HandLandmarker.__del__ = lambda self: None
        vision.PoseLandmarker.__del__ = lambda self: None
    except Exception:
        pass

from ai.continuous.temporal_features import extract_temporal_features
from ai.continuous.sentence_classifier import TemporalSentenceClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("signova.recognize_sentence")


def load_sentence_classes(
    classes_json_path: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    """Load 101 sentence classes mapping."""
    if classes_json_path is None or not classes_json_path.exists():
        classes_json_path = (
            PROJECT_ROOT / "models" / "isl_csltr_continuous" / "sentence_classes.json"
        )

    mapping: Dict[int, Dict[str, Any]] = {}
    if classes_json_path.exists():
        with open(classes_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("classes", []):
            sid = int(c["sentence_id"])
            mapping[sid] = {
                "sentence_id": sid,
                "sentence": c.get("sentence", f"Sentence {sid}"),
                "isl_gloss": c.get("isl_gloss", ""),
                "tokens": c.get("tokens", []),
            }
    return mapping


def load_classifier_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> Tuple[TemporalSentenceClassifier, Dict[str, Any]]:
    """Load trained TemporalSentenceClassifier from checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("model_config", {})
    input_dim = cfg.get("input_dim", 678)
    num_classes = cfg.get("num_classes", 101)
    hidden_dim = cfg.get("hidden_dim", 128)
    num_layers = cfg.get("num_layers", 2)
    dropout = cfg.get("dropout", 0.4)

    model = TemporalSentenceClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except Exception:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt


def extract_features_from_npz(npz_path: Union[str, Path], fps: float = 25.0) -> np.ndarray:
    """Load landmarks and masks from .npz and extract 678 temporal features."""
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ file not found: {p}")

    data = np.load(p)
    landmarks = data["landmarks"].astype(np.float32)  # (T, 75, 4)
    detection_mask = data["detection_mask"] if "detection_mask" in data else None
    timestamps_ms = data["timestamps_ms"] if "timestamps_ms" in data else None

    features = extract_temporal_features(
        landmarks=landmarks,
        detection_mask=detection_mask,
        timestamps_ms=timestamps_ms,
        fps=fps,
    )
    return features


def extract_features_from_video(
    video_path: Union[str, Path],
    max_frames: Optional[int] = 300,
) -> np.ndarray:
    """Extract MediaPipe landmarks from raw video and compute 678 temporal features."""
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

    extractor = ModernMediaPipeExtractor()
    raw_spatial, meta = extractor.extract_isign_video(
        vpath,
        hand_missing_strategy="zero",
        max_frames=max_frames,
    )
    n_frames = meta["read_frames"]
    if n_frames == 0 or raw_spatial.size == 0:
        raise RuntimeError(f"0 frames decoded from video: {vpath}")

    landmarks = raw_spatial.reshape(-1, 75, 4).astype(np.float32)
    fps = float(meta["fps"]) if meta["fps"] > 0 else 25.0
    timestamps_ms = np.array([int(round(i * 1000.0 / fps)) for i in range(n_frames)], dtype=np.int64)

    det_pose = np.any(raw_spatial[:, 0:132] != 0, axis=1)
    det_lh = np.any(raw_spatial[:, 132:216] != 0, axis=1)
    det_rh = np.any(raw_spatial[:, 216:300] != 0, axis=1)
    detection_mask = np.column_stack([det_pose, det_lh, det_rh]).astype(bool)

    features = extract_temporal_features(
        landmarks=landmarks,
        detection_mask=detection_mask,
        timestamps_ms=timestamps_ms,
        fps=fps,
    )
    return features


def recognize_isl_sentence(
    source: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    top_k: int = 5,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform end-to-end ISL continuous sentence recognition on a video or NPZ file.

    Args:
        source: Path to video file (.mp4/.avi) or landmark archive (.npz)
        checkpoint_path: Path to best_temporal_sentence_classifier.pt
        top_k: Number of ranked predictions to return (default: 5)
        device_name: 'cuda', 'cpu', or None for auto-detect

    Returns:
        Dict with top-1 sentence, top-k candidates, confidences, and metadata.
    """
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Input source not found: {source_path}")

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "best_temporal_sentence_classifier.pt"

    t0 = time.time()

    # 1. Feature Extraction
    if source_path.suffix.lower() == ".npz":
        features_np = extract_features_from_npz(source_path)
    else:
        features_np = extract_features_from_video(source_path)

    t_feat = time.time() - t0

    # 2. Model Prediction
    model, ckpt = load_classifier_model(checkpoint_path, device)
    class_map = load_sentence_classes()
    ckpt_label_map = ckpt.get("label_map", {})

    x = torch.from_numpy(features_np).unsqueeze(0).to(device)  # (1, T, 678)
    length = torch.tensor([features_np.shape[0]], dtype=torch.long, device=device)

    t_infer_start = time.time()
    with torch.no_grad():
        logits = model(x, lengths=length)
        probs = F.softmax(logits, dim=-1)[0]  # (101,)
        top_scores, top_indices = torch.topk(probs, k=min(top_k, probs.size(0)))

    t_infer = time.time() - t_infer_start
    t_total = time.time() - t0

    candidates = []
    for rank, (score_t, idx_t) in enumerate(zip(top_scores, top_indices)):
        sid = int(idx_t.item())
        conf = float(score_t.item())
        cinfo = class_map.get(sid, {})
        sentence_str = cinfo.get("sentence", ckpt_label_map.get(str(sid), f"Class {sid}"))
        gloss_str = cinfo.get("isl_gloss", "")

        candidates.append({
            "rank": rank + 1,
            "sentence_id": sid,
            "sentence": sentence_str,
            "isl_gloss": gloss_str,
            "confidence": round(conf, 4),
            "confidence_percentage": f"{conf * 100:.1f}%",
        })

    top1 = candidates[0]
    return {
        "source": str(source_path),
        "source_type": source_path.suffix.lower()[1:],
        "sequence_frames": int(features_np.shape[0]),
        "feature_dimension": int(features_np.shape[1]),
        "predicted_sentence": top1["sentence"],
        "predicted_gloss": top1["isl_gloss"],
        "predicted_class_id": top1["sentence_id"],
        "confidence": top1["confidence"],
        "confidence_percentage": top1["confidence_percentage"],
        "top_candidates": candidates,
        "timing_seconds": {
            "feature_extraction": round(t_feat, 3),
            "inference": round(t_infer, 3),
            "total": round(t_total, 3),
        },
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signova AI: Continuous ISL Sentence Recognition Inference"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to raw ISL video file (.mp4, .avi, etc.)")
    group.add_argument("--npz", type=str, help="Path to pre-extracted landmark .npz file")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "models" / "isl_csltr_continuous" / "best_temporal_sentence_classifier.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of top candidates to display")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--json", action="store_true", help="Output result formatted as JSON")

    args = parser.parse_args()
    source = args.video if args.video else args.npz

    result = recognize_isl_sentence(
        source=source,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        device_name=args.device,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 65)
        print(" SIGNOVA AI: CONTINUOUS ISL SENTENCE RECOGNITION")
        print("=" * 65)
        print(f" Source File:         {result['source']}")
        print(f" Sequence Length:     {result['sequence_frames']} frames (678 features/frame)")
        print(f" Top-1 Sentence:      \"{result['predicted_sentence']}\"")
        print(f" Top-1 ISL Gloss:     {result['predicted_gloss']}")
        print(f" Class ID / Conf:     ID {result['predicted_class_id']} ({result['confidence_percentage']})")
        print(f" Inference Time:      {result['timing_seconds']['total']:.3f}s (Extraction: {result['timing_seconds']['feature_extraction']:.3f}s, Model: {result['timing_seconds']['inference']:.3f}s)")
        print("-" * 65)
        print(" Ranked Candidates:")
        for cand in result["top_candidates"]:
            print(f"   [{cand['rank']}] ({cand['confidence_percentage']:>5}) ID {cand['sentence_id']:3d}: \"{cand['sentence']}\"  [{cand['isl_gloss']}]")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
