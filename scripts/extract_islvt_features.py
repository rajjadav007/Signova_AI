"""
Phase 8.2: ISLVT Feature Extraction Pipeline for 4 Target Signs (I, GO, EAT, FOOD).

Extracts MediaPipe landmarks (33 pose + 21 left hand + 21 right hand = 75 landmarks x 4 channels = 300 features)
from ISLVT continuous sentence video recordings using ModernMediaPipeExtractor.
Maintains leakage-safe sentence-level splits (paired front/side views in same split).
"""

import argparse
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("signova.phase8_2_extraction")

DEFAULT_DATASET_DIR = Path(r"D:\Signova AI\Indian Sign Language Video and Text dataset for sentences (ISLVT)")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "islvt_continuous"

# Approved experimental vocabulary
TARGET_VOCAB = {"i", "go", "eat", "food"}

# Leakage-safe train / test split for the 16 sentences containing target vocabulary:
# Test: 4 sentences covering all 4 target words (21: go, 34: food, 50: i/eat, 72: i/go)
# Train: 12 sentences (4, 11, 12, 13, 17, 18, 20, 26, 31, 44, 63, 67)
SPLIT_ASSIGNMENTS = {
    4: "train",
    11: "train",
    12: "train",
    13: "train",
    17: "train",
    18: "train",
    20: "train",
    21: "test",
    26: "train",
    31: "train",
    34: "test",
    44: "train",
    50: "test",
    63: "train",
    67: "train",
    72: "test"
}


def parse_annotations(excel_path: Path) -> Dict[int, Dict[str, Any]]:
    """Parse marathi-English_ISLgloss1.xlsx and extract clean sentence mappings."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {excel_path}")

    df_raw = pd.read_excel(excel_path, header=3)
    df_clean = df_raw.dropna(subset=['sr.no']).copy()

    sentences: Dict[int, Dict[str, Any]] = {}
    for _, row in df_clean.iterrows():
        sid = int(row['sr.no'])
        eng = str(row['English sentence']).strip() if pd.notna(row['English sentence']) else ""
        mar = str(row['Marathi sentence ']).strip() if pd.notna(row['Marathi sentence ']) else ""
        isl = str(row['Indian sign language gloss']).strip() if pd.notna(row['Indian sign language gloss']) else ""

        # Tokenize gloss
        clean_gloss = re.sub(r"[?,!.]", "", isl)
        tokens = [t.lower().strip() for t in clean_gloss.split() if t.strip()]
        targets = [t for t in tokens if t in TARGET_VOCAB]

        sentences[sid] = {
            "sentence_id": sid,
            "english": eng,
            "marathi": mar,
            "isl_raw": isl,
            "tokens": tokens,
            "target_glosses": targets,
            "has_target": len(targets) > 0
        }

    logger.info(f"Loaded {len(sentences)} total sentences from {excel_path.name}.")
    target_count = sum(1 for s in sentences.values() if s["has_target"])
    logger.info(f"Sentences containing target vocabulary {TARGET_VOCAB}: {target_count}")
    return sentences


def detect_camera_view(filename: str, sentence_id: int) -> Tuple[str, str]:
    """
    Determine camera view and pairing status.
    In ISLVT, '(2)' or ' 2.MOV' indicates side-view take.
    Special case: Sentence ID 4 has two side-view duplicate takes and 0 front view.
    """
    fn_lower = filename.lower()
    has_side_tag = bool(re.search(r"(\(2\)|(?:\s2\s*\.mov$))", filename, re.IGNORECASE))

    if sentence_id == 4:
        return "side", "duplicate_side_take_no_front"

    if has_side_tag:
        return "side", "normal_pair"
    return "front", "normal_pair"


def run_extraction(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_only: bool = True
) -> Dict[str, Any]:
    """Execute complete Phase 8.2 feature extraction pipeline."""
    t0_global = time.perf_counter()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    excel_path = dataset_dir / "marathi-English_ISLgloss1.xlsx"
    sentences = parse_annotations(excel_path)

    # Filter sentences
    if target_only:
        active_sentences = {sid: s for sid, s in sentences.items() if s["has_target"]}
    else:
        active_sentences = sentences

    # Map video files in dataset_dir
    all_video_files = sorted([f for f in dataset_dir.iterdir() if f.suffix.lower() == ".mov"])
    video_map: Dict[int, List[Path]] = {}
    for vf in all_video_files:
        m = re.match(r"^\s*(\d+)", vf.name)
        if m:
            sid = int(m.group(1))
            video_map.setdefault(sid, []).append(vf)

    # Prepare output directories
    features_dir = output_dir / "features"
    train_dir = features_dir / "train"
    test_dir = features_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    extractor = ModernMediaPipeExtractor()
    logger.info("Initialized ModernMediaPipeExtractor for feature extraction.")

    manifest_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    total_attempted = 0
    total_successful = 0

    for sid in sorted(active_sentences.keys()):
        s_info = active_sentences[sid]
        vids = video_map.get(sid, [])
        if not vids:
            logger.warning(f"No video files found for Sentence ID {sid} ({s_info['isl_raw']})")
            continue

        split = SPLIT_ASSIGNMENTS.get(sid, "train")
        target_dir = train_dir if split == "train" else test_dir

        for vpath in vids:
            total_attempted += 1
            camera_view, pair_status = detect_camera_view(vpath.name, sid)
            stem = vpath.stem.strip()
            # Clean safe filename for npz
            safe_stem = re.sub(r"[^\w\-.]", "_", stem)
            out_npz_path = target_dir / f"{safe_stem}.npz"

            logger.info(f"[{total_attempted}] Processing ID {sid} ({split}) | {vpath.name} [{camera_view}]...")

            try:
                # ModernMediaPipeExtractor in VIDEO mode with hand_missing_strategy='zero'
                raw_spatial, meta = extractor.extract_isign_video(vpath, hand_missing_strategy="zero")
                num_frames = meta["read_frames"]

                if num_frames == 0 or raw_spatial.size == 0:
                    raise RuntimeError("Extractor returned 0 frames.")

                # Shape: (T, 75, 4) and (T, 300)
                landmarks = raw_spatial.reshape(-1, 75, 4).astype(np.float32)
                spatial_features = raw_spatial.astype(np.float32)

                fps = float(meta["fps"]) if meta["fps"] > 0 else 25.0
                timestamps_ms = np.array([int(round(i * 1000.0 / fps)) for i in range(num_frames)], dtype=np.int64)
                frame_indices = np.arange(num_frames, dtype=np.int32)

                # Detection mask: (T, 3) -> [pose, left_hand, right_hand]
                det_pose = np.any(raw_spatial[:, 0:132] != 0, axis=1)
                det_lh = np.any(raw_spatial[:, 132:216] != 0, axis=1)
                det_rh = np.any(raw_spatial[:, 216:300] != 0, axis=1)
                detection_mask = np.column_stack([det_pose, det_lh, det_rh]).astype(bool)

                record_meta = {
                    "video_id": safe_stem,
                    "video_filename": vpath.name,
                    "sentence_id": sid,
                    "split": split,
                    "camera_view": camera_view,
                    "pair_status": pair_status,
                    "english": s_info["english"],
                    "marathi": s_info["marathi"],
                    "isl_gloss": s_info["isl_raw"],
                    "tokens": s_info["tokens"],
                    "target_glosses": s_info["target_glosses"],
                    "frame_count": num_frames,
                    "fps": fps,
                    "duration_seconds": round(num_frames / fps, 2),
                    "resolution": meta.get("resolution", "1920x1080"),
                    "feature_dims": {
                        "landmarks": list(landmarks.shape),
                        "spatial_features": list(spatial_features.shape)
                    },
                    "pose_detection_rate": round(float(meta.get("pose_detection_rate", 0.0)), 4),
                    "left_hand_detection_rate": round(float(meta.get("left_hand_detection_rate", 0.0)), 4),
                    "right_hand_detection_rate": round(float(meta.get("right_hand_detection_rate", 0.0)), 4),
                    "average_pose_visibility": round(float(meta.get("average_pose_visibility", 0.0)), 4),
                    "missing_landmark_percentage": round(float(meta.get("missing_landmark_percentage", 0.0)), 2),
                    "extraction_fps": round(float(meta.get("extraction_fps", 0.0)), 2),
                    "extraction_time_seconds": round(float(meta.get("extraction_time_seconds", 0.0)), 2),
                    "npz_relative_path": str(out_npz_path.relative_to(output_dir)).replace("\\", "/")
                }

                # Save compressed npz archive
                np.savez_compressed(
                    out_npz_path,
                    landmarks=landmarks,
                    spatial_features=spatial_features,
                    timestamps_ms=timestamps_ms,
                    frame_indices=frame_indices,
                    detection_mask=detection_mask,
                    metadata_json=json.dumps(record_meta)
                )

                total_successful += 1
                manifest_records.append(record_meta)

            except Exception as e:
                logger.error(f"Failed extracting {vpath.name}: {e}")
                failures.append({
                    "video_filename": vpath.name,
                    "sentence_id": sid,
                    "error": str(e)
                })

    t_total = time.perf_counter() - t0_global

    # Build manifest DataFrame
    df_manifest = pd.DataFrame(manifest_records)
    manifest_csv = output_dir / "manifest.csv"
    manifest_json = output_dir / "manifest.json"
    df_manifest.to_csv(manifest_csv, index=False)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    # Build splits metadata
    splits_summary = {
        "split_policy": "sentence_level_leakage_safe (camera pairs kept together)",
        "train_sentence_ids": sorted(list(set(r["sentence_id"] for r in manifest_records if r["split"] == "train"))),
        "test_sentence_ids": sorted(list(set(r["sentence_id"] for r in manifest_records if r["split"] == "test"))),
        "train_video_count": sum(1 for r in manifest_records if r["split"] == "train"),
        "test_video_count": sum(1 for r in manifest_records if r["split"] == "test"),
        "target_vocabulary": sorted(list(TARGET_VOCAB)),
        "single_signer_limitation": True
    }
    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits_summary, f, indent=2)

    # Build feature summary
    feature_summary = {
        "pipeline": "Phase 8.2 ISLVT Feature Extraction",
        "extractor": "ModernMediaPipeExtractor (Tasks API: PoseLandmarker + HandLandmarker)",
        "total_attempted": total_attempted,
        "total_successful": total_successful,
        "total_failed": len(failures),
        "total_time_seconds": round(t_total, 2),
        "average_fps": round(df_manifest["frame_count"].sum() / t_total, 2) if total_successful > 0 else 0,
        "total_frames_extracted": int(df_manifest["frame_count"].sum()) if total_successful > 0 else 0,
        "landmarks_per_frame": 75,
        "channels_per_landmark": 4,
        "spatial_feature_dim": 300,
        "missing_strategy": "zero_unobserved (no fabricated landmarks)",
        "mean_pose_detection_rate": round(float(df_manifest["pose_detection_rate"].mean()), 4) if total_successful > 0 else 0,
        "mean_left_hand_detection_rate": round(float(df_manifest["left_hand_detection_rate"].mean()), 4) if total_successful > 0 else 0,
        "mean_right_hand_detection_rate": round(float(df_manifest["right_hand_detection_rate"].mean()), 4) if total_successful > 0 else 0,
        "mean_missing_landmark_percentage": round(float(df_manifest["missing_landmark_percentage"].mean()), 2) if total_successful > 0 else 0,
        "failures": failures
    }
    with open(output_dir / "feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(feature_summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Phase 8.2 Extraction Completed in {t_total:.2f}s")
    logger.info(f"Successful: {total_successful} / {total_attempted} | Failed: {len(failures)}")
    logger.info(f"Features saved to: {output_dir}")
    logger.info("=" * 60)

    return feature_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 8.2 ISLVT Feature Extraction Pipeline")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Path to raw ISLVT dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for features")
    parser.add_argument("--all", action="store_true", help="Extract all sentences, not just 4-sign target sentences")
    args = parser.parse_args()

    run_extraction(dataset_dir=args.dataset_dir, output_dir=args.output_dir, target_only=not args.all)


if __name__ == "__main__":
    main()
