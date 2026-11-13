"""
Diagnostic script for F-015.1: Real Video Pose Compatibility Investigation.
Runs controlled diagnostic experiments on '10 granfather sick  (2).mov'.
"""

import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "models" / "assets")

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ai.inference.model_loader import load_model, get_model, get_tokenizer
from ai.inference.pose_translator import PoseTranslator, predict_with_metadata
from ai.preprocessing.pose_preprocessor import (
    BASE_SPATIAL_DIM,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    assemble_frame_landmarks,
    preprocess_pose,
)

VIDEO_PATH = PROJECT_ROOT / "datasets" / "videos" / "10 granfather sick  (2).mov"
TASK_PATH = PROJECT_ROOT / "models" / "assets" / "holistic_landmarker.task"


def extract_landmarks_detailed(video_path: Path):
    """Extract landmarks and analyze frame-by-frame hand positions."""
    cap = cv2.VideoCapture(str(video_path))
    opts = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK_PATH)),
        min_hand_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(opts)

    frames_kps = []
    lh_detections = []
    rh_detections = []
    lh_positions = []
    rh_positions = []

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        # Body pose
        pose_pts = np.zeros((POSE_LANDMARK_COUNT, 4), dtype=np.float32)
        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            for i, lm in enumerate(res.pose_landmarks[:POSE_LANDMARK_COUNT]):
                vis = getattr(lm, "visibility", 1.0)
                pose_pts[i] = [lm.x, lm.y, lm.z, vis if vis is not None else 1.0]

        # Left hand
        lh_pts = np.zeros((HAND_LANDMARK_COUNT, 4), dtype=np.float32)
        if res.left_hand_landmarks and len(res.left_hand_landmarks) > 0:
            lh_detections.append(True)
            wrist = res.left_hand_landmarks[0]
            lh_positions.append((wrist.x, wrist.y, wrist.z))
            for i, lm in enumerate(res.left_hand_landmarks[:HAND_LANDMARK_COUNT]):
                lh_pts[i] = [lm.x, lm.y, lm.z, 1.0]
        else:
            lh_detections.append(False)
            lh_positions.append(None)

        # Right hand
        rh_pts = np.zeros((HAND_LANDMARK_COUNT, 4), dtype=np.float32)
        if res.right_hand_landmarks and len(res.right_hand_landmarks) > 0:
            rh_detections.append(True)
            wrist = res.right_hand_landmarks[0]
            rh_positions.append((wrist.x, wrist.y, wrist.z))
            for i, lm in enumerate(res.right_hand_landmarks[:HAND_LANDMARK_COUNT]):
                rh_pts[i] = [lm.x, lm.y, lm.z, 1.0]
        else:
            rh_detections.append(False)
            rh_positions.append(None)

        vec = assemble_frame_landmarks(pose=pose_pts, left_hand=lh_pts, right_hand=rh_pts)
        frames_kps.append(vec)
        f_idx += 1

    cap.release()
    landmarker.close()

    raw_kps = np.asarray(frames_kps, dtype=np.float32)
    return raw_kps, lh_detections, rh_detections, lh_positions, rh_positions


def main():
    print("=" * 70)
    print("DIAGNOSTIC INVESTIGATION: F-015.1")
    print("=" * 70)

    raw_kps, lh_det, rh_det, lh_pos, rh_pos = extract_landmarks_detailed(VIDEO_PATH)
    total_frames = len(lh_det)

    print(f"\n1. HAND DETECTION CHRONOLOGY (Total: {total_frames} frames)")
    # Analyze continuous intervals for LH
    lh_intervals = []
    current_state = lh_det[0]
    start_f = 0
    for i in range(1, total_frames):
        if lh_det[i] != current_state:
            lh_intervals.append((current_state, start_f, i - 1, i - start_f))
            current_state = lh_det[i]
            start_f = i
    lh_intervals.append((current_state, start_f, total_frames - 1, total_frames - start_f))

    print("  Left Hand Detection Intervals (State, Start, End, Count):")
    for state, s, e, count in lh_intervals:
        print(f"    {'PRESENT' if state else 'MISSING'}: frames {s:3d} to {e:3d} ({count:2d} frames)")

    # Right hand intervals
    rh_missing_idx = [i for i, d in enumerate(rh_det) if not d]
    print(f"  Right Hand Missing Frame indices: {rh_missing_idx}")

    # Check wrist positions of Left Hand when detected
    valid_lh = [p for p in lh_pos if p is not None]
    if valid_lh:
        xs, ys, zs = zip(*valid_lh)
        print(f"  Left Hand Wrist when detected -> X: [{min(xs):.3f}, {max(xs):.3f}], Y: [{min(ys):.3f}, {max(ys):.3f}]")

    # Check wrist positions of Right Hand when detected
    valid_rh = [p for p in rh_pos if p is not None]
    if valid_rh:
        rxs, rys, rzs = zip(*valid_rh)
        print(f"  Right Hand Wrist when detected -> X: [{min(rxs):.3f}, {max(rxs):.3f}], Y: [{min(rys):.3f}, {max(rys):.3f}]")

    print("\n2. CONTROLLED DECODING EXPERIMENTS")
    load_model(device="cpu")
    model = get_model()
    tokenizer = get_tokenizer()

    # Experiment A: Baseline beam=4 (default)
    prep_a = preprocess_pose(raw_kps, max_length=500, use_motion=True, device="cpu")
    translator = PoseTranslator(auto_load=True, device="cpu")

    for beams in [1, 2, 4, 8]:
        res = translator.predict_with_metadata(prep_a, num_beams=beams, length_penalty=2.0)
        print(f"  Beam Width = {beams:2d} -> \"{res.text}\" (latency: {res.latency_ms:.1f}ms)")

    # Experiment B: Length penalty variants (at beam=4)
    print("\n3. LENGTH PENALTY EXPERIMENTS (num_beams=4)")
    for lp in [0.5, 1.0, 1.5, 2.0]:
        res = translator.predict_with_metadata(prep_a, num_beams=4, length_penalty=lp)
        print(f"  Length Penalty = {lp:.1f} -> \"{res.text}\"")

    # Experiment C: Motion vs Spatial-only
    print("\n4. SPATIAL VS MOTION REPRESENTATION")
    # For spatial-only, checkpoint is t5-small (pose_dim=300), but we are restricted to t5-small-motion.
    # What if velocity is zeroed out or smoothed?
    kps_motion_smooth = raw_kps.copy()
    # Linear interpolation of missing left hand positions
    kps_interp = raw_kps.copy()
    # Left hand channels are 33*4 .. 54*4 = 132 .. 216
    lh_slice = slice(132, 216)
    # Check if forward-fill / linear interpolation of missing frames changes output
    for c in range(132, 216):
        col = kps_interp[:, c]
        # find non-zero indices
        nz = np.where(col != 0.0)[0]
        if len(nz) > 1:
            # Interpolate
            xp = nz
            fp = col[nz]
            kps_interp[:, c] = np.interp(np.arange(total_frames), xp, fp)

    prep_interp = preprocess_pose(kps_interp, max_length=500, use_motion=True, device="cpu")
    res_interp = translator.predict_with_metadata(prep_interp, num_beams=4, length_penalty=2.0)
    print(f"  With LH Interpolation (no sudden 0-teleportation) -> \"{res_interp.text}\"")

    # Experiment D: Top-K tokens at step 1
    print("\n5. TOP FIRST-TOKEN LOGITS ANALYSIS")
    with torch.inference_mode():
        # Encode pose
        encoder_outputs = model.t5.encoder(
            inputs_embeds=model.pose_encoder(prep_a.pose_tensor),
            attention_mask=torch.ones((1, 500), dtype=torch.long, device="cpu"),
            return_dict=True,
        )
        # Decoder start token is 0 (<pad>)
        decoder_input_ids = torch.tensor([[0]], dtype=torch.long)
        out = model.t5(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )
        logits = out.logits[0, -1]  # (vocab_size,)
        topk = torch.topk(logits, k=10)
        print("  Top 10 candidates for first generated token:")
        for score, tid in zip(topk.values.tolist(), topk.indices.tolist()):
            tok_str = tokenizer.decode([tid])
            print(f"    Token ID {tid:5d} ({tok_str!r:15s}): logit = {score:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
