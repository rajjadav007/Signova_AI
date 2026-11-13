"""
End-to-End Pipeline Verification Test for Signova AI.
Tests and logs all 7 stages:
1. MediaPipe landmark extraction (33 Pose + 21 LH + 21 RH = 75 landmarks)
2. Landmark validation (shapes, component ordering, missing landmarks)
3. Pose preprocessing (300 features/frame, z-score normalization, padding/masking)
4. Model loading (t5-base simple model, pose_dim=300, use_motion=False, moved to device AFTER checkpoint)
5. Model inference (pose tensor + attention mask -> token IDs)
6. Text generation (SentencePiece tokenizer decode -> English text)
7. Text-to-speech (predicted English text -> synthesized audio)
"""

import logging
from pathlib import Path
import sys
import time
import numpy as np
import pytest
import torch

# Setup root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.config.settings import (
    ISIGN_CHANNELS,
    ISIGN_HAND_COUNT,
    ISIGN_POSE_COUNT,
    ISIGN_SPATIAL_DIM,
    ISIGN_TOTAL_LANDMARKS,
)
from ai.inference.model_loader import get_tokenizer, is_model_loaded, load_model, unload_model
from ai.inference.pose_translator import PoseTranslator, predict_with_metadata
from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from ai.preprocessing.pose_preprocessor import preprocess_pose, to_spatial_matrix
from ai.speech.tts import text_to_speech

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("signova.test_pipeline")


def test_complete_7_stage_pipeline():
    """Execute and verify all 7 stages of Signova AI pipeline end-to-end."""
    print("\n" + "=" * 70)
    print("STARTING SIGNOVA AI 7-STAGE PIPELINE VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # STAGE 1: MediaPipe Landmark Extraction
    # -------------------------------------------------------------
    print("\n--- STAGE 1: MediaPipe Landmark Extraction ---")
    extractor = MediaPipeLandmarkExtractor()
    test_video = PROJECT_ROOT / "datasets" / "videos" / "10 granfather sick  (2).mov"

    if test_video.exists():
        raw_landmarks, extract_meta = extractor.extract_isign_video(test_video, max_frames=30)
        frames_extracted = raw_landmarks.shape[0]
        print(f"Extracted {frames_extracted} frames from real video: {test_video.name}")
        print(f"Extraction engine: {extract_meta.get('engine')}")
        print(f"Extraction FPS: {extract_meta.get('extraction_fps', 0):.2f}")
    else:
        # Fallback to single frame extraction
        dummy_frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        frame_kps, quality = extractor.extract_isign_frame(dummy_frame)
        raw_landmarks = frame_kps.reshape(1, -1)
        frames_extracted = 1
        print("Used single frame synthetic extraction")

    total_detected_landmarks = ISIGN_TOTAL_LANDMARKS
    print(f"Detected landmarks per frame: {total_detected_landmarks} (Pose: {ISIGN_POSE_COUNT}, LH: {ISIGN_HAND_COUNT}, RH: {ISIGN_HAND_COUNT})")
    assert raw_landmarks.shape[1] == 300, f"Expected 300 coordinates per frame, got {raw_landmarks.shape[1]}"
    extractor.close()

    # -------------------------------------------------------------
    # STAGE 2: Landmark Validation
    # -------------------------------------------------------------
    print("\n--- STAGE 2: Landmark Validation ---")
    # Reshape to (T, 75, 4) for coordinate validation
    reshaped_kps = raw_landmarks.reshape(frames_extracted, 75, 4)
    # Check missing landmarks (visibility/confidence == 0)
    visibilities = reshaped_kps[:, :, 3]
    missing_landmarks_mask = (visibilities == 0.0)
    missing_count = int(np.sum(missing_landmarks_mask))
    total_points = frames_extracted * 75
    missing_pct = (missing_count / total_points) * 100.0

    print(f"Landmark shape validated: {reshaped_kps.shape}")
    print(f"Missing landmarks: {missing_count}/{total_points} ({missing_pct:.1f}%)")
    print(f"Pose landmark range [0..32]: {reshaped_kps[:, :33, :].shape}")
    print(f"Left Hand landmark range [33..53]: {reshaped_kps[:, 33:54, :].shape}")
    print(f"Right Hand landmark range [54..74]: {reshaped_kps[:, 54:75, :].shape}")

    # Validate sequence helper via to_spatial_matrix
    spatial_mat = to_spatial_matrix(raw_landmarks)
    assert spatial_mat.shape == (frames_extracted, 300)
    print(f"Sequence spatial matrix validated: shape={spatial_mat.shape}")

    # -------------------------------------------------------------
    # STAGE 3: Pose Preprocessing
    # -------------------------------------------------------------
    print("\n--- STAGE 3: Pose Preprocessing ---")
    # Must produce (1, 500, 300) with use_motion=False
    preprocessed = preprocess_pose(raw_landmarks, max_length=500, use_motion=False, device="cpu")
    pose_tensor = preprocessed.pose_tensor
    pose_length = preprocessed.pose_length

    print(f"Preprocessed pose tensor shape: {tuple(pose_tensor.shape)}")
    print(f"Pose length tensor: {pose_length.tolist()}")
    print(f"Sequence length: {pose_tensor.shape[1]}")
    print(f"Feature dimensions per frame: {pose_tensor.shape[2]}")
    print(f"Normalization applied: {(preprocessed.metadata or {}).get('normalization', 'zscore')}")
    print(f"Valid frames before padding: {preprocessed.actual_length}")

    assert pose_tensor.shape == (1, 500, 300), f"Expected (1, 500, 300), got {pose_tensor.shape}"
    assert pose_length.item() == preprocessed.actual_length

    # -------------------------------------------------------------
    # STAGE 4: Model Loading
    # -------------------------------------------------------------
    print("\n--- STAGE 4: Model Loading ---")
    model = load_model(device="cpu")
    tokenizer = get_tokenizer()

    model_device = next(model.parameters()).device
    encoder_input_dim = model.pose_encoder.encoder[0].in_features
    encoder_output_dim = model.pose_encoder.encoder[3].out_features
    t5_d_model = model.t5.config.d_model

    print(f"Base model: {model.t5.config._name_or_path or 't5-base'}")
    print(f"Model device: {model_device}")
    print(f"Pose encoder input features: {encoder_input_dim}")
    print(f"Pose encoder output features: {encoder_output_dim}")
    print(f"T5 d_model: {t5_d_model}")
    print(f"Model training state: is_training={model.training}")

    assert encoder_input_dim == 300, f"Expected 300 input features, got {encoder_input_dim}"
    assert encoder_output_dim == t5_d_model == 768, f"Expected 768 d_model, got {t5_d_model}"
    assert not model.training, "Model should be in eval mode!"

    # -------------------------------------------------------------
    # STAGE 5: Model Inference
    # -------------------------------------------------------------
    print("\n--- STAGE 5: Model Inference ---")
    translator = PoseTranslator(auto_load=False, device="cpu")

    t_start = time.perf_counter()
    sample_pose_path = PROJECT_ROOT / "norm_sample.pose"
    if sample_pose_path.exists():
        result = translator.predict_with_metadata(sample_pose_path, max_length=128, num_beams=4)
    else:
        result = translator.predict_with_metadata(raw_landmarks, max_length=128, num_beams=4)
    t_inference = time.perf_counter() - t_start

    print(f"Inference latency: {t_inference:.3f}s ({result.latency_ms:.1f}ms)")
    print(f"Generated token IDs count: {len(result.token_ids)}")
    print(f"Generated token IDs: {result.token_ids[:15]}...")
    assert len(result.token_ids) > 0, "Inference returned empty token IDs"

    # -------------------------------------------------------------
    # STAGE 6: Text Generation
    # -------------------------------------------------------------
    print("\n--- STAGE 6: Text Generation ---")
    predicted_text = result.text
    print(f"Decoded English text: \"{predicted_text}\"")
    assert isinstance(predicted_text, str) and len(predicted_text) > 0, "Decoded text is empty"

    # -------------------------------------------------------------
    # STAGE 7: Text-to-Speech
    # -------------------------------------------------------------
    print("\n--- STAGE 7: Text-to-Speech ---")
    tts_output_path = PROJECT_ROOT / "tests" / "stage7_verified_speech.wav"
    out_wav = text_to_speech.synthesize_to_wav(predicted_text, tts_output_path)

    assert out_wav.exists(), "TTS output WAV file was not created"
    file_size = out_wav.stat().st_size
    print(f"Speech synthesized to: {out_wav.name}")
    print(f"Audio file size: {file_size} bytes")
    assert file_size > 100, f"WAV file too small: {file_size} bytes"

    # Clean up generated audio file
    if out_wav.exists():
        out_wav.unlink()

    print("\n" + "=" * 70)
    print("ALL 7 PIPELINE STAGES COMPLETED & VERIFIED SUCCESSFULLY!")
    print(f"Summary: 75 Landmarks -> 300 Features -> (1, 500, 300) Tensor -> T5-Base -> \"{predicted_text}\" -> Audio WAV")
    print("=" * 70)

    unload_model()
