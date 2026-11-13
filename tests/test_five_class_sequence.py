"""
Unit and Integration Tests for Signova AI Five-Class ISL Sentence Sequence Recognition.
Covers all 15 required verification targets:
1. Dataset directory discovery
2. Numeric frame sorting
3. Image decoding
4. Missing or corrupted image handling
5. MediaPipe landmark extraction
6. Feature dimensions
7. Temporal sequence ordering
8. Variable-length sequence padding
9. Sequence masking
10. Correct class mapping
11. Model forward pass
12. Model output dimensions
13. Inference on one complete sequence
14. Checkpoint loading
15. Reproducible preprocessing
"""

import json
from pathlib import Path
import re
import sys
import tempfile
import cv2
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.five_class_model import FiveClassBiGRUAttentionModel
from ai.continuous.five_class_sequence_recognizer import (
    FiveClassSequenceRecognizer,
    natural_sort_key
)
from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor
from scripts.extract_five_class_features import normalize_torso_landmarks

DATASET_ROOT = Path(r"D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Sentence_Level")
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "five_class_sequence"


# 1. Dataset directory discovery
def test_01_dataset_directory_discovery():
    assert DATASET_ROOT.exists(), f"Dataset root directory missing: {DATASET_ROOT}"
    expected_classes = ["comb your hair", "help me", "i am fine. thank you sir", "i need water", "who are you"]
    for c in expected_classes:
        cdir = DATASET_ROOT / c
        assert cdir.exists(), f"Class directory missing: {cdir}"
        for s in ["1", "2", "3", "4", "5", "6", "7"]:
            assert (cdir / s).is_dir(), f"Signer directory {s} missing in {cdir}"


# 2. Numeric frame sorting
def test_02_numeric_frame_sorting():
    unsorted_names = [
        Path("frame (10).jpg"),
        Path("frame (2).jpg"),
        Path("frame (1).jpg"),
        Path("frame (20).jpg")
    ]
    sorted_paths = sorted(unsorted_names, key=natural_sort_key)
    sorted_names = [p.name for p in sorted_paths]
    assert sorted_names == ["frame (1).jpg", "frame (2).jpg", "frame (10).jpg", "frame (20).jpg"], (
        f"Incorrect numeric sort order: {sorted_names}"
    )


# 3. Image decoding
def test_03_image_decoding():
    sample_img_path = DATASET_ROOT / "comb your hair" / "1" / "comb your hair 01.jpg"
    assert sample_img_path.exists()
    img = cv2.imread(str(sample_img_path))
    assert img is not None, "Failed to decode valid image"
    assert len(img.shape) == 3 and img.shape[2] == 3, f"Unexpected image shape: {img.shape}"


# 4. Missing or corrupted image handling
def test_04_missing_or_corrupted_image_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        corrupted_file = tmp_path / "corrupted.jpg"
        corrupted_file.write_bytes(b"not a real jpeg image garbage content")
        
        img = cv2.imread(str(corrupted_file))
        assert img is None, "Corrupted image should return None from cv2.imread"
        
        non_existent = tmp_path / "does_not_exist.jpg"
        img_missing = cv2.imread(str(non_existent))
        assert img_missing is None, "Missing file should return None"


# 5. MediaPipe landmark extraction
def test_05_mediapipe_landmark_extraction():
    extractor = ModernMediaPipeExtractor()
    sample_img_path = DATASET_ROOT / "comb your hair" / "1" / "comb your hair 01.jpg"
    bgr = cv2.imread(str(sample_img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    kps, info = extractor.extract_isign_frame(rgb)
    assert kps.shape == (75, 4), f"Expected (75, 4) landmark shape, got {kps.shape}"
    assert "pose" in info and "left_hand" in info and "right_hand" in info


# 6. Feature dimensions
def test_06_feature_dimensions():
    extractor = ModernMediaPipeExtractor()
    sample_img_path = DATASET_ROOT / "comb your hair" / "1" / "comb your hair 01.jpg"
    bgr = cv2.imread(str(sample_img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    raw_kps, _ = extractor.extract_isign_frame(rgb)
    norm_kps = normalize_torso_landmarks(raw_kps)
    flat_feat = norm_kps.reshape(-1)
    
    assert flat_feat.shape == (300,), f"Expected 300 features per frame, got {flat_feat.shape}"
    assert flat_feat.dtype == np.float32


# 7. Temporal sequence ordering
def test_07_temporal_sequence_ordering():
    sdir = DATASET_ROOT / "help me" / "1"
    files = sorted([f for f in sdir.glob("*.jpg")], key=natural_sort_key)
    assert len(files) == 11
    
    extracted_numbers = []
    for f in files:
        m = re.search(r'\((\d+)\)', f.name)
        if m:
            extracted_numbers.append(int(m.group(1)))
    assert extracted_numbers == list(range(1, 12)), f"Order mismatch: {extracted_numbers}"


# 8. Variable-length sequence padding
def test_08_variable_length_sequence_padding():
    seq1 = torch.ones(15, 300)
    seq2 = torch.ones(25, 300) * 2.0
    
    lengths = [len(seq1), len(seq2)]
    max_len = max(lengths)
    
    padded = torch.zeros(2, max_len, 300)
    padded[0, :lengths[0]] = seq1
    padded[1, :lengths[1]] = seq2
    
    assert padded.shape == (2, 25, 300)
    assert torch.all(padded[0, 15:] == 0.0), "Padded steps must be zero"
    assert torch.all(padded[1, :25] == 2.0)


# 9. Sequence masking
def test_09_sequence_masking():
    lengths = [10, 20]
    max_len = 20
    mask = torch.zeros(2, max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
        
    assert mask[0, :10].all() and (~mask[0, 10:]).all()
    assert mask[1, :20].all()
    
    model = FiveClassBiGRUAttentionModel()
    x = torch.randn(2, 20, 300)
    _, attn = model(x, mask=mask)
    
    assert torch.allclose(attn[0, 10:], torch.zeros(10), atol=1e-6), "Masked steps must receive zero attention"
    assert torch.isclose(attn[0, :10].sum(), torch.tensor(1.0), atol=1e-5), "Attention weights must sum to 1.0"


# 10. Correct class mapping
def test_10_correct_class_mapping():
    mapping_file = ARTIFACTS_DIR / "class_mapping.json"
    assert mapping_file.exists(), f"Missing class mapping: {mapping_file}"
    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    classes = data["classes"]
    assert len(classes) == 5
    expected_sentences = [
        "comb your hair",
        "help me",
        "i am fine. thank you sir.",
        "i need water",
        "who are you"
    ]
    for i, exp in enumerate(expected_sentences):
        assert classes[i]["class_id"] == i
        assert classes[i]["sentence"] == exp


# 11. Model forward pass
def test_11_model_forward_pass():
    model = FiveClassBiGRUAttentionModel()
    model.eval()
    x = torch.randn(2, 18, 300)
    mask = torch.ones(2, 18, dtype=torch.bool)
    with torch.no_grad():
        logits, attn = model(x, mask=mask)
    assert logits is not None and attn is not None


# 12. Model output dimensions
def test_12_model_output_dimensions():
    model = FiveClassBiGRUAttentionModel(num_classes=5)
    model.eval()
    batch_size = 3
    seq_len = 22
    x = torch.randn(batch_size, seq_len, 300)
    with torch.no_grad():
        logits, attn = model(x)
    assert logits.shape == (batch_size, 5), f"Expected logits (3, 5), got {logits.shape}"
    assert attn.shape == (batch_size, seq_len), f"Expected attn (3, 22), got {attn.shape}"


# 13. Inference on one complete sequence
def test_13_inference_on_one_complete_sequence():
    recognizer = FiveClassSequenceRecognizer(model_dir=ARTIFACTS_DIR)
    dummy_seq = np.random.randn(20, 300).astype(np.float32)
    result = recognizer.predict_features(dummy_seq)
    
    assert "predicted_class_id" in result
    assert "predicted_sentence" in result
    assert "class_probabilities" in result
    assert 0 <= result["predicted_class_id"] < 5
    assert result["frame_count"] == 20
    assert result["latency_ms"] > 0.0
    assert len(result["class_probabilities"]) == 5


# 14. Checkpoint loading
def test_14_checkpoint_loading():
    checkpoint_path = ARTIFACTS_DIR / "best_model.pt"
    assert checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = FiveClassBiGRUAttentionModel()
    model.load_state_dict(state_dict)
    assert model is not None


# 15. Reproducible preprocessing
def test_15_reproducible_preprocessing():
    kps = np.random.uniform(0.1, 0.9, size=(75, 4)).astype(np.float32)
    # Ensure shoulders are present
    kps[11, 3] = 0.9
    kps[12, 3] = 0.9
    
    norm1 = normalize_torso_landmarks(kps)
    norm2 = normalize_torso_landmarks(kps)
    assert np.allclose(norm1, norm2), "Preprocessing must be deterministic and reproducible"
