"""
Test offline demonstration clips in datasets/demo_samples/
Ensures that all 10 signs are recognized correctly by the champion model.
"""

import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "datasets" / "demo_samples"
MANIFEST_PATH = DEMO_DIR / "demo_manifest.json"

TARGET_CLASSES = [
    "food", "hello", "help", "no", "please", 
    "sorry", "stop", "thank you", "water", "yes"
]

@pytest.fixture(scope="module")
def recognizer():
    from ai.isolated.inference.classifier import IsolatedSignRecognizer
    rec = IsolatedSignRecognizer()
    rec.load_model()
    return rec

def test_demo_manifest_completeness():
    assert MANIFEST_PATH.exists(), f"Demo manifest not found at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for cls in TARGET_CLASSES:
        assert cls in data, f"Missing demo class '{cls}' in manifest"
        entry = data[cls]
        assert entry["predicted_sign"] == cls
        assert entry["confidence"] >= 0.90
        video_path = DEMO_DIR / entry["video_file"]
        assert video_path.exists(), f"Video file {video_path} does not exist"
        assert video_path.stat().st_size > 1000, f"Video file {video_path} is too small"

def test_demo_samples_landmark_inference(recognizer):
    """Verify that landmark extraction and model prediction matches for each demo sample."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    extraction_index_path = PROJECT_ROOT / "datasets" / "landmarks" / "extraction_index.csv"
    import csv
    with open(extraction_index_path, "r", encoding="utf-8") as f:
        samples_by_id = {row["video_id"]: row for row in csv.DictReader(f)}

    for cls in TARGET_CLASSES:
        entry = data[cls]
        vid_id = entry["original_video_id"]
        assert vid_id in samples_by_id, f"Video ID {vid_id} not in extraction index"
        
        npz_rel = samples_by_id[vid_id]["npz_path"]
        npz_path = PROJECT_ROOT / npz_rel
        assert npz_path.exists(), f"NPZ file not found: {npz_path}"
        
        landmarks = np.load(npz_path)["landmarks"]
        res = recognizer.predict_landmarks(landmarks)
        
        assert res["predicted_sign"] == cls, f"Expected {cls}, got {res['predicted_sign']}"
        assert res["status"] == "recognized"
        assert res["confidence"] >= 0.75
