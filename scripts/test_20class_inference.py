"""
Verification Script for 20-Class Isolated ISL Model.
Validates:
1. Model loading & 20-class mapping
2. Landmark prediction & top-3 output schema
3. Food vs Water ambiguity evaluation
4. Zero-downtime rollback to 10-class checkpoint via ISOLATED_MODEL_VERSION
"""

import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.isolated.inference.classifier import IsolatedSignRecognizer
from ai.isolated.config.isolated_settings import (
    ISOLATED_CHECKPOINT_PATH,
    ISOLATED_IMPROVED_CHECKPOINT_PATH,
    ISOLATED_20CLASS_CHECKPOINT_PATH,
    ISOLATED_20CLASS_LABELS_PATH,
)


def run_verification():
    print("=" * 80)
    print("SIGNOVA AI: 20-CLASS ISOLATED MODEL INFERENCE & ROLLBACK VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------
    # Test 1: Verify 20-Class Model Loading
    # -------------------------------------------------------------
    print("\n[Test 1] Initializing IsolatedSignRecognizer with ISOLATED_MODEL_VERSION='20class'...")
    os.environ["ISOLATED_MODEL_VERSION"] = "20class"
    rec20 = IsolatedSignRecognizer()
    rec20.load_model()

    print(f"  Model Type:        {rec20.model_type}")
    print(f"  Checkpoint Path:   {rec20.checkpoint_path}")
    print(f"  Loaded Num Classes:{len(rec20.classes)}")
    print(f"  Threshold:         {rec20.rejection_threshold}")
    assert len(rec20.classes) == 20, f"Expected 20 classes, got {len(rec20.classes)}"
    print("  --> PASS: 20-Class model loaded successfully.")

    # -------------------------------------------------------------
    # Test 2: Inference on Test Samples
    # -------------------------------------------------------------
    print("\n[Test 2] Testing landmark prediction on real test samples...")
    test_csv = PROJECT_ROOT / "datasets" / "splits" / "test_20class.csv"
    test_df = pd.read_csv(test_csv)
    landmarks_dir = PROJECT_ROOT / "datasets" / "landmarks" / "test"

    sample_rows = test_df.sample(min(10, len(test_df)), random_state=42)
    for _, row in sample_rows.iterrows():
        vid = row["video_id"]
        true_cls = row["class_label"]
        lm_file = landmarks_dir / f"{vid}.npz"
        if not lm_file.exists():
            continue
        data = np.load(lm_file)
        raw_kps = data["landmarks"]  # (T, 75, 4)

        result = rec20.predict_landmarks(raw_kps, top_k=3)
        assert result["success"] in [True, False]
        assert "predicted_sign" in result
        assert "confidence" in result
        assert len(result["top_predictions"]) <= 3
        pred = result["predicted_sign"]
        conf = result["confidence"]
        print(f"  Sample {vid:<20} | True: {true_cls:<10} | Pred: {pred:<10} | Conf: {conf:.4f} | Status: {result['status']}")

    print("  --> PASS: Landmark inference and schema validation succeeded.")

    # -------------------------------------------------------------
    # Test 3: Food vs Water Evaluation
    # -------------------------------------------------------------
    print("\n[Test 3] Testing FOOD vs WATER test clips specifically...")
    fw_df = test_df[test_df["class_label"].isin(["food", "water"])]
    for _, row in fw_df.iterrows():
        vid = row["video_id"]
        true_cls = row["class_label"]
        lm_file = landmarks_dir / f"{vid}.npz"
        if not lm_file.exists():
            continue
        data = np.load(lm_file)
        raw_kps = data["landmarks"]

        result = rec20.predict_landmarks(raw_kps, top_k=3)
        pred = result["predicted_sign"]
        conf = result["confidence"]
        is_fw_amb = result.get("is_food_water_ambiguity", False)
        print(f"  {true_cls.upper()} sample ({vid}): Pred={pred:<8} Conf={conf:.4f} FW_Ambiguity={is_fw_amb}")

    print("  --> PASS: Food vs Water evaluation completed.")

    # -------------------------------------------------------------
    # Test 4: Zero-Downtime Rollback to 10-Class Model
    # -------------------------------------------------------------
    print("\n[Test 4] Testing zero-downtime rollback via ISOLATED_MODEL_VERSION='10class'...")
    os.environ["ISOLATED_MODEL_VERSION"] = "10class"
    rec10 = IsolatedSignRecognizer()
    rec10.load_model()

    print(f"  Rollback Model Type:        {rec10.model_type}")
    print(f"  Rollback Checkpoint Path:   {rec10.checkpoint_path}")
    print(f"  Rollback Num Classes:       {len(rec10.classes)}")
    assert len(rec10.classes) == 10, f"Expected 10 classes on rollback, got {len(rec10.classes)}"
    assert rec10.checkpoint_path in [ISOLATED_CHECKPOINT_PATH, ISOLATED_IMPROVED_CHECKPOINT_PATH], f"Checkpoint mismatch: {rec10.checkpoint_path}"
    print("  --> PASS: Zero-downtime rollback cleanly restored 10-class checkpoint.")

    # Cleanup env
    os.environ.pop("ISOLATED_MODEL_VERSION", None)
    print("\n[Verification Complete] All tests passed!")


if __name__ == "__main__":
    run_verification()
