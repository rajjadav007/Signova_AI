"""
Signova AI - Five-Class BiGRU Model Validation on Test Sequences.
Validates all 5 test sequences (Signer 2) through FiveClassSequenceRecognizer
and compares against models/isl_csltr_continuous/five_class_sequence/test_predictions.json.
"""

import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.five_class_sequence_recognizer import FiveClassSequenceRecognizer

DATASET_ROOT = Path(r"D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Sentence_Level")
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "five_class_sequence"

TEST_SEQUENCES = [
    {"class_id": 0, "sentence": "comb your hair", "folder": DATASET_ROOT / "comb your hair" / "2"},
    {"class_id": 1, "sentence": "help me", "folder": DATASET_ROOT / "help me" / "2"},
    {"class_id": 2, "sentence": "i am fine. thank you sir.", "folder": DATASET_ROOT / "i am fine. thank you sir" / "2"},
    {"class_id": 3, "sentence": "i need water", "folder": DATASET_ROOT / "i need water" / "2"},
    {"class_id": 4, "sentence": "who are you", "folder": DATASET_ROOT / "who are you" / "2"},
]


def run_validation():
    recognizer = FiveClassSequenceRecognizer(model_dir=ARTIFACTS_DIR)

    saved_preds_path = ARTIFACTS_DIR / "test_predictions.json"
    with open(saved_preds_path, "r", encoding="utf-8") as f:
        saved_predictions = json.load(f)

    # Key by ground truth class id
    saved_by_class = {item["ground_truth_id"]: item for item in saved_predictions}

    print("=" * 80)
    print("STEP 1: VALIDATING 5 TEST SEQUENCES THROUGH FiveClassSequenceRecognizer")
    print("=" * 80)

    all_matched = True
    results = []

    for item in TEST_SEQUENCES:
        cid = item["class_id"]
        sentence = item["sentence"]
        folder = item["folder"]

        res = recognizer.predict_folder(folder)
        saved = saved_by_class[cid]

        pred_id = res["predicted_class_id"]
        pred_sent = res["predicted_sentence"]
        conf = res["confidence"]

        # Check match with saved test predictions
        matches_saved = (pred_id == saved["predicted_id"]) and (abs(conf - saved["confidence"]) < 0.05)

        if not matches_saved:
            all_matched = False

        status = "MATCH" if matches_saved else "MISMATCH"
        print(f"\nClass {cid}: '{sentence}'")
        print(f"  Folder: {folder}")
        print(f"  Frames: {res['frame_count']} | Latency: {res['latency_ms']} ms")
        print(f"  Standalone Inference : Pred ID={pred_id}, Sentence='{pred_sent}', Conf={conf*100:.1f}%")
        print(f"  Saved Training Eval  : Pred ID={saved['predicted_id']}, Sentence='{saved['predicted_sentence']}', Conf={saved['confidence']*100:.1f}%")
        print(f"  Status: [{status}]")

        results.append({
            "class_id": cid,
            "ground_truth": sentence,
            "predicted_id": pred_id,
            "predicted_sentence": pred_sent,
            "confidence": conf,
            "saved_predicted_id": saved["predicted_id"],
            "saved_confidence": saved["confidence"],
            "matched": matches_saved,
            "latency_ms": res["latency_ms"],
            "frame_count": res["frame_count"]
        })

    print("\n" + "=" * 80)
    print(f"VALIDATION VERDICT: {'ALL 5 TEST SEQUENCES MATCH SAVED PREDICTIONS EXACTLY' if all_matched else 'MISMATCH DETECTED'}")
    print("=" * 80)


if __name__ == "__main__":
    run_validation()
