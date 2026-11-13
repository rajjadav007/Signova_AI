"""
Signova AI - Five-Class ISL Sentence Sequence Recognition CLI.
Usage:
    python scripts/recognize_five_class.py --folder "path/to/frames_folder"
    python scripts/recognize_five_class.py --npz "path/to/sequence.npz"
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from ai.continuous.five_class_sequence_recognizer import FiveClassSequenceRecognizer


def main():
    parser = argparse.ArgumentParser(description="Recognize ISL Sentence Sequence using FiveClassBiGRUAttentionModel")
    parser.add_argument("--folder", type=str, default=None, help="Path to folder containing sequence JPG frames")
    parser.add_argument("--npz", type=str, default=None, help="Path to pre-extracted .npz sequence file")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to model artifacts directory")
    args = parser.parse_args()

    recognizer = FiveClassSequenceRecognizer(model_dir=args.model_dir)

    if args.folder:
        folder_path = Path(args.folder)
        print(f"Loading frames from: {folder_path}...")
        result = recognizer.predict_folder(folder_path)
    elif args.npz:
        npz_path = Path(args.npz)
        print(f"Loading features from: {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        feats = data["features"]
        result = recognizer.predict_features(feats)
    else:
        # Default demo sequence: test signer 2 "comb your hair"
        default_folder = Path(r"D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Sentence_Level\comb your hair\2")
        print(f"No input provided. Running on default test sequence: {default_folder}")
        result = recognizer.predict_folder(default_folder)

    print("\n" + "=" * 50)
    print("FIVE-CLASS SEQUENCE RECOGNITION RESULT")
    print("=" * 50)
    print(f"Predicted Class ID:   {result['predicted_class_id']}")
    print(f"Predicted Sentence:   '{result['predicted_sentence']}'")
    print(f"Confidence:           {result['confidence'] * 100:.2f}%")
    print(f"Frames Processed:     {result['frame_count']}")
    print(f"Inference Latency:    {result['latency_ms']:.2f} ms")
    if "total_pipeline_time_ms" in result:
        print(f"Total Pipeline Time:  {result['total_pipeline_time_ms']:.2f} ms (Extraction + Preproc + Inference)")

    print("\nClass Probabilities:")
    for sent, prob in result["class_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {sent:28s}: {prob*100:5.1f}% {bar}")


if __name__ == "__main__":
    main()
