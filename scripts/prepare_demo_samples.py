"""
Select and prepare 10 verified sample clips (one for each sign) for offline hackathon demonstration.
Uses pre-extracted landmarks from datasets/landmarks to evaluate all samples, selects the highest
confidence representative clip for each sign, and copies the original video to datasets/demo_samples/.
"""

import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.isolated.inference.classifier import IsolatedSignRecognizer

DEMO_DIR = PROJECT_ROOT / "datasets" / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = [
    "food", "hello", "help", "no", "please", 
    "sorry", "stop", "thank you", "water", "yes"
]

DATASET_ROOT = Path(r"D:\Signova AI\dataset\isl-isolated-40words")

def main():
    recognizer = IsolatedSignRecognizer()
    recognizer.load_model()
    print(f"Loaded classifier: {recognizer.model_type} from {recognizer.checkpoint_path}", flush=True)

    extraction_index_path = PROJECT_ROOT / "datasets" / "landmarks" / "extraction_index.csv"
    with open(extraction_index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_samples = [row for row in reader if row.get("extraction_status") == "SUCCESS"]

    print(f"Total extracted landmark sequences: {len(all_samples)}", flush=True)

    demo_results = {}

    for target_class in TARGET_CLASSES:
        class_samples = [s for s in all_samples if s["class_label"] == target_class]
        print(f"\nEvaluating candidates for class: '{target_class}' ({len(class_samples)} available)", flush=True)

        best_sample = None
        best_conf = -1.0
        best_res = None

        for s in class_samples:
            npz_rel = s["npz_path"]
            npz_path = PROJECT_ROOT / npz_rel
            if not npz_path.exists():
                continue

            try:
                data = np.load(npz_path)
                landmarks = data["landmarks"]  # (T, 75, 4)
                res = recognizer.predict_landmarks(landmarks)

                if res["predicted_sign"] == target_class:
                    if res["confidence"] > best_conf:
                        best_conf = res["confidence"]
                        best_sample = s
                        best_res = res
            except Exception as e:
                continue

        # If no sample predicted as target with top-1, check top-3 candidates
        if best_sample is None:
            for s in class_samples:
                npz_rel = s["npz_path"]
                npz_path = PROJECT_ROOT / npz_rel
                if not npz_path.exists():
                    continue
                try:
                    data = np.load(npz_path)
                    landmarks = data["landmarks"]
                    res = recognizer.predict_landmarks(landmarks)
                    for cand in res.get("top_predictions", []):
                        if cand["sign"] == target_class and cand["confidence"] > best_conf:
                            best_conf = cand["confidence"]
                            best_sample = s
                            best_res = res
                except Exception:
                    continue

        if best_sample is not None:
            safe_name = target_class.replace(" ", "_")
            dest_video = DEMO_DIR / f"{safe_name}.mp4"
            src_video = DATASET_ROOT / best_sample["relative_video_path"]
            
            if src_video.exists():
                shutil.copy2(src_video, dest_video)
                dest_exists = True
            else:
                dest_exists = False
                print(f"  [!] Source video not found at: {src_video}", flush=True)

            demo_results[target_class] = {
                "target_class": target_class,
                "video_file": f"{safe_name}.mp4",
                "original_video_id": best_sample["video_id"],
                "source_dataset": best_sample["source_dataset"],
                "split": best_sample["split"],
                "signer_id": best_sample.get("signer_id", "unknown"),
                "frame_count": int(best_sample.get("frame_count", 0)),
                "predicted_sign": best_res["predicted_sign"],
                "confidence": round(best_res["confidence"], 4),
                "status": best_res["status"],
                "top_predictions": best_res.get("top_predictions", [])
            }
            print(f"  -> Selected {best_sample['video_id']} ({best_sample['split']}) | Conf: {best_res['confidence']:.4f} | Pred: {best_res['predicted_sign']} | Video copied: {dest_exists}", flush=True)
        else:
            print(f"  [!] No valid sample found for {target_class}", flush=True)

    manifest_dest = DEMO_DIR / "demo_manifest.json"
    with open(manifest_dest, "w", encoding="utf-8") as f:
        json.dump(demo_results, f, indent=2)

    print(f"\n==========================================", flush=True)
    print(f"Demo manifest saved to: {manifest_dest}", flush=True)
    print(f"Total demo clips prepared: {len(demo_results)}/{len(TARGET_CLASSES)}", flush=True)
    print(f"==========================================", flush=True)

if __name__ == "__main__":
    main()
