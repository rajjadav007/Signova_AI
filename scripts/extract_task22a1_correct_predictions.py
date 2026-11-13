"""Extract correctly and incorrectly classified validation videos for Task 2.2A.1."""

import json
from pathlib import Path
import pickle
import numpy as np
import sys

PROJECT_ROOT = Path("D:/Projects/Signova AI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier


def main():
    model_path = PROJECT_ROOT / "models/isl_csltr_continuous/best_dtw_knn_task22a1_model.pkl"
    cache_path = PROJECT_ROOT / "datasets/isl_csltr_continuous/ml_cache/dtw_dataset_improved.pkl"
    manifest_path = PROJECT_ROOT / "datasets/isl_csltr_continuous/manifests/validation_manifest.json"
    classes_path = PROJECT_ROOT / "models/isl_csltr_continuous/sentence_classes.json"

    clf = DTWKNNClassifier.load(model_path)
    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    classes_data = json.load(open(classes_path, "r", encoding="utf-8"))["classes"]
    id2sent = {c["sentence_id"]: c["sentence"] for c in classes_data}

    val_feats = data["validation"]["seqs"]
    val_labels = data["validation"]["labels"]

    ALIAS_MAP = {49: 46, 63: 14}
    val_labels_m = np.array([ALIAS_MAP.get(int(y), int(y)) for y in val_labels])

    dist_matrix = clf.compute_distance_matrix(val_feats, max_workers=8)
    y_pred_raw, _ = clf.predict_from_distances(dist_matrix, k=clf.n_neighbors)
    y_pred_m = np.array([ALIAS_MAP.get(int(y), int(y)) for y in y_pred_raw])

    correct = []
    incorrect = []

    for idx, (m_item, y_true_orig, y_true, y_p_orig, y_p) in enumerate(zip(manifest, val_labels, val_labels_m, y_pred_raw, y_pred_m)):
        item_info = {
            "index": idx,
            "video_id": m_item["video_id"],
            "video_filename": m_item["video_filename"],
            "gt_class_id_orig": int(y_true_orig),
            "gt_class_id": int(y_true),
            "pred_class_id_orig": int(y_p_orig),
            "pred_class_id": int(y_p),
            "gt_sentence": id2sent.get(int(y_true), m_item["sentence"]),
            "pred_sentence": id2sent.get(int(y_p), "Unknown"),
            "manifest_sentence": m_item["sentence"],
        }
        if y_true == y_p:
            correct.append(item_info)
        else:
            incorrect.append(item_info)

    print(f"Total validation samples: {len(manifest)}")
    print(f"Total correct: {len(correct)}")
    print(f"Total incorrect: {len(incorrect)}")
    print(f"Sum check: {len(correct) + len(incorrect)} == {len(manifest)}")

    print("\n--- 15 CORRECTLY CLASSIFIED SAMPLES ---")
    for i, c in enumerate(correct, 1):
        print(f"{i}. Video: {c['video_filename']} | GT Class: {c['gt_class_id']} | Pred Class: {c['pred_class_id']} | Sentence: \"{c['gt_sentence']}\"")

    out_data = {
        "summary": {
            "total_samples": len(manifest),
            "correct_count": len(correct),
            "incorrect_count": len(incorrect),
            "accuracy": len(correct) / len(manifest),
        },
        "correct": correct,
        "incorrect": incorrect,
    }

    out_json_path = PROJECT_ROOT / "models/isl_csltr_continuous/task_2_2a_1_validation_predictions_detail.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nSaved predictions detail JSON to: {out_json_path}")


if __name__ == "__main__":
    main()
