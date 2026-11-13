"""
Build 5-Class Reference Dataset for Signova AI Hackathon Demo Mode.

Selected 5 Sentences:
- SID 5: 'comb your hair' (COMB YOU HAIR)
- SID 23: 'help me' (HELP ME)
- SID 35: 'i am fine. thank you sir' (I FINE. THANK YOU SIR)
- SID 53: 'i need water' (I NEED WATER)
- SID 91: 'who are you' (WHO YOU)

Picks exactly 2 canonical training videos per class (10 reference videos total).
Extracts / loads landmarks, applies standard temporal feature preprocessing,
and fits & saves a standalone DTWKNNClassifier model artifact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier
from scripts.recognize_sentence_dtw import extract_features_from_npz, preprocess_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("signova.build_demo_references")

TARGET_SIDS = [5, 23, 35, 53, 91]


def build_references() -> None:
    manifest_path = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "manifests" / "manifest.csv"
    classes_json_path = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "sentence_classes.json"
    output_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "hackathon_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    with open(classes_json_path, "r", encoding="utf-8") as f:
        all_classes = {c["sentence_id"]: c for c in json.load(f)["classes"]}

    reference_records: List[Dict[str, Any]] = []
    ref_seqs: List[np.ndarray] = []
    ref_labels: List[int] = []

    # Map original SIDs to demo class index (0..4) and keep original SID
    demo_classes_info: List[Dict[str, Any]] = []
    for demo_id, sid in enumerate(TARGET_SIDS):
        c_info = all_classes[sid]
        demo_classes_info.append({
            "demo_class_id": demo_id,
            "original_sentence_id": sid,
            "sentence": c_info["sentence"],
            "isl_gloss": c_info["isl_gloss"],
            "tokens": c_info["tokens"],
        })

        train_vids = df[(df["sentence_id"] == sid) & (df["split"] == "train")].copy()
        train_vids = train_vids.sort_values(by="missing_landmark_percentage", ascending=True)
        top2 = train_vids.iloc[:2]

        if len(top2) < 2:
            raise ValueError(f"Class {sid} has fewer than 2 training videos!")

        for r_idx, (_, row) in enumerate(top2.iterrows()):
            npz_rel = row["npz_relative_path"]
            npz_abs = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / npz_rel
            raw_feat = extract_features_from_npz(npz_abs)
            proc_seq = preprocess_sequence(raw_feat)

            ref_seqs.append(proc_seq)
            ref_labels.append(demo_id)

            rec = {
                "demo_class_id": demo_id,
                "original_sentence_id": sid,
                "sentence": c_info["sentence"],
                "isl_gloss": c_info["isl_gloss"],
                "reference_index": r_idx + 1,
                "signer_id": int(row["signer_id"]),
                "video_id": row["video_id"],
                "video_filename": row["video_filename"],
                "canonical_video_path": row["canonical_video_path"],
                "missing_landmark_percentage": float(row["missing_landmark_percentage"]),
                "duration_seconds": float(row["duration_seconds"]),
                "feature_shape": list(proc_seq.shape),
            }
            reference_records.append(rec)
            logger.info(
                "Added Ref: DemoID %d (SID %d) '%s' | Signer %d | %s | shape %s",
                demo_id, sid, c_info["sentence"], row["signer_id"], row["video_filename"], proc_seq.shape
            )

    # 1. Save demo classes JSON
    classes_out_path = output_dir / "demo_classes.json"
    with open(classes_out_path, "w", encoding="utf-8") as f:
        json.dump({"classes": demo_classes_info, "num_classes": len(demo_classes_info)}, f, indent=2)
    logger.info("Saved %s", classes_out_path)

    # 2. Save reference metadata JSON
    meta_out_path = output_dir / "reference_metadata.json"
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump({"references": reference_records, "total_references": len(reference_records)}, f, indent=2)
    logger.info("Saved %s", meta_out_path)

    # 3. Save reference dataset pickle
    ref_dataset_path = output_dir / "reference_dataset.pkl"
    with open(ref_dataset_path, "wb") as f:
        pickle.dump({
            "classes": demo_classes_info,
            "references": reference_records,
            "sequences": ref_seqs,
            "labels": np.array(ref_labels, dtype=np.int64),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved %s", ref_dataset_path)

    # 4. Fit & save DTWKNNClassifier
    clf = DTWKNNClassifier(
        n_neighbors=1,
        metric="cosine",
        window=10,
        weighting="uniform",
        num_classes=len(TARGET_SIDS),
    )
    clf.fit(ref_seqs, ref_labels)
    model_out_path = output_dir / "demo_dtw_knn_model.pkl"
    clf.save(model_out_path)
    logger.info("Fitted and saved model to %s", model_out_path)


if __name__ == "__main__":
    build_references()
