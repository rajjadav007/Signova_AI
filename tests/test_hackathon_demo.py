"""
Self-check unit test for Signova AI Hackathon Demo Mode.
Validates 5-class reference integrity, DTW recognition, and unknown input rejection.
"""

import json
from pathlib import Path
import pickle
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_recognize_sentence import load_demo_artifacts, recognize_sequence


def test_hackathon_demo_integrity():
    demo_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "hackathon_demo"
    classes, ref_meta, ref_seqs, ref_labels = load_demo_artifacts(demo_dir)

    # 1. Scope checks
    assert len(classes) == 5, f"Expected 5 classes, got {len(classes)}"
    assert len(ref_meta) == 10, f"Expected 10 reference videos, got {len(ref_meta)}"
    assert len(ref_seqs) == 10, f"Expected 10 reference sequences, got {len(ref_seqs)}"
    assert len(ref_labels) == 10, f"Expected 10 reference labels, got {len(ref_labels)}"

    # 2. Split preservation check
    manifest_path = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "manifests" / "manifest.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_lines = f.readlines()
    train_vids = {line.split(",")[0] for line in manifest_lines if ",train," in line}
    test_vids = {line.split(",")[0] for line in manifest_lines if ",test," in line}
    val_vids = {line.split(",")[0] for line in manifest_lines if ",validation," in line}

    for ref in ref_meta:
        vid_id = ref["video_id"]
        assert vid_id in train_vids, f"Ref video {vid_id} must be in train split!"
        assert vid_id not in test_vids, f"Ref video {vid_id} must NOT be in test split!"
        assert vid_id not in val_vids, f"Ref video {vid_id} must NOT be in validation split!"

    # 3. Reference self-recognition check
    for idx, (seq, true_label) in enumerate(zip(ref_seqs, ref_labels)):
        res = recognize_sequence(
            proc_seq=seq,
            ref_seqs=ref_seqs,
            ref_labels=ref_labels,
            classes=classes,
            distance_threshold=0.40,
            min_margin=0.015,
        )
        assert res["is_recognized"] is True
        assert res["predicted_class_id"] == true_label
        assert res["top_distance"] < 1e-4, f"Self-distance should be 0, got {res['top_distance']}"
        assert res["top_similarity"] > 0.999

    # 4. Unknown input rejection check
    # Generate random sequence of length 50 with shape (50, 48)
    rng = np.random.RandomState(42)
    noise_seq = rng.randn(50, 48).astype(np.float32)
    res_noise = recognize_sequence(
        proc_seq=noise_seq,
        ref_seqs=ref_seqs,
        ref_labels=ref_labels,
        classes=classes,
        distance_threshold=0.40,
        min_margin=0.015,
    )
    assert res_noise["is_recognized"] is False
    assert res_noise["status"] == "Sentence not recognized."
    assert res_noise["predicted_sentence"] is None

    print("All Hackathon Demo integrity assertions PASSED!")


if __name__ == "__main__":
    test_hackathon_demo_integrity()
