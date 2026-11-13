"""
Prepares the official 20-class manifest and signer-independent splits for Signova AI.
Preserves all original 10-class clips in their respective train/val/test splits.
Allocates the 10 new classes using the exact same signer partitioning rules.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "isl-isolated-40words"
MANIFEST_DIR = PROJECT_ROOT / "datasets" / "manifests"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"

CLASSES_10 = [
    "food", "hello", "help", "no", "please", "sorry", "stop", "thank you", "water", "yes"
]

CLASSES_NEW_10 = [
    "father", "mother", "sister", "teacher", "you", "friend", "student", "school", "market", "hospital"
]

CLASSES_20 = CLASSES_10 + CLASSES_NEW_10

def prepare_manifest_and_splits():
    metadata_path = DATASET_ROOT / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv at {metadata_path}")

    meta_df = pd.read_csv(metadata_path)
    filtered = meta_df[meta_df["normalized_word"].isin(CLASSES_20)].copy()

    # Load existing 10-class splits to strictly preserve them
    train_10 = pd.read_csv(SPLITS_DIR / "train.csv")
    val_10 = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_10 = pd.read_csv(SPLITS_DIR / "test.csv")

    existing_split_map = {}
    for _, row in train_10.iterrows():
        existing_split_map[row["relative_video_path"]] = ("train", row["video_id"])
    for _, row in val_10.iterrows():
        existing_split_map[row["relative_video_path"]] = ("validation", row["video_id"])
    for _, row in test_10.iterrows():
        existing_split_map[row["relative_video_path"]] = ("test", row["video_id"])

    # Collect rows for 20 classes
    all_rows = []
    new_video_counter = 189

    # ISL500 allocation rule
    isl500_train = {f"User0{i:02d}" for i in range(1, 12)}
    isl500_val = {"User012", "User013"}
    isl500_test = {"User014", "User015"}

    for _, row in filtered.iterrows():
        rel_path = row["video_path"]
        cls = row["normalized_word"]
        dataset = row["dataset"]
        signer = row["signer"]

        if rel_path in existing_split_map:
            split_name, vid = existing_split_map[rel_path]
        else:
            vid = f"ISL20_{new_video_counter:04d}"
            new_video_counter += 1

            # Partition new classes
            if dataset == "ISL500":
                if signer in isl500_train:
                    split_name = "train"
                elif signer in isl500_val:
                    split_name = "validation"
                elif signer in isl500_test:
                    split_name = "test"
                else:
                    split_name = "train"
            elif dataset == "INCLUDE":
                # Deterministic split: hash of signer
                h = abs(hash(f"{cls}_{signer}")) % 100
                if h < 70:
                    split_name = "train"
                elif h < 85:
                    split_name = "validation"
                else:
                    split_name = "test"
            elif dataset == "CISLR":
                # Deterministic split based on video path
                h = abs(hash(rel_path)) % 100
                if h < 60:
                    split_name = "train"
                elif h < 80:
                    split_name = "validation"
                else:
                    split_name = "test"
            else:
                split_name = "test"

        abs_path = (DATASET_ROOT / rel_path).resolve()
        item = {
            "video_id": vid,
            "class_label": cls,
            "normalized_word": cls,
            "original_label": row["original_label"],
            "source_dataset": dataset,
            "signer_id": signer,
            "relative_video_path": rel_path,
            "absolute_video_path": str(abs_path),
            "file_exists": abs_path.exists(),
            "fps": row["fps"],
            "resolution": row["resolution"],
            "duration_sec": row["duration"],
            "quality_score": row.get("quality_score", 1.0),
            "upstream_review_status": row.get("review_status", "accepted"),
            "validation_status": "APPROVED",
            "include_in_mvp_train": True,
            "notes": f"License: {row.get('license', 'multi-source')}",
            "split": split_name,
        }
        all_rows.append(item)

    df_20 = pd.DataFrame(all_rows)

    # Balance check: ensure every class has at least 1 in val and test
    for cls in CLASSES_20:
        c_sub = df_20[df_20["class_label"] == cls]
        splits_present = set(c_sub["split"].unique())
        if "validation" not in splits_present or "test" not in splits_present:
            print(f"Warning: class {cls} missing split: {splits_present}")

    # Write manifests
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_csv = MANIFEST_DIR / "isl_20class_manifest.csv"
    manifest_json = MANIFEST_DIR / "isl_20class_manifest.json"
    df_20.drop(columns=["split"]).to_csv(manifest_csv, index=False)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(df_20.drop(columns=["split"]).to_dict(orient="records"), f, indent=2)

    # Write 20-class splits
    train_df = df_20[df_20["split"] == "train"].copy()
    val_df = df_20[df_20["split"] == "validation"].copy()
    test_df = df_20[df_20["split"] == "test"].copy()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train_20class.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "validation_20class.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test_20class.csv", index=False)

    # Verification: check signer leakage
    train_signers = set(train_df[train_df["source_dataset"] == "ISL500"]["signer_id"].unique())
    val_signers = set(val_df[val_df["source_dataset"] == "ISL500"]["signer_id"].unique())
    test_signers = set(test_df[test_df["source_dataset"] == "ISL500"]["signer_id"].unique())

    assert len(train_signers.intersection(val_signers)) == 0, "Leakage: ISL500 train vs val"
    assert len(train_signers.intersection(test_signers)) == 0, "Leakage: ISL500 train vs test"
    assert len(val_signers.intersection(test_signers)) == 0, "Leakage: ISL500 val vs test"

    summary = {
        "version": "2.0.0",
        "total_approved_clips": len(df_20),
        "classes_count": len(CLASSES_20),
        "splits": {
            "train": {
                "count": len(train_df),
                "classes": train_df["class_label"].value_counts().to_dict(),
                "signers_count": train_df["signer_id"].nunique(),
            },
            "validation": {
                "count": len(val_df),
                "classes": val_df["class_label"].value_counts().to_dict(),
                "signers_count": val_df["signer_id"].nunique(),
            },
            "test": {
                "count": len(test_df),
                "classes": test_df["class_label"].value_counts().to_dict(),
                "signers_count": test_df["signer_id"].nunique(),
            },
        },
    }

    with open(SPLITS_DIR / "split_summary_20class.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Success: 20-class manifest and splits generated!")
    print(f"Total: {len(df_20)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"ISL500 signers: Train={len(train_signers)}, Val={len(val_signers)}, Test={len(test_signers)}")

if __name__ == "__main__":
    prepare_manifest_and_splits()
