"""
Signova AI - Five-Class ISL Sentence Dataset Inspection and Verification.
Phase 1 & Phase 2 verification script.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import re
import sys
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dataset_inspection")

DATASET_ROOT = Path(r"D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Sentence_Level")

CLASSES = [
    {"class_id": 0, "name": "comb your hair", "folder": "comb your hair"},
    {"class_id": 1, "name": "help me", "folder": "help me"},
    {"class_id": 2, "name": "i am fine. thank you sir.", "folder": "i am fine. thank you sir"},
    {"class_id": 3, "name": "i need water", "folder": "i need water"},
    {"class_id": 4, "name": "who are you", "folder": "who are you"},
]

SPLITS = {
    "train": {"1", "3", "4", "5", "6"},
    "validation": {"7"},
    "test": {"2"}
}


def natural_sort_key(s: Path):
    """Sort filenames numerically, e.g. (1), (2), (10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s.name)]


def inspect():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {DATASET_ROOT}")

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "classes": [],
        "split_counts": {"train": 0, "validation": 0, "test": 0},
        "total_sequences": 0,
        "total_frames": 0,
        "sequence_lengths": [],
        "resolutions": set(),
        "corrupted_images": [],
        "ignored_files": [],
        "sequences": []
    }

    per_class_summary = []

    for c in CLASSES:
        cdir = DATASET_ROOT / c["folder"]
        if not cdir.exists():
            raise FileNotFoundError(f"Missing class folder: {cdir}")

        c_info = {
            "class_id": c["class_id"],
            "sentence": c["name"],
            "folder": c["folder"],
            "train_videos": 0,
            "validation_videos": 0,
            "test_videos": 0,
            "total_videos": 0,
            "total_frames": 0,
            "signers": []
        }

        # Check all signer folders
        for signer_str in ["1", "2", "3", "4", "5", "6", "7"]:
            sdir = cdir / signer_str
            if not sdir.exists():
                logger.warning(f"Missing signer {signer_str} for class {c['name']}")
                continue

            all_files = sorted(list(sdir.glob("*")), key=natural_sort_key)
            valid_frames = []
            for f in all_files:
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    valid_frames.append(f)
                else:
                    summary["ignored_files"].append(str(f))

            # Decode check
            for f in valid_frames:
                try:
                    with Image.open(f) as img:
                        summary["resolutions"].add(f"{img.size[0]}x{img.size[1]}")
                except Exception as e:
                    summary["corrupted_images"].append({"file": str(f), "error": str(e)})

            n_frames = len(valid_frames)
            summary["sequence_lengths"].append(n_frames)
            summary["total_frames"] += n_frames
            c_info["total_frames"] += n_frames
            c_info["total_videos"] += 1
            summary["total_sequences"] += 1

            split_name = "train" if signer_str in SPLITS["train"] else ("validation" if signer_str in SPLITS["validation"] else "test")
            summary["split_counts"][split_name] += 1

            if split_name == "train":
                c_info["train_videos"] += 1
            elif split_name == "validation":
                c_info["validation_videos"] += 1
            else:
                c_info["test_videos"] += 1

            seq_id = f"s{c['class_id']:03d}_signer{signer_str}_{re.sub(r'[^a-zA-Z0-9]', '_', c['name'])}"
            seq_record = {
                "sequence_id": seq_id,
                "class_id": c["class_id"],
                "sentence": c["name"],
                "signer_id": int(signer_str),
                "split": split_name,
                "folder_path": str(sdir),
                "frame_count": n_frames,
                "first_frame": valid_frames[0].name if valid_frames else None,
                "last_frame": valid_frames[-1].name if valid_frames else None,
            }
            summary["sequences"].append(seq_record)
            c_info["signers"].append(signer_str)

        per_class_summary.append(c_info)

    summary["classes"] = per_class_summary
    summary["resolutions"] = sorted(list(summary["resolutions"]))
    summary["min_frames_per_sequence"] = min(summary["sequence_lengths"])
    summary["max_frames_per_sequence"] = max(summary["sequence_lengths"])
    summary["avg_frames_per_sequence"] = round(sum(summary["sequence_lengths"]) / len(summary["sequence_lengths"]), 2)
    summary["num_distinct_signers"] = len(set(s["signer_id"] for s in summary["sequences"]))

    # Save summary
    out_dir = Path(r"D:\Projects\Signova AI\datasets\isl_csltr_continuous\metadata")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "five_class_inspection_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Inspection complete. Summary saved to {out_file}")
    print("\n=== PHASE 1: DATASET SUMMARY REPORT ===")
    print("| Sentence | Train videos | Validation videos | Test videos | Total videos | Total frames |")
    print("|---|---:|---:|---:|---:|---:|")
    for c in per_class_summary:
        print(f"| {c['sentence']} | {c['train_videos']} | {c['validation_videos']} | {c['test_videos']} | {c['total_videos']} | {c['total_frames']} |")

    print("\n--- Additional Stats ---")
    print(f"Total training sequences: {summary['split_counts']['train']}")
    print(f"Total validation sequences: {summary['split_counts']['validation']}")
    print(f"Total testing sequences: {summary['split_counts']['test']}")
    print(f"Total sequences: {summary['total_sequences']}")
    print(f"Total JPG frames: {summary['total_frames']}")
    print(f"Min frames per sequence: {summary['min_frames_per_sequence']}")
    print(f"Max frames per sequence: {summary['max_frames_per_sequence']}")
    print(f"Avg frames per sequence: {summary['avg_frames_per_sequence']}")
    print(f"Distinct signers: {summary['num_distinct_signers']}")
    print(f"Resolutions: {summary['resolutions']}")
    print(f"Corrupted images: {len(summary['corrupted_images'])}")
    print(f"Ignored non-image files (e.g. Thumbs.db): {len(summary['ignored_files'])}")


if __name__ == "__main__":
    inspect()
