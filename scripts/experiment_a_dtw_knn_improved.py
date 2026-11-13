"""
Experiment A Improved: Dynamic Time Warping + K-Nearest Neighbors (DTW + KNN).

Task 2.2A Sprint Deliverable:
- Group-wise normalization:
  - Spatial (0:225): Active std scaling without center shift (zero stays 0.0)
  - Velocity (225:450): Active std scaling without center shift
  - Acceleration (450:675): 3-sigma clipping + active std scaling
  - Presence masks (675:678): Preserved
- Group-partitioned PCA: 24 spatial + 16 velocity + 8 acceleration = 48 dims
- 504 train exemplars (canonical + duplicates) with explicit alias map {49: 46, 63: 14}
- Constrained Sakoe-Chiba warping window tuning (5, 8, 10, 15)
- Group-weighted DTW distance: spatial vs velocity vs acceleration weights
- Neighbor counts k in {1, 3, 5, 7} with inverse-distance voting
- Preserves baseline checkpoint: models/isl_csltr_continuous/best_dtw_knn_model.pkl (untouched)
- Saves improved champion model to: models/isl_csltr_continuous/best_dtw_knn_improved_model.pkl
- Saves comprehensive evaluation report to: models/isl_csltr_continuous/eval_exp_a_dtw_knn_improved.json
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
from sklearn.decomposition import PCA

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier, compute_dtw_distance
from ai.continuous.sentence_dataset import ISLCSLTRSentenceDataset
from scripts.eval_utils import compute_benchmark_metrics

# Explicit alias-to-canonical mapping for verified duplicate video sentences
ALIAS_MAP: Dict[int, int] = {
    49: 46,  # 'i dont agree' -> 'i do not agree' (shares identical video)
    63: 14,  # 'No need to worry dont worry' -> 'do not worry' (shares identical video)
}


def get_active_channel_std(arr: np.ndarray, min_std: float = 1e-4) -> np.ndarray:
    """Compute standard deviation of active (non-zero) elements per feature channel."""
    stds = np.ones(arr.shape[1], dtype=np.float32)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        active = col[col != 0.0]
        if len(active) > 10:
            s = float(np.std(active))
            stds[j] = max(s, min_std)
        else:
            stds[j] = 1.0
    return stds


def extract_and_normalize_features() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract, group-wise normalize, and PCA-project train and validation sequences.
    Fitted strictly on train data only.
    """
    cache_dir = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    improved_cache_path = cache_dir / "dtw_dataset_improved.pkl"
    preprocessors_path = cache_dir / "preprocessors_improved.pkl"

    manifests_dir = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "manifests"

    print("Loading train (504) and validation (85) datasets...")
    ds_train = ISLCSLTRSentenceDataset(
        manifest_path=manifests_dir / "train_manifest.json",
        canonical_only=False,
        cache_features=True,
        augment=False,
    )
    ds_val = ISLCSLTRSentenceDataset(
        manifest_path=manifests_dir / "validation_manifest.json",
        canonical_only=False,
        cache_features=True,
        augment=False,
    )

    train_raw = [ds_train[i]["features"].numpy() for i in range(len(ds_train))]
    val_raw = [ds_val[i]["features"].numpy() for i in range(len(ds_val))]

    # Map training labels to canonical using explicit ALIAS_MAP
    y_train = np.array([ALIAS_MAP.get(int(ds_train[i]["label"]), int(ds_train[i]["label"])) for i in range(len(ds_train))], dtype=np.int64)
    y_val = np.array([int(ds_val[i]["label"]) for i in range(len(ds_val))], dtype=np.int64)

    print(f"Loaded {len(train_raw)} train sequences and {len(val_raw)} validation sequences.")

    # 2x temporal stride downsampling
    train_sub = [seq[::2] for seq in train_raw]
    val_sub = [seq[::2] for seq in val_raw]

    # Concatenate all training frames for group-wise statistics
    train_frames = np.concatenate(train_sub, axis=0)

    # 1. Inspect and extract group features
    # Spatial: 0:225, Velocity: 225:450, Acceleration: 450:675, Mask: 675:678
    tr_spatial = train_frames[:, 0:225]
    tr_vel = train_frames[:, 225:450]
    tr_acc = train_frames[:, 450:675]

    # Compute group statistics on train active elements only
    print("Computing active standard deviations on train split...")
    spatial_std = get_active_channel_std(tr_spatial)
    vel_std = get_active_channel_std(tr_vel)
    acc_std = get_active_channel_std(tr_acc)

    # Acceleration 3-sigma thresholds per channel
    acc_threshold_3sigma = 3.0 * acc_std

    # Normalize train frames per group
    tr_spatial_norm = tr_spatial / spatial_std
    tr_vel_norm = tr_vel / vel_std
    tr_acc_clipped = np.clip(tr_acc, -acc_threshold_3sigma, acc_threshold_3sigma)
    tr_acc_norm = tr_acc_clipped / acc_std

    # Fit Group-partitioned PCA: 24 spatial + 16 velocity + 8 acceleration = 48 dims
    print("Fitting Group-partitioned PCA strictly on train split...")
    pca_spatial = PCA(n_components=24, random_state=42)
    pca_spatial.fit(tr_spatial_norm)

    pca_vel = PCA(n_components=16, random_state=42)
    pca_vel.fit(tr_vel_norm)

    pca_acc = PCA(n_components=8, random_state=42)
    pca_acc.fit(tr_acc_norm)

    print(f"PCA Explained Variance - Spatial(24): {pca_spatial.explained_variance_ratio_.sum():.4f}, "
          f"Vel(16): {pca_vel.explained_variance_ratio_.sum():.4f}, "
          f"Acc(8): {pca_acc.explained_variance_ratio_.sum():.4f}")

    def project_sequence(seq: np.ndarray) -> np.ndarray:
        s = seq[:, 0:225] / spatial_std
        v = seq[:, 225:450] / vel_std
        a_clip = np.clip(seq[:, 450:675], -acc_threshold_3sigma, acc_threshold_3sigma)
        a = a_clip / acc_std

        s_pca = pca_spatial.transform(s)
        v_pca = pca_vel.transform(v)
        a_pca = pca_acc.transform(a)
        return np.concatenate([s_pca, v_pca, a_pca], axis=-1).astype(np.float32)

    train_projected = [project_sequence(s) for s in train_sub]
    val_projected = [project_sequence(s) for s in val_sub]

    dtw_dataset = {
        "train": {
            "seqs": train_projected,
            "labels": y_train,
            "num_samples": len(train_projected),
        },
        "validation": {
            "seqs": val_projected,
            "labels": y_val,
            "num_samples": len(val_projected),
        },
        "group_dims": {
            "spatial": (0, 24),
            "velocity": (24, 40),
            "acceleration": (40, 48),
            "total": 48,
        },
        "alias_map": ALIAS_MAP,
    }

    preprocessors = {
        "spatial_std": spatial_std,
        "vel_std": vel_std,
        "acc_std": acc_std,
        "acc_threshold_3sigma": acc_threshold_3sigma,
        "pca_spatial": pca_spatial,
        "pca_vel": pca_vel,
        "pca_acc": pca_acc,
    }

    with open(improved_cache_path, "wb") as f:
        pickle.dump(dtw_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(preprocessors_path, "wb") as f:
        pickle.dump(preprocessors, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Cached improved feature representations to {improved_cache_path}")
    return dtw_dataset, preprocessors


def evaluate_predictions_with_aliases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    alias_map: Dict[int, int],
    num_classes: int = 101,
    train_time_sec: float = 0.0,
    inference_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute benchmark metrics applying explicit alias-to-canonical equivalence.
    If true label is 46 and predicted is 49 (or true 14 and predicted 63), it is evaluated as a match.
    """
    y_true_mapped = np.array([alias_map.get(int(y), int(y)) for y in y_true], dtype=np.int64)
    y_pred_mapped = np.array([alias_map.get(int(y), int(y)) for y in y_pred], dtype=np.int64)

    # Re-map posterior probabilities by merging alias probability mass
    remapped_probs = y_probs.copy()
    for alias_cls, canon_cls in alias_map.items():
        if alias_cls < num_classes and canon_cls < num_classes:
            remapped_probs[:, canon_cls] += remapped_probs[:, alias_cls]
            remapped_probs[:, alias_cls] = 0.0

    metrics = compute_benchmark_metrics(
        y_true=y_true_mapped,
        y_pred=y_pred_mapped,
        y_probs=remapped_probs,
        num_classes=num_classes,
        train_time_sec=train_time_sec,
        inference_time_sec=inference_time_sec,
    )
    return metrics


def main() -> None:
    process = psutil.Process()
    mem_start_mb = process.memory_info().rss / (1024 * 1024)

    models_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== TASK 2.2A: IMPROVED DTW + KNN SPRINT ===")
    dtw_data, _ = extract_and_normalize_features()

    train_feats = dtw_data["train"]["seqs"]
    y_train = dtw_data["train"]["labels"]
    val_feats = dtw_data["validation"]["seqs"]
    y_val = dtw_data["validation"]["labels"]

    print(f"Ready for evaluation: {len(train_feats)} train exemplars, {len(val_feats)} val sequences.")
    mem_after_load_mb = process.memory_info().rss / (1024 * 1024)

    results: Dict[str, Any] = {
        "experiment": "Task 2.2A Improvement Sprint: DTW + KNN",
        "description": "Group-wise normalized features + 48-dim Group PCA + Constrained Window + Group-Weighted DTW",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_split": {
            "train_samples": len(train_feats),
            "val_samples": len(val_feats),
            "classes": 101,
            "alias_mapping": ALIAS_MAP,
        },
        "system_profile": {
            "initial_memory_mb": round(mem_start_mb, 2),
            "memory_after_data_load_mb": round(mem_after_load_mb, 2),
        },
        "baseline_recorded": {
            "config": "cosine_k5",
            "top1_accuracy": 0.0471,
            "top3_accuracy": 0.0941,
            "macro_f1": 0.0368,
            "unique_predicted_classes": 24,
            "most_frequently_predicted_class": "42",
            "most_frequently_predicted_count": 19,
        },
        "configurations": {},
    }

    # Experimental parameter search space
    # 1. Warping windows: [5, 8, 10, 15]
    # 2. Group weights: (spatial, velocity, accel)
    # 3. Metrics: ['cosine', 'euclidean']
    # 4. k values: [1, 3, 5, 7]
    window_candidates = [5, 8, 10, 15]
    group_weight_candidates = [
        ("balanced_low_accel", (1.0, 0.5, 0.2)),
        ("spatial_heavy", (1.0, 0.3, 0.1)),
        ("equal_groups", (1.0, 1.0, 1.0)),
        ("no_group_weighting", None),
    ]
    metrics_list = ["cosine", "euclidean"]
    k_list = [1, 3, 5, 7]

    best_config_name = ""
    best_top1 = -1.0
    best_f1 = -1.0
    best_metrics = None
    best_clf = None

    # Step through configurations systematically
    for metric in metrics_list:
        for gw_name, gw_weights in group_weight_candidates:
            for window in window_candidates:
                # Ponytail optimization: skip unnecessary Euclidean variants if cosine consistently dominates
                if metric == "euclidean" and (window != 10 or gw_weights is not None):
                    continue

                config_family = f"{metric}_win{window}_{gw_name}"
                print(f"\n--- Computing Pairwise DTW Matrix: {config_family} ---")

                clf = DTWKNNClassifier(
                    n_neighbors=5,
                    metric=metric,
                    window=window,
                    weighting="distance",
                    num_classes=101,
                    group_weights=gw_weights,
                )
                clf.fit(train_feats, y_train)

                t0 = time.time()
                dist_matrix = clf.compute_distance_matrix(val_feats, max_workers=8)
                dtw_calc_time = time.time() - t0
                mem_after_dtw_mb = process.memory_info().rss / (1024 * 1024)

                print(f"Computed {len(val_feats)}x{len(train_feats)} DTW in {dtw_calc_time:.2f}s | RSS: {mem_after_dtw_mb:.2f} MB")

                for k in k_list:
                    t_infer_0 = time.time()
                    y_pred, y_probs = clf.predict_from_distances(dist_matrix, k=k)
                    total_infer_time = (time.time() - t_infer_0) + dtw_calc_time
                    latency_per_sample_ms = (total_infer_time / len(val_feats)) * 1000.0

                    # Evaluate with alias handling
                    metrics = evaluate_predictions_with_aliases(
                        y_true=y_val,
                        y_pred=y_pred,
                        y_probs=y_probs,
                        alias_map=ALIAS_MAP,
                        num_classes=101,
                        train_time_sec=0.0,
                        inference_time_sec=total_infer_time,
                    )

                    config_key = f"{config_family}_k{k}"
                    pred_dist = metrics["prediction_distribution"]
                    top_pred_class = max(pred_dist.keys(), key=lambda c: pred_dist[c]) if pred_dist else "none"
                    top_pred_count = pred_dist.get(top_pred_class, 0)

                    results["configurations"][config_key] = {
                        "distance_metric": metric,
                        "window": window,
                        "group_weights": gw_weights,
                        "group_weights_name": gw_name,
                        "k": k,
                        "weighting": "distance" if k > 1 else "uniform",
                        "top1_accuracy": round(metrics["top1_accuracy"], 4),
                        "top3_accuracy": round(metrics["top3_accuracy"], 4),
                        "macro_f1": round(metrics["macro_f1"], 4),
                        "unique_predicted_classes": metrics["unique_predicted_classes"],
                        "most_frequently_predicted_class": top_pred_class,
                        "most_frequently_predicted_count": top_pred_count,
                        "prediction_distribution": pred_dist,
                        "total_inference_time_sec": round(total_infer_time, 2),
                        "latency_per_sample_ms": round(latency_per_sample_ms, 2),
                        "per_class": metrics["per_class"],
                        "confusion_matrix": metrics["confusion_matrix"],
                    }

                    print(
                        f"Config {config_key:35s} | Top-1: {metrics['top1_accuracy']*100:5.2f}% | "
                        f"Top-3: {metrics['top3_accuracy']*100:5.2f}% | "
                        f"Macro F1: {metrics['macro_f1']:6.4f} | "
                        f"Unique: {metrics['unique_predicted_classes']:2d}/85 | "
                        f"Most freq: cls {top_pred_class} ({top_pred_count}) | "
                        f"Latency: {latency_per_sample_ms:5.1f}ms"
                    )

                    # Model Selection Priority: 1. Top-1, 2. Macro F1, 3. Top-3, 4. Unique classes
                    is_better = False
                    if metrics["top1_accuracy"] > best_top1:
                        is_better = True
                    elif metrics["top1_accuracy"] == best_top1:
                        if metrics["macro_f1"] > best_f1:
                            is_better = True
                        elif metrics["macro_f1"] == best_f1:
                            if metrics["top3_accuracy"] > (best_metrics["top3_accuracy"] if best_metrics else 0):
                                is_better = True

                    if is_better:
                        best_top1 = metrics["top1_accuracy"]
                        best_f1 = metrics["macro_f1"]
                        best_config_name = config_key
                        best_metrics = metrics
                        best_clf = DTWKNNClassifier(
                            n_neighbors=k,
                            metric=metric,
                            window=window,
                            weighting="distance",
                            num_classes=101,
                            group_weights=gw_weights,
                        )
                        best_clf.fit(train_feats, y_train)

    print("\n" + "=" * 80)
    print(f"CHAMPION MODEL SELECTED: {best_config_name}")
    print("=" * 80)
    print(f"Top-1 Accuracy: {best_metrics['top1_accuracy']*100:.2f}% (Baseline: 4.71%)")
    print(f"Top-3 Accuracy: {best_metrics['top3_accuracy']*100:.2f}% (Baseline: 9.41%)")
    print(f"Macro F1:       {best_metrics['macro_f1']:.4f} (Baseline: 0.0368)")
    print(f"Unique Preds:   {best_metrics['unique_predicted_classes']}/85 (Baseline: 24)")

    # Save Champion Model separately
    champion_path = models_dir / "best_dtw_knn_improved_model.pkl"
    best_clf.save(champion_path)
    print(f"\nSaved improved champion model checkpoint to {champion_path}")

    # Save Evaluation results separately
    results["best_configuration"] = best_config_name
    best_cfg_info = results["configurations"][best_config_name]
    results["best_validation_metrics"] = {
        "config": best_config_name,
        "distance_metric": best_cfg_info["distance_metric"],
        "window": best_cfg_info["window"],
        "group_weights": best_cfg_info["group_weights"],
        "k": best_cfg_info["k"],
        "top1_accuracy": best_cfg_info["top1_accuracy"],
        "top3_accuracy": best_cfg_info["top3_accuracy"],
        "macro_f1": best_cfg_info["macro_f1"],
        "unique_predicted_classes": best_cfg_info["unique_predicted_classes"],
        "most_frequently_predicted_class": best_cfg_info["most_frequently_predicted_class"],
        "most_frequently_predicted_count": best_cfg_info["most_frequently_predicted_count"],
        "latency_per_sample_ms": best_cfg_info["latency_per_sample_ms"],
    }

    eval_out_path = models_dir / "eval_exp_a_dtw_knn_improved.json"
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved improved evaluation report to {eval_out_path}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'SPRINT BENCHMARK COMPARISON: BASELINE vs IMPROVED DTW+KNN':^80}")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Original Baseline':<20} | {'Improved Model':<20} | {'Delta':<10}")
    print("-" * 80)
    b_top1 = 0.0471
    b_top3 = 0.0941
    b_f1 = 0.0368
    b_uniq = 24
    i_top1 = best_metrics["top1_accuracy"]
    i_top3 = best_metrics["top3_accuracy"]
    i_f1 = best_metrics["macro_f1"]
    i_uniq = best_metrics["unique_predicted_classes"]

    d_top1 = (i_top1 - b_top1) * 100
    d_top3 = (i_top3 - b_top3) * 100
    d_f1 = i_f1 - b_f1
    d_uniq = i_uniq - b_uniq

    print(f"{'Configuration':<32} | {'Cosine DTW k=5':<20} | {best_config_name:<20} | {'-':<10}")
    print(f"{'Validation Top-1 Acc':<32} | {b_top1*100:6.2f}%              | {i_top1*100:6.2f}%              | {d_top1:+6.2f}%")
    print(f"{'Validation Top-3 Acc':<32} | {b_top3*100:6.2f}%              | {i_top3*100:6.2f}%              | {d_top3:+6.2f}%")
    print(f"{'Validation Macro F1':<32} | {b_f1:6.4f}              | {i_f1:6.4f}              | {d_f1:+6.4f}")
    print(f"{'Unique Classes Predicted':<32} | {b_uniq:2d} / 85               | {i_uniq:2d} / 85               | {d_uniq:+3d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
