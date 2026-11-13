import json
from pathlib import Path

out_dir = Path("Signova AI/models/isl_csltr_continuous/phase_8_5b")
results = {}
for e in ["a", "b", "c"]:
    with open(out_dir / f"exp_{e}" / "results.json", "r", encoding="utf-8") as f:
        results[e.upper()] = json.load(f)

print("=" * 85)
print("EXPERIMENT COMPARISON SUMMARY: PHASE 8.5B")
print("=" * 85)
print(f"{'Metric':<35} | {'Exp A (Baseline)':<15} | {'Exp B (Blank Reg)':<16} | {'Exp C (Rarity Wt)':<16}")
print("-" * 85)
metrics = [
    ("Best Epoch", lambda d: d["best_epoch"]),
    ("Total Epochs Trained", lambda d: d["total_epochs"]),
    ("Training Time (s)", lambda d: d["training_time_seconds"]),
    ("Train Loss (Best Epoch)", lambda d: [h["train_loss"] for h in d["history"] if h["epoch"] == d["best_epoch"]][0]),
    ("Val Loss", lambda d: d["validation_metrics"]["loss"]),
    ("Val WER (%)", lambda d: d["validation_metrics"]["wer"]),
    ("Val TER (%)", lambda d: d["validation_metrics"]["ter"]),
    ("Val Exact Match (%)", lambda d: d["validation_metrics"]["exact_match_rate"]),
    ("Val Vocab Diversity (/20)", lambda d: d["validation_metrics"]["vocabulary_diversity"]),
    ("Val Macro F1 (%)", lambda d: d["validation_metrics"]["macro_f1"]),
    ("Val Blank Posterior Mean", lambda d: d["validation_metrics"]["blank_posterior_mean"]),
    ("Val Interior Nonblank (%)", lambda d: d["validation_metrics"]["quintile_stats"]["interior_nonblank_pct"]),
    ("Val Q5 Nonblank (%)", lambda d: d["validation_metrics"]["quintile_stats"]["Q5_nonblank_pct"]),
    ("Test Loss", lambda d: d["test_metrics"]["loss"]),
    ("Test WER (%)", lambda d: d["test_metrics"]["wer"]),
    ("Test TER (%)", lambda d: d["test_metrics"]["ter"]),
    ("Test Exact Match (%)", lambda d: d["test_metrics"]["exact_match_rate"]),
    ("Test Vocab Diversity (/20)", lambda d: d["test_metrics"]["vocabulary_diversity"]),
    ("Test Macro F1 (%)", lambda d: d["test_metrics"]["macro_f1"]),
    ("Test Blank Posterior Mean", lambda d: d["test_metrics"]["blank_posterior_mean"]),
    ("Test Interior Nonblank (%)", lambda d: d["test_metrics"]["quintile_stats"]["interior_nonblank_pct"]),
    ("Test Q5 Nonblank (%)", lambda d: d["test_metrics"]["quintile_stats"]["Q5_nonblank_pct"]),
]

for label, fn in metrics:
    v_a = fn(results["A"])
    v_b = fn(results["B"])
    v_c = fn(results["C"])
    print(f"{label:<35} | {str(v_a):<15} | {str(v_b):<16} | {str(v_c):<16}")
print("=" * 85)
