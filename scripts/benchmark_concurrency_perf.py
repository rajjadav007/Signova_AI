"""
Performance Benchmark for MediaPipe and Isolated Inference Concurrency.
Tests 1, 2, and 4 concurrent requests.
Records:
- Average latency
- Median latency
- P95 latency
- Throughput (req/s)
- Errors
- Recognition consistency
"""

import concurrent.futures
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.isolated_inference_service import isolated_inference_service

DEMO_VIDEOS = [
    PROJECT_ROOT / "datasets/demo_samples/food.mp4",
    PROJECT_ROOT / "datasets/demo_samples/hello.mp4",
    PROJECT_ROOT / "datasets/demo_samples/help.mp4",
    PROJECT_ROOT / "datasets/demo_samples/no.mp4",
]

def run_single_inference(video_path: Path) -> Dict[str, Any]:
    video_bytes = video_path.read_bytes()
    t0 = time.perf_counter()
    res = isolated_inference_service.predict_video_file(video_bytes, video_path.name)
    dt = time.perf_counter() - t0
    return {
        "video": video_path.name,
        "latency_sec": dt,
        "sign": res.get("predicted_sign"),
        "confidence": res.get("confidence", 0.0),
        "status": res.get("status"),
        "success": res.get("success", False),
    }

def benchmark_level(concurrency: int, total_requests: int = 4) -> Dict[str, Any]:
    print(f"\n--- Benchmarking Concurrency Level: {concurrency} (Total Requests: {total_requests}) ---")
    requests_to_run = [DEMO_VIDEOS[i % len(DEMO_VIDEOS)] for i in range(total_requests)]

    t_wall_start = time.perf_counter()
    results = []

    if concurrency == 1:
        for vp in requests_to_run:
            results.append(run_single_inference(vp))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_single_inference, vp) for vp in requests_to_run]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

    wall_time = time.perf_counter() - t_wall_start
    latencies = [r["latency_sec"] for r in results]
    errors = sum(1 for r in results if not r["success"])

    avg_lat = float(np.mean(latencies))
    med_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = round(total_requests / max(wall_time, 1e-4), 3)

    consistency = {}
    for r in results:
        v = r["video"]
        if v not in consistency:
            consistency[v] = []
        consistency[v].append((r["sign"], round(r["confidence"], 3)))

    metrics = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "wall_time_sec": round(wall_time, 3),
        "avg_latency_sec": round(avg_lat, 3),
        "median_latency_sec": round(med_lat, 3),
        "p95_latency_sec": round(p95_lat, 3),
        "throughput_req_per_sec": throughput,
        "errors": errors,
        "consistency": {k: list(set(v)) for k, v in consistency.items()},
    }

    print(f"  Wall Time:       {metrics['wall_time_sec']}s")
    print(f"  Avg Latency:     {metrics['avg_latency_sec']}s")
    print(f"  Median Latency:  {metrics['median_latency_sec']}s")
    print(f"  P95 Latency:     {metrics['p95_latency_sec']}s")
    print(f"  Throughput:      {metrics['throughput_req_per_sec']} req/s")
    print(f"  Errors:          {metrics['errors']}")
    print(f"  Consistency:     {metrics['consistency']}")
    return metrics

def main():
    print("=" * 70)
    print("SIGNOVA AI PHASE 7.5: CONCURRENCY PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Pre-warm model and extractor
    isolated_inference_service.load_model()
    _ = run_single_inference(DEMO_VIDEOS[0])
    print("Pre-warm complete.")

    benchmarks = []
    for c in [1, 2, 4]:
        metrics = benchmark_level(concurrency=c, total_requests=4)
        benchmarks.append(metrics)

    output_file = PROJECT_ROOT / "docs/reports/mediapipe_concurrency_perf.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    print("\nBenchmark results saved to:", output_file)

if __name__ == "__main__":
    main()
