"""
Tests for MediaPipe Concurrency, State Isolation, and Thread Safety.
Verifies Phase 7.5 state isolation requirements:
1. Independent hand history across two requests (no shared instance state).
2. Concurrent left/right hand assignment without crosstalk.
3. Genuine two-handed gesture stability.
4. Repeated requests on the same extractor instance.
5. Concurrent video inference without exceptions.
6. MediaPipe timestamp safety across concurrent video streams.
7. Regression verification for single-request behavior.
"""

import concurrent.futures
from pathlib import Path
import threading
import time
from typing import Any, List
import numpy as np
import pandas as pd
import pytest
import sklearn
import torch

from ai.pose_extraction.mediapipe_extractor import (
    ModernMediaPipeExtractor,
    _assign_hands_anatomically,
)
from backend.app.services.isolated_inference_service import isolated_inference_service


class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MockCategory:
    def __init__(self, category_name: str, score: float = 0.99):
        self.category_name = category_name
        self.score = score


# ---------------------------------------------------------------------------
# Test 1: Independent Hand History Across Two Requests
# ---------------------------------------------------------------------------
def test_1_independent_hand_history_across_requests():
    """Verify two requests maintain isolated hand history and instance has no shared state."""
    extractor = ModernMediaPipeExtractor()

    # Ensure instance does not have shared _hand_history attribute
    assert not hasattr(extractor, "_hand_history") or getattr(extractor, "_hand_history", None) is None

    history_a: List[str] = []
    history_b: List[str] = []

    h_right = [MockLandmark(0.7, 0.7) for _ in range(21)]
    h_left = [MockLandmark(0.3, 0.7) for _ in range(21)]

    # Stream A: Right-handed frames (MediaPipe Category "Left" -> Anatomical Right)
    for _ in range(5):
        _assign_hands_anatomically([h_right], [[MockCategory("Left")]], None, hand_history=history_a)

    # Stream B: Left-handed frames (MediaPipe Category "Right" -> Anatomical Left)
    for _ in range(5):
        _assign_hands_anatomically([h_left], [[MockCategory("Right")]], None, hand_history=history_b)

    # Histories must be completely isolated
    assert len(history_a) == 5
    assert len(history_b) == 5
    assert all(h == "right" for h in history_a)
    assert all(h == "left" for h in history_b)

    # Extractor instance itself must NOT store either history
    assert getattr(extractor, "_hand_history", None) is None


# ---------------------------------------------------------------------------
# Test 2: Concurrent Left/Right Hand Assignment
# ---------------------------------------------------------------------------
def test_2_concurrent_hand_assignment_no_crosstalk():
    """Verify concurrent threads running hand assignment do not cross-pollinate hysteresis."""
    h_right = [MockLandmark(0.7, 0.7) for _ in range(21)]
    h_left = [MockLandmark(0.3, 0.7) for _ in range(21)]

    def run_stream_right():
        hist: List[str] = []
        results = []
        for i in range(10):
            # Frame 4 has an inverted glitch category
            cat = "Right" if i == 4 else "Left"
            lh, rh = _assign_hands_anatomically([h_right], [[MockCategory(cat)]], None, hand_history=hist)
            results.append((lh is None, rh is not None))
            time.sleep(0.001)
        return hist, results

    def run_stream_left():
        hist: List[str] = []
        results = []
        for i in range(10):
            # Frame 4 has an inverted glitch category
            cat = "Left" if i == 4 else "Right"
            lh, rh = _assign_hands_anatomically([h_left], [[MockCategory(cat)]], None, hand_history=hist)
            results.append((lh is not None, rh is None))
            time.sleep(0.001)
        return hist, results

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_right = executor.submit(run_stream_right)
        f_left = executor.submit(run_stream_left)

        hist_right, res_right = f_right.result()
        hist_left, res_left = f_left.result()

    # Right stream should have right assignments for all frames including glitch (due to hysteresis)
    assert all(r == (True, True) for r in res_right)
    # Left stream should have left assignments for all frames including glitch
    assert all(l == (True, True) for l in res_left)

    # Check that raw choices recorded are independent
    assert hist_right.count("right") == 9
    assert hist_right.count("left") == 1  # only the glitch frame
    assert hist_left.count("left") == 9
    assert hist_left.count("right") == 1  # only the glitch frame


# ---------------------------------------------------------------------------
# Test 3: Genuine Two-Handed Gesture Stability
# ---------------------------------------------------------------------------
def test_3_two_handed_gesture_stability():
    """Verify genuine two-handed gestures are never collapsed or corrupted under concurrency."""
    pose = [MockLandmark(0.5, 0.5) for _ in range(33)]
    pose[15] = MockLandmark(0.3, 0.7, visibility=0.9)  # Left wrist
    pose[16] = MockLandmark(0.7, 0.7, visibility=0.9)  # Right wrist

    h_r = [MockLandmark(0.69, 0.71) for _ in range(21)]
    h_l = [MockLandmark(0.31, 0.69) for _ in range(21)]

    def run_two_hand_stream():
        hist: List[str] = []
        for _ in range(8):
            lh, rh = _assign_hands_anatomically(
                [h_r, h_l],
                [[MockCategory("Left")], [MockCategory("Right")]],
                pose,
                hand_history=hist,
            )
            assert lh is not None and rh is not None
            assert lh == h_l and rh == h_r
        return hist

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_two_hand_stream) for _ in range(4)]
        for f in futures:
            hist = f.result()
            assert len(hist) == 8
            assert all(entry == "both" for entry in hist)


# ---------------------------------------------------------------------------
# Test 4: Repeated Requests on Single Extractor
# ---------------------------------------------------------------------------
def test_4_repeated_requests_on_cached_extractor():
    """Verify repeated video requests on a shared extractor instance execute cleanly and reproducibly."""
    extractor = ModernMediaPipeExtractor()
    demo_video = Path("Signova AI/datasets/demo_samples/hello.mp4")
    if not demo_video.exists():
        pytest.skip(f"Demo video missing: {demo_video}")

    # Run request 1
    kps1, meta1 = extractor.extract_isign_video(demo_video)
    # Run request 2 on same extractor
    kps2, meta2 = extractor.extract_isign_video(demo_video)

    assert kps1.shape == kps2.shape
    assert meta1["read_frames"] == meta2["read_frames"]
    # Numerical reproducibility
    np.testing.assert_allclose(kps1, kps2, atol=1e-5)
    # Extractor must remain clean
    assert getattr(extractor, "_hand_history", None) is None


# ---------------------------------------------------------------------------
# Test 5: Concurrent Video Inference Without Exceptions
# ---------------------------------------------------------------------------
def test_5_concurrent_inference_without_exceptions():
    """Verify concurrent calls to isolated_inference_service execute without failure or crash."""
    video_food = Path("Signova AI/datasets/demo_samples/food.mp4")
    video_hello = Path("Signova AI/datasets/demo_samples/hello.mp4")
    if not (video_food.exists() and video_hello.exists()):
        pytest.skip("Demo videos not available")

    bytes_food = video_food.read_bytes()
    bytes_hello = video_hello.read_bytes()

    isolated_inference_service.load_model()

    def run_req(idx, video_bytes, name):
        return isolated_inference_service.predict_video_file(video_bytes, name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run_req, 0, bytes_food, "food.mp4")
        f2 = executor.submit(run_req, 1, bytes_hello, "hello.mp4")

        res_food = f1.result()
        res_hello = f2.result()

    assert res_food["success"] is True
    assert res_food["predicted_sign"] == "food"
    assert res_hello["success"] is True
    assert res_hello["predicted_sign"] == "hello"


# ---------------------------------------------------------------------------
# Test 6: MediaPipe Timestamp Safety Across Streams
# ---------------------------------------------------------------------------
def test_6_mediapipe_timestamp_safety():
    """Verify concurrent video extractions maintain independent monotonic timestamps without collision."""
    extractor = ModernMediaPipeExtractor()
    demo_video = Path("Signova AI/datasets/demo_samples/hello.mp4")
    if not demo_video.exists():
        pytest.skip("Demo video missing")

    def extract_worker(w_id):
        # Each worker processes the video stream starting from timestamp 0
        kps, meta = extractor.extract_isign_video(demo_video, max_frames=15)
        return w_id, kps.shape, meta["read_frames"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(extract_worker, 1)
        f2 = executor.submit(extract_worker, 2)

        id1, shape1, frames1 = f1.result()
        id2, shape2, frames2 = f2.result()

    assert frames1 == 15
    assert frames2 == 15
    assert shape1 == shape2


# ---------------------------------------------------------------------------
# Test 7: Regression Against Single-Request Behavior
# ---------------------------------------------------------------------------
def test_7_regression_single_request_behavior():
    """Verify single-request recognition confidence and output format are completely preserved."""
    video_food = Path("Signova AI/datasets/demo_samples/food.mp4")
    if not video_food.exists():
        pytest.skip("Demo food video missing")

    isolated_inference_service.load_model()
    bytes_food = video_food.read_bytes()

    result = isolated_inference_service.predict_video_file(bytes_food, "food.mp4")

    assert result["success"] is True
    assert result["predicted_sign"] == "food"
    assert result["confidence"] > 0.85
    assert "top_predictions" in result
    assert len(result["top_predictions"]) >= 1
    assert result["status"] in ("recognized", "confirmed")
