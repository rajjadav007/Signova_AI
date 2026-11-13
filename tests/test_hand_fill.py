import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from ai.pose_extraction.mediapipe_extractor import apply_hand_forward_fill

def test_hand_forward_fill_basic():
    # 5 frames, 75 landmarks, 4 channels
    arr = np.zeros((5, 75, 4), dtype=np.float32)

    # Frame 0: LH (slot 33) has valid detection
    arr[0, 33, :3] = [0.1, 0.2, 0.3]
    arr[0, 33, 3] = 1.0

    # Frame 1: Missing LH
    # Frame 2: Missing LH
    # Frame 3: Missing LH (exceeds max_consecutive_frames if max=2)
    # Frame 4: Valid LH
    arr[4, 33, :3] = [0.8, 0.9, 1.0]
    arr[4, 33, 3] = 1.0

    # Body landmark (slot 0) missing on frame 1
    arr[0, 0, :3] = [0.5, 0.5, 0.5]
    arr[0, 0, 3] = 1.0

    filled = apply_hand_forward_fill(arr, max_consecutive_frames=2)

    # Frame 1 & 2 should have coords from Frame 0, but confidence 0.0
    np.testing.assert_allclose(filled[1, 33, :3], [0.1, 0.2, 0.3])
    assert filled[1, 33, 3] == 0.0
    np.testing.assert_allclose(filled[2, 33, :3], [0.1, 0.2, 0.3])
    assert filled[2, 33, 3] == 0.0

    # Frame 3 exceeded max_consecutive_frames (3 > 2), should be 0.0
    np.testing.assert_allclose(filled[3, 33, :], 0.0)

    # Frame 4 retains original detection
    np.testing.assert_allclose(filled[4, 33, :3], [0.8, 0.9, 1.0])
    assert filled[4, 33, 3] == 1.0

    # Body landmark slot 0 on frame 1 must remain 0.0 (NEVER forward-filled)
    np.testing.assert_allclose(filled[1, 0, :], 0.0)

def test_hand_forward_fill_leading_zeros():
    # Frames before first detection must stay 0.0
    arr = np.zeros((3, 75, 4), dtype=np.float32)
    arr[2, 54, :3] = [0.4, 0.5, 0.6]  # RH slot 54 detected at frame 2
    arr[2, 54, 3] = 1.0

    filled = apply_hand_forward_fill(arr, max_consecutive_frames=5)
    np.testing.assert_allclose(filled[0, 54, :], 0.0)
    np.testing.assert_allclose(filled[1, 54, :], 0.0)
    np.testing.assert_allclose(filled[2, 54, :3], [0.4, 0.5, 0.6])

if __name__ == "__main__":
    test_hand_forward_fill_basic()
    test_hand_forward_fill_leading_zeros()
    print("All forward-fill unit tests passed!")
