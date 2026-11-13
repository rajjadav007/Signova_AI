"""
MediaPipe Landmark Extractor for Signova AI.
Supports:
1. Modern MediaPipe Tasks API (PoseLandmarker + HandLandmarker) in VIDEO and IMAGE modes.
2. Legacy MediaPipe Holistic fallback (LegacyMediaPipeExtractor).
3. Continuous iSign Pose-to-Text: 75 keypoints (33 pose, 21 LH, 21 RH) -> 300 spatial features.
4. Isolated Sign Recognition: 94 keypoints (6 pose, 21 LH, 21 RH, 46 face).
"""

import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    import sys
    if sys.platform.startswith("win"):
        # Prevent MediaPipe C++ LiteRT double-free STATUS_INVALID_HANDLE (0xc0000008) crash on Windows
        try:
            vision.HandLandmarker.__del__ = lambda self: None
            vision.PoseLandmarker.__del__ = lambda self: None
        except Exception:
            pass
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ai.config.settings import (
    DEFAULT_HAND_TASK_PATH,
    DEFAULT_MEDIAPIPE_TASK_PATH,
    DEFAULT_POSE_TASK_PATH,
    FACE_INDICES,
    HAND_LANDMARK_COUNT,
    ISIGN_CHANNELS,
    ISIGN_HAND_COUNT,
    ISIGN_POSE_COUNT,
    ISIGN_SPATIAL_DIM,
    ISIGN_TOTAL_LANDMARKS,
    NUM_KEYPOINTS,
    POSE_INDICES,
)

logger = logging.getLogger("signova.ai.pose_extraction")


# ---------------------------------------------------------------------------
# Handedness and Anatomical Mapping Helper
# ---------------------------------------------------------------------------
def _assign_hands_anatomically(
    detected_hands: List[Any],
    detected_handedness: List[Any],
    pose_landmarks: Optional[List[Any]] = None,
    hand_history: Optional[List[str]] = None,
    history_window: int = 5,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Map detected hands to anatomical (left_hand, right_hand) slots with hysteresis.
    MediaPipe HandLandmarker assumes a selfie mirror view.
    In third-person observer camera:
      - MediaPipe 'Left' is anatomical Right.
      - MediaPipe 'Right' is anatomical Left.
    When pose is present, wrist distance (pose left #15, right #16) provides ground truth.
    When hand_history is provided, a 5-frame majority vote stabilizes single-hand assignments,
    preventing frame-to-frame left/right channel switching while preserving two-hand gestures.
    """
    if not detected_hands:
        if hand_history is not None:
            hand_history.append("none")
        return None, None

    pose_lw = None
    pose_rw = None
    if pose_landmarks and len(pose_landmarks) > 16:
        lw = pose_landmarks[15]
        rw = pose_landmarks[16]
        lw_vis = getattr(lw, "visibility", 1.0)
        rw_vis = getattr(rw, "visibility", 1.0)
        if lw_vis is None or lw_vis > 0.35:
            pose_lw = np.array([lw.x, lw.y], dtype=np.float32)
        if rw_vis is None or rw_vis > 0.35:
            pose_rw = np.array([rw.x, rw.y], dtype=np.float32)

    num_hands = len(detected_hands)

    if num_hands == 1:
        hand = detected_hands[0]
        cat = detected_handedness[0][0].category_name if detected_handedness else "Right"
        hw = np.array([hand[0].x, hand[0].y], dtype=np.float32)

        raw_choice = None
        is_decisive = False

        if pose_lw is not None and pose_rw is not None:
            dist_l = float(np.linalg.norm(hw - pose_lw))
            dist_r = float(np.linalg.norm(hw - pose_rw))
            if dist_l + 0.05 < dist_r:
                raw_choice = "left"
                is_decisive = True
            elif dist_r + 0.05 < dist_l:
                raw_choice = "right"
                is_decisive = True
            else:
                raw_choice = "left" if dist_l < dist_r else "right"
                is_decisive = False
        elif pose_lw is not None and float(np.linalg.norm(hw - pose_lw)) < 0.16:
            raw_choice = "left"
            is_decisive = True
        elif pose_rw is not None and float(np.linalg.norm(hw - pose_rw)) < 0.16:
            raw_choice = "right"
            is_decisive = True
        else:
            # Inverted category fallback (MediaPipe selfie-mirror to third-person)
            raw_choice = "right" if cat == "Left" else "left"
            is_decisive = False

        final_choice = raw_choice

        # Apply 5-frame majority voting if history is maintained
        if hand_history is not None:
            recent_single = [h for h in hand_history[-history_window:] if h in ("left", "right")]
            if recent_single:
                l_count = recent_single.count("left")
                r_count = recent_single.count("right")

                if not is_decisive:
                    if r_count > l_count:
                        final_choice = "right"
                    elif l_count > r_count:
                        final_choice = "left"
                else:
                    # Prevent single-frame glitch against strong unanimous majority
                    if raw_choice == "left" and r_count >= 4 and len(recent_single) >= 4:
                        if pose_lw is not None and pose_rw is not None and abs(dist_l - dist_r) < 0.08:
                            final_choice = "right"
                    elif raw_choice == "right" and l_count >= 4 and len(recent_single) >= 4:
                        if pose_lw is not None and pose_rw is not None and abs(dist_l - dist_r) < 0.08:
                            final_choice = "left"

            hand_history.append(raw_choice)

        if final_choice == "left":
            return hand, None
        return None, hand

    # Two or more hands: Genuine two-hand gesture!
    h0 = detected_hands[0]
    h1 = detected_hands[1]
    hw0 = np.array([h0[0].x, h0[0].y], dtype=np.float32)
    hw1 = np.array([h1[0].x, h1[0].y], dtype=np.float32)

    if hand_history is not None:
        hand_history.append("both")

    if pose_lw is not None and pose_rw is not None:
        cost_normal = float(np.linalg.norm(hw0 - pose_lw) + np.linalg.norm(hw1 - pose_rw))
        cost_swapped = float(np.linalg.norm(hw0 - pose_rw) + np.linalg.norm(hw1 - pose_lw))
        if cost_normal <= cost_swapped:
            return h0, h1
        return h1, h0

    cat0 = detected_handedness[0][0].category_name if detected_handedness else "Right"
    if cat0 == "Left":
        return h1, h0
    return h0, h1


def apply_hand_forward_fill(
    raw_spatial: np.ndarray,
    max_consecutive_frames: int = 15,
) -> np.ndarray:
    """
    Apply temporal forward-fill to missing hand landmarks in a (T, 300) or (T, 75, 4) array.
    Only fills coordinates (x, y, z) for hands missing for <= max_consecutive_frames.
    Preserves confidence channel as 0.0 (unobserved/imputed).
    Body landmarks (0..32) are NEVER forward-filled.
    Frames before first valid detection remain exact zeros.
    """
    orig_shape = raw_spatial.shape
    reshaped = raw_spatial.reshape(-1, ISIGN_TOTAL_LANDMARKS, ISIGN_CHANNELS).copy()
    num_frames = reshaped.shape[0]

    # Hand index ranges: Left Hand 33..53, Right Hand 54..74
    HAND_SLICES = [
        ("left_hand", ISIGN_POSE_COUNT, ISIGN_POSE_COUNT + ISIGN_HAND_COUNT),
        ("right_hand", ISIGN_POSE_COUNT + ISIGN_HAND_COUNT, ISIGN_TOTAL_LANDMARKS),
    ]

    for _, start_idx, end_idx in HAND_SLICES:
        last_valid_coords = None
        consecutive_missing = 0

        for t in range(num_frames):
            hand_conf = reshaped[t, start_idx:end_idx, 3]
            is_detected = bool(np.any(hand_conf > 0.0))

            if is_detected:
                last_valid_coords = reshaped[t, start_idx:end_idx, :3].copy()
                consecutive_missing = 0
            else:
                consecutive_missing += 1
                if last_valid_coords is not None and consecutive_missing <= max_consecutive_frames:
                    # Impute coordinates (x, y, z) only; confidence remains 0.0
                    reshaped[t, start_idx:end_idx, :3] = last_valid_coords
                    reshaped[t, start_idx:end_idx, 3] = 0.0
                else:
                    reshaped[t, start_idx:end_idx, :] = 0.0

    return reshaped.reshape(orig_shape)


# ---------------------------------------------------------------------------
# 1. Modern Tasks Extractor (PoseLandmarker + HandLandmarker)
# ---------------------------------------------------------------------------
class ModernMediaPipeExtractor:
    """
    Modern MediaPipe Tasks API extractor using standalone PoseLandmarker and HandLandmarker.
    Supports VIDEO mode with monotonic timestamps for video and IMAGE mode for single frames.
    """

    def __init__(
        self,
        pose_model_path: Optional[Union[str, Path]] = None,
        hand_model_path: Optional[Union[str, Path]] = None,
        min_pose_detection_confidence: float = 0.5,
        min_hand_landmarks_confidence: float = 0.5,
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe package is not installed.")

        self.pose_model_path = Path(pose_model_path or DEFAULT_POSE_TASK_PATH).resolve()
        self.hand_model_path = Path(hand_model_path or DEFAULT_HAND_TASK_PATH).resolve()

        if not self.pose_model_path.exists():
            raise FileNotFoundError(f"Pose task model not found at: {self.pose_model_path}")
        if not self.hand_model_path.exists():
            raise FileNotFoundError(f"Hand task model not found at: {self.hand_model_path}")

        self.min_pose_confidence = min_pose_detection_confidence
        self.min_hand_confidence = min_hand_landmarks_confidence

        # Lazy IMAGE mode landmarkers for single-frame calls (avoid 110ms startup overhead when unused)
        self._pose_image = None
        self._hand_image = None

    def _ensure_image_landmarkers(self) -> None:
        """Lazily initialize persistent IMAGE mode landmarkers on first single-frame call."""
        if self._pose_image is None:
            pose_options_image = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.pose_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=self.min_pose_confidence,
            )
            self._pose_image = vision.PoseLandmarker.create_from_options(pose_options_image)
        if self._hand_image is None:
            hand_options_image = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.hand_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=self.min_hand_confidence,
            )
            self._hand_image = vision.HandLandmarker.create_from_options(hand_options_image)

    def extract_isign_frame(
        self, rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract 75 landmarks from a single RGB frame using IMAGE mode.
        """
        self._ensure_image_landmarkers()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_result = self._pose_image.detect(mp_image)
        hand_result = self._hand_image.detect(mp_image)

        return self._format_75_landmarks(pose_result, hand_result)

    def extract_isign_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        hand_missing_strategy: str = "zero",
        max_forward_fill_frames: int = 15,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract 75-landmark sequence using dedicated VIDEO mode with monotonic timestamps.
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for video landmark extraction.")

        path_obj = Path(video_path).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Video file not found at: {path_obj}")

        cap = cv2.VideoCapture(str(path_obj))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file: {path_obj}")

        container_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create VIDEO mode landmarkers for this video stream
        pose_options_video = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.pose_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_pose_confidence,
        )
        hand_options_video = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.hand_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.min_hand_confidence,
        )

        pose_video = vision.PoseLandmarker.create_from_options(pose_options_video)
        hand_video = vision.HandLandmarker.create_from_options(hand_options_video)

        frames_spatial_300: List[np.ndarray] = []
        pose_detected_count = 0
        lh_detected_count = 0
        rh_detected_count = 0
        pose_visibilities: List[float] = []
        corrupted_frames = 0
        read_frames = 0
        last_timestamp_ms = -1
        self._hand_history: List[str] = []

        t_start = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None or frame.size == 0:
                    corrupted_frames += 1
                    continue

                # Ensure strictly monotonically increasing timestamp
                calculated_ts = int(round(read_frames * 1000.0 / fps))
                ts_ms = max(calculated_ts, last_timestamp_ms + 1)
                last_timestamp_ms = ts_ms
                read_frames += 1

                # Scale down high-resolution frames (e.g. 1080p, 720p) to 640px width to cut memory bandwidth
                if width > 640:
                    scale = 640.0 / width
                    frame = cv2.resize(frame, (640, int(height * scale)), interpolation=cv2.INTER_AREA)

                # Consistent RGB conversion
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                pose_res = pose_video.detect_for_video(mp_image, ts_ms)
                hand_res = hand_video.detect_for_video(mp_image, ts_ms)

                kps, info = self._format_75_landmarks(pose_res, hand_res)
                frames_spatial_300.append(kps.reshape(-1))

                if info["pose"]:
                    pose_detected_count += 1
                    pose_visibilities.append(info["pose_visibility_mean"])
                if info["left_hand"]:
                    lh_detected_count += 1
                if info["right_hand"]:
                    rh_detected_count += 1

                if max_frames is not None and read_frames >= max_frames:
                    break
        finally:
            cap.release()
            try:
                pose_video.close()
            except Exception:
                pass
            try:
                hand_video.close()
            except Exception:
                pass

        extraction_time = time.perf_counter() - t_start
        extraction_fps = round(read_frames / max(extraction_time, 1e-4), 2)

        if read_frames == 0:
            raise RuntimeError(f"No valid frames could be decoded from video: {path_obj}")

        raw_spatial_array = np.asarray(frames_spatial_300, dtype=np.float32)
        if hand_missing_strategy == "forward_fill":
            raw_spatial_array = apply_hand_forward_fill(
                raw_spatial_array, max_consecutive_frames=max_forward_fill_frames
            )

        pose_rate = round(pose_detected_count / read_frames, 4)
        lh_rate = round(lh_detected_count / read_frames, 4)
        rh_rate = round(rh_detected_count / read_frames, 4)
        avg_pose_vis = float(np.mean(pose_visibilities)) if pose_visibilities else 0.0

        total_slots = read_frames * ISIGN_TOTAL_LANDMARKS
        missing_slots = (
            (read_frames - pose_detected_count) * ISIGN_POSE_COUNT
            + (read_frames - lh_detected_count) * ISIGN_HAND_COUNT
            + (read_frames - rh_detected_count) * ISIGN_HAND_COUNT
        )
        overall_missing_pct = round(missing_slots / total_slots * 100.0, 2)
        missing_hand_pct = round(
            ((read_frames * 2) - (lh_detected_count + rh_detected_count))
            / (read_frames * 2)
            * 100.0,
            2,
        )

        occlusion_warning = (
            abs(lh_rate - rh_rate) > 0.25
            or min(lh_rate, rh_rate) < 0.60
            or avg_pose_vis < 0.60
        )

        metadata: Dict[str, Any] = {
            "video_path": str(path_obj),
            "container_frame_count": container_frame_count,
            "read_frames": read_frames,
            "corrupted_frames": corrupted_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "pose_detection_rate": pose_rate,
            "left_hand_detection_rate": lh_rate,
            "right_hand_detection_rate": rh_rate,
            "missing_hand_percentage": missing_hand_pct,
            "average_pose_visibility": round(avg_pose_vis, 4),
            "missing_landmark_percentage": overall_missing_pct,
            "occlusion_warning": occlusion_warning,
            "raw_spatial_shape": list(raw_spatial_array.shape),
            "engine": "modern_tasks",
            "hand_missing_strategy": hand_missing_strategy,
            "max_forward_fill_frames": max_forward_fill_frames,
            "extraction_fps": extraction_fps,
            "extraction_time_seconds": round(extraction_time, 4),
        }

        return raw_spatial_array, metadata

    def _format_75_landmarks(
        self, pose_result: Any, hand_result: Any
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Assemble pose and hand detection results into the official 75-landmark format.
        Slots 0..32:   Body (33) [x, y, z, visibility]
        Slots 33..53:  Left Hand (21) [x, y, z, confidence]
        Slots 54..74:  Right Hand (21) [x, y, z, confidence]
        Missing components are exact zeros.
        """
        keypoints = np.zeros((ISIGN_TOTAL_LANDMARKS, ISIGN_CHANNELS), dtype=np.float32)
        pose_detected = False
        pose_visibilities: List[float] = []

        pose_lms = None
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            pose_detected = True
            pose_lms = pose_result.pose_landmarks[0]
            for i, lm in enumerate(pose_lms[:ISIGN_POSE_COUNT]):
                vis = getattr(lm, "visibility", 1.0)
                if vis is None:
                    vis = 1.0
                keypoints[i] = [lm.x, lm.y, lm.z, vis]
                pose_visibilities.append(float(vis))

        # Hand assignment with anatomical validation and hysteresis stabilization
        hands = hand_result.hand_landmarks if hand_result.hand_landmarks else []
        handedness = hand_result.handedness if hand_result.handedness else []
        lh, rh = _assign_hands_anatomically(
            hands, handedness, pose_lms, hand_history=getattr(self, "_hand_history", None)
        )

        left_detected = lh is not None
        right_detected = rh is not None

        if left_detected:
            for i, lm in enumerate(lh[:ISIGN_HAND_COUNT]):
                keypoints[ISIGN_POSE_COUNT + i] = [lm.x, lm.y, lm.z, 1.0]

        if right_detected:
            for i, lm in enumerate(rh[:ISIGN_HAND_COUNT]):
                keypoints[ISIGN_POSE_COUNT + ISIGN_HAND_COUNT + i] = [lm.x, lm.y, lm.z, 1.0]

        missing_count = 0
        if not pose_detected:
            missing_count += ISIGN_POSE_COUNT
        if not left_detected:
            missing_count += ISIGN_HAND_COUNT
        if not right_detected:
            missing_count += ISIGN_HAND_COUNT

        avg_pose_vis = float(np.mean(pose_visibilities)) if pose_visibilities else 0.0

        quality_info: Dict[str, Any] = {
            "pose": pose_detected,
            "left_hand": left_detected,
            "right_hand": right_detected,
            "pose_visibility_mean": avg_pose_vis,
            "missing_landmarks_count": missing_count,
            "missing_percentage": round(missing_count / ISIGN_TOTAL_LANDMARKS * 100.0, 2),
        }

        return keypoints, quality_info

    def close(self) -> None:
        # On Windows, explicitly calling .close() on both landmarker C++ instances
        # causes a double-free / STATUS_INVALID_HANDLE (0xc0000008) in LiteRT thread pool.
        # Setting references to None allows Python GC and native destructor to release cleanly.
        self._pose_image = None
        self._hand_image = None


# ---------------------------------------------------------------------------
# 2. Legacy MediaPipe Holistic Extractor (Preserved Fallback)
# ---------------------------------------------------------------------------
class LegacyMediaPipeExtractor:
    """
    Legacy MediaPipe HolisticLandmarker extractor.
    Preserved as fallback implementation for backwards compatibility.
    """

    def __init__(
        self,
        model_asset_path: Optional[Union[str, Path]] = None,
        min_face_detection_confidence: float = 0.5,
        min_pose_detection_confidence: float = 0.5,
        min_hand_landmarks_confidence: float = 0.5,
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe package is not installed.")

        self.model_asset_path = Path(model_asset_path or DEFAULT_MEDIAPIPE_TASK_PATH).resolve()
        if not self.model_asset_path.exists():
            raise FileNotFoundError(f"MediaPipe holistic model asset not found at: {self.model_asset_path}")

        self.options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_asset_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=min_face_detection_confidence,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_hand_landmarks_confidence=min_hand_landmarks_confidence,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(self.options)

    def extract_frame(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
        """Extract 94 landmarks from a single RGB frame for isolated sign recognition."""
        if self._landmarker is None:
            raise RuntimeError("LegacyMediaPipeExtractor is closed.")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        keypoints = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)
        detections = {"pose": False, "left_hand": False, "right_hand": False, "face": False}

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pose_lms = result.pose_landmarks
            detections["pose"] = True
            for out_idx, p_idx in enumerate(POSE_INDICES):
                if p_idx < len(pose_lms):
                    lm = pose_lms[p_idx]
                    vis = getattr(lm, "visibility", 1.0) or 1.0
                    keypoints[out_idx] = [lm.x, lm.y, lm.z, vis]

        if result.left_hand_landmarks and len(result.left_hand_landmarks) > 0:
            detections["left_hand"] = True
            for i, lm in enumerate(result.left_hand_landmarks[:HAND_LANDMARK_COUNT]):
                keypoints[6 + i] = [lm.x, lm.y, lm.z, 1.0]

        if result.right_hand_landmarks and len(result.right_hand_landmarks) > 0:
            detections["right_hand"] = True
            for i, lm in enumerate(result.right_hand_landmarks[:HAND_LANDMARK_COUNT]):
                keypoints[27 + i] = [lm.x, lm.y, lm.z, 1.0]

        if result.face_landmarks and len(result.face_landmarks) > 0:
            detections["face"] = True
            face_lms = result.face_landmarks
            for out_offset, f_idx in enumerate(FACE_INDICES):
                if f_idx < len(face_lms):
                    lm = face_lms[f_idx]
                    keypoints[48 + out_offset] = [lm.x, lm.y, lm.z, 1.0]

        return keypoints, detections

    def extract_isign_frame(
        self, rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract 75 landmarks from a single RGB frame for continuous iSign translation."""
        if self._landmarker is None:
            raise RuntimeError("LegacyMediaPipeExtractor is closed.")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        keypoints = np.zeros((ISIGN_TOTAL_LANDMARKS, ISIGN_CHANNELS), dtype=np.float32)
        pose_detected = False
        left_hand_detected = False
        right_hand_detected = False
        pose_visibilities: List[float] = []

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pose_detected = True
            pose_lms = result.pose_landmarks
            for i, lm in enumerate(pose_lms[:ISIGN_POSE_COUNT]):
                vis = getattr(lm, "visibility", 1.0) or 1.0
                keypoints[i] = [lm.x, lm.y, lm.z, vis]
                pose_visibilities.append(float(vis))

        if result.left_hand_landmarks and len(result.left_hand_landmarks) > 0:
            left_hand_detected = True
            for i, lm in enumerate(result.left_hand_landmarks[:ISIGN_HAND_COUNT]):
                keypoints[ISIGN_POSE_COUNT + i] = [lm.x, lm.y, lm.z, 1.0]

        if result.right_hand_landmarks and len(result.right_hand_landmarks) > 0:
            right_hand_detected = True
            for i, lm in enumerate(result.right_hand_landmarks[:ISIGN_HAND_COUNT]):
                keypoints[ISIGN_POSE_COUNT + ISIGN_HAND_COUNT + i] = [lm.x, lm.y, lm.z, 1.0]

        missing_landmarks = 0
        if not pose_detected:
            missing_landmarks += ISIGN_POSE_COUNT
        if not left_hand_detected:
            missing_landmarks += ISIGN_HAND_COUNT
        if not right_hand_detected:
            missing_landmarks += ISIGN_HAND_COUNT

        avg_pose_vis = float(np.mean(pose_visibilities)) if pose_visibilities else 0.0

        quality_info: Dict[str, Any] = {
            "pose": pose_detected,
            "left_hand": left_hand_detected,
            "right_hand": right_hand_detected,
            "pose_visibility_mean": avg_pose_vis,
            "missing_landmarks_count": missing_landmarks,
            "missing_percentage": round(missing_landmarks / ISIGN_TOTAL_LANDMARKS * 100.0, 2),
        }

        return keypoints, quality_info

    def extract_isign_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        hand_missing_strategy: str = "zero",
        max_forward_fill_frames: int = 15,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract 75-landmark sequence using legacy holistic landmarker."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for video landmark extraction.")

        path_obj = Path(video_path).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Video file not found at: {path_obj}")

        cap = cv2.VideoCapture(str(path_obj))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file: {path_obj}")

        container_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_spatial_300: List[np.ndarray] = []
        pose_detected_count = 0
        lh_detected_count = 0
        rh_detected_count = 0
        pose_visibilities: List[float] = []
        corrupted_frames = 0
        read_frames = 0
        t_start = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                corrupted_frames += 1
                continue

            read_frames += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kps, info = self.extract_isign_frame(rgb_frame)
            frames_spatial_300.append(kps.reshape(-1))

            if info["pose"]:
                pose_detected_count += 1
                pose_visibilities.append(info["pose_visibility_mean"])
            if info["left_hand"]:
                lh_detected_count += 1
            if info["right_hand"]:
                rh_detected_count += 1

            if max_frames is not None and read_frames >= max_frames:
                break

        cap.release()
        extraction_time = time.perf_counter() - t_start
        extraction_fps = round(read_frames / max(extraction_time, 1e-4), 2)

        if read_frames == 0:
            raise RuntimeError(f"No valid frames could be decoded from video: {path_obj}")

        raw_spatial_array = np.asarray(frames_spatial_300, dtype=np.float32)
        if hand_missing_strategy == "forward_fill":
            raw_spatial_array = apply_hand_forward_fill(
                raw_spatial_array, max_consecutive_frames=max_forward_fill_frames
            )

        pose_rate = round(pose_detected_count / read_frames, 4)
        lh_rate = round(lh_detected_count / read_frames, 4)
        rh_rate = round(rh_detected_count / read_frames, 4)
        avg_pose_vis = float(np.mean(pose_visibilities)) if pose_visibilities else 0.0

        total_slots = read_frames * ISIGN_TOTAL_LANDMARKS
        missing_slots = (
            (read_frames - pose_detected_count) * ISIGN_POSE_COUNT
            + (read_frames - lh_detected_count) * ISIGN_HAND_COUNT
            + (read_frames - rh_detected_count) * ISIGN_HAND_COUNT
        )
        overall_missing_pct = round(missing_slots / total_slots * 100.0, 2)
        missing_hand_pct = round(
            ((read_frames * 2) - (lh_detected_count + rh_detected_count))
            / (read_frames * 2)
            * 100.0,
            2,
        )

        occlusion_warning = (
            abs(lh_rate - rh_rate) > 0.25
            or min(lh_rate, rh_rate) < 0.60
            or avg_pose_vis < 0.60
        )

        metadata: Dict[str, Any] = {
            "video_path": str(path_obj),
            "container_frame_count": container_frame_count,
            "read_frames": read_frames,
            "corrupted_frames": corrupted_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "pose_detection_rate": pose_rate,
            "left_hand_detection_rate": lh_rate,
            "right_hand_detection_rate": rh_rate,
            "missing_hand_percentage": missing_hand_pct,
            "average_pose_visibility": round(avg_pose_vis, 4),
            "missing_landmark_percentage": overall_missing_pct,
            "occlusion_warning": occlusion_warning,
            "raw_spatial_shape": list(raw_spatial_array.shape),
            "engine": "legacy_holistic",
            "hand_missing_strategy": hand_missing_strategy,
            "max_forward_fill_frames": max_forward_fill_frames,
            "extraction_fps": extraction_fps,
            "extraction_time_seconds": round(extraction_time, 4),
        }

        return raw_spatial_array, metadata

    def close(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


# ---------------------------------------------------------------------------
# 3. Unified MediaPipeLandmarkExtractor (Primary Public Interface)
# ---------------------------------------------------------------------------
class MediaPipeLandmarkExtractor:
    """
    Primary landmark extractor for Signova AI.
    Defaults to modern MediaPipe Tasks API (PoseLandmarker + HandLandmarker in VIDEO mode).
    Maintains complete backward compatibility and automatic fallback to LegacyMediaPipeExtractor.
    """

    def __init__(
        self,
        engine: Literal["modern", "legacy"] = "modern",
        model_asset_path: Optional[Union[str, Path]] = None,
        pose_model_path: Optional[Union[str, Path]] = None,
        hand_model_path: Optional[Union[str, Path]] = None,
        min_pose_detection_confidence: float = 0.5,
        min_hand_landmarks_confidence: float = 0.5,
        min_face_detection_confidence: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.engine_name = engine
        self._modern_extractor: Optional[ModernMediaPipeExtractor] = None
        self._legacy_extractor: Optional[LegacyMediaPipeExtractor] = None

        if engine == "modern":
            try:
                self._modern_extractor = ModernMediaPipeExtractor(
                    pose_model_path=pose_model_path,
                    hand_model_path=hand_model_path,
                    min_pose_detection_confidence=min_pose_detection_confidence,
                    min_hand_landmarks_confidence=min_hand_landmarks_confidence,
                )
                logger.info("Initialized ModernMediaPipeExtractor (PoseLandmarker + HandLandmarker).")
            except Exception as exc:
                logger.warning(
                    "Failed to initialize ModernMediaPipeExtractor (%s); falling back to legacy holistic.",
                    exc,
                )
                self.engine_name = "legacy"
                self._legacy_extractor = LegacyMediaPipeExtractor(
                    model_asset_path=model_asset_path,
                    min_face_detection_confidence=min_face_detection_confidence,
                    min_pose_detection_confidence=min_pose_detection_confidence,
                    min_hand_landmarks_confidence=min_hand_landmarks_confidence,
                )
        else:
            self._legacy_extractor = LegacyMediaPipeExtractor(
                model_asset_path=model_asset_path,
                min_face_detection_confidence=min_face_detection_confidence,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_hand_landmarks_confidence=min_hand_landmarks_confidence,
            )

    def extract_isign_frame(
        self, rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract 75 landmarks from a single RGB frame for continuous iSign translation.
        """
        if self._modern_extractor is not None:
            return self._modern_extractor.extract_isign_frame(rgb_frame)
        if self._legacy_extractor is not None:
            return self._legacy_extractor.extract_isign_frame(rgb_frame)
        raise RuntimeError("MediaPipeLandmarkExtractor is closed.")

    def extract_isign_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        hand_missing_strategy: str = "zero",
        max_forward_fill_frames: int = 15,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract sequence of 75-landmark vectors from a video file.
        Uses VIDEO mode for modern engine.
        """
        if self._modern_extractor is not None:
            return self._modern_extractor.extract_isign_video(
                video_path,
                max_frames=max_frames,
                hand_missing_strategy=hand_missing_strategy,
                max_forward_fill_frames=max_forward_fill_frames,
            )
        if self._legacy_extractor is not None:
            return self._legacy_extractor.extract_isign_video(
                video_path,
                max_frames=max_frames,
                hand_missing_strategy=hand_missing_strategy,
                max_forward_fill_frames=max_forward_fill_frames,
            )
        raise RuntimeError("MediaPipeLandmarkExtractor is closed.")

    def extract_isign_video_to_pose(
        self,
        video_path: Union[str, Path],
        output_pose_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        hand_missing_strategy: str = "zero",
        max_forward_fill_frames: int = 15,
    ) -> Tuple[Path, Any, Dict[str, Any]]:
        """
        Extract video landmarks and serialize into an official iSign .pose file.
        Returns:
            (saved_pose_path, pose_object, video_metadata)
        """
        from ai.pose_extraction.pose_generator import video_to_pose_file
        return video_to_pose_file(
            video_path=video_path,
            output_path=output_pose_path,
            extractor=self,
            max_frames=max_frames,
            hand_missing_strategy=hand_missing_strategy,
            max_forward_fill_frames=max_forward_fill_frames,
        )

    def extract_frame(
        self, rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Extract 94 landmarks for isolated sign recognition (face, pose, hands).
        Delegates to legacy holistic landmarker.
        """
        if self._legacy_extractor is None:
            # Lazily initialize legacy holistic extractor for isolated 94-landmark mode
            self._legacy_extractor = LegacyMediaPipeExtractor()
        return self._legacy_extractor.extract_frame(rgb_frame)

    def close(self) -> None:
        if self._modern_extractor is not None:
            self._modern_extractor.close()
            self._modern_extractor = None
        if self._legacy_extractor is not None:
            self._legacy_extractor.close()
            self._legacy_extractor = None
