"""Pose and landmark extraction package."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
    from ai.pose_extraction.pose_generator import (
        build_isign_pose_header,
        landmarks_to_pose,
        save_pose_file,
        validate_pose_file,
        video_to_pose_file,
    )


def __getattr__(name: str):
    if name == "MediaPipeLandmarkExtractor":
        from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
        return MediaPipeLandmarkExtractor
    if name in {
        "build_isign_pose_header",
        "landmarks_to_pose",
        "save_pose_file",
        "validate_pose_file",
        "video_to_pose_file",
    }:
        import ai.pose_extraction.pose_generator as pg
        return getattr(pg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MediaPipeLandmarkExtractor",
    "build_isign_pose_header",
    "landmarks_to_pose",
    "save_pose_file",
    "validate_pose_file",
    "video_to_pose_file",
]
