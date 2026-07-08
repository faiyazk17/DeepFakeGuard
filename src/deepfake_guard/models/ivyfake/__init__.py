"""
IvyFake-based detector (CLIP-based explainable AIGC detection)
Uses CLIP vision encoder with temporal and spatial artifact analyzers.

Architecture from: https://github.com/HamzaKhan760/IvyFakeGenDetector
"""

from .detector import (
    IvyXDetector,
    SimplifiedIvyDetector,
    TemporalArtifactAnalyzer,
    SpatialArtifactAnalyzer,
    IvyFakeDetector,
    load_pretrained_ivydetector,
    create_ivyfake_detector,
    detect_video_ivyfake,
)
from .multiscale_detector import (
    MultiScaleIvyDetector,
    MultiScaleIvyFakeDetector,
    TemporalPyramidExtractor,
    load_multiscale_detector,
)

__all__ = [
    # Standard
    "IvyXDetector",
    "SimplifiedIvyDetector",
    "TemporalArtifactAnalyzer",
    "SpatialArtifactAnalyzer",
    "IvyFakeDetector",
    "load_pretrained_ivydetector",
    "create_ivyfake_detector",
    "detect_video_ivyfake",
    # Multi-scale
    "MultiScaleIvyDetector",
    "MultiScaleIvyFakeDetector",
    "TemporalPyramidExtractor",
    "load_multiscale_detector",
]