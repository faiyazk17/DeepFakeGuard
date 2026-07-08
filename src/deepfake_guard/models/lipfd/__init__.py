"""
LipFD detector module for DeepFakeGuard.

Implements the LipFD model (NeurIPS 2024) — detects lip-sync deepfakes by
exploiting temporal inconsistency between audio and visual lip movements.
Pretrained on the AVLips dataset.

Original Paper:
    Liu et al., "Lips Are Lying: Spotting the Temporal Inconsistency
    between Audio and Visual in Lip-Syncing DeepFakes", NeurIPS 2024.
    https://arxiv.org/abs/2401.15668

Original Repository:
    https://github.com/AaronPeng920/LipFD

Quick start::

    from deepfake_guard.models.lipfd import LipFDDetector

    # With pretrained weights (recommended)
    det = LipFDDetector(weights_path="weights/lipfd_ckpt.pth")
    result = det.predict_video("suspicious_video.mp4")

    # Without weights (pipeline test only)
    det = LipFDDetector()
    result = det.predict_video("test.mp4")
"""

from .detector import LipFDDetector
from .model import LipFD, RALoss, build_model, get_loss

__all__ = [
    "LipFDDetector",
    "LipFD",
    "RALoss",
    "build_model",
    "get_loss",
]
