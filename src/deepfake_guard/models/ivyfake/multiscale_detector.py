"""
Multi-Scale Temporal IvyFake Detector

Analyzes videos at multiple temporal scales (0.5, 1.0, 2.0 fps) for superior
artifact detection.  This is the novel contribution from:
  https://github.com/HamzaKhan760/IvyFakeGenDetector

Architecture:
  Input Video
      │
      ├── Slow Branch  (0.5 fps)  ── Conv1d → Pool
      ├── Medium Branch (1.0 fps) ── Conv1d → Pool
      ├── Fast Branch  (2.0 fps)  ── Conv1d → Pool
      │
      └── Cross-Scale Fusion → Spatial Analyzer → Classifier
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from ...types import ModalityResult


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Pyramid
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPyramidExtractor(nn.Module):
    """Extracts features at multiple temporal scales."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.slow_branch = self._make_temporal_branch(embed_dim)
        self.medium_branch = self._make_temporal_branch(embed_dim)
        self.fast_branch = self._make_temporal_branch(embed_dim)

        self.scale_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _make_temporal_branch(embed_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(
        self,
        slow_features: torch.Tensor,
        medium_features: torch.Tensor,
        fast_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slow_features:   (batch, num_frames_slow,   embed_dim)
            medium_features: (batch, num_frames_medium, embed_dim)
            fast_features:   (batch, num_frames_fast,   embed_dim)
        Returns:
            fused_features: (batch, embed_dim)
        """
        # (B, D, T) → Conv1d → AdaptiveAvgPool1d → (B, D, 1) → (B, D)
        slow = self.slow_branch(slow_features.transpose(1, 2)).squeeze(-1)
        medium = self.medium_branch(medium_features.transpose(1, 2)).squeeze(-1)
        fast = self.fast_branch(fast_features.transpose(1, 2)).squeeze(-1)

        multi_scale = torch.cat([slow, medium, fast], dim=-1)
        return self.scale_fusion(multi_scale)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Scale Model
# ──────────────────────────────────────────────────────────────────────────────

class MultiScaleIvyDetector(nn.Module):
    """Enhanced IvyFake detector with multi-scale temporal analysis."""

    def __init__(
        self,
        backbone: str = "openai/clip-vit-base-patch32",
        num_classes: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Shared frozen CLIP vision encoder
        clip_model = CLIPModel.from_pretrained(backbone)
        self.vision_encoder = clip_model.vision_model
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Multi-scale temporal pyramid
        self.temporal_pyramid = TemporalPyramidExtractor(embed_dim)

        # Spatial analyzer (for single-frame artifacts)
        self.spatial_analyzer = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP features from images."""
        vision_outputs = self.vision_encoder(pixel_values=images)
        return vision_outputs.pooler_output

    def forward(
        self,
        slow_frames: torch.Tensor,
        medium_frames: torch.Tensor,
        fast_frames: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            slow_frames:   (batch, num_frames_slow,   C, H, W) — 0.5 fps
            medium_frames: (batch, num_frames_medium, C, H, W) — 1.0 fps
            fast_frames:   (batch, num_frames_fast,   C, H, W) — 2.0 fps
        Returns:
            dict with 'logits', 'temporal_features', 'spatial_features'
        """
        batch_size = slow_frames.shape[0]

        def extract_scale_features(frames: torch.Tensor) -> torch.Tensor:
            num_frames = frames.shape[1]
            frames_flat = frames.view(-1, *frames.shape[2:])
            features = self.extract_features(frames_flat)
            return features.view(batch_size, num_frames, -1)

        slow_features = extract_scale_features(slow_frames)
        medium_features = extract_scale_features(medium_frames)
        fast_features = extract_scale_features(fast_frames)

        # Multi-scale temporal analysis
        temporal_features = self.temporal_pyramid(
            slow_features, medium_features, fast_features
        )

        # Spatial analysis (use medium frames as reference)
        spatial_features = self.spatial_analyzer(medium_features.mean(dim=1))

        # Fuse temporal and spatial
        combined = torch.cat([temporal_features, spatial_features], dim=-1)
        fused = self.fusion(combined)

        logits = self.classifier(fused)

        return {
            "logits": logits,
            "temporal_features": temporal_features,
            "spatial_features": spatial_features,
        }

    def predict(
        self,
        slow_frames: torch.Tensor,
        medium_frames: torch.Tensor,
        fast_frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (predictions [B], probabilities [B, 2])."""
        self.eval()
        with torch.no_grad():
            output = self.forward(slow_frames, medium_frames, fast_frames)
            logits = output["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs


def load_multiscale_detector(
    weights_path: Optional[str] = None,
    device: str = "cpu",
) -> MultiScaleIvyDetector:
    """Load pretrained multi-scale detector."""
    model = MultiScaleIvyDetector()
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# High-level wrapper for DeepfakeGuard integration
# ──────────────────────────────────────────────────────────────────────────────

# CLIP normalisation constants
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class MultiScaleIvyFakeDetector:
    """
    Multi-scale IvyFake detector wrapper for DeepfakeGuard.

    Extracts frames at three temporal scales (slow / medium / fast fps)
    and processes them through the TemporalPyramid.

    Modes:
      • **Trained** (weights_path): full model → softmax → P(fake)
      • **Zero-shot** (no weights): frozen CLIP cosine-sim at each scale,
        weighted fusion (fast scale weighted highest for micro-artifacts)
    """

    def __init__(
        self,
        device: str = "cpu",
        slow_fps: float = 0.5,
        medium_fps: float = 1.0,
        fast_fps: float = 2.0,
        max_duration: float = 10.0,
        threshold: float = 0.5,
        weights_path: Optional[str] = None,
    ):
        self.device = device
        self.slow_fps = slow_fps
        self.medium_fps = medium_fps
        self.fast_fps = fast_fps
        self.max_duration = max_duration
        self.threshold = threshold
        self.weights_loaded = False

        self.model = MultiScaleIvyDetector()
        if weights_path:
            state_dict = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.weights_loaded = True

        self.model.to(device)
        self.model.eval()

    # ── Frame extraction at specific fps ─────────────────────────────────

    def _extract_frames_at_fps(
        self, video_path: str, fps: float
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        actual_duration = min(duration, self.max_duration)
        num_frames = max(int(actual_duration * fps), 1)
        frame_interval = video_fps / fps

        frames: List[np.ndarray] = []
        for i in range(num_frames):
            frame_idx = int(i * frame_interval)
            if frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    # ── Preprocessing ────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> torch.Tensor:
        frame = cv2.resize(frame, (224, 224))
        t = torch.from_numpy(frame).float() / 255.0
        t = t.permute(2, 0, 1)
        mean = torch.tensor(_CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(_CLIP_STD).view(3, 1, 1)
        return (t - mean) / std

    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert frame list → (1, T, 3, 224, 224) tensor on device."""
        if not frames:
            return torch.zeros(1, 1, 3, 224, 224, device=self.device)
        tensors = torch.stack([self._preprocess_frame(f) for f in frames])
        return tensors.unsqueeze(0).to(self.device)

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_video(self, video_path: str) -> Dict[str, Any]:
        slow_frames = self._extract_frames_at_fps(video_path, self.slow_fps)
        medium_frames = self._extract_frames_at_fps(video_path, self.medium_fps)
        fast_frames = self._extract_frames_at_fps(video_path, self.fast_fps)

        if not medium_frames:
            return {
                "score": 0.0,
                "label": "UNKNOWN",
                "details": {"error": "Could not extract frames"},
            }

        slow_t = self._frames_to_tensor(slow_frames)
        medium_t = self._frames_to_tensor(medium_frames)
        fast_t = self._frames_to_tensor(fast_frames)

        if self.weights_loaded:
            preds, probs = self.model.predict(slow_t, medium_t, fast_t)
            fake_prob = float(probs[0, 1].item())
            mode = "trained"
        else:
            # Zero-shot: frozen CLIP cosine-similarity at each scale
            def _get_feats(tensor: torch.Tensor) -> torch.Tensor:
                b, t_ = tensor.shape[:2]
                flat = tensor.view(-1, *tensor.shape[2:])
                raw = self.model.extract_features(flat)
                return F.normalize(raw, dim=-1).view(b, t_, -1)

            def _temporal_consistency(feats: torch.Tensor) -> torch.Tensor:
                if feats.shape[1] < 2:
                    return torch.ones(1, device=feats.device)
                cos = (feats[:, :-1] * feats[:, 1:]).sum(-1)
                return cos.mean(-1)

            slow_sim = _temporal_consistency(_get_feats(slow_t))
            medium_sim = _temporal_consistency(_get_feats(medium_t))
            fast_sim = _temporal_consistency(_get_feats(fast_t))

            # Weighted: fast scale most informative for micro-artifacts
            combined_sim = 0.2 * slow_sim + 0.3 * medium_sim + 0.5 * fast_sim
            fake_prob = float(
                torch.sigmoid(10.0 * ((1.0 - combined_sim) - 0.25)).item()
            )
            mode = "zero-shot"

        return {
            "score": fake_prob,
            "label": "FAKE" if fake_prob > self.threshold else "REAL",
            "details": {
                "frame_counts": {
                    "slow": len(slow_frames),
                    "medium": len(medium_frames),
                    "fast": len(fast_frames),
                },
                "fps_settings": {
                    "slow": self.slow_fps,
                    "medium": self.medium_fps,
                    "fast": self.fast_fps,
                },
                "detector_type": "ivyfake-multiscale",
                "mode": mode,
                "backbone": "CLIP-ViT-B/32",
            },
        }
