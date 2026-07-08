"""
IvyFake-based video detector (CLIP-based explainable AIGC detection)

Faithfully implements the architecture from:
  https://github.com/HamzaKhan760/IvyFakeGenDetector

Two model variants:
  - IvyXDetector (full): temporal + spatial analyzers → fusion → classifier
  - SimplifiedIvyDetector (lightweight): frozen CLIP → MLP classifier

When trained weights are loaded the full model heads produce calibrated
softmax scores.  Without weights the detector falls back to a zero-shot
heuristic based on frozen CLIP cosine-similarity and spatial variance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from ...types import ModalityResult


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules (match HamzaKhan760/IvyFakeGenDetector/models/detector.py)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalArtifactAnalyzer(nn.Module):
    """Analyzes temporal inconsistencies in video frames."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (batch, num_frames, embed_dim)
        Returns:
            temporal_features: (batch, embed_dim)
        """
        x = frame_features.transpose(1, 2)          # (B, D, T) for Conv1d
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)                       # (B, T, D)

        x = x.transpose(0, 1)                       # (T, B, D) for MHA
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.transpose(0, 1)          # (B, T, D)

        return attn_out.mean(dim=1)                  # global average pool


class SpatialArtifactAnalyzer(nn.Module):
    """Analyzes spatial artifacts in individual frames."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (batch, num_frames, embed_dim)
        Returns:
            spatial_features: (batch, embed_dim)
        """
        attention_weights = self.spatial_attention(frame_features)
        attended = frame_features * attention_weights
        return attended.mean(dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Full model (IvyXDetector)
# ──────────────────────────────────────────────────────────────────────────────

class IvyXDetector(nn.Module):
    """
    Unified explainable AIGC detector for images and videos.
    Architecture from https://github.com/HamzaKhan760/IvyFakeGenDetector:

      1. Vision Backbone: CLIP ViT-B/32
      2. Temporal Analyzer: Conv1d + MultiheadAttention over frame embeddings
      3. Spatial Analyzer: Learned attention gate per frame
      4. Fusion Layer: MLP combining temporal and spatial features
      5. Classification Head: Linear → 2-class logits
      6. Explanation Head: Linear → 768-d features (for text generation)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 2,
        embed_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # CLIP vision backbone
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Temporal and spatial artifact analyzers
        self.temporal_analyzer = TemporalArtifactAnalyzer(embed_dim)
        self.spatial_analyzer = SpatialArtifactAnalyzer(embed_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification head
        self.classifier = nn.Linear(embed_dim // 2, num_classes)

        # Explanation generator (simplified — pooled features → text dim)
        self.explanation_head = nn.Linear(embed_dim // 2, 768)

    def extract_frame_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from frames using CLIP vision encoder.

        Returns projected 512-dim embeddings (not raw 768-dim hidden states).
        """
        vision_outputs = self.clip_model.vision_model(pixel_values=images)
        pooled = vision_outputs.pooler_output          # (N, 768)
        projected = self.clip_model.visual_projection(pooled)  # (N, 512)
        return projected

    def forward(
        self,
        images: torch.Tensor,
        return_explanations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (batch, num_frames, C, H, W) or (batch, C, H, W)
            return_explanations: whether to produce explanation features

        Returns:
            dict with 'logits', 'temporal_features', 'spatial_features',
            and optionally 'explanation_features'.
        """
        # Handle both image and video inputs
        if images.dim() == 5:
            batch_size, num_frames = images.shape[:2]
            images_flat = images.view(-1, *images.shape[2:])
        else:
            batch_size = images.shape[0]
            num_frames = 1
            images_flat = images

        # Extract features
        frame_features = self.extract_frame_features(images_flat)
        frame_features = frame_features.view(batch_size, num_frames, -1)

        # Analyze temporal and spatial artifacts
        if num_frames > 1:
            temporal_features = self.temporal_analyzer(frame_features)
        else:
            temporal_features = frame_features.squeeze(1)

        spatial_features = self.spatial_analyzer(frame_features)

        # Fuse features
        combined = torch.cat([temporal_features, spatial_features], dim=-1)
        fused = self.fusion(combined)

        # Classification
        logits = self.classifier(fused)

        output: Dict[str, torch.Tensor] = {
            "logits": logits,
            "temporal_features": temporal_features,
            "spatial_features": spatial_features,
        }

        if return_explanations:
            output["explanation_features"] = self.explanation_head(fused)

        return output

    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (predictions [B], probabilities [B, 2])."""
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
            logits = output["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight model (SimplifiedIvyDetector)
# ──────────────────────────────────────────────────────────────────────────────

class SimplifiedIvyDetector(nn.Module):
    """
    Lightweight version: frozen CLIP vision encoder + simple MLP classifier.
    Suitable for deployment and real-time applications.
    """

    def __init__(
        self,
        backbone: str = "openai/clip-vit-base-patch32",
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        clip_model = CLIPModel.from_pretrained(backbone)
        self.vision_encoder = clip_model.vision_model

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        embed_dim = self.vision_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, C, H, W)
        Returns:
            logits: (batch, num_classes)
        """
        vision_outputs = self.vision_encoder(pixel_values=images)
        features = vision_outputs.pooler_output
        return self.classifier(features)

    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs


# ──────────────────────────────────────────────────────────────────────────────
# Utility: weight loading
# ──────────────────────────────────────────────────────────────────────────────

def load_pretrained_ivydetector(
    weights_path: Optional[str] = None,
    device: str = "cpu",
) -> IvyXDetector:
    """Load pretrained IvyXDetector model."""
    model = IvyXDetector()
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


class IvyFakeDetector:
    """
    IvyFake-based video detector for DeepfakeGuard.

    Operation modes:
      • **Trained** (weights_path provided): the full IvyXDetector architecture
        runs through temporal + spatial analyzers → fusion → classifier →
        softmax.  This is the proper IvyFake pipeline.
      • **Zero-shot** (no weights): uses CLIP text-image zero-shot
        classification with prompts for "real" vs "AI-generated" content.
        This is how CLIP is designed for zero-shot tasks.
    """

    # Text prompts for zero-shot CLIP classification (class 0 = real, class 1 = fake)
    _ZS_REAL_PROMPTS: List[str] = [
        "a real photograph",
        "an authentic unedited photo",
        "a genuine camera photograph of a real scene",
        "a natural unmanipulated photograph",
    ]
    _ZS_FAKE_PROMPTS: List[str] = [
        "an AI generated image",
        "a deepfake manipulated photo",
        "a synthetic computer generated image",
        "an artificially created fake image",
    ]

    def __init__(
        self,
        device: str = "cpu",
        num_frames: int = 16,
        threshold: float = 0.5,
        model_name: str = "openai/clip-vit-base-patch32",
        weights_path: Optional[str] = None,
    ):
        self.device = device
        self.num_frames = num_frames
        self.threshold = threshold
        self.weights_loaded = False

        # Build model — freeze backbone when no weights (zero-shot mode)
        self.model = IvyXDetector(
            model_name=model_name,
            freeze_backbone=(weights_path is None),
        )

        if weights_path:
            state_dict = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.weights_loaded = True
        else:
            pass  # Zero-shot mode via CLIP text-image similarity

        self.model.to(device)
        self.model.eval()

        # Pre-compute text embeddings for zero-shot (only needed without weights)
        if not self.weights_loaded:
            self._precompute_text_embeddings()

    # ── Frame extraction ─────────────────────────────────────────────────

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Uniformly sample frames from video."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return []

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single RGB frame for CLIP input."""
        frame_resized = cv2.resize(frame, (224, 224))
        t = torch.from_numpy(frame_resized).float() / 255.0
        t = t.permute(2, 0, 1)  # HWC → CHW
        mean = torch.tensor(_CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(_CLIP_STD).view(3, 1, 1)
        return (t - mean) / std

    # ── Trained-model inference ──────────────────────────────────────────

    def _predict_trained(self, video_tensor: torch.Tensor) -> Dict[str, Any]:
        """Full model: temporal + spatial → fusion → classifier → softmax."""
        preds, probs = self.model.predict(video_tensor)
        fake_prob = float(probs[0, 1].item())

        # Also compute per-frame scores by running frame-by-frame
        num_frames = video_tensor.shape[1] if video_tensor.dim() == 5 else 1
        per_frame_probs: List[float] = []
        if video_tensor.dim() == 5 and num_frames > 1:
            for i in range(num_frames):
                frame_input = video_tensor[:, i:i + 1, :, :, :]
                _, fp = self.model.predict(frame_input)
                per_frame_probs.append(float(fp[0, 1].item()))

        return {
            "score": fake_prob,
            "label": "FAKE" if fake_prob > self.threshold else "REAL",
            "details": {
                "per_frame_fake_probs": per_frame_probs if per_frame_probs else [fake_prob],
                "frame_count": num_frames,
                "detector_type": "ivyfake",
                "mode": "trained",
                "backbone": "CLIP-ViT-B/32",
                "features": ["temporal_analyzer", "spatial_analyzer", "fusion", "classifier"],
            },
        }

    # ── Zero-shot text-image classification ────────────────────────────

    def _precompute_text_embeddings(self) -> None:
        """Pre-compute and cache CLIP text embeddings for real/fake prompts."""
        tokenizer = self.model.processor.tokenizer
        clip = self.model.clip_model

        def _encode_prompts(prompts: List[str]) -> torch.Tensor:
            tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
            text_out = clip.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
            )
            # pooler_output → text_projection → normalize
            pooled = text_out.pooler_output                      # (N, hidden)
            if hasattr(clip, "text_projection") and clip.text_projection is not None:
                pooled = clip.text_projection(pooled)            # (N, embed)
            return F.normalize(pooled, dim=-1)

        with torch.no_grad():
            real_embeds = _encode_prompts(self._ZS_REAL_PROMPTS)
            self._real_text_embed = F.normalize(real_embeds.mean(dim=0, keepdim=True), dim=-1)

            fake_embeds = _encode_prompts(self._ZS_FAKE_PROMPTS)
            self._fake_text_embed = F.normalize(fake_embeds.mean(dim=0, keepdim=True), dim=-1)

        # Stack: (2, embed_dim) — row 0 = real, row 1 = fake
        self._text_embeds = torch.cat(
            [self._real_text_embed, self._fake_text_embed], dim=0
        )

    def _predict_zeroshot(self, video_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Zero-shot CLIP text-image classification.

        Compares each frame's CLIP vision embedding against text prompts
        for "real photograph" vs "AI-generated deepfake", then averages
        across frames.  This is how CLIP zero-shot classification works
        and matches the spirit of the IvyFake CLIP-based approach.
        """
        if video_tensor.dim() == 5:
            batch_size, num_frames = video_tensor.shape[:2]
            images_flat = video_tensor.view(-1, *video_tensor.shape[2:])
        else:
            batch_size = video_tensor.shape[0]
            num_frames = 1
            images_flat = video_tensor

        # Get CLIP vision embeddings for all frames
        raw_feats = self.model.extract_frame_features(images_flat)
        feats = F.normalize(raw_feats, dim=-1)  # (N, embed_dim)

        # Cosine similarity against text embeddings (N, 2)
        # Column 0 = similarity to "real", column 1 = similarity to "fake"
        logit_scale = self.model.clip_model.logit_scale.exp()
        sim = logit_scale * (feats @ self._text_embeds.T)  # (N, 2)
        probs = torch.softmax(sim, dim=-1)  # (N, 2)

        per_frame_fake_probs = probs[:, 1].tolist()  # P(fake) per frame

        # Average across all frames
        mean_probs = probs.view(batch_size, num_frames, 2).mean(dim=1)  # (B, 2)
        fake_prob = float(mean_probs[0, 1].item())

        real_sim_avg = float(sim[:, 0].mean().item())
        fake_sim_avg = float(sim[:, 1].mean().item())

        return {
            "score": fake_prob,
            "label": "FAKE" if fake_prob > self.threshold else "REAL",
            "details": {
                "per_frame_fake_probs": per_frame_fake_probs,
                "frame_count": num_frames,
                "detector_type": "ivyfake",
                "mode": "zero-shot (CLIP text-image)",
                "backbone": "CLIP-ViT-B/32 (frozen)",
                "features": ["clip_text_image_similarity"],
                "real_similarity": round(real_sim_avg, 4),
                "fake_similarity": round(fake_sim_avg, 4),
                "note": "CLIP zero-shot text-image classification — load trained weights for full IvyFake accuracy",
            },
        }

    # ── Public API ───────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Detect AI-generated content in video.

        Returns:
            dict with score, label, and details.
        """
        frames = self.extract_frames(video_path)
        if not frames:
            return {
                "score": 0.0,
                "label": "UNKNOWN",
                "details": {"error": "Could not extract frames"},
            }

        inputs = torch.stack([self.preprocess_frame(f) for f in frames]).to(self.device)
        inputs = inputs.unsqueeze(0)  # (T,3,H,W) → (1,T,3,H,W)

        if self.weights_loaded:
            return self._predict_trained(inputs)
        else:
            return self._predict_zeroshot(inputs)


# ──────────────────────────────────────────────────────────────────────────────
# Factory / standalone helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_ivyfake_detector(
    device: str = "cpu",
    weights_path: Optional[str] = None,
    **kwargs,
) -> IvyFakeDetector:
    """Factory function for IvyFake detector."""
    return IvyFakeDetector(device=device, weights_path=weights_path, **kwargs)


def detect_video_ivyfake(
    video_path: str,
    device: str = "cpu",
    weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Standalone detection function for DeepfakeGuard modality registration.

    Args:
        video_path: Path to video file
        device: "cpu" or "cuda"
        weights_path: Optional path to trained IvyFake weights

    Returns:
        dict compatible with DeepfakeGuard format
    """
    detector = IvyFakeDetector(device=device, weights_path=weights_path)
    result = detector.predict_video(video_path)
    return ModalityResult(
        score=result["score"],
        label=result["label"],
        details=result["details"],
    ).__dict__