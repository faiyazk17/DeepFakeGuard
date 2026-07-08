"""
D3 Detector — Detection by Difference of Differences
Training-free AI-generated video detection using second-order temporal features.

Faithfully reimplements the core algorithm from:
  "D3: Training-Free AI-Generated Video Detection Using Second-Order Features"
  Zheng et al., ICCV 2025  —  https://arxiv.org/abs/2508.00701
  Reference repo: https://github.com/eitanzur/D3
"""

from __future__ import annotations

import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List

# Encoder names whose HuggingFace output is a dataclass (not a plain tensor).
_TRANSFORMER_ENCODERS = {
    "clip-16", "clip-32", "xclip-16", "xclip-32", "dino-base", "dino-large",
}


class D3Model(nn.Module):
    """
    Core D3 model: encoder -> 1st-order distances -> 2nd-order diffs -> volatility.
    Mirrors ``models/D3_model.py`` from the reference implementation.
    """

    def __init__(
        self,
        encoder_type: str = "xclip-16",
        loss_type: str = "l2",
        device: str = "cpu",
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.loss_type = loss_type
        self.device = device
        self.encoder = self._build_encoder(self.encoder_type)
        self.encoder.eval()
        self.to(device)

    # ------------------------------------------------------------------
    # Encoder factory (matches original D3_model.py encoder options)
    # ------------------------------------------------------------------
    def _build_encoder(self, name: str) -> nn.Module:
        if name == "clip-16":
            from transformers import CLIPVisionModel
            return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

        if name == "clip-32":
            from transformers import CLIPVisionModel
            return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        if name == "xclip-16":
            try:
                from transformers import XCLIPVisionModel
                return XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")
            except Exception:
                from transformers import CLIPVisionModel
                return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

        if name == "xclip-32":
            try:
                from transformers import XCLIPVisionModel
                return XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")
            except Exception:
                from transformers import CLIPVisionModel
                return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        if name == "dino-base":
            from transformers import AutoModel
            return AutoModel.from_pretrained("facebook/dinov2-base")

        if name == "dino-large":
            from transformers import AutoModel
            return AutoModel.from_pretrained("facebook/dinov2-large")

        if name == "resnet-18":
            import torchvision.models as models
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            modules = list(resnet.children())[:-1]  # strip FC layer
            return nn.Sequential(*modules)

        if name == "mobilenet-v3":
            import timm
            mob = timm.create_model("mobilenetv3_large_100", pretrained=True)
            modules = list(mob.children())[:-1]
            return nn.Sequential(*modules)

        raise ValueError(
            f"Unknown encoder: {name}. "
            "Choose from: clip-16, clip-32, xclip-16, xclip-32, "
            "dino-base, dino-large, resnet-18, mobilenet-v3"
        )

    # ------------------------------------------------------------------
    # Forward pass — mirrors original D3_model.forward exactly
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: ``(batch, num_frames, 3, H, W)``
        Returns:
            ``(features, dis_2nd_avg, dis_2nd_std)``
        """
        b, t, c, h, w = x.shape
        images = x.reshape(-1, c, h, w)  # (B*T, 3, H, W)

        if self.encoder_type in _TRANSFORMER_ENCODERS:
            out = self.encoder(images, output_hidden_states=True)
            features = out.pooler_output  # (B*T, D)
        else:
            features = self.encoder(images)  # (B*T, D, ...)
            features = features.reshape(features.size(0), -1)  # flatten spatial

        features = features.reshape(b, t, -1)  # (B, T, D)

        # 1st-order: scalar distance between consecutive frame features
        vec1 = features[:, :-1, :]  # (B, T-1, D)
        vec2 = features[:, 1:, :]   # (B, T-1, D)

        if self.loss_type == "cos":
            dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)  # (B, T-1)
        else:  # l2
            dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)     # (B, T-1)

        # 2nd-order: difference of consecutive 1st-order distances
        dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]  # (B, T-2)

        dis_2nd_avg = torch.mean(dis_2nd, dim=1)  # (B,)
        dis_2nd_std = torch.std(dis_2nd, dim=1)    # (B,)  — **this is the volatility**

        return features, dis_2nd_avg, dis_2nd_std


class D3Detector:
    """
    High-level wrapper: extract frames -> D3Model -> result dict.
    Compatible with the DeepfakeGuard pipeline.
    """

    # Default threshold — calibrated for xCLIP-16 / L2 with 32 sampled frames.
    # The reference D3 demo used 2.5 but with longer clips; 1.8 is more stable
    # for typical conference-demo videos (5-30 s, 24-30 fps).
    DEFAULT_THRESHOLD = 1.8

    def __init__(
        self,
        encoder_name: str = "xclip-16",
        loss_type: str = "l2",
        threshold: float | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.encoder_name = encoder_name.lower()
        self.loss_type = loss_type
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        self.model = D3Model(
            encoder_type=self.encoder_name,
            loss_type=loss_type,
            device=device,
        )

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_frames(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    # ------------------------------------------------------------------
    # Pre-processing (ImageNet normalisation at 224x224)
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> torch.Tensor:
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # (3, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_video(self, video_path: str, sample_frames: int = 32) -> Dict[str, Any]:
        frames = self._extract_frames(video_path, sample_frames)
        if len(frames) < 3:
            return {
                "score": 0.5,
                "label": "UNKNOWN",
                "details": {"error": "Insufficient frames"},
            }

        tensors = torch.stack([self._preprocess_frame(f) for f in frames])  # (T,3,224,224)
        batch = tensors.unsqueeze(0).to(self.device)  # (1,T,3,224,224)

        _, dis_2nd_avg_tensor, volatility_tensor = self.model(batch)
        volatility = float(volatility_tensor.item())
        dis_2nd_avg = float(dis_2nd_avg_tensor.item())

        # Higher volatility -> real; lower -> AI-generated.
        # Score is a sigmoid centred at the threshold so it:
        #   - equals exactly 0.5 when volatility == threshold
        #   - approaches 1.0 for very low volatility (AI-generated)
        #   - approaches 0.0 for very high volatility (real)
        # k controls sharpness; k=1.5 gives a smooth, well-calibrated transition.
        is_real = volatility > self.threshold
        k = 1.5
        try:
            fake_score = 1.0 / (1.0 + math.exp(k * (volatility - self.threshold)))
        except OverflowError:
            fake_score = 0.0

        return {
            "score": fake_score,
            "label": "REAL" if is_real else "FAKE",
            "details": {
                "volatility": volatility,
                "threshold": self.threshold,
                "encoder": self.encoder_name,
                "loss_type": self.loss_type,
                "frame_count": len(frames),
                "detector_type": "d3",
            },
        }


def create_d3_detector(encoder: str = "xclip-16", device: str = "cpu") -> D3Detector:
    """Factory helper."""
    return D3Detector(encoder_name=encoder, device=device)
