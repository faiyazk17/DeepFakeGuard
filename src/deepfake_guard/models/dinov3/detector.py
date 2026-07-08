from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .classifier_head import LinearProbe
from .frame_encoder import FrameEncoder
from ...utils.preprocess import build_preprocess


@dataclass
class PredictDetails:
    per_frame_fake_probs: List[float]
    instability: float
    avg_confidence: float
    frame_count: int


class Detector(nn.Module):
    """Encoder + classifier head wrapper.

    - forward(images): images [B,3,H,W] -> (logits [B,2], embeddings [B,D])
    - predict_video(frames): frames [T,3,H,W] -> dict(score, label, details)
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

        self.encoder = FrameEncoder(
            device=device,
            layernorm_tuning=True,
            unfreeze_last_block=True,
        )
        self.head = LinearProbe(
            input_dim=self.encoder.embed_dim,
            num_classes=2,
            normalize_inputs=True,
            detach_classifier_inputs=False
        )

        # Exposed for smoke_test / apps
        self.preprocess = build_preprocess(image_size=self.encoder.image_size)

        self.to(device)

    def forward(self, images: torch.Tensor):
        x = self.encoder(images)
        head_output = self.head(x)
        return head_output.logits_labels, head_output.l2_embeddings

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor) -> Dict[str, Any]:
        """Predict deepfake probability for a stack of frames.

        Args:
            frames: [T,3,H,W] tensor.

        Returns:
            dict with:
              - score: mean P(fake)
              - label: "FAKE" if score > 0.5 else "REAL"
              - details: per-frame probs + simple uncertainty stats
        """
        self.eval()

        if frames.dim() != 4:
            raise ValueError(f"Expected frames [T,3,H,W], got shape {tuple(frames.shape)}")

        frames = frames.to(self.device)
        logits, _ = self.forward(frames)
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(fake)

        per_frame = probs.detach().cpu().tolist()
        score = float(probs.mean().item())
        instability = float(probs.std(unbiased=False).item())
        confidence = float(abs(score - 0.5) * 2.0)

        details = PredictDetails(
            per_frame_fake_probs=[float(x) for x in per_frame],
            instability=instability,
            avg_confidence=confidence,
            frame_count=int(probs.numel()),
        )

        return {
            "score": score,
            "label": "FAKE" if score > 0.5 else "REAL",
            "details": details.__dict__,
        }
