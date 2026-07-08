from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class EncoderInfo:
    name: str
    embed_dim: int
    image_size: int

def _try_build_dinov3() -> tuple[nn.Module, EncoderInfo] | None:
    """
    Attempts to build a DINOv3 ViT-B/16 encoder via timm.
    Falls back through DINOv2 variants if the primary model is unavailable.
    Returns None if timm is not installed or no candidate loads successfully.
    """
    try:
        import timm  # type: ignore
    except Exception:
        return None

    candidates = [
        "vit_base_patch16_dinov3.lvd1689m",
        "vit_base_patch16_dinov3",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2",
        "vit_large_patch16_dinov3.lvd1689m",
        "vit_large_patch16_dinov3",
        "vit_large_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2",
    ]

    for name in candidates:
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0)
            embed_dim = getattr(model, "num_features", None) or getattr(model, "embed_dim", 1024)
            info = EncoderInfo(name=f"timm:{name}", embed_dim=int(embed_dim), image_size=224)
            return model, info
        except Exception:
            continue

    return None

def _build_resnet50_fallback() -> tuple[nn.Module, EncoderInfo]:
    from torchvision.models import resnet50, ResNet50_Weights  # type: ignore

    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Remove classifier: output will be pooled features (2048)
    backbone.fc = nn.Identity()
    return backbone, EncoderInfo(name="torchvision:resnet50", embed_dim=2048, image_size=224)

class FrameEncoder(nn.Module):
    """
    DINOv3 ViT-B/16 frame encoder with two-stage parameter-efficient fine-tuning:

    Stage 1 — LayerNorm tuning (~40K params, 0.04% of backbone):
        Re-calibrates feature normalization statistics for the forensic domain
        while keeping all other pre-trained representations frozen.

    Stage 2 — Last-block unfreezing (~7.1M params total, 8.3% of backbone):
        Adapts high-level semantic representations to deepfake-specific cues
        while retaining low- and mid-level features frozen.

    Falls back to a frozen ResNet50 backbone if timm / DINOv3 is unavailable.
    """

    def __init__(
        self,
        device: str = "cpu",
        layernorm_tuning: bool = True,
        unfreeze_last_block: bool = True,
    ):
        super().__init__()
        self.device = device

        built = _try_build_dinov3()
        if built is None:
            warnings.warn("DINOv3 not available; using ResNet50 fallback.")
            model, info = _build_resnet50_fallback()
            self.is_dinov3 = False
        else:
            model, info = built
            self.is_dinov3 = True

        self.model = model
        self.info = info
        self.image_size = info.image_size
        self.embed_dim = info.embed_dim

        # Stage 0: freeze entire backbone
        for p in self.model.parameters():
            p.requires_grad = False

        if self.is_dinov3:
            # Stage 1: unfreeze LayerNorm (gamma, beta) across all transformer blocks
            if layernorm_tuning:
                for m in self.model.modules():
                    if isinstance(m, nn.LayerNorm):
                        for p in m.parameters():
                            p.requires_grad = True

            # Stage 2: unfreeze the final transformer block
            if unfreeze_last_block:
                blocks = getattr(self.model, "blocks", None)
                if blocks is not None and len(blocks) > 0:
                    for p in blocks[-1].parameters():
                        p.requires_grad = True

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images [B, 3, H, W] -> L2-normalised embeddings [B, D]."""
        emb = self.model(x)
        return torch.nn.functional.normalize(emb, dim=-1)

    def trainable_param_count(self) -> int:
        """Number of parameters with requires_grad=True."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


