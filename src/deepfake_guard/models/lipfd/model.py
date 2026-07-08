"""
LipFD – Lip Forgery Detection model.

Combines an OpenAI CLIP vision encoder (for global audio-visual features)
with a Region-Awareness ResNet-50 backbone (for multi-scale spatial features)
to detect lip-sync deepfakes by spotting temporal inconsistency between
audio and visual signals.

Original Paper:
    Liu et al., "Lips Are Lying: Spotting the Temporal Inconsistency
    between Audio and Visual in Lip-Syncing DeepFakes", NeurIPS 2024.
    https://arxiv.org/abs/2401.15668

Original Implementation:
    https://github.com/AaronPeng920/LipFD

Adapted for DeepFakeGuard by the DeepFakeGuard team.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .region_awareness import get_backbone


# ---------------------------------------------------------------------------
# CLIP feature dimension lookup
# ---------------------------------------------------------------------------
CLIP_FEAT_DIM = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
}

# Supported CLIP architectures for LipFD
VALID_ARCH_NAMES = [
    "CLIP:ViT-B/32",
    "CLIP:ViT-B/16",
    "CLIP:ViT-L/14",
]


# ---------------------------------------------------------------------------
# LipFD model
# ---------------------------------------------------------------------------
class LipFD(nn.Module):
    """
    Lip Forgery Detection network.

    Architecture:
        1. A CLIP vision encoder extracts a global feature vector from the
           full composite image (mel-spectrogram + video frames).
        2. A 5×5 stride-5 convolution down-samples the composite image from
           1120×1120 to 224×224 (the CLIP input resolution).
        3. A Region-Awareness ResNet-50 backbone processes multi-scale
           crops of individual frames, concatenating each regional feature
           with the global CLIP feature and computing attention weights.
        4. A final linear layer produces a binary prediction logit.

    Parameters
    ----------
    clip_arch : str
        CLIP architecture name, e.g. ``"ViT-L/14"``.
    num_classes : int
        Number of output classes (default 1 for binary).
    """

    def __init__(self, clip_arch: str = "ViT-L/14", num_classes: int = 1):
        super().__init__()

        if clip_arch not in CLIP_FEAT_DIM:
            raise ValueError(
                f"Unsupported CLIP architecture '{clip_arch}'. "
                f"Choose from {list(CLIP_FEAT_DIM.keys())}"
            )

        self.clip_arch = clip_arch
        self.global_feat_dim = CLIP_FEAT_DIM[clip_arch]

        # Down-sample conv: (B, 3, 1120, 1120) → (B, 3, 224, 224)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=5)

        # CLIP vision encoder (frozen during fine-tuning)
        try:
            import clip as openai_clip
        except ImportError as _clip_err:
            # Workaround: openai-clip uses deprecated pkg_resources.packaging
            # which breaks with newer setuptools.  Patch it in.
            try:
                import packaging  # noqa: F811
                import pkg_resources
                pkg_resources.packaging = packaging  # type: ignore[attr-defined]
                import clip as openai_clip
            except Exception:
                raise ImportError(
                    "OpenAI CLIP is required. Install with:\n"
                    "  pip install git+https://github.com/openai/CLIP.git\n"
                    "or\n"
                    "  pip install openai-clip packaging"
                ) from _clip_err
        self.encoder, self.clip_preprocess = openai_clip.load(
            clip_arch, device="cpu",
        )
        # Freeze CLIP encoder by default
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Region-awareness backbone (ResNet-50)
        self.backbone = get_backbone(global_feat_dim=self.global_feat_dim)

    # ------------------------------------------------------------------
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global CLIP features from the full composite image.

        Parameters
        ----------
        x : Tensor of shape (B, 3, 1120, 1120)
            The full composite image (mel-spectrogram stacked on video frames).

        Returns
        -------
        features : Tensor of shape (B, global_feat_dim)
        """
        x = self.conv1(x)  # (B, 3, 224, 224)
        features = self.encoder.encode_image(x)
        return features.float()

    # ------------------------------------------------------------------
    def forward(self, crops, feature):
        """
        Forward pass.

        Parameters
        ----------
        crops : list[list[Tensor]]
            Multi-scale crops: ``crops[scale][frame]``, each (B, 3, 224, 224).
        feature : Tensor
            Global CLIP features, shape (B, global_feat_dim).

        Returns
        -------
        pred_score   : (B, 1) — raw logit
        weights_max  : list[Tensor] — for RA loss
        weights_org  : list[Tensor] — for RA loss
        """
        return self.backbone(crops, feature)


# ---------------------------------------------------------------------------
# Region-Awareness Loss (RA-Loss)
# ---------------------------------------------------------------------------
class RALoss(nn.Module):
    """
    Region-Awareness Loss from the LipFD paper.

    Encourages the model to focus on the most discriminative region
    by penalising when attention weights are uniformly distributed.
    """

    def forward(self, alphas_max, alphas_org):
        loss = torch.tensor(0.0, device=alphas_org[0].device)
        batch_size = alphas_org[0].shape[0]

        for i in range(len(alphas_org)):
            loss_wt = torch.tensor(0.0, device=alphas_org[0].device)
            for j in range(batch_size):
                diff = (alphas_max[i][j] - alphas_org[i][j]).sum()
                loss_wt = loss_wt + 10.0 / torch.exp(diff)
            loss = loss + loss_wt / batch_size

        return loss


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def build_model(arch_name: str) -> LipFD:
    """
    Build a LipFD model from an architecture string.

    Parameters
    ----------
    arch_name : str
        e.g. ``"CLIP:ViT-L/14"`` or ``"CLIP:ViT-B/32"``.
    """
    assert arch_name in VALID_ARCH_NAMES, (
        f"Unknown arch '{arch_name}'. Valid: {VALID_ARCH_NAMES}"
    )
    clip_variant = arch_name.split(":", 1)[1]
    return LipFD(clip_arch=clip_variant)


def get_loss() -> RALoss:
    """Return the region-awareness loss function."""
    return RALoss()
