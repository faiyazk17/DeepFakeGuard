"""
Region-Awareness Backbone for LipFD.

Implements a modified ResNet-50 backbone that processes multi-scale face crops
and fuses them with global CLIP features using learned attention weights.
This enables "region awareness" — the model attends to different spatial
regions (full face, mid-face, lip area) with learned importance.

Original Paper:
    Liu et al., "Lips Are Lying: Spotting the Temporal Inconsistency
    between Audio and Visual in Lip-Syncing DeepFakes", NeurIPS 2024.
    https://arxiv.org/abs/2401.15668

Original Implementation:
    https://github.com/AaronPeng920/LipFD

Adapted for DeepFakeGuard by the DeepFakeGuard team.
ResNet-50 architecture based on torchvision (BSD-3-Clause license).
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax
from typing import Type, Any, Callable, Union, List, Optional

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# ---------------------------------------------------------------------------
# Pre-trained weight URLs (ImageNet-1K)
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
}


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------
def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1,
    groups: int = 1, dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
    )


# ---------------------------------------------------------------------------
# BasicBlock & Bottleneck
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ---------------------------------------------------------------------------
# Region-Aware ResNet
# ---------------------------------------------------------------------------
class RegionAwareResNet(nn.Module):
    """
    Modified ResNet-50 that processes multi-scale face crops with
    learned attention weighting, fused with global CLIP features.

    Input:
        x       – list of lists: ``x[scale][frame]`` each (B, 3, 224, 224)
                  typically 3 scales × 5 frames.
        feature – global features from CLIP encoder, shape (B, global_dim).

    Output:
        pred_score  – (B, 1) logit (not sigmoided).
        weights_max – list of max attention weights per frame (for RA loss).
        weights_org – list of first-scale attention weights (for RA loss).
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        global_feat_dim: int = 768,
        num_classes: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a "
                f"3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        # ---- ResNet stem ----
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- ResNet stages ----
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Region-awareness heads ----
        regional_dim = 512 * block.expansion  # 2048 for Bottleneck
        fused_dim = regional_dim + global_feat_dim
        self.get_weight = nn.Sequential(
            nn.Linear(fused_dim, 1),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(fused_dim, num_classes)

        # ---- Weight initialisation ----
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # ------------------------------------------------------------------
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [
            block(self.inplanes, planes, stride, downsample,
                  self.groups, self.base_width, previous_dilation, norm_layer)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups,
                      base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer)
            )
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def _forward_impl(
        self,
        x: List[List[Tensor]],
        feature: Tensor,
    ):
        """
        Parameters
        ----------
        x : list[list[Tensor]]
            ``x[scale_idx][frame_idx]`` – each tensor is (B, 3, 224, 224).
            Typically 3 scales, 5 frames each.
        feature : Tensor
            Global features from CLIP, shape (B, global_feat_dim).

        Returns
        -------
        pred_score   : (B, 1)
        weights_max  : list[Tensor]  — for RA loss
        weights_org  : list[Tensor]  — for RA loss
        """
        num_frames = len(x[0])
        num_scales = len(x)

        parts: List[Tensor] = []
        weights_max: List[Tensor] = []
        weights_org: List[Tensor] = []

        for i in range(num_frames):
            features_list: List[Tensor] = []
            weight_list: List[Tensor] = []

            for j in range(num_scales):
                f = x[j][i]                          # (B, 3, 224, 224)
                f = self.conv1(f)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)
                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)
                f = torch.flatten(f, 1)              # (B, 2048)
                fused = torch.cat([f, feature], dim=1)  # (B, 2048+global)
                features_list.append(fused)
                weight_list.append(self.get_weight(fused))

            features_stack = torch.stack(features_list, dim=2)  # (B, D, S)
            weights_stack = torch.stack(weight_list, dim=2)     # (B, 1, S)
            weights_stack = softmax(weights_stack, dim=2)

            weights_max.append(
                weights_stack[:, :, :num_scales].max(dim=2)[0]
            )
            weights_org.append(weights_stack[:, :, 0])

            # Weighted combination across scales
            weighted = features_stack.mul(weights_stack).sum(2)
            weighted = weighted.div(weights_stack.sum(2))
            parts.append(weighted)

        # Average across frames
        parts_stack = torch.stack(parts, dim=0)             # (F, B, D)
        out = parts_stack.sum(0).div(parts_stack.shape[0])  # (B, D)

        pred_score = self.fc(out)
        return pred_score, weights_max, weights_org

    def forward(self, x: List[List[Tensor]], feature: Tensor):
        return self._forward_impl(x, feature)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_backbone(
    global_feat_dim: int = 768,
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> RegionAwareResNet:
    """
    Build a ResNet-50 based region-awareness backbone.

    Parameters
    ----------
    global_feat_dim : int
        Dimensionality of the global CLIP features that will be
        concatenated with regional ResNet features.
    pretrained : bool
        If True, load ImageNet-pretrained ResNet-50 weights
        (classification head will NOT match — partial load only).
    """
    model = RegionAwareResNet(
        Bottleneck, [3, 4, 6, 3],
        global_feat_dim=global_feat_dim,
        num_classes=1,
        **kwargs,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(
            MODEL_URLS["resnet50"], progress=progress,
        )
        # Partial load — skip keys that don't match (fc, get_weight)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
