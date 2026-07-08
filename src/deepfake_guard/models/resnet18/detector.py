"""
ResNet18-based video detector (pretrained ImageNet weights)
Simple frame-level classification without face cropping
"""

from __future__ import annotations

import os
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from timm import create_model
from torchvision import transforms

from ...types import ModalityResult


class ResNet18Detector(nn.Module):
    """
    ResNet18-based video detector.
    
    Differences from DINOv3 approach:
    - No face cropping (analyzes full frames)
    - Uniform frame sampling
    - ResNet18 backbone (lighter than ViT)
    - Uses pretrained ImageNet weights (not fine-tuned on deepfakes)
    """
    
    def __init__(
        self,
        device: str = "cpu",
        num_frames: int = 16,
        threshold: float = 0.5
    ):
        super().__init__()
        self.device = device
        self.num_frames = num_frames
        self.threshold = threshold
        
        # Load pretrained ResNet18
        self.model = create_model("resnet18", pretrained=True, num_classes=2)
        self.model.eval()
        self.model.to(device)

        # Preprocessing (ImageNet normalisation)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_frames(self, video_path: str) -> list:
        """Uniformly sample frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return []
            
        step = max(total // self.num_frames, 1)
        
        idx = 0
        while cap.isOpened() and len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            idx += 1
        
        cap.release()
        return frames
    
    @torch.no_grad()
    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Detect AI-generated content in video.
        
        Returns:
            dict with:
                - score: P(fake) 0-1
                - label: "FAKE" or "REAL"
                - details: per-frame info
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        if not frames:
            return {
                "score": 0.0,
                "label": "UNKNOWN",
                "details": {"error": "Could not extract frames"}
            }
        
        # Preprocess
        inputs = torch.stack([self.transform(f) for f in frames]).to(self.device)
        
        # Inference
        outputs = self.model(inputs)
        probs = torch.softmax(outputs, dim=1)
        fake_probs = probs[:, 1]  # P(fake) for each frame
        
        # Aggregate
        mean_score = float(fake_probs.mean().item())
        per_frame = fake_probs.cpu().tolist()
        
        return {
            "score": mean_score,
            "label": "FAKE" if mean_score > self.threshold else "REAL",
            "details": {
                "per_frame_fake_probs": [float(p) for p in per_frame],
                "frame_count": len(frames),
                "detector_type": "resnet18",
                "note": "Uses pretrained ImageNet weights (not fine-tuned)"
            }
        }
    
    def forward(self, x):
        """Forward pass for single images."""
        return self.model(x)


def create_resnet18_detector(device: str = "cpu", **kwargs) -> ResNet18Detector:
    """Factory function for ResNet18 detector."""
    return ResNet18Detector(device=device, **kwargs)


# Detection function for DeepfakeGuard integration
def detect_video_resnet18(video_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Standalone detection function for DeepfakeGuard modality registration.
    
    Args:
        video_path: Path to video file
        device: "cpu" or "cuda"
        
    Returns:
        dict compatible with DeepfakeGuard format
    """
    detector = ResNet18Detector(device=device)
    result = detector.predict_video(video_path)
    
    # Convert to ModalityResult format
    return ModalityResult(
        score=result["score"],
        label=result["label"],
        details=result["details"]
    ).__dict__