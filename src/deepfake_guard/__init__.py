from .core import DeepfakeGuard
from .types import ModalityResult
from .models.dinov3.detector import Detector as DINOv3Detector
from .models.resnet18.detector import ResNet18Detector, detect_video_resnet18
from .models.ivyfake.detector import IvyFakeDetector, detect_video_ivyfake
from .models.d3.detector import D3Detector, create_d3_detector

__version__ = "0.4.0"
__all__ = [
    "DeepfakeGuard",
    "ModalityResult",
    "DINOv3Detector",
    "ResNet18Detector",
    "detect_video_resnet18",
    "IvyFakeDetector",
    "detect_video_ivyfake",
    "D3Detector",
    "create_d3_detector",
]