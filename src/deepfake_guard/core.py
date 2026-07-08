from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Literal

import torch

from .models.dinov3.detector import Detector as DINOv3Detector
from .models.resnet18.detector import ResNet18Detector, detect_video_resnet18
from .models.ivyfake.detector import IvyFakeDetector
from .models.ivyfake.multiscale_detector import MultiScaleIvyFakeDetector
from .models.d3.detector import D3Detector
from .models.lipfd.detector import LipFDDetector
from .types import ModalityResult
from .utils.face_crop import FaceCropper
from .utils.preprocess import simulate_compression, stack_frames
from .utils.video_io import read_video_frames
from .utils.weights import load_weights


DetectorFn = Callable[[str], Dict[str, Any]]


class DeepfakeGuard:
    """
    Orchestrates multimodal deepfake detection pipelines.
    
    Supports multiple detector backends:
    - "dinov3": DINOv3-based detector (face cropping, 0.88+ AUROC)
    - "resnet18": ResNet18-based detector (full frames, pretrained ImageNet)
    - "ivyfake": IvyFake detector (CLIP-based with temporal/spatial analysis)
    - "ivyfake-multiscale": Multi-scale IvyFake (temporal pyramid at 0.5/1.0/2.0 fps)
    - "d3": D3 detector (training-free, second-order temporal features)
    - "lipfd": LipFD detector (audio-visual lip-sync detection, NeurIPS 2024)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        detector_type: Literal["dinov3", "resnet18", "ivyfake", "ivyfake-multiscale", "d3", "lipfd"] = "dinov3"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_type = detector_type


        self._pipelines: Dict[str, DetectorFn] = {}
        self._model_info: Dict[str, str] = {}
        
        # Visual detector (switchable)
        self.visual_detector = None
        self.face_cropper = None
        self.resnet_detector = None
        self.ivyfake_detector = None
        self.ivyfake_multiscale_detector = None
        self.d3_detector = None
        self.lipfd_detector = None
        self.visual_weights_loaded = False

        if detector_type == "dinov3":
            self._init_dinov3(weights_path)
        elif detector_type == "resnet18":
            self._init_resnet18()
        elif detector_type == "ivyfake":
            self._init_ivyfake(weights_path)
        elif detector_type == "ivyfake-multiscale":
            self._init_ivyfake_multiscale(weights_path)
        elif detector_type == "d3":
            self._init_d3()
        elif detector_type == "lipfd":
            self._init_lipfd(weights_path)
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        # Register visual modality
        self.register_modality(
            "visual",
            self._run_visual_analysis,
            f"{detector_type.upper()} deepfake detection"
        )

    # Path to the weights bundled inside the installed package
    _BUNDLED_DINOV3_WEIGHTS = Path(__file__).parent / "weights" / "dinov3_best_v3.pth"

    def _init_dinov3(self, weights_path: Optional[str] = None) -> None:
        """Initialize DINOv3-based detector.

        Weights are resolved in this order:
          1. Explicit ``weights_path`` argument (if provided and exists)
          2. Bundled weights shipped with the package
        """
        self.visual_detector = DINOv3Detector(device=self.device)
        self.face_cropper = FaceCropper(
            device=self.device,
            padding_ratio=0.3,
        )

        resolved = None
        if weights_path and os.path.exists(weights_path):
            resolved = weights_path
        elif self._BUNDLED_DINOV3_WEIGHTS.exists():
            resolved = str(self._BUNDLED_DINOV3_WEIGHTS)
        else:
            import warnings
            warnings.warn("No DINOv3 weights found — running with random initialisation.")

        if resolved:
            self.load_visual_weights(resolved)

    def _init_resnet18(self) -> None:
        """Initialize ResNet18-based detector (pretrained ImageNet weights, no fine-tuning)."""
        self.resnet_detector = ResNet18Detector(device=self.device)
        self.visual_weights_loaded = True

    def _init_ivyfake(self, weights_path: Optional[str] = None) -> None:
        """Initialize IvyFake detector (CLIP-based with explainable features)."""
        self.ivyfake_detector = IvyFakeDetector(
            device=self.device,
            weights_path=weights_path,
        )
        self.visual_weights_loaded = True

    def _init_ivyfake_multiscale(self, weights_path: Optional[str] = None) -> None:
        """Initialize multi-scale IvyFake detector (temporal pyramid)."""
        self.ivyfake_multiscale_detector = MultiScaleIvyFakeDetector(
            device=self.device,
            weights_path=weights_path,
        )
        self.visual_weights_loaded = True

    def _init_d3(self, encoder: str = "xclip-16") -> None:
        """Initialize D3 detector (training-free, second-order temporal features)."""
        self.d3_detector = D3Detector(encoder_name=encoder, device=self.device)
        self.visual_weights_loaded = True

    def _init_lipfd(self, weights_path: Optional[str] = None) -> None:
        """Initialize LipFD detector (audio-visual lip-sync deepfake detection, NeurIPS 2024).

        Requires pretrained weights (~1.68 GB). Download from:
          https://github.com/AaronComo/LipFD
        """
        if weights_path and not os.path.exists(weights_path):
            import warnings
            warnings.warn(f"LipFD weights not found at '{weights_path}' — running without weights (accuracy will be poor).")
            weights_path = None
        self.lipfd_detector = LipFDDetector(weights_path=weights_path, device=self.device)
        self.visual_weights_loaded = weights_path is not None

    def set_detector(
        self,
        detector_type: Literal["dinov3", "resnet18", "ivyfake", "ivyfake-multiscale", "d3", "lipfd"],
        weights_path: Optional[str] = None,
    ) -> None:
        """
        Switch detector backend.
        
        Args:
            detector_type: "dinov3", "resnet18", "ivyfake", "ivyfake-multiscale", "d3", or "lipfd"
            weights_path: Path to weights (used for dinov3, ivyfake, and lipfd)
        """
        if detector_type == self.detector_type:
            return
            
        self.detector_type = detector_type

        # Reset all detector handles
        self._pipelines.clear()
        self._model_info.clear()
        self.visual_detector = None
        self.face_cropper = None
        self.resnet_detector = None
        self.ivyfake_detector = None
        self.ivyfake_multiscale_detector = None
        self.d3_detector = None
        self.lipfd_detector = None
        self.visual_weights_loaded = False
        
        # Reinitialize
        if detector_type == "dinov3":
            self._init_dinov3(weights_path)
        elif detector_type == "resnet18":
            self._init_resnet18()
        elif detector_type == "ivyfake":
            self._init_ivyfake(weights_path)
        elif detector_type == "ivyfake-multiscale":
            self._init_ivyfake_multiscale(weights_path)
        elif detector_type == "d3":
            self._init_d3()
        elif detector_type == "lipfd":
            self._init_lipfd(weights_path)
            
        # Re-register visual modality
        self.register_modality(
            "visual",
            self._run_visual_analysis,
            f"{detector_type.upper()} deepfake detection"
        )

    def register_modality(self, name: str, pipeline: DetectorFn, description: str) -> None:
        """Register an additional modality pipeline."""
        self._pipelines[name] = pipeline
        self._model_info[name] = description

    def load_visual_weights(self, path: str) -> None:
        """Load weights for DINOv3 detector."""
        if self.detector_type != "dinov3":
            import warnings
            warnings.warn(f"load_visual_weights is only applicable to the dinov3 detector (current: {self.detector_type}).")
            return
            
        if os.path.exists(path):
            load_weights(self.visual_detector, path)
            self.visual_weights_loaded = True
            return

        import warnings
        warnings.warn(f"Weights file not found: {path}")

    def detect_video(self, video_path: str) -> Dict[str, Any]:
        """Run the full detection pipeline on a video file."""
        result: Dict[str, Any] = {
            "model_info": {
                "version": "0.4.0",
                "detector_type": self.detector_type,
                **self._model_info
            },
            "overall_label": "UNKNOWN",
            "overall_score": 0.0,
            "modality_results": {},
            "errors": [],
        }

        if not os.path.exists(video_path):
            result["errors"].append(f"Video file not found: {video_path}")
            return result

        for name, pipeline in self._pipelines.items():
            try:
                modality_res = pipeline(video_path)
                if "error" in modality_res:
                    result["errors"].append(modality_res["error"])
                    continue
                result["modality_results"][name] = modality_res
            except Exception as e:
                result["errors"].append(f"Error in {name}: {str(e)}")

        overall_score = self._aggregate_scores(result["modality_results"])
        result["overall_score"] = overall_score
        result["overall_label"] = "FAKE" if overall_score > 0.5 else "REAL"

        return result

    # Per-detector trust priors: reflect how calibrated each score is.
    # DINOv3 is fine-tuned on deepfakes  → high trust.
    # D3 is training-free but principled  → medium trust.
    # ResNet18 + IvyFake have untrained classification heads → low trust.
    _DETECTOR_TRUST: Dict[str, float] = {
        "dinov3":   1.0,
        "lipfd":    0.85,             # NeurIPS 2024, trained on lip-sync deepfakes; 91.2% acc on FakeAVCeleb
        "d3":       0.6,
        "ivyfake":  0.5,              # principled CLIP cosine-sim signal, not trained on deepfakes
        "ivyfake-multiscale": 0.55,   # multi-scale temporal pyramid — slightly better coverage
        "resnet18": 0.2,              # untrained head on ImageNet features
    }

    def _aggregate_scores(self, modality_results: Dict[str, Dict[str, Any]]) -> float:
        """Confidence-weighted aggregate across modalities.

        Each score is weighted by:
          trust_prior(detector) × certainty(score)

        where certainty = 2 * |score - 0.5|  (0 = completely uncertain, 1 = maximal).
        This ensures that a highly confident trained detector dominates an uncertain
        untrained one, and that scores close to 0.5 contribute little to the result.
        """
        weighted_sum = 0.0
        weight_total = 0.0
        for res in modality_results.values():
            if not isinstance(res, dict):
                continue
            score = res.get("score")
            if not isinstance(score, (int, float)):
                continue
            det_type = res.get("details", {}).get("detector_type", self.detector_type)
            trust = self._DETECTOR_TRUST.get(det_type, 0.5)
            certainty = abs(float(score) - 0.5) * 2.0  # ∈ [0, 1]
            weight = trust * (0.1 + 0.9 * certainty)   # floor at 10 % of trust so zero-certainty still votes
            weighted_sum += float(score) * weight
            weight_total += weight
        if weight_total == 0.0:
            return 0.0
        return float(weighted_sum / weight_total)

    def _run_visual_analysis(self, video_path: str) -> Dict[str, Any]:
        """Run visual analysis based on current detector type."""
        if self.detector_type == "dinov3":
            return self._run_dinov3_analysis(video_path)
        elif self.detector_type == "resnet18":
            return self._run_resnet18_analysis(video_path)
        elif self.detector_type == "ivyfake":
            return self._run_ivyfake_analysis(video_path)
        elif self.detector_type == "ivyfake-multiscale":
            return self._run_ivyfake_multiscale_analysis(video_path)
        elif self.detector_type == "d3":
            return self._run_d3_analysis(video_path)
        elif self.detector_type == "lipfd":
            return self._run_lipfd_analysis(video_path)
        else:
            return {"error": f"Unknown detector type: {self.detector_type}"}

    def _run_dinov3_analysis(self, video_path: str) -> Dict[str, Any]:
        """DINOv3-based visual analysis with face cropping."""
        try:
            frames = read_video_frames(video_path)
        except Exception as exc:
            return {"error": f"Error reading video: {exc}"}

        if not frames:
            return {"error": "Could not read frames from video"}

        cropped_frames = []
        for frame in frames:
            crop = self.face_cropper.crop(frame, return_metadata=False)
            if crop:
                cropped_frames.append(crop)

        if not cropped_frames:
            return {"error": "No faces detected in video"}

        processed_frames = simulate_compression(cropped_frames)
        input_tensor = stack_frames(
            processed_frames,
            self.visual_detector.preprocess,
            self.device
        )

        det_result = self.visual_detector.predict_video(input_tensor)
        det_result = det_result.__dict__ if hasattr(det_result, "__dict__") else det_result

        score = float(det_result.get("score", 0.0))
        label = det_result.get("label", "UNKNOWN")
        details = det_result.get("details", {})
        details["detector_type"] = "dinov3"

        return ModalityResult(score=score, label=label, details=details).__dict__

    def _run_resnet18_analysis(self, video_path: str) -> Dict[str, Any]:
        """ResNet18-based visual analysis (full frames, no face cropping)."""
        result = self.resnet_detector.predict_video(video_path)
        return ModalityResult(
            score=result["score"],
            label=result["label"],
            details=result["details"]
        ).__dict__

    def _run_ivyfake_analysis(self, video_path: str) -> Dict[str, Any]:
        """IvyFake-based visual analysis (CLIP-based with temporal/spatial features)."""
        result = self.ivyfake_detector.predict_video(video_path)
        return ModalityResult(
            score=result["score"],
            label=result["label"],
            details=result["details"]
        ).__dict__

    def _run_ivyfake_multiscale_analysis(self, video_path: str) -> Dict[str, Any]:
        """Multi-scale IvyFake analysis (temporal pyramid at multiple fps)."""
        result = self.ivyfake_multiscale_detector.predict_video(video_path)
        return ModalityResult(
            score=result["score"],
            label=result["label"],
            details=result["details"]
        ).__dict__

    def _run_d3_analysis(self, video_path: str) -> Dict[str, Any]:
        """D3-based visual analysis (training-free, second-order temporal features)."""
        result = self.d3_detector.predict_video(video_path)
        return ModalityResult(
            score=result["score"],
            label=result["label"],
            details=result["details"]
        ).__dict__

    def _run_lipfd_analysis(self, video_path: str) -> Dict[str, Any]:
        """LipFD audio-visual lip-sync deepfake analysis (NeurIPS 2024)."""
        if self.lipfd_detector is None:
            return {"error": "LipFD detector not initialised"}

        try:
            result = self.lipfd_detector.predict_video(video_path)
        except Exception as exc:
            return {"error": f"LipFD analysis failed: {exc}"}

        if result.get("overall_label") == "ERROR":
            errors = result.get("errors", ["Unknown LipFD error"])
            return {"error": "; ".join(errors)}

        score = float(result.get("overall_score", 0.0))
        label = result.get("overall_label", "UNKNOWN")
        details = result.get("modality_results", {}).get("audio_visual", {}).get("details", {})
        details["detector_type"] = "lipfd"
        details["model"] = result.get("model_info", {})

        return ModalityResult(score=score, label=label, details=details).__dict__