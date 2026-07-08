import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from deepfake_guard.models.lipfd import LipFDDetector
from deepfake_guard.models.lipfd.model import LipFD, build_model, VALID_ARCH_NAMES
from deepfake_guard.models.lipfd.preprocessing import preprocess_video, build_composite_images
from deepfake_guard.models.lipfd.region_awareness import RegionAwareResNet, get_backbone

print("All LipFD imports OK")
print("Valid architectures:", VALID_ARCH_NAMES)

det = LipFDDetector(arch="CLIP:ViT-L/14", device="cpu")
print("LipFDDetector instantiated:", det)

# Test via DeepfakeGuard orchestrator
from deepfake_guard import DeepfakeGuard
g = DeepfakeGuard(detector_type="lipfd")
print("DeepfakeGuard(detector_type='lipfd') OK:", g)
