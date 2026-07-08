# DeepFakeGuard v0.4.0

A multimodal Python library for deepfake detection with **five switchable detector backends** — ready for research and conference demos.

---

## Detectors

| | DINOv3 | ResNet18 | IvyFake | D3 | LipFD |
|---|---|---|---|---|---|
| **Architecture** | ViT-B/16 | CNN | CLIP ViT-B/32 | Encoder-agnostic | CLIP ViT-L/14 + ResNet-50 |
| **Method** | Face-based | Full-frame | Temporal + spatial artifacts | Second-order temporal features | Audio-visual lip-sync |
| **Training** | Fine-tuned on deepfakes | Pretrained ImageNet | Pretrained CLIP | **Training-free** | Trained on lip-sync fakes |
| **Accuracy** | 0.88+ AUROC | Baseline | Good | Varies by encoder | 0.962 AUROC (FakeAVCeleb) |
| **Weights required** | ✅ Yes | ❌ No | ❌ No | ❌ No | ✅ Yes (1.68 GB) |
| **Paper** | DINOv2 (Meta, 2023) | — | IvyFake (2024) | ICCV 2025 | NeurIPS 2024 |

---

## Quick Start

### Prerequisites
Python 3.8+, PyTorch 2.0+

### Install (local development)
```bash
git clone https://github.com/aryanbiswas16/DeepFakeGuard.git
cd DeepFakeGuard
pip install -e .

# Or with UI + API extras:
pip install -e ".[all]"
```

### Run the demo GUI
```bash
streamlit run ui/enhanced_gui.py
```
Open **http://localhost:8501** — use the sidebar to select a detector and upload a video.

### Run the API server
```bash
# No weights needed (ResNet18)
python -m uvicorn app.main:app --reload

# With DINOv3 weights
$env:DEEPFAKE_GUARD_DETECTOR="dinov3"
$env:DEEPFAKE_GUARD_WEIGHTS="weights/dinov3_best_v3.pth"
python -m uvicorn app.main:app --reload
```

### Smoke test all detectors
```bash
python test_detectors.py
# or with a real video:
python test_detectors.py --video path/to/video.mp4
```

---

## Python API

```python
from deepfake_guard import DeepfakeGuard

# Training-free — no weights needed
guard = DeepfakeGuard(detector_type="d3")
result = guard.detect_video("video.mp4")
print(result["overall_label"], result["overall_score"])

# Switch detectors at runtime
guard.set_detector("resnet18")
result2 = guard.detect_video("video.mp4")

# DINOv3 — highest accuracy (requires weights)
guard.set_detector("dinov3", weights_path="weights/dinov3_best_v3.pth")
result3 = guard.detect_video("video.mp4")

# LipFD — audio-visual lip-sync detection (requires weights)
guard.set_detector("lipfd", weights_path="weights/lipfd_ckpt.pth")
result4 = guard.detect_video("video.mp4")
```

### Result format
```python
{
    "overall_label": "FAKE",        # or "REAL"
    "overall_score": 0.73,          # 0.0–1.0, higher = more likely fake
    "model_info": { "detector_type": "d3", ... },
    "modality_results": {
        "visual": {
            "score": 0.73,
            "label": "FAKE",
            "details": { "volatility": 0.12, "frame_count": 16, ... }
        }
    },
    "errors": []
}
```

---

## LipFD Detector — Audio-Visual Lip-Sync Detection

LipFD (*Lip Forgery Detection*, Liu et al. NeurIPS 2024) detects **lip-sync deepfakes** by
analysing the temporal consistency between audio and visual lip movements.

- **Dual pathway**: CLIP ViT-L/14 (global audio-visual features) + Region-Aware ResNet-50 (multi-scale spatial attention)
- **Input**: Mel-spectrogram + 5-frame composite image (1000×2500)
- **Weights**: `weights/lipfd_ckpt.pth` (1.68 GB) — download from [AaronComo/LipFD](https://github.com/AaronComo/LipFD)

```python
from deepfake_guard.models.lipfd import LipFDDetector

det = LipFDDetector(weights_path="weights/lipfd_ckpt.pth")
result = det.predict_video("video.mp4")
```

**FakeAVCeleb v1.2 benchmark** (250 samples per class):

| Subset | Accuracy | AUROC |
|--------|----------|-------|
| FakeVideo-FakeAudio | **91.2%** | **0.962** |
| FakeVideo-RealAudio | 75.2% | 0.821 |
| RealVideo-RealAudio (specificity) | 87.6% TNR | — |

---

## D3 Detector — Encoder Options

D3 (*Detection by Difference of Differences*, Zheng et al. ICCV 2025) is training-free.
It measures **second-order temporal volatility**: real videos have high motion variance,
AI-generated videos have constrained, low-variance motion.

The algorithm works with any pre-trained encoder — select one in the GUI sidebar or via code:

| Encoder | Notes |
|---------|-------|
| `xclip-16` | **Recommended** — video-language model, best temporal sensitivity |
| `xclip-32` | Larger patch size variant |
| `clip-16` | OpenAI CLIP ViT-B/16 |
| `clip-32` | OpenAI CLIP ViT-B/32 |
| `dino-base` | Meta DINOv2-Base (facebook/dinov2-base) |
| `dino-large` | Meta DINOv2-Large (facebook/dinov2-large) |
| `resnet-18` | Lightweight CNN, fastest |
| `mobilenet-v3` | Lightweight CNN alternative |

```python
from deepfake_guard.models.d3 import D3Detector

det = D3Detector(encoder_name="xclip-16", threshold=2.5)
result = det.predict_video("video.mp4")
```

---

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/detect?detector=d3` | POST | Analyse uploaded video |
| `/switch/{type}` | GET | Switch active detector |
| `/detectors` | GET | List all detectors |

```bash
curl -X POST "http://localhost:8000/detect?detector=d3" -F "file=@video.mp4"
```

---

## File Structure

```
deepfake-guard/
├── src/deepfake_guard/
│   ├── core.py                # DeepfakeGuard — main API, detector switching
│   ├── types.py               # ModalityResult dataclass
│   ├── models/
│   │   ├── dinov3/            # ViT-B/16 face-based detector
│   │   ├── resnet18/          # CNN full-frame detector
│   │   ├── ivyfake/           # CLIP explainable detector
│   │   ├── d3/                # Training-free temporal detector (ICCV 2025)
│   │   └── lipfd/             # Audio-visual lip-sync detector (NeurIPS 2024)
│   └── utils/                 # Face cropping, preprocessing, video I/O, weights
├── app/main.py                # FastAPI server
├── ui/
│   ├── enhanced_gui.py        # Streamlit demo (standalone, no server needed)
│   └── demo_frontend.py       # Streamlit client for the API server
├── weights/                   # Place trained weights here
├── test_detectors.py          # Smoke test for all 5 detectors
└── pyproject.toml
```

---

## Adding a New Detector

```python
# 1. Create src/deepfake_guard/models/mydetector/detector.py
def predict_video(video_path: str) -> dict:
    return {"score": 0.6, "label": "FAKE", "details": {}}

# 2. Add to core.py — _init_mydetector() + _run_mydetector_analysis()

# 3. Register a modality at runtime (for one-off use):
guard.register_modality("audio", my_audio_fn, "Audio deepfake detection")
```

---

## Citations

**D3 (ICCV 2025):**
```bibtex
@inproceedings{zheng2025d3,
  title     = {D3: Training-Free AI-Generated Video Detection Using Second-Order Features},
  author    = {Zheng, Chende and Suo, Ruiqi and Lin, Chenhao and Zhao, Zhengyu and
               Yang, Le and Liu, Shuai and Yang, Minghui and Wang, Cong and Shen, Chao},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2508.00701}
}
```

**IvyFake:**
```bibtex
@software{ivyfake2024,
  author = {Khan, Hamza and et al.},
  title  = {IvyFake: An Explainable and Optimized AI-Generated Image Detection},
  url    = {https://github.com/HamzaKhan760/IvyFakeGenDetector},
  year   = {2024}
}
```

**DINOv2 (Meta):**
```bibtex
@article{oquab2023dinov2,
  title   = {DINOv2: Learning Robust Visual Features without Supervision},
  author  = {Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
  journal = {arXiv preprint arXiv:2304.07193},
  year    = {2023}
}
```

**DeepFakeGuard:**
```bibtex
@software{deepfakeguard2026,
  author = {Biswas, Aryan},
  title  = {DeepFakeGuard: A Unified Multi-Detector Framework for Deepfake Detection},
  url    = {https://github.com/aryanbiswas16/DeepFakeGuard},
  year   = {2026}
}
```

**LipFD (NeurIPS 2024):**
```bibtex
@inproceedings{liu2024lipfd,
  title     = {Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes},
  author    = {Liu, Weian and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2401.15668}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
