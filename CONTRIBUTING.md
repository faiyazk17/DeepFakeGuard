# Contributing to DeepFakeGuard

Thank you for your interest in contributing to DeepFakeGuard! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/aryanbiswas16/DeepfakeVidDetection.git
cd DeepfakeVidDetection
git checkout v0.4.0-multi-detector
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e ".[dev,ui,api]"
```

4. **Run tests:**
```bash
pytest
```

## 📝 Contribution Guidelines

### Code Style

We use **Black** for code formatting:
```bash
black src/ tests/
```

And **flake8** for linting:
```bash
flake8 src/
```

### Adding a New Detector

To add a new detector (e.g., Audio, Metadata, or a new visual method):

1. **Create the detector module:**
```bash
mkdir src/deepfake_guard/models/your_detector
touch src/deepfake_guard/models/your_detector/__init__.py
touch src/deepfake_guard/models/your_detector/detector.py
```

2. **Implement the detector class:**
```python
# src/deepfake_guard/models/your_detector/detector.py

from typing import Dict, Any

class YourDetector:
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Initialize your model
    
    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Detect deepfakes in video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with keys:
                - score: float (0-1, higher = more fake)
                - label: "REAL" or "FAKE"
                - details: Dict with additional info
        """
        # Your detection logic
        return {
            "score": 0.5,
            "label": "UNKNOWN",
            "details": {"detector_type": "your_detector"}
        }
```

3. **Register in core.py:**
Add your detector to the `set_detector()` method in `src/deepfake_guard/core.py`.

4. **Update documentation:**
- Add to README.md detector comparison table
- Update this CONTRIBUTING.md
- Add citation if using an existing method

### Pull Request Process

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** with clear, descriptive commit messages

3. **Add tests** for new functionality

4. **Update documentation** (README, docstrings, etc.)

5. **Run linting and tests:**
```bash
black src/ tests/
flake8 src/
pytest
```

6. **Submit a pull request** with:
   - Clear description of changes
   - Motivation for the change
   - Any breaking changes
   - Screenshots (if UI changes)

## 🐛 Reporting Bugs

When reporting bugs, please include:

- **System information:** OS, Python version, GPU/CPU
- **Library versions:** `pip list` output
- **Error message:** Full traceback
- **Minimal reproducible example:** Code that triggers the bug
- **Expected vs actual behavior**

## 💡 Feature Requests

We welcome feature requests! Please:

- Check existing issues first
- Clearly describe the feature
- Explain the use case
- Consider implementation complexity

## 📋 Code Review Criteria

Pull requests are reviewed for:

- ✅ Code quality and style
- ✅ Documentation completeness
- ✅ Test coverage
- ✅ Performance impact
- ✅ Backward compatibility
- ✅ Proper error handling

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Citation file (for significant contributions)

## 📞 Contact

- **Issues:** GitHub Issues
- **Email:** aryanbiswas16@gmail.com

---

Thank you for contributing to DeepFakeGuard!
