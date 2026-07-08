"""
Preprocessing pipeline for the LipFD detector.

Handles the full inference pre-processing chain:
    video file  →  frames + audio  →  mel-spectrogram  →  composite image
                →  multi-scale crops  →  tensors ready for LipFD.

The composite image layout matches the original LipFD training format
(see https://github.com/AaronPeng920/LipFD for reference):
    ┌──────────────────────────────────────────────────┐
    │         mel-spectrogram  (500 × 2500 px)         │  ← audio
    ├──────────────────────────────────────────────────┤
    │ frame₀ │ frame₁ │ frame₂ │ frame₃ │ frame₄     │  ← video
    │ 500×500│ 500×500│ 500×500│ 500×500│ 500×500     │
    └──────────────────────────────────────────────────┘

Dependencies: opencv-python, librosa, numpy, torch, torchvision, matplotlib
Optional:     ffmpeg on PATH (for audio extraction from video)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T

try:
    import librosa
    from librosa import feature as audio_feat
except ImportError:
    librosa = None  # type: ignore[assignment]
    audio_feat = None

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants (matching original LipFD preprocessing)
# ---------------------------------------------------------------------------
WINDOW_LEN = 5          # frames per sample group
N_EXTRACT = 10           # number of starting-point samples from a video
FRAME_SIZE = 500         # resize each frame to 500×500
CLIP_INPUT_SIZE = 1120   # composite image resize for CLIP encoder
CROP_SIZE = 224          # crop resize for ResNet backbone

# CLIP normalisation (OpenAI CLIP stats)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Multi-scale crop indices (from 224×224 base)
# crops[0]: 1.0× (full 224)   → identity
# crops[1]: 0.65× (center)    → [28:196, 28:196]
# crops[2]: 0.45× (center)    → [61:163, 61:163]
CROP_INDICES = [(28, 196), (61, 163)]


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------
def extract_audio(video_path: str, output_wav: str) -> bool:
    """
    Extract audio track from a video file to a WAV file using ffmpeg.

    Returns True on success, False if ffmpeg is unavailable or the video
    has no audio track.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn",                     # no video
                "-acodec", "pcm_s16le",    # PCM 16-bit
                "-ar", "16000",            # 16 kHz
                "-ac", "1",                # mono
                output_wav,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        return result.returncode == 0 and os.path.exists(output_wav)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Mel-spectrogram generation
# ---------------------------------------------------------------------------
def generate_mel_spectrogram(
    audio_path: str,
    sr: int = 22050,
) -> Optional[np.ndarray]:
    """
    Generate a mel-spectrogram image (H×W×3, uint8) from a WAV file.

    Returns None if librosa/matplotlib are unavailable or audio is too short.
    """
    if librosa is None or plt is None:
        warnings.warn(
            "librosa and matplotlib are required for audio processing. "
            "Install them with: pip install librosa matplotlib"
        )
        return None

    try:
        data, sample_rate = librosa.load(audio_path, sr=sr)
        if len(data) < 1024:
            return None

        mel = librosa.power_to_db(
            audio_feat.melspectrogram(y=data, sr=sample_rate),
            ref=np.min,
        )

        # Render to image via matplotlib (matches original LipFD)
        tmp_path = os.path.join(tempfile.gettempdir(), "_avlips_mel_tmp.png")
        plt.imsave(tmp_path, mel)
        mel_img = (plt.imread(tmp_path) * 255).astype(np.uint8)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        return mel_img[:, :, :3]  # drop alpha if present

    except Exception as e:
        warnings.warn(f"Mel-spectrogram generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(
    video_path: str,
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], float, int]:
    """
    Extract all (or up to *max_frames*) frames from a video.

    Returns
    -------
    frames : list[np.ndarray]
        BGR frames.
    fps : float
    total_frame_count : int
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames, fps, total


# ---------------------------------------------------------------------------
# Composite image builder
# ---------------------------------------------------------------------------
def build_composite_images(
    video_path: str,
    n_extract: int = N_EXTRACT,
    window_len: int = WINDOW_LEN,
    frame_size: int = FRAME_SIZE,
) -> List[np.ndarray]:
    """
    Build composite images (mel spectrogram + frame groups) from a video.

    Each composite image has shape ``(1000, 2500, 3)`` — the mel-spectrogram
    (500 × 2500) stacked on top of 5 horizontally-concatenated frames
    (500 × 2500).

    If audio extraction or mel-spectrogram generation fails, a blank
    (grey) spectrogram placeholder is used so that the visual pathway
    still functions.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    n_extract : int
        Number of frame groups to sample.
    window_len : int
        Number of consecutive frames per group.
    frame_size : int
        Resize dimension for individual frames.

    Returns
    -------
    composites : list[np.ndarray]
        Each element is an (H, W, 3) uint8 composite image.
    """
    frames, fps, total_count = extract_frames(video_path)

    if len(frames) < window_len + 1:
        warnings.warn(
            f"Video too short ({len(frames)} frames). "
            f"Need at least {window_len + 1}."
        )
        return []

    # --- Sample starting indices (evenly spaced) ---
    max_start = len(frames) - window_len
    if n_extract >= max_start:
        start_indices = list(range(max_start))
    else:
        start_indices = np.linspace(
            0, max_start - 1, n_extract, endpoint=True, dtype=int,
        ).tolist()

    # --- Resize frames to FRAME_SIZE × FRAME_SIZE ---
    # Convert BGR→RGB to match original preprocess.py which uses
    # cv2.cvtColor(frame, COLOR_BGR2RGBA)[:,:,:3] → RGB
    resized_frames = [
        cv2.resize(
            cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (frame_size, frame_size)
        )
        for f in frames
    ]

    # --- Extract audio & generate mel-spectrogram ---
    mel_img: Optional[np.ndarray] = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = os.path.join(tmp_dir, "audio.wav")
        if extract_audio(video_path, wav_path):
            mel_img = generate_mel_spectrogram(wav_path)

    composites: List[np.ndarray] = []

    for start in start_indices:
        group_frames = resized_frames[start: start + window_len]
        if len(group_frames) < window_len:
            continue

        # Concatenate frames horizontally: (500, 2500, 3)
        frame_row = np.concatenate(group_frames, axis=1)[:, :, :3]

        # Build mel-spectrogram slice for this time window
        if mel_img is not None:
            mapping = mel_img.shape[1] / len(frames)
            begin = int(np.round(start * mapping))
            end = int(np.round((start + window_len) * mapping))
            if end <= begin:
                end = begin + 1
            sub_mel = mel_img[:, begin:end, :3]
            sub_mel = cv2.resize(
                sub_mel, (frame_size * window_len, frame_size),
            )
        else:
            # Fallback: grey placeholder
            sub_mel = np.full(
                (frame_size, frame_size * window_len, 3), 128, dtype=np.uint8,
            )

        # Stack vertically: mel on top, frames below → (1000, 2500, 3)
        # Both mel (from matplotlib) and frames are now RGB — matching
        # the original preprocess.py which saves RGB PNGs via plt.imsave.
        composite_rgb = np.concatenate([sub_mel, frame_row], axis=0)

        # Simulate cv2.imread read-back: the original training pipeline
        # saves composites as RGB PNGs then reads them with cv2.imread
        # which returns BGR.  The model was trained on BGR data.
        composite_bgr = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
        composites.append(composite_bgr)

    return composites


# ---------------------------------------------------------------------------
# Tensor conversion & multi-scale crops
# ---------------------------------------------------------------------------
def composite_to_tensors(
    composite: np.ndarray,
    frame_size: int = FRAME_SIZE,
    window_len: int = WINDOW_LEN,
) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    """
    Convert a composite image to the tensors expected by LipFD.

    Parameters
    ----------
    composite : np.ndarray
        Shape (1000, 2500, 3), uint8, BGR.

    Returns
    -------
    full_img : Tensor
        Shape (3, 1120, 1120) — CLIP input (normalised).
    crops : list[list[Tensor]]
        ``crops[scale][frame]``, each (3, 224, 224).
        3 scales × 5 frames.
    """
    # IMPORTANT: The original LipFD training pipeline uses raw BGR pixel
    # values in [0, 255] with NO normalisation.  The Normalize() call in the
    # original data/datasets.py is a dead assignment — its result is
    # immediately overwritten.  We must replicate this exactly for the
    # pretrained weights to work correctly.
    tensor = torch.tensor(composite, dtype=torch.float32).permute(2, 0, 1)
    # tensor is now (3, H, W) in BGR order, values [0, 255]

    # --- Full image for CLIP encoder ---
    full_img = T.Resize((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE))(tensor)

    # --- Multi-scale crops from the frame row (bottom half) ---
    resize_224 = T.Resize((CROP_SIZE, CROP_SIZE))

    # crops[0]: full-frame crops (1.0×)
    # IMPORTANT: The original LipFD dataset code uses `range(5)` which gives
    # pixel offsets i=0,1,2,3,4 — NOT i*500.  This means 5 nearly-identical
    # crops from the first frame area with tiny 1-pixel shifts.  The model
    # was trained with this, so we must match it exactly.
    crops_full: List[torch.Tensor] = []
    for i in range(window_len):
        # Bottom half = frames: tensor[:, frame_size:, :]
        crop = tensor[:, frame_size:, i:i + frame_size]  # (3, 500, 500)
        crops_full.append(resize_224(crop))

    # crops[1]: 0.65× center crop
    crops_mid: List[torch.Tensor] = []
    for c in crops_full:
        y0, y1 = CROP_INDICES[0]
        crops_mid.append(resize_224(c[:, y0:y1, y0:y1]))

    # crops[2]: 0.45× center crop (lip region)
    crops_lip: List[torch.Tensor] = []
    for c in crops_full:
        y0, y1 = CROP_INDICES[1]
        crops_lip.append(resize_224(c[:, y0:y1, y0:y1]))

    crops = [crops_full, crops_mid, crops_lip]

    return full_img, crops


# ---------------------------------------------------------------------------
# High-level: video → model-ready batched tensors
# ---------------------------------------------------------------------------
def preprocess_video(
    video_path: str,
    n_extract: int = N_EXTRACT,
    window_len: int = WINDOW_LEN,
    max_composites: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[List[List[torch.Tensor]]]]:
    """
    End-to-end preprocessing: video file → batched tensors for LipFD.

    Returns
    -------
    full_imgs : Tensor | None
        Batch of full composite images, shape (N, 3, 1120, 1120).
    crops : list[list[Tensor]] | None
        ``crops[scale][frame]`` where each tensor is (N, 3, 224, 224).
        Returns None if the video is too short or cannot be processed.
    """
    composites = build_composite_images(
        video_path, n_extract=n_extract, window_len=window_len,
    )
    if not composites:
        return None, None

    if max_composites and len(composites) > max_composites:
        composites = composites[:max_composites]

    full_imgs_list: List[torch.Tensor] = []
    # crops_batched[scale][frame] will accumulate tensors across composites
    crops_batched: List[List[List[torch.Tensor]]] = [
        [[] for _ in range(window_len)] for _ in range(3)
    ]

    for comp in composites:
        full_img, crops = composite_to_tensors(comp, window_len=window_len)
        full_imgs_list.append(full_img)
        for s in range(3):
            for f in range(window_len):
                crops_batched[s][f].append(crops[s][f])

    # Stack into batch tensors
    full_imgs = torch.stack(full_imgs_list, dim=0)  # (N, 3, 1120, 1120)
    crops_tensors: List[List[torch.Tensor]] = [
        [torch.stack(crops_batched[s][f], dim=0) for f in range(window_len)]
        for s in range(3)
    ]

    return full_imgs, crops_tensors
