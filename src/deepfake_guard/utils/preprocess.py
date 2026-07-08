from __future__ import annotations

import torch
from torchvision import transforms
import random
from PIL import Image, ImageFilter

class RandomCompression:
    def __init__(self, quality_range=(40, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(*self.quality_range)
            # Save to buffer and reload
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=q)
            buffer.seek(0)
            return Image.open(buffer)
        return img

class GaussianBlur:
    def __init__(self, kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5):
        self.p = p
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

# Robust training pipeline with augmentations
def build_training_preprocess(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)), # Ensure base size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomCompression(quality_range=(50, 90), p=0.3),
        GaussianBlur(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

# Minimal preprocess function; tuned later if DINOv3 provides a specific pipeline
def build_preprocess(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# Backwards-compatible aliases (in case older code uses different names)
buildpreprocess = build_preprocess


def stack_frames(frames, preprocess_fn, device: str):
    tensor_list = [preprocess_fn(f) for f in frames]
    return torch.stack(tensor_list).to(device)


stackframes = stack_frames

def simulate_compression(frames, quality=90):
    from io import BytesIO
    processed_frames = []
    for frame in frames:
        if frame is None:
            continue
        buffer = BytesIO()
        frame.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        processed_frames.append(compressed)
    return processed_frames
