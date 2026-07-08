import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

class FaceCropper:
    def __init__(
        self,
        device='cuda',
        margin_px=0,
        padding_ratio=0.25,
        min_face_size=50,
        confidence_threshold=0.95,
        vertical_shift_ratio=0.1,
    ):
        self.device = device
        self.margin_px = int(margin_px)
        self.padding_ratio = float(padding_ratio)
        self.min_face_size = int(min_face_size)
        self.confidence_threshold = float(confidence_threshold)
        self.vertical_shift_ratio = float(vertical_shift_ratio)
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=device, margin=0)
    
    @staticmethod
    def _square_and_clip_box(box, width, height, margin_px, padding_ratio, vertical_shift_ratio):
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = x2 - x1
        bh = y2 - y1
        
        pad = max(float(margin_px), max(bw, bh) * float(padding_ratio))
        
        # Center of the face box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Shift centre upward to better capture lower-face artifacts
        cy -= (bh * vertical_shift_ratio)
        
        side = max(bw, bh) + 2 * pad
        
        # Square box
        x1_sq = max(0, int(round(cx - side / 2)))
        y1_sq = max(0, int(round(cy - side / 2)))
        x2_sq = min(width, int(round(cx + side / 2)))
        y2_sq = min(height, int(round(cy + side / 2)))
        
        # Adjust if out of bounds (keeps it square)
        if x2_sq - x1_sq != y2_sq - y1_sq:
            side = min(x2_sq - x1_sq, y2_sq - y1_sq)
            # Re-center based on available space
            if x1_sq == 0: 
                x2_sq = x1_sq + side
                y2_sq = y1_sq + side
            elif y1_sq == 0:
                y2_sq = y1_sq + side
                x2_sq = x1_sq + side
            elif x2_sq == width:
                x1_sq = x2_sq - side
                y1_sq = y2_sq - side
            else: # Hit bottom
                y1_sq = y2_sq - side
                x1_sq = x2_sq - side
                
        return (x1_sq, y1_sq, x2_sq, y2_sq)

    def crop(self, image_path_or_pil, return_metadata=True):
        metadata = {}
        
        try:
            if isinstance(image_path_or_pil, str):
                img = Image.open(image_path_or_pil).convert('RGB')
            else:
                img = image_path_or_pil.convert('RGB')
            
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None:
                return (None, {'status': 'no_face'}) if return_metadata else None
            
            box = boxes[0]
            confidence = float(probs[0]) if probs is not None else 0.0
            
            if confidence < self.confidence_threshold:
                return (None, {'status': f'low_confidence ({confidence:.3f})'}) if return_metadata else None
            
            x1, y1, x2, y2 = box
            face_size = min(x2 - x1, y2 - y1)
            
            if face_size < self.min_face_size:
                return (None, {'status': 'face_too_small'}) if return_metadata else None

            w, h = img.size
            crop_box = self._square_and_clip_box(
                box,
                width=w,
                height=h,
                margin_px=self.margin_px,
                padding_ratio=self.padding_ratio,
                vertical_shift_ratio=self.vertical_shift_ratio
            )
            crop = img.crop(crop_box)
            
            metadata = {
                'status': 'success',
                'confidence': confidence,
                'face_size': face_size,
                'crop_box': crop_box,
                'cropped_size': crop.size
            }
            
            return (crop, metadata) if return_metadata else crop
        
        except Exception as e:
            return (None, {'status': f'error: {str(e)}'}) if return_metadata else None
