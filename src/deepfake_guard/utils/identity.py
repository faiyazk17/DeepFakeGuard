import warnings
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

class IdentityMatcher:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        try:
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        except ImportError:
            warnings.warn("facenet-pytorch not installed — identity verification disabled.")
        except Exception as e:
            warnings.warn(f"Could not load identity model: {e}")

    def get_embedding(self, image):
        if self.model is None:
            return None
        
        try:
            # Resize to 160x160 as expected by InceptionResnetV1
            if isinstance(image, Image.Image):
                img = image.resize((160, 160))
            else:
                # Assume it might be a tensor or numpy array, but for now let's stick to PIL
                return None

            # Convert to tensor and normalize
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            img_tensor = trans(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(img_tensor)
            return embedding
        except Exception as e:
            warnings.warn(f"Error computing face embedding: {e}")
            return None

    def compute_similarity(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        return F.cosine_similarity(emb1, emb2).item()
