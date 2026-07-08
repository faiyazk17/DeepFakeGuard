from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class HeadOutput:
    logits_labels: Optional[torch.Tensor] = None
    l2_embeddings: Optional[torch.Tensor] = None

class LinearProbe(nn.Module):
    """Best-practice linear classification head for SSL representations."""
    
    def __init__(self, input_dim, num_classes, normalize_inputs=False, detach_classifier_inputs=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_inputs = normalize_inputs
        self.detach_classifier_inputs = detach_classifier_inputs

    def forward(self, x: torch.Tensor, **kwargs) -> HeadOutput:
        l2_embeddings = F.normalize(x, p=2, dim=1)
        
        if self.normalize_inputs:
            x = l2_embeddings
        
        logits = self.linear(x if not self.detach_classifier_inputs else x.detach())
        
        return HeadOutput(logits_labels=logits, l2_embeddings=l2_embeddings)