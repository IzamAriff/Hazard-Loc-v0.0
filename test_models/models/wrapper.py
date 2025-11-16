"""
Centralized Wrapper for Model Standardization
Provides a single, authoritative definition for ClassificationWrapper.
"""

import torch
import torch.nn as nn

class ClassificationWrapper(nn.Module):
    """
    Wraps segmentation models to output 2D classification logits [B, C].
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)  # [B, C, H, W]
        x = self.pool(x)   # [B, C, 1, 1]
        return torch.flatten(x, 1)  # [B, C]