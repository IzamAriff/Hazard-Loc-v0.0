"""
Ensemble Model for testing
Combines multiple 2D outputs via averaging
Uses EnsembleFocalLoss
"""

import torch
import torch.nn as nn


class ClassificationWrapper(nn.Module):
    """Wraps segmentation models to output 2D classification logits"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class EnsembleModel(nn.Module):
    """
    Ensemble that averages outputs from multiple models
    All members are wrapped to 2D classification
    Outputs 2D logits [B, C]
    Paired with: EnsembleFocalLoss
    """
    def __init__(self, models_list, num_classes=2):
        super().__init__()
        # FIX: Accept a list of pre-initialized models to prevent instance sharing
        if not isinstance(models_list, list) or len(models_list) < 2:
            raise ValueError("EnsembleModel requires a list of at least two model instances.")
        
        self.models = nn.ModuleList(models_list)
        print(f"✓ EnsembleModel initialized with {len(models_list)} members")

    def forward(self, x):
        # Get outputs from each member
        outputs = [model(x) for model in self.models]

        # Average the logits
        stacked = torch.stack(outputs)
        avg_output = torch.mean(stacked, dim=0)

        return avg_output