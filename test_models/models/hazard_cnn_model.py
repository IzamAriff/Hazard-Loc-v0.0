"""
HazardCNN (ResNet-18) Model for testing
Outputs 2D classification logits [B, C]
Uses HazardCNNFocalLoss
"""

import torch.nn as nn
from torchvision import models

class HazardCNNModel(nn.Module):
    """
    ResNet-18 baseline for hazard detection
    Outputs 2D logits [B, C]
    Paired with: HazardCNNFocalLoss
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        print(f"✓ HazardCNNModel (ResNet-18) initialized with {num_classes} classes (2D output)")

    def forward(self, x):
        return self.backbone(x)