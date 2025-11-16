"""
DeepLabV3+ Model for testing with classification wrapper
Outputs 2D logits [B, C] after global average pooling
Uses DeepLabFocalLoss
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .wrapper import ClassificationWrapper # Use the single, correct wrapper


class DeepLabV3PlusModel(nn.Module):
    """
    DeepLabV3+ wrapped for classification
    Outputs 2D logits [B, C] (via wrapper)
    Paired with: DeepLabFocalLoss
    """
    def __init__(self, num_classes=2):
        super().__init__()
        deeplab = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        self.model = ClassificationWrapper(deeplab)
        print(f"✓ DeepLabV3PlusModel initialized with {num_classes} classes (wrapped)")

    def forward(self, x):
        return self.model(x)