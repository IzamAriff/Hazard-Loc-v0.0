"""
SegFormer Model for testing with classification wrapper
Outputs 2D logits [B, C] after global average pooling
Uses SegFormerCrossEntropyLoss
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .wrapper import ClassificationWrapper # Use the single, correct wrapper


class SegFormerModel(nn.Module):
    """
    SegFormer transformer-based model wrapped for classification
    Outputs 2D logits [B, C] (via wrapper)
    Paired with: SegFormerCrossEntropyLoss
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # FIX: Instantiate the correct Segformer model, not a Unet with a mit_b0 encoder.
        segformer = smp.Segformer(
            encoder_name="mit_b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        self.model = ClassificationWrapper(segformer)
        print(f"✓ SegFormerModel initialized with {num_classes} classes (wrapped)")

    def forward(self, x):
        return self.model(x)