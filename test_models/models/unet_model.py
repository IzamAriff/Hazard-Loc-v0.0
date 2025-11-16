"""
U-Net Model for testing
Outputs 2D classification logits [B, C] after global average pooling.
Uses a 2D-compatible loss like FocalLoss.
"""
import torch.nn as nn
import segmentation_models_pytorch as smp

from .wrapper import ClassificationWrapper # Use the single, correct wrapper

class UNetModel(nn.Module):
    """
    U-Net with ResNet18 encoder
    Outputs 2D logits [B, C] (via wrapper)
    Paired with: FocalLoss (or other 2D loss)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        self.model = ClassificationWrapper(unet)
        print(f"✓ UNetModel initialized with {num_classes} classes (wrapped to 2D output)")

    def forward(self, x):
        return self.model(x)