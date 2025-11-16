"""
SegFormer Specific Loss Function: CrossEntropy Loss
Independent file - does not interfere with src/utils/losses.py
Per dissertation: SegFormer transformer-based model uses CrossEntropy per literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerCrossEntropyLoss(nn.Module):
    """
    CrossEntropy Loss for SegFormer (wrapped to 2D classification)
    Standard loss for transformer-based semantic segmentation
    """
    def __init__(self, weight=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        print(f"✓ SegFormerCrossEntropyLoss initialized (label_smoothing={label_smoothing})")

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] classification logits (after wrapper)
            targets: [B] class labels
        """
        if inputs.dim() != 2:
            raise ValueError(f"SegFormerCrossEntropyLoss expects 2D input [B, C], got {inputs.shape}")

        # Standard PyTorch CrossEntropy
        loss = F.cross_entropy(
            inputs, 
            targets,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )

        return loss