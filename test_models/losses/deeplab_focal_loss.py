"""
DeepLabV3+ Specific Loss Function: Focal Loss
Independent file - does not interfere with src/utils/losses.py
Per dissertation: DeepLabV3+ for edge detection needs robustness against extreme class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabFocalLoss(nn.Module):
    """
    Focal Loss for DeepLabV3+ (wrapped to 2D classification)
    Robust against extreme class imbalance in edge detection
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Alpha can be a float for binary, or a list/tensor for multi-class.
        self.alpha = alpha if alpha is None else torch.tensor(alpha)
        print(f"✓ DeepLabFocalLoss initialized (α={'class-weighted' if alpha is not None else 'None'}, γ={gamma})")

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] classification logits (after wrapper)
            targets: [B] class labels
        """
        if inputs.dim() != 2:
            raise ValueError(f"DeepLabFocalLoss expects 2D input [B, C], got {inputs.shape}")

        # Calculate log probabilities and gather them for the correct class
        log_pt = F.log_softmax(inputs, dim=1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        # --- FIX: Apply alpha as a per-class weight ---
        if self.alpha is not None:
            if self.alpha.device != log_pt.device:
                self.alpha = self.alpha.to(log_pt.device)
            # Select the alpha weight for each sample's target class
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -(1 - pt) ** self.gamma * log_pt


        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss