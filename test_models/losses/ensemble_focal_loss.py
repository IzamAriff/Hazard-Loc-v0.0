"""
Ensemble Specific Loss Function: Focal Loss
Independent file - does not interfere with src/utils/losses.py
Per dissertation: Ensemble combines multiple 2D outputs via averaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleFocalLoss(nn.Module):
    """
    Focal Loss for Ensemble model (averaged 2D outputs)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        print(f"✓ EnsembleFocalLoss initialized (α={alpha}, γ={gamma})")

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] averaged ensemble logits
            targets: [B] class labels
        """
        if inputs.dim() != 2:
            raise ValueError(f"EnsembleFocalLoss expects 2D input [B, C], got {inputs.shape}")

        # --- STABILITY FIX ---
        # Calculate pt directly from logits to avoid instability from torch.exp(-ce_loss)
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        # Gather the probabilities for the correct class
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss # alpha is applied to the final loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss