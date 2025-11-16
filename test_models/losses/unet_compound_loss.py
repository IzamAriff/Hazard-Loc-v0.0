"""
U-Net Specific Loss Function: Compound Loss (Focal + Dice)
Independent file - does not interfere with src/utils/losses.py
Per dissertation: U-Net optimized for dense masks requires Focal-Dice balance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetFocalLoss(nn.Module):
    """
    Focal Loss component for U-Net (segmentation)
    Works with 2D flattened logits [B*H*W, C]
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B*H*W, C] flattened logits
            targets: [B*H*W] flattened class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UNetDiceLoss(nn.Module):
    """
    Dice Loss component for U-Net (segmentation)
    Works with 4D segmentation outputs [B, C, H, W]
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] segmentation logits
            targets: [B] class labels (expanded to spatial dims internally)
        """
        if inputs.dim() != 4:
            raise ValueError(f"UNetDiceLoss expects 4D input, got {inputs.shape}")

        B, C, H, W = inputs.shape
        if targets.dim() == 1:
            targets = targets.view(B, 1, 1).expand(B, H, W)

        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=C).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        cardinality = (inputs + targets_one_hot).sum(dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()


class UNetCompoundLoss(nn.Module):
    """
    Compound Loss = α * Focal Loss + β * Dice Loss
    For U-Net architecture (4D outputs)
    """
    def __init__(self, alpha_weight=1.0, beta_weight=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight
        self.focal_loss_fn = UNetFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = UNetDiceLoss()
        print(f"✓ UNetCompoundLoss initialized (Focal α={alpha_weight}, Dice β={beta_weight})")

    def forward(self, inputs, targets):
        if inputs.dim() != 4:
            raise ValueError(f"UNetCompoundLoss expects 4D input, got {inputs.shape}")

        B, C, H, W = inputs.shape
        loss_components = {}

        inputs_flat = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.view(B, 1, 1).expand(B, H, W).contiguous().view(-1)

        focal = self.focal_loss_fn(inputs_flat, targets_flat)
        loss_components['focal'] = focal.item()

        dice = self.dice_loss(inputs, targets)
        loss_components['dice'] = dice.item()

        total_loss = self.alpha_weight * focal + self.beta_weight * dice
        return total_loss, loss_components