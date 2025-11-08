"""
Custom loss functions for HazardLoc crack detection
Implements compound loss (Focal Loss + Dice Loss) as per dissertation methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: Lin et al. (2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for optimizing IoU directly
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CompoundLoss(nn.Module):
    """
    Compound Loss = α * Focal Loss + β * Dice Loss
    As specified in dissertation Section 3.4.3.1
    
    Default weights: α=1.0, β=0.5
    """
    def __init__(self, alpha_weight=1.0, beta_weight=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.alpha_weight = alpha_weight  # Weight for Focal Loss
        self.beta_weight = beta_weight    # Weight for Dice Loss
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        total_loss = self.alpha_weight * focal + self.beta_weight * dice
        
        return total_loss, {'focal': focal.item(), 'dice': dice.item()}
