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
        # --- FIX: Numerically stable implementation for multi-class Focal Loss ---
        # Get the log probabilities from the model's logits
        log_prob = F.log_softmax(inputs, dim=-1)
        
        # Gather the log probabilities of the ground truth classes
        log_pt = log_prob.gather(1, targets.view(-1, 1)).view(-1)
        
        # Convert log probabilities to probabilities
        pt = log_pt.exp()
        
        # The focal loss formula
        # The -log_pt is equivalent to the cross-entropy loss for that sample
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * log_pt
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()


class DiceLoss(nn.Module):
    """
    Dice Loss for optimizing IoU directly
    """
    def __init__(self, smooth=1.0, from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
    
    def forward(self, inputs, targets):
        # --- FIX: Use softmax for multi-class problems ---
        if self.from_logits:
            # Get probabilities from logits
            inputs = F.softmax(inputs, dim=1)
        
        # --- FIX: Correct Dice score for multi-class classification ---
        # Do not flatten all dimensions. We want to compare class predictions.
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        
        # Sum over the batch dimension (dim=0) for a per-class score
        intersection = (inputs * targets_one_hot).sum(dim=0)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        # Calculate the Dice score per class, then average
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice_per_class.mean()


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
        self.dice_loss = DiceLoss(from_logits=True) # Ensure DiceLoss knows it's getting logits
    
    def forward(self, inputs, targets):
        # Both loss functions now correctly handle logits and integer targets
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        total_loss = self.alpha_weight * focal + self.beta_weight * dice
        
        return total_loss, {'focal': focal.item(), 'dice': dice.item()}