"""
losses.py
---------
Dice + Focal combined loss (mirrors the original notebook's
"Total Loss = Dice loss + Focal loss") and an IoU (Jaccard) metric,
implemented with plain PyTorch - no segmentation_models dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class DiceLoss(nn.Module):
    def __init__(self, smooth=None):
        super().__init__()
        self.smooth = config.DICE_SMOOTH if smooth is None else smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_onehot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = config.FOCAL_GAMMA if gamma is None else gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CombinedLoss(nn.Module):
    """Total loss = Dice loss + Focal loss."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.focal(logits, targets)


def iou_score(logits, targets, smooth: float = 1.0) -> float:
    """Mean IoU (Jaccard) over the batch - used as the training metric."""
    preds = torch.argmax(logits, dim=1)
    preds_onehot = F.one_hot(preds, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

    intersection = (preds_onehot * targets_onehot).sum(dim=(0, 2, 3))
    union = preds_onehot.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
