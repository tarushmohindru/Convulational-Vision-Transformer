from __future__ import annotations

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """0.5 * BCE + 0.5 * Dice, computed per-class then averaged."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def _dice(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(2)   # (B, C, H*W)
        tgt = targets.flatten(2)
        inter = (probs * tgt).sum(-1)
        union = probs.sum(-1) + tgt.sum(-1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + (1 - self.bce_weight) * self._dice(logits, targets)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    preds = (torch.sigmoid(logits) > threshold).float().flatten(2)
    tgt = targets.flatten(2)
    inter = (preds * tgt).sum(-1)
    union = preds.sum(-1) + tgt.sum(-1)
    return ((2 * inter + smooth) / (union + smooth)).mean().item()
