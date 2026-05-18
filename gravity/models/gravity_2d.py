"""Gravity 2D — image classification with 4-direction field propagation.

Architecture: patch_embed + pos_embed → L × GravityBlock2D → norm → pool → head

Patch embedding converts image into a 2D grid of tokens. Each block runs
4 directional scans (right, left, down, up) via the density→field→features pipeline.

Reference: Paper 1, Table 10 (CIFAR-10: 82.3% with 653K params).
"""

import torch
import torch.nn as nn

from gravity.layers.gravity_block import GravityBlock2D


class Gravity2D(nn.Module):
    """Gravity 2D image classifier.

    Args:
        num_classes: Number of output classes (default 10).
        in_channels: Input image channels (default 3).
        d_model: Model dimension (default 128).
        K: Coin parameter dimension (default 15).
        S: Number of field scales (default 3).
        R: Window radius (default 1).
        n_layers: Number of Gravity blocks (default 4).
        patch_size: Patch size (default 4).
        img_size: Input image size (default 32).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 d_model: int = 128, K: int = 15, S: int = 3, R: int = 1,
                 n_layers: int = 4, patch_size: int = 4, img_size: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.ps = patch_size
        self.gh = img_size // patch_size
        self.gw = img_size // patch_size

        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.gh, self.gw, d_model) * 0.02
        )
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GravityBlock2D(d_model, K, S, R, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, imgs):
        """
        Args:
            imgs: [B, C, H, W] input images.
        Returns:
            logits: [B, num_classes].
        """
        B = imgs.shape[0]
        P = self.ps

        # Extract patches: [B, C, H, W] → [B, gh, gw, C*P*P]
        x = imgs.unfold(2, P, P).unfold(3, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, self.gh, self.gw, -1)

        x = self.drop(self.patch_embed(x) + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        # Global average pooling over spatial dims
        return self.head(self.norm(x).mean(dim=(1, 2)))
