"""Gravity 4D — video classification with spatiotemporal field propagation.

Architecture: patchify + embed + pos → L × GravityBlock4D → norm → pool → head

5-direction scan decomposition:
  - 4 spatial scans: bidirectional on H and W
  - 1 temporal scan: causal (forward only — no future leakage)

Reference: Paper 1, Table 12 (MovingMNIST: 88.1% with 120K params).
"""

import torch
import torch.nn as nn

from gravity.layers.gravity_block import GravityBlock4D


class Gravity4D(nn.Module):
    """Gravity 4D video classifier.

    Args:
        num_classes: Number of output classes (default 10).
        in_channels: Input frame channels (default 1 for grayscale).
        d_model: Model dimension (default 64).
        K: Coin parameter dimension (default 8).
        S: Number of field scales (default 3).
        R_space: Spatial window radius (default 1).
        R_time: Temporal window radius (default 1).
        n_layers: Number of Gravity blocks (default 2).
        patch_size: Spatial patch size (default 4).
        n_frames: Number of input frames (default 10).
        frame_size: Frame spatial size (default 32).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1,
                 d_model: int = 64, K: int = 8, S: int = 3,
                 R_space: int = 1, R_time: int = 1,
                 n_layers: int = 2, patch_size: int = 4,
                 n_frames: int = 10, frame_size: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.ps = patch_size
        self.in_channels = in_channels
        self.gh = frame_size // patch_size
        self.gw = frame_size // patch_size
        self.n_frames = n_frames

        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_frames, self.gh, self.gw, d_model) * 0.02
        )

        self.blocks = nn.ModuleList([
            GravityBlock4D(d_model, K, S, R_space, R_time, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def patchify(self, videos):
        """Convert video frames to spatial patches.

        Args:
            videos: [B, T, H, W] for single-channel or [B, T, C, H, W].
        Returns:
            patches: [B, T, gh, gw, patch_dim].
        """
        P = self.ps
        if videos.dim() == 4:
            # [B, T, H, W] → single channel
            B, T, H, W = videos.shape
            x = videos.unfold(2, P, P).unfold(3, P, P)  # [B, T, gh, gw, P, P]
            return x.reshape(B, T, self.gh, self.gw, P * P)
        else:
            # [B, T, C, H, W] → multi-channel
            B, T, C, H, W = videos.shape
            x = videos.unfold(3, P, P).unfold(4, P, P)  # [B, T, C, gh, gw, P, P]
            x = x.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, gh, gw, C, P, P]
            return x.reshape(B, T, self.gh, self.gw, C * P * P)

    def forward(self, videos):
        """
        Args:
            videos: [B, T, H, W] or [B, T, C, H, W].
        Returns:
            logits: [B, num_classes].
        """
        x = self.patchify(videos)
        x = self.patch_embed(x) + self.pos_embed

        for block in self.blocks:
            x = block(x)

        # Global average pool over T, H, W
        x = self.norm(x.mean(dim=(1, 2, 3)))
        return self.head(x)
