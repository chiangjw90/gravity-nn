"""Gravity Blocks — composed processing units for each dimensionality.

Pipeline: norm → coin_proj → density → field_solve → attention →
          features → feat_proj (residual) → FFN (residual)

Each block variant (1D/2D/3D/4D) uses the appropriate field solver,
attention, and feature extractor for its dimensionality.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gravity.layers.field_solver import MCFieldSolver
from gravity.layers.field_solver_nd import FieldSolver2D, FieldSolver3D, FieldSolver4D
from gravity.layers.local_field_attention import (
    LocalFieldAttention1D, LocalFieldAttention2D,
    LocalFieldAttention3D, LocalFieldAttention4D,
)
from gravity.layers.physics_features import (
    PhysicsFeatures1D, PhysicsFeatures2D,
    PhysicsFeatures3D, PhysicsFeatures4D,
)


class GravityBlock1D(nn.Module):
    """1D Gravity block for sequence processing.

    Args:
        d: Model dimension.
        K: Coin parameter dimension (default 64).
        S: Number of field scales (default 3).
        C: Number of density channels (default 16).
        R: Window half-size (default 5).
        dropout: Dropout rate (default 0.1).
        use_position_feature: Include position/N in features.
    """

    def __init__(self, d: int, K: int = 64, S: int = 3, C: int = 16,
                 R: int = 5, dropout: float = 0.1,
                 use_position_feature: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.coin_proj = nn.Linear(d, K)

        self.solver = MCFieldSolver(S, C)
        self.attn = LocalFieldAttention1D(K, S, C, R)
        self.feat = PhysicsFeatures1D(K, S, C, R, use_position_feature)

        self.feat_proj = nn.Sequential(
            nn.Linear(self.feat.feat_dim, d * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 2, d), nn.Dropout(dropout)
        )
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 4, d), nn.Dropout(dropout)
        )

        nn.init.normal_(self.coin_proj.weight, 0, 0.02)
        nn.init.zeros_(self.coin_proj.bias)

    def forward(self, x):
        """x: [B, N, d] → [B, N, d]"""
        h = self.norm1(x)
        coins = math.pi * torch.tanh(self.coin_proj(h))
        rho = self.attn.compute_density(coins)
        phi, gp = self.solver(rho)
        at = self.attn(coins, phi, gp)
        ci = getattr(self.attn, '_ci', None)
        vm = getattr(self.attn, '_vm', None)
        f = self.feat(coins, at, rho, phi, gp, ci, vm)
        x = x + self.feat_proj(f)
        x = x + self.ffn(self.norm2(x))
        return x


class GravityBlock2D(nn.Module):
    """2D Gravity block for image processing.

    Args:
        d: Model dimension.
        K: Coin parameter dimension (default 15).
        S: Number of field scales (default 3).
        R: Window radius (default 1).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(self, d: int, K: int = 15, S: int = 3, R: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.coin_proj = nn.Linear(d, K)

        self.solver = FieldSolver2D(S)
        self.attn = LocalFieldAttention2D(K, S, R)
        win = (2 * R + 1) ** 2
        self.feats = PhysicsFeatures2D(K, S, 4, win)

        self.feat_proj = nn.Sequential(
            nn.Linear(self.feats.feat_dim, d), nn.GELU(), nn.Dropout(dropout)
        )
        self.gate = nn.Linear(d, d)
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 4, d), nn.Dropout(dropout)
        )

        nn.init.normal_(self.coin_proj.weight, 0, 0.02)
        nn.init.zeros_(self.coin_proj.bias)

    def forward(self, x):
        """x: [B, H, W, d] → [B, H, W, d]"""
        h = self.norm1(x)
        coins = math.pi * torch.tanh(self.coin_proj(h))
        rho = self.attn.compute_density(coins)
        phi, gp = self.solver(rho)
        at, nf = self.attn(coins, phi)
        f = self.feats(coins, at, nf, rho, phi, gp)
        po = self.feat_proj(f)
        x = x + torch.sigmoid(self.gate(h)) * po
        return x + self.ffn(self.norm2(x))


class GravityBlock3D(nn.Module):
    """3D Gravity block for volume processing.

    Args:
        d: Model dimension.
        K: Coin parameter dimension (default 8).
        S: Number of field scales (default 3).
        R: Window radius (default 1).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(self, d: int, K: int = 8, S: int = 3, R: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.coin_proj = nn.Linear(d, K)

        self.solver = FieldSolver3D(S)
        self.attn = LocalFieldAttention3D(K, S, R)
        win = (2 * R + 1) ** 3
        self.feats = PhysicsFeatures3D(K, S, 6, win)

        self.feat_proj = nn.Sequential(
            nn.Linear(self.feats.feat_dim, d), nn.GELU(), nn.Dropout(dropout)
        )
        self.gate = nn.Linear(d, d)
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 4, d), nn.Dropout(dropout)
        )

        nn.init.normal_(self.coin_proj.weight, 0, 0.02)
        nn.init.zeros_(self.coin_proj.bias)

    def forward(self, x):
        """x: [B, D, H, W, d] → [B, D, H, W, d]"""
        h = self.norm1(x)
        coins = math.pi * torch.tanh(self.coin_proj(h))
        rho = self.attn.compute_density(coins)
        phi, gp = self.solver(rho)
        at, nf = self.attn(coins, phi)
        f = self.feats(coins, at, nf, rho, phi, gp)
        po = self.feat_proj(f)
        x = x + torch.sigmoid(self.gate(h)) * po
        return x + self.ffn(self.norm2(x))


class GravityBlock4D(nn.Module):
    """4D Gravity block for video processing.

    Args:
        d: Model dimension.
        K: Coin parameter dimension (default 8).
        S: Number of field scales (default 3).
        R_space: Spatial window radius (default 1).
        R_time: Temporal window radius (default 1).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(self, d: int, K: int = 8, S: int = 3, R_space: int = 1,
                 R_time: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.field_proj = nn.Linear(d, K)

        self.solver = FieldSolver4D(S)
        self.attn = LocalFieldAttention4D(K, S, R_space, R_time)
        win = (2 * R_space + 1) ** 2 * (2 * R_time + 1)
        self.feats = PhysicsFeatures4D(K, S, 5, win)

        self.feat_proj = nn.Sequential(
            nn.Linear(self.feats.feat_dim, d), nn.GELU(), nn.Dropout(dropout)
        )
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 4, d), nn.Dropout(dropout)
        )

        nn.init.normal_(self.field_proj.weight, 0, 0.02)
        nn.init.zeros_(self.field_proj.bias)

    def forward(self, x):
        """x: [B, T, H, W, d] → [B, T, H, W, d]"""
        h = self.norm1(x)
        params = math.pi * torch.tanh(self.field_proj(h))
        rho = self.attn.compute_density(params)
        phi, gp = self.solver(rho)
        at, nf = self.attn(params, phi)
        f = self.feats(params, at, nf, rho, phi, gp)
        x = x + self.feat_proj(f)
        x = x + self.ffn(self.norm2(x))
        return x
