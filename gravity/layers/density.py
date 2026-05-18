"""Density Bottleneck — the core architectural innovation of Gravity.

Compresses K-dimensional field parameters to scalar (or C-channel) density
via a learnable nonlinear projection: ρ = softplus(W) · params².

This bottleneck-before-propagation is the source of empirical strength
(Paper 1, Section 5.1: two qualitatively different field solvers achieve
identical PPL, identifying the density bottleneck as the key mechanism).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityBottleneck(nn.Module):
    """Compress K-dim field parameters to C-channel scalar density.

    Two modes:
    - Multi-channel (C > 1): ρ = softplus(W · params²), W is [K, C] linear.
      Used by 1D language model at 100M+ scale (K=64, C=16).
    - Single-channel (C = 1): ρ = params² @ softplus(w), w is [K] vector.
      Used by 2D/3D/4D proven code for lightweight density.

    Args:
        K: Input parameter dimension (coin dimension).
        C: Number of output density channels.
            C > 1  → nn.Linear projection (multi-channel mode).
            C == 1 → weighted sum with learnable weights (single-channel mode).
    """

    def __init__(self, K: int, C: int = 1):
        super().__init__()
        self.K = K
        self.C = C
        if C > 1:
            # Multi-channel: linear projection K → C
            self.proj = nn.Linear(K, C)
        else:
            # Single-channel: learnable weight vector
            self.weights = nn.Parameter(torch.ones(K) / K)

    def forward(self, params):
        """
        Args:
            params: [..., K] field parameters (any leading dims).
        Returns:
            density: [..., C] if C > 1, or [...] if C == 1.
        """
        squared = params ** 2
        if self.C > 1:
            return F.softplus(self.proj(squared))
        else:
            return squared @ F.softplus(self.weights)
