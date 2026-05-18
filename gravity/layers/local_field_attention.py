"""Local Field Attention — field-physics-derived windowed interaction.

Computes interaction weights within a fixed-size local window using
physics-informed quantities (field potentials, gradients, density),
not query-key inner products. The window is always local, so the
attention cost is O(N) rather than O(N²).

Variants:
  - 1D: causal window (left-only padding), used for language modeling.
  - 2D: symmetric spatial window, used for image classification.
  - 3D: symmetric spatial window, used for volume classification.
  - 4D: asymmetric — temporal causal, spatial symmetric.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gravity.layers.density import DensityBottleneck


class LocalFieldAttention1D(nn.Module):
    """1D local field attention with causal masking.

    Args:
        K: Coin parameter dimension.
        S: Number of field scales.
        C: Number of density channels.
        R: Window half-size (window = 2R+1).
    """

    def __init__(self, K: int, S: int = 3, C: int = 16, R: int = 5):
        super().__init__()
        self.S = S
        self.C = C
        self.W = 2 * R + 1

        self.w_phi = nn.Parameter(torch.tensor(0.5))
        self.w_force = nn.Parameter(torch.tensor(0.5))
        self.w_dist = nn.Parameter(torch.tensor(0.5))
        self.w_coin = nn.Parameter(torch.tensor(0.5))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('temps', torch.ones(S))

        # Density projection: K → C channels
        # Named 'density_proj' for backward compatibility with old checkpoints
        self.density_proj = nn.Linear(K, C)

        self._ci = None
        self._vm = None

    def compute_density(self, c):
        """Compute density from coin parameters: ρ = softplus(W · c²)"""
        return F.softplus(self.density_proj(c ** 2))

    def forward(self, c, phi, gp):
        """
        Args:
            c:   [B, N, K] coin parameters.
            phi: [B, S, C, N] field potential.
            gp:  [B, S, C, N] field gradient.
        Returns:
            at:  [B, S, N, W] attention weights within causal window.
        """
        B, N, K = c.shape
        S, C, W = self.S, self.C, self.W
        dev = c.device

        wp, wf, wd, wc = [F.softplus(x) for x in
                          [self.w_phi, self.w_force, self.w_dist, self.w_coin]]
        s2 = torch.exp(self.log_sigma2).clamp(0.01, 100)

        # Build causal window indices
        ti = torch.arange(N, device=dev).unsqueeze(1)
        wi = torch.arange(W, device=dev).unsqueeze(0)
        raw = ti - W + 1 + wi
        ci = raw.clamp(min=0)
        vm = (raw >= 0).float()
        self._ci = ci
        self._vm = vm

        # Coin similarity
        csim = -wc * ((c.unsqueeze(2) - c[:, ci, :]) ** 2).sum(-1) / s2

        # Distance penalty
        dv = torch.arange(W - 1, -1, -1, device=dev, dtype=torch.float32)

        # Field potential difference
        pd = (phi.unsqueeze(-1) - phi[:, :, :, ci]).abs().sum(2)

        # Field gradient difference
        fd = (gp.unsqueeze(-1) - gp[:, :, :, ci]).abs().sum(2)

        # Normalization
        eps = 1 + pd.mean(dim=(1, 3), keepdim=True).clamp(min=0.1)

        # Combined score
        sc = (-wp * pd - wf * fd - wd * dv.view(1, 1, 1, W) / eps +
              csim.unsqueeze(1)).clamp(-20, 20)

        # Apply causal mask
        sc = sc + (vm.unsqueeze(0).unsqueeze(0) - 1) * 1e9

        # Softmax with temperature
        at = F.softmax(
            (sc / self.temps.view(1, S, 1, 1).detach()).clamp(-20, 20), dim=-1
        )

        if at.isnan().any():
            at = torch.where(at.isnan(), torch.ones_like(at) / W, at)

        return at


class LocalFieldAttention2D(nn.Module):
    """2D local field attention with symmetric spatial windows.

    Args:
        K: Coin parameter dimension.
        S: Number of field scales.
        R: Window radius (window = (2R+1)²).
    """

    def __init__(self, K: int, S: int = 3, R: int = 1):
        super().__init__()
        self.S = S
        self.R = R
        self.W = (2 * R + 1) ** 2

        self.raw_w_phi = nn.Parameter(torch.tensor(0.5))
        self.raw_w_coin = nn.Parameter(torch.tensor(0.5))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))

        self.density = DensityBottleneck(K, C=1)

    def compute_density(self, c):
        return self.density(c)

    def forward(self, coins, phi):
        """
        Args:
            coins: [B, H, W, K] coin parameters on 2D grid.
            phi:   [B, 4, S, H, W] field potential from 4-dir solver.
        Returns:
            attn: [B, H, W, win] attention weights.
            nf:   [B, H, W, K] neighbor-aggregated features.
        """
        B, H, Wg, K = coins.shape
        S = self.S
        R = self.R
        wp = F.softplus(self.raw_w_phi)
        wc = F.softplus(self.raw_w_coin)
        s2 = torch.exp(self.log_sigma2).clamp(0.01, 100)

        phi_agg = phi.mean(dim=1)  # [B, S, H, W] — average over 4 directions

        # Pad + unfold for 2D windows
        cp = F.pad(coins.permute(0, 3, 1, 2), (R, R, R, R))
        cp = cp.unfold(2, 2 * R + 1, 1).unfold(3, 2 * R + 1, 1)
        cw = cp.permute(0, 2, 3, 4, 5, 1).reshape(B, H, Wg, self.W, K)

        pp = F.pad(phi_agg, (R, R, R, R))
        pp = pp.unfold(2, 2 * R + 1, 1).unfold(3, 2 * R + 1, 1)
        pp = pp.reshape(B, S, H, Wg, self.W)

        csim = -wc * ((coins.unsqueeze(3) - cw) ** 2).sum(-1) / s2
        pdiff = -wp * (phi_agg.unsqueeze(-1) - pp).abs().mean(1)
        score = (csim + pdiff).clamp(-20, 20)

        attn = F.softmax(score, dim=-1)
        nf = (attn.unsqueeze(-1) * cw).sum(3)

        return attn, nf


class LocalFieldAttention3D(nn.Module):
    """3D local field attention with symmetric spatial windows.

    Args:
        K: Coin parameter dimension.
        S: Number of field scales.
        R: Window radius (window = (2R+1)³).
    """

    def __init__(self, K: int, S: int = 3, R: int = 1):
        super().__init__()
        self.S = S
        self.R = R
        self.W = (2 * R + 1) ** 3

        self.w_phi = nn.Parameter(torch.tensor(0.5))
        self.w_coin = nn.Parameter(torch.tensor(0.5))
        self.log_s2 = nn.Parameter(torch.tensor(0.0))

        self.density = DensityBottleneck(K, C=1)

    def compute_density(self, c):
        return self.density(c)

    def forward(self, coins, phi):
        """
        Args:
            coins: [B, D, H, W, K] coin parameters on 3D grid.
            phi:   [B, 6, S, D, H, W] field potential from 6-dir solver.
        Returns:
            attn: [B, D, H, W, win] attention weights.
            nf:   [B, D, H, W, K] neighbor-aggregated features.
        """
        B, D, H, W, K = coins.shape
        R = self.R
        wp = F.softplus(self.w_phi)
        wc = F.softplus(self.w_coin)
        s2 = torch.exp(self.log_s2).clamp(0.01, 100)

        phi_agg = phi.mean(1)  # [B, S, D, H, W]

        cp = F.pad(coins.permute(0, 4, 1, 2, 3), (R, R, R, R, R, R))
        cp = cp.unfold(2, 2 * R + 1, 1).unfold(3, 2 * R + 1, 1).unfold(4, 2 * R + 1, 1)
        cw = cp.permute(0, 2, 3, 4, 5, 6, 7, 1).reshape(B, D, H, W, self.W, K)

        pp = F.pad(phi_agg, (R, R, R, R, R, R))
        pp = pp.unfold(2, 2 * R + 1, 1).unfold(3, 2 * R + 1, 1).unfold(4, 2 * R + 1, 1)
        pp = pp.reshape(B, self.S, D, H, W, self.W)

        csim = -wc * ((coins.unsqueeze(4) - cw) ** 2).sum(-1) / s2
        pdiff = -wp * (phi_agg.unsqueeze(-1) - pp).abs().mean(1)
        attn = F.softmax((csim + pdiff).clamp(-20, 20), dim=-1)
        nf = (attn.unsqueeze(-1) * cw).sum(4)

        return attn, nf


class LocalFieldAttention4D(nn.Module):
    """4D local field attention — temporal causal, spatial symmetric.

    Args:
        K: Coin parameter dimension.
        S: Number of field scales.
        R_space: Spatial window radius.
        R_time: Temporal window radius (causal — past only).
    """

    def __init__(self, K: int, S: int = 3, R_space: int = 1, R_time: int = 1):
        super().__init__()
        self.S = S
        self.R_s = R_space
        self.R_t = R_time
        self.W = (2 * R_space + 1) ** 2 * (2 * R_time + 1)

        self.w_phi = nn.Parameter(torch.tensor(0.5))
        self.w_coin = nn.Parameter(torch.tensor(0.5))
        self.log_s2 = nn.Parameter(torch.tensor(0.0))

        self.density = DensityBottleneck(K, C=1)

    def compute_density(self, c):
        return self.density(c)

    def forward(self, coins, phi):
        """
        Args:
            coins: [B, T, H, W, K] coin parameters.
            phi:   [B, 5, S, T, H, W] field potential from 5-dir solver.
        Returns:
            attn: [B, T, H, W, win] attention weights.
            nf:   [B, T, H, W, K] neighbor-aggregated features.
        """
        B, T, H, W, K = coins.shape
        Rs = self.R_s
        Rt = self.R_t
        wp = F.softplus(self.w_phi)
        wc = F.softplus(self.w_coin)
        s2 = torch.exp(self.log_s2).clamp(0.01, 100)

        phi_agg = phi.mean(1)  # [B, S, T, H, W]

        # Pad: temporal causal (2*Rt past, 0 future), spatial symmetric
        # This ensures unfold produces exactly T outputs on temporal dim
        cp = F.pad(coins.permute(0, 4, 1, 2, 3), (Rs, Rs, Rs, Rs, 2 * Rt, 0))
        cp = cp.unfold(2, 2 * Rt + 1, 1).unfold(3, 2 * Rs + 1, 1).unfold(4, 2 * Rs + 1, 1)
        cw = cp.permute(0, 2, 3, 4, 5, 6, 7, 1).reshape(B, T, H, W, self.W, K)

        pp = F.pad(phi_agg, (Rs, Rs, Rs, Rs, 2 * Rt, 0))
        pp = pp.unfold(2, 2 * Rt + 1, 1).unfold(3, 2 * Rs + 1, 1).unfold(4, 2 * Rs + 1, 1)
        pp = pp.reshape(B, self.S, T, H, W, self.W)

        csim = -wc * ((coins.unsqueeze(4) - cw) ** 2).sum(-1) / s2
        pdiff = -wp * (phi_agg.unsqueeze(-1) - pp).abs().mean(1)
        attn = F.softmax((csim + pdiff).clamp(-20, 20), dim=-1)
        nf = (attn.unsqueeze(-1) * cw).sum(4)

        return attn, nf
