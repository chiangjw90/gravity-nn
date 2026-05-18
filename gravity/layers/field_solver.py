"""1D Multi-Scale Causal Field Solver — the scan primitive for all dimensions.

Solves φ_s(t) = α_s · φ_s(t-1) + β_s · ρ(t) for S scales simultaneously
using parallel associative scan (O(N log N) work, O(N) memory).

Binary operator: (α₁, b₁) ⊗ (α₂, b₂) = (α₁·α₂, α₂·b₁ + b₂)

After solving, applies running mean removal and causal gradient computation:
  - Mean removal: φ(t) ← φ(t) - (1/t) Σ_{i=1}^{t} φ(i)
  - Gradient: ∇φ(t) = φ(t) - φ(t-1)

Higher-dimensional solvers (2D/3D/4D) decompose into multiple 1D scans
along different directions, each calling this primitive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCFieldSolver(nn.Module):
    """Multi-scale causal field solver (1D primitive).

    Args:
        S: Number of scales (default 3).
        C: Number of density channels (default 1).
            For 1D language model: C=16 (multi-channel density).
            For 2D/3D/4D: C=1 (single-channel density, called per-direction).
        n_dirs: Number of scan directions sharing this solver.
            1 for standalone 1D; 4 for 2D; 6 for 3D; 5 for 4D.
            When n_dirs > 1, each direction has its own learnable
            (log_lambdas, log_betas) parameters.
    """

    def __init__(self, S: int = 3, C: int = 1, n_dirs: int = 1):
        super().__init__()
        self.S = S
        self.C = C
        self.n_dirs = n_dirs

        if n_dirs == 1:
            # Single-direction: parameters are [S]
            self.log_lambdas = nn.Parameter(torch.linspace(0.7, 4.9, S))
            self.log_betas = nn.Parameter(torch.linspace(-0.5, 0.5, S))
        else:
            # Multi-direction: parameters are [n_dirs, S]
            self.log_lambdas = nn.Parameter(
                torch.linspace(0.7, 4.9, S).unsqueeze(0).expand(n_dirs, -1).clone()
            )
            self.log_betas = nn.Parameter(
                torch.linspace(-0.5, 0.5, S).unsqueeze(0).expand(n_dirs, -1).clone()
            )

    def scan_1d(self, rho, dir_idx=None):
        """Run parallel associative scan on a 1D density sequence.

        Args:
            rho: [B, N] or [B, N, C] density values along scan axis.
            dir_idx: Direction index (required when n_dirs > 1).

        Returns:
            phi: [B, S, N] or [B, S, C, N] potential field (mean-removed).
            gp:  [B, S, N] or [B, S, C, N] field gradient.
        """
        multi_channel = rho.dim() == 3
        if multi_channel:
            B, N, C = rho.shape
        else:
            B, N = rho.shape
            C = 1

        dev = rho.device

        # Get direction-specific parameters
        if self.n_dirs == 1:
            log_l = self.log_lambdas
            log_b = self.log_betas
        else:
            assert dir_idx is not None, "dir_idx required for multi-direction solver"
            log_l = self.log_lambdas[dir_idx]
            log_b = self.log_betas[dir_idx]

        alphas = torch.exp(-1.0 / torch.exp(log_l).clamp(1, 1000))
        betas = torch.exp(log_b).clamp(0.01, 100)

        # Reshape for scan
        if multi_channel:
            rt = rho.permute(0, 2, 1)  # [B, C, N]
            b = rt.unsqueeze(1) * betas.view(1, self.S, 1, 1)  # [B, S, C, N]
            a = alphas.view(1, self.S, 1, 1).expand(B, self.S, C, N).clone()
        else:
            b = rho.unsqueeze(1) * betas.view(1, -1, 1)  # [B, S, N]
            a = alphas.view(1, self.S, 1).expand(B, self.S, N).clone()

        # Parallel associative scan: O(N log N)
        stride = 1
        while stride < N:
            b = a * F.pad(b, (stride, 0))[..., :N] + b
            a = a * F.pad(a, (stride, 0), value=1.0)[..., :N]
            stride *= 2

        # Mean removal (causal running mean)
        phi = b
        cs = phi.cumsum(-1)
        cnt = torch.arange(1, N + 1, device=dev, dtype=phi.dtype)
        phi = (phi - cs / cnt).clamp(-1000, 1000)

        # Causal gradient: ∇φ(t) = φ(t) - φ(t-1)
        gp = torch.zeros_like(phi)
        gp[..., 0] = phi[..., 0]
        gp[..., 1:] = phi[..., 1:] - phi[..., :-1]

        return phi, gp

    def forward(self, rho):
        """Forward pass for single-direction solver.

        Args:
            rho: [B, N, C] density field (multi-channel) or [B, N] (single).
        Returns:
            phi: Potential field (mean-removed).
            gp:  Field gradient.
        """
        return self.scan_1d(rho, dir_idx=None if self.n_dirs == 1 else 0)

    def get_effective_ranges(self):
        """Return effective dependency ranges (3 × λ) for each scale."""
        with torch.no_grad():
            if self.n_dirs == 1:
                lambdas = torch.exp(self.log_lambdas).clamp(1, 1000)
            else:
                lambdas = torch.exp(self.log_lambdas[0]).clamp(1, 1000)
            return (3 * lambdas).cpu().tolist()
