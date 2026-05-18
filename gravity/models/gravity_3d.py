"""Gravity 3D — volume classification and 3D diffusion generation.

Classification: patch_embed + pos_embed → L × GravityBlock3D → norm → pool → head
Diffusion:      patchify → embed + time_cond → L × GravityDiffusionBlock → unpatchify

The diffusion variant uses AdaLN-Zero conditioning, SwiGLU FFN, and RMSNorm
instead of the standard block's LayerNorm + GELU FFN.

Reference: Paper 1, Table 11 (OrganMNIST3D: 86.4%), Section 4.12 (3D diffusion).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gravity.layers.gravity_block import GravityBlock3D
from gravity.layers.field_solver_nd import FieldSolver3D
from gravity.layers.local_field_attention import LocalFieldAttention3D
from gravity.layers.physics_features import PhysicsFeatures3D


# ─── Classification Model ───────────────────────────────────────────


class Gravity3D(nn.Module):
    """Gravity 3D volume classifier.

    Args:
        num_classes: Number of output classes (default 11).
        in_channels: Input volume channels (default 1).
        d_model: Model dimension (default 64).
        K: Coin parameter dimension (default 8).
        S: Number of field scales (default 3).
        R: Window radius (default 1).
        n_layers: Number of Gravity blocks (default 2).
        vol_size: Input volume size per dim (default 28).
        patch_size: 3D patch size per dim (default 4).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(self, num_classes: int = 11, in_channels: int = 1,
                 d_model: int = 64, K: int = 8, S: int = 3, R: int = 1,
                 n_layers: int = 2, vol_size: int = 28, patch_size: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.ps = patch_size
        self.gd = vol_size // patch_size

        self.patch_embed = nn.Linear(in_channels * patch_size ** 3, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.gd, self.gd, self.gd, d_model) * 0.02
        )
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GravityBlock3D(d_model, K, S, R, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, vols):
        """
        Args:
            vols: [B, C, D, H, W] input volumes.
        Returns:
            logits: [B, num_classes].
        """
        B = vols.shape[0]
        P = self.ps
        G = self.gd

        # Squeeze channel dim if single-channel, then extract 3D patches
        x = vols.squeeze(1) if vols.shape[1] == 1 else vols[:, 0]
        x = x.unfold(1, P, P).unfold(2, P, P).unfold(3, P, P)
        x = x.reshape(B, G, G, G, P * P * P)

        x = self.drop(self.patch_embed(x) + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        return self.head(self.norm(x).mean(dim=(1, 2, 3)))


# ─── Diffusion Components ───────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (faster than LayerNorm)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(d * mult * 2 / 3)
        self.w1 = nn.Linear(d, hidden)
        self.w2 = nn.Linear(d, hidden)
        self.w3 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TimestepEmbed(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""

    def __init__(self, d: int):
        super().__init__()
        half = d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d))

    def forward(self, t):
        args = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], -1))


class GravityDiffusionBlock3D(nn.Module):
    """3D Gravity block with AdaLN-Zero conditioning for diffusion.

    Uses RMSNorm + SwiGLU FFN instead of LayerNorm + GELU FFN.

    Args:
        d: Model dimension.
        K: Coin parameter dimension (default 8).
        S: Number of field scales (default 3).
        R: Window radius (default 1).
    """

    def __init__(self, d: int, K: int = 8, S: int = 3, R: int = 1):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.field_proj = nn.Linear(d, K)

        self.solver = FieldSolver3D(S)
        self.attn = LocalFieldAttention3D(K, S, R)
        win = (2 * R + 1) ** 3
        self.feats = PhysicsFeatures3D(K, S, 6, win)
        self.feat_proj = nn.Sequential(nn.Linear(self.feats.feat_dim, d), nn.SiLU())
        self.ffn = SwiGLU(d)

        # AdaLN-Zero: 6 modulation parameters (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.normal_(self.field_proj.weight, 0, 0.02)
        nn.init.zeros_(self.field_proj.bias)

    def forward(self, x, cond):
        """
        Args:
            x:    [B, D, H, W, d] input.
            cond: [B, d] conditioning (timestep embedding).
        Returns:
            x: [B, D, H, W, d] output.
        """
        B, D, H, W, d = x.shape
        g1, b1, a1, g2, b2, a2 = self.adaLN(cond).chunk(6, dim=-1)
        g1 = g1[:, None, None, None, :]
        b1 = b1[:, None, None, None, :]
        a1 = a1[:, None, None, None, :]
        g2 = g2[:, None, None, None, :]
        b2 = b2[:, None, None, None, :]
        a2 = a2[:, None, None, None, :]

        # Field propagation sublayer
        h = self.norm1(x) * (1 + g1) + b1
        params = math.pi * torch.tanh(self.field_proj(h))
        rho = self.attn.compute_density(params)
        phi, gp = self.solver(rho)
        at, nf = self.attn(params, phi)
        f = self.feats(params, at, nf, rho, phi, gp)
        x = x + a1 * self.feat_proj(f)

        # FFN sublayer
        h = self.norm2(x) * (1 + g2) + b2
        x = x + a2 * self.ffn(h)
        return x


# ─── Diffusion Model ────────────────────────────────────────────────


class Gravity3DDiffusion(nn.Module):
    """Gravity 3D Diffusion model for volumetric generation.

    Noise prediction network for DDPM. Uses patchify/unpatchify for
    input/output conversion and AdaLN-Zero conditioned blocks.

    Args:
        channels: Input volume channels (default 1).
        d_model: Model dimension (default 128).
        K: Coin parameter dimension (default 8).
        S: Number of field scales (default 3).
        R: Window radius (default 1).
        n_layers: Number of diffusion blocks (default 4).
        patch_size: 3D patch size per dim (default 4).
        vol_size: Input volume size per dim (default 28).
    """

    def __init__(self, channels: int = 1, d_model: int = 128, K: int = 8,
                 S: int = 3, R: int = 1, n_layers: int = 4,
                 patch_size: int = 4, vol_size: int = 28):
        super().__init__()
        self.ps = patch_size
        self.channels = channels
        self.G = vol_size // patch_size
        patch_dim = channels * patch_size ** 3

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.G, self.G, self.G, d_model) * 0.02
        )
        self.time_embed = TimestepEmbed(d_model)

        self.blocks = nn.ModuleList([
            GravityDiffusionBlock3D(d_model, K, S, R)
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.output_proj = nn.Linear(d_model, patch_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def patchify(self, vols):
        """[B, C, D, H, W] → [B, G, G, G, patch_dim]"""
        B = vols.shape[0]
        P = self.ps
        G = self.G
        x = vols.squeeze(1) if vols.shape[1] == 1 else vols[:, 0]
        x = x.unfold(1, P, P).unfold(2, P, P).unfold(3, P, P)
        return x.reshape(B, G, G, G, P * P * P)

    def unpatchify(self, x):
        """[B, G, G, G, patch_dim] → [B, C, D, H, W]"""
        B = x.shape[0]
        P = self.ps
        G = self.G
        C = self.channels
        x = x.reshape(B, G, G, G, C, P, P, P)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, C, G * P, G * P, G * P)
        return x

    def forward(self, noisy_vols, timesteps):
        """
        Args:
            noisy_vols: [B, C, D, H, W] noisy input volumes.
            timesteps:  [B] diffusion timestep indices.
        Returns:
            pred_noise: [B, C, D, H, W] predicted noise.
        """
        x = self.patchify(noisy_vols)
        x = self.patch_embed(x) + self.pos_embed
        cond = self.time_embed(timesteps)

        for block in self.blocks:
            x = block(x, cond)

        g, b = self.final_adaLN(cond).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + g[:, None, None, None, :]) + b[:, None, None, None, :]

        return self.unpatchify(self.output_proj(x))


class DDPMSchedule:
    """DDPM noise schedule (cosine or linear).

    Args:
        T: Number of diffusion timesteps (default 1000).
        schedule: 'cosine' or 'linear' (default 'cosine').
    """

    def __init__(self, T: int = 1000, schedule: str = 'cosine'):
        self.T = T
        if schedule == 'cosine':
            steps = torch.linspace(0, T, T + 1)
            alpha_bars = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            self.betas = betas.clamp(max=0.999)
        else:
            self.betas = torch.linspace(1e-4, 0.02, T)

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        prev_ab = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1.0 - prev_ab) / (1.0 - self.alpha_bars)

    def to(self, device):
        for attr in ['betas', 'alphas', 'alpha_bars', 'sqrt_alpha_bars',
                      'sqrt_one_minus_alpha_bars', 'sqrt_recip_alphas',
                      'posterior_variance']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        s_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        return s_ab * x0 + s_1mab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_idx):
        """Reverse diffusion: denoise one step."""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)
        pred_noise = model(x_t, t_tensor)
        mean = self.sqrt_recip_alphas[t_idx] * (
            x_t - self.betas[t_idx] / self.sqrt_one_minus_alpha_bars[t_idx] * pred_noise
        )
        if t_idx > 0:
            return mean + torch.sqrt(self.posterior_variance[t_idx]) * torch.randn_like(x_t)
        return mean

    @torch.no_grad()
    def sample(self, model, shape, device='cpu'):
        """Generate samples by running full reverse diffusion."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)
