"""Physics Feature Extraction — composite feature vectors from field quantities.

Assembles features from:
  - Attention patterns
  - Coin parameters and neighbor-aggregated coins
  - Field potentials and gradients
  - Attention entropy
  - Density statistics (1D only: running mean, z-score, ratio)

The 1D variant produces a 227-dim composite vector (at default K=64, S=3, C=16, R=5).
The 2D/3D/4D variants produce dimension-appropriate feature vectors.
"""

import torch
import torch.nn as nn


class PhysicsFeatures1D(nn.Module):
    """1D physics feature extraction for language modeling.

    Produces a composite feature vector including attention patterns,
    coin parameters, windowed coins, field potentials/gradients,
    entropy, curvature, and density statistics.

    Args:
        K: Coin parameter dimension.
        S: Number of field scales.
        C: Number of density channels.
        R: Window half-size.
        use_position_feature: Whether to include position/N feature.
            False (default) for variable-seq curriculum training.
            True for fixed-seq training. See paper for details.
    """

    def __init__(self, K: int = 64, S: int = 3, C: int = 16, R: int = 5,
                 use_position_feature: bool = False):
        super().__init__()
        self.K = K
        self.S = S
        self.C = C
        self.W = 2 * R + 1
        self.use_position_feature = use_position_feature

        pos_dim = 1 if use_position_feature else 0
        self.feat_dim = (
            S * self.W +          # attention patterns
            K +                   # coins
            self.W * K +          # windowed coins
            S * C +               # phi
            S * C +               # gradient
            3 * S +               # entropy + mean_pos + curvature
            pos_dim +             # optional position/N
            1 + 3                 # density stats (raw + running_mean + z + ratio)
        )

    def forward(self, c, at, rho, phi, gp, ci=None, vm=None):
        """
        Args:
            c:   [B, N, K] coin parameters.
            at:  [B, S, N, W] attention weights.
            rho: [B, N, C] density.
            phi: [B, S, C, N] field potential.
            gp:  [B, S, C, N] field gradient.
            ci:  [N, W] window column indices (from attention).
            vm:  [N, W] validity mask.
        Returns:
            features: [B, N, feat_dim].
        """
        B, N, K = c.shape
        S, C, W = self.S, self.C, self.W
        dev = c.device
        f = []

        if ci is None:
            t = torch.arange(N, device=dev).unsqueeze(1)
            w = torch.arange(W, device=dev).unsqueeze(0)
            r = t - W + 1 + w
            ci = r.clamp(min=0)
            vm = (r >= 0).float()

        # 1. Attention patterns [B, N, S*W]
        f.append(at.permute(0, 2, 1, 3).reshape(B, N, S * W))

        # 2. Coin parameters [B, N, K]
        f.append(c)

        # 3. Windowed coins [B, N, W*K]
        cw = c[:, ci, :] * vm.unsqueeze(0).unsqueeze(-1)
        f.append(cw.reshape(B, N, W * K))

        # 4. Field potential [B, N, S*C]
        f.append(phi.permute(0, 3, 1, 2).reshape(B, N, S * C))

        # 5. Field gradient [B, N, S*C]
        f.append(gp.permute(0, 3, 1, 2).reshape(B, N, S * C))

        # 6. Attention entropy [B, N, S]
        ac = at.clamp(min=1e-15)
        f.append(-(ac * ac.log()).sum(-1).permute(0, 2, 1))

        # 7. Attention mean position [B, N, S]
        wp = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W)
        f.append((at * wp).sum(-1).permute(0, 2, 1) / W)

        # 8. Attention curvature [B, N, S]
        As = at[:, :, :, W - 1]
        Am1 = at[:, :, :, W - 2]
        Am2 = at[:, :, :, max(0, W - 3)]
        f.append((Am1 + Am2 - 2 * As).permute(0, 2, 1))

        # 9. Optional position feature
        if self.use_position_feature:
            pos = torch.arange(N, device=dev, dtype=torch.float32).view(1, N, 1).expand(B, N, 1) / N
            f.append(pos)

        # 10. Density statistics [B, N, 4]
        rm = rho.mean(-1)  # [B, N]
        f.append(rm.unsqueeze(-1))
        cs = rm.cumsum(-1)
        cnt = torch.arange(1, N + 1, device=dev, dtype=torch.float32)
        rmn = cs / cnt
        f.append(rmn.unsqueeze(-1))
        rv = ((rm ** 2).cumsum(-1) / cnt - rmn ** 2).clamp(min=1e-8)
        f.append(((rm - rmn) / rv.sqrt().clamp(min=1e-6)).clamp(-5, 5).unsqueeze(-1))
        f.append((rm / rmn.clamp(min=1e-6)).clamp(0, 10).unsqueeze(-1))

        return torch.cat(f, -1)


class PhysicsFeatures2D(nn.Module):
    """2D physics feature extraction for image processing.

    Args:
        K: Coin parameter dimension.
        S: Number of field scales.
        n_dirs: Number of scan directions (default 4).
        win_size: Local window size ((2R+1)², default 9).
    """

    def __init__(self, K: int, S: int = 3, n_dirs: int = 4, win_size: int = 9):
        super().__init__()
        self.feat_dim = K + K + win_size + n_dirs * S + n_dirs * S + 1 + 1

    def forward(self, coins, attn, nf, rho, phi, gp):
        """
        Args:
            coins: [B, H, W, K].
            attn:  [B, H, W, win].
            nf:    [B, H, W, K] neighbor features.
            rho:   [B, H, W] density.
            phi:   [B, 4, S, H, W] potential.
            gp:    [B, 4, S, H, W] gradient.
        Returns:
            features: [B, H, W, feat_dim].
        """
        B, H, W, K = coins.shape
        f = []
        f.append(coins)
        f.append(nf)
        f.append(attn)
        f.append(phi.permute(0, 3, 4, 1, 2).reshape(B, H, W, -1))
        f.append(gp.permute(0, 3, 4, 1, 2).reshape(B, H, W, -1))
        f.append(rho.unsqueeze(-1))
        ac = attn.clamp(min=1e-15)
        f.append(-(ac * ac.log()).sum(-1, keepdim=True))
        return torch.cat(f, dim=-1)


class PhysicsFeatures3D(nn.Module):
    """3D physics feature extraction for volume processing.

    Args:
        K: Coin parameter dimension.
        S: Number of field scales.
        n_dirs: Number of scan directions (default 6).
        win_size: Local window size ((2R+1)³, default 27).
    """

    def __init__(self, K: int, S: int = 3, n_dirs: int = 6, win_size: int = 27):
        super().__init__()
        self.feat_dim = K + K + win_size + n_dirs * S + n_dirs * S + 1 + 1

    def forward(self, coins, attn, nf, rho, phi, gp):
        """
        Args:
            coins: [B, D, H, W, K].
            attn:  [B, D, H, W, win].
            nf:    [B, D, H, W, K].
            rho:   [B, D, H, W].
            phi:   [B, 6, S, D, H, W].
            gp:    [B, 6, S, D, H, W].
        Returns:
            features: [B, D, H, W, feat_dim].
        """
        B, D, H, W, K = coins.shape
        f = []
        f.append(coins)
        f.append(nf)
        f.append(attn)
        f.append(phi.permute(0, 3, 4, 5, 1, 2).reshape(B, D, H, W, -1))
        f.append(gp.permute(0, 3, 4, 5, 1, 2).reshape(B, D, H, W, -1))
        f.append(rho.unsqueeze(-1))
        ac = attn.clamp(min=1e-15)
        f.append(-(ac * ac.log()).sum(-1, keepdim=True))
        return torch.cat(f, -1)


class PhysicsFeatures4D(nn.Module):
    """4D physics feature extraction for video processing.

    Args:
        K: Coin parameter dimension.
        S: Number of field scales.
        n_dirs: Number of scan directions (default 5).
        win_size: Local window size (default 27).
    """

    def __init__(self, K: int, S: int = 3, n_dirs: int = 5, win_size: int = 27):
        super().__init__()
        self.feat_dim = K + K + win_size + n_dirs * S + n_dirs * S + 1 + 1

    def forward(self, coins, attn, nf, rho, phi, gp):
        """
        Args:
            coins: [B, T, H, W, K].
            attn:  [B, T, H, W, win].
            nf:    [B, T, H, W, K].
            rho:   [B, T, H, W].
            phi:   [B, 5, S, T, H, W].
            gp:    [B, 5, S, T, H, W].
        Returns:
            features: [B, T, H, W, feat_dim].
        """
        B, T, H, W, K = coins.shape
        f = []
        f.append(coins)
        f.append(nf)
        f.append(attn)
        f.append(phi.permute(0, 3, 4, 5, 1, 2).reshape(B, T, H, W, -1))
        f.append(gp.permute(0, 3, 4, 5, 1, 2).reshape(B, T, H, W, -1))
        f.append(rho.unsqueeze(-1))
        ac = attn.clamp(min=1e-15)
        f.append(-(ac * ac.log()).sum(-1, keepdim=True))
        return torch.cat(f, -1)
