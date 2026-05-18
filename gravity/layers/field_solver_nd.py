"""Multi-dimensional Field Solvers — directional scan decomposition.

Decomposes N-dimensional lattices into 1D parallel scans:
  - 2D (images):  4 directions — right, left, down, up
  - 3D (volumes): 6 directions — ±x, ±y, ±z
  - 4D (video):   5 directions — 4 spatial (bidirectional) + 1 temporal (causal)

Each direction runs the 1D MCFieldSolver primitive independently with its own
learnable (α, β) parameters, then stacks the results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gravity.layers.field_solver import MCFieldSolver


class FieldSolver2D(nn.Module):
    """4-direction field propagation on 2D grid (images).

    Directions: right (→), left (←), down (↓), up (↑).

    Args:
        S: Number of scales (default 3).
    """

    def __init__(self, S: int = 3):
        super().__init__()
        self.S = S
        self.n_dirs = 4
        self.solver = MCFieldSolver(S, C=1, n_dirs=4)

    def forward(self, rho_2d):
        """
        Args:
            rho_2d: [B, H, W] density on 2D grid.
        Returns:
            phi: [B, 4, S, H, W] potential per direction.
            gp:  [B, 4, S, H, W] gradient per direction.
        """
        B, H, W = rho_2d.shape
        S = self.S
        all_phi = []
        all_gp = []

        # → rightward: scan along W for each row
        rr = rho_2d.reshape(B * H, W)
        p, g = self.solver.scan_1d(rr, 0)
        all_phi.append(p.reshape(B, H, S, W).permute(0, 2, 1, 3))
        all_gp.append(g.reshape(B, H, S, W).permute(0, 2, 1, 3))

        # ← leftward: flip, scan, flip back
        p, g = self.solver.scan_1d(rr.flip(-1), 1)
        all_phi.append(p.flip(-1).reshape(B, H, S, W).permute(0, 2, 1, 3))
        all_gp.append(g.flip(-1).reshape(B, H, S, W).permute(0, 2, 1, 3))

        # ↓ downward: scan along H for each column
        rc = rho_2d.permute(0, 2, 1).reshape(B * W, H)
        p, g = self.solver.scan_1d(rc, 2)
        all_phi.append(p.reshape(B, W, S, H).permute(0, 2, 3, 1))
        all_gp.append(g.reshape(B, W, S, H).permute(0, 2, 3, 1))

        # ↑ upward
        p, g = self.solver.scan_1d(rc.flip(-1), 3)
        all_phi.append(p.flip(-1).reshape(B, W, S, H).permute(0, 2, 3, 1))
        all_gp.append(g.flip(-1).reshape(B, W, S, H).permute(0, 2, 3, 1))

        return torch.stack(all_phi, dim=1), torch.stack(all_gp, dim=1)


class FieldSolver3D(nn.Module):
    """6-direction field propagation on 3D grid (volumes).

    Directions: +x, -x, +y, -y, +z, -z.

    Args:
        S: Number of scales (default 3).
    """

    def __init__(self, S: int = 3):
        super().__init__()
        self.S = S
        self.n_dirs = 6
        self.solver = MCFieldSolver(S, C=1, n_dirs=6)

    def forward(self, rho_3d):
        """
        Args:
            rho_3d: [B, D, H, W] density on 3D grid.
        Returns:
            phi: [B, 6, S, D, H, W] potential per direction.
            gp:  [B, 6, S, D, H, W] gradient per direction.
        """
        B, D, H, W = rho_3d.shape
        S = self.S
        phis = []
        gps = []

        # ±x (along D axis)
        flat = rho_3d.permute(0, 2, 3, 1).reshape(B * H * W, D)
        for di, flip in [(0, False), (1, True)]:
            inp = flat.flip(-1) if flip else flat
            p, g = self.solver.scan_1d(inp, di)
            if flip:
                p, g = p.flip(-1), g.flip(-1)
            phis.append(p.reshape(B, H, W, S, D).permute(0, 3, 4, 1, 2))
            gps.append(g.reshape(B, H, W, S, D).permute(0, 3, 4, 1, 2))

        # ±y (along H axis)
        flat = rho_3d.permute(0, 1, 3, 2).reshape(B * D * W, H)
        for di, flip in [(2, False), (3, True)]:
            inp = flat.flip(-1) if flip else flat
            p, g = self.solver.scan_1d(inp, di)
            if flip:
                p, g = p.flip(-1), g.flip(-1)
            phis.append(p.reshape(B, D, W, S, H).permute(0, 3, 1, 4, 2))
            gps.append(g.reshape(B, D, W, S, H).permute(0, 3, 1, 4, 2))

        # ±z (along W axis)
        flat = rho_3d.reshape(B * D * H, W)
        for di, flip in [(4, False), (5, True)]:
            inp = flat.flip(-1) if flip else flat
            p, g = self.solver.scan_1d(inp, di)
            if flip:
                p, g = p.flip(-1), g.flip(-1)
            phis.append(p.reshape(B, D, H, S, W).permute(0, 3, 1, 2, 4))
            gps.append(g.reshape(B, D, H, S, W).permute(0, 3, 1, 2, 4))

        return torch.stack(phis, dim=1), torch.stack(gps, dim=1)


class FieldSolver4D(nn.Module):
    """5-direction field propagation for video (4D).

    Spatial: 4 bidirectional scans (right, left, down, up).
    Temporal: 1 causal scan (forward only — no future leakage).

    Args:
        S: Number of scales (default 3).
    """

    def __init__(self, S: int = 3):
        super().__init__()
        self.S = S
        self.n_dirs = 5
        self.solver = MCFieldSolver(S, C=1, n_dirs=5)

    def forward(self, rho_4d):
        """
        Args:
            rho_4d: [B, T, H, W] density on 4D grid.
        Returns:
            phi: [B, 5, S, T, H, W] potential per direction.
            gp:  [B, 5, S, T, H, W] gradient per direction.
        """
        B, T, H, W = rho_4d.shape
        S = self.S
        phis = []
        gps = []

        # Spatial: right (di=0)
        flat = rho_4d.reshape(B * T * H, W)
        p, g = self.solver.scan_1d(flat, 0)
        phis.append(p.reshape(B, T, H, S, W).permute(0, 3, 1, 2, 4))
        gps.append(g.reshape(B, T, H, S, W).permute(0, 3, 1, 2, 4))

        # Spatial: left (di=1)
        p, g = self.solver.scan_1d(flat.flip(-1), 1)
        phis.append(p.flip(-1).reshape(B, T, H, S, W).permute(0, 3, 1, 2, 4))
        gps.append(g.flip(-1).reshape(B, T, H, S, W).permute(0, 3, 1, 2, 4))

        # Spatial: down (di=2)
        flat = rho_4d.permute(0, 1, 3, 2).reshape(B * T * W, H)
        p, g = self.solver.scan_1d(flat, 2)
        phis.append(p.reshape(B, T, W, S, H).permute(0, 3, 1, 4, 2))
        gps.append(g.reshape(B, T, W, S, H).permute(0, 3, 1, 4, 2))

        # Spatial: up (di=3)
        p, g = self.solver.scan_1d(flat.flip(-1), 3)
        phis.append(p.flip(-1).reshape(B, T, W, S, H).permute(0, 3, 1, 4, 2))
        gps.append(g.flip(-1).reshape(B, T, W, S, H).permute(0, 3, 1, 4, 2))

        # Temporal: forward causal only (di=4) — NO backward scan
        flat = rho_4d.permute(0, 2, 3, 1).reshape(B * H * W, T)
        p, g = self.solver.scan_1d(flat, 4)
        phis.append(p.reshape(B, H, W, S, T).permute(0, 3, 4, 1, 2))
        gps.append(g.reshape(B, H, W, S, T).permute(0, 3, 4, 1, 2))

        return torch.stack(phis, dim=1), torch.stack(gps, dim=1)
