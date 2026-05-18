"""Shape tests for all Gravity model variants.

Verifies that forward passes produce correct output shapes
for 1D (language), 2D (image), 3D (volume), and 4D (video) models.
"""

import torch
import pytest


def test_gravity_lm_shape():
    from gravity import GravityLM
    model = GravityLM(vocab_size=256, max_seq_len=128, d_model=64,
                      n_layers=2, K=15, S=3, C=4, R=3)
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert out.shape == (2, 32, 256), f"Expected (2, 32, 256), got {out.shape}"


def test_gravity_2d_shape():
    from gravity import Gravity2D
    model = Gravity2D(num_classes=10, in_channels=3, d_model=32,
                      K=8, S=3, R=1, n_layers=2, patch_size=4, img_size=32)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"


def test_gravity_3d_shape():
    from gravity import Gravity3D
    model = Gravity3D(num_classes=11, in_channels=1, d_model=32,
                      K=8, S=3, R=1, n_layers=1, vol_size=28, patch_size=4)
    x = torch.randn(2, 1, 28, 28, 28)
    out = model(x)
    assert out.shape == (2, 11), f"Expected (2, 11), got {out.shape}"


def test_gravity_3d_diffusion_shape():
    from gravity import Gravity3DDiffusion
    model = Gravity3DDiffusion(channels=1, d_model=32, K=8, S=3, R=1,
                                n_layers=1, patch_size=4, vol_size=28)
    x = torch.randn(2, 1, 28, 28, 28)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == (2, 1, 28, 28, 28), f"Expected (2, 1, 28, 28, 28), got {out.shape}"


def test_gravity_4d_shape():
    from gravity import Gravity4D
    model = Gravity4D(num_classes=10, in_channels=1, d_model=32,
                      K=8, S=3, R_space=1, R_time=1, n_layers=1,
                      patch_size=4, n_frames=5, frame_size=16)
    x = torch.randn(2, 5, 16, 16)
    out = model(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"


def test_transformer_lm_shape():
    from gravity import TransformerLM
    model = TransformerLM(vocab_size=256, max_seq_len=128, d_model=64,
                          n_heads=4, n_layers=2)
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert out.shape == (2, 32, 256), f"Expected (2, 32, 256), got {out.shape}"


def test_backward_compat_imports():
    """Verify backward-compatible import names still work."""
    from gravity import PoissonBlock, MCFieldSolver, MCAttention, MCFeatures
    assert PoissonBlock is not None
    assert MCFieldSolver is not None
    assert MCAttention is not None
    assert MCFeatures is not None


def test_layers_imports():
    """Verify layer-level imports work."""
    from gravity.layers import (
        DensityBottleneck, MCFieldSolver,
        GravityBlock1D, GravityBlock2D, GravityBlock3D, GravityBlock4D,
    )
    db = DensityBottleneck(K=15, C=4)
    x = torch.randn(2, 10, 15)
    out = db(x)
    assert out.shape == (2, 10, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
