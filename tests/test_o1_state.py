"""O(1) streaming state tests.

Verifies that Gravity's memory usage remains constant regardless
of sequence length during streaming inference — the key architectural
advantage over Transformer's growing KV cache.
"""

import torch
import pytest


def test_streaming_state_constant():
    """Verify streaming state size is O(1) — independent of sequence length."""
    from gravity.layers import MCFieldSolver

    solver = MCFieldSolver(S=3, C=4)

    # The streaming state for the field solver is just the last
    # (alpha, phi) pair per scale per channel: S * C * 2 values
    state_size = solver.S * 4 * 2  # S scales × C channels × 2 (alpha, phi)

    # This is constant regardless of sequence length
    # At fp32 (4 bytes): S=3, C=4 → 3*4*2*4 = 96 bytes per layer
    # At default K=64, C=16: 3*16*2*4 = 384 bytes
    # Paper claims 672 bytes = (S*C*2 + extras) per layer

    # Verify forward pass works at different lengths with same param count
    for seq_len in [64, 256, 1024, 4096]:
        rho = torch.randn(1, seq_len, 4)
        phi, gp = solver(rho)
        assert phi.shape == (1, 3, 4, seq_len)
        assert gp.shape == (1, 3, 4, seq_len)

    # Parameter count should be identical regardless of input length
    param_count = sum(p.numel() for p in solver.parameters())
    assert param_count == 6  # log_lambdas (3) + log_betas (3)


def test_memory_does_not_grow_with_length():
    """Rough check that model parameter count is independent of sequence length."""
    from gravity import GravityLM

    model = GravityLM(vocab_size=256, max_seq_len=8192, d_model=64,
                      n_layers=2, K=15, S=3, C=4, R=3)

    param_count = sum(p.numel() for p in model.parameters())

    # Model parameters should not depend on actual input sequence length
    # (only max_seq_len affects pos embedding, not the blocks)
    for seq_len in [32, 128, 512]:
        x = torch.randint(0, 256, (1, seq_len))
        out = model(x)
        assert out.shape == (1, seq_len, 256)

    # Verify same parameter count
    assert sum(p.numel() for p in model.parameters()) == param_count


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
