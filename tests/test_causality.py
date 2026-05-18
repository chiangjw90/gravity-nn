"""Causality tests for 1D Gravity model.

Verifies that future tokens do not affect past outputs,
ensuring the causal property required for autoregressive generation.
"""

import torch
import pytest


def test_causal_masking():
    """Changing future tokens should not affect past outputs."""
    from gravity import GravityLM

    model = GravityLM(vocab_size=256, max_seq_len=64, d_model=32,
                      n_layers=2, K=8, S=3, C=4, R=3)
    model.eval()

    seq_a = torch.randint(0, 256, (1, 32))
    seq_b = seq_a.clone()
    seq_b[0, 16:] = torch.randint(0, 256, (16,))  # Change future tokens

    with torch.no_grad():
        out_a = model(seq_a)
        out_b = model(seq_b)

    # Outputs at positions 0–15 should be identical
    diff = (out_a[0, :16] - out_b[0, :16]).abs().max().item()
    assert diff < 1e-5, f"Causal violation: max diff at past positions = {diff}"


def test_causal_different_lengths():
    """Output at position t should not depend on sequence length beyond t."""
    from gravity import GravityLM

    model = GravityLM(vocab_size=256, max_seq_len=64, d_model=32,
                      n_layers=2, K=8, S=3, C=4, R=3)
    model.eval()

    tokens = torch.randint(0, 256, (1, 32))

    with torch.no_grad():
        out_short = model(tokens[:, :16])
        out_long = model(tokens)

    # Output at position 15 should be very close
    diff = (out_short[0, 15] - out_long[0, 15]).abs().max().item()
    assert diff < 1e-4, f"Length-dependent output: max diff = {diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
