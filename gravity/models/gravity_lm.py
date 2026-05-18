"""Gravity Language Model — 1D causal sequence model.

Architecture: token_embed + pos_embed → L × GravityBlock1D → norm → head

This model uses multi-channel density (C=16 by default) and causal
windowed attention. Weight tying between token embedding and LM head.

Reference: Paper 1, Tables 2 and 5–8.
"""

import torch
import torch.nn as nn

from gravity.layers.gravity_block import GravityBlock1D


class GravityLM(nn.Module):
    """Gravity Language Model.

    Args:
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
        d_model: Model dimension (default 768).
        n_layers: Number of Gravity blocks (default 12).
        K: Coin parameter dimension (default 64).
        S: Number of field scales (default 3).
        C: Number of density channels (default 16).
        R: Window half-size (default 5).
        dropout: Dropout rate (default 0.1).
        use_position_feature: Include position/N in features.
            False (default) for variable-seq curriculum training.
            True for fixed-seq training.
    """

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 768,
                 n_layers: int = 12, K: int = 64, S: int = 3, C: int = 16,
                 R: int = 5, dropout: float = 0.1,
                 use_position_feature: bool = False):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GravityBlock1D(d_model, K, S, C, R, dropout, use_position_feature)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # Weight tying

        nn.init.normal_(self.tok.weight, 0, 0.02)
        nn.init.normal_(self.pos.weight, 0, 0.01)

    def forward(self, ids):
        """
        Args:
            ids: [B, N] token indices.
        Returns:
            logits: [B, N, vocab_size].
        """
        B, N = ids.shape
        p = torch.arange(N, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok(ids) + self.pos(p))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def get_effective_ranges(self):
        """Return learned dependency ranges for each layer and scale."""
        ranges = {}
        for i, block in enumerate(self.blocks):
            ranges[f'layer_{i}'] = block.solver.get_effective_ranges()
        return ranges


def make_gravity(vocab_size: int, max_seq_len: int, **kwargs) -> GravityLM:
    """Create a Gravity language model."""
    return GravityLM(vocab_size, max_seq_len, **kwargs)
