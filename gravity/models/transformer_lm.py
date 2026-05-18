"""Transformer Language Model — baseline for comparison.

Standard Transformer with pre-norm, GELU activation, and weight tying.
Included for fair benchmarking against GravityLM.

Reference: Paper 1, Tables 2 and 5–8.
"""

import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    """Standard Transformer language model baseline.

    Args:
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
        d_model: Model dimension (default 768).
        n_heads: Number of attention heads (default 12).
        n_layers: Number of layers (default 12).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 768,
                 n_heads: int = 12, n_layers: int = 12, dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # Weight tying

        nn.init.normal_(self.tok.weight, 0, 0.02)
        nn.init.normal_(self.pos.weight, 0, 0.01)

    def forward(self, ids, mask=None):
        """
        Args:
            ids: [B, N] token indices.
            mask: Optional attention mask [N, N].
        Returns:
            logits: [B, N, vocab_size].
        """
        B, N = ids.shape
        dev = ids.device
        h = self.drop(self.tok(ids) + self.pos(torch.arange(N, device=dev).unsqueeze(0)))
        if mask is None:
            m = torch.triu(torch.full((N, N), float('-inf'), device=dev), diagonal=1)
            try:
                return self.head(self.norm(self.enc(h, mask=m, is_causal=True)))
            except TypeError:
                # PyTorch < 2.0 doesn't support is_causal
                return self.head(self.norm(self.enc(h, mask=m)))
        else:
            try:
                return self.head(self.norm(self.enc(h, mask=mask, is_causal=False)))
            except TypeError:
                return self.head(self.norm(self.enc(h, mask=mask)))


def make_transformer(vocab_size: int, max_seq_len: int, **kwargs) -> TransformerLM:
    """Create a Transformer language model."""
    return TransformerLM(vocab_size, max_seq_len, **kwargs)
