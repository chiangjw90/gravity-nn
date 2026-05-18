"""
Gravity: A Physics-Inspired O(N) Neural Network Architecture

Replace Transformer's O(N²) self-attention with O(N) field-equation processing.
672 bytes streaming state per layer (fp32), independent of context length.
Same density→field→features mechanism across sequences, images, volumes, and video.

Quick start:
    from gravity import GravityLM
    model = GravityLM(vocab_size=256, max_seq_len=1024)
    logits = model(token_ids)  # [B, N, vocab_size]

Multi-dimensional:
    from gravity import Gravity2D, Gravity3D, Gravity4D
    img_model = Gravity2D(num_classes=10)    # 4-direction scan
    vol_model = Gravity3D(num_classes=11)    # 6-direction scan
    vid_model = Gravity4D(num_classes=10)    # 5-direction scan

Building blocks:
    from gravity.layers import GravityBlock1D, DensityBottleneck, MCFieldSolver

Reference: Chiang (2026), "Gravity: A Physics-Inspired O(N) Framework
with O(1) Streaming State Across Dimensions"

Patent pending. Apache 2.0 for academic/research use.
Commercial licensing: chiangjw90@gmail.com
"""

# Models
from gravity.models.gravity_lm import GravityLM, make_gravity
from gravity.models.gravity_2d import Gravity2D
from gravity.models.gravity_3d import Gravity3D, Gravity3DDiffusion
from gravity.models.gravity_4d import Gravity4D
from gravity.models.transformer_lm import TransformerLM, make_transformer

# Backward compatibility aliases
from gravity.layers.gravity_block import GravityBlock1D as PoissonBlock
from gravity.layers.field_solver import MCFieldSolver
from gravity.layers.local_field_attention import LocalFieldAttention1D as MCAttention
from gravity.layers.physics_features import PhysicsFeatures1D as MCFeatures

__version__ = "0.1.0"
__author__ = "Chia-Wei Chiang"

__all__ = [
    # Models
    "GravityLM",
    "Gravity2D",
    "Gravity3D",
    "Gravity3DDiffusion",
    "Gravity4D",
    "TransformerLM",
    # Factories
    "make_gravity",
    "make_transformer",
    # Backward compat aliases
    "PoissonBlock",
    "MCFieldSolver",
    "MCAttention",
    "MCFeatures",
]
