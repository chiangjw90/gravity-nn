"""Gravity building blocks: density → field → attention → features → block."""

from gravity.layers.density import DensityBottleneck
from gravity.layers.field_solver import MCFieldSolver
from gravity.layers.field_solver_nd import FieldSolver2D, FieldSolver3D, FieldSolver4D
from gravity.layers.local_field_attention import (
    LocalFieldAttention1D, LocalFieldAttention2D,
    LocalFieldAttention3D, LocalFieldAttention4D,
)
from gravity.layers.physics_features import (
    PhysicsFeatures1D, PhysicsFeatures2D,
    PhysicsFeatures3D, PhysicsFeatures4D,
)
from gravity.layers.gravity_block import (
    GravityBlock1D, GravityBlock2D, GravityBlock3D, GravityBlock4D,
)

__all__ = [
    "DensityBottleneck",
    "MCFieldSolver",
    "FieldSolver2D", "FieldSolver3D", "FieldSolver4D",
    "LocalFieldAttention1D", "LocalFieldAttention2D",
    "LocalFieldAttention3D", "LocalFieldAttention4D",
    "PhysicsFeatures1D", "PhysicsFeatures2D",
    "PhysicsFeatures3D", "PhysicsFeatures4D",
    "GravityBlock1D", "GravityBlock2D", "GravityBlock3D", "GravityBlock4D",
]
