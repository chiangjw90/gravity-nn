"""Model metrics and profiling utilities."""

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch module.
        trainable_only: If True, count only parameters with requires_grad.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_memory_mb(model: nn.Module, input_shape: tuple,
                      dtype: torch.dtype = torch.float32) -> float:
    """Estimate peak memory usage for a forward pass (MB).

    This is a rough estimate based on parameter + activation memory.

    Args:
        model: PyTorch module.
        input_shape: Input tensor shape (without batch dim).
        dtype: Data type (default float32).

    Returns:
        Estimated memory in MB.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Rough activation estimate: 2× parameter memory
    return (param_bytes * 3) / (1024 ** 2)
