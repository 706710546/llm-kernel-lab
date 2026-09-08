"""PyTorch reference implementation for vector addition."""

import torch


def vector_add_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the elementwise sum used as the correctness reference."""
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} != {y.shape}")
    if x.device != y.device:
        raise ValueError(f"device mismatch: {x.device} != {y.device}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} != {y.dtype}")
    return x + y

