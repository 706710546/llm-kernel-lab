"""Triton implementation of elementwise vector addition."""

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    """Each Triton program processes one contiguous block of elements."""
    program_id = tl.program_id(axis=0)
    offsets = program_id * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def vector_add_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    """Add two contiguous CUDA tensors using a one-dimensional Triton grid."""
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} != {y.shape}")
    if x.device != y.device:
        raise ValueError(f"device mismatch: {x.device} != {y.device}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} != {y.dtype}")
    if not x.is_cuda:
        raise ValueError("Triton implementation requires CUDA tensors")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("this first kernel intentionally supports contiguous tensors only")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("this first kernel supports float16 and float32 only")

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output
    grid = (triton.cdiv(n_elements, block_size),)
    _vector_add_kernel[grid](
        x,
        y,
        output,
        n_elements=n_elements,
        block_size=block_size,
    )
    return output
