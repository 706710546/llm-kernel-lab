"""逐元素 Vector Add 的 Triton 实现。"""

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
    """每个 Triton Program 处理一段连续元素。"""
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
    """使用一维 Triton Grid 将两个连续 CUDA Tensor 相加。"""
    if x.shape != y.shape:
        raise ValueError(f"形状不一致：{x.shape} != {y.shape}")
    if x.device != y.device:
        raise ValueError(f"设备不一致：{x.device} != {y.device}")
    if x.dtype != y.dtype:
        raise ValueError(f"数据类型不一致：{x.dtype} != {y.dtype}")
    if not x.is_cuda:
        raise ValueError("Triton 实现要求输入位于 CUDA 设备上")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("第一个 Kernel 暂时只支持内存连续的 Tensor")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("第一个 Kernel 暂时只支持 float16 和 float32")

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
