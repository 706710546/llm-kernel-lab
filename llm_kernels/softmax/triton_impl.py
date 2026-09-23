"""Softmax 的 Triton V0：一个 Program 处理一整行。"""

import torch
import triton
import triton.language as tl


MAX_BLOCK_SIZE = 65_536


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    n_columns,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_columns

    input_offsets = row * n_columns + offsets
    values = tl.load(
        input_ptr + input_offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    row_max = tl.max(values, axis=0)
    numerators = tl.exp(values - row_max)
    denominator = tl.sum(numerators, axis=0)
    probabilities = numerators / denominator

    tl.store(output_ptr + input_offsets, probabilities, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """使用一个 Triton Program 计算输入矩阵的一行 Softmax。"""
    if x.ndim != 2:
        raise ValueError("x 必须是二维 Tensor")
    if not x.is_cuda:
        raise ValueError("x 必须位于 CUDA 设备上")
    if not x.is_contiguous():
        raise ValueError("Softmax V0 暂时只支持连续 Tensor")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("Softmax V0 暂时只支持 float16 和 float32")

    n_rows, n_columns = x.shape
    output = torch.empty_like(x)
    if x.numel() == 0:
        return output

    block_size = triton.next_power_of_2(n_columns)
    if block_size > MAX_BLOCK_SIZE:
        raise ValueError(
            f"Softmax V0 每行最多支持 {MAX_BLOCK_SIZE:,} 个元素，"
            f"实际为 {n_columns:,}；更宽的行需要分块或在线算法"
        )

    _softmax_kernel[(n_rows,)](
        x,
        output,
        n_columns,
        BLOCK_SIZE=block_size,
    )
    return output
