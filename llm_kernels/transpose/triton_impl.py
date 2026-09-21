"""Matrix Transpose 的 Triton V0 实现。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_columns,
    BLOCK_SIZE: tl.constexpr,
):
    block_row = tl.program_id(axis=0)
    block_column = tl.program_id(axis=1)

    row_offsets = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    column_offsets = block_column * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    input_offsets = (
        row_offsets[:, None] * n_columns + column_offsets[None, :]
    )
    input_mask = (
        (row_offsets[:, None] < n_rows)
        & (column_offsets[None, :] < n_columns)
    )
    tile = tl.load(input_ptr + input_offsets, mask=input_mask)

    output_offsets = (
        column_offsets[:, None] * n_rows + row_offsets[None, :]
    )
    output_mask = tl.trans(input_mask)
    tl.store(
        output_ptr + output_offsets,
        tl.trans(tile),
        mask=output_mask,
    )


def transpose_triton(
    x: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """使用二维 Tile 将连续矩阵 ``[M, N]`` 转置为 ``[N, M]``。"""
    if x.ndim != 2:
        raise ValueError("x 必须是二维 Tensor")
    if not x.is_cuda:
        raise ValueError("x 必须位于 CUDA 设备上")
    if not x.is_contiguous():
        raise ValueError("x 必须是连续 Tensor")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("当前只支持 float16 和 float32")
    if block_size not in (16, 32):
        raise ValueError("block_size 当前只支持 16 或 32")

    n_rows, n_columns = x.shape
    output = torch.empty((n_columns, n_rows), device=x.device, dtype=x.dtype)
    if x.numel() == 0:
        return output

    grid = (
        triton.cdiv(n_rows, block_size),
        triton.cdiv(n_columns, block_size),
    )
    _transpose_kernel[grid](
        x,
        output,
        n_rows,
        n_columns,
        BLOCK_SIZE=block_size,
    )
    return output
