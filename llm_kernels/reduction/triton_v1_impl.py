"""按行求和 Reduction 的 Triton V1：两阶段分块归约。"""

import torch
import triton
import triton.language as tl


# 每个 Stage 1 Program 读取一行中的多少个元素。
CHUNK_SIZE = 1_024
# Stage 2 仍由一个 Program 合并一行的局部和，因此局部和数量也有上限。
MAX_PARTIALS_PER_ROW = 65_536


@triton.jit
def _row_sum_stage1_kernel(
    x_ptr,
    partials_ptr,
    n_columns,
    partials_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    """计算一行中一个块的局部和，并写入 partials[row, block]。"""
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_columns

    values = tl.load(x_ptr + row * n_columns + offsets, mask=mask, other=0.0)
    partial_sum = tl.sum(values.to(tl.float32), axis=0)
    tl.store(partials_ptr + row * partials_per_row + block, partial_sum)


@triton.jit
def _row_sum_stage2_kernel(
    partials_ptr,
    output_ptr,
    partials_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    """将同一行的所有局部和再次归约为最终结果。"""
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < partials_per_row

    partials = tl.load(
        partials_ptr + row * partials_per_row + offsets,
        mask=mask,
        other=0.0,
    )
    row_sum = tl.sum(partials, axis=0)
    tl.store(output_ptr + row, row_sum)


def row_sum_triton_v1(x: torch.Tensor) -> torch.Tensor:
    """用两阶段 Triton Reduction 计算二维输入的每行和。

    Stage 1 将每一行拆成固定大小的块，写出 FP32 局部和；Stage 2 再将
    一行的所有局部和合并。中间结果使用 FP32，避免 FP16 的局部和过早舍入。
    """
    if x.ndim != 2:
        raise ValueError(f"输入必须是二维 Tensor，实际维度为 {x.ndim}")
    if not x.is_cuda:
        raise ValueError("Triton 实现要求输入位于 CUDA 设备上")
    if not x.is_contiguous():
        raise ValueError("Reduction V1 暂时只支持内存连续的 Tensor")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("Reduction V1 暂时只支持 float16 和 float32")

    n_rows, n_columns = x.shape
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    if n_rows == 0:
        return output
    if n_columns == 0:
        return torch.zeros_like(output)

    partials_per_row = triton.cdiv(n_columns, CHUNK_SIZE)
    if partials_per_row > MAX_PARTIALS_PER_ROW:
        max_columns = CHUNK_SIZE * MAX_PARTIALS_PER_ROW
        raise ValueError(
            f"Reduction V1 每行最多支持 {max_columns:,} 个元素，"
            f"实际为 {n_columns:,}；更大行宽需要更多归约阶段"
        )

    # partials 是 [M, ceil(N / CHUNK_SIZE)]，每个元素对应一个块的 FP32 局部和。
    partials = torch.empty(
        (n_rows, partials_per_row), device=x.device, dtype=torch.float32
    )
    _row_sum_stage1_kernel[(n_rows, partials_per_row)](
        x,
        partials,
        n_columns,
        partials_per_row,
        BLOCK_SIZE=CHUNK_SIZE,
    )

    partial_block_size = triton.next_power_of_2(partials_per_row)
    _row_sum_stage2_kernel[(n_rows,)](
        partials,
        output,
        partials_per_row,
        BLOCK_SIZE=partial_block_size,
    )
    return output
