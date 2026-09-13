"""按行求和 Reduction 的 Triton V0：一个 Program 处理一整行。"""

import torch
import triton
import triton.language as tl


MAX_BLOCK_SIZE = 65_536


@triton.jit
def _row_sum_kernel(
    x_ptr,
    output_ptr,
    n_columns,
    BLOCK_SIZE: tl.constexpr,
):
    """将第 program_id 行的 BLOCK_SIZE 个位置归约为一个输出。"""
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_columns

    # Mask 外的位置在数学上不属于这一行；以 0 补齐不会改变求和结果。
    values = tl.load(x_ptr + row * n_columns + offsets, mask=mask, other=0.0)
    # FP16 输入也使用 FP32 累加，降低归约树中反复舍入带来的误差。
    row_sum = tl.sum(values.to(tl.float32), axis=0)
    tl.store(output_ptr + row, row_sum)


def row_sum_triton(x: torch.Tensor) -> torch.Tensor:
    """使用一个 Triton Program 计算输入矩阵的一行之和。"""
    if x.ndim != 2:
        raise ValueError(f"输入必须是二维 Tensor，实际维度为 {x.ndim}")
    if not x.is_cuda:
        raise ValueError("Triton 实现要求输入位于 CUDA 设备上")
    if not x.is_contiguous():
        raise ValueError("Reduction V0 暂时只支持内存连续的 Tensor")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("Reduction V0 暂时只支持 float16 和 float32")

    n_rows, n_columns = x.shape
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    if n_rows == 0:
        return output
    if n_columns == 0:
        return torch.zeros_like(output)

    block_size = triton.next_power_of_2(n_columns)
    if block_size > MAX_BLOCK_SIZE:
        raise ValueError(
            f"Reduction V0 每行最多支持 {MAX_BLOCK_SIZE:,} 个元素，"
            f"实际为 {n_columns:,}；更大行宽需要多阶段归约"
        )

    # 一个 Program 对应一行，因此 Grid 中 Program 总数就是 n_rows。
    _row_sum_kernel[(n_rows,)](
        x,
        output,
        n_columns,
        BLOCK_SIZE=block_size,
    )
    return output
