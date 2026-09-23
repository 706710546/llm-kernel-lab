"""Softmax V2：单 Program 分块在线统计，再读取一次输入写出结果。"""

import torch
import triton
import triton.language as tl


CHUNK_SIZE = 1024
MAX_COLUMNS = 1_048_576


@triton.jit
def _softmax_v2_kernel(input_ptr, output_ptr, n_columns, CHUNK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, CHUNK_SIZE)
    n_chunks = tl.cdiv(n_columns, CHUNK_SIZE)

    # 前向扫描：只保留截至当前块的最大值 m 和指数和 l。
    m = -float("inf")
    l = 0.0
    for chunk in range(n_chunks):
        columns = chunk * CHUNK_SIZE + lanes
        values = tl.load(
            input_ptr + row * n_columns + columns,
            mask=columns < n_columns,
            other=-float("inf"),
        ).to(tl.float32)
        block_max = tl.max(values, 0)
        new_m = tl.maximum(m, block_max)
        l = l * tl.exp(m - new_m) + tl.sum(tl.exp(values - new_m), 0)
        m = new_m

    # 独立 Softmax 需要知道最终分母；第二遍重新读取并写出概率。
    for chunk in range(n_chunks):
        columns = chunk * CHUNK_SIZE + lanes
        values = tl.load(
            input_ptr + row * n_columns + columns,
            mask=columns < n_columns,
            other=-float("inf"),
        ).to(tl.float32)
        probabilities = tl.exp(values - m) / l
        tl.store(
            output_ptr + row * n_columns + columns,
            probabilities,
            mask=columns < n_columns,
        )


def softmax_triton_v2(x: torch.Tensor) -> torch.Tensor:
    """逐行扫描连续二维 FP16/FP32 Tensor，不分配跨 Kernel 中间缓冲。"""
    if x.ndim != 2:
        raise ValueError("x 必须是二维 Tensor")
    if not x.is_cuda:
        raise ValueError("x 必须位于 CUDA 设备上")
    if not x.is_contiguous():
        raise ValueError("Softmax V2 暂时只支持连续 Tensor")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("Softmax V2 暂时只支持 float16 和 float32")

    n_rows, n_columns = x.shape
    output = torch.empty_like(x)
    if x.numel() == 0:
        return output
    if n_columns > MAX_COLUMNS:
        raise ValueError(f"Softmax V2 每行最多支持 {MAX_COLUMNS:,} 个元素")

    _softmax_v2_kernel[(n_rows,)](x, output, n_columns, CHUNK_SIZE=CHUNK_SIZE)
    return output
