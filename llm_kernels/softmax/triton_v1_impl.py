"""Softmax V1：分块统计最大值与指数和，再归一化输出。"""

import torch
import triton
import triton.language as tl


CHUNK_SIZE = 1024
MAX_CHUNKS_PER_ROW = 1024


@triton.jit
def _softmax_v1_partials_kernel(
    input_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    n_columns,
    n_chunks,
    CHUNK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    columns = chunk * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    values = tl.load(
        input_ptr + row * n_columns + columns,
        mask=columns < n_columns,
        other=-float("inf"),
    ).to(tl.float32)

    local_max = tl.max(values, 0)
    local_sum = tl.sum(tl.exp(values - local_max), 0)
    partial_offset = row * n_chunks + chunk
    tl.store(partial_max_ptr + partial_offset, local_max)
    tl.store(partial_sum_ptr + partial_offset, local_sum)


@triton.jit
def _softmax_v1_merge_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    row_max_ptr,
    row_sum_ptr,
    n_chunks,
    MERGE_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunks = tl.arange(0, MERGE_SIZE)
    valid = chunks < n_chunks
    partial_offsets = row * n_chunks + chunks
    partial_max = tl.load(
        partial_max_ptr + partial_offsets,
        mask=valid,
        other=-float("inf"),
    )
    partial_sum = tl.load(partial_sum_ptr + partial_offsets, mask=valid, other=0.0)

    row_max = tl.max(partial_max, 0)
    row_sum = tl.sum(partial_sum * tl.exp(partial_max - row_max), 0)
    tl.store(row_max_ptr + row, row_max)
    tl.store(row_sum_ptr + row, row_sum)


@triton.jit
def _softmax_v1_write_kernel(
    input_ptr,
    output_ptr,
    row_max_ptr,
    row_sum_ptr,
    n_columns,
    CHUNK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    columns = chunk * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    offsets = row * n_columns + columns
    valid = columns < n_columns
    values = tl.load(input_ptr + offsets, mask=valid, other=-float("inf")).to(tl.float32)
    row_max = tl.load(row_max_ptr + row)
    row_sum = tl.load(row_sum_ptr + row)
    probabilities = tl.exp(values - row_max) / row_sum
    tl.store(output_ptr + offsets, probabilities, mask=valid)


def softmax_triton_v1(x: torch.Tensor) -> torch.Tensor:
    """分块处理连续二维 FP16/FP32 Tensor 的按行 Softmax。"""
    if x.ndim != 2:
        raise ValueError("x 必须是二维 Tensor")
    if not x.is_cuda:
        raise ValueError("x 必须位于 CUDA 设备上")
    if not x.is_contiguous():
        raise ValueError("Softmax V1 暂时只支持连续 Tensor")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("Softmax V1 暂时只支持 float16 和 float32")

    n_rows, n_columns = x.shape
    output = torch.empty_like(x)
    if x.numel() == 0:
        return output

    n_chunks = triton.cdiv(n_columns, CHUNK_SIZE)
    if n_chunks > MAX_CHUNKS_PER_ROW:
        raise ValueError(
            f"Softmax V1 每行最多支持 {CHUNK_SIZE * MAX_CHUNKS_PER_ROW:,} 个元素"
        )

    partial_max = torch.empty((n_rows, n_chunks), device=x.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    row_max = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    row_sum = torch.empty_like(row_max)

    _softmax_v1_partials_kernel[(n_rows, n_chunks)](
        x,
        partial_max,
        partial_sum,
        n_columns,
        n_chunks,
        CHUNK_SIZE=CHUNK_SIZE,
    )
    _softmax_v1_merge_kernel[(n_rows,)](
        partial_max,
        partial_sum,
        row_max,
        row_sum,
        n_chunks,
        MERGE_SIZE=triton.next_power_of_2(n_chunks),
    )
    _softmax_v1_write_kernel[(n_rows, n_chunks)](
        x,
        output,
        row_max,
        row_sum,
        n_columns,
        CHUNK_SIZE=CHUNK_SIZE,
    )
    return output
