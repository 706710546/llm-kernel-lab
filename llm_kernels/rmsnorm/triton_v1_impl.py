"""RMSNorm Triton V1：用两阶段 Kernel 支持超过单 Program 上限的宽行。"""

import math

import torch
import triton
import triton.language as tl


V1_TILE_SIZE = 4_096
MAX_V1_CHUNKS = 1_024


@triton.jit
def _rmsnorm_partial_kernel(
    input_ptr,
    partial_ptr,
    N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    tile = tl.program_id(axis=0)
    row = tl.program_id(axis=1)
    columns = tile * TILE_SIZE + tl.arange(0, TILE_SIZE)
    values = tl.load(input_ptr + row * N + columns, mask=columns < N, other=0).to(tl.float32)
    square_sum = tl.sum(values * values, axis=0)
    tl.store(partial_ptr + row * NUM_CHUNKS + tile, square_sum)


@triton.jit
def _rmsnorm_output_kernel(
    input_ptr,
    weight_ptr,
    partial_ptr,
    output_ptr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    PARTIALS_BLOCK: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    tile = tl.program_id(axis=0)
    row = tl.program_id(axis=1)

    partial_ids = tl.arange(0, PARTIALS_BLOCK)
    partial_values = tl.load(
        partial_ptr + row * NUM_CHUNKS + partial_ids,
        mask=partial_ids < NUM_CHUNKS,
        other=0,
    )
    mean_square = tl.sum(partial_values, axis=0) / N
    inverse_rms = tl.rsqrt(mean_square + EPS)

    columns = tile * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = columns < N
    offsets = row * N + columns
    values = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float32)
    weights = tl.load(weight_ptr + columns, mask=mask, other=0).to(tl.float32)
    tl.store(output_ptr + offsets, values * inverse_rms * weights, mask=mask)


def rmsnorm_triton_v1(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """两阶段分块 RMSNorm，面向归一化宽度大于 Triton V0 上限的连续输入。"""
    if x.ndim < 1:
        raise ValueError("x 至少必须是一维 Tensor")
    if x.shape[-1] == 0:
        raise ValueError("RMSNorm 的归一化维度不能是空的")
    if not x.is_cuda:
        raise ValueError("x 必须位于 CUDA 设备上")
    if not x.is_contiguous():
        raise ValueError("RMSNorm V1 暂时只支持连续 x")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("RMSNorm V1 只支持 float16、bfloat16 和 float32")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(f"weight 形状必须是 ({x.shape[-1]},)，实际为 {tuple(weight.shape)}")
    if weight.device != x.device:
        raise ValueError("x 与 weight 必须位于同一设备")
    if weight.dtype != x.dtype:
        raise TypeError("x 与 weight 必须使用相同 dtype")
    if not weight.is_contiguous():
        raise ValueError("RMSNorm V1 暂时只支持连续 weight")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError(f"eps 必须是有限的非负数，实际为 {eps}")

    n_columns = x.shape[-1]
    if n_columns <= 16_384:
        raise ValueError("V1 用于超过 16,384 列的宽行；较窄行请使用 rmsnorm_triton")
    n_chunks = triton.cdiv(n_columns, V1_TILE_SIZE)
    if n_chunks > MAX_V1_CHUNKS:
        raise ValueError(f"RMSNorm V1 最多支持 {MAX_V1_CHUNKS * V1_TILE_SIZE:,} 列")

    output = torch.empty_like(x)
    if x.numel() == 0:
        return output

    n_rows = x.numel() // n_columns
    partials = torch.empty((n_rows, n_chunks), device=x.device, dtype=torch.float32)
    _rmsnorm_partial_kernel[(n_chunks, n_rows)](
        x,
        partials,
        N=n_columns,
        NUM_CHUNKS=n_chunks,
        TILE_SIZE=V1_TILE_SIZE,
        num_warps=4,
    )
    _rmsnorm_output_kernel[(n_chunks, n_rows)](
        x,
        weight,
        partials,
        output,
        N=n_columns,
        EPS=eps,
        NUM_CHUNKS=n_chunks,
        PARTIALS_BLOCK=triton.next_power_of_2(n_chunks),
        TILE_SIZE=V1_TILE_SIZE,
        num_warps=4,
    )
    return output
