"""RMSNorm Triton V0：每个 Program 负责一行归一化。"""

import math

import torch
import triton
import triton.language as tl


MAX_BLOCK_SIZE = 16_384


@triton.jit
def _rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < N

    offsets = row * N + columns
    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0).to(tl.float32)

    # Padding lanes load as zero; divide by the true width, not BLOCK_SIZE.
    mean_square = tl.sum(x * x, axis=0) / N
    inverse_rms = tl.rsqrt(mean_square + EPS)
    normalized = x * inverse_rms * weight

    tl.store(output_ptr + offsets, normalized, mask=mask)


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    num_warps: int = 4,
) -> torch.Tensor:
    """用 Triton V0 沿最后一维执行 RMSNorm。

    V0 要求 x 与 weight 都是 CUDA 上连续的 FP16、BF16 或 FP32 Tensor，
    并限制每行不超过 MAX_BLOCK_SIZE；低精度计算会在 Kernel 内提升到 FP32。
    num_warps 支持 1、2、4、8、16，用于实验每个 Program 的线程配置。
    """
    if x.ndim < 1:
        raise ValueError("x 至少必须是一维 Tensor")
    if x.shape[-1] == 0:
        raise ValueError("RMSNorm 的归一化维度不能是空的")
    if not x.is_cuda:
        raise ValueError("x 必须位于 CUDA 设备上")
    if not x.is_contiguous():
        raise ValueError("RMSNorm V0 暂时只支持连续 x")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("RMSNorm V0 暂时只支持 float16、bfloat16 和 float32")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(
            f"weight 形状必须是 ({x.shape[-1]},)，实际为 {tuple(weight.shape)}"
        )
    if weight.device != x.device:
        raise ValueError(f"x 与 weight 必须在同一设备：{x.device} != {weight.device}")
    if weight.dtype != x.dtype:
        raise TypeError(f"x 与 weight 必须使用相同 dtype：{x.dtype} != {weight.dtype}")
    if not weight.is_contiguous():
        raise ValueError("RMSNorm V0 暂时只支持连续 weight")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError(f"eps 必须是有限的非负数，实际为 {eps}")
    if num_warps not in (1, 2, 4, 8, 16):
        raise ValueError("num_warps 必须是 1、2、4、8 或 16")

    n_columns = x.shape[-1]
    block_size = triton.next_power_of_2(n_columns)
    if block_size > MAX_BLOCK_SIZE:
        raise ValueError(
            f"RMSNorm V0 每行最多支持 {MAX_BLOCK_SIZE:,} 个元素，"
            f"实际需要 {block_size:,} 个 lane；更宽的行需要分块算法"
        )

    output = torch.empty_like(x)
    if x.numel() == 0:
        return output

    n_rows = x.numel() // n_columns
    _rmsnorm_kernel[(n_rows,)](
        x,
        weight,
        output,
        N=n_columns,
        EPS=eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def rmsnorm_triton_auto(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    num_warps: int = 4,
) -> torch.Tensor:
    """按最后一维宽度分派：窄行使用 V0，宽行使用两阶段 V1。"""
    if x.ndim < 1:
        raise ValueError("x 至少必须是一维 Tensor")
    if x.shape[-1] <= MAX_BLOCK_SIZE:
        return rmsnorm_triton(x, weight, eps=eps, num_warps=num_warps)

    from llm_kernels.rmsnorm.triton_v1_impl import rmsnorm_triton_v1

    return rmsnorm_triton_v1(x, weight, eps=eps)
