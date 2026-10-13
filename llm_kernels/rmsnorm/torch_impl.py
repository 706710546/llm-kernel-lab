"""RMSNorm 的 PyTorch 参考实现。"""

import math

import torch


def rmsnorm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """沿最后一维执行 RMSNorm，并返回与输入相同 dtype 的结果。

    x 可以有任意正维数，weight 的形状必须等于 x 的最后一维。
    FP16/BF16 输入会先转成 FP32 完成平方、均值和归一化，再转回输入 dtype。
    """
    if x.ndim < 1:
        raise ValueError("x 至少必须是一维 Tensor")
    if x.shape[-1] == 0:
        raise ValueError("RMSNorm 的归一化维度不能是空的")
    if not x.is_floating_point():
        raise TypeError(f"x 必须是浮点 Tensor，实际 dtype 为 {x.dtype}")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(
            f"weight 形状必须是 ({x.shape[-1]},)，实际为 {tuple(weight.shape)}"
        )
    if weight.device != x.device:
        raise ValueError(f"x 与 weight 必须在同一设备：{x.device} != {weight.device}")
    if weight.dtype != x.dtype:
        raise TypeError(f"x 与 weight 必须使用相同 dtype：{x.dtype} != {weight.dtype}")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError(f"eps 必须是有限的非负数，实际为 {eps}")

    accumulation_dtype = (
        torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    )
    x_acc = x.to(accumulation_dtype)
    weight_acc = weight.to(accumulation_dtype)

    mean_square = x_acc.square().mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_square + eps)
    output = x_acc * inverse_rms * weight_acc
    return output.to(x.dtype)
