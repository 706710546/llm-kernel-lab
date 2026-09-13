"""按行求和 Reduction 的 PyTorch 参考实现。"""

import torch


def row_sum_torch(x: torch.Tensor) -> torch.Tensor:
    """沿最后一维求和，将形状 [M, N] 的输入变为形状 [M] 的输出。"""
    if x.ndim != 2:
        raise ValueError(f"输入必须是二维 Tensor，实际维度为 {x.ndim}")
    return x.sum(dim=-1)

