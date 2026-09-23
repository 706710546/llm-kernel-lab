"""Softmax 的 PyTorch 参考实现。"""

import torch


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """沿二维 Tensor 的最后一维计算 Softmax。"""
    if x.ndim != 2:
        raise ValueError("x 必须是二维 Tensor")
    return torch.softmax(x, dim=-1)
