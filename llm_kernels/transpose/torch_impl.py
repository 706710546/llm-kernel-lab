"""Matrix Transpose 的 PyTorch 参考实现。"""

import torch


def transpose_torch(x: torch.Tensor) -> torch.Tensor:
    """交换二维 Tensor 的两个维度，并返回连续的输出。"""
    if x.ndim != 2:
        raise ValueError("x 必须是二维 Tensor")
    return x.transpose(0, 1).contiguous()
