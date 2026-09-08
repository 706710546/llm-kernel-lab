"""Vector Add 的 PyTorch 参考实现。"""

import torch


def vector_add_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """返回逐元素相加结果，作为正确性验证的参考。"""
    if x.shape != y.shape:
        raise ValueError(f"形状不一致：{x.shape} != {y.shape}")
    if x.device != y.device:
        raise ValueError(f"设备不一致：{x.device} != {y.device}")
    if x.dtype != y.dtype:
        raise ValueError(f"数据类型不一致：{x.dtype} != {y.dtype}")
    return x + y
