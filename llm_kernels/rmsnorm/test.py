"""RMSNorm PyTorch 参考实现的手算、形状、精度与梯度检查。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.rmsnorm.torch_impl import rmsnorm_torch
from llm_kernels.rmsnorm.triton_impl import (
    MAX_BLOCK_SIZE,
    rmsnorm_triton,
    rmsnorm_triton_auto,
)
from llm_kernels.rmsnorm.triton_v1_impl import rmsnorm_triton_v1
from llm_kernels.rmsnorm.cuda_impl import rmsnorm_cuda, rmsnorm_cuda_shuffle


def test_known_example(device: str) -> None:
    """用 [3, 4] 手算 RMS，确认没有减均值且逐元素乘上 weight。"""
    x = torch.tensor([[3.0, 4.0]], device=device)
    weight = torch.tensor([2.0, 0.5], device=device)
    expected = torch.tensor(
        [[6.0 / (12.5**0.5), 2.0 / (12.5**0.5)]], device=device
    )

    actual = rmsnorm_torch(x, weight, eps=0.0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    print(f"通过手算样例 device={device}：RMS=sqrt((3^2+4^2)/2)=sqrt(12.5)")


def test_shapes_and_dtypes(device: str) -> None:
    """确认不同隐藏维、前置维数和低精度输入都保持预期形状与 dtype。"""
    torch.manual_seed(0)
    shapes = ((17,), (3, 128), (2, 3, 257))
    dtypes = (torch.float32, torch.float16, torch.bfloat16)

    for dtype in dtypes:
        for shape in shapes:
            x = torch.randn(shape, device=device, dtype=dtype)
            weight = torch.randn((shape[-1],), device=device, dtype=dtype)
            actual = rmsnorm_torch(x, weight)

            x_fp32 = x.float()
            weight_fp32 = weight.float()
            expected = (
                x_fp32
                * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + 1e-6)
                * weight_fp32
            ).to(dtype)

            tolerance = {
                torch.float32: (1e-5, 1e-6),
                torch.float16: (1e-3, 1e-3),
                torch.bfloat16: (1e-2, 1e-2),
            }[dtype]
            torch.testing.assert_close(
                actual, expected, rtol=tolerance[0], atol=tolerance[1]
            )
            assert actual.shape == x.shape
            assert actual.dtype == x.dtype
            print(
                f"通过 device={device}, dtype={str(dtype).removeprefix('torch.')}, "
                f"shape={tuple(shape)}"
            )


def test_autograd(device: str) -> None:
    """PyTorch 参考实现应保留对输入和缩放参数的自动微分关系。"""
    x = torch.randn((2, 17), device=device, requires_grad=True)
    weight = torch.randn((17,), device=device, requires_grad=True)
    rmsnorm_torch(x, weight).square().sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    print(f"通过 autograd device={device}：x 与 weight 的梯度均有限")


def test_invalid_inputs() -> None:
    x = torch.randn((2, 17))
    for invalid_weight, eps, error_type in (
        (torch.randn(16), 1e-6, ValueError),
        (torch.randn(17, dtype=torch.float64), 1e-6, TypeError),
        (torch.randn(17), -1.0, ValueError),
    ):
        try:
            rmsnorm_torch(x, invalid_weight, eps=eps)
        except error_type:
            continue
        raise AssertionError(f"输入应触发 {error_type.__name__}")
    print("通过非法输入检查：形状、dtype 与 eps 错误均被明确拒绝")


def test_triton_shapes() -> None:
    """验证 Triton V0 的 padding、广播、dtype 与空 batch 行为。"""
    shapes = (
        (0, 17),
        (1, 1),
        (2, 17),
        (3, 127),
        (3, 257),
        (2, 3, 1_024),
        (8, 4_096),
        (2, 8_192),
        (2, MAX_BLOCK_SIZE),
    )
    tolerances = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }

    for dtype, (rtol, atol) in tolerances.items():
        for shape in shapes:
            x = torch.randn(shape, device="cuda", dtype=dtype)
            weight = torch.randn((shape[-1],), device="cuda", dtype=dtype)
            expected = rmsnorm_torch(x, weight)
            actual = rmsnorm_triton(x, weight)

            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            assert actual.shape == x.shape
            assert actual.dtype == x.dtype
            print(
                f"通过 Triton V0 dtype={str(dtype).removeprefix('torch.'):7s} "
                f"shape={tuple(shape)}"
            )


def test_triton_invalid_inputs() -> None:
    x = torch.randn((2, 17), device="cuda")
    valid_weight = torch.ones((17,), device="cuda")
    noncontiguous_x = torch.randn((3, 17), device="cuda").t()
    too_wide = torch.randn((1, MAX_BLOCK_SIZE + 1), device="cuda")
    checks = (
        (x, torch.ones((16,), device="cuda"), 1e-6, ValueError),
        (x, valid_weight.to(torch.float16), 1e-6, TypeError),
        (
            torch.randn((2, 17), device="cuda", dtype=torch.float64),
            valid_weight,
            1e-6,
            TypeError,
        ),
        (noncontiguous_x, torch.ones((3,), device="cuda"), 1e-6, ValueError),
        (
            too_wide,
            torch.ones((MAX_BLOCK_SIZE + 1,), device="cuda"),
            1e-6,
            ValueError,
        ),
    )
    for input_x, weight, eps, error_type in checks:
        try:
            rmsnorm_triton(input_x, weight, eps=eps)
        except error_type:
            continue
        raise AssertionError(f"Triton V0 输入应触发 {error_type.__name__}")
    try:
        rmsnorm_triton(x, valid_weight, num_warps=3)
    except ValueError:
        pass
    else:
        raise AssertionError("num_warps=3 应触发 ValueError")
    print("通过 Triton V0 参数检查：形状、dtype、stride 和宽行限制均生效")


def test_num_warps() -> None:
    """不同 CTA warp 数必须保持数值结果一致。"""
    x = torch.randn((64, 4_096), device="cuda", dtype=torch.float32)
    weight = torch.randn((4_096,), device="cuda", dtype=torch.float32)
    expected = rmsnorm_torch(x, weight)
    for num_warps in (4, 8):
        actual = rmsnorm_triton(x, weight, num_warps=num_warps)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        print(f"通过 num_warps={num_warps} 数值对照")


def test_triton_v1_wide_rows() -> None:
    """验证分块 V1 的 partial sum 合并、尾块 mask 与 dtype。"""
    shapes = ((2, 16_385), (2, 32_769), (2, 65_536))
    tolerances = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }
    for dtype, (rtol, atol) in tolerances.items():
        for shape in shapes:
            x = torch.randn(shape, device="cuda", dtype=dtype)
            weight = torch.randn((shape[-1],), device="cuda", dtype=dtype)
            expected = rmsnorm_torch(x, weight)
            actual = rmsnorm_triton_v1(x, weight)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            assert actual.shape == x.shape and actual.dtype == dtype
            print(f"通过 Triton V1 dtype={dtype} shape={shape}")


def test_triton_auto_dispatch() -> None:
    """确认自动入口按宽度选择 V0/V1，结果仍与 PyTorch 参考一致。"""
    for columns in (4_096, 32_769):
        x = torch.randn((2, columns), device="cuda", dtype=torch.float32)
        weight = torch.randn((columns,), device="cuda", dtype=torch.float32)
        actual = rmsnorm_triton_auto(x, weight)
        expected = rmsnorm_torch(x, weight)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        print(f"通过 Triton Auto Dispatch N={columns:,}")


def test_cuda_shapes() -> None:
    """验证 CUDA C++ 共享内存与 Warp Shuffle 两个版本。"""
    shapes = (
        (0, 17),
        (1, 1),
        (2, 17),
        (3, 257),
        (2, 3, 1_024),
        (64, 4_096),
        (2, 32_769),
    )
    tolerances = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }
    for dtype, (rtol, atol) in tolerances.items():
        for shape in shapes:
            x = torch.randn(shape, device="cuda", dtype=dtype)
            weight = torch.randn((shape[-1],), device="cuda", dtype=dtype)
            expected = rmsnorm_torch(x, weight)
            for name, implementation in (
                ("shared", rmsnorm_cuda),
                ("shuffle", rmsnorm_cuda_shuffle),
            ):
                actual = implementation(x, weight)
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                assert actual.shape == x.shape
                assert actual.dtype == dtype
                print(
                    f"通过 CUDA C++ {name} "
                    f"{str(dtype).removeprefix('torch.')} shape={tuple(shape)}"
                )


def test_cuda_invalid_inputs() -> None:
    """确认 CUDA 包装层在编译和启动前拒绝不支持的输入。"""
    x = torch.randn((2, 17), device="cuda")
    weight = torch.ones((17,), device="cuda")
    checks = (
        (x.to(torch.float64), weight.to(torch.float64), 1e-6, TypeError),
        (x, weight.to(torch.float16), 1e-6, TypeError),
        (x.t(), torch.ones((2,), device="cuda"), 1e-6, ValueError),
        (x, torch.ones((16,), device="cuda"), 1e-6, ValueError),
        (x, weight, -1.0, ValueError),
    )
    for input_x, input_weight, eps, error_type in checks:
        try:
            rmsnorm_cuda(input_x, input_weight, eps=eps)
        except error_type:
            continue
        raise AssertionError(f"CUDA RMSNorm 输入应触发 {error_type.__name__}")
    print("通过 CUDA C++ 非法输入检查")


def main() -> None:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        test_known_example(device)
        test_shapes_and_dtypes(device)
        test_autograd(device)
    test_invalid_inputs()
    if torch.cuda.is_available():
        test_triton_shapes()
        test_triton_invalid_inputs()
        test_num_warps()
        test_triton_v1_wide_rows()
        test_triton_auto_dispatch()
        test_cuda_shapes()
        test_cuda_invalid_inputs()
    print("所有 RMSNorm PyTorch、Triton V0 与 CUDA C++ 测试均已通过。")


if __name__ == "__main__":
    main()
