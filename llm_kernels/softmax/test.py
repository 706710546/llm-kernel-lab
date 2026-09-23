"""验证 PyTorch 与 Triton Softmax 的结果一致。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.softmax.torch_impl import softmax_torch
from llm_kernels.softmax.cuda_impl import softmax_cuda
from llm_kernels.softmax.triton_impl import MAX_BLOCK_SIZE, softmax_triton
from llm_kernels.softmax.triton_v1_impl import softmax_triton_v1
from llm_kernels.softmax.triton_v2_impl import softmax_triton_v2


def test_numerical_stability() -> None:
    """大正数和大负数输入不应产生 inf 或 nan。"""
    x = torch.tensor(
        [
            [10_000.0, 10_001.0, 10_002.0],
            [-10_000.0, -9_999.0, -9_998.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    expected = softmax_torch(x)
    for name, implementation in (
        ("V0", softmax_triton),
        ("V1", softmax_triton_v1),
        ("V2", softmax_triton_v2),
        ("CUDA", softmax_cuda),
    ):
        actual = implementation(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(actual).all()
        print(f"通过 {name} 数值稳定性样例：大正数与大负数均未产生 inf/nan")

    # 最大值位于最后一个块，验证跨块的 max 与 sum 合并。
    wide = torch.full((2, 1_025), -10_000.0, device="cuda", dtype=torch.float32)
    wide[0, 1_024] = 10_000.0
    wide[1, 0] = 10_000.0
    for name, implementation in (
        ("V1", softmax_triton_v1),
        ("V2", softmax_triton_v2),
        ("CUDA", softmax_cuda),
    ):
        torch.testing.assert_close(implementation(wide), softmax_torch(wide), rtol=0, atol=0)
        print(f"通过 {name} 跨块极值样例：最大值分别位于首块和尾块")


def test_shapes() -> None:
    shapes = (
        (0, 0),
        (0, 17),
        (7, 0),
        (1, 1),
        (1, 17),
        (7, 256),
        (31, 1_003),
        (2, 1_024),
        (2, 1_025),
        (64, 4_096),
        (2, 65_536),
        (2, 65_537),
        (2, 131_072),
    )
    for dtype in (torch.float32, torch.float16):
        rtol, atol = {
            torch.float32: (1e-4, 1e-6),
            torch.float16: (2e-3, 2e-3),
        }[dtype]
        for rows, columns in shapes:
            x = torch.randn((rows, columns), device="cuda", dtype=dtype)
            expected = softmax_torch(x)
            implementations = [("V1", softmax_triton_v1), ("V2", softmax_triton_v2)]
            if dtype == torch.float32:
                implementations.append(("CUDA", softmax_cuda))
            if columns <= MAX_BLOCK_SIZE:
                implementations.insert(0, ("V0", softmax_triton))
            for name, implementation in implementations:
                actual = implementation(x)
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                assert actual.shape == x.shape

                if rows > 0 and columns > 0:
                    row_sums = actual.to(torch.float32).sum(dim=-1)
                    torch.testing.assert_close(
                        row_sums,
                        torch.ones_like(row_sums),
                        rtol=2e-3,
                        atol=2e-3,
                    )
                print(
                    f"通过 {name} dtype={str(dtype).removeprefix('torch.'):7s} "
                    f"shape=({rows:,}, {columns:,})"
                )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    test_numerical_stability()
    test_shapes()
    print("所有 Softmax V0 / V1 / V2 / CUDA 正确性测试均已通过。")


if __name__ == "__main__":
    main()
