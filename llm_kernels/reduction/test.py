"""按行求和 Reduction 的正确性测试，不依赖外部测试框架。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.reduction.torch_impl import row_sum_torch
from llm_kernels.reduction.triton_impl import row_sum_triton


def test_known_example() -> None:
    """先用可以手算的例子确认数学定义。"""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
    actual = row_sum_triton(x)
    expected = torch.tensor([6.0, 15.0], device="cuda")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("通过 Triton 手算样例：[[1, 2, 3], [4, 5, 6]] → [6, 15]")


def test_shapes() -> None:
    """覆盖未来 Triton Kernel 必须支持的普通和边界形状。"""
    test_shapes = ((1, 1), (1, 17), (7, 256), (31, 1_003), (64, 4_096), (8, 0))
    for dtype in (torch.float32, torch.float16):
        for rows, columns in test_shapes:
            x = torch.randn((rows, columns), device="cuda", dtype=dtype)
            expected = row_sum_torch(x)
            actual = row_sum_triton(x)
            tolerance = {torch.float32: (1e-5, 1e-5), torch.float16: (1e-2, 1e-2)}[dtype]
            torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
            assert actual.shape == (rows,)
            print(
                f"通过 dtype={str(dtype).removeprefix('torch.'):7s} "
                f"shape=({rows:,}, {columns:,}) → ({rows:,})"
            )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    test_known_example()
    test_shapes()
    print("所有 Reduction Triton V0 正确性测试均已通过。")


if __name__ == "__main__":
    main()
