"""Vector Add 正确性检查，不依赖外部测试框架。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.vector_add.torch_impl import vector_add_torch
from llm_kernels.vector_add.triton_impl import vector_add_triton


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")

    test_sizes = (0, 1, 17, 256, 1_000, 65_537, 1 << 20)
    test_dtypes = (torch.float32, torch.float16)

    for dtype in test_dtypes:
        for size in test_sizes:
            x = torch.randn(size, device="cuda", dtype=dtype)
            y = torch.randn(size, device="cuda", dtype=dtype)

            expected = vector_add_torch(x, y)
            actual = vector_add_triton(x, y)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(f"通过 dtype={str(dtype).removeprefix('torch.'):7s} N={size:,}")

    print("所有 Vector Add 正确性测试均已通过。")


if __name__ == "__main__":
    main()
