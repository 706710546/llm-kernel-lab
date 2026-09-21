"""验证 PyTorch 与 Triton Matrix Transpose 的结果一致。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.transpose.torch_impl import transpose_torch
from llm_kernels.transpose.triton_impl import transpose_triton


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")

    shapes = (
        (0, 0),
        (0, 7),
        (7, 0),
        (1, 1),
        (1, 17),
        (17, 1),
        (31, 33),
        (32, 32),
        (33, 65),
        (513, 1000),
    )
    for dtype in (torch.float32, torch.float16):
        for shape in shapes:
            x = torch.randn(shape, device="cuda", dtype=dtype)
            expected = transpose_torch(x)
            actual = transpose_triton(x)

            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.is_contiguous()
            print(f"通过 dtype={dtype} shape={shape}")

    print("所有 Matrix Transpose 正确性测试均已通过。")


if __name__ == "__main__":
    main()
