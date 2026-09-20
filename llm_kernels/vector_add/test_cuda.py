"""验证 PyTorch、Triton 与 CUDA C++ Vector Add 的结果一致。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.vector_add.cuda_impl import vector_add_cuda
from llm_kernels.vector_add.torch_impl import vector_add_torch
from llm_kernels.vector_add.triton_impl import vector_add_triton


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")

    test_sizes = (0, 1, 17, 256, 1_000, 65_537, 1 << 20)
    thread_counts = (128, 256, 512)
    for threads_per_block in thread_counts:
        for size in test_sizes:
            x = torch.randn(size, device="cuda", dtype=torch.float32)
            y = torch.randn(size, device="cuda", dtype=torch.float32)

            expected = vector_add_torch(x, y)
            triton_actual = vector_add_triton(x, y)
            cuda_actual = vector_add_cuda(x, y, threads_per_block=threads_per_block)

            torch.testing.assert_close(triton_actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(cuda_actual, expected, rtol=0, atol=0)
            print(
                "通过 PyTorch / Triton / CUDA "
                f"FP32 threads={threads_per_block} N={size:,}"
            )

    try:
        vector_add_cuda(x, y, threads_per_block=96)
    except ValueError:
        print("通过非法 Threads / Block 参数检查")
    else:
        raise AssertionError("threads_per_block=96 应该触发 ValueError")

    print("所有 CUDA C++ Vector Add 正确性测试均已通过。")


if __name__ == "__main__":
    main()
