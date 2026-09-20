"""为 Nsight Compute 提供单一、可重复的 CUDA C++ Vector Add 分析目标。"""

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.vector_add.cuda_impl import vector_add_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行用于性能分析的 CUDA C++ Vector Add")
    parser.add_argument("--size", type=int, default=1 << 24, help="向量元素数量")
    parser.add_argument("--threads", type=int, default=256, help="每个 CUDA Block 的线程数")
    parser.add_argument("--warmup", type=int, default=10, help="正式分析前的预热次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    if args.size <= 0:
        raise ValueError("--size 必须大于 0")
    if not 32 <= args.threads <= 1024 or args.threads & (args.threads - 1):
        raise ValueError("--threads 必须是 [32, 1024] 范围内的 2 的幂")
    if args.warmup < 0:
        raise ValueError("--warmup 不能小于 0")

    x = torch.randn(args.size, device="cuda", dtype=torch.float32)
    y = torch.randn(args.size, device="cuda", dtype=torch.float32)

    # 第一次调用会加载扩展；随后的调用让 GPU 和缓存进入稳定状态。
    for _ in range(args.warmup):
        output = vector_add_cuda(x, y, threads_per_block=args.threads)
    torch.cuda.synchronize()

    # NCU 使用 --launch-skip 跳过预热，只捕获下面这一次调用。
    output = vector_add_cuda(x, y, threads_per_block=args.threads)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, x + y, rtol=0, atol=0)
    block_count = (args.size + args.threads - 1) // args.threads
    print(f"元素数量：{args.size:,}")
    print(f"Threads / Block：{args.threads}")
    print(f"Block 数量：{block_count:,}")
    print("正确性检查：通过")


if __name__ == "__main__":
    main()
