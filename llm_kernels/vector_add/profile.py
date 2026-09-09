"""为 Nsight Compute 提供单一、可重复的 Vector Add 分析目标。"""

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.vector_add.triton_impl import vector_add_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行用于性能分析的 Triton Vector Add")
    parser.add_argument("--size", type=int, default=1 << 24, help="向量元素数量")
    parser.add_argument("--block-size", type=int, default=256, help="每个 Program 处理的元素数")
    parser.add_argument("--warmup", type=int, default=10, help="正式分析前的预热次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    if args.size <= 0:
        raise ValueError("--size 必须大于 0")
    if args.block_size <= 0 or args.block_size & (args.block_size - 1):
        raise ValueError("--block-size 必须是正的 2 的幂")

    x = torch.randn(args.size, device="cuda", dtype=torch.float32)
    y = torch.randn(args.size, device="cuda", dtype=torch.float32)

    # 前十次调用用于完成 JIT 编译、填充缓存并让 GPU 进入稳定状态。
    for _ in range(args.warmup):
        output = vector_add_triton(x, y, block_size=args.block_size)
    torch.cuda.synchronize()

    # Nsight Compute 使用 --launch-skip 跳过预热，只捕获下面这一次调用。
    output = vector_add_triton(x, y, block_size=args.block_size)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, x + y, rtol=0, atol=0)
    program_count = (args.size + args.block_size - 1) // args.block_size
    print(f"元素数量：{args.size:,}")
    print(f"Block Size：{args.block_size}")
    print(f"Program 数量：{program_count:,}")
    print("正确性检查：通过")


if __name__ == "__main__":
    main()

