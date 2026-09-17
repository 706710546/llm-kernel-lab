"""为 Nsight Compute 提供可重复的 Row Sum V0 / V1 分析目标。"""

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.reduction.torch_impl import row_sum_torch
from llm_kernels.reduction.triton_impl import MAX_BLOCK_SIZE, row_sum_triton
from llm_kernels.reduction.triton_v1_impl import row_sum_triton_v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行用于性能分析的 Triton Row Sum")
    parser.add_argument("--version", choices=("v0", "v1"), default="v1", help="待分析的实现版本")
    parser.add_argument("--rows", type=int, default=256, help="输入矩阵的行数 M")
    parser.add_argument("--columns", type=int, default=65_536, help="输入矩阵的列数 N")
    parser.add_argument("--warmup", type=int, default=10, help="正式分析前的预热次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    if args.rows <= 0 or args.columns <= 0:
        raise ValueError("--rows 和 --columns 必须大于 0")
    if args.warmup < 0:
        raise ValueError("--warmup 不能小于 0")
    if args.version == "v0" and args.columns > MAX_BLOCK_SIZE:
        raise ValueError(f"V0 每行最多支持 {MAX_BLOCK_SIZE:,} 个元素")

    x = torch.randn((args.rows, args.columns), device="cuda", dtype=torch.float32)
    implementation = row_sum_triton if args.version == "v0" else row_sum_triton_v1

    # 预热完成 JIT 编译，并使后续捕获只包含稳定运行的 kernel。
    for _ in range(args.warmup):
        output = implementation(x)
    torch.cuda.synchronize()

    # NCU 使用 --launch-skip 跳过预热后，捕获本次调用产生的 kernel。
    output = implementation(x)
    torch.cuda.synchronize()

    expected = row_sum_torch(x)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-4)
    print(f"版本：{args.version.upper()}")
    print(f"输入形状：({args.rows:,}, {args.columns:,})")
    if args.version == "v1":
        partials_per_row = (args.columns + 1_024 - 1) // 1_024
        print(f"Stage 1 Program 数量：{args.rows * partials_per_row:,}")
        print(f"Stage 2 Program 数量：{args.rows:,}")
    else:
        print(f"Program 数量：{args.rows:,}")
    print("正确性检查：通过")


if __name__ == "__main__":
    main()
