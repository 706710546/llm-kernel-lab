"""为 Nsight Compute 提供单一、可重复的 Triton Transpose 分析目标。"""

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.transpose.triton_impl import transpose_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行用于性能分析的 Triton Transpose")
    parser.add_argument("--rows", type=int, default=4096, help="输入矩阵行数")
    parser.add_argument("--columns", type=int, default=4096, help="输入矩阵列数")
    parser.add_argument("--block-size", type=int, default=32, help="方形 Tile 的边长")
    parser.add_argument("--warmup", type=int, default=10, help="正式分析前的预热次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    if args.rows <= 0 or args.columns <= 0:
        raise ValueError("矩阵行数和列数必须大于 0")
    if args.block_size not in (16, 32):
        raise ValueError("--block-size 当前只支持 16 或 32")
    if args.warmup < 0:
        raise ValueError("--warmup 不能小于 0")

    x = torch.randn(
        (args.rows, args.columns),
        device="cuda",
        dtype=torch.float32,
    )
    for _ in range(args.warmup):
        output = transpose_triton(x, block_size=args.block_size)
    torch.cuda.synchronize()

    output = transpose_triton(x, block_size=args.block_size)
    torch.cuda.synchronize()

    expected = x.transpose(0, 1).contiguous()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    grid_rows = (args.rows + args.block_size - 1) // args.block_size
    grid_columns = (args.columns + args.block_size - 1) // args.block_size
    print(f"输入形状：({args.rows:,}, {args.columns:,})")
    print(f"Tile：{args.block_size} × {args.block_size}")
    print(f"Grid：({grid_rows:,}, {grid_columns:,})")
    print("正确性检查：通过")


if __name__ == "__main__":
    main()
