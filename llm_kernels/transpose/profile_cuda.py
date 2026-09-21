"""为 Nsight Compute 提供 CUDA Naive/Tiled Transpose 分析目标。"""

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.transpose.cuda_impl import (
    transpose_cuda_naive,
    transpose_cuda_tiled,
    transpose_cuda_tiled_unpadded,
)


IMPLEMENTATIONS = {
    "naive": transpose_cuda_naive,
    "tiled_unpadded": transpose_cuda_tiled_unpadded,
    "tiled": transpose_cuda_tiled,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行用于 NCU 分析的 CUDA Transpose")
    parser.add_argument("--version", choices=IMPLEMENTATIONS, default="tiled")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    if args.rows <= 0 or args.columns <= 0:
        raise ValueError("矩阵行数和列数必须大于 0")
    if args.warmup < 0:
        raise ValueError("--warmup 不能小于 0")

    implementation = IMPLEMENTATIONS[args.version]
    x = torch.randn(
        (args.rows, args.columns),
        device="cuda",
        dtype=torch.float32,
    )
    for _ in range(args.warmup):
        output = implementation(x)
    torch.cuda.synchronize()

    output = implementation(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        x.transpose(0, 1).contiguous(),
        rtol=0,
        atol=0,
    )

    grid_rows = (args.rows + 31) // 32
    grid_columns = (args.columns + 31) // 32
    print(f"版本：{args.version}")
    print(f"输入形状：({args.rows:,}, {args.columns:,})")
    print("CUDA Block：(32, 8)，共 256 Threads")
    print(f"CUDA Grid：({grid_columns:,}, {grid_rows:,})")
    print("正确性检查：通过")


if __name__ == "__main__":
    main()
