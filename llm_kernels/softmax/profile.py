"""为 Nsight Compute 提供 V0/V1/V2 Softmax 分阶段分析目标。"""

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.softmax.triton_impl import softmax_triton
from llm_kernels.softmax.triton_v1_impl import CHUNK_SIZE, softmax_triton_v1
from llm_kernels.softmax.triton_v2_impl import softmax_triton_v2
from llm_kernels.softmax.cuda_impl import softmax_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行用于 NCU 分析的 Triton Softmax")
    parser.add_argument("--version", choices=("v0", "v1", "v2", "cuda"), default="v0")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=8192)
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

    x = torch.randn(
        (args.rows, args.columns),
        device="cuda",
        dtype=torch.float32,
    )
    implementations = {
        "v0": softmax_triton,
        "v1": softmax_triton_v1,
        "v2": softmax_triton_v2,
        "cuda": softmax_cuda,
    }
    implementation = implementations[args.version]
    for _ in range(args.warmup):
        output = implementation(x)
    torch.cuda.synchronize()

    output = implementation(x)
    torch.cuda.synchronize()
    expected = torch.softmax(x, dim=-1)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-6)

    print(f"输入形状：({args.rows:,}, {args.columns:,})")
    print(f"版本：{args.version}")
    if args.version in ("v0", "v2"):
        print(f"Program 数量：{args.rows:,}")
    elif args.version == "cuda":
        print(f"CUDA Block 数量：{args.rows:,}；每个 Block 256 个线程")
    else:
        chunks = (args.columns + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"Stage 1/3 Grid：({args.rows:,}, {chunks:,})")
        print(f"Stage 2 Grid：({args.rows:,},)")
    print("正确性检查：通过")


if __name__ == "__main__":
    main()
