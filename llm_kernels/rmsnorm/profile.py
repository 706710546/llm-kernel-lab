"""提供固定输入的 RMSNorm V0/V1 与 CUDA 目标，供 Nsight Compute 捕获。"""

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.rmsnorm.torch_impl import rmsnorm_torch
from llm_kernels.rmsnorm.triton_impl import rmsnorm_triton
from llm_kernels.rmsnorm.triton_v1_impl import rmsnorm_triton_v1
from llm_kernels.rmsnorm.cuda_impl import rmsnorm_cuda, rmsnorm_cuda_shuffle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行用于 NCU 分析的 RMSNorm Kernel")
    parser.add_argument("--rows", type=int, default=1_024)
    parser.add_argument("--columns", type=int, default=4_096)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--provider",
        choices=("triton", "triton-v1", "cuda", "cuda-shuffle"),
        default="triton",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    if args.rows < 0 or args.columns <= 0:
        raise ValueError("rows 必须非负，columns 必须为正数")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    x = torch.randn((args.rows, args.columns), device="cuda", dtype=dtype)
    weight = torch.randn((args.columns,), device="cuda", dtype=dtype)
    functions = {
        "triton": lambda: rmsnorm_triton(x, weight, num_warps=args.num_warps),
        "triton-v1": lambda: rmsnorm_triton_v1(x, weight),
        "cuda": lambda: rmsnorm_cuda(x, weight),
        "cuda-shuffle": lambda: rmsnorm_cuda_shuffle(x, weight),
    }
    function = functions[args.provider]
    for _ in range(args.warmup):
        output = function()
    torch.cuda.synchronize()

    output = function()
    torch.cuda.synchronize()
    expected = rmsnorm_torch(x, weight)
    tolerances = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }
    rtol, atol = tolerances[dtype]
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    print(f"输入形状：({args.rows:,}, {args.columns:,})，dtype={args.dtype}")
    if args.provider == "triton":
        print(f"Triton：每行一个 Program，Grid：({args.rows:,},)，num_warps={args.num_warps}")
    elif args.provider == "triton-v1":
        print("Triton V1：Partial 与输出两阶段 Kernel，每行按 4,096 列切块")
    elif args.provider == "cuda":
        print(f"CUDA C++：每行一个 Block，Grid：({args.rows:,},)，Block：256 threads")
    else:
        print(f"CUDA Shuffle：每行一个 Block，Grid：({args.rows:,},)，Block：256 threads")
    print("正确性检查：通过")


if __name__ == "__main__":
    main()
