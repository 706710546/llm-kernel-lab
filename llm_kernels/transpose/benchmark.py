"""测量 PyTorch、Triton 和 CUDA Matrix Transpose 的延迟与有效带宽。"""

from pathlib import Path
import sys

import torch
import triton


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.transpose.cuda_impl import transpose_cuda_naive, transpose_cuda_tiled
from llm_kernels.transpose.torch_impl import transpose_torch
from llm_kernels.transpose.triton_impl import transpose_triton


WARMUP_MS = 25
REPEAT_MS = 100


def effective_bandwidth_gbps(
    n_rows: int,
    n_columns: int,
    element_size: int,
    latency_ms: float,
) -> float:
    """按照一次读取和一次写入计算逻辑数据搬运量。"""
    bytes_moved = 2 * n_rows * n_columns * element_size
    return bytes_moved / (latency_ms * 1e-3) / 1e9


def measure_ms(function) -> tuple[float, float, float]:
    """返回以毫秒为单位的 P50、P20 和 P80 延迟。"""
    p50, p20, p80 = triton.testing.do_bench(
        function,
        warmup=WARMUP_MS,
        rep=REPEAT_MS,
        quantiles=[0.5, 0.2, 0.8],
    )
    return float(p50), float(p20), float(p80)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")

    dtype = torch.float32
    shapes = (
        (256, 256),
        (1024, 1024),
        (4096, 1024),
        (1024, 4096),
        (4096, 4096),
        (8192, 8192),
    )

    compile_input = torch.zeros((1, 1), device="cuda", dtype=dtype)
    transpose_cuda_naive(compile_input)
    transpose_cuda_tiled(compile_input)
    torch.cuda.synchronize()

    print("provider,M,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps")

    for n_rows, n_columns in shapes:
        x = torch.randn((n_rows, n_columns), device="cuda", dtype=dtype)
        providers = {
            "pytorch": lambda: transpose_torch(x),
            "triton": lambda: transpose_triton(x),
            "cuda_naive": lambda: transpose_cuda_naive(x),
            "cuda_tiled": lambda: transpose_cuda_tiled(x),
        }
        for provider, function in providers.items():
            p50_ms, p20_ms, p80_ms = measure_ms(function)
            bandwidth = effective_bandwidth_gbps(
                n_rows,
                n_columns,
                x.element_size(),
                p50_ms,
            )
            print(
                f"{provider},{n_rows},{n_columns},{p50_ms * 1000:.3f},"
                f"{p20_ms * 1000:.3f},{p80_ms * 1000:.3f},{bandwidth:.2f}"
            )


if __name__ == "__main__":
    main()
