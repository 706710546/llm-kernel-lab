"""测量 PyTorch、Triton Softmax V0/V1/V2 的延迟与有效带宽。"""

from pathlib import Path
import sys

import torch
import triton


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.softmax.torch_impl import softmax_torch
from llm_kernels.softmax.cuda_impl import softmax_cuda
from llm_kernels.softmax.triton_impl import MAX_BLOCK_SIZE, softmax_triton
from llm_kernels.softmax.triton_v1_impl import softmax_triton_v1
from llm_kernels.softmax.triton_v2_impl import softmax_triton_v2


WARMUP_MS = 25
REPEAT_MS = 100
BENCHMARK_SHAPES = (
    (4_096, 128),
    (4_096, 512),
    (4_096, 2_048),
    (4_096, 8_192),
    (2_048, 16_384),
    (1_024, 32_768),
    (512, 131_072),
)


def effective_bandwidth_gbps(
    rows: int,
    columns: int,
    element_size: int,
    latency_ms: float,
) -> float:
    """按一次读取和一次写入计算最低逻辑数据搬运量。"""
    bytes_moved = 2 * rows * columns * element_size
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
    # 把首次 Extension 编译排除在计时范围之外。
    softmax_cuda(torch.zeros((1, 1), device="cuda", dtype=dtype))
    torch.cuda.synchronize()
    print("provider,M,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps")
    for rows, columns in BENCHMARK_SHAPES:
        x = torch.randn((rows, columns), device="cuda", dtype=dtype)
        providers = {
            "pytorch": lambda: softmax_torch(x),
            "triton_v1": lambda: softmax_triton_v1(x),
            "triton_v2": lambda: softmax_triton_v2(x),
            "cuda": lambda: softmax_cuda(x),
        }
        if columns <= MAX_BLOCK_SIZE:
            providers["triton_v0"] = lambda: softmax_triton(x)
        for provider, function in providers.items():
            p50_ms, p20_ms, p80_ms = measure_ms(function)
            bandwidth = effective_bandwidth_gbps(
                rows,
                columns,
                x.element_size(),
                p50_ms,
            )
            print(
                f"{provider},{rows},{columns},{p50_ms * 1000:.3f},"
                f"{p20_ms * 1000:.3f},{p80_ms * 1000:.3f},{bandwidth:.2f}"
            )


if __name__ == "__main__":
    main()
