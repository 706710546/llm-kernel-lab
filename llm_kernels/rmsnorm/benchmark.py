"""比较 PyTorch、Triton V0 和 CUDA RMSNorm 的延迟与逻辑有效带宽。"""

from pathlib import Path
import sys

import torch
import triton


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.rmsnorm.torch_impl import rmsnorm_torch
from llm_kernels.rmsnorm.triton_impl import rmsnorm_triton
from llm_kernels.rmsnorm.cuda_impl import rmsnorm_cuda, rmsnorm_cuda_shuffle


WARMUP_MS = 25
REPEAT_MS = 100
BENCHMARK_SHAPES = (
    (1, 4_096),
    (32, 4_096),
    (256, 4_096),
    (1_024, 4_096),
    (512, 8_192),
)


def measure_ms(function) -> tuple[float, float, float]:
    """返回以毫秒为单位的 P50、P20 和 P80 延迟。"""
    p50, p20, p80 = triton.testing.do_bench(
        function,
        warmup=WARMUP_MS,
        rep=REPEAT_MS,
        quantiles=[0.5, 0.2, 0.8],
    )
    return float(p50), float(p20), float(p80)


def logical_bandwidth_gbps(
    rows: int,
    columns: int,
    element_size: int,
    latency_ms: float,
) -> float:
    """按最低逻辑流量（一遍读 x、一遍读 weight、一遍写 y）估算吞吐。"""
    bytes_moved = (2 * rows * columns + columns) * element_size
    return bytes_moved / (latency_ms * 1e-3) / 1e9


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")

    print("provider,dtype,M,N,p50_us,p20_us,p80_us,logical_bandwidth_GBps")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        dtype_name = str(dtype).removeprefix("torch.")
        for rows, columns in BENCHMARK_SHAPES:
            x = torch.randn((rows, columns), device="cuda", dtype=dtype)
            weight = torch.randn((columns,), device="cuda", dtype=dtype)
            # 首次调用会编译 Extension；先触发编译，避免把构建时间纳入计时。
            rmsnorm_cuda(x, weight)
            rmsnorm_cuda_shuffle(x, weight)
            providers = {
                "pytorch": lambda: rmsnorm_torch(x, weight),
                "triton_v0_w4": lambda: rmsnorm_triton(x, weight, num_warps=4),
                "triton_v0_w8": lambda: rmsnorm_triton(x, weight, num_warps=8),
                "cuda_shared_256": lambda: rmsnorm_cuda(x, weight),
                "cuda_shuffle_256": lambda: rmsnorm_cuda_shuffle(x, weight),
            }
            for provider, function in providers.items():
                p50_ms, p20_ms, p80_ms = measure_ms(function)
                bandwidth = logical_bandwidth_gbps(
                    rows,
                    columns,
                    x.element_size(),
                    p50_ms,
                )
                print(
                    f"{provider},{dtype_name},{rows},{columns},{p50_ms * 1000:.3f},"
                    f"{p20_ms * 1000:.3f},{p80_ms * 1000:.3f},{bandwidth:.2f}"
                )


if __name__ == "__main__":
    main()
