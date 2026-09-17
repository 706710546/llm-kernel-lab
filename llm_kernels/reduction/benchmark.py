"""测量 PyTorch 和 Triton Row Sum 的延迟与有效带宽。"""

from pathlib import Path
import sys

import torch
import triton


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.reduction.torch_impl import row_sum_torch
from llm_kernels.reduction.triton_impl import MAX_BLOCK_SIZE, row_sum_triton
from llm_kernels.reduction.triton_v1_impl import row_sum_triton_v1


WARMUP_MS = 25
REPEAT_MS = 100
BENCHMARK_SHAPES = (
    (8_192, 128),
    (8_192, 512),
    (8_192, 2_048),
    (8_192, 8_192),
    (256, 65_536),
    (128, 131_072),
)


def effective_bandwidth_gbps(
    rows: int,
    columns: int,
    input_element_size: int,
    output_element_size: int,
    latency_ms: float,
) -> float:
    """按一次读取输入矩阵、一次写入行和计算逻辑数据搬运量。"""
    bytes_moved = rows * columns * input_element_size + rows * output_element_size
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
    print("provider,M,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps")

    for rows, columns in BENCHMARK_SHAPES:
        x = torch.randn((rows, columns), device="cuda", dtype=dtype)

        providers = {
            "pytorch": lambda: row_sum_torch(x),
            "triton_v1": lambda: row_sum_triton_v1(x),
        }
        if columns <= MAX_BLOCK_SIZE:
            providers["triton_v0"] = lambda: row_sum_triton(x)

        for provider, function in providers.items():
            p50_ms, p20_ms, p80_ms = measure_ms(function)
            bandwidth = effective_bandwidth_gbps(
                rows,
                columns,
                x.element_size(),
                x.element_size(),
                p50_ms,
            )
            print(
                f"{provider},{rows},{columns},{p50_ms * 1000:.3f},"
                f"{p20_ms * 1000:.3f},{p80_ms * 1000:.3f},{bandwidth:.2f}"
            )


if __name__ == "__main__":
    main()
