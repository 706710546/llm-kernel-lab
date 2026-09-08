"""Benchmark PyTorch and Triton Vector Add latency and effective bandwidth."""

from pathlib import Path
import sys

import torch
import triton


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.vector_add.torch_impl import vector_add_torch
from llm_kernels.vector_add.triton_impl import vector_add_triton


WARMUP_MS = 25
REPEAT_MS = 100


def effective_bandwidth_gbps(n_elements: int, element_size: int, latency_ms: float) -> float:
    """Model two reads and one write: 3 * N * element_size bytes."""
    bytes_moved = 3 * n_elements * element_size
    return bytes_moved / (latency_ms * 1e-3) / 1e9


def measure_ms(function) -> tuple[float, float, float]:
    """Return the p50, p20, and p80 latency in milliseconds."""
    p50, p20, p80 = triton.testing.do_bench(
        function,
        warmup=WARMUP_MS,
        rep=REPEAT_MS,
        quantiles=[0.5, 0.2, 0.8],
    )
    return float(p50), float(p20), float(p80)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    dtype = torch.float32
    sizes = [2**power for power in range(10, 27, 2)]
    print("provider,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps")

    for size in sizes:
        x = torch.randn(size, device="cuda", dtype=dtype)
        y = torch.randn(size, device="cuda", dtype=dtype)

        providers = {
            "pytorch": lambda: vector_add_torch(x, y),
            "triton": lambda: vector_add_triton(x, y),
        }
        for provider, function in providers.items():
            p50_ms, p20_ms, p80_ms = measure_ms(function)
            bandwidth = effective_bandwidth_gbps(size, x.element_size(), p50_ms)
            print(
                f"{provider},{size},{p50_ms * 1000:.3f},"
                f"{p20_ms * 1000:.3f},{p80_ms * 1000:.3f},{bandwidth:.2f}"
            )


if __name__ == "__main__":
    main()

