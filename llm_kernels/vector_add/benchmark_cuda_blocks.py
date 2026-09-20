"""比较不同 CUDA Threads / Block 配置的 Vector Add 性能。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.vector_add.benchmark import effective_bandwidth_gbps, measure_ms
from llm_kernels.vector_add.cuda_impl import vector_add_cuda


THREAD_COUNTS = (128, 256, 512)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")

    dtype = torch.float32
    sizes = [2**power for power in range(10, 27, 2)]

    compile_input = torch.zeros(1, device="cuda", dtype=dtype)
    vector_add_cuda(compile_input, compile_input)
    torch.cuda.synchronize()

    print("threads_per_block,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps")
    for threads_per_block in THREAD_COUNTS:
        for size in sizes:
            x = torch.randn(size, device="cuda", dtype=dtype)
            y = torch.randn(size, device="cuda", dtype=dtype)
            function = lambda: vector_add_cuda(
                x,
                y,
                threads_per_block=threads_per_block,
            )
            p50_ms, p20_ms, p80_ms = measure_ms(function)
            bandwidth = effective_bandwidth_gbps(size, x.element_size(), p50_ms)
            print(
                f"{threads_per_block},{size},{p50_ms * 1000:.3f},"
                f"{p20_ms * 1000:.3f},{p80_ms * 1000:.3f},{bandwidth:.2f}"
            )


if __name__ == "__main__":
    main()
