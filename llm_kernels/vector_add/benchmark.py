"""测量 PyTorch、Triton 和 CUDA C++ Vector Add 的延迟与有效带宽。"""

from pathlib import Path
import sys

import torch
import triton


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.vector_add.cuda_impl import vector_add_cuda
from llm_kernels.vector_add.torch_impl import vector_add_torch
from llm_kernels.vector_add.triton_impl import vector_add_triton


WARMUP_MS = 25
REPEAT_MS = 100


def effective_bandwidth_gbps(n_elements: int, element_size: int, latency_ms: float) -> float:
    """按照两次读取、一次写入计算理论数据搬运量。"""
    bytes_moved = 3 * n_elements * element_size
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
    sizes = [2**power for power in range(10, 27, 2)]

    # CUDA Extension 第一次调用会触发 C++/CUDA 编译。先在计时区间外完成加载，
    # 避免把一次性的编译成本误认为 Kernel 执行时间。
    compile_input = torch.zeros(1, device="cuda", dtype=dtype)
    vector_add_cuda(compile_input, compile_input)
    torch.cuda.synchronize()

    print("provider,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps")

    for size in sizes:
        x = torch.randn(size, device="cuda", dtype=dtype)
        y = torch.randn(size, device="cuda", dtype=dtype)

        providers = {
            "pytorch": lambda: vector_add_torch(x, y),
            "triton": lambda: vector_add_triton(x, y),
            "cuda": lambda: vector_add_cuda(x, y),
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
