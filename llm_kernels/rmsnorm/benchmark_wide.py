"""比较超宽行 Triton V1 两阶段归约与 CUDA C++ 单 Block/行实现。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.rmsnorm.benchmark import logical_bandwidth_gbps, measure_ms
from llm_kernels.rmsnorm.torch_impl import rmsnorm_torch
from llm_kernels.rmsnorm.triton_v1_impl import rmsnorm_triton_v1
from llm_kernels.rmsnorm.cuda_impl import rmsnorm_cuda, rmsnorm_cuda_shuffle


WIDE_SHAPES = ((256, 16_385), (256, 32_769), (256, 65_537))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")

    print("provider,dtype,M,N,p50_us,p20_us,p80_us,logical_bandwidth_GBps")
    dtype = torch.float32
    for rows, columns in WIDE_SHAPES:
        x = torch.randn((rows, columns), device="cuda", dtype=dtype)
        weight = torch.randn((columns,), device="cuda", dtype=dtype)
        # Trigger the CUDA Extension build before timing.
        rmsnorm_cuda(x, weight)
        rmsnorm_cuda_shuffle(x, weight)
        providers = {
            "pytorch": lambda: rmsnorm_torch(x, weight),
            "triton_v1_two_stage": lambda: rmsnorm_triton_v1(x, weight),
            "cuda_shared_256": lambda: rmsnorm_cuda(x, weight),
            "cuda_shuffle_256": lambda: rmsnorm_cuda_shuffle(x, weight),
        }
        expected = rmsnorm_torch(x, weight)
        for provider, function in providers.items():
            output = function()
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
            p50_ms, p20_ms, p80_ms = measure_ms(function)
            bandwidth = logical_bandwidth_gbps(
                rows, columns, x.element_size(), p50_ms
            )
            print(
                f"{provider},float32,{rows},{columns},{p50_ms * 1000:.3f},"
                f"{p20_ms * 1000:.3f},{p80_ms * 1000:.3f},{bandwidth:.2f}"
            )


if __name__ == "__main__":
    main()
