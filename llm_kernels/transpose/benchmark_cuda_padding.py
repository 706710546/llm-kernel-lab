"""比较 CUDA Tiled Transpose 有无共享内存 Padding 的性能。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.transpose.benchmark import effective_bandwidth_gbps, measure_ms
from llm_kernels.transpose.cuda_impl import (
    transpose_cuda_tiled,
    transpose_cuda_tiled_unpadded,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")

    dtype = torch.float32
    shapes = (
        (1024, 1024),
        (4096, 1024),
        (1024, 4096),
        (4096, 4096),
        (8192, 8192),
    )
    compile_input = torch.zeros((1, 1), device="cuda", dtype=dtype)
    transpose_cuda_tiled_unpadded(compile_input)
    transpose_cuda_tiled(compile_input)
    torch.cuda.synchronize()

    print("padding,M,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps")
    for padding, implementation in (
        (0, transpose_cuda_tiled_unpadded),
        (1, transpose_cuda_tiled),
    ):
        for n_rows, n_columns in shapes:
            x = torch.randn((n_rows, n_columns), device="cuda", dtype=dtype)
            function = lambda: implementation(x)
            p50_ms, p20_ms, p80_ms = measure_ms(function)
            bandwidth = effective_bandwidth_gbps(
                n_rows,
                n_columns,
                x.element_size(),
                p50_ms,
            )
            print(
                f"{padding},{n_rows},{n_columns},{p50_ms * 1000:.3f},"
                f"{p20_ms * 1000:.3f},{p80_ms * 1000:.3f},{bandwidth:.2f}"
            )


if __name__ == "__main__":
    main()
