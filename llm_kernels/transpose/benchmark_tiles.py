"""比较不同 Triton Tile 大小的 Matrix Transpose 性能。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.transpose.benchmark import effective_bandwidth_gbps, measure_ms
from llm_kernels.transpose.triton_impl import transpose_triton


BLOCK_SIZES = (16, 32)


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
    print("block_size,M,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps")

    for block_size in BLOCK_SIZES:
        for n_rows, n_columns in shapes:
            x = torch.randn((n_rows, n_columns), device="cuda", dtype=dtype)
            function = lambda: transpose_triton(x, block_size=block_size)
            p50_ms, p20_ms, p80_ms = measure_ms(function)
            bandwidth = effective_bandwidth_gbps(
                n_rows,
                n_columns,
                x.element_size(),
                p50_ms,
            )
            print(
                f"{block_size},{n_rows},{n_columns},{p50_ms * 1000:.3f},"
                f"{p20_ms * 1000:.3f},{p80_ms * 1000:.3f},{bandwidth:.2f}"
            )


if __name__ == "__main__":
    main()
