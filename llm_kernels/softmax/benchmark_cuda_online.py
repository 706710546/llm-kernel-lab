"""交替 provider 顺序，重复比较 CUDA Shuffle 与 Online 的短行/宽行。"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_kernels.softmax.benchmark import measure_ms, effective_bandwidth_gbps
from llm_kernels.softmax.cuda_impl import softmax_cuda_shuffle, softmax_cuda_online


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    torch.manual_seed(0)
    implementations = (("cuda_shuffle", softmax_cuda_shuffle), ("cuda_online", softmax_cuda_online))
    for _, implementation in implementations:
        implementation(torch.zeros((1, 1), device="cuda"))
    torch.cuda.synchronize()

    print("round,provider,M,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps")
    for rows, columns in ((4_096, 128), (1_024, 32_768), (512, 131_072)):
        x = torch.randn((rows, columns), device="cuda", dtype=torch.float32)
        for round_index in range(3):
            ordered = implementations if round_index % 2 == 0 else implementations[::-1]
            for provider, implementation in ordered:
                p50, p20, p80 = measure_ms(lambda: implementation(x))
                bandwidth = effective_bandwidth_gbps(rows, columns, x.element_size(), p50)
                print(
                    f"{round_index + 1},{provider},{rows},{columns},"
                    f"{p50 * 1000:.3f},{p20 * 1000:.3f},{p80 * 1000:.3f},{bandwidth:.2f}"
                )


if __name__ == "__main__":
    main()
