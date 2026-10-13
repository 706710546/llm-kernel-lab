# RMSNorm Triton V1 超宽行：NCU Basic

## 实验条件

- GPU：NVIDIA GeForce RTX 3080 Ti，CC 8.6
- 输入：FP32，`M=256, N=32769`，每行分为 9 个 4,096 列块
- Stage 1 Grid：`(9, 256)`，每个 Program 产出一个 FP32 局部平方和
- Stage 2 Grid：`(9, 256)`，每个 Program 合并 9 个 partial，再处理一个输出块
- 两个命令分别采集两个阶段（profile 每次预热并运行整个 V1）：

```powershell
ncu --set basic --kernel-name regex:rmsnorm_partial_kernel --launch-skip 10 --launch-count 1 python llm_kernels/rmsnorm/profile.py --rows 256 --columns 32769 --provider triton-v1
ncu --set basic --kernel-name regex:rmsnorm_output_kernel --launch-skip 10 --launch-count 1 python llm_kernels/rmsnorm/profile.py --rows 256 --columns 32769 --provider triton-v1
```

## NCU 结果

| 指标 | Stage 1 Partial | Stage 2 Output |
|---|---:|---:|
| NCU Duration | 45.92 μs | 83.74 μs |
| DRAM Throughput | 88.71% | 90.05% |
| Compute (SM) Throughput | 17.45% | 25.31% |
| Threads / Program | 128 | 128 |
| Registers / Thread | 39 | 80 |
| Dynamic Shared Memory | 16 B | 2.05 KB |
| Theoretical / Achieved Occupancy | 100% / 90.74% | 50% / 46.86% |

两个 NCU Duration 相加约 129.66 μs，与普通 Benchmark 的 V1 P50 127.984 μs 同一量级，
但 profiler replay 数据不应与普通计时直接混用。Stage 2 的 80 registers/thread 将理论
Occupancy 限制为 50%；它既要合并 partial，也要读取原输入和 weight 并写输出。虽然逻辑
Bandwidth 和 DRAM Throughput 都高，本结果仍只针对这个形状，不能推导所有宽度的瓶颈。

## 性能对照

普通 Benchmark 的 `M=256` 结果：

| N | Triton V1 P50 | CUDA Shuffle P50 | V1 相对 CUDA Shuffle |
|---:|---:|---:|---:|
| 16,385 | 68.608 μs | 74.752 μs | 1.09× 快 |
| 32,769 | 127.984 μs | 151.552 μs | 1.18× 快 |
| 65,537 | 247.808 μs | 321.536 μs | 1.30× 快 |

该结果说明 Triton V1 的 tile-parallel 映射在这三组宽行上较合适。其代价是多启动一次
Kernel，并在每个输出 tile 重读少量 partials；当前 chunk 数较少（5、9、17），因此这个
重复开销可控。完整原始数据见 [`宽行 Benchmark CSV`](rmsnorm_v1_wide_rtx3080ti_fp32.csv)。
