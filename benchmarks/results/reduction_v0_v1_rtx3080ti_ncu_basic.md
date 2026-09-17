# Row Sum Reduction：V0 / V1 的 Nsight Compute 分析

## 实验信息

| 项目 | 数值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 3080 Ti |
| Compute Capability | 8.6 |
| 数据类型 | FP32 |
| 输入形状 | `[256, 65,536]` |
| 输入元素数量 | 16,777,216（`2^24`） |
| 输入数据量 | 64 MiB |
| SM 数量 | 80 |
| V1 分块大小 | 1,024 个元素 |
| V1 局部和矩阵 | `[256, 64]`，共 64 KiB（FP32） |

V0 为“一行一个 Program”的单 kernel 实现。V1 有两个 kernel：Stage 1 将每行拆为
64 个块并写入局部和；Stage 2 合并每行的 64 个局部和。

分析命令：

```powershell
ncu --set basic --kernel-name regex:row_sum_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/reduction/profile.py --version v0

ncu --set basic --kernel-name regex:row_sum_stage1_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/reduction/profile.py --version v1

ncu --set basic --kernel-name regex:row_sum_stage2_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/reduction/profile.py --version v1
```

每条命令都先执行 10 次预热，NCU 跳过预热后捕获一次目标 kernel。运行
`profile.py` 末尾还会与 PyTorch 参考实现进行正确性比较。

## 关键结果

| 指标 | V0 | V1 Stage 1 | V1 Stage 2 |
|---|---:|---:|---:|
| Duration | 84.03 μs | 84.10 μs | 3.58 μs |
| Memory Throughput | 92.85% | 92.80% | 6.55% |
| DRAM Throughput | 92.85% | 92.80% | 2.33% |
| Compute (SM) Throughput | 3.62% | 18.34% | 6.55% |
| Grid Size | 256 | 16,384 | 256 |
| CUDA Block Size | 128 threads | 128 threads | 128 threads |
| Registers Per Thread | 93 | 18 | 16 |
| Theoretical Occupancy | 41.67% | 100% | 100% |
| Achieved Occupancy | 26.32% | 94.07% | 23.93% |
| Waves Per SM | 0.64 | 17.07 | 0.27 |

V1 两个阶段的 kernel 时间相加为：

```text
84.10 μs + 3.58 μs = 87.68 μs
```

独立 Benchmark 对同一形状测得 V0 约 87.04 μs、V1 约 90.11 μs。NCU 捕获与独立
Benchmark 的数值接近；两者之间的少量差异来自 profiling 的多次采集、GPU 时钟和测量
环境，并不改变结论。

## V0：Occupancy 较低，但不是当前瓶颈

V0 使用 256 个 CUDA Block。每个 Block 有 128 个线程，即 4 个 Warp。80 个 SM 平均只
分到约 3.2 个 Block，因此实际活跃 Warp 数大约为：

```text
3.2 Block / SM × 4 Warp / Block ≈ 12.8 Warp / SM
```

这与 NCU 测得的 `12.63 Achieved Active Warps Per SM`、`26.32% Achieved Occupancy`
一致。V0 的 93 个寄存器/线程还将理论 Occupancy 限制为 41.67%，但目前更直接的限制是
Grid 总数只有 256，无法让所有 SM 长时间保持满载。

这并不构成“立刻优化 Occupancy”的理由：V0 的 DRAM Throughput 已经达到 92.85%，而
Compute Throughput 只有 3.62%。它已经接近显存带宽上限，额外增加活跃 Warp 很可能不会
提供更多可用的 DRAM 带宽。

## V1 Stage 1：更高 Occupancy 没有缩短核心时间

V1 Stage 1 的 Grid 为：

```text
256 行 × 64 块 / 行 = 16,384 Programs
```

其寄存器使用仅为 18 个/线程，Achieved Occupancy 达到 94.07%。不过它的耗时为 84.10 μs，
与 V0 的 84.03 μs 几乎相同；两者的 DRAM Throughput 也都约为 92.8%。

这证明 Stage 1 与 V0 都已是 Memory-Bound：V1 增加并行度能够改善 Occupancy，却不能突破
已经饱和的显存带宽。

## V1 Stage 2：低效率，但绝对成本很小

Stage 2 只读取 `[256, 64]` 的 `partials`，总共 64 KiB 数据，却只启动 256 个 CUDA Block。
因此它只有 0.27 Waves Per SM，Achieved Occupancy 为 23.93%，DRAM Throughput 也只有
2.33%。

它的 GPU 利用率并不高，但耗时仅 3.58 μs。这里的关键是区分“相对利用率”和“绝对耗时”：
Stage 2 的工作量非常小，优化它不是当前最高优先级。

## 性能结论与派发策略

1. V0 与 V1 Stage 1 都把原始输入读取的 DRAM 带宽推到约 93%。
2. V1 没有让核心读取阶段更快；它的价值是突破 V0 每行最多 65,536 个元素的边界。
3. V1 的额外 Stage 2 带来约 3～4 μs 固定成本，因此在 V0 支持的行宽内通常略慢。
4. NCU 的自动 Occupancy 建议是排查线索，而不是优化结论；必须结合 DRAM Throughput 和
   实测时间判断。

当前推荐的实现选择为：

```text
N ≤ 65,536   → 使用 V0
N > 65,536   → 使用 V1
```

未来 V2 可以研究不同的分块大小、自动调参和更大行宽的递归多阶段归约，但不应以“提高
Occupancy”本身作为目标。
