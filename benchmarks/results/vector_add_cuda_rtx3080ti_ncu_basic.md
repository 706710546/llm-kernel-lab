# CUDA Vector Add：Nsight Compute 基础分析

## 实验信息

| 项目 | 数值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 3080 Ti |
| Compute Capability | 8.6 |
| 数据类型 | FP32 |
| 元素数量 | 16,777,216（`2^24`） |
| CUDA Threads / Block | 256 |
| CUDA Block 数量 | 65,536 |
| SM 数量 | 80 |
| Nsight Compute | 2024.3.2 |

分析命令：

```powershell
ncu --set basic --kernel-name regex:^vector_add_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/vector_add/profile_cuda.py
```

`profile_cuda.py` 先执行 10 次预热。Nsight Compute 跳过这些调用，只分析第 11 次
`vector_add_kernel`。

## CUDA Kernel 关键结果

| 指标 | 测量值 |
|---|---:|
| Duration | 246.75 μs |
| Memory Throughput | 91.86% |
| DRAM Throughput | 91.86% |
| Compute (SM) Throughput | 15.99% |
| L1/TEX Cache Throughput | 20.75% |
| L2 Cache Throughput | 39.37% |
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 78.36% |
| Achieved Active Warps / SM | 37.61 |
| Registers / Thread | 16 |
| Static/Dynamic Shared Memory | 0 B |
| Waves / SM | 136.53 |

## 与有效带宽交叉验证

FP32 Vector Add 对每个元素读取两个输入并写入一个输出：

```text
逻辑数据量 = 16,777,216 × 12 Byte
           = 201,326,592 Byte

有效带宽 = 201,326,592 Byte ÷ 246.75 μs
         ≈ 815.91 GB/s
```

独立 Benchmark 在同一尺寸上测得 CUDA 有效带宽 `815.80 GB/s`，两种测量方法几乎完全
吻合，说明计时方法和逻辑数据量公式是可信的。

## CUDA 与 Triton 对照

| 指标 | Triton | CUDA C++ |
|---|---:|---:|
| Duration | 247.10 μs | 246.75 μs |
| DRAM Throughput | 91.67% | 91.86% |
| Compute (SM) Throughput | 7.53% | 15.99% |
| Achieved Occupancy | 86.10% | 78.36% |
| Registers / Thread | 16 | 16 |
| 底层 CUDA Block Size | 128 | 256 |

两者的延迟只相差 `0.35 μs`，约为 `0.14%`，不能据此声称某一种实现稳定更快。CUDA
版本的 Compute Throughput 较高、Achieved Occupancy 反而较低，但最终 DRAM Throughput
和运行时间基本相同。这说明本算子的决定性瓶颈是显存，而不是计算单元或 Occupancy。

Nsight Compute 给出的 Occupancy 优化提示是通用诊断，不等于实际一定能获得提示中的
加速比例。当前 DRAM Throughput 已超过 91%，即使把 Achieved Occupancy 提升到 100%，
也未必还有足够的显存带宽可供利用。

## 本阶段结论

1. 手写 CUDA、Triton 和 PyTorch 在大尺寸连续 Vector Add 上进入同一显存带宽平台。
2. `256 Threads / Block` 的朴素 CUDA 映射已经能达到约 `816 GB/s` 有效带宽。
3. Triton 的 `BLOCK_SIZE=256` 最终编译为 128 个 CUDA Threads，而手写 CUDA 明确使用
   256 个 Threads；两者映射不同，但性能几乎相同。
4. “手写 CUDA 一定更快”不是可靠结论；应该用 Benchmark 和 Profiler 数据判断。

## Threads / Block 对比实验

最终版本把 CUDA Launcher 的 Threads / Block 改为运行时参数，并在相同的 `N=2^24`
输入上依次采集 `128`、`256`、`512` 三种配置：

| Threads / Block | Grid Size | Duration | DRAM Throughput | Compute Throughput | Achieved Occupancy |
|---:|---:|---:|---:|---:|---:|
| 128 | 131,072 | 247.42 μs | 91.73% | 15.98% | 80.37% |
| 256 | 65,536 | 249.60 μs | 91.29% | 15.87% | 76.54% |
| 512 | 32,768 | 247.94 μs | 91.96% | 15.63% | 66.70% |

改变 Threads / Block 会反向改变 Grid Size，但总线程数始终是 16,777,216。三种配置的
Duration 最大相差 2.18 μs，约 0.88%；DRAM Throughput 全部超过 91%。

Achieved Occupancy 随线程块增大而下降，但 Kernel 时间并没有按相同比例增加。这再次
说明 Occupancy 是隐藏延迟的手段，不是最终性能目标。当前三种配置都已经提供足够多的
活跃 Warp 来覆盖显存访问延迟，真正限制性能的是 DRAM 带宽。

默认选择 256 Threads / Block，理由是它在 CUDA 工程中常见、映射直观，并在本实验中
达到带宽平台。128 或 512 也都是有效配置；不应根据亚微秒级的单次差异宣称某个配置
绝对最优。
