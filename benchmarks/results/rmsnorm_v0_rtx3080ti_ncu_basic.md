# RMSNorm Triton V0：NCU Basic 对照

采集日期：2026-10-13。GPU：NVIDIA GeForce RTX 3080 Ti，Compute Capability 8.6。
输入：FP32，`M=1024`，`N=4096`；Grid 为 1024 个 Program。分别使用 4 和 8 warps。

## 采集命令

```powershell
ncu --set basic --kernel-name regex:rmsnorm_kernel --launch-skip 10 --launch-count 1 python llm_kernels/rmsnorm/profile.py --rows 1024 --columns 4096 --num-warps 4
ncu --set basic --kernel-name regex:rmsnorm_kernel --launch-skip 10 --launch-count 1 python llm_kernels/rmsnorm/profile.py --rows 1024 --columns 4096 --num-warps 8
```

## 结果

| 指标 | 4 warps | 8 warps |
|---|---:|---:|
| Threads / Block | 128 | 256 |
| Registers / Thread | 80 | 48 |
| Theoretical Occupancy | 50.00% | 83.33% |
| Achieved Active Warps / SM | 20.90 | 34.08 |
| DRAM Throughput | 86.61% | 83.88% |
| L1/TEX Cache Throughput | 27.43% | 28.01% |
| L2 Cache Throughput | 39.77% | 40.25% |
| Compute (SM) Throughput | 6.23% | 8.62% |
| NCU Duration | 42.94 μs | 44.99 μs |

两种配置的 NCU Duration 不是普通 Benchmark 的 P50：NCU 为收集指标会重放 Kernel，
因此应将这些数字作为单次采样对照，而非直接替代 `benchmark.py` 数据。普通 Benchmark
中这一形状两种配置的 P50 都约为 45 μs。

## 如何解释

- DRAM Throughput 达到约 84%～87%，而 SM 计算吞吐只有约 6%～9%。对这组形状和实现，
  瓶颈更接近内存通路，而不是算术吞吐。
- 从 4 增加到 8 warps 后，线程数翻倍、寄存器/线程由 80 降到 48，理论占用率由 50% 升至
  83.33%。这验证了 `num_warps` 会改变每个 CTA 的线程组织和寄存器分摊。
- 更高的占用率并没有带来 Benchmark 延迟改善。DRAM 吞吐也没有继续上升，说明增加可驻留
  warp 不能消除当前主要的内存瓶颈。不能因此断言 8 warps 对其他 M/N 也无效。
- 本版只报告 `--set basic` 范围的指标。没有采集或分析 Local Load/Store、具体 Memory
  Workload、Spill 源码行等更细粒度诊断；若进一步优化，应围绕一个具体假设扩展指标集。

## 局限

这是单一 shape、单一 GPU、单次 profiler 捕获。它不足以建立自动调优规则，也不足以证明
所有 RMSNorm 工作负载都受 DRAM 带宽限制。下一步应在多个 M/N 上比较 4/8 warps，并在需要时
用更详细的 NCU 集合验证 local-memory spill 与访存行为。
