# RMSNorm CUDA Warp Shuffle：NCU Basic

## 实验条件

- GPU：NVIDIA GeForce RTX 3080 Ti，CC 8.6
- 输入：FP32，`M=1024, N=4096`
- Kernel：`rmsnorm_shuffle_kernel`，Grid `(1024,)`，Block `(256,)`
- 每个 Warp 用 `__shfl_down_sync` 做 Warp 内归约；8 个 Warp 的部分和通过 32 B 共享内存合并
- 命令：

```powershell
ncu --set basic --kernel-name regex:rmsnorm_shuffle_kernel --launch-skip 10 --launch-count 1 python llm_kernels/rmsnorm/profile.py --rows 1024 --columns 4096 --provider cuda-shuffle
```

## 采样结果

| 指标 | Shuffle 版 | 上一阶段共享内存版 |
|---|---:|---:|
| NCU Duration | 60.03 μs | 60.96 μs |
| DRAM Throughput | 82.56% | 81.69% |
| Compute (SM) Throughput | 19.74% | 24.27% |
| Registers / Thread | 40 | 40 |
| Static Shared Memory / Block | 32 B | 1,024 B |
| Theoretical / Achieved Occupancy | 100% / 86.67% | 100% / 87.91% |

NCU Duration 由 profiler replay 测得，不可与普通 Benchmark P50 混为一谈。普通 Benchmark
在 `1024×4096` 上两个 CUDA 版本的 P50 均约 60.416 μs；NCU 差异也不足以证明 Shuffle
版更快。

## 结论

这次优化达成了资源目标：显式共享内存由每 Block 1,024 B 减少到 32 B，并把树形归约中
多次 Block 级同步改为两次跨 Warp 汇合所需的同步。但五组形状的普通 Benchmark 几乎持平，
说明“同步更少 / 共享内存更少”并不自动等于端到端变快。该形状的 DRAM Throughput 已超过
82%，Kernel 还要做两遍输入读取及输出写入；这可能限制归约改写带来的收益。后续应使用更
短行、更大 M/N、CUDA Event 重复采样或更精确的指标来继续定位，而不是把本轮微小波动说成
性能提升。
