# RMSNorm CUDA FP16 Warp Shuffle：NCU Basic

## 条件和命令

- GPU：NVIDIA GeForce RTX 3080 Ti，CC 8.6
- 输入：FP16，`M=1024, N=4096`
- Kernel：`rmsnorm_shuffle_kernel<Half>`，Grid `(1024,)`，Block `(256,)`
- 命令：

```powershell
ncu --set basic --kernel-name regex:rmsnorm_shuffle_kernel --launch-skip 10 --launch-count 1 python llm_kernels/rmsnorm/profile.py --rows 1024 --columns 4096 --provider cuda-shuffle --dtype float16
```

## 结果

| 指标 | 数值 |
|---|---:|
| NCU Duration | 32.13 μs |
| DRAM Throughput | 59.04% |
| L1 / L2 Throughput | 45.91% / 29.32% |
| Compute (SM) Throughput | 38.34% |
| Registers / Thread | 31 |
| Static Shared Memory / Block | 32 B |
| Theoretical / Achieved Occupancy | 100% / 89.49% |

普通 Benchmark 的 FP16 P50 是 30.720 μs；NCU Duration 包含 profiler 采集影响，不能与
普通 P50 直接比较。NCU 将该 Kernel 标记为低于峰值 60% 的计算和内存吞吐，提示可能有
延迟因素；这不是单凭 Basic 报告就能定位根因的证据。更精确的原因需要进一步查看
Scheduler / Warp State 等指标。

FP16 相较 FP32 的普通 P50 约减半（`1024×4096` CUDA Shuffle：60.416 → 30.720 μs），
符合输入和输出元素字节数减半的预期方向，但本测试不证明延迟一定严格减半，也不能把
Bandwidth 数值当成实际 DRAM 字节数。
