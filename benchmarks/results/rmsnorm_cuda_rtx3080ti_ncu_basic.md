# RMSNorm CUDA C++ FP32 共享内存基线：NCU Basic

## 实验条件

- GPU：NVIDIA GeForce RTX 3080 Ti，CC 8.6
- 输入：FP32，`M=1024, N=4096`
- CUDA Kernel：`rmsnorm_shared_kernel`，Grid `(1024,)`，Block `(256,)`
- 每个 Block 用 1,024 B 静态共享内存保存 256 个线程的平方和部分结果
- 命令：

```powershell
ncu --set basic --kernel-name regex:rmsnorm_shared_kernel --launch-skip 10 --launch-count 1 python llm_kernels/rmsnorm/profile.py --rows 1024 --columns 4096 --provider cuda
```

## 采样结果

| 指标 | 结果 |
|---|---:|
| NCU Duration | 60.96 μs |
| DRAM Throughput | 81.69% |
| L1 / L2 Throughput | 26.05% / 36.98% |
| Compute (SM) Throughput | 24.27% |
| Block Size | 256 threads |
| Registers / Thread | 40 |
| Static Shared Memory / Block | 1,024 B |
| Theoretical / Achieved Occupancy | 100% / 87.91% |

NCU Duration 受到 profiler replay 影响，不能直接和普通 Benchmark P50 等同。相同形状普通
Benchmark 的 CUDA P50 是 60.416 μs，Triton 4-warps P50 是 45.056 μs。

## 怎么理解

这个最直观的 CUDA 版本有较高 DRAM Throughput，也有比 Triton 版本更高的 SM 活跃度，
但普通 Benchmark 仍比 Triton 慢约 34%。原因不矛盾：CUDA 版本显式进行 8 轮 Block
级共享内存归约，每一轮都有 `__syncthreads()`；Triton 编译器会根据布局生成自己的归约，
并可用 Warp 内指令减少 Block 级协作。高 DRAM 利用率不是唯一耗时来源。

结论只适用于此形状、设备和实现。下一步可以在保持算法与测试不变的条件下，将 CUDA
归约改成 Warp Shuffle + 少量共享内存，观察同步次数与性能是否改善；不要仅凭 Occupancy
断言更快。
