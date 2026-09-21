# Matrix Transpose V0：Nsight Compute 基础分析

## 实验信息

| 项目 | 数值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 3080 Ti |
| Compute Capability | 8.6 |
| 数据类型 | FP32 |
| 输入形状 | 4,096 × 4,096 |
| Triton Tile | 32 × 32 |
| Triton Grid | 128 × 128 |
| 底层 CUDA Block Size | 128 Threads |

分析命令：

```powershell
ncu --set basic --kernel-name regex:transpose_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/transpose/profile.py
```

## 关键指标

| 指标 | 测量值 |
|---|---:|
| Duration | 168.86 μs |
| Memory Throughput | 89.22% |
| DRAM Throughput | 89.22% |
| Compute (SM) Throughput | 15.36% |
| L1/TEX Cache Throughput | 29.26% |
| L2 Cache Throughput | 38.37% |
| Registers / Thread | 25 |
| Dynamic Shared Memory / Block | 4.10 KB |
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 93.44% |
| Waves / SM | 17.07 |

## 与 Benchmark 交叉验证

输入包含 `4,096 × 4,096` 个 FP32 元素。每个元素读取一次、写入一次：

```text
逻辑数据量 = 4,096 × 4,096 × 4 Byte × 2
           = 134,217,728 Byte

有效带宽 = 134,217,728 Byte ÷ 168.86 μs
         ≈ 794.85 GB/s
```

独立 Benchmark 测得 `775.57 GB/s`，与 NCU 结果处于同一范围。Profiler 会重放 Kernel
多个 Pass，因此应关注数量级和瓶颈判断，而不是要求两个工具给出完全相同的微秒数。

## 为什么出现 4.10 KB 动态共享内存？

源码没有显式编写共享内存，但 NCU 显示每个 CUDA Block 使用约 `4.10 KB`：

```text
32 × 32 × 4 Byte = 4,096 Byte ≈ 4.10 KB
```

这与一个 FP32 Tile 的大小一致，表明 Triton 编译器为 `tl.trans` 生成了共享内存中转。
线程先以合并方式读取输入 Tile，在片上存储中完成访问方向转换，再以合并方式写出转置
Tile。这正是经典 CUDA Tiled Transpose 的核心思想。

## 性能结论

DRAM Throughput 为 89.22%，明显高于 15.36% 的 Compute Throughput，说明 V0 主要受
显存系统限制。它没有浮点运算，所谓 Compute Throughput 主要来自地址计算、控制指令和
数据搬运相关指令，不能理解为算术单元在做大量矩阵计算。

Achieved Occupancy 达到 93.44%，说明当前寄存器和共享内存使用没有严重限制并发。但
Occupancy 已经足够高，下一步更值得比较的是 16×16 与 32×32 Tile 对访问效率和启动
数量的影响，而不是继续追求 100% Occupancy。

## 16×16 与 32×32 Tile 对比

相同的 `4096×4096` FP32 输入：

| 指标 | 16×16 | 32×32 |
|---|---:|---:|
| Triton Grid | 256×256 | 128×128 |
| Program 数量 | 65,536 | 16,384 |
| 底层 CUDA Block Size | 128 | 128 |
| Duration | 218.59 μs | 168.86 μs |
| DRAM Throughput | 69.70% | 89.22% |
| Compute Throughput | 16.73% | 15.36% |
| Dynamic Shared Memory / Block | 1.02 KB | 4.10 KB |
| Registers / Thread | 16 | 25 |
| Achieved Occupancy | 93.97% | 93.44% |

16×16 和 32×32 最终都被 Triton 编译为每个 CUDA Block 128 Threads，Occupancy 也几乎
相同，因此性能差距不能用线程数或 Occupancy 解释。

32×32 每个 Program 搬运的元素数量是 16×16 的四倍，使 Program 数量减少到四分之一，
并把 DRAM Throughput 从 69.70% 提高到 89.22%。虽然它使用更多共享内存和寄存器，但
没有降低实际 Occupancy，最终速度提高约 29.5%。因此 V0 默认选择 32×32 Tile。
