# CUDA Matrix Transpose：Naive 与 Tiled NCU 对照

## 实验配置

| 项目 | 数值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 3080 Ti |
| 数据类型 | FP32 |
| 输入形状 | 4,096 × 4,096 |
| CUDA Grid | 128 × 128 |
| CUDA Block | 32 × 8 = 256 Threads |
| 每个 Block 处理 | 32 × 32 个元素 |

## 关键结果

| 指标 | Naive CUDA | Tiled CUDA |
|---|---:|---:|
| Duration | 647.49 μs | 173.79 μs |
| Memory Throughput | 84.89% | 87.11% |
| DRAM Throughput | 25.29% | 87.11% |
| L1/TEX Throughput | 96.46% | 27.73% |
| L2 Throughput | 84.89% | 37.56% |
| Compute Throughput | 5.93% | 24.96% |
| Registers / Thread | 22 | 26 |
| Shared Memory / Block | 0 B | 4.22 KB Static |
| Achieved Occupancy | 79.86% | 95.20% |

## Naive 为什么慢？

Naive Kernel 中，一个 Warp 的 32 个线程读取同一行的连续元素，所以读取是合并的。但
写出表达式是：

```text
output[column × M + row]
```

相邻线程的 `column` 相差 1，因此它们的输出地址相差 `M×4 Byte`。在本实验中：

```text
4,096 × 4 Byte = 16 KB
```

同一个 Warp 的线程无法把这些写入合并成少量连续显存事务。NCU 中 L1/TEX 达到
96.46%、L2 达到 84.89%，但 DRAM 只有 25.29%，说明内存访问路径非常忙，却没有形成
同等规模的有效连续数据吞吐。

## Tiled 为什么快？

Tiled Kernel 分为两个阶段：

```text
合并读取输入
    ↓
写入共享内存 tile
    ↓ __syncthreads()
交换 Block 坐标和共享内存索引
    ↓
合并写入输出
```

共享内存是片上存储。它允许线程先按照适合输入的方向读取，再按照适合输出的方向读取
Tile，从而让全局显存的两端都保持连续访问。Tiled 版本把 DRAM Throughput 提高到
87.11%，并获得约 `3.73×` 的 NCU 加速。

## 为什么是 `tile[32][33]`？

共享内存通常有 32 个 Bank。FP32 每个元素 4 Byte，如果二维数组的行跨度正好是 32 个
float，那么转置读取一列时，一个 Warp 的地址会映射到相同 Bank，产生严重 Bank
Conflict。

增加一列 Padding：

```cpp
__shared__ float tile[32][32 + 1];
```

行跨度从 32 变为 33。对 Bank 编号取模后，相邻行会错开一个 Bank，使 Warp 的 32 个
线程分散到 32 个 Bank。这个额外的 128 Byte 不存放有效矩阵元素，只用于改变地址布局。

共享内存大小也与 NCU 一致：

```text
32 × 33 × 4 Byte = 4,224 Byte ≈ 4.22 KB
```

## 为什么 Block 是 32×8？

一个 Tile 有 1,024 个元素，但没有必要真的启动 1,024 个线程。当前使用 256 个线程：

```text
32 × 8 = 256 Threads
```

每个线程通过循环搬运 4 个元素：

```text
offset = 0, 8, 16, 24
```

这样既覆盖完整的 32×32 Tile，又使用常见的 256 Threads / Block 配置。

## 与 Triton 的联系

Triton 源码只写了 `tl.load → tl.trans → tl.store`，但 NCU 显示它使用约 4.10 KB 动态
共享内存。手写 CUDA Tiled 版本显式使用 4.22 KB 静态共享内存。两者性能接近，说明
Triton 编译器自动生成了与经典 CUDA Tiled Transpose 相似的底层策略。

## Padding 与 Bank Conflict 专项实验

为了单独验证 `tile[32][33]` 的作用，增加了一个完全相同但使用 `tile[32][32]` 的
Unpadded Kernel。两者只相差共享内存第二维的长度。

针对 `4096×4096` 输入采集共享内存专项计数器：

| 指标 | `[32][32]` 无 Padding | `[32][33]` 有 Padding |
|---|---:|---:|
| GPU Duration | 196.58 μs | 170.11 μs |
| Shared Load Bank Conflicts | 16,252,928 | 0 |
| Shared Store Bank Conflicts | 0 | 0 |
| Shared Load Wavefronts | 16,934,521 | 525,742 |
| Shared Store Wavefronts | 524,288 | 524,288 |

无 Padding 时，共享内存转置读取产生约 1625 万次 Bank Conflict，Load Wavefront 数量是
有 Padding 时的约 32.2 倍。这正是 32 路 Bank Conflict 的硬件表现。

写入共享内存时两个版本都没有冲突，因为线程按照连续列写入；冲突只发生在后半段的
转置读取。增加一列 Padding 后，读取地址分散到不同 Bank，冲突降为 0。

普通 Benchmark 中的性能差距可能小于 32 倍，因为 Bank Conflict 只影响整个 Kernel 的
共享内存读取阶段，而且大矩阵还受到全局显存带宽限制。专项采集中，Padding 让耗时降低
约 13.5%，但它最重要的价值是消除了可确认的结构性冲突。

专项指标命令：

```powershell
ncu --metrics `
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,`
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,`
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,`
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,`
  gpu__time_duration.sum `
  --kernel-name regex:^transpose_tiled_kernel `
  --launch-skip 10 --launch-count 1 `
  python llm_kernels/transpose/profile_cuda.py --version tiled
```
