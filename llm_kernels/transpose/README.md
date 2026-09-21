# Matrix Transpose

Matrix Transpose 是项目的第三个算子。它把一维连续访存推进到二维索引与 Tile，重点是
理解逻辑转置、实际数据搬运和合并访存之间的区别。

## 1. 数学定义

输入 `x` 的形状为 `[M, N]`，输出 `y` 的形状为 `[N, M]`：

```text
y[j, i] = x[i, j]
```

例如：

```text
[[1, 2, 3],       [[1, 4],
 [4, 5, 6]]   →    [2, 5],
                    [3, 6]]
```

## 2. View 与真正的数据搬运

PyTorch 的 `x.transpose(0, 1)` 通常只交换 Shape 和 Stride，返回一个不连续 View，并没有
真的重新排列显存。本实验使用：

```python
x.transpose(0, 1).contiguous()
```

`contiguous()` 会分配 `[N, M]` 的连续输出并搬运数据，这才与 Triton Kernel 完成的是
同一个任务，二者性能才可以公平比较。

## 3. Triton V0：32×32 Tile

V0 将输入矩阵划分成 `32×32` Tile：

```text
Grid = (ceil(M / 32), ceil(N / 32))
```

每个 Triton Program：

1. 根据两个 Program ID 定位一个输入 Tile；
2. 连续读取 Tile 中的数据；
3. 使用 `tl.trans` 交换 Tile 的两个维度；
4. 将结果连续写入输出矩阵对应的转置 Tile；
5. 对不能填满 32×32 的边缘 Tile 使用 Mask。

输入位置与输出位置分别是：

```text
input_offset  = row × N + column
output_offset = column × M + row
```

## 4. 理论数据量

转置不做浮点计算。每个元素读取一次、写入一次。FP32 的逻辑数据量为：

```text
读取：4MN Byte
写入：4MN Byte
合计：8MN Byte
```

因此它和 Vector Add 一样，预期主要受到显存访问效率限制。但转置比 Vector Add 更难，
因为如果映射方式不合适，就会出现非合并读取或非合并写入。

## 5. 运行方式

```powershell
python llm_kernels/transpose/test.py
python llm_kernels/transpose/benchmark.py
```

Triton V0 支持连续的二维 FP16/FP32 CUDA Tensor；手写 CUDA 实验当前使用 FP32。

## 6. V0 Benchmark

RTX 3080 Ti、FP32 的第一轮结果：

| Shape | PyTorch GB/s | Triton GB/s |
|---:|---:|---:|
| 256 × 256 | 85.33 | 102.40 |
| 1024 × 1024 | 431.16 | 630.15 |
| 4096 × 1024 | 436.91 | 744.73 |
| 1024 × 4096 | 520.13 | 744.73 |
| 4096 × 4096 | 388.40 | 775.57 |
| 8192 × 8192 | 338.03 | 784.86 |

大矩阵下，Triton V0 达到约 775～785 GB/s。专用方形 Tile Kernel 明显快于当前 PyTorch
参考路径，但这不意味着 Triton 普遍比 PyTorch 快；这里只比较了特定形状、连续输入和
“生成连续转置副本”这一项任务。

完整原始数据保存在：

```text
benchmarks/results/transpose_v0_rtx3080ti_fp32.csv
```

NCU 分析命令：

```powershell
ncu --set basic --kernel-name regex:transpose_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/transpose/profile.py
```

## 7. NCU 结论

对 `4096×4096` FP32 输入进行分析：

| 指标 | 测量值 |
|---|---:|
| Duration | 168.86 μs |
| DRAM Throughput | 89.22% |
| Compute Throughput | 15.36% |
| Achieved Occupancy | 93.44% |
| Dynamic Shared Memory / Block | 4.10 KB |

`4.10 KB` 接近一个 `32×32` FP32 Tile 的 `4,096 Byte`。虽然 Triton 源码没有显式声明
共享内存，编译器仍然为 `tl.trans` 生成了共享内存中转，从而让输入读取和输出写入都能
保持高效。完整分析见：

```text
benchmarks/results/transpose_v0_rtx3080ti_ncu_basic.md
```

## 8. Tile Size 实验

`4096×4096` FP32 输入的 NCU 对比：

| 指标 | 16×16 | 32×32 |
|---|---:|---:|
| Program 数量 | 65,536 | 16,384 |
| 底层 Threads / Block | 128 | 128 |
| Duration | 218.59 μs | 168.86 μs |
| DRAM Throughput | 69.70% | 89.22% |
| Achieved Occupancy | 93.97% | 93.44% |

两者的底层线程数和 Occupancy 几乎相同，但 32×32 用更少的 Program 搬运更多连续数据，
显著提高了 DRAM 吞吐。因此 V0 最终默认使用 32×32 Tile。

运行完整 Tile Benchmark：

```powershell
python llm_kernels/transpose/benchmark_tiles.py
```

原始数据保存在：

```text
benchmarks/results/transpose_tile_comparison_rtx3080ti_fp32.csv
```

## 9. 手写 CUDA：Naive 与 Tiled

CUDA 阶段同时保留两个 Kernel：

```text
transpose_naive_kernel：合并读取，跨行写出
transpose_tiled_kernel：合并读取 → 共享内存 → 合并写出
```

两者都使用 `32×8=256` Threads / Block，每个线程循环处理 4 个元素，共同覆盖一个
`32×32` Tile。`4096×4096` 的 Benchmark：

| Provider | P50 | 有效带宽 |
|---|---:|---:|
| PyTorch | 353.25 μs | 379.95 GB/s |
| Triton | 176.13 μs | 762.05 GB/s |
| CUDA Naive | 467.46 μs | 287.12 GB/s |
| CUDA Tiled | 173.06 μs | 775.57 GB/s |

Naive CUDA 因非合并写入明显更慢；Tiled CUDA 与 Triton 基本持平。

## 10. 共享内存与同步

Tiled CUDA 声明：

```cpp
__shared__ float tile[32][33];
```

线程先合作写入共享内存，然后必须调用：

```cpp
__syncthreads();
```

因为后半段读取的元素可能是其他线程写入的。如果没有同步，部分线程可能在 Tile 尚未
写完时就开始读取，从而产生数据竞争。

第二维使用 `33` 而不是 `32`，是为了改变共享内存行跨度，避免转置读列时发生严重的
Bank Conflict。完整 NCU 对照见：

```text
benchmarks/results/transpose_cuda_rtx3080ti_ncu_basic.md
```

运行四方正确性与性能测试：

```powershell
python llm_kernels/transpose/test_cuda.py
python llm_kernels/transpose/benchmark.py
```

## 11. Bank Conflict 实验证据

项目保留了一个只用于实验的 `cuda_tiled_unpadded` 版本：

```text
cuda_tiled_unpadded：tile[32][32]
cuda_tiled：         tile[32][33]
```

`4096×4096` 输入的专项 NCU 结果：

| 指标 | 无 Padding | 有 Padding |
|---|---:|---:|
| Duration | 196.58 μs | 170.11 μs |
| Shared Load Bank Conflicts | 16,252,928 | 0 |
| Shared Load Wavefronts | 16,934,521 | 525,742 |

无 Padding 版本产生约 1625 万次冲突；增加一列后冲突完全消失，Load Wavefront 数量下降
约 32.2 倍。普通 Benchmark 的差距没有 32 倍，因为共享内存读取只是 Kernel 的一个
阶段，最终性能还受到全局显存带宽限制。

运行 Padding Benchmark：

```powershell
python llm_kernels/transpose/benchmark_cuda_padding.py
```

原始数据保存在：

```text
benchmarks/results/transpose_cuda_padding_rtx3080ti_fp32.csv
```
