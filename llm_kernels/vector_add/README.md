# Vector Add

Vector Add 是 LLM Kernel Lab 的第一个完整实验。它的计算非常简单，因此可以帮助我们
单独观察 Kernel 启动开销和全局显存带宽。

## 1. 数学定义

对于两个长度相同的向量：

```text
output[i] = x[i] + y[i]
```

## 2. 输入与输出形状

第一个版本接受形状相同、内存连续且位于 CUDA 设备上的 Tensor。Kernel 会把任意形状
的 Tensor 看作长度为 `N` 的一维向量。

## 3. PyTorch 参考实现

参考结果使用 `torch.add`，代码写作 `x + y`。它首先用于验证 Triton 输出是否正确，
同时也是性能对照组。

## 4. 理论 FLOPs

每个元素只进行一次浮点加法，所以总计算量约为 `N` FLOPs。

## 5. 理论显存流量

每个元素需要读取两个输入，并写出一个结果。对于 FP32：

```text
读取 x：4 字节
读取 y：4 字节
写出 output：4 字节
总计：12 字节/元素，即 12N 字节
```

## 6. 算术强度

对于 FP32：

```text
AI = N FLOPs / 12N Bytes = 1/12 FLOP/Byte
```

这个数值非常低，说明每做一次加法，就需要搬运大量数据。

## 7. 性能瓶颈假设

- 大向量应该受到显存带宽限制（Memory Bound）。
- 很小的向量搬运数据很少，固定的 Kernel Launch 和计时开销会占据主要部分。

## 8. GPU 映射方式

每个 Triton Program 处理连续的 `block_size` 个元素。相邻 Lane 访问相邻地址，有利于
形成合并访存（Coalesced Memory Access）。当 `N` 不能被 `block_size` 整除时，最后
一个 Program 使用 Mask 避免越界读写。

## 9. 当前使用的优化

第一个版本只使用一维 Grid、连续访存和单个 `load → add → store` Kernel，暂时不做
Autotuning。这样可以先建立一个容易解释的基线。

## 10. Benchmark 是否支持假设？

第一次 RTX 3080 Ti 本地实验中，当 `N = 2^26` 时，PyTorch 和 Triton 的 FP32 有效
带宽都达到约 816–818 GB/s。当 `N <= 2^14` 时，延迟保持在约 4–5 微秒，说明此时
主要受固定开销影响。

原始数据保存在：

```text
benchmarks/results/vector_add_rtx3080ti_fp32.csv
```

大尺寸结果支持 Memory Bound 假设。与此同时，简单 Triton 实现与 PyTorch 基本持平，
所以我们不能只写“Triton 更快”，而应该继续回答：

1. 有效带宽从哪个尺寸开始进入平台期？
2. 它与 Profiler 测得的 DRAM Throughput 有什么差别？
3. 为什么小尺寸下两者存在几微秒差异？
4. P20 到 P80 的范围是否说明测量稳定？

## 11. Nsight Compute 分析入口

`profile.py` 默认先预热 10 次，再执行一次待分析的 Kernel。下面的命令跳过 10 次
预热，只捕获最后一次调用：

```powershell
ncu --set basic --kernel-name regex:vector_add_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/vector_add/profile.py
```

第一次实测结果：

| 指标 | 测量值 |
|---|---:|
| Duration | 247.10 μs |
| DRAM Throughput | 91.67% |
| Compute (SM) Throughput | 7.53% |
| Achieved Occupancy | 86.10% |

DRAM Throughput 远高于 Compute Throughput，直接验证了 Memory Bound 判断。虽然实际
Occupancy 没有达到理论值，但显存系统已经接近饱和，因此不能假设把 Occupancy 提高到
100% 就一定能获得相同比例的加速。

详细实验信息和推导见：

```text
benchmarks/results/vector_add_rtx3080ti_ncu_basic.md
```

## 12. Block Size 对比图

使用 `128`、`256`、`512` 三种 Triton Block Size 进行实测：

![Block Size 性能比较](../../assets/benchmark/vector_add_block_size_comparison.svg)

在大尺寸输入下，三条有效带宽曲线最终都达到约 810～826 GB/s。Block Size 改变了
Program 数量，但没有明显改变带宽平台，进一步支持显存带宽是主要瓶颈的判断。

重新生成图表：

```powershell
python scripts/plot_vector_add.py
```

## 13. CUDA C++ Extension

CUDA 版本用于理解 Triton 隐藏起来的 Host / Device、Grid / Block / Thread 映射。当前版本
支持连续的 FP32 CUDA Tensor，并允许配置 `threads_per_block`；每个 CUDA Thread 负责一个元素：

```text
index = blockIdx.x × blockDim.x + threadIdx.x
```

源码分工：

```text
cuda_impl.py                 Python 包装与扩展加载
csrc/bindings.cpp            Python 与 C++ 的函数绑定
csrc/vector_add_cuda.cu      CUDA Kernel 和 Host Launcher
test_cuda.py                 PyTorch / Triton / CUDA 三方正确性测试
```

Windows 下可以直接从普通 PowerShell 运行。包装层会通过 `vswhere` 在当前 Python 进程中
查找并加载已安装的 MSVC Build Tools，不会修改系统环境变量：

```powershell
python llm_kernels/vector_add/test_cuda.py
```

第一次运行会编译扩展。为兼容仓库父目录中的中文路径，包装层会将两份 C++/CUDA 源码
同步到系统临时目录下的 `llm_kernel_lab_extensions/vector_add`，并在那里保存构建缓存；
仓库中的 `csrc` 始终是唯一源码。后续运行只在源码变化时重新编译。

CUDA 第一版先建立正确、可解释的 FP32 基线；下面继续用 Benchmark 和 NCU 验证性能。

## 14. PyTorch / Triton / CUDA 性能对比

三种实现共用 `benchmark.py` 中相同的输入、计时器和有效带宽公式。CUDA Extension
会在正式计时之前完成编译，因此表中的延迟只包含算子调用和 GPU 执行，不包含编译时间。

RTX 3080 Ti 的第一次三方实验结果：

| N | PyTorch GB/s | Triton GB/s | CUDA GB/s |
|---:|---:|---:|---:|
| 1,024 | 0.83 | 2.40 | 3.00 |
| 262,144 | 341.33 | 384.00 | 384.00 |
| 4,194,304 | 780.19 | 768.00 | 768.00 |
| 16,777,216 | 815.80 | 812.43 | 815.80 |
| 67,108,864 | 825.65 | 826.08 | 831.32 |

最大输入下三者的性能差距不到 1%，都进入约 830 GB/s 的显存带宽平台。这说明对于
连续 Vector Add，PyTorch、Triton 和朴素 CUDA 都已经接近同一个硬件瓶颈。CUDA 的价值
在这里主要是暴露 Grid、Block、Thread 和 Stream，而不是保证比高级框架更快。

完整数据保存在：

```text
benchmarks/results/vector_add_cuda_rtx3080ti_fp32.csv
```

重新运行：

```powershell
python llm_kernels/vector_add/benchmark.py
```

用于 NCU 的 CUDA 单 Kernel 入口：

```powershell
ncu --set basic --kernel-name regex:^vector_add_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/vector_add/profile_cuda.py
```

CUDA 与 Triton 的 NCU 对照报告保存在：

```text
benchmarks/results/vector_add_cuda_rtx3080ti_ncu_basic.md
```

## 15. CUDA Threads / Block 实验

CUDA 包装函数默认使用 256 Threads / Block，也可以显式传入其他配置：

```python
output = vector_add_cuda(x, y, threads_per_block=128)
```

参数必须是 `[32, 1024]` 范围内的 2 的幂。最终实验选择 `128`、`256`、`512` 三组：

| Threads / Block | 最大输入 P50 | 最大输入有效带宽 | NCU Duration | DRAM Throughput | Achieved Occupancy |
|---:|---:|---:|---:|---:|---:|
| 128 | 980.992 μs | 820.91 GB/s | 247.42 μs | 91.73% | 80.37% |
| 256 | 983.040 μs | 819.20 GB/s | 249.60 μs | 91.29% | 76.54% |
| 512 | 979.968 μs | 821.77 GB/s | 247.94 μs | 91.96% | 66.70% |

三种配置在最大输入上的 Benchmark 差距不到 0.4%，NCU Duration 差距不到 0.9%。虽然
512 线程的 Achieved Occupancy 明显更低，但它的 DRAM Throughput 和运行时间没有明显
变差，因为三种配置都提供了足够的并行度，并且已经到达相同的显存带宽瓶颈。

因此最终默认值保留为 256：它是常见且容易解释的折中配置，而不是因为本次测量证明它
最快。完整原始数据保存在：

```text
benchmarks/results/vector_add_cuda_block_comparison_rtx3080ti_fp32.csv
```

重新运行 Block 实验：

```powershell
python llm_kernels/vector_add/benchmark_cuda_blocks.py
```
