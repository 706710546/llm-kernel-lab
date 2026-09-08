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
