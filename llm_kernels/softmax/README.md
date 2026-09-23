# Softmax

Softmax 是项目的第四个算子。它把 Vector Add 的连续访存、Reduction 的按行归约和新的
数值稳定性问题组合在一起，也是 Attention 中把分数转换为概率的核心步骤。

## 1. 数学定义

对一行输入 `x`：

```text
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

输出满足：

```text
每个元素位于 0 到 1 之间
每一行所有元素之和约等于 1
```

## 2. 为什么不能直接计算 `exp(x)`？

浮点数的表示范围有限。如果输入很大：

```text
exp(10,000) → inf
```

后续就可能出现：

```text
inf / inf → nan
```

Softmax 对整行同时平移一个常数不敏感，因此使用稳定形式：

```text
m = max(x)
softmax(xᵢ) = exp(xᵢ - m) / Σⱼ exp(xⱼ - m)
```

最大元素变为 `exp(0)=1`，其他指数都不大于 1，从而避免正向溢出。

## 3. Triton V0 映射

V0 使用一个 Program 处理一整行：

```text
一行输入
  ↓ tl.max
减去最大值
  ↓ tl.exp
计算指数
  ↓ tl.sum
得到分母
  ↓ 除法
写出一行概率
```

Grid 为：

```text
Grid = (M,)
```

输入行宽会向上补齐到 2 的幂。例如 `N=1003` 使用 `BLOCK_SIZE=1024`。

## 4. 为什么 Mask 使用 `-inf`？

补齐位置不属于真实输入。在求最大值时，如果补 0，而真实元素全是负数，0 会错误地
成为最大值。因此读取时使用：

```python
other=-float("inf")
```

它同时满足：

```text
max(x, -inf) = max(x)
exp(-inf) = 0
```

所以补齐位置既不影响最大值，也不影响指数和。

## 5. FP32 中间计算

FP16 输入加载后会转换为 FP32，再执行 `max、exp、sum、divide`。最后写入 FP16 输出时
才发生舍入。这能减少指数和归约中的低精度误差。

## 6. V0 限制

与 Reduction V0 类似，一个 Program 必须容纳完整一行。当前最大 Block Size 为 65,536，
更宽的行需要分块 Softmax 或 Online Softmax。当前阶段先建立容易解释的基线。

## 7. 运行方式

```powershell
python llm_kernels/softmax/test.py
python llm_kernels/softmax/benchmark.py
```

## 8. V0 Benchmark

RTX 3080 Ti、FP32：

| Shape | PyTorch GB/s | Triton GB/s |
|---:|---:|---:|
| 4096×128 | 409.60 | 372.36 |
| 4096×512 | 630.15 | 658.65 |
| 4096×2048 | 762.05 | 789.59 |
| 4096×8192 | 796.79 | 816.65 |
| 1024×32768 | 426.25 | 143.56 |

`N≤8192` 时，V0 与 PyTorch 性能接近；`N=32768` 时，V0 性能明显下降。超宽行让单个
Program 同时维护更多输入、指数值和归约中间状态，可能增加寄存器、共享内存或 Local
Memory 压力。第 9 节使用 NCU 对比正常点和退化点，验证了具体原因。

原始数据保存在：

```text
benchmarks/results/softmax_v0_rtx3080ti_fp32.csv
```

NCU 入口：

```powershell
ncu --set basic --kernel-name regex:softmax_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/softmax/profile.py --columns 8192
```

## 9. NCU：为什么超宽行性能下降？

选择两个元素总数相同的输入：

```text
4096×8192  = 33,554,432
1024×32768 = 33,554,432
```

| 指标 | N=8,192 | N=32,768 |
|---|---:|---:|
| Duration | 333.60 μs | 1.87 ms |
| DRAM Throughput | 91.10% | 87.24% |
| Registers / Thread | 102 | 40 |
| Achieved Occupancy | 32.21% | 81.83% |
| Local Load | 0 B | 701.50 MB |
| Local Store | 0 B | 510.13 MB |
| 实际 DRAM 总流量 | 267.79 MB | 1,460.43 MB |

超宽行版本的 Occupancy 更高，却慢约 5.6 倍。根因是寄存器无法容纳所有中间值，编译器
把约 1.21 GB 数据 Spill 到 Local Memory。Local Memory 最终由显存和 Cache 承载，使
实际 DRAM 流量扩大到逻辑数据量的约 5.44 倍。

因此不能把高 Occupancy 单独当作性能良好的证据。完整报告见：

```text
benchmarks/results/softmax_v0_rtx3080ti_ncu_basic.md
```

## 10. V1：分块统计与合并

V1 每块处理 1,024 个元素，共启动三个 Kernel：

```text
Stage 1：每块求局部最大值 mₖ 和局部指数和 lₖ
Stage 2：每行合并所有 (mₖ, lₖ)，得到全局 (m, l)
Stage 3：再次读取输入，用 exp(xᵢ-m)/l 写出概率
```

合并公式是：

```text
m = maxₖ(mₖ)
l = Σₖ lₖ × exp(mₖ-m)
```

各块必须重新缩放 `lₖ`，因为它们分别减去了自己的局部最大值。V1 使用 FP32 中间
统计量，能处理到 1,048,576 列；当前测试覆盖到 131,072 列。

例如一行拆为 `[1, 2]` 和 `[3, 4]`。前一块用 `m₁=2` 计算 `l₁=e⁻¹+1`，后一块用
`m₂=4` 计算 `l₂=e⁻¹+1`。全行最大值为 `m=4`，所以前一块的统计量必须乘以
`exp(2-4)=e⁻²`，得到 `l=l₁e⁻²+l₂`。直接把两个 `l` 相加会得到错误分母。

## 11. V0 与 V1 的取舍

同一轮 FP32 Benchmark：

| 形状 | V0 P50 | V1 P50 | 结论 |
|---:|---:|---:|---|
| 4096×8192 | 331.26 μs | 510.43 μs | V0 更快 |
| 2048×16384 | 333.82 μs | 510.98 μs | V0 更快 |
| 1024×32768 | 1881.60 μs | 501.76 μs | V1 约快 3.75 倍 |
| 512×131072 | 不支持 | 985.60 μs | V1 可处理 |

V1 短行较慢，因为它启动三个 Kernel、读取输入两遍并写入中间统计量；超宽行时，这些
明确的额外成本远小于 V0 的寄存器 Spill。

NCU 对 `1024×32768` 的 V1 三阶段分别测得 Local Load/Store 都为 0，合计实际 DRAM
流量约 410 MB，远低于同形状 V0 的 1,460 MB。详见：

```text
benchmarks/results/softmax_v1_rtx3080ti_ncu_basic.md
```

这一步为理解 Online Softmax 和 FlashAttention 做准备。当前 V1 是三阶段并行算法，
不是单 Kernel 在线实现。

## 12. V2：在线更新最大值与指数和

V2 与 V1 同样把一行切成长度为 1,024 的块，但只启动一个 Kernel、每行一个
Program。扫描到当前块时，跨块只需保留两个 FP32 标量；当前块仍会产生临时向量：

```text
m = 已扫描元素的最大值
l = Σ已扫描元素 exp(xᵢ - m)
```

假设新块的最大值为 `b`，本块元素为 `xᵢ`，则更新公式是：

```text
m_new = max(m, b)
l_new = l × exp(m - m_new) + Σ本块 exp(xᵢ - m_new)
```

第一项把旧的指数和换算到新最大值的基准下，第二项加入本块贡献。例子：先扫描
`[1, 2]`，得到 `m=2, l=e⁻¹+1`；再扫描 `[3, 4]`，新最大值为 4，旧 `l`
必须乘 `e⁻²`，然后加上 `e⁻¹+1`。如果直接相加，两个块的指数和不在同一个
基准下，结果就错了。

第一遍扫描结束后才知道整行最终的 `(m, l)`，因此独立 Softmax 仍要第二遍读取
输入，计算 `exp(xᵢ-m)/l` 并写出输出。这里的“在线”指分块更新归一化统计量，
不等于单遍写出完整 Softmax，也不等于已经实现 FlashAttention。

## 13. V0 / V1 / V2 对比

RTX 3080 Ti、FP32，同一轮 Benchmark 的 P50：

| 形状 | V0 | V1 | V2 |
|---:|---:|---:|---:|
| 4096×128 | 10.24 μs | 23.55 μs | 13.31 μs |
| 4096×8192 | 329.82 μs | 506.88 μs | 491.52 μs |
| 2048×16384 | 333.82 μs | 512.00 μs | 493.57 μs |
| 1024×32768 | 1866.75 μs | 502.78 μs | 494.59 μs |
| 512×131072 | 不支持 | 980.99 μs | 977.92 μs |

短行首选 V0：一次读取、一次启动，没有跨块循环或中间缓冲。超宽行中，V0
发生寄存器 Spill；V1/V2 都控制了每个 Program 的工作集。V2 在两种超宽
形状下与 V1 接近，微小差异不能作为稳定胜出的结论。V1 的块可以跨 Program
并行，V2 的同一行需要依次更新 `(m, l)`；V2 的优势是只启动一个 Kernel，
不分配跨 Kernel 的统计量缓冲。

NCU 对 `1024×32768` 的 V2 测得 Local Load/Store 均为 0，实际 DRAM 读写
约 406.93 MB，接近“两读一写”的 402.65 MB 逻辑下界。详细数据见
`benchmarks/results/softmax_v2_rtx3080ti_ncu_basic.md`；同轮完整数据见
`benchmarks/results/softmax_v0_v1_v2_rtx3080ti_fp32.csv`。

注意 Benchmark 中的“有效带宽”统一按一次读取和一次写入计算；V1/V2 实际
读取输入两遍，因此该数字不是 NCU 测得的实际 DRAM 带宽。

## 14. CUDA V0：把 Triton 归约展开成线程协作

CUDA 版本目前只支持连续二维 FP32。`blockIdx.x` 指定一行，256 个线程共同处理
这一行；线程 `t` 负责 `t, t+256, t+512, ...` 等列。对应关系如下：

| Triton V0 | CUDA V0 |
|---|---|
| `tl.program_id(0)` | `blockIdx.x` |
| `tl.arange(...)` | `threadIdx.x` 加步长循环 |
| `tl.max(values)` | 线程局部最大值 + 共享内存树形归约 |
| `tl.sum(values)` | 线程局部和 + 共享内存树形归约 |
| Program 内隐式协作 | `__shared__` 和显式 `__syncthreads()` |

为什么第一次归约后要同步？线程 0 读取 `partial[128]` 前，线程 128 必须已经
写完；而每轮归约的写入又会成为下一轮的输入。删掉同步会产生数据竞争。

CUDA V0 分三次扫描输入：

1. 求整行最大值 `m`；
2. 求 `Σ exp(xᵢ-m)`；
3. 重新读取并写出 `exp(xᵢ-m)/l`。

它没有缓存整行，所以宽行不需要让大量中间值长期占用寄存器，但比 Triton V2
多读一次输入。`1024×32768` 的本轮 P50 为 CUDA 684.03 μs、Triton V2
492.54 μs；NCU 测得 CUDA DRAM Read 402.68 MB、Local Load/Store 均为 0。
更高的 CUDA Occupancy 没有转化为更低延迟；读取次数、显式归约同步和指数计算
也都影响时间。完整报告见 `benchmarks/results/softmax_cuda_rtx3080ti_ncu_basic.md`。

编译用 PyTorch CUDA Extension；在本项目的中文 Windows 路径下，会把仓库中的
源码暂存到 ASCII 临时目录再交给 NVCC。Windows 下 `.cu` 注释使用 ASCII，中文
教学解释集中放在此文档中，以避免 NVCC 与系统编码组合导致的解析问题。
