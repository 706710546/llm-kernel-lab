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

## 15. CUDA V1：Warp Shuffle 归约

一个 Warp 有 32 个线程。`__shfl_down_sync(mask, value, offset)` 可以让线程
读取同一 Warp 中指定偏移线程的寄存器值，不需要先将这个值写入共享内存。
本实现用偏移 `16, 8, 4, 2, 1` 做树形归约，最终只使用 lane 0 的归约结果。

256 个线程是 8 个 Warp，因此 Block 归约分两层：

```text
每个线程的局部结果
  → 8 个 Warp 各自 Shuffle 归约
  → 每个 Warp 的 lane 0 写入一个共享内存位置
  → 第一个 Warp 读取这 8 个值并再次归约
  → 共享内存广播最终结果给全 Block
```

第一个 Warp 中剩余 24 个 lane 用中性值补齐：求最大值补 `-inf`，求和补 0。
即使行宽不足 32，整个 Warp 仍参与 Shuffle；没有读取真实元素的线程先使用
中性值。当前 Block 固定有 256 个线程，Shuffle 位于所有参与线程都执行的位置，
因此使用 `0xffffffff` 全 Warp 掩码。不要把它直接照搬到只有部分 lane 执行
Shuffle 的分支中。

这里仍需 Block 同步：Warp Shuffle 只能交换 Warp 内的数据，不能替代跨 Warp
共享内存交接。最后还要保证所有线程读完广播值再复用缓冲。此次也补上了旧版
读取 `row_max` 后、复用共享缓冲前缺少的保护同步。

共享缓冲从 1,024 B 降到 32 B，源码中 Block Barrier 从 19 次降到 6 次。
同轮测试 `4096×128` 从 30.72 μs 降到 17.41 μs；`1024×32768` 则仍约
683–684 μs。NCU 确认宽行读流量仍为 402.68 MB、无 Local Memory Spill。
这是一个控制变量实验：归约更便宜了，但三遍读取输入的成本没有改变。

完整报告见 `benchmarks/results/softmax_cuda_shuffle_rtx3080ti.md`。

## 16. CUDA Online V2：归约对象从一个数变成一对状态

此前 CUDA 先扫描求最大值，再扫描求指数和，最后扫描写输出。Online V2
把前两步合并：每个线程扫描自己负责的列，同时维护 `SoftmaxState{maximum, sum}`。
它表示该线程已经处理的集合 A：

```text
m_A = max(A)
l_A = Σₓ∈A exp(x-m_A)
```

线程 t 的列仍然是 `t, t+256, t+512, ...`。这些是离散的线程局部集合，但同一轮
相邻线程加载的列相邻，仍有利于合并访存。每线程更新有顺序依赖；不同线程可以
并行统计自己的集合，最后再合并。

### 16.1 加入一个新元素

新值为 v。如果 `v > m`，旧指数和要换到更大的基准：

```text
l = l * exp(m-v) + 1
m = v
```

否则最大值不变，只要 `l += exp(v-m)`。两种分支每次只计算一个指数，
这与通用双指数公式数学上等价。初始状态 `(-inf, 0)` 表示尚未处理任何元素。

### 16.2 合并两个线程的集合

不能将 l_A 与 l_B 直接相加。若 `m_A >= m_B`：

```text
m = m_A
l = l_A + l_B * exp(m_B-m_A)
```

否则保留 m_B，并缩放 l_A。这一操作能合并两个不相交输入集合；在精确算术下
与分组顺序无关，在浮点计算中仍会有舍入差异，所以测试使用容差。

例如集合 A=[1,2]、B=[3,4]：A 的指数和以 2 为基准，B 以 4 为基准。
最终采用基准 4，A 的和乘 `exp(2-4)` 后才能加入 B 的和。

### 16.3 空状态不能照搬普通公式

当 N 小于线程数时，一些线程没有元素。我们用 `(-inf,0)` 表示空集合。
若两个空集合直接算 `exp(m_A-m_B)`，就会出现 `exp(-inf-(-inf))=NaN`。
因此 `merge_states` 先判断空状态，空集合与 A 合并仍返回 A。
对于本项目测试的有限、非空输入集合，至少一个最大值的指数贡献为 1，
所以 l 不为 0，可用 l=0 识别空状态。这里不承诺 NaN/Inf 输入的特殊语义。

### 16.4 Warp 到 Block 的合并

`warp_reduce_state` 分别 Shuffle maximum 和 sum，再按状态合并规则计算。
每个 Warp 的 lane 0 将一对结果写入共享内存；第一个 Warp 合并这 8 对，
再向整个 Block 广播最终结果。共享内存为两个 8-float 数组，共 64 B。
两次 Block Barrier 分别保护 Warp 结果交接与最终结果广播；之后不复用缓冲。

第二遍输入扫描用最终的 `(m,l)` 写出概率。因此它仍是独立 Softmax 的两遍
实现，不是 FlashAttention。减少输入读取并不保证更快：在线依赖、指数运算
和分支也有成本，需要 Benchmark 与实际 DRAM 计数器验证。

### 16.5 实验结果与运行方式

三轮交替 provider 顺序实验的 P50 中位数：`1024×32768` 从 Shuffle 的
685.52 μs 降到 Online 的 529.41 μs；`4096×128` 从 17.41 μs 增到
26.62 μs。宽行收益来自少读一遍输入，但短行没有足够数据量抵消状态合并成本。
NCU 测得 Online 宽行读流量约 268.47 MB、Local Load/Store 为 0 B，符合
两遍读取；此版本仍不是普适最快的 Softmax。

```powershell
python llm_kernels/softmax/test.py
python llm_kernels/softmax/benchmark_cuda_online.py
```

完整报告与测试范围见 `benchmarks/results/softmax_cuda_online_rtx3080ti.md`。
