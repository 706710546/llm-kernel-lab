# Reduction：按行求和

Reduction 的含义是：将一组元素合并成更少的结果。这个模块从最基础的按行求和开始。

## 1. 数学定义

输入 `x` 的形状为 `[M, N]`：

```text
x = [
  [x₀₀, x₀₁, ..., x₀,N-1],
  [x₁₀, x₁₁, ..., x₁,N-1],
  ...
]
```

输出 `y` 的形状为 `[M]`：

```text
yᵢ = Σⱼ xᵢⱼ
```

例如：

```text
[[1, 2, 3],
 [4, 5, 6]]
        ↓ 按行求和
[6, 15]
```

## 2. 为什么它比 Vector Add 难？

Vector Add 中，每个输出元素只依赖一对输入：

```text
output[i] = x[i] + y[i]
```

每个位置可以完全独立计算。

Reduction 中，一个输出元素依赖一整行输入：

```text
y[0] = x[0, 0] + x[0, 1] + ... + x[0, N-1]
```

多个 GPU 线程必须合作，先计算局部和，再合并中间结果。这会引入并行归约、同步、Warp
协作和数值累加顺序等新问题。

## 3. 当前阶段：可信的 PyTorch 参考实现

```python
y = x.sum(dim=-1)
```

当前已完成数学定义、输入输出形状、PyTorch 参考实现和 Triton V0 正确性测试。

## 4. Triton V0 设计

最简单的映射是：

```text
一个输出 y[i]
        ↓
一个 Triton Program
        ↓
读取输入 x[i, :]
        ↓
tl.sum 在 Program 内归约
        ↓
写入 y[i]
```

例如输入形状为 `[7, 1003]`：

```text
Grid = (7,)
Program 0 → 第 0 行 → 输出 y[0]
Program 1 → 第 1 行 → 输出 y[1]
...
Program 6 → 第 6 行 → 输出 y[6]
```

`tl.arange` 的长度必须在编译期确定，因此 V0 会将 `1003` 向上补齐到下一个 2 的幂
`1024`。后面额外的 21 个位置通过 Mask 屏蔽，并以 `0` 参与求和；这不会改变结果。

FP16 输入不会直接以 FP16 连续累加。V0 会先转换为 FP32，再执行 `tl.sum`，最后按输出
Tensor 的类型写回。这是因为浮点加法不满足严格结合律：并行归约的加法顺序与 PyTorch
内部实现可能不同，低精度累加会放大这种差异。

## 5. Triton V1：两阶段分块归约

V0 的一个 Program 必须一次读取、累加一整行。因此它的 `BLOCK_SIZE` 最多是
`65,536`，更宽的行无法处理。

V1 将一行分块。固定每块 `1,024` 个元素，分两个 kernel 完成：

```text
输入 x，形状 [M, N]
        │
        ├─ Stage 1：每个 Program 处理 row 的一个块
        │             将块内 1,024 个元素求和
        ▼
partials，形状 [M, ceil(N / 1024)]，元素类型 FP32
        │
        ├─ Stage 2：每个 Program 处理 partials 的一行
        │             合并这一行的所有局部和
        ▼
输出 y，形状 [M]
```

例如 `x` 的形状为 `[2, 2,500]`：

```text
Stage 1：每行拆为 3 块
  Program (0, 0) → x[0,    0:1024] → partials[0, 0]
  Program (0, 1) → x[0, 1024:2048] → partials[0, 1]
  Program (0, 2) → x[0, 2048:2500] → partials[0, 2]（末尾用 Mask 补 0）
  Program (1, *) → 处理第 1 行的三个块

Stage 2：每行合并 3 个局部和
  Program 0 → partials[0, :] → y[0]
  Program 1 → partials[1, :] → y[1]
```

V1 的代价是：多了一次写入 `partials` 和一次读取 `partials`，还启动了第二个 kernel。
因此对于 V0 已能支持的短行，V1 不保证更快；它的主要价值是支持更宽的行。中间结果采用
FP32，避免 FP16 在每个块完成时就发生额外舍入。

## 6. 当前实现路线

```text
V0：一个 Triton Program 处理一整行
        ↓
V1：分块写出 FP32 局部和，再进行第二阶段归约
        ↓
V2：进入 CUDA Shared Memory 与 Warp Reduction
```

不要提前跳到 V2。我们先理解：一行输入如何被映射到一个 Triton Program，以及为什么
树状求和的顺序和 CPU 的串行循环不同。

## 当前检查清单

| 项目 | 状态 |
|---|---|
| 数学定义 | ✓ |
| PyTorch 参考实现 | ✓ |
| 手算样例 | ✓ |
| 边界形状测试 | ✓ |
| Triton V0 | ✓ |
| Triton V1 两阶段归约 | ✓ |
| V0 / V1 Benchmark | ✓ |
| NCU Profiler 入口 | ✓ |

## 7. Nsight Compute 验证

对 V1 而言，不能只捕获一次 Kernel：它有两个阶段，需要分别分析。

```powershell
# V0：一个 Kernel，一行对应一个 Program
ncu --set basic --kernel-name regex:row_sum_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/reduction/profile.py --version v0

# V1 Stage 1：分块读取原始输入并写入 partials
ncu --set basic --kernel-name regex:row_sum_stage1_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/reduction/profile.py --version v1

# V1 Stage 2：读取 partials 并合并为最终输出
ncu --set basic --kernel-name regex:row_sum_stage2_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/reduction/profile.py --version v1
```

默认输入为 `[256, 65,536]` 的 FP32 矩阵。分析得到以下结论：

1. V0 和 V1 Stage 1 的 DRAM Throughput 都约为 92.8%，两者都是 Memory-Bound；
2. V1 Stage 1 虽然 Occupancy 更高，但无法突破已饱和的显存带宽；
3. V1 Stage 2 的 GPU 利用率低，但只耗时约 3.58 μs，当前不是优化重点；
4. V0 在 `N ≤ 65,536` 时略快；更宽的行使用 V1。

完整的原始指标、推导过程和工程结论见
[V0 / V1 Nsight Compute 实验报告](../../benchmarks/results/reduction_v0_v1_rtx3080ti_ncu_basic.md)。
