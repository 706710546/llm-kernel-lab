# RMSNorm

## 这个算子做什么？

RMSNorm 对每个向量的最后一维计算均方根（Root Mean Square），按这个尺度归一化，
再乘上每个通道可学习的缩放参数 `weight`：

```text
rms(x) = sqrt(mean(x * x) + eps)
y      = (x / rms(x)) * weight
```

若 `x` 的形状是 `[batch, hidden]`，每个 batch 行各自归一化，`weight` 的形状是
`[hidden]`，输出形状与 `x` 相同。输入也可以有多个前置维度，归一化始终沿最后一维。

RMSNorm 不减去均值；LayerNorm 则会先减均值，再用方差归一化。不要把二者混为一谈。

## Triton V0：一行一个 Program

V0 将输入最右侧维度视作 `N` 个通道。一个 Triton Program（可类比一个 CTA）处理一行：
加载这一行及对应 `weight`，计算 `sum(x*x) / N`，做 `rsqrt`，再把缩放结果写回。
所有行的 Programs 并行运行。输入若有多个前置维度，包装函数会把这些维度展平为行数，
但只接受连续 Tensor。

宽度 `N` 不一定是 2 的幂。Triton 的 `tl.arange` 要求 V0 使用向上取整后的
`BLOCK_SIZE = next_power_of_2(N)`。超过真实宽度的 lane 通过 Mask 读 0、写入时被屏蔽。
平方和因此没有额外贡献，但均值必须除以真实 `N`，不能除以补齐后的 `BLOCK_SIZE`。
比如 `N=17` 时 block 是 32，后 15 个 lane 是补齐位置；分母仍然是 17。

低精度输入加载后显式转为 FP32 做平方、归约和归一化。最终 `tl.store` 按输出 Tensor
的 dtype 转回原精度。

V0 当前支持 CUDA 连续的 FP16/BF16/FP32，最多 16,384 个元素/行，使用 4 个 warps。
不连续输入、FP64 和更宽的行会明确报错，不做隐式拷贝或假装支持。之后再讨论更宽行
如何分块，以及不同 `num_warps` 的取舍。

## CUDA C++ 教学基线：一个 Block 处理一行

`cuda_impl.py` 通过 PyTorch CUDA Extension 编译并调用 `csrc/rmsnorm_cuda.cu`。两个 CUDA
版本支持 FP32、FP16 和 BF16；输入连续，`weight` 必须是同设备、同 dtype 的一维张量。
半精度和 BF16 加载后转换成 FP32 计算平方和与归一化，再转换回输入 dtype 写出。
前置维会展平为行数；零行返回空张量，归一化维度必须非空。当前不支持反向传播或非连续输入。

CUDA 版用每线程的 stride loop 处理整行，不要求行宽为 2 的幂，也不设 Triton V0 的
16,384 元素上限；测试包含 `N=32,769`。但这仅表示当前算法可正确处理更宽行，不代表该
简单单 Block/行策略在任意极宽行上都高效。

启动配置为 Grid `(M,)`、Block `(256,)`：每个 Block 负责一整行。线程 `t` 先读列
`t, t+256, t+512, ...` 并累加自己的 `x*x`。每个线程把局部和写到
`partial[t]` 共享内存，然后全体线程通过 256→128→…→1 的树形归约求出整行平方和。
每轮的加法之后都要 `__syncthreads()`，确保下一轮不会读到其他线程尚未写完的值。
归约结束后所有线程读取均方值和 `rsqrt`，再按相同的列分工计算输出。

输入会读取两遍：第一遍算 RMS，第二遍计算并写输出。这是刻意选择的简单基线；不把整行
留在寄存器里，避免长行造成寄存器压力。权重也被各行重复读取，实际缓存行为由硬件决定。

### 对照结果

RTX 3080 Ti、FP32、一次本机采样，`cuda_shared_256` 相比 Triton V0 4-warps：

| M × N | CUDA P50 | Triton P50 | CUDA / Triton |
|---:|---:|---:|---:|
| 1 × 4,096 | 10.240 μs | 5.120 μs | 2.00× |
| 32 × 4,096 | 10.240 μs | 6.144 μs | 1.67× |
| 256 × 4,096 | 17.408 μs | 14.336 μs | 1.21× |
| 1,024 × 4,096 | 60.416 μs | 45.056 μs | 1.34× |
| 512 × 8,192 | 68.608 μs | 46.080 μs | 1.49× |

这版更慢并不代表 CUDA 本身比 Triton 慢：这是初学用的朴素实现，用 8 道 Block 同步
完成归约。Triton 会替你生成布局对应的归约代码；两者算法表达相同，但线程协作实现不同。
完整数据在 [`CUDA FP32 Benchmark`](../../benchmarks/results/rmsnorm_cuda_rtx3080ti_fp32.csv)，
NCU 在 [`CUDA NCU Basic 报告`](../../benchmarks/results/rmsnorm_cuda_rtx3080ti_ncu_basic.md)。

### CUDA Shuffle 版本：降低 Block 级同步

`rmsnorm_shuffle_kernel` 把归约拆成两层：首先 8 个 Warp 各自用 5 条
`__shfl_down_sync` 指令把 32 个 lane 的局部和合并到 lane 0；8 个 Warp 的结果写入
共享内存。第一次 `__syncthreads()` 等待 8 个 Warp 都写好结果。然后 Warp 0 读取这 8 个
部分和（其余 24 个 lane 用加法单位元 0），再做 Warp 内 Shuffle；lane 0 写回总和，第二次
Block 同步后全体线程读取均方值并写输出。

与共享内存树形版相比，归约逻辑从 9 次 Block Barrier（1 次初始发布 + 8 轮树形步骤）降到
2 次，静态共享内存从 256 个 float（1,024 B）降为 8 个 float（32 B）。然而 RTX 3080 Ti
上的普通 Benchmark 五组数据几乎持平：`1024×4096` 两者均约 60.416 μs。这是有价值的
负结果：少了同步和共享内存不保证端到端时间下降，输入/输出访存、指令与调度等成本也
共同决定整体时间。NCU 的 DRAM Throughput 均约 82%，也没有证据支持把微小差异称为确定
提速。详细数据见 [`Shuffle CSV`](../../benchmarks/results/rmsnorm_cuda_shuffle_rtx3080ti_fp32.csv)
和 [`Shuffle NCU 报告`](../../benchmarks/results/rmsnorm_cuda_shuffle_rtx3080ti_ncu_basic.md)。

测试 Shuffle Kernel：

```powershell
python llm_kernels/rmsnorm/profile.py --provider cuda-shuffle --rows 1024 --columns 4096
ncu --set basic --kernel-name regex:rmsnorm_shuffle_kernel --launch-skip 10 --launch-count 1 python llm_kernels/rmsnorm/profile.py --rows 1024 --columns 4096 --provider cuda-shuffle
```

Benchmark 现在对 FP32、FP16 和 BF16 都比较 PyTorch、Triton 4/8 warps、CUDA shared 和
CUDA Shuffle。`1024×4096` 时 CUDA Shuffle 的 P50 分别约为 60.416、30.720、30.720 μs；
Triton 4-warps 分别约为 45.056、24.576、24.576 μs。低精度减少了每元素显存字节数，
所以该形状延迟降低；但并未让此 CUDA 实现超过 Triton。各 dtype 的逻辑带宽按各自元素
尺寸计算，仍是逻辑吞吐估算，不等于 NCU DRAM Throughput。完整结果在
[`三 dtype Benchmark CSV`](../../benchmarks/results/rmsnorm_cuda_dtypes_rtx3080ti.csv)。

## Triton V1：两阶段分块支持超宽行

Triton V0 一个 Program 需要在一个 block 里处理整行，并受最大 Block Size 限制。V1 用
两次 Kernel 启动解除这个限制：

1. Stage 1 Grid 为 `(NUM_CHUNKS, M)`。每个 Program 读取一行中的 4,096 列，FP32 累加
   平方和，写一个 FP32 partial。
2. Stage 2 仍按行和列块启动。每个 Program 读取该行的所有 partial（超过块数的 lane
   Mask 为 0），求出整行 `mean_square` 和 `inverse_rms`；然后读取自己负责的 4,096 个
   输入/weight 元素，写输出。

由于所有 partial 只有在 Stage 1 结束后才齐全，两个阶段之间需要 Kernel 边界。Stage 2
的每个输出块都会重复求和该行较短的 partial 列表，这是避免第三次 Kernel 启动的取舍。
当前每行最多 1,024 个 partial（最大支持 4,194,304 列）；V1 只接受宽度大于 16,384 的
输入，窄行应使用单 Kernel V0。两版本都支持连续 FP16/BF16/FP32，累加使用 FP32。

测试覆盖 `N=16,385、32,769、65,536`，含不规则尾块和三种 dtype。对 `M=256` 的 FP32
宽行，V1 对比 CUDA Shuffle：

| N | Triton V1 P50 | CUDA Shuffle P50 | 对照结论 |
|---:|---:|---:|---|
| 16,385 | 68.608 μs | 74.752 μs | V1 约快 1.09× |
| 32,769 | 127.984 μs | 151.552 μs | V1 约快 1.18× |
| 65,537 | 247.808 μs | 321.536 μs | V1 约快 1.30× |

V1 消除了 V0 的宽度限制，并在这些样例中优于逐线程循环读取整行的 CUDA 单 Block/行
版本；这并不表示它适合所有 Shape。`M`、partial 个数、两次启动成本都会影响选择。完整
计时在 [`V1 宽行 Benchmark CSV`](../../benchmarks/results/rmsnorm_v1_wide_rtx3080ti_fp32.csv)，
Stage 1/2 指标见 [V1 NCU Basic 报告](../../benchmarks/results/rmsnorm_v1_wide_rtx3080ti_ncu_basic.md)。

运行命令：

```powershell
python llm_kernels/rmsnorm/benchmark_wide.py
python llm_kernels/rmsnorm/profile.py --provider triton-v1 --rows 256 --columns 32769
```

V1 Stage 2 在 NCU 中测得 80 registers/thread、50% 理论 Occupancy；两阶段 P50 仍优于
本组 CUDA 对照。不要据此单独追逐 Occupancy，需结合 Kernel 总时间、访存和工作负载规模。

高层入口 `rmsnorm_triton_auto` 会依据 `N` 自动选择实现：`N≤16,384` 使用 V0，较宽行
使用 V1。若学习或测量时需要固定算法，可以直接调用 `rmsnorm_triton`（V0）或
`rmsnorm_triton_v1`。Auto Dispatch 只按当前已验证的行宽分界路由，不是基于 GPU 架构、
dtype 或完整性能模型的自动调优器。

### 代码阅读顺序

1. 在 `cuda_impl.py` 看输入约束、Extension 如何从 `.cpp/.cu` 编译加载。
2. 在 `bindings.cpp` 找 Python 函数如何绑定到 C++ 函数。
3. 在 `.cu` 先确认 Grid/Block：`blockIdx.x` 是行号，`threadIdx.x` 是线程号。
4. 逐段追踪 `local_square_sum → partial[] → partial[0] → inverse_rms → row_output`。
5. 对照 shared 与 shuffle 两个 Kernel：Warp Shuffle 仅能交换同一 Warp 内的值，为什么
   仍然要把 8 个 Warp 的部分和写入共享内存并同步？
6. 运行 test/benchmark/profile，对比结果而非只看代码“看起来并行”。

## 当前进度

已完成 PyTorch 参考实现、Triton V0/V1、CUDA C++ FP32/FP16/BF16 共享内存与 Warp Shuffle
版本、按宽度自动分派、正确性测试、Benchmark 和 NCU 基础分析。Warp Shuffle 版将共享内存从 1,024 B 降至
32 B、把 Block 级同步从 9 次降至 2 次，但本轮五组 Benchmark 未观察到稳定的端到端提速。
V1 两阶段分块已覆盖超宽行；基于硬件指标的自动调优尚未实现。

## 运行测试

在仓库根目录运行：

```powershell
python llm_kernels/rmsnorm/test.py
python llm_kernels/rmsnorm/benchmark.py
python llm_kernels/rmsnorm/benchmark_wide.py
python llm_kernels/rmsnorm/profile.py --provider cuda --rows 1024 --columns 4096
python llm_kernels/rmsnorm/profile.py --provider triton-v1 --rows 256 --columns 32769
```

测试包含手算样例、不同形状和 dtype、PyTorch 参考实现自动微分、PyTorch/Triton/CUDA 对照
及边界参数检查。CUDA Extension 当前是仅前向的教学 Kernel，不注册 autograd backward。
Benchmark 的带宽只按最低逻辑流量估算，不能当成 NCU 测出的 DRAM 实际流量；PyTorch
参考公式会产生多个中间运算，而 Triton V0 将这些步骤融合在一个 Kernel 中。

## 初次 Benchmark 记录

RTX 3080 Ti、FP32、单次运行的示例：

| M | N | PyTorch P50 | Triton V0，4 warps | Triton V0，8 warps | 4-warp 逻辑带宽 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,096 | 20.480 μs | 5.120 μs | 5.120 μs | 9.60 GB/s |
| 32 | 4,096 | 26.624 μs | 6.144 μs | 6.144 μs | 173.33 GB/s |
| 256 | 4,096 | 50.176 μs | 14.336 μs | 13.312 μs | 586.29 GB/s |
| 1,024 | 4,096 | 161.696 μs | 45.056 μs | 45.056 μs | 745.09 GB/s |
| 512 | 8,192 | 163.840 μs | 45.056 μs | 45.056 μs | 745.45 GB/s |

这是教学基线的一次本机结果，不是跨设备保证。原始计时数据保存在
[`rmsnorm_v0_rtx3080ti_fp32.csv`](../../benchmarks/results/rmsnorm_v0_rtx3080ti_fp32.csv)。
`M=1` 时只有一个 Program，GPU 大量
SM 没有同一时刻的行任务可执行，因此有效吞吐很低；随着 M 增大，更多行可并行，逻辑
带宽上升。4→8 warps 只在 `M=256,N=4096` 这一点快约 7%，其他表中规模基本持平，
因此目前没有根据把 8 warps 设为通用默认值。大尺寸接近本机流式读写的水平，但
**仅靠这个 Benchmark 还不能证明**实际
DRAM Throughput、Occupancy 或寄存器压力，需要 NCU 指标验证。PyTorch P50 较高的一个
重要原因是参考公式拆成了多个 Tensor 运算；Triton V0 把归约与缩放融合为一次 Kernel。

## NCU 采样

先单独运行目标，检查输入形状与结果正确，再从仓库根目录采集一个 Kernel：

```powershell
python llm_kernels/rmsnorm/profile.py --rows 1024 --columns 4096
ncu --set basic --kernel-name regex:rmsnorm_shared_kernel --launch-skip 10 --launch-count 1 python llm_kernels/rmsnorm/profile.py --rows 1024 --columns 4096 --provider cuda
```

`profile.py` 默认预热 10 次、同步后再额外运行一次；NCU 跳过前 10 次并捕获下一次，
避免把首次 JIT 编译混入要看的 Kernel。`--set basic` 适合入门查看耗时、线程配置和
基础吞吐指标；如果要查具体指标，再按问题单独扩大采集范围。完整结果和解读见
[`RMSNorm V0 NCU Basic 对照`](../../benchmarks/results/rmsnorm_v0_rtx3080ti_ncu_basic.md)。

对 `M=1024,N=4096` 的 NCU 对照（CC 8.6）：

| 配置 | Block 大小 | 寄存器/线程 | 理论占用率 | DRAM Throughput | Compute Throughput | NCU Duration |
|---|---:|---:|---:|---:|---:|---:|
| 4 warps | 128 threads | 80 | 50.0% | 86.61% | 6.23% | 42.94 μs |
| 8 warps | 256 threads | 48 | 83.33% | 83.88% | 8.62% | 44.99 μs |

这里 DRAM 吞吐高而 SM 计算吞吐低，说明这个形状在当前配置下主要受内存通路限制。
8 warps 减少了每线程寄存器、提高理论占用率，但 NCU 中 DRAM 吞吐没有提高，耗时也没有
改善。结论不是“8 warps 无用”，而是这个形状的瓶颈已经更靠近内存侧；占用率只是解释
执行资源的一个指标，不能单独当成性能目标。NCU Duration 会受采集重放影响，不应与
普通 Benchmark 的 P50 混成同一组计时。

## 阅读代码时可以自问

1. `mean(..., keepdim=True)` 为什么要保留最后一维？
2. 低精度输入为什么先转成 FP32 再平方和归约？
3. `weight` 为什么能从 `[hidden]` 广播到 `[batch, hidden]`？
4. 若错误地除以补齐后的 `BLOCK_SIZE`，`N=17` 的结果会怎样偏差？
5. 这个单 Program-per-row 版本处理 32,768 个通道时会遇到什么资源问题？
6. 哪些操作融合进一个 Kernel 后可以消除中间 Tensor 和额外的全局内存流量？
7. 为什么大矩阵上的高逻辑带宽仍不能替代 NCU 的 DRAM Throughput 指标？
