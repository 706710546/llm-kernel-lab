# LLM Kernel Lab 阶段复习与后续学习路线

更新日期：2026-10-08。已纳入 CUDA Softmax Online 两遍读取实验。
本文面向 C++ 基础较好、Python/Triton/CUDA 正在入门的学习者。

> 当前最重要的成果，不是“写了几个算子”，而是开始建立：从数学与数据映射出发，
> 用正确性测试、Benchmark 和硬件指标解释性能的工程方法。

阅读顺序：先看第 1、2 节确认项目位置，再逐算子复习第 3 节；第 5、6 节用于准备
项目介绍与面试，第 7、8 节用于安排后续实践。不要求一天全部学会。

## 1. 项目目前进行到哪一步？

已完成四个算子，当前停在 **Softmax 的 CUDA Online 两遍读取实验**。
这不是完整推理框架，也不是编译器项目，而是一个有可复现实验的 GPU 算子实验室。

| 模块 | 已完成实现 | 已完成验证 | 尚未完成 |
|---|---|---|---|
| Vector Add | PyTorch、Triton、FP32 CUDA | 正确性、带宽 Benchmark、Block Size 实验、NCU | 无需继续追逐微小差异 |
| Row Sum | PyTorch、Triton V0/V1 | 边界与低精度测试、Benchmark、两阶段 NCU | 独立 CUDA Row Sum |
| Matrix Transpose | PyTorch 连续副本、Triton Tile、CUDA Naive/Tiled | 正确性、Tile 实验、Padding/Bank Conflict 对照、NCU | 通用任意 Stride 支持 |
| Softmax | PyTorch、Triton V0/V1/V2、FP32 CUDA 基线/Shuffle/Online | 数值稳定性与边界测试、版本 Benchmark、Spill/流量/归约实验 | 自动形状分派、更多 dtype/Stride 支持 |

各模块中的 V0/V1/V2 是该模块自己的版本号，不是全仓库发布版本。
例如 Reduction V1 是两阶段求和，Softmax V1 是三阶段分块归一化。

尚未实现的重点：RMSNorm/LayerNorm、MatMul、Tensor Core 实验、Attention。
尚不能宣称完成：FlashAttention、LLM 端到端推理优化、生产级算子库、编译器 Pass。

现有统一结构无需大改：

```text
torch_impl.py       可信参考结果
triton*_impl.py     Triton 实现及不同算法版本
cuda_impl.py        Python 到 CUDA Extension 的包装
csrc/              C++ 绑定、CUDA Kernel、Host Launcher
test.py            正确性和边界检查
benchmark.py       预热后计时与吞吐比较
profile*.py        针对某个 Kernel 的 NCU 采样入口
README.md          模块的中文原理与运行说明
benchmarks/results 实验数据与分析报告
```

## 2. 已经建立的完整工作流程

对每一个新算子，都按下面的顺序完成：

1. 写出数学定义与输入输出形状，确认比较的是同一项工作。
2. 使用 PyTorch 建立参考实现，而不是用自己的两个 Kernel 互相证明正确。
3. 设计 GPU 映射：一个 Program/Block 负责哪些行、列或 Tile？
4. 处理边界：空 Tensor、短行、非整块尺寸、类型与连续性限制。
5. 测试结果、输出形状和必要的数学性质，例如 Softmax 每行和接近 1。
6. 编译与预热后 Benchmark，记录 P50、P20、P80，而不是只截取最快一次。
7. 提出瓶颈假设，再用 NCU 检查吞吐、资源占用和实际流量。
8. 每次集中改变一个主要因素，重新跑正确性和 Benchmark。
9. 保存原始数据、条件、命令与结论，说明哪些结论不能外推。

最新 Shuffle 实验也同时修复了基线同步问题，因此没有把历史基线直接拿来对比：
正式比较使用的是同轮、修复同步后的两个版本。这是控制变量之外的正确性前提。

## 3. 四个算子的重点复习

### 3.1 Vector Add：从 GPU 映射到带宽上限

数学：`z[i] = x[i] + y[i]`。每个输出独立，不需要跨线程合并结果。

CUDA 映射：`i = blockIdx.x * blockDim.x + threadIdx.x`。
Triton 映射：`offsets = program_id * BLOCK_SIZE + arange(0, BLOCK_SIZE)`。

如果 `N=2050、BLOCK_SIZE=512`，需要 5 个 Program。最后一个负责逻辑位置
2048～2559，其中只有 2048、2049 合法，其余 510 个位置必须 Mask。

FP32 每个元素读取两个输入、写一次输出，共 12 B；计算一次加法，算术强度为
`1/12 FLOP/B`。大尺寸预期主要受显存带宽限制，小尺寸更容易被固定启动和计时
开销影响。

项目证据：大尺寸有效带宽约 809～826 GB/s；一轮 NCU 的 DRAM Throughput
为 91.67%、SM Throughput 为 7.53%。Block Size 改变了并行映射，却没有明显
抬高最终带宽平台。

应该记住：连续地址有利于合并访存；低算术强度算子不能靠增加线程无限加速。
不过“每个线程隔步处理元素”本身不等于非合并访存，关键是同一条访存指令中
相邻线程访问的地址。在 CUDA Softmax 中，每线程步长为 256，但同轮的相邻线程
仍访问相邻列。

复习入口：[模块文档](../llm_kernels/vector_add/README.md)、
[NCU 报告](../benchmarks/results/vector_add_rtx3080ti_ncu_basic.md)。

### 3.2 Row Sum：多个元素如何合作得到一个结果？

输入 `[M,N]`，输出 `[M]`，`y[row] = Σ x[row,column]`。

V0：每行一个 Program，整行加载后 `tl.sum`。补齐到 2 的幂，非法位置补 0。
V1：每行按 1024 元素分块，先写 FP32 局部和，再启动第二个 Kernel 合并。

重要代价：分块带来更多并行任务和较小工作集，但也带来中间缓冲、额外读写、
一次额外 Kernel 启动。因此 V1 的价值包括支持超宽行，而不只是追求速度。

FP16 输入先用 FP32 累加，减少中间精度损失。浮点加法不严格满足结合律，
并行归约与串行顺序可能不同，正确性一般用容差比较，不要求逐位相同。

项目证据：V0 与 V1 Stage 1 的 DRAM Throughput 均约 92.8%；更高 Occupancy
没有降低主阶段耗时。V1 第二阶段约 3.58 μs，虽利用率低，却不是主要耗时来源。

复习入口：[模块文档](../llm_kernels/reduction/README.md)、
[两阶段 NCU 报告](../benchmarks/results/reduction_v0_v1_rtx3080ti_ncu_basic.md)。

### 3.3 Matrix Transpose：数据布局比算术更重要

输入 `[M,N]`，输出 `[N,M]`：`y[column,row] = x[row,column]`。
线性位置分别是 `row*N+column` 与 `column*M+row`。

公平比较很重要：`x.transpose(0,1)` 通常只改变 Shape/Stride，返回 View；
项目使用 `.contiguous()` 生成连续副本，才与真正搬运数据的 Kernel 对齐。

Naive CUDA 的相邻线程连续读入，却跨行写出，写入难以有效合并。
Tiled CUDA 通过共享内存中转，把“连续读”和“连续写”结合起来。
线程需要读取其他线程写入的 Tile，必须先执行 Block 同步。

`tile[32][33]` 的多一列不是扩大计算范围，而是改变共享内存行跨度，避免转置
读列时集中访问相同 Bank。在本实验的 FP32、32×32 Tile 映射下有效，不能
把“加一列消除冲突”当成所有类型和访问布局的通用定理。

项目证据：无 Padding 的一次 NCU 采样有 16,252,928 次 Shared Load Bank
Conflict，Padding 后为 0；Load Wavefront 数约降低 32.2 倍。但整体运行时间
没有改善 32 倍，因为共享内存阶段只是 Kernel 总成本的一部分。

复习入口：[模块文档](../llm_kernels/transpose/README.md)、
[CUDA 与 Bank Conflict 报告](../benchmarks/results/transpose_cuda_rtx3080ti_ncu_basic.md)。

### 3.4 Softmax：稳定性、Spill、在线统计与 Warp 协作

稳定公式：

```text
m = max(x)
l = Σ exp(xᵢ-m)
yᵢ = exp(xᵢ-m)/l
```

减去最大值使指数输入不大于 0，避免大正数导致的指数溢出。Mask 位置补
`-inf`：不影响最大值，且在有限行最大值下其指数贡献为 0。本项目主要验证
有限输入；不要把它宣称为对所有 NaN/Inf 输入都定义了特殊处理语义。

| 版本 | 并行组织 | 输入扫描 | 主要取舍 |
|---|---|---|---|
| Triton V0 | 每行一个 Program，整行工作集 | 源码一次加载 | 中短行快，宽行可能 Spill |
| Triton V1 | 三个 Kernel，块并行、行合并、块写出 | 两次 | 多启动与缓冲，避免过大工作集 |
| Triton V2 | 每行一个 Program，在线逐块统计 | 两次 | 一个 Kernel，同一行块间有依赖 |
| CUDA V0 | 每行一个 Block，共享内存树形归约 | 三次 | 清楚易懂，但归约同步较多 |
| CUDA Shuffle V1 | Warp 内 Shuffle、Warp 间共享内存 | 三次 | 归约更轻，未减少输入读取 |
| CUDA Online V2 | 每线程在线维护 `(m,l)`，再跨 Warp/Block 合并 | 两次 | 宽行少读一遍，短行增加统计成本 |

“源码只读一次”不是实际 DRAM 只读一次：Spill、缓存与内存事务会改变真实流量。

跨块统计必须换算到相同最大值基准：局部块有 `(mₖ,lₖ)`，整行归并时
`m=max(mₖ)`，`l=Σ lₖ*exp(mₖ-m)`。

在线更新也是相同原理：

```text
m_new = max(m, block_max)
l_new = l*exp(m-m_new) + Σ本块 exp(xᵢ-m_new)
```

如果新块出现更大的最大值，必须缩放旧的指数和。在线表示跨块只保留 `(m,l)`
统计量；当前块仍有临时向量。独立 Softmax 等最终分母确定后仍要再读输入，
因此 Online Softmax 不等于 FlashAttention，也不等于已经单遍写出了所有概率。

Spill 实验：宽行 V0 一轮 NCU 测得 Local Load 701.50 MB、Local Store
510.13 MB，实际 DRAM 流量约 1460 MB；V1/V2 将工作集分块后未观测到这些
Local Memory 访问。V0 的 Occupancy 更高，却更慢。

Shuffle 实验：256 线程为 8 个 Warp，各 Warp 内先用偏移 `16,8,4,2,1`
的 Shuffle 归约，再用 8 个共享内存槽合并。当前 Shuffle 的全 Warp 掩码
依赖所有相应 lane 都参与，不能随意照搬到部分线程执行的分支中。

共享内存从 1024 B 降到 32 B，源码中的 Block Barrier 从 19 次降到 6 次。
短行 `4096×128` 同轮 P50 从 30.72 μs 到 17.41 μs（约 1.76×）；宽行
`1024×32768` 两版仍约 683～684 μs，读流量仍是三份输入。

这次还修复了旧版读取 `row_max` 后、复用共享缓冲前缺少保护同步的潜在竞态。
通过数值测试不证明代码没有竞争；现有报告不宣称已完成全面竞争检测。

复习入口：[模块文档](../llm_kernels/softmax/README.md)、
[Spill 报告](../benchmarks/results/softmax_v0_rtx3080ti_ncu_basic.md)、
[Shuffle 报告](../benchmarks/results/softmax_cuda_shuffle_rtx3080ti.md)。

## 4. 必须分清的概念

### 4.1 执行单位与存储层次

- CUDA Thread 是线程，Warp 是线程执行组，Block 是可共享数据与协作的线程组。
- Grid 包含本次启动的 Blocks，SM 是执行硬件；Block 不是 SM 的同义词。
- Triton Program 是程序实例，不能把 `BLOCK_SIZE=1024` 解释成 1024 个线程。
  当前 Softmax V2 的一次 NCU 采样显示一个 Program 对应 128 个底层线程。
- Register、Shared Memory、Global Memory 是不同存储资源。
- CUDA Local Memory 虽是线程私有的地址空间，却不是“就近的快速共享内存”；
  Spill 到它可能引入显存和缓存流量。

### 4.2 三个性能数字不是一回事

- 有效带宽：按指定逻辑字节数除以耗时，是实验中人为选定口径的吞吐指标。
- NCU DRAM Throughput：实际 DRAM 吞吐相对工具所用峰值口径的指标。
- Occupancy：驻留的活跃 Warp 数相对硬件可容纳上限，不是任务管理器里的
  GPU 使用率，也不是所有线程持续做有效工作的比例。

Softmax CSV 按“一读一写”计算有效带宽；Triton V1/V2 与 CUDA Online
实际两读一写，CUDA V0/Shuffle 三读一写。所以不能把 CSV 的 GB/s 当成
各版本实际显存吞吐。

### 4.3 如何读一次性能实验？

先确认 GPU、dtype、Shape、输入是否连续、算法版本，再看 P50 与 P20/P80。
分位数是重复计时的分布信息，不是严格的置信区间。微小差异需要重复多轮、
更换 provider 顺序或交错测量，不能据一次 1% 差异宣布稳定胜出。

GPU 执行是异步的，普通 CPU 时钟包住一次调用不一定等于 Kernel 耗时。
项目用 `triton.testing.do_bench` 测量；第一次扩展编译与 JIT 应在计时前完成。
这些结果不是包含 Python、传输和完整模型的生产端到端延迟，也不是固定设备
输出缓冲下的纯 Kernel 下界。比较口径需要明确。

NCU 可能通过 replay 多次执行采样，会改变运行环境。版本延迟比较以独立
Benchmark 为主，NCU 用来解释资源与流量。多阶段 Kernel 要逐阶段采样，
不能把其中一段当成完整算子。

### 4.4 Python 先学哪些就够用？

先掌握函数与返回值、for 循环、列表/元组/字典、解包、条件判断、模块导入、
Path、异常和 f-string。再补 Tensor 的 shape/dtype/device/stride/contiguous。
不必先学 Web 框架或复杂 Python 元编程。

Benchmark 中 `lambda: implementation(x)` 是把“待执行动作”交给计时器，
不是立即运行。`providers.items()` 遍历名字和函数，`p50,p20,p80 = ...`
是解包。当前代码在每轮内立即计时，能使用这一轮的 `x`；若把 lambda 全部
存起来到循环结束后才调用，就要注意 Python 闭包晚绑定的问题。

## 5. 对求职有什么实际帮助？

以下是项目能力与岗位工作内容的映射，不是招聘行情或录用保证。

| 方向 | 现有项目能提供的证据 | 仍需补充 |
|---|---|---|
| GPU 算子开发与优化 | CUDA/Triton 映射、归约、Tile、访存、Profiler 对照 | MatMul、Tensor Core、更多类型与形状覆盖 |
| AI 推理性能工程 | 分辨瓶颈、稳定数值、减少流量的意识、实验复现 | 真实模型接入、融合算子、端到端性能分析 |
| AI/GPU 编译器 | 理解布局、Spill、资源限制、生成代码的性能后果 | IR、Pass、Lowering、合法性分析与编译器测试 |
| 通用 C++ 工程 | Python/C++ 绑定、编译调试、边界检查、Git 交付 | 更完整工程测试、可移植性、构建与错误处理 |

对你更有价值的组合是：**这个项目证明你理解 GPU 上什么样的代码执行得好；
第二个编译器项目证明你能够把高层计算转换、优化并生成这样的代码。**

当前项目不能单独证明你熟悉 LLVM/MLIR，写过 Triton Kernel 也不等于开发过
Triton 编译器。继续保持两个仓库独立，不把整个算子库重构成编译器框架。

### 5.1 简历可用描述模板

以下描述只在你能独立解释并复现实验后使用；“实现/分析”的参与范围要如实写：

> LLM Kernel Lab：基于 PyTorch、Triton 与 CUDA 构建四类 GPU 算子的正确性、
> Benchmark 与 Nsight Compute 分析流程；对比连续访存、分块归约、共享内存
> 转置及 Online Softmax 的性能与资源代价。

可以选两个最能讲清楚的结果作为子项：

- 在 RTX 3080 Ti 的 FP32 转置实验中，通过共享内存 Padding 将特定 32×32
  Tile 的 Shared Load Bank Conflict 从约 1625 万次降为 0，记录流量与延迟对照。
- 在 FP32 `1024×32768` Softmax 上，定位宽行单 Program 的寄存器 Spill，
  用分块方案将同轮 P50 从 1881.60 μs 降到 501.76 μs（约 3.75×）。
- 用 Warp Shuffle 改进 CUDA Softmax 归约，在 FP32 `4096×128` 上相对
  修复同步后的共享内存基线提升约 1.76×；说明宽行几乎不受益的原因。

不要写“所有算子加速 3.75 倍”“性能全面超过 PyTorch”“实现 FlashAttention”
或“精通 CUDA/编译器”。每个数字必须带比较对象、形状、dtype 和硬件条件。

### 5.2 面试时用一个闭环讲故事

按五句话组织：任务是什么 → 原方案怎么映射 → 看到什么退化 → 什么指标支持
原因判断 → 改了什么、效果怎样、还有什么限制。

最推荐讲 Softmax Spill 与 Transpose Padding。前者体现算法/资源取舍，后者
体现硬件访问布局。Shuffle 实验可以作为“优化不总有效”的补充：没有宽行
收益也有价值，因为它帮助排除错误优化方向。

## 6. 阶段复习问题：用自己的数据作答

先不看文档，用 2～3 分钟解释，再回到源码和报告核对。

1. 为什么 Vector Add 很难靠改 Block Size 大幅加速？
   要点：低算术强度、带宽平台、NCU 的 DRAM/SM 指标。
2. Triton 的 BLOCK_SIZE 与 CUDA Threads/Block 有什么区别？
   要点：数据 Tile 大小与底层线程数不同，需要结合配置和实际编译结果。
3. 为什么求和 Mask 补 0、Softmax Mask 补 -inf？
   要点：归约的中性元素与稳定指数的零贡献。
4. FP16 输入为什么先转换到 FP32？为什么仍可能和 PyTorch 不逐位相同？
   要点：累加精度、归约顺序、最终输出舍入。
5. 为什么直接比较 transpose View 与 CUDA 搬运 Kernel 不公平？
   要点：逻辑元数据操作与实际数据重排的任务差别。
6. 转置 Padding 为什么消除冲突？为何整体性能没有提升 32 倍？
   要点：具体 Bank 映射、局部指标与总体耗时的关系。
7. Occupancy 更高为什么 Softmax 反而更慢？
   要点：Spill 带来的额外流量，而不是单看活跃 Warp。
8. Online Softmax 为什么要缩放旧的 l？为什么仍读两遍？
   要点：最大值基准变化、最终归一化分母尚未知。
9. Shuffle 能不能替代所有 __syncthreads？
   要点：Warp 内交换不能保证跨 Warp 共享内存交接。
10. 为什么全 Warp mask 不能随便放进分支？
    要点：被命名的相关线程必须满足参与要求，当前实现让整 Warp 参与。
11. 数值测试通过能证明共享内存无竞态吗？
    要点：不能；仍要分析访问与同步，必要时增加工具检查。
12. Shuffle 为什么只明显加速短行？Online 宽行为什么受益、短行为什么变慢？
    要点：Shuffle 没减少读取；Online 少读一遍但增加状态更新与合并计算。

掌握标准不是背答案：能画出一次映射、指出对应代码、找到支撑指标，并说明
结论适用范围。回答不出来的地方就是下一次实践任务，而不是需要记更多术语。

## 7. 后续项目怎么推进？保持架构，只补关键能力

下面是下一阶段的建议路线，不是已经实现的功能。每个阶段都以验收条件结束，不按“写完
一个文件”结束。一个实践单元可安排约 2～3 小时，基础补课与硬件问题可能增加
时间；不要把这些估计当成固定完结日期。

| 顺序 | 工作内容 | 验收结果 | 建议投入 |
|---|---|---|---|
| 1 | RMSNorm，然后按需要扩展 LayerNorm | FP32 累加、稳定性、Triton/CUDA 至少一组对照 | 3～4 个单元 |
| 2 | MatMul 基线与 Tile 复用 | 明确 M/N/K、FLOPs、精度模式、参考实现与资源分析 | 4～6 个单元 |
| 3 | 教学版 Attention | 先朴素基线，再研究融合与在线统计；明确支持边界 | 3～5 个单元 |
| 4 | 发布整理 | 统一运行说明、汇总结果、限制、简历与复现演示 | 2～3 个单元 |

### 7.1 已完成的衔接：CUDA Softmax 两遍读取

Online 版本保留了旧 CUDA 基线和 Shuffle 版作对照，用在线 `(m,l)` 更新把
统计阶段合为一次扫描，再第二遍写输出。重点难点不是复制 Triton 公式，
而是不同线程局部状态如何稳定地合并。

NCU 测得宽行输入读取约 268.47 MB，接近两份输入；历史 Shuffle 约 402.68 MB。
三轮交替顺序实验显示宽行较快、短行较慢。这说明减少流量的收益与额外指数
计算/依赖的成本依 Shape 而变，不能无条件切换。完整数据见
[CUDA Online 报告](../benchmarks/results/softmax_cuda_online_rtx3080ti.md)。

### 7.2 RMSNorm/LayerNorm：从归约到融合

先做 RMSNorm，复用平方和归约、FP32 累加与按行映射。再考虑 LayerNorm
中的均值、方差及稳定计算。学习收益是把多步计算合成一个 Kernel，减少中间
Tensor 与流量，而不是只追求算子数量。

### 7.3 MatMul：补齐计算密集型能力

目前实验大多偏访存和归约。MatMul 能加入输入复用、计算强度、K 维循环、
Tile 资源取舍以及 Tensor Core/精度模式。先实现易解释的正确基线，再研究
共享内存或 Triton Tile 优化，不直接跳到复杂流水线。

与 PyTorch 比较时固定 dtype、累加方式及 TF32 等设置；不能把精度模式不同
造成的差别当成算法优化收益。使用目标 GPU 实际支持的特性，不照搬更新架构
专有优化。官方 [Triton MatMul 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
可作逐步对照资料。

### 7.4 Attention：最后整合，而不是现在跳跃

先实现可验证的 `softmax(QKᵀ/√d)V` 参考路径，再研究 Tile、融合和在线输出
更新。需要明确是否支持 Causal Mask、Batch/Head、FP16、反向等。
初版不必覆盖全部功能，更不能仅复用 Online Softmax 名称就声称实现了
FlashAttention。若时间紧，先发布含 Norm 与 MatMul 的完整阶段版本，Attention
作为独立扩展里程碑。

### 7.5 有用但不应抢占主线的工作

- 将重复的 Windows Extension 加载逻辑抽成小工具，避免三个模块长期复制；
  等算子主线稳定再做，不需要重建整个架构。
- 小步统一 Benchmark 的输出、seed、环境记录与 provider 顺序。
- 检查 CUDA Stream 和错误处理，按需加入竞争/越界检测工具验证。
- 自动分派和 Autotuning 建立在多轮数据上，不能凭一组形状写死普适阈值。
- 独立 CUDA Row Sum 可作为归约复习练习，不必为了对称性阻塞 Norm/MatMul。

## 8. 接下来一周如何复习？

这是可调整的 7 个学习单元，不要求连续 7 个自然日。每个单元都留下一页自己的
笔记或一次可复现输出，比只浏览代码更重要。

1. Vector Add：手算 Grid 和末块 Mask，运行 test/benchmark，解释带宽公式。
2. Reduction：画出 `[2,2500]` 的两阶段映射，说明额外成本与 FP32 中间值。
3. Transpose：画 4×4 转置与 Tile 中转，比较 View/副本，读 Padding 报告。
4. Softmax 数学：手算 `[1,2]` 与 `[3,4]` 两块的最大值基准换算。
5. Softmax 性能：对照 V0/V1/V2，估算逻辑流量并解释 Spill 证据。
6. CUDA Shuffle：逐行讲解 warp_reduce/block_reduce，解释每道同步保护谁。
7. 项目演示：不看本文，用 5 分钟介绍一个成功优化和一个效果不明显的优化。

运行命令从仓库根目录执行：

```powershell
python llm_kernels/vector_add/test_cuda.py
python llm_kernels/reduction/test.py
python llm_kernels/transpose/test_cuda.py
python llm_kernels/softmax/test.py
python llm_kernels/softmax/benchmark.py
```

NCU 前先确定要回答的问题，不要每次都盲目采集所有指标。改变代码后先做
正确性检查；运行慢不等于错误，运行快也不等于正确。

## 9. 编译器第二项目怎么与当前项目衔接？

等你能独立解释一个访存优化、一个归约优化和一个资源退化后，再把当前 Kernel
作为第二项目的执行后端或回归用例，不把两个仓库混在一起。

第二项目优先建立小型完整闭环：高层计算表示 → IR → 一个有明确合法性条件的
优化 Pass → Lowering → 可运行结果 → 正确性和性能回归。
选择 Tile、融合或布局变换中的一个点深入，而不是一开始声称支持完整 LLM。

MLIR 的 [Toy 官方教程](https://mlir.llvm.org/docs/Tutorials/Toy/)
提供从语言表示到变换与 Lowering 的学习路径。它是基础教学，不是现成的高性能
GPU 编译器项目；后续要自己选择优化问题并建立测试。

两项目的共同接口是“可解释的性能后果”：为什么这种布局利于合并访存？
为什么融合减少中间流量？为什么 Tile 过大会增加资源压力？这些问题能把
编译器变换与真实硬件连接起来。

## 10. 资料怎么读？

优先顺序：自己的源码和实验 → 中文模块说明 → 对应官方小节 → 新实验。
英文资料可按术语和短段落阅读，不必先通读一本指南。

- [CUDA 12.6 Programming Guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)：
  与本地 Toolkit 主版本一致，按 Thread Hierarchy、Shared Memory、Synchronization、
  Warp Shuffle 小节查阅，而非从头逐页读。
- [Triton Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：
  与自己的 V0 对照，关注工作集、归约和数据流量；不要把教程条件下的性能结论
  无条件套到本项目。
- [Triton MatMul 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：
  留到 MatMul 阶段，先理解 Tile 与 K 循环，再看参数调优。
- [MLIR Toy 教程](https://mlir.llvm.org/docs/Tutorials/Toy/)：
  留作第二项目的 IR/Pass/Lowering 基础资料。

最后的复习标准：**能自己说明算法、画出映射、运行测试、读懂计时、用指标
解释优化与局限。**这比掌握更多库名、背更多 GPU 术语更能支撑可信的项目经历。
