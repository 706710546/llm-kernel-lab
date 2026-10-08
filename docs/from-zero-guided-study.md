# LLM Kernel Lab：从零开始、跟着代码学的完整手册

适用对象：有 C++ 基础，但隔了一段时间没碰 Python、Triton、CUDA，希望重新掌握本项目的人。
更新日期：2026-10-08。文中的性能数字来自 RTX 3080 Ti 上的特定实验，不是其他机器的承诺。

> **怎么用本文：**不要从头到尾只阅读一遍。每一章依次完成「手算 → 打开指定代码 → 运行命令 →
> 解释输出 → 完成检查点」。全部走通后，你应当能够独立讲解现有实现及其证据；“完全掌握”仍需要
> 自己修改代码、复现测量和处理新形状，无法靠被动阅读保证。

本手册不覆盖尚未实现的 RMSNorm、MatMul 或 Attention，也不把 Online Softmax 称为 FlashAttention。
另有一份偏项目总结和求职表达的[阶段复习文档](project-review-and-learning-roadmap.md)；建议先按本手册动手，
再用那份文档复盘。

## 0. 先建立全局地图

仓库做的事情：用 PyTorch 写参考结果，分别用 Triton/CUDA 实现 GPU 算子，测试正确性、
测 Benchmark，再用 NCU（Nsight Compute）解释资源使用和瓶颈。

```text
数学定义 ─→ PyTorch 参考实现 ─→ Triton/CUDA 映射
                                    │
                                    ├─ 正确性与边界测试
                                    ├─ Benchmark：延迟与逻辑吞吐
                                    └─ NCU：实际 GPU 指标与瓶颈解释
```

当前四个模块的推荐顺序：

| 顺序 | 模块 | 先掌握什么 | 难点 |
|---:|---|---|---|
| 1 | Vector Add | Grid、Program、Mask、合并访存 | 小尺寸启动开销与大尺寸带宽 |
| 2 | Row Sum / Reduction | 归约与两阶段分块 | 加法顺序、FP32 中间结果 |
| 3 | Matrix Transpose | 二维索引、Tile、共享内存 | View 与真实搬运、Bank Conflict |
| 4 | Softmax | 稳定归一化、在线状态 | Spill、两遍/三遍读取、Warp 协作 |

不要混淆模块版本号：Reduction V1 是**两阶段行求和**，Softmax V1 是**三阶段分块
Softmax**。两者的 V1 只是各自模块内部的演进标签。

### 0.1 文件各做什么？

以 `llm_kernels/softmax/` 为例：

| 文件 | 作用 | 阅读时先问的问题 |
|---|---|---|
| `torch_impl.py` | 可信的 PyTorch 参考结果 | 数学任务到底是什么？ |
| `triton_impl.py`、`triton_v*_impl.py` | Triton GPU 实现 | 一个 Program 负责什么？Grid 怎么定？ |
| `cuda_impl.py` | Python 包装与 Extension 编译加载 | 如何从 Python 调用 C++/CUDA？ |
| `csrc/bindings.cpp` | 将 C++ 函数暴露给 Python | Python 里看到的函数名是什么？ |
| `csrc/*.cu` | CUDA Kernel 与 Host Launcher | Thread/Block 如何映射？何时同步？ |
| `test.py`、`test_cuda.py` | 正确性与边界检查 | 参考结果、误差、空输入怎么处理？ |
| `benchmark.py` | 预热后计时与有效带宽 | 测了什么，没测什么？ |
| `profile.py`、`profile_cuda.py` | NCU 目标入口 | 捕获了哪一次 Kernel？ |
| `README.md` | 中文原理与实验结论 | 版本为什么演进？ |

不是每个模块都拥有上表每个文件；例如 Reduction 尚未实现 CUDA 版。
原始数据和 NCU 记录在 `benchmarks/results/`。查看一个结果时，先核对数据文件名称中的算子、
版本、GPU、dtype，再核对文档中写的 Shape。

## 1. 第一次坐到电脑前：环境和运行约定

以下命令在 **PowerShell** 中执行。复制时逐条运行，先不要改代码。

### 步骤 1：进入仓库并确认 Git 状态

```powershell
Set-Location 'E:\vibecode\llm学习项目\llm-kernel-lab'
git status --short
```

`git status --short` 有输出并不等于项目坏了：当前可能有尚未提交的教学文档和 CUDA Online
实验。不要执行 `git reset --hard` 或删除不认识的文件。先学会区分“已提交版本”和
“工作目录里的新增改动”。

### 步骤 2：检查 Python 与 GPU

```powershell
python --version
python -c "import torch,triton; print('torch',torch.__version__); print('triton',triton.__version__); print('cuda available',torch.cuda.is_available()); print('device',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi
```

预期：Python 3.11、PyTorch 与 Triton 可导入、`cuda available True`、识别 RTX 3080 Ti。
此前验证过的版本记录在[环境检查](environment.md)。驱动、Toolkit、PyTorch 自带 CUDA
Runtime 的版本号**不必逐字相同**；以真实 GPU 测试是否通过为准。不要因为版本显示不同就先重装。

### 步骤 3：设置本次 PowerShell 会话的 Triton 缓存目录

```powershell
$env:TRITON_CACHE_DIR = Join-Path $env:TEMP 'llm-kernel-lab-triton-cache'
```

这台 Windows 机器曾出现默认 `.triton/cache` 路径的写入权限问题；此设置只影响当前
PowerShell 会话。开新终端后如再次遇到缓存权限错误，重新执行即可。

### 步骤 4：先跑一个最小正确性测试

```powershell
python llm_kernels/vector_add/test.py
```

预期出现 FP32/FP16 多个 `通过 ...`，最后有“所有 Vector Add 正确性测试均已通过”。
第一次运行会 JIT 编译；`remark: ... instructions in function` 是编译器备注，不是报错。
以后使用 `test.py` 判断“是否算对”，使用 `benchmark.py` 判断“花了多少时间”，两者不能互相替代。

### 步骤 5：暂时只记住两种运行入口

```powershell
python llm_kernels/vector_add/test.py       # 验证正确性
python llm_kernels/vector_add/benchmark.py  # 测性能
```

后续每个模块也遵守相似结构。CUDA Extension 的首次调用可能触发较慢的 MSVC/NVCC 编译；
仓库把源码暂存到 ASCII 临时路径，是为了避开中文路径下的构建问题。正式 Benchmark
会在计时前完成首次编译。

## 2. 只学本项目用到的 Python 和 Tensor 基础

你不需要先补完 Python 全课程。先能看懂下面这些写法，再进入 GPU 代码。

### 步骤 1：变量、函数、类型标注、返回值

```python
def effective_bandwidth_gbps(n_elements: int, element_size: int, latency_ms: float) -> float:
    bytes_moved = 3 * n_elements * element_size
    return bytes_moved / (latency_ms * 1e-3) / 1e9
```

冒号后的 `int`、`float` 是帮助人和工具理解接口的类型标注。`return` 交回函数结果。
`1e-3` 把毫秒换成秒，`1e9` 把 B/s 换成 GB/s。这里乘 3 是两个输入各读一次、输出写一次。
实际代码在 [`vector_add/benchmark.py`](../llm_kernels/vector_add/benchmark.py)。

### 步骤 2：循环、字典、解包、lambda

```python
providers = {
    "pytorch": lambda: vector_add_torch(x, y),
    "triton": lambda: vector_add_triton(x, y),
}
for provider, function in providers.items():
    p50_ms, p20_ms, p80_ms = measure_ms(function)
```

字典把名字映射到待测函数。`lambda: ...` 是**暂不执行**的无参数函数；计时器需要时才调用。
`.items()` 一次给出“名字、函数”，`for provider, function` 把两个值分别接住。
`p50_ms, p20_ms, p80_ms = ...` 是把返回的三个值解包。若把这些 lambda 留到外层
循环结束后再统一执行，需要额外注意 Python 的闭包晚绑定；本项目是在每轮内立即计时。

### 步骤 3：Tensor 的 shape、dtype、device、contiguous

```python
x = torch.randn((2, 3), device="cuda", dtype=torch.float32)
print(x.shape)       # (2, 3)
print(x.numel())     # 6
print(x.element_size())  # FP32 时是 4 字节
```

`shape` 决定逻辑维度，`dtype` 决定单元素占用和数值精度，`device` 决定 CPU/GPU。
`stride` 决定各维度下标如何转为物理地址。`contiguous` 表示符合当前形状的标准连续
内存布局；本项目多数自定义 Kernel 暂只支持它，所以包装函数会先检查。

特别注意：`x.transpose(0,1)` 通常只是改变 Shape/Stride 的 View，不搬运全部数据；
`x.transpose(0,1).contiguous()` 才是生成连续转置副本。后面比较转置性能时必须使用
后者，才和自定义 Kernel 做了同一件事。

### 步骤 4：用参考结果验证，而不是肉眼看十几个数

```python
expected = vector_add_torch(x, y)
actual = vector_add_triton(x, y)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

Vector Add 两边执行相同的逐元素加法，本项目可要求严格相等。Reduction/Softmax 的
并行顺序、指数计算和 FP16 舍入可能不同，测试会使用明确的容差。`assert_close`
失败时，先检查形状、dtype、数值最大误差及特殊边界，不要直接放宽容差。

**本章检查点：**能说明 `lambda` 为什么不是立即执行、`numel()` 与 `shape` 的区别、
以及为什么 `transpose` 的 View 不能作为“搬运转置”的公平基线。

## 3. GPU 执行模型：先分清 Program、Thread、Block、Warp

CPU 上一个 for 循环可依次处理很多元素；GPU 要让许多工作组并行处理。理解本项目
只需先画出“谁负责什么”，不要一开始背全部硬件规格。

- CUDA **Thread**：执行 Kernel 的一个线程；`threadIdx.x` 是它在 Block 中的下标。
- CUDA **Block**：一组可用共享内存与 `__syncthreads()` 协作的线程；`blockIdx.x` 标识它。
- CUDA **Warp**：当前 NVIDIA GPU 中由 32 个线程组成的执行组；Warp 内可用 Shuffle 交换寄存器值。
- **Grid**：本次 Kernel 启动所需的所有 Block；可为一维或二维。
- Triton **Program**：一个 Triton Kernel 的实例，处理逻辑 Tile；它**不是**一个 CUDA Thread。
- **SM**：GPU 上执行 Block 的硬件单元，不等于 Block，也不等于“一个 Program”。

### 步骤 1：手算一维切块

先假设 `N=10, BLOCK_SIZE=4`。`triton.cdiv(10,4)=3`，因此 Grid 有 3 个 Program。

| Program ID | `id*4 + tl.arange(0,4)` | `offset < 10` |
|---:|---|---|
| 0 | 0、1、2、3 | 都合法 |
| 1 | 4、5、6、7 | 都合法 |
| 2 | 8、9、10、11 | 只有 8、9 合法 |

若 `N=2049, BLOCK_SIZE=512`，Grid 是 5。最后一个 Program 的逻辑位置为
2048～2559，只有 2048 合法，其余 **511** 个位置被 Mask。注意它和
`N=2050`（2048、2049 两个合法，其余 510 个 Mask）差 1 个元素。

Mask 控制实际的 `load/store`，不会把非法地址当成真实输入。不要把
`tl.arange(0,4)` 的四个值简单理解成“四个 CUDA 线程”。一个 Program 的
底层线程数由 Triton 编译布局和配置决定；需要时用 NCU 的 Block Size 验证。

### 步骤 2：运行一个真正的 10 元素例子

```powershell
python -c "import torch; from llm_kernels.vector_add.triton_impl import vector_add_triton; x=torch.arange(10,device='cuda',dtype=torch.float32); print(vector_add_triton(x,torch.ones_like(x),block_size=4).cpu().tolist())"
```

预期输出 `[1.0, 2.0, ..., 10.0]`。第一次以 `block_size=4` 运行会生成新的
Triton 特化版本；若出现 `remark: ... instructions in function`，仍是正常编译备注。

### 步骤 3：再对比 CUDA 的映射

[`vector_add/csrc/vector_add_cuda.cu`](../llm_kernels/vector_add/csrc/vector_add_cuda.cu) 的
单线程索引是：

```cpp
index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n_elements) output[index] = x[index] + y[index];
```

Triton 版本一次 Program 生成一组 `offsets`；CUDA 版本每个 Thread 算一个
`index`。二者都要根据总元素数防止末尾越界，但映射单位不同。

### 步骤 4：理解“合并访存”而非只盯单线程

关注**同一条访存指令中相邻线程访问的地址**。在 CUDA Softmax 中，线程 0
随循环访问 `0、256、512...`，看起来是离散的；但同一轮线程 0、1、2 ...
访问 `0、1、2...`，仍有利于合并访存。不要由单线程的步长直接推断整体是否合并。

**本章检查点：**能在纸上列出 Program/Block 对应的位置，指出末尾 Mask，
并能说出为什么 Program、CUDA Thread 和 SM 不是同一种东西。

## 4. Vector Add：完整地读第一个算子

数学定义：`z[i]=x[i]+y[i]`。每个输出只依赖同一位置的两个输入，所以各位置
无需彼此通信。这使它成为观察启动开销和显存带宽的干净基线。

### 步骤 1：按固定顺序读四个文件

1. [`torch_impl.py`](../llm_kernels/vector_add/torch_impl.py)：参考结果 `x+y`。
2. [`triton_impl.py`](../llm_kernels/vector_add/triton_impl.py)：先读下面的 Python 包装函数，再读上面的 JIT Kernel。
3. [`test.py`](../llm_kernels/vector_add/test.py)：`0、1、17、256、1000、65537、2^20` 为什么都要测？
4. [`benchmark.py`](../llm_kernels/vector_add/benchmark.py)：输入如何生成、三个 provider 如何计时？

包装函数的阅读顺序：输入限制 → 分配输出 → `n_elements` → 空输入提前返回 →
Grid → `_vector_add_kernel[grid](...)`。JIT Kernel 的阅读顺序：Program ID →
Offsets → Mask → 两次 `tl.load` → 加法 → `tl.store`。这样读比从第一行机械往下看更容易。

### 步骤 2：运行正确性测试，再运行 Benchmark

```powershell
python llm_kernels/vector_add/test.py
python llm_kernels/vector_add/benchmark.py
```

`test.py` 通过只说明已测的输入算对，不说明性能最优。`benchmark.py` 的表头是
`provider,N,p50_us,p20_us,p80_us,effective_bandwidth_GBps`。`p50_us` 是测量
中位延迟（微秒），P20/P80 用于观察波动；不是“平均值±误差”。

### 步骤 3：手算有效带宽

FP32 每元素 4 B。Vector Add 读 x、读 y、写 z，共 `12N` B。
若 `N=16,777,216`，逻辑数据量为 `201,326,592 B`；假设 P50 为 250.880 μs：

```text
effective GB/s = 201,326,592 / (250.880×10^-6) / 10^9
                 ≈ 802.48 GB/s
```

这叫**有效带宽**：按约定的逻辑搬运量除以时间，并不直接等于 NCU 测得的真实
DRAM 字节数或 DRAM Throughput。对大向量，PyTorch/Triton/CUDA 接近同一带宽
平台是合理的；不是“Triton 写得不好”，也不能宣称 CUDA 必然更快。

### 步骤 4：读 CUDA C++ Extension 的三层

1. [`cuda_impl.py`](../llm_kernels/vector_add/cuda_impl.py) 在 Python 中加载编译后的扩展；
2. [`csrc/bindings.cpp`](../llm_kernels/vector_add/csrc/bindings.cpp) 把 C++ 函数暴露给 Python；
3. [`csrc/vector_add_cuda.cu`](../llm_kernels/vector_add/csrc/vector_add_cuda.cu) 检查参数、选择当前 CUDA Stream、启动 Kernel。

`.cu` 中 `__global__` 是设备执行的 Kernel，普通 `torch::Tensor vector_add_cuda(...)`
是 Host 侧入口。`<<<blocks, threads, 0, stream>>>` 指定 Grid、Block、动态共享
内存字节数和 CUDA Stream。Kernel 启动是异步的，不能用裸 CPU 时钟包一行调用
就断言 GPU 执行时间。

### 步骤 5：读已有实验，不急着重新做 NCU

[Vector Add NCU 报告](../benchmarks/results/vector_add_rtx3080ti_ncu_basic.md) 记录一次
DRAM Throughput 91.67%、SM Throughput 7.53%。这支持大尺寸接近显存带宽瓶颈；
不同 Block Size 在大尺寸下也进入相近平台。先学会复述证据链，再考虑调参。

**本章动手检查点：**用自己的话说明为什么逻辑数据量是 `12N`，解释
`N=0` 为什么直接返回，以及为什么最大尺寸一次测量波动大时不能选出“最快版本”。

## 5. Row Sum：从独立元素到并行归约

数学任务：输入 `[M,N]`，输出 `[M]`，`y[row]=Σ_column x[row,column]`。
同一输出依赖整行输入，必须把许多局部结果合并，这叫 Reduction（归约）。

### 步骤 1：先手算

```text
输入 [[1,2,3], [4,5,6]]
输出 [6,15]
```

[`torch_impl.py`](../llm_kernels/reduction/torch_impl.py) 的参考实现就是按最后一维求和。
运行：

```powershell
python llm_kernels/reduction/test.py
```

首先看输出中的手算样例，再看 FP32/FP16、`N=0`、`N=1003`、超宽行样例。

### 步骤 2：读 Triton V0

打开 [`triton_impl.py`](../llm_kernels/reduction/triton_impl.py)：一个 Program 对应
一行，`tl.arange(0,BLOCK_SIZE)` 生成列位置，`tl.sum(...,axis=0)` 压成一个值。
`N=1003` 会补齐为 `BLOCK_SIZE=1024`。非法位置 `other=0.0`，因为 0 是加法
中性元。FP16 输入先转 FP32 累加，最后写回原始 dtype。

`N=0` 时包装函数返回全 0 输出；若行数为 0，返回空输出。V0 限制最大 Block
Size 为 65,536。不要为了让 V0 支持任意宽行无限增加 Block Size：工作集过大
可能带来资源压力，而且当前实现明确设置了上限。

### 步骤 3：手算 V1 的两阶段分块

对 `[M,N]=[2,2500]`、每块 1024：每行 3 块，Stage 1 的 Grid 是 `(2,3)`。

```text
第 0 行：块 0 处理 0..1023；块 1 处理 1024..2047；块 2 处理 2048..2499
第 1 行：同样拆 3 块
partials 形状 = [2,3]，dtype=FP32
Stage 2 Grid = (2,)，分别合并每行 3 个局部和
```

打开 [`triton_v1_impl.py`](../llm_kernels/reduction/triton_v1_impl.py)，从
`partials_per_row=triton.cdiv(...)` 和两个 Grid 看起，再分别读 `_stage1_kernel`
与 `_stage2_kernel`。Stage 1 末块补 0；Stage 2 读取不足 2 的幂的 partials
时也补 0。两个阶段分开启动，有额外中间缓冲和启动成本。

### 步骤 4：理解数值误差与性能取舍

浮点加法并不严格满足结合律：`(a+b)+c` 与 `a+(b+c)` 可能最后几位不同。
所以行求和测试使用容差，不要求与 PyTorch 位位相同。

阅读 [V0/V1 NCU 报告](../benchmarks/results/reduction_v0_v1_rtx3080ti_ncu_basic.md)：
V0 与 V1 Stage 1 的 DRAM Throughput 都约 92.8%；V1 第二阶段仅几微秒。
V1 的价值是支持超宽行，不意味着所有短行都会更快。

```powershell
python llm_kernels/reduction/benchmark.py
```

**本章检查点：**能画出 `[2,2500]` 的两个 Grid，解释 `partials` 的 shape、
为什么中间结果用 FP32，以及为什么 `N=0` 时输出应是 0。

## 6. Matrix Transpose：二维索引、Tile、共享内存

数学任务：输入 `x[M,N]`，输出 `y[N,M]`，满足 `y[column,row]=x[row,column]`。
二维连续矩阵在显存中的位置是：

```text
input_offset  = row*N + column
output_offset = column*M + row
```

### 步骤 1：先分清 View 和真正复制

在 PowerShell 中尝试：

```powershell
python -c "import torch; x=torch.arange(6,device='cuda').reshape(2,3); v=x.transpose(0,1); c=v.contiguous(); print('x',x.shape,x.stride()); print('view',v.shape,v.stride(),v.is_contiguous()); print('copy',c.shape,c.stride(),c.is_contiguous())"
```

`v` 与 `c` 数值相同，但 `v` 通常是非连续 View，`c` 是连续副本。
项目的 [`transpose_torch`](../llm_kernels/transpose/torch_impl.py) 返回
`x.transpose(0,1).contiguous()`；如果只拿 View 和 CUDA Kernel 比耗时，
实际上在比较两种不同工作量。

### 步骤 2：先手算二维 Grid 和边缘 Mask

Triton V0 用 32×32 Tile。对输入 `[M,N]=[33,65]`：

```text
ceil(33/32)=2 个行 Tile
ceil(65/32)=3 个列 Tile
Grid=(2,3)，共 6 个 Program
```

最后一个行 Tile 只有 1 行合法，最后一个列 Tile 只有 1 列合法。
边缘 Tile 仍然生成 32×32 个逻辑位置，非法位置由二维 Mask 排除。

打开 [`triton_impl.py`](../llm_kernels/transpose/triton_impl.py)，先找
`row_offsets`、`column_offsets`，再看 `[:,None]` 与 `[None,:]` 如何组合成
二维 `input_offsets`。`tl.trans(tile)` 改变 Tile 内的排列方向；输出地址使用
`column*n_rows+row`。`output_mask=tl.trans(input_mask)` 与转置后的数据方向一致。

### 步骤 3：对照 CUDA Naive 与 Tiled

打开 [`transpose_cuda.cu`](../llm_kernels/transpose/csrc/transpose_cuda.cu)。先只比较：

```text
Naive：直接读 input[row*N+column]，写 output[column*M+row]
Tiled：先把 32×32 数据放进 __shared__ tile，再按另一方向取出并写回
```

Naive 的相邻线程可以连续读输入，但写输出时可能跨行，无法形成同样高效的
合并写入。Tiled 用共享内存进行中转，使输入读取和输出写入都更连续。
`__syncthreads()` 必须位于“写完 Tile”与“读取别的线程写入的 Tile”之间。

Tiled 版声明 `tile[32][33]`，而专项对照版使用 `tile[32][32]`。多出的
一列不对应输入元素；它改变共享内存行跨度，以缓解这个转置访问模式下的
Bank Conflict。此结论依赖本实验的元素类型与访问模式，不能泛化为“共享
内存数组统一加一列”。

### 步骤 4：运行与读报告

```powershell
python llm_kernels/transpose/test.py
python llm_kernels/transpose/test_cuda.py
python llm_kernels/transpose/benchmark.py
python llm_kernels/transpose/benchmark_cuda_padding.py
```

第一次 CUDA 测试会编译 Extension。实验中 4096×4096 的无 Padding 版本
出现 16,252,928 次 Shared Load Bank Conflict，Padding 后为 0；这是一组
硬件计数器证据，不是“性能提升 1600 万倍”。整体时间还受显存读写等其他
阶段影响。详见 [Transpose CUDA 报告](../benchmarks/results/transpose_cuda_rtx3080ti_ncu_basic.md)。

**本章检查点：**能手写两个线性地址公式、解释为什么 `x.transpose()` View
不适合作为搬运性能基线，并说明 Tile 与 `__syncthreads()` 各起什么作用。

## 7. Softmax：从数学稳定性走到 Online 状态

对一行 `x`：

```text
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

直接算 `exp(10000)` 会溢出。Softmax 对整行加减同一个常数不变，因此采用：

```text
m = max(x)
l = Σⱼ exp(xⱼ-m)
yᵢ = exp(xᵢ-m)/l
```

减去最大值后，最大元素的指数是 1，其他有限元素的指数不大于 1。
本项目主要测试有限输入，不承诺所有 NaN/Inf 输入有特殊定义的输出。

### 步骤 1：先手算 `[1,2]`

```text
m=2
未归一化指数=[exp(-1), exp(0)]≈[0.3679,1]
l≈1.3679
概率≈[0.2689,0.7311]，和约为 1
```

然后读 [`softmax/torch_impl.py`](../llm_kernels/softmax/torch_impl.py) 的 PyTorch
参考实现，运行：

```powershell
python llm_kernels/softmax/test.py
```

测试包括 FP16/FP32、空矩阵、非 2 的幂行宽、超宽行、大正负值、极值分别位于
首尾块、递增/递减值及一个非默认 CUDA Stream。通过测试说明这些情形与参考
结果相符；不能证明所有可能输入和并发调度都被验证。

### 步骤 2：读 Triton V0，理解为什么 Mask 用 `-inf`

[`triton_impl.py`](../llm_kernels/softmax/triton_impl.py)：每行一个 Program，
加载一整行，依次 `tl.max` → `tl.exp` → `tl.sum` → 除法 → `tl.store`。
`N=1003` 会补到 1024。补齐位置若用 0，而真实输入全是负值，0 可能错误地
成为最大值；用 `-inf` 则不影响最大值，而且 `exp(-inf-m)=0`（m 有限）。
FP16 输入先转 FP32 进行最大值、指数和归约，最后写回 FP16。

V0 的 Python 包装函数限制每行最多 65,536 个元素。更宽的行不仅超过该限制，
即使可编译也可能使单个 Program 的中间值过多。

### 步骤 3：读 V0 Spill 实验

[V0 NCU 报告](../benchmarks/results/softmax_v0_rtx3080ti_ncu_basic.md) 对比两种
总元素数相同、行宽不同的输入。`N=32768` 的一轮采样有 Local Load 701.50 MB
与 Local Store 510.13 MB，实际 DRAM 流量远超“一读一写”的逻辑量。
这是寄存器压力导致中间值被 Spill 的直接证据之一。

特别记住：V0 宽行时 Occupancy 反而更高，运行却更慢。Occupancy 高不等于
没有额外数据搬运，也不代表高效率；要结合 Local Memory 与 DRAM 流量读。

### 步骤 4：读 Triton V1 的三个阶段

[`triton_v1_impl.py`](../llm_kernels/softmax/triton_v1_impl.py) 每块 1024 元素：

1. Stage 1：每块求局部最大值 `mₖ` 与局部指数和 `lₖ`；
2. Stage 2：每行合并所有块的状态，得到整行 `(m,l)`；
3. Stage 3：重新读取输入，写 `exp(xᵢ-m)/l`。

合并时不能直接把不同块的 `lₖ` 相加，因为它们各自减去了不同的最大值：

```text
m = maxₖ mₖ
l = Σₖ lₖ * exp(mₖ-m)
```

例如 `[1,2]` 与 `[3,4]` 两块，前一块最大值为 2，后一块为 4。
最终以 4 为统一基准，前一块的指数和必须乘 `exp(2-4)`。
V1 有三个 Kernel 启动、中间状态缓冲和两次输入读取，但块工作集较小，
超宽行不再出现 V0 那样的 Spill。[V1 报告](../benchmarks/results/softmax_v1_rtx3080ti_ncu_basic.md)
记录了三个阶段各自的 NCU 结果；不要只看其中一个阶段就评价整个算子。

### 步骤 5：读 Triton V2 的 Online 更新

[`triton_v2_impl.py`](../llm_kernels/softmax/triton_v2_impl.py) 是每行一个 Program、
逐块扫描。已处理部分的统计状态记为 `(m,l)`，加入新块后：

```text
m_new = max(m, block_max)
l_new = l*exp(m-m_new) + Σ本块 exp(xᵢ-m_new)
```

公式前半项把旧指数和换到新最大值基准；后半项加入新块。
Online 指**扫描时只保留跨块统计状态**，当前块仍有临时向量。
独立 Softmax 要等最终分母确定才能写出概率，所以 V2 仍第二遍读取输入。
它只启动一个 Kernel，但一行内块间有顺序依赖；不能断言它总比 V1 快。

### 步骤 6：读 CUDA V0 → Shuffle → Online 的演进

在 [`softmax_cuda.cu`](../llm_kernels/softmax/csrc/softmax_cuda.cu) 中先找到三种
Kernel，不要一开始逐行读整个 C++ 文件：

| CUDA 版本 | 每行怎样合作 | 输入读取次数 | 主要教学价值 |
|---|---|---:|---|
| V0 `softmax_cuda_kernel` | 256 线程，整个 Block 用共享内存树形归约最大值与和 | 3 | 显式同步、共享内存 |
| Shuffle `softmax_cuda_shuffle_kernel` | Warp 内 Shuffle，8 个 Warp 结果再合并 | 3 | 减少归约同步与缓冲 |
| Online `softmax_cuda_online_kernel` | 每线程维护 `(m,l)`，再做 Warp/Block 状态合并 | 2 | 减少输入扫描 |

V0 的一行通过 `blockIdx.x` 选择；线程 t 扫描 `t,t+256,t+512...`。
共享内存里的数据由其他线程读取前必须同步。Shuffle 可在同一 Warp 内交换
寄存器值，但**不能**替代跨 Warp 的共享内存交接。当前实现所有相关 lane
参加 Shuffle，才使用全 Warp 掩码；不能随意把它放进仅部分 lane 执行的分支。

Online 的归约对象从一个数变成 `SoftmaxState{maximum,sum}`：

```text
若 m_A >= m_B：
  合并后的 m = m_A
  合并后的 l = l_A + l_B*exp(m_B-m_A)
```

若一侧没有元素，直接返回另一侧；否则空状态 `(-inf,0)` 与 `(-inf,0)`
直接代入指数式会出现 `-inf-(-inf)`，产生 NaN。这也是边界测试特别重要的原因。

### 步骤 7：运行专项对照，解释为什么短行与宽行相反

```powershell
python llm_kernels/softmax/benchmark.py
python llm_kernels/softmax/benchmark_cuda_online.py
```

在三轮交替顺序的历史实验中，`1024×32768` 的 CUDA Shuffle/Online P50
中位数约 685.52/529.41 μs；短行 `4096×128` 则约 17.41/26.62 μs。
NCU 对前一形状测得 Online 读流量约 268.47 MB，接近读 FP32 输入两遍；
Shuffle 读约 402.68 MB，接近三遍。

这说明减少读取对宽行有利；短行节省的数据量小，在线更新的指数、分支和
状态合并开销反而占比较大。不能把 Online 直接替换为所有形状的默认算法。
详见 [Online 实验报告](../benchmarks/results/softmax_cuda_online_rtx3080ti.md)。

**本章检查点：**能手算 `[1,2]` 与 `[3,4]` 的状态合并；说出五个版本的
Kernel 数和输入读取次数；解释为什么 Online 不是 FlashAttention。

## 8. Benchmark：怎么看输出，不被数字带偏

### 步骤 1：先确认比较任务和条件一致

每次记录 GPU、输入 Shape、dtype、是否连续、provider、是否预热、计时方法。
如果一个实现只返回转置 View，另一个实际复制矩阵，时间没有可比性。
如果一边 FP16、一边 FP32，或者数学精度模式不同，也不要直接把速度差称作优化收益。

### 步骤 2：理解 P20、P50、P80

`triton.testing.do_bench` 在当前代码中返回三个延迟分位数（单位 ms），
打印时乘 1000 变成 μs。P50 是中位数，不是最小值；P20～P80 展示中间
一部分样本范围，也**不是**严格的统计置信区间。

示例：如果两个版本分别 494 μs 和 503 μs，差距约 1.8%，而多轮运行的
波动与这个数量级接近，就不能宣称前者在所有情况下稳定更快。
若最大尺寸出现异常长的 P80，先重复实验、交替 provider 顺序，再判断趋势。

### 步骤 3：区分逻辑字节数和实际 DRAM 流量

| 算子或实现 | 逻辑数据搬运口径 | 注意 |
|---|---|---|
| FP32 Vector Add | `2 读 + 1 写 = 12N B` | 输入只有两个向量 |
| FP32 Transpose | `1 读 + 1 写 = 8MN B` | 要比较连续输出副本 |
| FP32 Row Sum V0 | 输入约 `4MN B`，输出 `4M B` | V1 还有 partials |
| FP32 Softmax Benchmark CSV | 统一按 `1 读 + 1 写 = 8MN B` | 仅用于同口径有效带宽比较 |
| FP32 Softmax V1/V2/Online | 算法至少两遍输入读、一遍输出写 | 实际 DRAM 可受缓存/事务影响 |
| 当前 FP32 CUDA Softmax V0/Shuffle | 三遍输入读、一遍输出写 | 不等同于 CSV 的一读一写公式 |

`effective_bandwidth_GBps` 是**按约定逻辑量**算出来的，不保证等于显存芯片
实际搬运量。发生 Spill、缓存命中或多阶段缓冲时，两者差异可能明显。

### 步骤 4：区分“测函数”与“纯 Kernel 下界”

Benchmark 的 provider 是 Python 函数，通常会分配输出 Tensor 并发起 GPU
工作；代码先完成首次 JIT/Extension 编译再计时。它不是“模型端到端延迟”，
也不是完全剔除 Python 与分配影响、固定输出缓冲的纯 Kernel 极限。
小尺寸时这些固定成本特别容易影响解释。

### 步骤 5：你自己写一条实验结论

用下面模板，每次只填**实测**内容：

```text
在 [GPU] 上，对 [Shape、dtype、连续性] 的 [算子]，比较 [A] 与 [B]；
使用 [测试命令与计时口径]，P50 分别为 [数值]。
我原先假设 [原因]；NCU 的 [明确指标] 支持/反驳这一假设。
结论只适用于当前条件；[尚未验证的条件] 不能外推。
```

推荐先给 Softmax Shuffle vs Online 各写一条短行、宽行结论，再给 Transpose
无 Padding vs Padding 写一条。这比截图一整张 CSV 更能体现工程能力。

## 9. NCU：从问题出发，而不是把所有指标都采一遍

`ncu` 是 NVIDIA Nsight Compute 命令行工具，用来观察某个 GPU Kernel 的硬件
执行情况。它与 Benchmark 分工不同：Benchmark 先判断“差多少”，NCU 再帮助
判断“为什么”。官方 [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
说明部分指标可能需要多次 replay，Profiler 环境会影响计时，所以不要用 NCU
采样时间代替普通 Benchmark 的版本排名。

### 步骤 1：先跑正常程序，再选一个问题

例如：为什么 Softmax V0 的宽行明显慢？要看 Local Memory 和 DRAM 读写量。
为什么 Shuffle 短行快、宽行不快？要比较同步开销与输入读取量。
为什么 Transpose Padding 有效？要看 Shared Load Bank Conflict 等指标。

### 步骤 2：使用项目的 profile 入口

```powershell
ncu --set basic --kernel-name regex:vector_add_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/vector_add/profile.py
```

`profile.py` 默认预热 10 次，再执行一次待捕获 Kernel。`--kernel-name` 过滤
目标名称；`--launch-skip 10` 跳过预热；`--launch-count 1` 只捕获一次匹配的
正式启动。不同版本的 Kernel 名称应以源码/实际编译名称与报告为准。

例：Softmax CUDA Online 的单 Kernel 入口：

```powershell
ncu --set basic --kernel-name regex:softmax_cuda_online_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/softmax/profile.py `
    --version cuda_online --rows 1024 --columns 32768
```

如果 `ncu` 返回 `ERR_NVGPUCTRPERM`，表示当前进程不能访问 GPU 性能计数器，
不是 Kernel 算错。先在有权限的管理员 PowerShell 里运行，不要为了完成本章
急于修改系统级设置。普通测试与 Benchmark 不依赖 NCU 权限。

### 步骤 3：四组核心指标各回答什么？

| 指标 | 能回答的问题 | 不能单独证明的事 |
|---|---|---|
| Duration | 单个捕获 Kernel 的时间 | 多阶段算子总耗时 |
| DRAM Throughput / DRAM Read-Write | 显存利用率与实际数据量 | 算法必然最佳 |
| Compute (SM) Throughput | SM 某些计算资源利用情况 | 浮点操作一定是唯一瓶颈 |
| Registers、Shared Memory、Local Load/Store | 工作集资源与可能的 Spill | 只看寄存器数就判断性能 |
| Achieved Occupancy | 驻留活跃 Warp 的比例 | 等同于 GPU 利用率或更高必更快 |

“Local Memory”是 CUDA 线程私有的地址空间，不是快速的 Block Shared Memory。
发生寄存器 Spill 时，它可能引入缓存和 DRAM 数据量。Softmax 宽行 V0 是本项目
最重要的反例：Occupancy 高，但 Spill 和实际流量大，运行反而慢。

### 步骤 4：采集具体 DRAM/Local 指标

项目的 CUDA Online 报告使用以下四项指标：

```text
dram__bytes_read.sum
dram__bytes_write.sum
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
```

先读已有[Online 报告](../benchmarks/results/softmax_cuda_online_rtx3080ti.md)，
理解“约 268 MB 输入读取”如何对应 `1024×32768×4 B×2`。之后再自行运行
NCU；不需要为了学习每个算子都采一套 `--set full`。Reduction V1 与
Softmax V1 有多个阶段，应分别捕获、汇总分析，不能将 Stage 1 当成完整算子。

## 10. 常见错误：按现象检查，不盲目重装

| 现象 | 先做什么 | 不要先做什么 |
|---|---|---|
| `torch.cuda.is_available()` 为 False | 查 `nvidia-smi`、Python 环境与设备识别 | 直接重写 Kernel |
| Triton 默认缓存目录 PermissionError | 重新设置第 1 章的 `TRITON_CACHE_DIR` | 删除整个用户目录 |
| `remark: ... instructions in function` | 继续看测试是否通过 | 当成编译失败 |
| 首次 CUDA Extension 编译较慢 | 等待 MSVC/NVCC，观察最终是否报错 | 把编译时间算进 Kernel 延迟 |
| NVCC 在中文 Windows 环境解析 `.cu` 异常 | 使用项目现有 ASCII 源码注释与临时源码路径 | 随意改仓库父目录 |
| NCU `ERR_NVGPUCTRPERM` | 用有权限的管理员 PowerShell 试 NCU | 认为 GPU Kernel 错误 |
| `assert_close` 失败 | 查 shape、dtype、越界、Mask、最大误差、特殊值 | 无理由放宽容差 |
| Benchmark P80 异常长 | 多轮复测、交替 provider、记录环境 | 只挑最好一次写简历 |
| CUDA Out of Memory | 缩小 Benchmark Shape、关闭其他占显存程序 | 删除项目测试或当成数学错误 |
| Git 提示配置文件读取警告 | 看真正的命令退出码与 Git 状态 | 把无关 warning 当作推送成功/失败结论 |

读错误栈时优先找**第一条真正的 `error:` 或 Python 异常类型**，不要只看最后的
“ninja stopped”。修复一处后重跑正确性，再测性能。源码中的 CUDA 同步问题
尤其不能靠“当前随机测试碰巧通过”证明不存在。

## 11. 按这个顺序完成 8 次实操复习

不要求连续 8 天。每次先动手、再写 5～10 句自己的解释。若前一次检查点说不清，
先返回源码，不急着进入下一算子。

### 第 1 次：熟悉目录和工具

1. 完成第 1 章环境检查和缓存设置。
2. 用 `git status --short` 看哪些文件还没提交；不清理它们。
3. 打开四个算子目录，各指出 `torch_impl.py`、实现、测试和 Benchmark。
4. 运行 Vector Add 的 `test.py`。
5. 笔记写下：我当前的 GPU、Python、PyTorch、Triton 是什么；哪里是参考结果。

### 第 2 次：Vector Add

1. 手算 `[N=10,block_size=4]` 的 Grid/Mask。
2. 运行第 3 章的 10 元素命令。
3. 分“Python 包装函数”和“GPU JIT Kernel”读 `triton_impl.py`。
4. 运行 Benchmark，手算一个尺寸的 `12N/t`。
5. 笔记写下：大尺寸为什么可能接近带宽上限，小尺寸为什么看不出吞吐优势。

### 第 3 次：Row Sum

1. 手算 `[[1,2,3],[4,5,6]] → [6,15]`。
2. 画 `[2,2500]`、每块 1024 的 Stage 1/Stage 2 Grid。
3. 读 V0/V1 的 Python 包装与 Kernel，标出 `partials`。
4. 运行 test/benchmark。
5. 笔记写下：`other=0`、FP32 中间和、额外 Kernel 启动分别解决或付出什么。

### 第 4 次：Transpose

1. 运行第 6 章的 View/contiguous 小实验。
2. 手写 `row*N+column` 与 `column*M+row`，画 2×3 的转置。
3. 算 `[33,65]` 输入在 32×32 Tile 下的 Grid。
4. 对比 CUDA Naive/Tiled 的访存路径与同步位置。
5. 运行 test 与 Padding Benchmark，阅读 Bank Conflict 报告。

### 第 5 次：Softmax 数学与 Triton

1. 手算 `[1,2]` 的概率；解释为什么先减最大值。
2. 读 Triton V0 的 `-inf` Mask、`tl.max/tl.sum`。
3. 手算两块 `[1,2]`、`[3,4]` 的状态合并。
4. 画 V1 三阶段、V2 两遍扫描。
5. 运行 `softmax/test.py`，阅读 V0 Spill 报告。

### 第 6 次：Softmax CUDA

1. 在 CUDA `.cu` 中找到三种 Kernel 与 Host Launcher。
2. 画 256 线程分成 8 个 Warp；指出何处需要 `__syncthreads()`。
3. 解释为什么空状态不能直接代入普通 `(m,l)` 合并公式。
4. 运行 `benchmark_cuda_online.py`。
5. 笔记分别解释短行变慢、宽行变快，不只写“优化有效”。

### 第 7 次：性能证据

1. 任选 Vector Add、Transpose、Softmax 三份 NCU 报告。
2. 对每份记录：Shape/dtype、P50、一个最关键硬件指标、结论边界。
3. 判断指标是“逻辑口径”还是“实际硬件计数器”。
4. 有 NCU 权限再自己捕获一次；没权限可以先分析保存的报告。
5. 用第 8 章模板写出两条有条件的实验结论。

### 第 8 次：能否独立讲给别人听？

1. 不看文档，用 5 分钟解释四个算子为什么按当前顺序学习。
2. 选择 Transpose Padding 或 Softmax Spill，按“问题→假设→指标→修改→结果”讲清楚。
3. 选择 Shuffle 宽行没有明显加速的例子，解释为什么**负结果也有价值**。
4. 对照第 12 章检查题核查薄弱环节，再定下一阶段学习计划。

读完后的下一项代码工作是 RMSNorm；先从 PyTorch 参考公式与 FP32 归约开始，
再做 Triton/CUDA 对照。MatMul、Attention 在后面。当前仓库还不是完整推理
引擎，也没有编译器 IR/Pass 实现；编译器工作仍应在第二个独立项目进行。

## 12. 自测题与核对要点

先只看问题，在纸上写出答案，再展开核对。回答时尽可能指出对应源码和实验文件。

1. `N=2050, BLOCK_SIZE=512`，需要几个 Program？最后有多少合法/Mask 位置？
2. `N=1003` 的 Row Sum 为什么 Mask 补 0？Softmax 为什么补 `-inf`？
3. Reduction `[2,2500]`、块 1024 时 `partials` 的 shape/dtype 和两个 Grid 各是什么？
4. 为什么 `x.transpose(0,1)` 的计时不能直接与 CUDA Tiled Transpose 比？
5. FP32 Vector Add 每元素为什么是 12 B？Transpose 为什么是 8 B？
6. 为什么 FP16 Reduction/Softmax 常用 FP32 中间计算，但仍允许与 PyTorch 有误差？
7. Softmax V0 的高 Occupancy 为什么没阻止宽行变慢？哪两个 NCU 指标更能解释？
8. 合并 `[1,2]` 与 `[3,4]` 时，旧指数和为什么乘 `exp(2-4)`？
9. CUDA Shuffle 能否取消所有 Block Barrier？为什么？
10. Online Softmax 已在线维护 `(m,l)`，为什么输出阶段仍读第二遍？
11. CUDA Online 宽行更快，就该替换短行 Shuffle 吗？依据是什么？
12. `effective_bandwidth_GBps` 是否一定是硬件实际 DRAM 带宽？

<details>
<summary>完成后展开：核对要点</summary>

1. 5 个；最后 2048、2049 两个合法，其余 510 个 Mask。
2. 0 是加法中性元；`-inf` 不会压过真实负值的最大值，指数贡献为 0（有限 m）。
3. `partials=[2,3]`、FP32；Stage 1 Grid `(2,3)`，Stage 2 Grid `(2,)`。
4. 前者通常是只改 Shape/Stride 的 View；项目比较的是连续转置副本的真实搬运。
5. Vector Add 两读一写，各 4 B；Transpose 一读一写，各 4 B。
6. FP32 降低中间舍入误差；并行计算顺序、指数实现和最终 FP16 写回仍可能不同。
7. Spill 到 Local Memory；看 Local Load/Store 和实际 DRAM 读写量，不能只看 Occupancy。
8. 两块原来分别以 2、4 为基准，整行改用 4，需要把前一块的指数和重定标。
9. 不能；Shuffle 只在 Warp 内交换，Warp 间状态仍靠共享内存交接与 Block 同步。
10. 整行最终的归一化分母在第一遍结束前未知；独立算子未缓存全部输入值。
11. 不该；同轮短行实验 Online 约 26.62 μs、Shuffle 约 17.41 μs，需按形状判断。
12. 不一定；它是指定逻辑字节数/时间，真实 DRAM 流量要看 Profiler。

</details>

如果 12 题能答对 9 题以上，且能运行四模块测试、独立解释一个 NCU 报告，
可以开始新算子。若能答对但无法在代码中定位依据，仍要重读对应 Kernel。

## 13. 求职与后续发展的连接

项目展示要给出可验证、带条件的结果，不写模糊的“精通 GPU”或“全面领先”。
例如：

> 在 RTX 3080 Ti 上实现并分析 PyTorch/Triton/CUDA 的 Vector Add、Row Sum、
> Matrix Transpose 和 Softmax；建立边界测试、延迟分位数 Benchmark 与 NCU
> 证据链，定位 Softmax 宽行寄存器 Spill，验证 Tile Padding 对共享内存
> Bank Conflict 的影响。

如果自己能复现实验，可进一步说清楚：哪个 Shape/dtype、哪两个版本、P50
从多少到多少、NCU 哪个指标变化，以及没覆盖哪些形状。项目的优势是你能
解释**为什么变快或为什么没有变快**；不是实现数量本身。

对 GPU 算子岗位，当前成果展示访存、归约、同步、数值与 profiling 的基本功；
接下来 MatMul 与真实模型工作负载仍需补齐。对 GPU/AI 编译器岗位，这个项目
帮助你理解 Lowering 后代码的硬件后果，但并不能代替第二项目中的 IR、Pass、
合法性和代码生成经验。两项目分开，保留清晰的能力证据。

## 14. 官方资料如何配合本项目读

不要先啃完所有英文资料。每次从本项目遇到的一个问题出发，只看对应小节。

- [CUDA 12.6 Programming Guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)：
  遇到 Thread/Block、Shared Memory、同步或 Shuffle 疑问时查阅。
- [Triton Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：
  完成第 7 章后，与本项目的 V0、工作集限制及 Mask 对照。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：
  需要解释硬件指标、replay 与 profiler 开销时查阅。
- [本仓库的 GPU 性能检查表](gpu-performance-playbook.md)：
  写每次实验报告时按步骤核查，不被单一指标带偏。

**最后的掌握标准：**拿到一个新形状，你能先写数学与地址映射，估算数据量，
提出性能假设，再用测试和工具验证；当结果违背预期，你知道回到哪里查证。
