# LLM Kernel Lab

> 使用 PyTorch、Triton 和 CUDA 实现、分析并优化大语言模型核心算子。
>
> A hands-on laboratory for implementing, profiling, and optimizing LLM
> operators with PyTorch, Triton, and CUDA.

项目采用统一的研究流程：

```text
PyTorch 参考实现 → Triton Kernel → 正确性测试 → 性能基准 → 瓶颈分析
```

本项目优先关注学习、正确性、性能分析和硬件原理，不以成为生产级算子库为目标。

## 当前里程碑

Vector Add 已完成 PyTorch、Triton、CUDA、Benchmark 和 Profiler 闭环。Reduction 已完成
Triton V0/V1 的正确性、Benchmark 与 NCU 分析闭环。Matrix Transpose V0 已完成 PyTorch
参考实现、Triton 32×32 Tile、CUDA Naive/Tiled、边界测试、Benchmark、Tile Size 实验和
NCU 分析。Softmax 已完成稳定公式、Triton V0 单 Program、V1 并行分块、V2 在线扫描
及 CUDA 共享内存归约基线，并完成正确性、Benchmark 和 NCU Spill 对照。

| 算子 | PyTorch | Triton | CUDA | 研究重点 |
|---|---:|---:|---:|---|
| Vector Add | ✓ | ✓ | ✓ | 显存带宽、CUDA 执行模型 |
| Reduction（按行求和） | ✓ | V0 / V1 ✓ | 计划中 | 并行归约 |
| Matrix Transpose | ✓ | V0 ✓ | Naive / Tiled ✓ | 二维 Tile、合并访存、共享内存 |
| Softmax | ✓ | V0 / V1 / V2 ✓ | V0 ✓ | 数值稳定性、分块归约、在线统计、Spill |

## 当前关键结果

RTX 3080 Ti 上的大尺寸 FP32 Vector Add：

- 独立 Benchmark 有效带宽约为 809～826 GB/s；
- Nsight Compute 测得 DRAM Throughput 为 91.67%；
- Compute (SM) Throughput 为 7.53%；
- PyTorch 与 Triton 在大尺寸下性能基本相同。

这些数据共同证明当前 Vector Add 是显存带宽受限，而不是计算能力受限。完整分析见
[Nsight Compute 实验报告](benchmarks/results/vector_add_rtx3080ti_ncu_basic.md)。

![Vector Add Block Size 性能比较](assets/benchmark/vector_add_block_size_comparison.svg)

三种 Block Size 在大尺寸下都进入约 810～826 GB/s 的带宽平台。它们之间的差异很小，
不足以支持某个 Block Size 显著更快的结论。

RTX 3080 Ti 上的 FP32 Row Sum V0 / V1：

- V0 与 V1 Stage 1 的 DRAM Throughput 都约为 92.8%；
- V1 Stage 1 的高 Occupancy 没有缩短时间，说明两者都受显存带宽限制；
- V1 的第二阶段约为 3.58 μs，是支持超过 65,536 列所付出的额外成本。

完整分析见 [Row Sum V0 / V1 Nsight Compute 实验报告](benchmarks/results/reduction_v0_v1_rtx3080ti_ncu_basic.md)。

RTX 3080 Ti 上的 FP32 Matrix Transpose V0：

- `4096×4096` 输入的 Triton 有效带宽约为 775.57 GB/s；
- Nsight Compute 测得 DRAM Throughput 为 89.22%；
- `32×32` Tile 被编译为 128 Threads，并使用约 4.10 KB 动态共享内存；
- Achieved Occupancy 为 93.44%。
- 手写 CUDA Naive 因非合并写入，在 NCU 中只有 25.29% DRAM Throughput；
- 手写 CUDA Tiled 使用 4.22 KB 静态共享内存，将 DRAM Throughput 提高到 87.11%，
  性能与 Triton 基本一致。
- `tile[32][32]` 产生 16,252,928 次共享内存读取 Bank Conflict；改为 `[32][33]` 后冲突
  降为 0。

完整分析见 [Transpose V0 Nsight Compute 实验报告](benchmarks/results/transpose_v0_rtx3080ti_ncu_basic.md)。
CUDA 对照见 [Transpose CUDA NCU 实验报告](benchmarks/results/transpose_cuda_rtx3080ti_ncu_basic.md)。

RTX 3080 Ti 上的 FP32 Softmax V0：

- `N≤8192` 时 Triton 与 PyTorch 性能接近，`4096×8192` 达到约 816.65 GB/s；
- `N=32768` 时 Triton 下降到约 143.56 GB/s；
- NCU 测得约 701.50 MB Local Load 和 510.13 MB Local Store；
- Spill 使实际 DRAM 流量扩大到最低逻辑流量的约 5.44 倍；
- 超宽行版本 Occupancy 更高但性能更差，证明 Occupancy 不能脱离 Spill 和数据流量判断。

完整分析见 [Softmax V0 Nsight Compute 实验报告](benchmarks/results/softmax_v0_rtx3080ti_ncu_basic.md)。

Softmax V1 将每行拆成 1,024 元素的块，在 FP32 中合并局部最大值和指数和。
`1024×32768` 上的 P50 从 V0 的 1881.60 μs 降至 V1 的 501.76 μs；NCU 测得 V1
三个阶段的 Local Load/Store 都为 0。详见
[Softmax V1 对照报告](benchmarks/results/softmax_v1_rtx3080ti_ncu_basic.md)。

Softmax V2 使用在线 `(m, l)` 更新，在 `1024×32768` 上 P50 为 494.59 μs，
与 V1 的 502.78 μs 接近；NCU 测得无 Local Memory Spill。V2 只启动一个
Kernel，但每行的块在同一 Program 内串行扫描，不能视为对 V1 的全面替代。
详见 [Softmax V2 实验报告](benchmarks/results/softmax_v2_rtx3080ti_ncu_basic.md)。

Softmax CUDA V0 用 256 线程和共享内存完成两次归约。在 `1024×32768` 上，
本轮 P50 为 684.03 μs，NCU 测得三次输入读取、无 Local Memory Spill。
它比 Triton V2 多一次全量读取，是用于理解线程协作和性能代价的教学基线。
详见 [Softmax CUDA 对照报告](benchmarks/results/softmax_cuda_rtx3080ti_ncu_basic.md)。

## 已验证的开发环境

- Windows 11、Python 3.11.9
- NVIDIA GeForce RTX 3080 Ti（12 GB，计算能力 8.6）
- NVIDIA 驱动 591.86
- CUDA Toolkit 12.6
- PyTorch 2.14.0+cu126
- triton-windows 3.8.0.post28
- Visual Studio Build Tools 2022 / MSVC 19.44

编译器和 Git 发布环境的详细情况见[开发环境检查](docs/environment.md)。

## 运行第一个实验

```powershell
python llm_kernels/vector_add/test.py
python llm_kernels/vector_add/benchmark.py
python llm_kernels/reduction/test.py
python llm_kernels/reduction/benchmark.py
python llm_kernels/transpose/test.py
python llm_kernels/transpose/benchmark.py
python llm_kernels/softmax/test.py
python llm_kernels/softmax/benchmark.py
```

完成正确性和 Benchmark 后，可以用 Nsight Compute 捕获一次预热后的 Kernel：

```powershell
ncu --set basic --kernel-name regex:vector_add_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/vector_add/profile.py
```

Benchmark 会输出延迟和有效带宽。对于包含 `N` 个 FP32 元素的向量，理论上的最低
全局显存流量为：

```text
读取 x + 读取 y + 写入 output = 12N 字节
```

## 当前目录结构

```text
llm-kernel-lab/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── scripts/
│   └── plot_vector_add.py
├── benchmarks/
│   └── results/
├── docs/
│   ├── environment.md
│   └── gpu-performance-playbook.md
└── llm_kernels/
    ├── vector_add/
    │   ├── csrc/                # CUDA C++ Extension
    │   ├── torch_impl.py
    │   ├── triton_impl.py
    │   ├── cuda_impl.py
    │   ├── test.py
    │   ├── benchmark.py
    │   ├── profile.py
    │   └── README.md
    ├── reduction/
    │   ├── torch_impl.py
    │   ├── triton_impl.py       # V0：一行一个 Program
    │   ├── triton_v1_impl.py    # V1：两阶段分块归约
    │   ├── test.py
    │   ├── benchmark.py
    │   ├── profile.py
    │   └── README.md
    ├── transpose/
    │   ├── torch_impl.py
    │   ├── triton_impl.py
    │   ├── test.py
    │   ├── benchmark.py
    │   ├── profile.py
    │   └── README.md
    └── softmax/
        ├── torch_impl.py
        ├── triton_impl.py
        ├── triton_v1_impl.py
        ├── triton_v2_impl.py
        ├── cuda_impl.py
        ├── csrc/                # CUDA C++ Extension
        ├── test.py
        ├── benchmark.py
        ├── profile.py
        └── README.md
```

## 项目的核心问题

对于每一个 Kernel，本项目都会回答同一个问题：

> 我们能否从算法、Tensor 形状、显存流量、GPU 执行模型和具体实现方式出发，解释
> 实际测得的性能？
