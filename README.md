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
NCU 分析。

| 算子 | PyTorch | Triton | CUDA | 研究重点 |
|---|---:|---:|---:|---|
| Vector Add | ✓ | ✓ | ✓ | 显存带宽、CUDA 执行模型 |
| Reduction（按行求和） | ✓ | V0 / V1 ✓ | 计划中 | 并行归约 |
| Matrix Transpose | ✓ | V0 ✓ | Naive / Tiled ✓ | 二维 Tile、合并访存、共享内存 |

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
    └── transpose/
        ├── torch_impl.py
        ├── triton_impl.py
        ├── test.py
        ├── benchmark.py
        ├── profile.py
        └── README.md
```

## 项目的核心问题

对于每一个 Kernel，本项目都会回答同一个问题：

> 我们能否从算法、Tensor 形状、显存流量、GPU 执行模型和具体实现方式出发，解释
> 实际测得的性能？
