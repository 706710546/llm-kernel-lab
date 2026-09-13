# Vector Add：Nsight Compute 基础分析

## 实验信息

| 项目 | 数值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 3080 Ti |
| Compute Capability | 8.6 |
| 数据类型 | FP32 |
| 元素数量 | 16,777,216（`2^24`） |
| Triton Block Size | 256 个元素/Program |
| Triton Program 数量 | 65,536 |
| CUDA Block Size | 128 个线程/Block |
| SM 数量 | 80 |

分析命令：

```powershell
ncu --set basic --kernel-name regex:vector_add_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/vector_add/profile.py
```

`profile.py` 先执行 10 次预热。Nsight Compute 跳过这些调用，只分析第 11 次
`_vector_add_kernel`。

## 关键结果

| 指标 | 测量值 |
|---|---:|
| Duration | 247.10 μs |
| Memory Throughput | 91.67% |
| DRAM Throughput | 91.67% |
| Compute (SM) Throughput | 7.53% |
| L1/TEX Cache Throughput | 19.40% |
| L2 Cache Throughput | 39.40% |
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 86.10% |
| Registers Per Thread | 16 |
| Static/Dynamic Shared Memory | 0 B |
| Waves Per SM | 68.27 |

## 与 Benchmark 交叉验证

FP32 Vector Add 对每个元素执行两次读取和一次写入，理论逻辑数据量是：

```text
16,777,216 × 12 字节 = 201,326,592 字节
```

根据 Nsight Compute 测得的 `247.10 μs` 计算：

```text
有效带宽 = 201,326,592 Byte ÷ 247.10 μs
         ≈ 814.8 GB/s
```

独立 Benchmark 在相同尺寸上测得约 809～815 GB/s。两种测量方式结果一致，说明
Benchmark 的计时和数据量模型可信。

## 性能结论

这个 Kernel 的 DRAM Throughput 已达到 91.67%，而 Compute Throughput 只有 7.53%。
因此它的主要瓶颈是显存带宽，不是浮点计算能力。

虽然 Achieved Occupancy 为 86.10%，低于理论值 100%，但当前不应该把“提高
Occupancy”当作首要优化目标：显存系统已经接近饱和，即使增加活跃 Warp，也不一定
能够获得更多可用显存带宽。

本实验支持以下结论：

> 对大尺寸连续 FP32 Vector Add，当前 Triton 实现已经是一个高带宽利用率的
> Memory-Bound Kernel。进一步优化应先检查内存访问和测量稳定性，而不是盲目追求
> 更高的 SM Compute Throughput 或 Occupancy。

## Triton Block Size 与 CUDA Block Size

源码中的 `block_size=256` 表示每个 Triton Program 处理 256 个元素；Nsight Compute
显示的 `Block Size=128` 表示编译后的 CUDA Thread Block 包含 128 个线程。二者不同，
说明 Triton 的一个向量位置不能简单等同于一个 CUDA 线程。底层线程映射由 Triton
编译器决定。

