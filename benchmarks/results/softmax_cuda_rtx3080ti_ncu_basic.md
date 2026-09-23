# Softmax CUDA V0：共享内存归约对照

## 实验对象

RTX 3080 Ti，FP32，输入 `1024×32768`。每行一个 CUDA Block、每 Block 256
线程。一个 Block 使用 1,024 B 静态共享内存（256 个 float）作为归约缓冲。
源码在 `llm_kernels/softmax/csrc/softmax_cuda.cu`。

三次扫描分别做：最大值归约、稳定指数和归约、结果写出。跨线程的归约采用共享内存
二分树，每轮 `__syncthreads()`；一行很宽时，线程用步长 256 逐段读取。

## 结果

| 指标 | CUDA V0 | Triton V2 |
|---|---:|---:|
| 同轮 Benchmark P50 | 684.03 μs | 492.54 μs |
| NCU Duration | 695.33 μs | 493.44 μs（前一轮） |
| DRAM Throughput | 86.60% | 91.92%（前一轮） |
| Compute (SM) Throughput | 14.88% | 14.49%（前一轮） |
| Registers / Thread | 39 | 33（前一轮） |
| Achieved Occupancy | 95.46% | 86.78%（前一轮） |
| DRAM Read | 402.68 MB | 272.93 MB（前一轮） |
| DRAM Write | 131.72 MB | 134.00 MB（前一轮） |
| Local Load / Store | 0 B / 0 B | 0 B / 0 B（前一轮） |

CUDA 版的三次读取约为 `3 × 134.22 = 402.65 MB`，与 NCU 读流量接近；
另有一次输出写入。V2 是两次读取，所以即使两者都没有 Spill，CUDA 版仍多搬运
约一份输入。CUDA 的 Occupancy 更高，却更慢，不能仅凭 Occupancy 判断性能。
这里还存在共享内存归约、同步和 `expf` 计算成本，不能把全部延迟差都归因于
一次额外读取。

短行 `4096×128`：CUDA 版 P50 为 29.70 μs，Triton V0 为 9.38 μs。
固定 256 线程和两次共享内存归约不适合这种形状；这属于教学基线，不是最终调优版。

所有形状的本轮原始数据见 `softmax_cuda_rtx3080ti_fp32.csv`。CSV 的“有效带宽”
统一按一次读取和一次写入计算，是比较用的逻辑指标，**不是** CUDA 版实际
DRAM 带宽。CUDA 版当前仅支持连续二维 FP32 Tensor；Triton 版本还测试了 FP16。

## 复现

```powershell
python llm_kernels/softmax/test.py
python llm_kernels/softmax/benchmark.py
ncu --set basic --kernel-name regex:softmax_cuda_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/softmax/profile.py `
    --version cuda --rows 1024 --columns 32768
```

显存和 Local Memory 的独立 NCU 采样使用：

```text
dram__bytes_read.sum
dram__bytes_write.sum
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
```

NCU 的多次 replay 会改变运行环境。跨实现延迟比较以未开 profiler 的 Benchmark
为准，NCU 用于解释流量、资源占用和瓶颈。
