# Softmax V2：在线扫描实验

## 方法

RTX 3080 Ti，FP32，形状 `1024×32768`。每行一个 Program，每次处理 1,024 列。
第一遍扫描维护在线状态 `(m, l)`；第二遍重新读取输入并写出概率。Benchmark 的
P50/P20/P80 原始数据见 `softmax_v0_v1_v2_rtx3080ti_fp32.csv`。

```powershell
python llm_kernels/softmax/test.py
python llm_kernels/softmax/benchmark.py
ncu --set basic --kernel-name regex:softmax_v2_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/softmax/profile.py `
    --version v2 --rows 1024 --columns 32768
```

Local Memory 与 DRAM 字节数另用同一输入、同一 Kernel 的四个计数器采集：

```text
dram__bytes_read.sum
dram__bytes_write.sum
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
```

## 观测

| 指标 | V2 |
|---|---:|
| Benchmark P50 | 494.59 μs |
| NCU Duration | 493.44 μs |
| DRAM Throughput | 91.92% |
| Compute (SM) Throughput | 14.49% |
| Registers / Thread | 33 |
| Achieved Occupancy | 86.78% |
| DRAM Read | 272.93 MB |
| DRAM Write | 134.00 MB |
| Local Load / Store | 0 B / 0 B |

`1024×32768` 的 FP32 输入是 134.22 MB。两遍读入加一遍输出的逻辑下界为
402.65 MB；NCU 实际 DRAM 读写约 406.93 MB，与预期接近。V2 没有观测到
Local Memory Spill。对照 V0，同形状曾测得约 1,460 MB DRAM 流量及约 1.21 GB
Local Load/Store，P50 为 1,881.60 μs（历史实验）；本轮 V0 P50 为 1,866.75 μs。

本轮 V1 P50 为 502.78 μs，V2 为 494.59 μs；差距约 1.6%，不足以宣称 V2
在此形状上稳定优于 V1。两者在超宽行都避免了 V0 的 Spill，但 V1 把块分配给
不同 Program 并行处理，V2 则在每行内部串行更新 `(m, l)`。V2 省去两次额外
Kernel 启动和中间统计量缓冲，V1 提供更多块级并行度。当前测量说明这些成本
在此 GPU 和形状上大致抵消。

## 指标解释

CSV 中的 `effective_bandwidth_GBps` 按“一次读、一次写”的逻辑数据量计算，
用于统一比较吞吐，不代表 V1/V2 的实际显存带宽。V1/V2 都至少读输入两遍；
V2 的实际 DRAM Throughput 由 NCU 直接测得。Profiler 的多次 replay 会扰动
运行环境，因此端到端版本比较以非 Profiler 的 Benchmark 为准。
