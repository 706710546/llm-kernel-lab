# CUDA Softmax V1：Warp Shuffle 归约实验

## 控制变量

保留 FP32、每行一个 Block、256 个线程、三次读取输入与一次输出写入。
只把共享内存二分归约替换为两层归约：先在每个 Warp 内用 Shuffle 归约，
再通过共享内存将 8 个 Warp 的结果交给第一个 Warp 归约。

基线同时修复了共享内存复用前的同步：所有线程读完 `row_max` 后才能将
`partial` 重新用于求和。旧基线缺少此保护，存在潜在竞态；历史测试通过不能
证明它在所有调度下安全。本报告的基线数据来自修复后版本。

源码层面，基线每次归约有 9 次 Block Barrier，加上复用前保护，共 19 次；
Shuffle 每次归约 3 次，共 6 次。最后一道 Barrier 保证所有线程读完结果才
复用共享缓冲。静态共享内存由 256 个 float（1,024 B）降为 8 个 float（32 B）。

## Benchmark

RTX 3080 Ti，FP32；从同轮完整 Benchmark 输出中保留两个 CUDA provider，
数据见 `softmax_cuda_shuffle_rtx3080ti_fp32.csv`。

| Shape | 基线 P50 | Shuffle P50 | 基线 / Shuffle |
|---:|---:|---:|---:|
| 4096×128 | 30.72 μs | 17.41 μs | 1.76× |
| 4096×512 | 35.78 μs | 27.65 μs | 1.29× |
| 4096×2048 | 100.35 μs | 99.33 μs | 1.01× |
| 1024×32768 | 684.03 μs | 683.01 μs | 1.00× |

短行收益明显；宽行差异在测量波动范围内，不能宣称全面加速。Shuffle 减少了
归约的共享内存访问与 Block 同步，但没有减少输入读取或指数运算。

## NCU

`4096×128` Shuffle：Duration 20.70 μs，Registers/Thread 39，Static Shared
Memory 32 B，Achieved Occupancy 84.28%，DRAM Throughput 11.56%，SM Throughput
58.05%。短行不应根据低 DRAM 利用率简单判断为“访存很差”；这轮没有采集
详细 Stall 指标，不能指定某一种 Stall 为唯一瓶颈。端到端比较以 Benchmark 为准。

`1024×32768` Shuffle：DRAM Read 402.68 MB、DRAM Write 131.69 MB、Local
Load/Store 都为 0 B。读取约等于 FP32 输入的三倍，和旧 CUDA 基线相同。
因此下一轮优化宽行应考察减少读取次数，而不是继续只减少归约 Barrier。

## 复现

```powershell
python llm_kernels/softmax/test.py
python llm_kernels/softmax/benchmark.py
ncu --set basic --kernel-name regex:softmax_cuda_shuffle_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/softmax/profile.py `
    --version cuda_shuffle --rows 4096 --columns 128
```

宽行流量另用 `dram__bytes_read.sum`、`dram__bytes_write.sum`、
`l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum` 和
`l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum` 采集。
