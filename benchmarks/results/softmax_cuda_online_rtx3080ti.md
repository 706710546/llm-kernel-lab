# CUDA Online Softmax V2：两遍读取与状态归约

实验日期：2026-09-28；RTX 3080 Ti、FP32、连续二维输入。

## 1. 实现与假设

每行一个 Block、256 线程。线程扫描 `t,t+256,t+512,...`，用在线递推同时
维护最大值和指数和。随后 Warp 内、Warp 间归约的是 `(m,l)` 状态对，
不是一个简单的最大值或求和。整行状态得到后，第二遍输入读取写出输出。

假设：宽行少读取一份输入能降低时间；短行则可能被状态合并中的额外指数
计算拖慢。这个算法改变了流量和统计计算，不是只改变一个线程配置参数。

当前共享内存为 64 B、源码中的 Block Barrier 为 2 次。它不缓存整行，
也不分配跨 Kernel 统计量缓冲。输入仅承诺本项目验证过的有限 FP32 值；
尚未为所有 NaN/Inf 情形定义特殊行为。

## 2. 正确性检查

- 现有空矩阵、短行、非整块行宽与最高 131,072 列测试；
- 大正负输入，最大值分别在首块和尾块；
- 1、17、33、257、1,025、4,097 列的递增、递减与相同值输入；
- 空线程状态与真实状态的合并；
- 一个非默认 CUDA Stream 上创建输入并调用算子的对照样例。

所有断言通过：与 PyTorch 容差一致、必要的有限性/非负性与行和检查。
这不代表全面的并发或竞争工具验证，也不代表任意 Stride/多 GPU 已覆盖。

## 3. Benchmark

完整首轮数据见 `softmax_cuda_online_rtx3080ti_fp32.csv`。后续专项三轮对照
复用相同输入并交替 provider 顺序，见 `softmax_cuda_online_repeated_rtx3080ti_fp32.csv`。
每轮先预热，首次 CUDA Extension 编译不进入正式计时。

下表是三轮 P50 的中位数，不是把所有计时样本合并重新计算的全局 P50：

| Shape | Shuffle 三轮 P50 中位数 | Online 三轮 P50 中位数 | 解释 |
|---:|---:|---:|---|
| 4096×128 | 17.408 μs | 26.624 μs | Online 慢约 53% |
| 1024×32768 | 685.520 μs | 529.408 μs | Online 耗时降低约 23%，约 1.29× |
| 512×131072 | 1478.656 μs | 1170.432 μs | Online 耗时降低约 21%，约 1.26× |

一些宽行 P80 有明显长尾，不把全部样本称为稳定。三轮 P50 仍支持此机器上
“短行较慢、宽行受益”的方向性结论，但不足以确定普适的自动分派阈值。
完整首轮的 Triton V2 在 1024×32768 上为 492.544 μs，仍快于当前 CUDA
Online；不能把减少一次读取解释成 CUDA 已全面胜过其他实现。

## 4. NCU：1024×32768 Online

| 指标 | 本轮 Online |
|---|---:|
| Duration | 539.20 μs |
| DRAM Throughput | 83.89% |
| Compute (SM) Throughput | 23.48% |
| Registers / Thread | 30 |
| Static Shared Memory / Block | 64 B |
| Achieved Occupancy | 95.11% |
| DRAM Read | 268.47 MB |
| DRAM Write | 133.86 MB |
| Local Load / Store | 0 B / 0 B |

输入大小为 134.22 MB，两次读取应约 268.44 MB。实测读流量与这个数字接近。
加上输出，实测读写约 402.33 MB，接近“两读一写”的逻辑量 402.65 MB。
历史 Shuffle 的相同形状读流量为 402.68 MB；差别接近少读一份输入。
缓存与内存事务会影响实际计数，所以写流量不必逐字节等于 Tensor 大小。

DRAM Throughput 的百分比并不要求比三遍读取版更高，才能证明优化：最终
任务量减少了，按逻辑任务衡量的时间可以更短。需要同时看耗时与搬运的字节数。
状态合并涉及指数运算和循环依赖，当前没有详细 Stall 或指令计数证据，不能
把剩余差距指定为某一种指令的唯一瓶颈。

## 5. 复现

```powershell
python llm_kernels/softmax/test.py
python llm_kernels/softmax/benchmark.py
python llm_kernels/softmax/benchmark_cuda_online.py
ncu --set basic --kernel-name regex:softmax_cuda_online_kernel --launch-skip 10 `
    --launch-count 1 python llm_kernels/softmax/profile.py `
    --version cuda_online --rows 1024 --columns 32768
```

独立流量采样使用 `dram__bytes_read.sum`、`dram__bytes_write.sum`、
`l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum` 和
`l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum`。基准中的有效带宽仍统一用
“一读一写”逻辑口径，不代表实际 DRAM 带宽；版本时间对照以独立 Benchmark
为准，不以 NCU replay 时间替代。

## 6. 本阶段结论

本次完成的是 FP32 独立 Online Softmax 的教学实现，而不是 FlashAttention。
最大的学习收获是：并行归约不只适用于加法，也可以归约能合并的统计状态；
优化需要交换存储、计算、依赖和同步成本。保留三版 CUDA 对照，不把 Online
作为所有形状的默认替代。接下来可以进入 RMSNorm，复用 FP32 累加、按行映射
和融合减少中间流量的分析方法。
