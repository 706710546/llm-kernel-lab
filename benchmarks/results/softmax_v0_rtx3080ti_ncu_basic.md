# Softmax V0：正常行宽与超宽行 NCU 对照

## 实验设计

选择两个元素总数完全相同的 FP32 输入：

```text
4,096 × 8,192  = 33,554,432 个元素
1,024 × 32,768 = 33,554,432 个元素
```

两者的最低逻辑数据量都约为：

```text
读取输入 + 写出输出
= 33,554,432 × 4 Byte × 2
= 268,435,456 Byte
```

唯一主要变量是每个 Triton Program 需要处理的行宽。

## Basic 指标

| 指标 | N=8,192 | N=32,768 |
|---|---:|---:|
| Program 数量 | 4,096 | 1,024 |
| Duration | 333.60 μs | 1.87 ms |
| DRAM Throughput | 91.10% | 87.24% |
| Registers / Thread | 102 | 40 |
| Achieved Occupancy | 32.21% | 81.83% |
| Waves / SM | 12.80 | 1.07 |

如果只看 Occupancy，32,768 列似乎更好；但它实际慢约 5.6 倍。这说明 Occupancy 不是
性能目标，必须继续检查数据流量和 Spill。

## 实际 DRAM 与 Local Memory 流量

| 指标 | N=8,192 | N=32,768 |
|---|---:|---:|
| DRAM Read | 134.26 MB | 816.35 MB |
| DRAM Write | 133.53 MB | 644.08 MB |
| DRAM Total | 267.79 MB | 1,460.43 MB |
| Local Memory Load | 0 B | 701.50 MB |
| Local Memory Store | 0 B | 510.13 MB |
| 专项采集 Duration | 326.43 μs | 1.91 ms |

8,192 列时，实际 DRAM 流量约等于理论的输入读取加输出写入，没有 Local Memory 访问。

32,768 列时，Local Memory 读写合计约 1.21 GB，实际 DRAM 流量扩大到约 1.46 GB，是
最低逻辑流量的约 5.44 倍。Kernel 的 DRAM Throughput 仍然很高，但大部分带宽在搬运
编译器 Spill 出去的中间值，而不是有效输入输出。

## 为什么会 Spill？

V0 的一个 Program 处理完整一行，并在逻辑上维护：

```text
输入 values
最大值 row_max
指数 numerators
分母 denominator
输出 probabilities
```

随着行宽增大，编译器无法把所有活跃中间值保存在寄存器中。它选择降低每线程寄存器数，
把部分值写入 Local Memory，之后再读取回来。

CUDA 中的 Local Memory 是“线程私有地址空间”，但不是每个线程旁边的一块高速片上
存储。它通常由设备显存承载，并经过 L1/L2 Cache。因此大规模 Spill 会制造额外 DRAM
流量。

## 为什么高 Occupancy 没有帮助？

32,768 列版本的寄存器数从 102 降到 40，因此可以同时驻留更多 Warp，Achieved
Occupancy 从 32.21% 上升到 81.83%。但每个 Program 要进行大量 Local Memory 读写，
增加的 Warp 无法抵消 5 倍以上的流量放大。

这组实验说明：

> 高 Occupancy 可能只是编译器减少寄存器并发生 Spill 后的副作用，不代表 Kernel 更快。

## V1 方向

V0 适合中等行宽。超宽行需要避免让一个 Program 同时保存完整行的所有中间值。后续可
研究：

1. 分块计算局部最大值；
2. 合并得到全局最大值；
3. 分块计算指数和；
4. 使用 Online Softmax 逐块更新最大值与归一化因子；
5. 在 Attention 中进一步演化为 FlashAttention 的在线归一化思想。
