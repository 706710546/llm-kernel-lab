# 我如何分析一个 GPU Kernel

这是一份会随着实验不断更新的检查表。只有真正从实验中得到结论后，才把结论写进来。

1. 理解算法和数学定义。
2. 明确输入、输出的 Tensor 形状。
3. 估算理论 FLOPs。
4. 估算需要搬运的字节数。
5. 计算算术强度（Arithmetic Intensity）。
6. 提出性能瓶颈假设。
7. 实现可信的参考版本。
8. 测试正确性，包括边界形状。
9. 预热后测量延迟中位数和分位数。
10. 使用 Profiler 检查怀疑的瓶颈。
11. 每次只优化一个因素。
12. 再次 Benchmark，并用数据解释结果。

## 实验记录

### Vector Add

- 假设：大尺寸 FP32 Vector Add 受到显存带宽限制。
- 证据：在 RTX 3080 Ti 上，当 `N = 2^26` 时，PyTorch 和 Triton 都稳定在约
  816–818 GB/s；小尺寸输入的耗时约为 4–6 微秒，主要受固定启动和计时开销影响。
- 观察：第一个 Triton 版本在大尺寸下没有明显领先 PyTorch。这支持“带宽受限”的
  假设，但不能支持“Triton 天生比 PyTorch 快”这样的笼统结论。
- Profiler 证据：Nsight Compute 测得 DRAM Throughput 91.67%、Compute Throughput
  7.53%、Achieved Occupancy 86.10%。这直接验证了显存带宽瓶颈。
- 方法结论：Occupancy 不是越高越好。分析优化方向时，应先看最接近饱和的硬件资源；
  当前显存吞吐比计算吞吐和 Occupancy 更能解释性能上限。
- 下一步问题：理解合并访存为什么能让 DRAM Throughput 达到 90% 以上。
