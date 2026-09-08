# How I Analyze a GPU Kernel

This document is a living checklist. Update it only when an experiment teaches
something concrete.

1. Understand the algorithm.
2. Determine input and output shapes.
3. Estimate FLOPs.
4. Estimate memory traffic.
5. Calculate arithmetic intensity.
6. Form a bottleneck hypothesis.
7. Implement a reference.
8. Test correctness, including edge shapes.
9. Benchmark with warmup and latency percentiles.
10. Profile the suspected bottleneck.
11. Optimize one factor at a time.
12. Benchmark again and explain the result.

## Experiment log

### Vector Add

- Hypothesis: large FP32 vectors are memory-bandwidth bound.
- Evidence: on the RTX 3080 Ti, both providers plateau near 816–818 GB/s at
  `N = 2^26`; small inputs are dominated by roughly 4–6 microseconds of fixed
  launch/timing cost.
- Observation: neither provider has a meaningful large-input advantage in this
  first implementation. The result supports the bandwidth-bound hypothesis;
  it does not support a broad claim that Triton is inherently faster.
- Next question: compare the plateau with a profiler-reported DRAM throughput
  before attributing the remaining gap to a specific hardware limit.
