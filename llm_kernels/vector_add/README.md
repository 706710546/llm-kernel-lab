# Vector Add

This is the first complete experiment in LLM Kernel Lab. The arithmetic is
simple on purpose: it isolates GPU launch overhead and global-memory bandwidth.

## 1. Mathematical definition

For two vectors of equal length:

```text
output[i] = x[i] + y[i]
```

## 2. Input and output shape

The first implementation accepts equal-shaped, contiguous CUDA tensors. The
kernel treats every tensor as a flat vector of `N` elements.

## 3. PyTorch baseline

The reference is `torch.add`, expressed as `x + y`.

## 4. Theoretical FLOPs

There is one floating-point addition per element, so the modeled work is `N`
FLOPs.

## 5. Modeled memory traffic

Each element requires two global-memory reads and one global-memory write. For
FP32, that is `4 + 4 + 4 = 12` bytes per element, or `12N` bytes in total.

## 6. Arithmetic intensity

For FP32:

```text
AI = N FLOPs / 12N bytes = 1/12 FLOP/byte
```

This is very low.

## 7. Bottleneck hypothesis

Large vectors should be memory-bandwidth bound. Very small vectors should be
dominated by fixed costs such as kernel launch and timing overhead.

## 8. GPU mapping

One Triton program processes `block_size` contiguous elements. Adjacent lanes
load adjacent addresses, enabling coalesced global-memory accesses. A mask keeps
the final partial block in bounds when `N` is not divisible by `block_size`.

## 9. Optimization used

The first version uses a one-dimensional grid, contiguous access, and one fused
load-add-store kernel. There is deliberately no autotuning yet.

## 10. Does the benchmark support the hypothesis?

The first local FP32 run on an RTX 3080 Ti reached approximately 816–818 GB/s
at `N = 2^26` for both PyTorch and Triton. At `N <= 2^14`, the measured
latency stayed around 4–5 microseconds, so fixed launch/timing costs dominated.
The raw measurements are committed in
`benchmarks/results/vector_add_rtx3080ti_fp32.csv`.

This supports the memory-bandwidth hypothesis for large vectors, while showing
that the simple Triton implementation is essentially tied with PyTorch at the
largest sizes. Next, answer these questions with profiler evidence:

1. At what size does bandwidth stop increasing rapidly?
2. How close is the plateau to the RTX 3080 Ti's practical memory bandwidth?
3. Which provider wins for small vectors, and why?
4. Do p20–p80 ranges indicate stable measurements?

Do not claim a speedup until the measurements have been recorded and explained.
