# LLM Kernel Lab

A hands-on laboratory for implementing, profiling, and optimizing LLM operators
with PyTorch, Triton, and CUDA.

PyTorch reference → Triton kernel → correctness → benchmark → performance analysis

This project prioritizes learning, correctness, profiling, and architectural
understanding over production completeness.

## Current milestone

Only Phase 0 / Vector Add is in scope. A new operator is added only after the
current one has a tested implementation, reproducible benchmark, and written
performance analysis.

| Operator | PyTorch | Triton | CUDA | Analysis |
|---|---:|---:|---:|---|
| Vector Add | ✓ | ✓ | planned | memory bandwidth |

## Tested environment

- Windows 11, Python 3.11.9
- NVIDIA GeForce RTX 3080 Ti (12 GB, compute capability 8.6)
- NVIDIA driver 591.86
- CUDA Toolkit 12.6
- PyTorch 2.14.0+cu126
- triton-windows 3.8.0.post28
- Visual Studio Build Tools 2022 / MSVC 19.44

See [the environment audit](docs/environment.md) for compiler and publishing
details.

## Run the first experiment

```powershell
python llm_kernels/vector_add/test.py
python llm_kernels/vector_add/benchmark.py
```

The benchmark reports latency and effective bandwidth. For a vector of `N`
FP32 elements, the minimum modeled global-memory traffic is:

```text
read x + read y + write output = 12N bytes
```

## Repository layout

```text
llm-kernel-lab/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── docs/
│   └── gpu-performance-playbook.md
└── llm_kernels/
    └── vector_add/
        ├── torch_impl.py
        ├── triton_impl.py
        ├── test.py
        ├── benchmark.py
        └── README.md
```

## Project question

For every kernel, this repository asks the same question:

> Can measured performance be explained from the algorithm, tensor shape,
> memory traffic, GPU execution model, and implementation choices?
