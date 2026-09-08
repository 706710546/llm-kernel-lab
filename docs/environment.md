# Development Environment Audit

Checked on 2026-09-08.

## Ready now

| Component | Detected version / state |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GB, compute capability 8.6 |
| NVIDIA driver | 591.86 (reports CUDA compatibility up to 13.1) |
| CUDA Toolkit | 12.6, `nvcc` 12.6.85 |
| Python | 3.11.9 |
| PyTorch | 2.14.0+cu126; CUDA available |
| Triton | triton-windows 3.8.0.post28 |
| MSVC | Visual Studio Build Tools 2022, compiler 19.44 |
| CMake | 4.4.3 |
| Ninja | 1.11.1 |
| Nsight Compute CLI | 2024.3.2 |

A Triton vector-add smoke test and an FP16 4096×4096 PyTorch matrix
multiplication both ran successfully on the GPU.

## Important details

- `cl.exe` is installed but is not loaded into a normal PowerShell session.
  Initialize the Visual Studio developer environment before compiling a CUDA
  extension, or let CMake select the installed Build Tools instance.
- The NVIDIA driver, PyTorch CUDA runtime, and CUDA Toolkit do not need identical
  version labels. The relevant 12.6 runtime/toolkit path was validated by an
  actual GPU run.
- Native Windows Triton is supplied by `triton-windows`. Linux installations
  should use the official `triton` package instead.

## Not needed for the first milestone

- Nsight Systems (`nsys`) is not installed.
- GNU Make is not installed. CMake + Ninja are available and are the preferred
  Windows build path for future CUDA code.
- GitHub CLI (`gh`) is not installed, and no SSH key was detected.

## Git publishing blocker

The global Git author is currently the placeholder `Administrator
<admin@local>`. Set a real author name and an email associated with GitHub in
this repository before creating the first commit. Do not commit with the
placeholder identity.

