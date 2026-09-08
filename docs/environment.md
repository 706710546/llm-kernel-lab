# 开发环境检查

检查日期：2026-09-08。

## 当前已经可用

| 组件 | 检测结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti，12 GB，计算能力 8.6 |
| NVIDIA 驱动 | 591.86，最高兼容 CUDA 13.1 |
| CUDA Toolkit | 12.6，`nvcc` 12.6.85 |
| Python | 3.11.9 |
| PyTorch | 2.14.0+cu126，CUDA 可用 |
| Triton | triton-windows 3.8.0.post28 |
| MSVC | Visual Studio Build Tools 2022，编译器 19.44 |
| CMake | 4.4.3 |
| Ninja | 1.11.1 |
| Nsight Compute CLI | 2024.3.2 |

已经在 RTX 3080 Ti 上实际运行并通过以下测试：

- Triton Vector Add；
- PyTorch FP16 `4096 × 4096` 矩阵乘法。

因此当前 PyTorch、Triton、驱动和 CUDA Runtime 的组合可以正常执行 GPU 代码。

## 需要注意的细节

- `cl.exe` 已安装，但普通 PowerShell 会话没有加载它。以后编译 CUDA 扩展时，需要先
  初始化 Visual Studio 开发者环境，或者让 CMake 自动选择已经安装的 Build Tools。
- NVIDIA 驱动、PyTorch CUDA Runtime 和 CUDA Toolkit 的版本号不必完全相同。
  当前关键的 CUDA 12.6 路径已经通过真实 GPU 程序验证。
- Windows 环境使用 `triton-windows`。如果以后迁移到 Linux，应改用官方 `triton` 包。

## 第一个里程碑暂时不需要

- 尚未安装 Nsight Systems（`nsys`）。
- 尚未安装 GNU Make。Windows 下已有 CMake + Ninja，后续 CUDA 构建优先使用它们。
- 尚未安装 GitHub CLI（`gh`），也没有检测到 SSH 密钥。目前使用 HTTPS 推送。

## Git 配置

本仓库使用以下本地身份，不影响其他 Git 仓库：

```text
706710546 <313013248+706710546@users.noreply.github.com>
```

