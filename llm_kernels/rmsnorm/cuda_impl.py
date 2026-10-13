"""RMSNorm CUDA C++ Extension 的 FP32 教学基线。"""

from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
import torch.utils.cpp_extension as cpp_extension


MODULE_DIR = Path(__file__).resolve().parent
EXTENSION_DIR = Path(tempfile.gettempdir()) / "llm_kernel_lab_extensions" / "rmsnorm"
SOURCE_STAGE_DIR = EXTENSION_DIR / "src"
BUILD_DIR = EXTENSION_DIR / "build"


def _ensure_msvc_environment() -> None:
    """在普通 Windows PowerShell 中为当前进程加载 MSVC 编译环境。"""
    if os.name != "nt" or shutil.which("cl") is not None:
        return

    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        raise RuntimeError("未找到 vswhere，请确认 Visual Studio C++ Build Tools 已安装")

    installation_path = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    if not installation_path:
        raise RuntimeError("未找到包含 MSVC x64 工具链的 Visual Studio 安装")

    vsdevcmd = Path(installation_path) / "Common7" / "Tools" / "VsDevCmd.bat"
    environment_output = subprocess.check_output(
        ["cmd.exe", "/d", "/c", "call", str(vsdevcmd), "-arch=x64", ">nul", "&&", "set"],
        text=True,
        errors="replace",
    )
    environment: dict[str, str] = {}
    for line in environment_output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.upper()
        if normalized_key == "PATH" and normalized_key in environment:
            if "\\VC\\Tools\\MSVC\\" not in value:
                continue
        environment[normalized_key] = value

    os.environ.update(environment)
    if shutil.which("cl") is None:
        raise RuntimeError("MSVC 环境加载失败，请改用 Visual Studio Developer PowerShell")


@lru_cache(maxsize=1)
def _load_extension():
    """首次调用编译，随后复用；使用 ASCII 临时路径避开中文路径下的 NVCC 问题。"""
    _ensure_msvc_environment()
    if os.name == "nt":
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

    SOURCE_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source_names = ("bindings.cpp", "rmsnorm_cuda.cu")
    for source_name in source_names:
        shutil.copy2(MODULE_DIR / "csrc" / source_name, SOURCE_STAGE_DIR / source_name)

    return cpp_extension.load(
        name="llm_kernel_lab_rmsnorm_cuda",
        sources=[str(SOURCE_STAGE_DIR / source_name) for source_name in source_names],
        build_directory=str(BUILD_DIR),
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def _validate_inputs(x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    if x.ndim < 1:
        raise ValueError("x 至少必须是一维 Tensor")
    if x.shape[-1] == 0:
        raise ValueError("RMSNorm 的归一化维度不能是空的")
    if not x.is_cuda:
        raise ValueError("x 必须位于 CUDA 设备上")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("CUDA RMSNorm 教学基线只支持 float16、bfloat16 和 float32")
    if not x.is_contiguous():
        raise ValueError("CUDA RMSNorm 教学基线暂时只支持连续 x")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(f"weight 形状必须是 ({x.shape[-1]},)，实际为 {tuple(weight.shape)}")
    if weight.device != x.device:
        raise ValueError("x 与 weight 必须位于同一设备")
    if weight.dtype != x.dtype:
        raise TypeError("CUDA RMSNorm 要求 x 与 weight 使用相同 dtype")
    if not weight.is_contiguous():
        raise ValueError("CUDA RMSNorm 教学基线暂时只支持连续 weight")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps < 0:
        raise ValueError("eps 必须是有限的非负数")


def rmsnorm_cuda(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """运行 CUDA C++ FP32 RMSNorm：每个 CUDA Block 处理 x 的一行。"""
    _validate_inputs(x, weight, eps)
    return _load_extension().rmsnorm(x, weight, float(eps))


def rmsnorm_cuda_shuffle(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """运行 Warp Shuffle 版 CUDA RMSNorm；输入约束与共享内存基线相同。"""
    _validate_inputs(x, weight, eps)
    return _load_extension().rmsnorm_shuffle(x, weight, float(eps))
