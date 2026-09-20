"""Vector Add 的 PyTorch CUDA Extension 包装层。"""

from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
import torch.utils.cpp_extension as cpp_extension


MODULE_DIR = Path(__file__).resolve().parent
EXTENSION_DIR = Path(tempfile.gettempdir()) / "llm_kernel_lab_extensions" / "vector_add"
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
        [
            "cmd.exe",
            "/d",
            "/c",
            "call",
            str(vsdevcmd),
            "-arch=x64",
            ">nul",
            "&&",
            "set",
        ],
        text=True,
        errors="replace",
    )

    environment: dict[str, str] = {}
    for line in environment_output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.upper()
        # 某些宿主环境同时存在 PATH 和 Path；优先保留含 MSVC 的版本。
        if normalized_key == "PATH" and normalized_key in environment:
            if "\\VC\\Tools\\MSVC\\" not in value:
                continue
        environment[normalized_key] = value

    os.environ.update(environment)
    if shutil.which("cl") is None:
        raise RuntimeError("MSVC 环境加载失败，请改用 Visual Studio Developer PowerShell")


@lru_cache(maxsize=1)
def _load_extension():
    """首次调用时编译扩展，之后复用进程内缓存和磁盘构建缓存。"""
    _ensure_msvc_environment()

    # 部分中文 Windows 环境会把 Python 的 OEM 编码错误识别为 ``cp1``，
    # 导致 PyTorch 在读取 ``cl`` 的版本信息时抛出 UnicodeDecodeError。
    # 编译器版本号本身是 ASCII，因此使用 UTF-8 并替换非 ASCII 字符是安全的。
    if os.name == "nt":
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

    # Ninja/NVCC 在部分 Windows 环境中无法正确处理中文源码路径。
    # 仓库里的 csrc 仍是唯一源码；这里只把它同步到纯 ASCII 临时目录用于编译。
    SOURCE_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source_names = ("bindings.cpp", "vector_add_cuda.cu")
    for source_name in source_names:
        shutil.copy2(MODULE_DIR / "csrc" / source_name, SOURCE_STAGE_DIR / source_name)

    return cpp_extension.load(
        name="llm_kernel_lab_vector_add_cuda",
        sources=[str(SOURCE_STAGE_DIR / source_name) for source_name in source_names],
        build_directory=str(BUILD_DIR),
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def vector_add_cuda(
    x: torch.Tensor,
    y: torch.Tensor,
    threads_per_block: int = 256,
) -> torch.Tensor:
    """调用自定义 CUDA C++ Extension 执行 FP32 Vector Add。"""
    if not 32 <= threads_per_block <= 1024 or threads_per_block & (threads_per_block - 1):
        raise ValueError("threads_per_block 必须是 [32, 1024] 范围内的 2 的幂")
    return _load_extension().vector_add(x, y, threads_per_block)
