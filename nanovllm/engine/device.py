"""Device selection helpers for the inference engine."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.config import Config


def is_rvv_device(config: "Config") -> bool:
    return config.device == "rvv"


def is_cuda_device(config: "Config") -> bool:
    return config.device == "cuda"


def torch_device_name(config: "Config") -> str:
    return "cpu" if is_rvv_device(config) else "cuda"


def dist_backend_name(config: "Config") -> str:
    return "gloo" if is_rvv_device(config) else "nccl"


def default_rvv_compile_only() -> bool:
    """Cross-compile by default on non-RISC-V hosts (see usage.md)."""
    return platform.machine() not in ("riscv64", "riscv")


def use_pin_memory(config: "Config") -> bool:
    return is_cuda_device(config)
