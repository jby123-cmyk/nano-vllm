import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.backends.tilelang.runtime import get_tilelang_execution_backend


_ACT_BACKEND = "torch"


def set_act_backend(backend: str) -> None:
    global _ACT_BACKEND
    if backend not in ("torch", "tilelang"):
        raise ValueError(
            f"act_backend must be 'torch' or 'tilelang', got {backend!r}"
        )
    _ACT_BACKEND = backend


def get_act_backend() -> str:
    return _ACT_BACKEND


class SiluAndMul(nn.Module):

    @torch.compile
    def _torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _ACT_BACKEND == "tilelang":
            from nanovllm.backends.tilelang.activation import run_tilelang_silu_mul
            return run_tilelang_silu_mul(x, backend=get_tilelang_execution_backend())
        return self._torch_forward(x)
