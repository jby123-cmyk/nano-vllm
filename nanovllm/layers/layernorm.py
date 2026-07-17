import torch
from torch import nn

from nanovllm.backends.tilelang.runtime import get_tilelang_execution_backend


# Set from ModelRunner via ``set_norm_backend(config.norm_backend)``.
_NORM_BACKEND = "torch"


def set_norm_backend(backend: str) -> None:
    global _NORM_BACKEND
    if backend not in ("torch", "tilelang"):
        raise ValueError(
            f"norm_backend must be 'torch' or 'tilelang', got {backend!r}"
        )
    _NORM_BACKEND = backend


def get_norm_backend() -> str:
    return _NORM_BACKEND


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if _NORM_BACKEND == "tilelang":
            from nanovllm.backends.tilelang.rmsnorm import run_tilelang_rmsnorm
            return run_tilelang_rmsnorm(
                x, self.weight, residual, eps=self.eps, backend=get_tilelang_execution_backend()
            )
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
