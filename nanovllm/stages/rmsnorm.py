"""
RMSNorm stage with PyTorch golden (mirrors ``nanovllm.layers.layernorm.RMSNorm``).
"""

from dataclasses import dataclass

import torch

from nanovllm.backends.tilelang.rmsnorm import run_tilelang_rmsnorm


@dataclass
class RMSNormResult:
    reference_output: torch.Tensor
    tilelang_output: torch.Tensor
    max_abs_diff: float
    residual_ref: torch.Tensor | None = None
    residual_tl: torch.Tensor | None = None
    residual_max_abs_diff: float | None = None


class RMSNormStage:
    def __init__(
        self,
        hidden: int,
        eps: float = 1e-6,
        fuse_residual: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        tilelang_backend: str = "cpu",
        seed: int = 0,
    ):
        self.hidden = hidden
        self.eps = eps
        self.fuse_residual = fuse_residual
        self.dtype = dtype
        self.device = device
        self.tilelang_backend = tilelang_backend
        self.seed = seed

    def reference(self, x, weight, residual=None):
        xf = x.float()
        if residual is not None:
            xf = xf + residual.float()
            residual_out = xf.to(x.dtype)
        else:
            residual_out = None
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        y = xf * torch.rsqrt(var + self.eps)
        y = y.to(x.dtype) * weight
        if residual_out is None:
            return y, None
        return y, residual_out

    def run(self, num_tokens: int) -> RMSNormResult:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        x = torch.randn(
            num_tokens, self.hidden, device=self.device, dtype=self.dtype, generator=gen
        )
        weight = torch.randn(
            self.hidden, device=self.device, dtype=self.dtype, generator=gen
        )
        residual = None
        if self.fuse_residual:
            residual = torch.randn(
                num_tokens,
                self.hidden,
                device=self.device,
                dtype=self.dtype,
                generator=gen,
            )
        ref_y, ref_r = self.reference(x, weight, residual)
        if self.fuse_residual:
            tl_y, tl_r = run_tilelang_rmsnorm(
                x, weight, residual, eps=self.eps, backend=self.tilelang_backend
            )
            rdiff = float((tl_r.float() - ref_r.float()).abs().max().item())
        else:
            tl_y = run_tilelang_rmsnorm(
                x, weight, None, eps=self.eps, backend=self.tilelang_backend
            )
            tl_r, rdiff = None, None
        return RMSNormResult(
            reference_output=ref_y,
            tilelang_output=tl_y,
            max_abs_diff=float((tl_y.float() - ref_y.float()).abs().max().item()),
            residual_ref=ref_r,
            residual_tl=tl_r,
            residual_max_abs_diff=rdiff,
        )
