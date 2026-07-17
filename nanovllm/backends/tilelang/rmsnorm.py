"""
TileLang RMSNorm matching ``nanovllm.layers.layernorm.RMSNorm``.

Two compile-time modes (``fuse_residual``):

  * False: ``y = rms(x) * weight``
  * True:  ``x = x + residual``; ``residual_out = x``; ``y = rms(x) * weight``

Tensor contract (2D; callers flatten higher-rank inputs)::

    x, y, residual : (num_tokens, hidden)
    weight         : (hidden,)

Do NOT add ``from __future__ import annotations``.
"""

import torch
import tilelang
import tilelang.language as T
from tilelang import jit
from tilelang import tvm as tvm

from nanovllm.backends.tilelang.attention import (
    _ATTENTION_PASS_CONFIGS,
    _round_up,
    tilelang_dtype,
)

DEFAULT_BLOCK_M = 1  # one token per program — simple, correct reduce


@jit(
    out_idx=[3, 4],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_rmsnorm_kernel(
    num_tokens: int,
    hidden: int,
    eps: float = 1e-6,
    fuse_residual: bool = False,
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = 128,
    in_dtype: str = "float32",
):
    """RMSNorm (+ optional fused residual). Fixed 5-tensor signature for Spike."""
    accum_dtype = "float32"
    x_shape = [num_tokens, hidden]
    w_shape = [hidden]

    @T.prim_func
    def main(
        X: T.Tensor(x_shape, in_dtype),
        Weight: T.Tensor(w_shape, in_dtype),
        Residual: T.Tensor(x_shape, in_dtype),
        Y: T.Tensor(x_shape, in_dtype),
        ResidualOut: T.Tensor(x_shape, in_dtype),
    ):
        with T.Kernel(num_tokens, threads=threads) as bx:
            x_local = T.alloc_fragment([hidden], accum_dtype)
            acc = T.alloc_fragment([1], accum_dtype)

            if fuse_residual:
                for j in T.Parallel(hidden):
                    val = T.Cast(accum_dtype, X[bx, j]) + T.Cast(
                        accum_dtype, Residual[bx, j]
                    )
                    x_local[j] = val
                    ResidualOut[bx, j] = T.Cast(in_dtype, val)
            else:
                for j in T.Parallel(hidden):
                    x_local[j] = T.Cast(accum_dtype, X[bx, j])

            T.fill(acc, 0)
            for j in T.serial(hidden):
                acc[0] += x_local[j] * x_local[j]
            mean = acc[0] / T.Cast(accum_dtype, hidden)
            scale = T.rsqrt(mean + T.Cast(accum_dtype, eps))
            for j in T.Parallel(hidden):
                Y[bx, j] = T.Cast(
                    in_dtype,
                    x_local[j] * scale * T.Cast(accum_dtype, Weight[j]),
                )

    return main


from nanovllm.backends.tilelang.runtime import compile_tilelang_kernel


def _compile_rmsnorm_kernel(build_args: tuple, backend: str):
    return compile_tilelang_kernel(
        build_rmsnorm_kernel,
        build_args,
        [3, 4],
        backend,
        kernel_name="rmsnorm",
    )


def run_tilelang_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
    backend: str = "cuda",
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = 128,
):
    """Run RMSNorm. Returns ``y`` or ``(y, residual_out)`` when residual is set."""
    orig_shape = x.shape
    hidden = weight.numel()
    if x.shape[-1] != hidden:
        raise ValueError(
            f"x last dim {x.shape[-1]} must equal weight.numel()={hidden}"
        )
    x2 = x.reshape(-1, hidden).contiguous()
    num_tokens = x2.shape[0]
    fuse = residual is not None
    if fuse:
        r2 = residual.reshape(-1, hidden).contiguous()
        if r2.shape != x2.shape:
            raise ValueError(
                f"residual shape {tuple(residual.shape)} must match x {tuple(x.shape)}"
            )
    else:
        r2 = torch.zeros_like(x2)

    in_dtype = tilelang_dtype(x.dtype)
    build_args = (
        num_tokens,
        hidden,
        float(eps),
        fuse,
        block_M,
        threads,
        in_dtype,
    )
    kernel = _compile_rmsnorm_kernel(build_args, backend)
    y2, r_out = kernel(x2, weight.contiguous(), r2)
    y = y2.reshape(orig_shape)
    if fuse:
        return y, r_out.reshape(orig_shape)
    return y
