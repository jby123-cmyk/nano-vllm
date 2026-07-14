"""
TileLang SiluAndMul matching ``nanovllm.layers.activation.SiluAndMul``.

  x: (tokens, 2 * intermediate) → silu(gate) * up → (tokens, intermediate)

Do NOT add ``from __future__ import annotations``.
"""

import torch
import tilelang
import tilelang.language as T
from tilelang import jit
from tilelang import tvm as tvm

from nanovllm.backends.tilelang.attention import (
    LOG2E,
    _ATTENTION_PASS_CONFIGS,
    _exp2_poly,
    _round_up,
    tilelang_dtype,
)

DEFAULT_BLOCK_M = 64


@jit(
    out_idx=[1],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_silu_mul_kernel(
    num_tokens: int,
    intermediate: int,
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = 128,
    in_dtype: str = "float32",
):
    accum_dtype = "float32"
    x_shape = [num_tokens, intermediate * 2]
    y_shape = [num_tokens, intermediate]

    @T.prim_func
    def main(
        X: T.Tensor(x_shape, in_dtype),
        Y: T.Tensor(y_shape, in_dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_M), threads=threads) as bx:
            for i, j in T.Parallel(block_M, intermediate):
                row = bx * block_M + i
                if row < num_tokens:
                    gate = T.Cast(accum_dtype, X[row, j])
                    up = T.Cast(accum_dtype, X[row, j + intermediate])
                    # silu(g) = g * sigmoid(g); sigmoid via exp2 poly (no libm expf)
                    sig = T.Cast(accum_dtype, 1.0) / (
                        T.Cast(accum_dtype, 1.0)
                        + _exp2_poly(-gate * T.Cast(accum_dtype, LOG2E))
                    )
                    Y[row, j] = T.Cast(in_dtype, gate * sig * up)

    return main


def _compile_silu_mul_kernel(build_args: tuple, backend: str):
    if backend == "cuda":
        return build_silu_mul_kernel(*build_args)
    if backend == "cpu":
        with tvm.target.Target("llvm"):
            return tilelang.compile(
                build_silu_mul_kernel.get_tir(*build_args),
                out_idx=[1],
                target="llvm",
                target_host="llvm",
                execution_backend="tvm_ffi",
            )
    raise ValueError(f"backend must be 'cuda' or 'cpu', got {backend!r}.")


def run_tilelang_silu_mul(
    x: torch.Tensor,
    backend: str = "cuda",
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = 128,
) -> torch.Tensor:
    if x.dim() != 2 or x.size(-1) % 2 != 0:
        raise ValueError(
            f"x must be 2D with even last dim; got {tuple(x.shape)}"
        )
    num_tokens, two_i = x.shape
    intermediate = two_i // 2
    padded_m = _round_up(max(num_tokens, 1), block_M)
    x_pad = x
    if padded_m != num_tokens:
        x_pad = torch.zeros(padded_m, two_i, device=x.device, dtype=x.dtype)
        x_pad[:num_tokens] = x

    in_dtype = tilelang_dtype(x.dtype)
    build_args = (padded_m, intermediate, block_M, threads, in_dtype)
    kernel = _compile_silu_mul_kernel(build_args, backend)
    out = kernel(x_pad.contiguous())
    return out[:num_tokens]
