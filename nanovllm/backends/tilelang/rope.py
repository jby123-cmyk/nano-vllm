"""
TileLang RoPE apply matching ``nanovllm.layers.rotary_embedding.apply_rotary_emb``.

Cos/sin are gathered in Python from the cache; this kernel only applies the
split-half rotation to Q and K.

  q/k : (tokens, heads, head_dim)
  cos/sin : (tokens, 1, head_dim // 2)  — broadcast over heads

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

DEFAULT_BLOCK_M = 64


@jit(
    out_idx=[4, 5],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_rope_kernel(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = 128,
    in_dtype: str = "float32",
):
    assert head_dim % 2 == 0
    half = head_dim // 2
    accum_dtype = "float32"
    q_shape = [num_tokens, num_heads, head_dim]
    k_shape = [num_tokens, num_kv_heads, head_dim]
    cs_shape = [num_tokens, 1, half]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, in_dtype),
        K: T.Tensor(k_shape, in_dtype),
        Cos: T.Tensor(cs_shape, in_dtype),
        Sin: T.Tensor(cs_shape, in_dtype),
        QOut: T.Tensor(q_shape, in_dtype),
        KOut: T.Tensor(k_shape, in_dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_M), threads=threads) as bx:
            for i, h, d in T.Parallel(block_M, num_heads, half):
                row = bx * block_M + i
                if row < num_tokens:
                    c = T.Cast(accum_dtype, Cos[row, 0, d])
                    s = T.Cast(accum_dtype, Sin[row, 0, d])
                    x1 = T.Cast(accum_dtype, Q[row, h, d])
                    x2 = T.Cast(accum_dtype, Q[row, h, d + half])
                    QOut[row, h, d] = T.Cast(in_dtype, x1 * c - x2 * s)
                    QOut[row, h, d + half] = T.Cast(in_dtype, x2 * c + x1 * s)

            for i, h, d in T.Parallel(block_M, num_kv_heads, half):
                row = bx * block_M + i
                if row < num_tokens:
                    c = T.Cast(accum_dtype, Cos[row, 0, d])
                    s = T.Cast(accum_dtype, Sin[row, 0, d])
                    x1 = T.Cast(accum_dtype, K[row, h, d])
                    x2 = T.Cast(accum_dtype, K[row, h, d + half])
                    KOut[row, h, d] = T.Cast(in_dtype, x1 * c - x2 * s)
                    KOut[row, h, d + half] = T.Cast(in_dtype, x2 * c + x1 * s)

    return main


def _compile_rope_kernel(build_args: tuple, backend: str):
    if backend == "cuda":
        return build_rope_kernel(*build_args)
    if backend == "cpu":
        with tvm.target.Target("llvm"):
            return tilelang.compile(
                build_rope_kernel.get_tir(*build_args),
                out_idx=[4, 5],
                target="llvm",
                target_host="llvm",
                execution_backend="tvm_ffi",
            )
    raise ValueError(f"backend must be 'cuda' or 'cpu', got {backend!r}.")


def run_tilelang_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    backend: str = "cuda",
    block_M: int = DEFAULT_BLOCK_M,
    threads: int = 128,
):
    """Apply RoPE to q/k; returns ``(q_out, k_out)``."""
    if q.dim() != 3 or k.dim() != 3:
        raise ValueError(
            f"q/k must be 3D (tokens, heads, dim); got {tuple(q.shape)}, {tuple(k.shape)}"
        )
    num_tokens, num_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    half = head_dim // 2
    if cos.shape != (num_tokens, 1, half) or sin.shape != (num_tokens, 1, half):
        raise ValueError(
            f"cos/sin must be ({num_tokens}, 1, {half}); "
            f"got {tuple(cos.shape)}, {tuple(sin.shape)}"
        )

    padded_m = _round_up(max(num_tokens, 1), block_M)
    q_pad, k_pad = q, k
    cos_pad, sin_pad = cos, sin
    if padded_m != num_tokens:
        q_pad = torch.zeros(
            padded_m, num_heads, head_dim, device=q.device, dtype=q.dtype
        )
        k_pad = torch.zeros(
            padded_m, num_kv_heads, head_dim, device=k.device, dtype=k.dtype
        )
        cos_pad = torch.zeros(
            padded_m, 1, half, device=cos.device, dtype=cos.dtype
        )
        sin_pad = torch.zeros(
            padded_m, 1, half, device=sin.device, dtype=sin.dtype
        )
        q_pad[:num_tokens] = q
        k_pad[:num_tokens] = k
        cos_pad[:num_tokens] = cos
        sin_pad[:num_tokens] = sin

    in_dtype = tilelang_dtype(q.dtype)
    build_args = (
        padded_m,
        num_heads,
        num_kv_heads,
        head_dim,
        block_M,
        threads,
        in_dtype,
    )
    kernel = _compile_rope_kernel(build_args, backend)
    q_out, k_out = kernel(
        q_pad.contiguous(),
        k_pad.contiguous(),
        cos_pad.contiguous(),
        sin_pad.contiguous(),
    )
    return q_out[:num_tokens], k_out[:num_tokens]
