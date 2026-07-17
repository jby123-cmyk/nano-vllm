"""
TileLang linear (GEMM) backend matching ``torch.nn.functional.linear``.

Tensor contract (PyTorch layout)::

    y = x @ weight.T (+ bias)
    x      : (num_tokens, in_features)
    weight : (out_features, in_features)
    bias   : (out_features,) or None
    y      : (num_tokens, out_features)

One parameterized kernel covers every nano-vllm linear call site
(``QKVParallelLinear``, ``MergedColumnParallelLinear``, ``RowParallelLinear``,
``ReplicatedLinear``, ``ParallelLMHead``). ``RowParallelLinear`` still owns its
``all_reduce`` wrapper outside this kernel.

Do NOT add ``from __future__ import annotations`` — it breaks TileLang's
``@T.prim_func`` type parsing.
"""

import os

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

# Default tiles; VLEN-friendly and divide Qwen3-0.6B N/K dims (1024/2048/3072/…).
DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64


@jit(
    out_idx=[3],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_linear_kernel(
    num_tokens: int,
    in_features: int,
    out_features: int,
    has_bias: bool = False,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = 128,
    in_dtype: str = "float32",
):
    """Tiled ``y = x @ weight.T (+ bias)`` with Weight in PyTorch ``[out, in]`` layout.

    Bias is always a kernel argument (pass zeros when unused). ``has_bias`` only
    controls whether the epilogue runs, so the signature stays fixed for JIT /
    Spike packing.
    """
    accum_dtype = "float32"
    x_shape = [num_tokens, in_features]
    w_shape = [out_features, in_features]
    b_shape = [out_features]
    y_shape = [num_tokens, out_features]

    @T.prim_func
    def main(
        X: T.Tensor(x_shape, in_dtype),
        Weight: T.Tensor(w_shape, in_dtype),
        Bias: T.Tensor(b_shape, in_dtype),
        Y: T.Tensor(y_shape, in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_features, block_N),
            T.ceildiv(num_tokens, block_M),
            threads=threads,
        ) as (bx, by):
            X_shared = T.alloc_shared([block_M, block_K], in_dtype)
            W_shared = T.alloc_shared([block_N, block_K], in_dtype)
            Y_local = T.alloc_fragment([block_M, block_N], accum_dtype)

            T.clear(Y_local)
            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=1):
                T.copy(
                    X[by * block_M : (by + 1) * block_M, ko * block_K : (ko + 1) * block_K],
                    X_shared,
                )
                T.copy(
                    Weight[
                        bx * block_N : (bx + 1) * block_N,
                        ko * block_K : (ko + 1) * block_K,
                    ],
                    W_shared,
                )
                T.gemm(
                    X_shared,
                    W_shared,
                    Y_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            if has_bias:
                for i, j in T.Parallel(block_M, block_N):
                    Y_local[i, j] += Bias[bx * block_N + j]

            # Tokens/features are padded to block multiples by the run helper,
            # so a full-tile store is always in-bounds (avoids a Parallel store race).
            T.copy(
                Y_local,
                Y[
                    by * block_M : (by + 1) * block_M,
                    bx * block_N : (bx + 1) * block_N,
                ],
            )

    return main


from nanovllm.backends.tilelang.runtime import compile_tilelang_kernel


def _compile_linear_kernel(build_args: tuple, out_idx: int, backend: str):
    """Compile ``build_linear_kernel`` for ``backend`` and return a callable."""
    return compile_tilelang_kernel(
        build_linear_kernel,
        build_args,
        out_idx,
        backend,
        kernel_name="linear",
    )


def run_tilelang_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    backend: str = "cuda",
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = 128,
) -> torch.Tensor:
    """Run tiled linear; returns ``(num_tokens, out_features)``.

    ``backend`` selects CUDA JIT (engine path) or host ``llvm`` (RVV golden).
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D (tokens, in_features); got {tuple(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(
            f"weight must be 2D (out_features, in_features); got {tuple(weight.shape)}"
        )

    num_tokens, in_features = x.shape
    out_features, in_w = weight.shape
    if in_w != in_features:
        raise ValueError(
            f"weight.in_features={in_w} must equal x.in_features={in_features}"
        )
    if bias is not None and tuple(bias.shape) != (out_features,):
        raise ValueError(
            f"bias must be ({out_features},); got {tuple(bias.shape)}"
        )

    # Pad so the final tile never reads past the allocation.
    padded_m = _round_up(max(num_tokens, 1), block_M)
    padded_n = _round_up(out_features, block_N)
    padded_k = _round_up(in_features, block_K)

    x_pad = x
    if padded_m != num_tokens or padded_k != in_features:
        x_pad = torch.zeros(
            padded_m, padded_k, device=x.device, dtype=x.dtype
        )
        x_pad[:num_tokens, :in_features] = x

    w_pad = weight
    if padded_n != out_features or padded_k != in_features:
        w_pad = torch.zeros(
            padded_n, padded_k, device=weight.device, dtype=weight.dtype
        )
        w_pad[:out_features, :in_features] = weight

    has_bias = bias is not None
    if bias is None:
        b_pad = torch.zeros(padded_n, device=x.device, dtype=x.dtype)
    elif padded_n != out_features:
        b_pad = torch.zeros(padded_n, device=bias.device, dtype=bias.dtype)
        b_pad[:out_features] = bias
    else:
        b_pad = bias

    in_dtype = tilelang_dtype(x.dtype)
    build_args = (
        padded_m,
        padded_k,
        padded_n,
        has_bias,
        block_M,
        block_N,
        block_K,
        threads,
        in_dtype,
    )
    kernel = _compile_linear_kernel(build_args, 3, backend)
    out = kernel(
        x_pad.contiguous(),
        w_pad.contiguous(),
        b_pad.contiguous(),
    )
    return out[:num_tokens, :out_features]


def dump_linear_tensorir(
    num_tokens: int,
    in_features: int,
    out_features: int,
    has_bias: bool = False,
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
    threads: int = 128,
    in_dtype: str = "float32",
    dump_path: str = "",
) -> str:
    tir = build_linear_kernel.get_tir(
        num_tokens,
        in_features,
        out_features,
        has_bias,
        block_M,
        block_N,
        block_K,
        threads,
        in_dtype,
    )
    tir_text = tir.script()
    banner = "TensorIR — linear (F.linear / GEMM transpose_B)"
    print("=" * 72)
    print(banner)
    print("=" * 72)
    print(tir_text)
    print("=" * 72)
    if dump_path:
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as handle:
            handle.write(tir_text)
        print(f"Saved TensorIR to: {dump_path}")
    return tir_text
