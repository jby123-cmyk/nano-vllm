"""
TileLang paged KV-cache store matching Triton ``store_kvcache_kernel``.

For each token ``i``, if ``slot_mapping[i] != -1``, write flattened K/V into
the cache at ``slot * D`` where ``D = num_kv_heads * head_dim``.

Contract (engine)::

    key/value : (N, num_kv_heads, head_dim)
    k_cache/v_cache : layout with stride(1) == D  (treated as flat slots)
    slot_mapping : (N,) int32

Do NOT add ``from __future__ import annotations``.
"""

import torch
import tilelang
import tilelang.language as T
from tilelang import jit
from tilelang import tvm as tvm

from nanovllm.backends.tilelang.attention import (
    _ATTENTION_PASS_CONFIGS,
    tilelang_dtype,
)


@jit(
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_kv_store_kernel(
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    num_slots: int,
    threads: int = 128,
    in_dtype: str = "float32",
):
    D = num_kv_heads * head_dim
    kv_shape = [num_tokens, num_kv_heads, head_dim]
    # Flat cache view: (num_slots, D)
    cache_shape = [num_slots, D]

    @T.prim_func
    def main(
        Key: T.Tensor(kv_shape, in_dtype),
        Value: T.Tensor(kv_shape, in_dtype),
        KCache: T.Tensor(cache_shape, in_dtype),
        VCache: T.Tensor(cache_shape, in_dtype),
        SlotMapping: T.Tensor([num_tokens], "int32"),
    ):
        with T.Kernel(num_tokens, threads=threads) as bx:
            slot = SlotMapping[bx]
            if slot >= 0:
                for d in T.Parallel(D):
                    h = d // head_dim
                    off = d % head_dim
                    KCache[slot, d] = Key[bx, h, off]
                    VCache[slot, d] = Value[bx, h, off]

    return main


from nanovllm.backends.tilelang.runtime import compile_tilelang_kernel, get_tilelang_execution_backend


def _compile_kv_store_kernel(build_args: tuple, backend: str):
    return compile_tilelang_kernel(
        build_kv_store_kernel,
        build_args,
        None,
        backend,
        kernel_name="kv_store",
    )


def run_tilelang_store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    backend: str = "cuda",
    threads: int = 128,
) -> None:
    """In-place store into k_cache/v_cache (mutates cache tensors)."""
    if key.dim() != 3 or value.dim() != 3:
        raise ValueError(
            f"key/value must be 3D; got {tuple(key.shape)}, {tuple(value.shape)}"
        )
    num_tokens, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    # Flatten cache to (num_slots, D) without copy when possible.
    k_flat = k_cache.view(-1, D)
    v_flat = v_cache.view(-1, D)
    num_slots = k_flat.shape[0]
    in_dtype = tilelang_dtype(key.dtype)
    build_args = (
        num_tokens,
        num_kv_heads,
        head_dim,
        num_slots,
        threads,
        in_dtype,
    )
    kernel = _compile_kv_store_kernel(build_args, backend)
    result = kernel(
        key.contiguous(),
        value.contiguous(),
        k_flat,
        v_flat,
        slot_mapping.to(torch.int32).contiguous(),
    )
    if get_tilelang_execution_backend() == "rvv" and isinstance(result, tuple):
        k_out, v_out = result
        k_flat.copy_(k_out.to(device=k_flat.device, dtype=k_flat.dtype))
        v_flat.copy_(v_out.to(device=v_flat.device, dtype=v_flat.dtype))
