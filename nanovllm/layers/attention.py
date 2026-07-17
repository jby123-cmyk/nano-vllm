import torch
from torch import nn

from nanovllm.utils.context import get_context
from nanovllm.backends.tilelang.runtime import get_tilelang_execution_backend


# Set from ModelRunner via ``set_attn_backend(config.attn_backend)``.
# Default preserves existing flash_attn behavior.
_ATTN_BACKEND = "flash_attn"
_STORE_KVCACHE_KERNEL = None


def set_attn_backend(backend: str) -> None:
    """Select decode attention backend: ``flash_attn`` (default) or ``tilelang``."""
    global _ATTN_BACKEND
    if backend not in ("flash_attn", "tilelang"):
        raise ValueError(
            f"attn_backend must be 'flash_attn' or 'tilelang', got {backend!r}"
        )
    _ATTN_BACKEND = backend


def get_attn_backend() -> str:
    return _ATTN_BACKEND


def _get_store_kvcache_kernel():
    """Lazily JIT-compile the Triton KV-store kernel (CUDA path only)."""
    global _STORE_KVCACHE_KERNEL
    if _STORE_KVCACHE_KERNEL is not None:
        return _STORE_KVCACHE_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    _STORE_KVCACHE_KERNEL = store_kvcache_kernel
    return _STORE_KVCACHE_KERNEL


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    if _ATTN_BACKEND == "tilelang":
        from nanovllm.backends.tilelang.kv_store import run_tilelang_store_kvcache
        run_tilelang_store_kvcache(
            key, value, k_cache, v_cache, slot_mapping,
            backend=get_tilelang_execution_backend(),
        )
        return
    store_kvcache_kernel = _get_store_kvcache_kernel()
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if _ATTN_BACKEND == "tilelang":
                from nanovllm.backends.tilelang.paged_prefill import (
                    run_tilelang_flash_attn_varlen,
                )
                o = run_tilelang_flash_attn_varlen(
                    q,
                    k if context.block_tables is None else k_cache,
                    v if context.block_tables is None else v_cache,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    context.max_seqlen_q,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                    backend=get_tilelang_execution_backend(),
                )
            else:
                from flash_attn import flash_attn_varlen_func

                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if _ATTN_BACKEND == "tilelang":
                from nanovllm.backends.tilelang.paged_decode import (
                    run_tilelang_flash_attn_with_kvcache,
                )
                o = run_tilelang_flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                    backend=get_tilelang_execution_backend(),
                    block_N=min(128, k_cache.size(1)),
                    block_H=max(1, self.num_heads // self.num_kv_heads),
                )
            else:
                from flash_attn import flash_attn_with_kvcache

                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        return o
