"""
Paged FlashAttention decode for nano-vllm's KV-cache layout.

Mirrors ``flash_attn.flash_attn_with_kvcache``:

  * K/V pool : ``[num_blocks, block_size, num_kv_heads, head_dim]``
  * ``block_table`` : ``[batch, max_num_blocks]`` int32 (logical -> physical)
  * ``cache_seqlens`` : ``[batch]`` int32 (real context length per sequence)

The softmax / GEMM / ``vec_exp2`` body is forked from
``build_flash_attention_decode_kernel``; only K/V loads and masking change.

Do NOT add ``from __future__ import annotations`` — it breaks TileLang's
``@T.prim_func`` type parsing.
"""

from dataclasses import dataclass

import torch
import tilelang.language as T
from tilelang import jit

from nanovllm.backends.tilelang.attention import (
    LOG2E,
    _ATTENTION_PASS_CONFIGS,
    _compile_attention_kernel,
    _exp2_poly,
    _round_up,
    tilelang_dtype,
)

# Engine default ``kvcache_block_size``; must be a multiple of decode ``block_N``.
PAGED_BLOCK_SIZE = 256


@dataclass
class PagedDecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    context_lens: list
    block_size: int
    num_blocks_pool: int
    max_num_blocks: int


def paged_pool_shapes(context_lens, block_size):
    """Derive pool / block_table shapes (must match ``build_paged_decode_inputs``)."""
    batch_size = len(context_lens)
    blocks_per_seq = [(length + block_size - 1) // block_size for length in context_lens]
    max_num_blocks = max(blocks_per_seq) if blocks_per_seq else 1
    total_needed = sum(blocks_per_seq)
    num_blocks_pool = max(total_needed * 2, total_needed + batch_size, 1)
    return {
        "batch_size": batch_size,
        "blocks_per_seq": blocks_per_seq,
        "max_num_blocks": max_num_blocks,
        "num_blocks_pool": num_blocks_pool,
    }


def build_paged_decode_inputs(
    context_lens,
    block_size,
    num_heads,
    num_kv_heads,
    head_dim,
    dtype=torch.float32,
    device="cpu",
    seed=0,
    num_blocks_pool=None,
):
    """Allocate a shared block pool with non-contiguous physical block ids.

    Returns tensors matching the engine / ``flash_attn_with_kvcache`` contract.
    """
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}."
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    shapes = paged_pool_shapes(context_lens, block_size)
    batch_size = shapes["batch_size"]
    blocks_per_seq = shapes["blocks_per_seq"]
    max_num_blocks = shapes["max_num_blocks"]
    if num_blocks_pool is None:
        num_blocks_pool = shapes["num_blocks_pool"]
    if num_blocks_pool < sum(blocks_per_seq):
        raise ValueError(
            f"num_blocks_pool={num_blocks_pool} < required {sum(blocks_per_seq)} blocks"
        )

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    k_cache = torch.zeros(
        num_blocks_pool, block_size, num_kv_heads, head_dim,
        device=device, dtype=dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    # Fill the whole pool with noise so unused blocks are not trivially zero.
    k_cache.normal_(generator=gen)
    v_cache.normal_(generator=gen)

    # Assign physical blocks from the *end* of the free list so ids are
    # non-contiguous relative to logical order (exercises block_table indirection).
    free_ids = list(range(num_blocks_pool - 1, -1, -1))
    block_table = torch.full(
        (batch_size, max_num_blocks), -1, dtype=torch.int32, device=device
    )
    for b, (length, n_blocks) in enumerate(zip(context_lens, blocks_per_seq)):
        phys = [free_ids.pop(0) for _ in range(n_blocks)]
        for i, pid in enumerate(phys):
            block_table[b, i] = pid
        # Overwrite only the live tokens in assigned blocks with fresh noise
        # so the golden and kernel see identical values.
        for t in range(length):
            pid = phys[t // block_size]
            off = t % block_size
            k_cache[pid, off].normal_(generator=gen)
            v_cache[pid, off].normal_(generator=gen)

    q = torch.randn(
        batch_size, num_heads, head_dim, device=device, dtype=dtype, generator=gen
    )
    cache_seqlens = torch.tensor(context_lens, dtype=torch.int32, device=device)
    return PagedDecodeInputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        context_lens=list(context_lens),
        block_size=block_size,
        num_blocks_pool=num_blocks_pool,
        max_num_blocks=max_num_blocks,
    )


def gather_dense_kv(
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    block_size,
):
    """Gather paged K/V into dense ``[batch, max_ctx, kv_heads, dim]`` tensors."""
    batch_size = block_table.size(0)
    max_ctx = int(cache_seqlens.max().item()) if cache_seqlens.numel() else 0
    num_kv_heads = k_cache.size(2)
    head_dim = k_cache.size(3)
    device = k_cache.device
    dtype = k_cache.dtype

    k_dense = torch.zeros(
        batch_size, max_ctx, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_dense = torch.zeros_like(k_dense)
    for b in range(batch_size):
        length = int(cache_seqlens[b].item())
        for t in range(length):
            pid = int(block_table[b, t // block_size].item())
            if pid < 0:
                raise ValueError(f"missing block_table entry for seq {b} token {t}")
            off = t % block_size
            k_dense[b, t] = k_cache[pid, off]
            v_dense[b, t] = v_cache[pid, off]
    return k_dense, v_dense


def paged_decode_reference(
    q,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    block_size,
    softmax_scale=None,
):
    """Pure-PyTorch single-query GQA golden (float32) over a paged KV pool."""
    if q.dim() == 4:
        if q.size(1) != 1:
            raise ValueError(f"q seq dim must be 1, got {tuple(q.shape)}")
        q = q.squeeze(1)
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.size(2)
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}."
        )
    group_size = num_heads // num_kv_heads
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    k_dense, v_dense = gather_dense_kv(
        k_cache, v_cache, block_table, cache_seqlens, block_size
    )
    out = torch.empty_like(q)
    qf, kf, vf = q.float(), k_dense.float(), v_dense.float()
    for b in range(batch_size):
        length = int(cache_seqlens[b].item())
        q_b = qf[b]
        k_b = kf[b, :length].repeat_interleave(group_size, dim=1)
        v_b = vf[b, :length].repeat_interleave(group_size, dim=1)
        scores = torch.einsum("hd,jhd->hj", q_b, k_b) * float(softmax_scale)
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hj,jhd->hd", probs, v_b).to(out.dtype)
    return out


# --------------------------------------------------------------------------- #
# TileLang paged decode kernel
# --------------------------------------------------------------------------- #
@jit(
    out_idx=[5],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_flash_attention_decode_paged_kernel(
    batch_size: int,
    num_blocks_pool: int,
    block_size: int,
    max_num_blocks: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: float,
    block_N: int = 128,
    block_H: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    vec_exp2: bool = False,
):
    """Paged-KV decode kernel matching ``flash_attn_with_kvcache``.

    Tensor contract (engine per-layer cache + decode metadata):
      Q            : (batch_size, num_heads, head_dim)
      K_cache      : (num_blocks_pool, block_size, num_kv_heads, head_dim)
      V_cache      : (num_blocks_pool, block_size, num_kv_heads, head_dim)
      block_table  : (batch_size, max_num_blocks) int32
      cache_seqlens: (batch_size,) int32
      -> Output    : (batch_size, num_heads, head_dim)

    Requires ``block_size % block_N == 0`` so each attention tile stays inside
    one physical KV block (same constraint as TileLang MLA paged decode).
    """
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    assert block_size % block_N == 0, (
        f"block_size={block_size} must be a multiple of block_N={block_N}"
    )
    scale = softmax_scale * LOG2E
    accum_dtype = "float32"
    kv_group_num = num_heads // num_kv_heads
    valid_block_H = min(block_H, kv_group_num)
    # Compile-time padded KV length (covers the widest block_table row).
    seqlen_kv = max_num_blocks * block_size

    def _exp2(v):
        return _exp2_poly(v) if vec_exp2 else T.exp2(v)

    need_cast = in_dtype != accum_dtype

    shape_q = [batch_size, num_heads, head_dim]
    shape_cache = [num_blocks_pool, block_size, num_kv_heads, head_dim]
    shape_bt = [batch_size, max_num_blocks]
    shape_cs = [batch_size]
    shape_o = [batch_size, num_heads, head_dim]

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, in_dtype),
        K_cache: T.Tensor(shape_cache, in_dtype),
        V_cache: T.Tensor(shape_cache, in_dtype),
        block_table: T.Tensor(shape_bt, "int32"),
        cache_seqlens: T.Tensor(shape_cs, "int32"),
        Output: T.Tensor(shape_o, in_dtype),
    ):
        with T.Kernel(batch_size, num_heads // valid_block_H, threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([block_H, head_dim], in_dtype)
            K_shared = T.alloc_shared([block_N, head_dim], in_dtype)
            V_shared = T.alloc_shared([block_N, head_dim], in_dtype)
            O_shared = T.alloc_shared([valid_block_H, head_dim], in_dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            if need_cast:
                acc_s_cast = T.alloc_fragment([block_H, block_N], in_dtype)
            acc_o = T.alloc_fragment([block_H, head_dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # Logical token base for this tile; map through block_table.
                phys = block_table[bid, (k * block_N) // block_size]
                off = (k * block_N) % block_size
                T.copy(K_cache[phys, off : off + block_N, cur_kv_head, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        k * block_N + j >= cache_seqlens[bid],
                        -T.infinity(accum_dtype),
                        acc_s[i, j],
                    )
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = _exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = _exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                if need_cast:
                    T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_H, head_dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(V_cache[phys, off : off + block_N, cur_kv_head, :], V_shared)
                T.gemm(acc_s_cast if need_cast else acc_s, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_H, head_dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

    return main


def run_tilelang_attention_decode_paged(
    q,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    softmax_scale,
    block_N=128,
    block_H=64,
    num_stages=2,
    threads=128,
    backend="cuda",
):
    """Run the paged decode kernel; returns ``(batch, num_heads, head_dim)``."""
    if q.dim() != 3:
        raise ValueError(f"q must be 3D (batch, heads, dim); got {tuple(q.shape)}")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError(
            "k_cache/v_cache must be 4D (num_blocks, block_size, kv_heads, dim); "
            f"got {tuple(k_cache.shape)}, {tuple(v_cache.shape)}"
        )
    if block_table.dim() != 2:
        raise ValueError(
            f"block_table must be 2D (batch, max_num_blocks); got {tuple(block_table.shape)}"
        )
    if cache_seqlens.dim() != 1:
        raise ValueError(
            f"cache_seqlens must be 1D (batch,); got {tuple(cache_seqlens.shape)}"
        )

    batch_size, num_heads, head_dim = q.shape
    num_blocks_pool, block_size, num_kv_heads, _ = k_cache.shape
    max_num_blocks = block_table.size(1)
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}."
        )
    if block_size % block_N != 0:
        raise ValueError(
            f"block_size={block_size} must be a multiple of block_N={block_N}"
        )
    if block_table.size(0) != batch_size or cache_seqlens.size(0) != batch_size:
        raise ValueError(
            f"batch mismatch: q={batch_size}, block_table={block_table.size(0)}, "
            f"cache_seqlens={cache_seqlens.size(0)}"
        )
    in_dtype = tilelang_dtype(q.dtype)

    # Pad block_table width so max_num_blocks * block_size covers the longest
    # context and stays a multiple of block_N (already true when block_size %
    # block_N == 0). Also pad so at least one full attention tile exists.
    max_ctx = int(cache_seqlens.max().item()) if cache_seqlens.numel() else 0
    need_blocks = max(1, (max_ctx + block_size - 1) // block_size)
    compile_max_blocks = max(max_num_blocks, need_blocks)
    # Ensure compile-time seqlen covers padded max_ctx for the dense-style loop.
    padded_ctx = _round_up(max(max_ctx, 1), block_N)
    compile_max_blocks = max(compile_max_blocks, (padded_ctx + block_size - 1) // block_size)

    # Unused block_table slots are -1 in the engine; remap to physical block 0
    # so padded tiles (masked by cache_seqlens) never OOB-index the pool.
    bt = block_table.to(torch.int32).contiguous().clone()
    if compile_max_blocks != max_num_blocks:
        padded = torch.zeros(
            (batch_size, compile_max_blocks),
            dtype=torch.int32,
            device=block_table.device,
        )
        padded[:, :max_num_blocks] = bt
        bt = padded
    bt[bt < 0] = 0

    build_args = (
        batch_size,
        num_blocks_pool,
        block_size,
        compile_max_blocks,
        num_heads,
        num_kv_heads,
        head_dim,
        float(softmax_scale),
        block_N,
        block_H,
        num_stages,
        threads,
        in_dtype,
        backend == "cpu",
    )
    kernel = _compile_attention_kernel(
        build_flash_attention_decode_paged_kernel, build_args, 5, backend
    )
    return kernel(
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        bt,
        cache_seqlens.to(torch.int32).contiguous(),
    )


def run_tilelang_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    softmax_scale=None,
    causal=True,
    backend="cuda",
    block_N=128,
    block_H=None,
    num_stages=2,
    threads=128,
):
    """``flash_attn_with_kvcache``-compatible TileLang decode entry point.

    Accepts ``q`` as ``(batch, 1, num_heads, head_dim)`` (engine call site) or
    ``(batch, num_heads, head_dim)``. Returns the same rank as the input ``q``.
    """
    del causal  # causal decode over a prefix cache is implicit via cache_seqlens
    squeeze = False
    if q.dim() == 4:
        if q.size(1) != 1:
            raise ValueError(f"q seq dim must be 1, got {tuple(q.shape)}")
        q = q.squeeze(1)
        squeeze = True
    elif q.dim() != 3:
        raise ValueError(f"q must be 3D or 4D; got {tuple(q.shape)}")

    num_heads = q.size(1)
    num_kv_heads = k_cache.size(2)
    head_dim = q.size(2)
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    if block_H is None:
        block_H = max(1, num_heads // num_kv_heads)

    out = run_tilelang_attention_decode_paged(
        q,
        k_cache,
        v_cache,
        block_table,
        cache_seqlens,
        softmax_scale=float(softmax_scale),
        block_N=block_N,
        block_H=block_H,
        num_stages=num_stages,
        threads=threads,
        backend=backend,
    )
    if squeeze:
        return out.unsqueeze(1)
    return out
