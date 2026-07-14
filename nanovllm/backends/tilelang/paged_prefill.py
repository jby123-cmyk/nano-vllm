"""
Paged FlashAttention prefill for nano-vllm's KV-cache layout.

Mirrors ``flash_attn.flash_attn_varlen_func`` with ``block_table`` (prefix cache):

  * Q : packed ``[total_q, num_heads, head_dim]`` (new query tokens only)
  * K/V pool : ``[num_blocks, block_size, num_kv_heads, head_dim]``
  * ``block_table`` : ``[batch, max_num_blocks]`` int32
  * ``cu_seqlens_q`` / ``cu_seqlens_k`` : prefix sums (K length >= Q when cached)

When ``block_table`` is ``None``, use ``run_tilelang_attention_prefill`` from
``attention.py`` (packed varlen K/V).

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
    run_tilelang_attention_prefill,
    tilelang_dtype,
)
from nanovllm.backends.tilelang.paged_decode import PAGED_BLOCK_SIZE, paged_pool_shapes

# Prefill tile N; must divide engine block_size (256).
PREFILL_BLOCK_N = 64


@dataclass
class PagedPrefillInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    specs: list[tuple[int, int]]
    block_size: int
    num_blocks_pool: int
    max_num_blocks: int


def flatten_prefill_specs(specs: list[tuple[int, int]]) -> list[int]:
    """Encode ``[(cached, new_q), ...]`` as a flat list for case ids."""
    out: list[int] = []
    for cached, new_q in specs:
        out.extend([cached, new_q])
    return out


def unflatten_prefill_specs(flat: list[int]) -> list[tuple[int, int]]:
    if len(flat) % 2 != 0:
        raise ValueError(f"prefill spec flat list must have even length, got {flat}")
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def prefill_paged_pool_shapes(specs: list[tuple[int, int]], block_size: int) -> dict:
    """Pool shapes for prefix-cache prefill (KV length = cached + new per seq)."""
    seqlen_k_list = [cached + new_q for cached, new_q in specs]
    return paged_pool_shapes(seqlen_k_list, block_size)


def _write_logical_kv(
    k_cache,
    v_cache,
    block_table,
    batch_idx,
    logical_pos,
    k_vec,
    v_vec,
    block_size,
):
    pid = int(block_table[batch_idx, logical_pos // block_size].item())
    off = logical_pos % block_size
    k_cache[pid, off] = k_vec
    v_cache[pid, off] = v_vec


def build_paged_prefill_inputs(
    specs: list[tuple[int, int]],
    block_size,
    num_heads,
    num_kv_heads,
    head_dim,
    dtype=torch.float32,
    device="cpu",
    seed=0,
    num_blocks_pool=None,
):
    """Build prefix-cache prefill tensors matching the engine contract.

    Each ``(num_cached, seqlen_q)`` tuple is one sequence: ``seqlen_k =
    num_cached + seqlen_q``.  Q holds only the ``seqlen_q`` new tokens; K/V
  live in the shared paged pool addressed by ``block_table``.
    """
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}."
        )
    shapes = prefill_paged_pool_shapes(specs, block_size)
    batch_size = len(specs)
    if num_blocks_pool is None:
        num_blocks_pool = shapes["num_blocks_pool"]
    max_num_blocks = shapes["max_num_blocks"]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    k_cache = torch.zeros(
        num_blocks_pool, block_size, num_kv_heads, head_dim,
        device=device, dtype=dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    k_cache.normal_(generator=gen)
    v_cache.normal_(generator=gen)

    free_ids = list(range(num_blocks_pool - 1, -1, -1))
    block_table = torch.full(
        (batch_size, max_num_blocks), -1, dtype=torch.int32, device=device
    )

    q_chunks = []
    cu_q = [0]
    cu_k = [0]
    max_seqlen_q = 0

    for b, (num_cached, seqlen_q) in enumerate(specs):
        seqlen_k = num_cached + seqlen_q
        n_blocks = (seqlen_k + block_size - 1) // block_size
        phys = [free_ids.pop(0) for _ in range(n_blocks)]
        for i, pid in enumerate(phys):
            block_table[b, i] = pid

        # Fill logical KV 0..seqlen_k-1 in the paged pool.
        for t in range(seqlen_k):
            k_vec = torch.randn(
                num_kv_heads, head_dim, device=device, dtype=dtype, generator=gen
            )
            v_vec = torch.randn(
                num_kv_heads, head_dim, device=device, dtype=dtype, generator=gen
            )
            _write_logical_kv(
                k_cache, v_cache, block_table, b, t, k_vec, v_vec, block_size
            )

        q_b = torch.randn(
            seqlen_q, num_heads, head_dim, device=device, dtype=dtype, generator=gen
        )
        q_chunks.append(q_b)
        cu_q.append(cu_q[-1] + seqlen_q)
        cu_k.append(cu_k[-1] + seqlen_k)
        max_seqlen_q = max(max_seqlen_q, seqlen_q)

    q = torch.cat(q_chunks, dim=0) if q_chunks else torch.empty(
        0, num_heads, head_dim, device=device, dtype=dtype
    )
    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)

    return PagedPrefillInputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        specs=list(specs),
        block_size=block_size,
        num_blocks_pool=num_blocks_pool,
        max_num_blocks=max_num_blocks,
    )


def gather_seq_kv(k_cache, v_cache, block_table, batch_idx, seqlen_k, block_size):
    """Gather one sequence's logical K/V from the paged pool."""
    num_kv_heads = k_cache.size(2)
    head_dim = k_cache.size(3)
    device = k_cache.device
    dtype = k_cache.dtype
    k_dense = torch.zeros(seqlen_k, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_dense = torch.zeros_like(k_dense)
    for t in range(seqlen_k):
        pid = int(block_table[batch_idx, t // block_size].item())
        off = t % block_size
        k_dense[t] = k_cache[pid, off]
        v_dense[t] = v_cache[pid, off]
    return k_dense, v_dense


def paged_prefill_reference(
    q,
    k_cache,
    v_cache,
    block_table,
    cu_seqlens_q,
    cu_seqlens_k,
    softmax_scale=None,
    block_size=PAGED_BLOCK_SIZE,
):
    """Pure-PyTorch varlen causal GQA golden over a paged KV pool (float32)."""
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    batch_size = len(cu_q) - 1
    num_heads = q.size(1)
    head_dim = q.size(2)
    num_kv_heads = k_cache.size(2)
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}."
        )
    group_size = num_heads // num_kv_heads
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    out = torch.empty_like(q)
    qf = q.float()
    for b in range(batch_size):
        qs, qe = cu_q[b], cu_q[b + 1]
        ks, ke = cu_k[b], cu_k[b + 1]
        lq, lk = qe - qs, ke - ks
        k_dense, v_dense = gather_seq_kv(
            k_cache, v_cache, block_table, b, lk, block_size
        )
        kf = k_dense.float()
        vf = v_dense.float()
        q_b = qf[qs:qe].transpose(0, 1)
        k_b = kf.repeat_interleave(group_size, dim=1).transpose(0, 1)
        v_b = vf.repeat_interleave(group_size, dim=1).transpose(0, 1)
        scores = torch.einsum("hid,hjd->hij", q_b, k_b) * float(softmax_scale)
        offset = lk - lq
        device = q.device
        i_idx = torch.arange(lq, device=device).view(1, lq, 1)
        j_idx = torch.arange(lk, device=device).view(1, 1, lk)
        causal = j_idx <= (i_idx + offset)
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out_b = torch.einsum("hij,hjd->hid", probs, v_b).transpose(0, 1)
        out[qs:qe] = out_b.to(out.dtype)
    return out


# --------------------------------------------------------------------------- #
# TileLang paged prefill kernel
# --------------------------------------------------------------------------- #
@jit(
    out_idx=[7],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_flash_attention_prefill_paged_kernel(
    batch_size: int,
    total_q: int,
    num_blocks_pool: int,
    block_size: int,
    max_num_blocks: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: float,
    is_causal: bool = True,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    in_dtype: str = "float16",
    vec_exp2: bool = False,
):
    """Prefix-cache varlen prefill over a paged KV pool.

    Tensor contract (engine prefill with ``block_tables``):
      Q_unpad       : (total_q, num_heads, head_dim)
      K_cache       : (num_blocks_pool, block_size, num_kv_heads, head_dim)
      V_cache       : same
      block_table   : (batch_size, max_num_blocks) int32
      cu_seqlens_q/k: (batch_size + 1,) int32
      max_seqlen_q  : runtime scalar
      -> Output     : (total_q, num_heads, head_dim)
    """
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    assert block_size % block_N == 0, (
        f"block_size={block_size} must be a multiple of block_N={block_N}"
    )
    group_size = num_heads // num_kv_heads
    scale = softmax_scale * LOG2E
    accum_dtype = "float32"

    def _exp2(v):
        return _exp2_poly(v) if vec_exp2 else T.exp2(v)

    need_cast = in_dtype != accum_dtype

    q_shape = [total_q, num_heads, head_dim]
    cache_shape = [num_blocks_pool, block_size, num_kv_heads, head_dim]
    bt_shape = [batch_size, max_num_blocks]
    cu_shape = [batch_size + 1]
    o_shape = [total_q, num_heads, head_dim]

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, in_dtype),
        K_cache: T.Tensor(cache_shape, in_dtype),
        V_cache: T.Tensor(cache_shape, in_dtype),
        block_table: T.Tensor(bt_shape, "int32"),
        cu_seqlens_q: T.Tensor(cu_shape, "int32"),
        cu_seqlens_k: T.Tensor(cu_shape, "int32"),
        max_seqlen_q: T.int32,
        Output_unpad: T.Tensor(o_shape, in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(max_seqlen_q, block_M), num_heads, batch_size, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, head_dim], in_dtype)
            K_shared = T.alloc_shared([block_N, head_dim], in_dtype)
            V_shared = T.alloc_shared([block_N, head_dim], in_dtype)
            O_shared = T.alloc_shared([block_M, head_dim], in_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            if need_cast:
                acc_s_cast = T.alloc_fragment([block_M, block_N], in_dtype)
            acc_o = T.alloc_fragment([block_M, head_dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            batch_idx = bz
            head_idx = by
            kv_head_idx = head_idx // group_size

            q_start_idx = cu_seqlens_q[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = k_end_idx - cu_seqlens_k[batch_idx]

            T.copy(
                Q_unpad[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :],
                Q_shared,
            )

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            offset = kv_current_seqlen - q_current_seqlen
            max_visible_k_idx = offset + (bx + 1) * block_M
            loop_range = (
                T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
                if is_causal
                else T.ceildiv(kv_current_seqlen, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                logical_kv_pos = k * block_N
                phys = block_table[batch_idx, logical_kv_pos // block_size]
                off = logical_kv_pos % block_size
                T.copy(K_cache[phys, off : off + block_N, kv_head_idx, :], K_shared)

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i + offset < k * block_N + j)
                            or (bx * block_M + i >= q_current_seqlen or k * block_N + j >= kv_current_seqlen),
                            -1e9,
                            0,
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i >= q_current_seqlen or k * block_N + j >= kv_current_seqlen),
                            -1e9,
                            0,
                        )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = _exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = _exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                if need_cast:
                    T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, head_dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V_cache[phys, off : off + block_N, kv_head_idx, :], V_shared)
                T.gemm(acc_s_cast if need_cast else acc_s, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, head_dim):
                acc_o[i, j] = (
                    0 if is_causal and bx * block_M + i + offset < 0 else acc_o[i, j] / logsum[i]
                )

            T.copy(acc_o, O_shared)
            for i, d in T.Parallel(block_M, head_dim):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    return main


def run_tilelang_attention_prefill_paged(
    q,
    k_cache,
    v_cache,
    block_table,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    softmax_scale,
    is_causal=True,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=128,
    backend="cuda",
):
    """Run paged prefix-cache prefill; returns ``(total_q, num_heads, head_dim)``."""
    if q.dim() != 3:
        raise ValueError(f"q must be 3D (total_q, heads, dim); got {tuple(q.shape)}")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError(
            "k_cache/v_cache must be 4D (num_blocks, block_size, kv_heads, dim)"
        )

    total_q, num_heads, head_dim = q.shape
    num_blocks_pool, block_size, num_kv_heads, _ = k_cache.shape
    batch_size = cu_seqlens_q.numel() - 1
    max_num_blocks = block_table.size(1)
    if block_size % block_N != 0:
        raise ValueError(
            f"block_size={block_size} must be a multiple of block_N={block_N}"
        )
    in_dtype = tilelang_dtype(q.dtype)

    padded_q = _round_up(total_q, block_M)
    compile_max_blocks = max_num_blocks
    max_ctx = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()) if batch_size else 0
    need_blocks = max(1, (max_ctx + block_size - 1) // block_size)
    compile_max_blocks = max(compile_max_blocks, need_blocks)
    padded_max_q = _round_up(max(max_seqlen_q, 1), block_M)

    q_pad = q
    if padded_q != total_q:
        q_pad = torch.zeros(padded_q, num_heads, head_dim, device=q.device, dtype=q.dtype)
        q_pad[:total_q] = q

    bt = block_table.to(torch.int32).contiguous().clone()
    if compile_max_blocks != max_num_blocks:
        padded_bt = torch.zeros(
            (batch_size, compile_max_blocks), dtype=torch.int32, device=block_table.device
        )
        padded_bt[:, :max_num_blocks] = bt
        bt = padded_bt
    bt[bt < 0] = 0

    build_args = (
        batch_size,
        padded_q,
        num_blocks_pool,
        block_size,
        compile_max_blocks,
        num_heads,
        num_kv_heads,
        head_dim,
        float(softmax_scale),
        is_causal,
        block_M,
        block_N,
        num_stages,
        threads,
        in_dtype,
        backend == "cpu",
    )
    kernel = _compile_attention_kernel(
        build_flash_attention_prefill_paged_kernel, build_args, 7, backend
    )
    out = kernel(
        q_pad.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        bt,
        cu_seqlens_q.to(torch.int32).contiguous(),
        cu_seqlens_k.to(torch.int32).contiguous(),
        int(padded_max_q),
    )
    return out[:total_q]


def run_tilelang_flash_attn_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    softmax_scale=None,
    causal=True,
    block_table=None,
    backend="cuda",
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=128,
):
    """``flash_attn_varlen_func``-compatible TileLang prefill entry point.

    When ``block_table`` is ``None``, ``k``/``v`` are packed
    ``(total_kv, kv_heads, head_dim)``.  When set, ``k``/``v`` are the paged
    pool ``(num_blocks, block_size, kv_heads, head_dim)``.
    """
    head_dim = q.size(-1)
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    if block_table is None:
        return run_tilelang_attention_prefill(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            softmax_scale=float(softmax_scale),
            is_causal=causal,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
            backend=backend,
        )

    return run_tilelang_attention_prefill_paged(
        q,
        k,
        v,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        softmax_scale=float(softmax_scale),
        is_causal=causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
        backend=backend,
    )
