"""
TileLang FlashAttention backends for the nano-vllm Qwen3 attention op.

Two independently compiled kernels, mirroring the two halves of
``nanovllm.layers.attention.Attention.forward``:

  * prefill : variable-length, causal, grouped-query attention
              (engine op: ``flash_attn_varlen_func``)
  * decode  : single-query-per-sequence attention over a KV cache
              (engine op: ``flash_attn_with_kvcache``)

Both kernels are faithful adaptations of the upstream TileLang examples
(``examples/flash_attention/example_gqa_fwd_varlen.py`` and
``examples/flash_decoding/example_gqa_decode.py``), parameterized by softmax
scale and dtype so they match the engine's tensor contract. Keeping prefill and
decode as separate ``@T.prim_func`` builders is deliberate: each lowers to its
own TIR + host/device codegen and can be retargeted/benchmarked on different
hardware (e.g. GPU vs vector processor) on its own.

Do NOT add ``from __future__ import annotations`` to this file — it breaks
TileLang's ``@T.prim_func`` type parsing.
"""

import os

import torch
import tilelang
import tilelang.language as T
from tilelang import jit
from tilelang import tvm as tvm

# exp(x) = exp2(x * log2(e)); folding the constant into the scale lets the
# kernel use a single fused-multiply for the softmax exponent.
LOG2E = 1.44269504

# Cephes single-precision 2**x polynomial (coeffs P0..P5, Horner) for the
# argument-reduced fractional part r in [-0.5, 0.5]. Accuracy ~1 ulp.
_EXP2_POLY = (
    1.535336188319500e-4,
    1.339887440266574e-3,
    9.618437357674640e-3,
    5.550332471162809e-2,
    2.402264791363012e-1,
    6.931472028550421e-1,
)


def _exp2_poly(x):
    """Vectorizable software ``2**x`` for the softmax exponent (Phase 2).

    ``T.exp2`` lowers to a scalar ``call exp2f`` on the CPU/RVV path, which
    prevents LLVM from vectorizing the softmax loops. This inlines the Cephes
    ``exp2f`` polynomial in pure arithmetic + an integer reinterpret so the whole
    expression vectorizes (``vfcvt`` / ``vfmadd`` / ``vsll``) instead of calling
    libm per element. Argument reduction ``2**x = 2**n * 2**r`` with
    ``n = round(x)`` and ``r in [-0.5, 0.5]``; ``2**n`` is built directly in the
    fp32 exponent field. Softmax exponents are ``<= 0`` (post max-subtract); the
    clamp flushes ``2**x`` to 0 for very negative / masked (``-inf``) inputs,
    matching ``exp2``. Used only on the vector path; CUDA keeps hardware ``exp2``.
    """
    x = T.max(x, T.float32(-127.0))
    n = T.floor(x + T.float32(0.5))
    r = x - n
    q = T.float32(_EXP2_POLY[0])
    for coeff in _EXP2_POLY[1:]:
        q = q * r + T.float32(coeff)
    two_r = q * r + T.float32(1.0)
    pow2n = T.reinterpret(T.shift_left(T.Cast("int32", n) + 127, 23), "float32")
    return two_r * pow2n

# Pass configs shared by both attention kernels.
#
# Why TL_DISABLE_WARP_SPECIALIZED:
#   On TMA-capable GPUs (Hopper sm_90 / Blackwell sm_100+/sm_120), TileLang's
#   ProducerConsumerWarpSpecialized pass routes the producer warpgroup's
#   global->shared Q/K/V loads through TMA bulk copies (see
#   ClassifyWarpSpecializedProducerCopy in src/cuda/op/copy_analysis.cc — the
#   TMA *load* path is gated ONLY by warp specialization). Each TMA copy needs a
#   CUtensorMap descriptor that the host runtime must place at a 64-byte-aligned
#   address. On this toolchain the descriptor is stack-allocated only 8-byte
#   aligned, so cuTensorMap setup aborts at launch with
#   "tensorMap address must be 64-byte aligned ... mod64=8". Disabling warp
#   specialization makes those loads fall back to cp.async/synchronous copies
#   (no descriptor), which is exactly what the upstream
#   examples/flash_decoding/example_gqa_decode.py does.
#
# Why TL_DISABLE_TMA_LOWER:
#   Belt-and-suspenders: the non-warp-specialized lowering path can still pick a
#   TMA *store* for the shared->global output copy, which would hit the same
#   misaligned-descriptor bug. This flag keeps plain T.copy() off the TMA store
#   path. It mirrors TileLang's own tilelang/utils/sparse.py, which pairs both
#   flags to opt out of TMA entirely. (The flag is deprecated but still honored
#   in this build; per-copy disable_tma=True is the long-term replacement.)
#
# Trade-off: warp specialization / TMA are throughput optimizations, so this is
# slightly slower on TMA GPUs but portable and correct. It does not change
# numerics. Revisit if/when the TileLang host-descriptor alignment bug is fixed.
_ATTENTION_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


def tilelang_dtype(torch_dtype: torch.dtype) -> str:
    if torch_dtype == torch.float16:
        return "float16"
    if torch_dtype == torch.float32:
        return "float32"
    raise ValueError(
        f"TileLang attention supports float16/float32, got {torch_dtype}. "
        "float16 is the validated path (tensor-core FlashAttention); cast q/k/v "
        "before calling the run_* helpers."
    )


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


# --------------------------------------------------------------------------- #
# Prefill: variable-length causal GQA forward (flash_attn_varlen_func)
# --------------------------------------------------------------------------- #
@jit(
    out_idx=[6],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_flash_attention_prefill_kernel(
    batch_size: int,
    total_q: int,
    total_kv: int,
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
    """Packed (unpadded) varlen FlashAttention, right-aligned causal masking.

    Tensor contract (matches model_runner.prepare_prefill + qwen3 q/k/v views):
      Q_unpad : (total_q,  num_heads,    head_dim)
      K_unpad : (total_kv, num_kv_heads, head_dim)
      V_unpad : (total_kv, num_kv_heads, head_dim)
      cu_seqlens_q / cu_seqlens_k : (batch_size + 1,) int32 prefix sums
      max_seqlen_q : runtime scalar
      -> Output_unpad : (total_q, num_heads, head_dim)

    ``vec_exp2`` (vector/RVV path, off for CUDA) uses the inlined polynomial
    ``2**x`` so softmax vectorizes instead of calling scalar ``exp2f`` (Phase 2).
    """
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    group_size = num_heads // num_kv_heads
    scale = softmax_scale * LOG2E
    accum_dtype = "float32"

    def _exp2(v):
        return _exp2_poly(v) if vec_exp2 else T.exp2(v)

    # Skip the acc_s_cast buffer + copy when acc (fp32) already is in_dtype
    # (fp32 vector path); fp16 (CUDA tensor-core) still needs the cast (Phase 3).
    need_cast = in_dtype != accum_dtype

    q_shape = [total_q, num_heads, head_dim]
    kv_shape = [total_kv, num_kv_heads, head_dim]
    o_shape = [total_q, num_heads, head_dim]

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, in_dtype),
        K_unpad: T.Tensor(kv_shape, in_dtype),
        V_unpad: T.Tensor(kv_shape, in_dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], "int32"),
        cu_seqlens_k: T.Tensor([batch_size + 1], "int32"),
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
            kv_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = k_end_idx - kv_start_idx

            T.copy(
                Q_unpad[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :],
                Q_shared,
            )

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # Right-align the causal mask so prefix-cache prefill (kv longer than
            # q) keeps the newest query tokens attending to the full prefix.
            offset = kv_current_seqlen - q_current_seqlen
            max_visible_k_idx = offset + (bx + 1) * block_M
            loop_range = (
                T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
                if is_causal
                else T.ceildiv(kv_current_seqlen, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    K_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :],
                    K_shared,
                )

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

                T.copy(
                    V_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :],
                    V_shared,
                )

                T.gemm(acc_s_cast if need_cast else acc_s, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, head_dim):
                # Queries that can see nothing (right-aligned offset) emit zeros.
                acc_o[i, j] = (
                    0 if is_causal and bx * block_M + i + offset < 0 else acc_o[i, j] / logsum[i]
                )

            T.copy(acc_o, O_shared)
            for i, d in T.Parallel(block_M, head_dim):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    return main


# --------------------------------------------------------------------------- #
# Decode: single-query GQA over a (dense, padded) KV cache (flash_attn_with_kvcache)
# --------------------------------------------------------------------------- #
@jit(
    out_idx=[4],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_flash_attention_decode_kernel(
    batch_size: int,
    seqlen_kv: int,
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
    """One query token per sequence attending to ``seqlen_kv`` cached keys.

    Tensor contract (matches model_runner.prepare_decode):
      Q    : (batch_size, num_heads,    head_dim)
      K    : (batch_size, seqlen_kv, num_kv_heads, head_dim)
      V    : (batch_size, seqlen_kv, num_kv_heads, head_dim)
      mask : (batch_size, seqlen_kv, num_kv_heads) uint8 (1 = valid kv position)
      -> Output : (batch_size, num_heads, head_dim)

    ``seqlen_kv`` is the padded cache length (multiple of ``block_N``); the mask
    encodes each sequence's real ``context_len``. GQA query heads sharing a kv
    head are batched into the ``block_H`` (M) dimension.

    ``vec_exp2`` (vector/RVV path, off for CUDA) uses the inlined polynomial
    ``2**x`` so softmax vectorizes instead of calling scalar ``exp2f`` (Phase 2).
    """
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    scale = softmax_scale * LOG2E
    accum_dtype = "float32"
    kv_group_num = num_heads // num_kv_heads
    valid_block_H = min(block_H, kv_group_num)

    def _exp2(v):
        return _exp2_poly(v) if vec_exp2 else T.exp2(v)

    # The P @ V input must match the GEMM's in_dtype. When acc (fp32) already is
    # in_dtype (fp32 vector path), skip the separate acc_s_cast buffer + copy
    # and feed acc_s straight in (Phase 3); fp16 (CUDA) still needs the cast.
    need_cast = in_dtype != accum_dtype

    shape_q = [batch_size, num_heads, head_dim]
    shape_kv = [batch_size, seqlen_kv, num_kv_heads, head_dim]
    shape_mask = [batch_size, seqlen_kv, num_kv_heads]
    shape_o = [batch_size, num_heads, head_dim]

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, in_dtype),
        K: T.Tensor(shape_kv, in_dtype),
        V: T.Tensor(shape_kv, in_dtype),
        mask: T.Tensor(shape_mask, "uint8"),
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
            mask_local = T.alloc_fragment([block_N], "uint8")
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
                T.copy(K[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_shared)
                T.copy(mask[bid, k * block_N : (k + 1) * block_N, cur_kv_head], mask_local)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        mask_local[j] != 0, acc_s[i, j], -T.infinity(accum_dtype)
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
                T.copy(V[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], V_shared)
                T.gemm(acc_s_cast if need_cast else acc_s, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_H, head_dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

    return main


# --------------------------------------------------------------------------- #
# Run helpers (compile + execute)
# --------------------------------------------------------------------------- #
from nanovllm.backends.tilelang.runtime import compile_tilelang_kernel


def _compile_attention_kernel(
    builder,
    build_args: tuple,
    out_idx: int,
    backend: str,
    *,
    kernel_name: str | None = None,
):
    """Compile ``builder`` for ``backend`` and return a callable kernel."""
    return compile_tilelang_kernel(
        builder,
        build_args,
        out_idx,
        backend,
        kernel_name=kernel_name or getattr(builder, "__name__", "attention"),
    )


def run_tilelang_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    is_causal: bool = True,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    backend: str = "cuda",
) -> torch.Tensor:
    """Run the prefill kernel on packed varlen q/k/v, returning (total_q, H, D).

    ``backend`` selects CUDA JIT (engine path) or host ``llvm`` (RVV golden).
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"q/k/v must be 3D (tokens, heads, dim); got {tuple(q.shape)}, "
            f"{tuple(k.shape)}, {tuple(v.shape)}"
        )

    total_q, num_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}."
        )
    batch_size = cu_seqlens_q.numel() - 1
    in_dtype = tilelang_dtype(q.dtype)

    # Pad packed lengths up to the block size so the final per-sequence block
    # never reads past the allocation. Padding rows belong to no sequence range
    # and are excluded by cu_seqlens, so results are unaffected.
    padded_q = _round_up(total_q, block_M)
    padded_kv = _round_up(total_kv, block_N)

    q_pad = q
    k_pad = k
    v_pad = v
    if padded_q != total_q:
        q_pad = torch.zeros(padded_q, num_heads, head_dim, device=q.device, dtype=q.dtype)
        q_pad[:total_q] = q
    if padded_kv != total_kv:
        k_pad = torch.zeros(padded_kv, num_kv_heads, head_dim, device=k.device, dtype=k.dtype)
        v_pad = torch.zeros(padded_kv, num_kv_heads, head_dim, device=v.device, dtype=v.dtype)
        k_pad[:total_kv] = k
        v_pad[:total_kv] = v

    # Vector-friendly software exp2 on the CPU/RVV path; CUDA keeps hardware exp2.
    build_args = (
        batch_size, padded_q, padded_kv, num_heads, num_kv_heads, head_dim,
        float(softmax_scale), is_causal, block_M, block_N, num_stages, threads, in_dtype,
        backend != "cuda",
    )
    kernel = _compile_attention_kernel(
        build_flash_attention_prefill_kernel, build_args, 6, backend, kernel_name="prefill"
    )
    out = kernel(
        q_pad.contiguous(),
        k_pad.contiguous(),
        v_pad.contiguous(),
        cu_seqlens_q.to(torch.int32).contiguous(),
        cu_seqlens_k.to(torch.int32).contiguous(),
        int(max_seqlen_q),
    )
    return out[:total_q]


def run_tilelang_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float,
    block_N: int = 128,
    block_H: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    backend: str = "cuda",
) -> torch.Tensor:
    """Run the decode kernel on dense padded KV, returning (batch, H, D).

    ``backend`` selects CUDA JIT (engine path) or host ``llvm`` (RVV golden).
    """
    if q.dim() != 3:
        raise ValueError(f"q must be 3D (batch, heads, dim); got {tuple(q.shape)}")
    if k_cache.dim() != 4:
        raise ValueError(
            f"k_cache must be 4D (batch, seqlen_kv, kv_heads, dim); got {tuple(k_cache.shape)}"
        )

    batch_size, num_heads, head_dim = q.shape
    _, seqlen_kv, num_kv_heads, _ = k_cache.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}."
        )
    if seqlen_kv % block_N != 0:
        raise ValueError(
            f"seqlen_kv={seqlen_kv} must be a multiple of block_N={block_N}; "
            "pad the KV cache before calling run_tilelang_attention_decode()."
        )
    in_dtype = tilelang_dtype(q.dtype)

    # Vector-friendly software exp2 on the CPU/RVV path; CUDA keeps hardware exp2.
    build_args = (
        batch_size, seqlen_kv, num_heads, num_kv_heads, head_dim,
        float(softmax_scale), block_N, block_H, num_stages, threads, in_dtype,
        backend != "cuda",
    )
    kernel = _compile_attention_kernel(
        build_flash_attention_decode_kernel, build_args, 4, backend, kernel_name="decode"
    )
    return kernel(
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        mask.to(torch.uint8).contiguous(),
    )


# --------------------------------------------------------------------------- #
# Source TIR dumps
# --------------------------------------------------------------------------- #
def _dump_tir(tir, dump_path: str, banner: str) -> str:
    tir_text = tir.script()
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


def dump_prefill_tensorir(
    batch_size: int,
    total_q: int,
    total_kv: int,
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
    dump_path: str = "",
) -> str:
    tir = build_flash_attention_prefill_kernel.get_tir(
        batch_size,
        total_q,
        total_kv,
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
    )
    return _dump_tir(tir, dump_path, "TensorIR — FlashAttention prefill (varlen causal GQA)")


def dump_decode_tensorir(
    batch_size: int,
    seqlen_kv: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: float,
    block_N: int = 128,
    block_H: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    dump_path: str = "",
) -> str:
    tir = build_flash_attention_decode_kernel.get_tir(
        batch_size,
        seqlen_kv,
        num_heads,
        num_kv_heads,
        head_dim,
        float(softmax_scale),
        block_N,
        block_H,
        num_stages,
        threads,
        in_dtype,
    )
    return _dump_tir(tir, dump_path, "TensorIR — FlashAttention decode (GQA KV-cache)")
