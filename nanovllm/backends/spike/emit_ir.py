"""Emit per-case RVV ``kernel.ll`` via ``lower_kernel_rvv``."""

from __future__ import annotations

import os
from dataclasses import dataclass

from nanovllm.backends.spike.config import DEFAULT_NR_LANES
from nanovllm.backends.spike.golden import _rvv_stage, make_case_id
from nanovllm.backends.tilelang.attention import (
    build_flash_attention_decode_kernel,
    build_flash_attention_prefill_kernel,
    tilelang_dtype,
)
from nanovllm.backends.tilelang.paged_decode import (
    PAGED_BLOCK_SIZE,
    build_flash_attention_decode_paged_kernel,
    paged_pool_shapes,
)
from nanovllm.backends.tilelang.paged_prefill import (
    build_flash_attention_prefill_paged_kernel,
    prefill_paged_pool_shapes,
    unflatten_prefill_specs,
)
from nanovllm.backends.tilelang.rvv_lower import (
    assert_target_vlen,
    lower_kernel_rvv,
    rvv_target,
)
from nanovllm.stages.attention import _round_up


@dataclass
class EmitIRResult:
    case_id: str
    case_dir: str
    ll_path: str
    compiled: bool
    error: str | None = None


def build_attention_tir(
    phase: str,
    lengths: list[int],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    nr_lanes: int = DEFAULT_NR_LANES,
):
    """Return (tir, signature_tuple) matching ``matrix_report._build_rvv_tir``."""
    if phase == "linear":
        from nanovllm.backends.tilelang.linear import (
            DEFAULT_BLOCK_K,
            DEFAULT_BLOCK_M,
            DEFAULT_BLOCK_N,
            build_linear_kernel,
        )

        if len(lengths) != 3:
            raise ValueError(f"linear lengths must be [M, N, K], got {lengths}")
        m, n, k = lengths
        padded_m = _round_up(max(m, 1), DEFAULT_BLOCK_M)
        padded_n = _round_up(n, DEFAULT_BLOCK_N)
        padded_k = _round_up(k, DEFAULT_BLOCK_K)
        build_args = (
            padded_m,
            padded_k,
            padded_n,
            False,
            DEFAULT_BLOCK_M,
            DEFAULT_BLOCK_N,
            DEFAULT_BLOCK_K,
            128,
            "float32",
        )
        tir = build_linear_kernel.get_tir(*build_args)
        sig = ("linear",) + build_args
        return tir, sig

    if phase == "rmsnorm":
        from nanovllm.backends.tilelang.rmsnorm import build_rmsnorm_kernel

        # lengths = [tokens, hidden]
        tokens, hidden = lengths[0], lengths[1]
        fuse = bool(lengths[2]) if len(lengths) > 2 else False
        build_args = (tokens, hidden, 1e-6, fuse, 1, 128, "float32")
        tir = build_rmsnorm_kernel.get_tir(*build_args)
        return tir, ("rmsnorm",) + build_args

    if phase == "silu_mul":
        from nanovllm.backends.tilelang.activation import build_silu_mul_kernel

        # lengths = [tokens, intermediate]
        tokens, inter = lengths[0], lengths[1]
        build_args = (tokens, inter, 64, 128, "float32")
        tir = build_silu_mul_kernel.get_tir(*build_args)
        return tir, ("silu_mul",) + build_args

    if phase == "rope":
        from nanovllm.backends.tilelang.rope import build_rope_kernel

        # lengths = [tokens, heads, kv_heads, head_dim]
        tokens, nh, nkv, hd = lengths
        build_args = (tokens, nh, nkv, hd, 64, 128, "float32")
        tir = build_rope_kernel.get_tir(*build_args)
        return tir, ("rope",) + build_args

    if phase == "kv_store":
        from nanovllm.backends.tilelang.kv_store import build_kv_store_kernel

        # lengths = [tokens, kv_heads, head_dim, num_slots]
        tokens, nkv, hd, slots = lengths
        build_args = (tokens, nkv, hd, slots, 128, "float32")
        tir = build_kv_store_kernel.get_tir(*build_args)
        return tir, ("kv_store",) + build_args

    if phase == "embedding":
        from nanovllm.backends.tilelang.embedding import build_embedding_kernel

        # lengths = [tokens, hidden, vocab]
        tokens, hidden, vocab = lengths
        build_args = (tokens, hidden, vocab, 1, 256, "float32")
        tir = build_embedding_kernel.get_tir(*build_args)
        return tir, ("embedding",) + build_args

    stage = _rvv_stage(num_heads, num_kv_heads, head_dim, nr_lanes)
    if phase == "decode":
        ctx = stage.prepare_decode_context(lengths)
        build_args = (
            ctx.batch_size,
            ctx.seqlen_kv_padded,
            stage.num_heads,
            stage.num_kv_heads,
            stage.head_dim,
            float(stage.softmax_scale),
            stage.decode_block_N,
            stage.decode_block_H,
            stage.decode_num_stages,
            stage.decode_threads,
            tilelang_dtype(stage.dtype),
            True,
        )
        tir = build_flash_attention_decode_kernel.get_tir(*build_args)
        sig = ("decode",) + build_args
        return tir, sig

    if phase == "decode_paged":
        shapes = paged_pool_shapes(lengths, PAGED_BLOCK_SIZE)
        if PAGED_BLOCK_SIZE % stage.decode_block_N != 0:
            raise ValueError(
                f"PAGED_BLOCK_SIZE={PAGED_BLOCK_SIZE} must be a multiple of "
                f"decode_block_N={stage.decode_block_N}"
            )
        build_args = (
            shapes["batch_size"],
            shapes["num_blocks_pool"],
            PAGED_BLOCK_SIZE,
            shapes["max_num_blocks"],
            stage.num_heads,
            stage.num_kv_heads,
            stage.head_dim,
            float(stage.softmax_scale),
            stage.decode_block_N,
            stage.decode_block_H,
            stage.decode_num_stages,
            stage.decode_threads,
            tilelang_dtype(stage.dtype),
            True,
        )
        tir = build_flash_attention_decode_paged_kernel.get_tir(*build_args)
        sig = ("decode_paged",) + build_args
        return tir, sig

    if phase == "prefill_paged":
        specs = unflatten_prefill_specs(lengths)
        shapes = prefill_paged_pool_shapes(specs, PAGED_BLOCK_SIZE)
        if PAGED_BLOCK_SIZE % stage.prefill_block_N != 0:
            raise ValueError(
                f"PAGED_BLOCK_SIZE={PAGED_BLOCK_SIZE} must be a multiple of "
                f"prefill_block_N={stage.prefill_block_N}"
            )
        total_q = sum(new_q for _cached, new_q in specs)
        max_seqlen_q = max((new_q for _cached, new_q in specs), default=0)
        padded_q = _round_up(total_q, stage.prefill_block_M)
        padded_max_q = _round_up(max(max_seqlen_q, 1), stage.prefill_block_M)
        max_ctx = max(cached + new_q for cached, new_q in specs) if specs else 0
        need_blocks = max(1, (max_ctx + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE)
        compile_max_blocks = max(shapes["max_num_blocks"], need_blocks)
        build_args = (
            shapes["batch_size"],
            padded_q,
            shapes["num_blocks_pool"],
            PAGED_BLOCK_SIZE,
            compile_max_blocks,
            stage.num_heads,
            stage.num_kv_heads,
            stage.head_dim,
            float(stage.softmax_scale),
            True,
            stage.prefill_block_M,
            stage.prefill_block_N,
            stage.prefill_num_stages,
            stage.prefill_threads,
            tilelang_dtype(stage.dtype),
            True,
        )
        tir = build_flash_attention_prefill_paged_kernel.get_tir(*build_args)
        sig = ("prefill_paged",) + build_args
        return tir, sig

    if phase != "prefill":
        raise ValueError(phase)
    ctx = stage.prepare_prefill_context(lengths)
    padded_q = _round_up(ctx.total_q, stage.prefill_block_M)
    padded_kv = _round_up(ctx.total_kv, stage.prefill_block_N)
    build_args = (
        len(lengths),
        padded_q,
        padded_kv,
        stage.num_heads,
        stage.num_kv_heads,
        stage.head_dim,
        float(stage.softmax_scale),
        True,
        stage.prefill_block_M,
        stage.prefill_block_N,
        stage.prefill_num_stages,
        stage.prefill_threads,
        tilelang_dtype(stage.dtype),
        True,
    )
    tir = build_flash_attention_prefill_kernel.get_tir(*build_args)
    sig = ("prefill",) + build_args
    return tir, sig


def emit_kernel_ll(
    phase: str,
    lengths: list[int],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    nr_lanes: int = DEFAULT_NR_LANES,
    out_root: str = "demos/build_rvv/spike",
    case_id: str | None = None,
) -> EmitIRResult:
    case_id = case_id or make_case_id(
        phase, num_heads, num_kv_heads, head_dim, lengths, nr_lanes
    )
    case_dir = os.path.join(out_root, case_id)
    os.makedirs(case_dir, exist_ok=True)

    target = rvv_target(nr_lanes=nr_lanes)
    assert_target_vlen(target, nr_lanes)
    tir, _sig = build_attention_tir(
        phase,
        lengths,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        nr_lanes=nr_lanes,
    )
    result = lower_kernel_rvv(
        tir,
        "kernel",
        case_dir,
        target,
        title=f"Spike matrix {case_id}",
        metadata={
            "case_id": case_id,
            "phase": phase,
            "lengths": lengths,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "nr_lanes": nr_lanes,
        },
    )
    ll_path = os.path.join(case_dir, "kernel.ll")
    # lower_kernel_rvv names files kernel.ll already when name="kernel".
    if result.ll_path and result.ll_path != ll_path and os.path.isfile(result.ll_path):
        os.replace(result.ll_path, ll_path)
    return EmitIRResult(
        case_id=case_id,
        case_dir=case_dir,
        ll_path=ll_path,
        compiled=result.compiled,
        error=result.error,
    )
