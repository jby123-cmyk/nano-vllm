"""Enumerate the RVV attention matrix cases (dedup by RVV JIT signature).

Lists mirror ``demos/rvv/matrix_report.py`` / ``tests/test_attention_rvv_numeric.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanovllm.backends.spike.config import DEFAULT_NR_LANES
from nanovllm.backends.spike.emit_ir import build_attention_tir
from nanovllm.backends.spike.golden import make_case_id

# Same matrix as demos/rvv/matrix_report.py
HEAD_CONFIGS = [
    (8, 8, 64),
    (8, 2, 64),
    (8, 4, 64),
    (16, 8, 128),
    (16, 2, 64),
]

DECODE_CONTEXT_LENS = [
    ([64], "single seq, sub-block KV (pad to block_N)"),
    ([128], "single seq, exactly one KV block"),
    ([64, 128], "ragged batch (2 seqs)"),
    ([1, 200], "min len + spans two KV blocks"),
]

PREFILL_SEQ_LENS = [
    ([64], "one block aligned"),
    ([80], "non block-aligned (pad to block_M)"),
    ([64, 64], "two equal sequences"),
    ([48, 96], "ragged, both non-aligned"),
]

# Flat (cached, new_q) pairs per sequence — see ``unflatten_prefill_specs``.
PREFILL_PAGED_SPECS = [
    ([0, 64], "cold prefill (no prefix)"),
    ([64, 16], "64-token prefix + 16 new"),
    ([32, 48, 128, 32], "ragged prefix batch"),
    ([200, 48], "multi-block prefix + new"),
]

# Linear GEMM shapes as [M, N, K] = [tokens, out_features, in_features].
LINEAR_SHAPES = [
    ([1, 256, 128], "decode-sized small GEMM"),
    ([1, 2048, 1024], "qkv-ish decode M=1"),
    ([64, 256, 128], "prefill tile M=64"),
]

# Layer-op Spike smokes (lengths encoding is phase-specific).
RMSNORM_SHAPES = [
    ([8, 128, 0], "plain RMSNorm"),
    ([8, 128, 1], "fused residual RMSNorm"),
]
SILU_SHAPES = [
    ([8, 256], "SiluAndMul"),
]
EMBED_SHAPES = [
    ([8, 64, 128], "embedding gather"),
]
ROPE_SHAPES = [
    ([8, 4, 2, 64], "decode-step RoPE (tokens, heads, kv_heads, head_dim)"),
]
KV_STORE_SHAPES = [
    ([8, 2, 64, 512], "paged KV scatter (tokens, kv_heads, head_dim, num_slots)"),
]


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    phase: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lengths: tuple[int, ...]
    coverage_note: str
    nr_lanes: int
    signature: tuple


def enumerate_cases(
    nr_lanes: int = DEFAULT_NR_LANES,
    *,
    head_configs: list[tuple[int, int, int]] | None = None,
    include_decode: bool = True,
    include_decode_paged: bool = True,
    include_prefill: bool = True,
    include_prefill_paged: bool = True,
    include_linear: bool = True,
    include_layer_ops: bool = True,
) -> list[MatrixCase]:
    """Return unique RVV-signature cases (attention + linear + layer ops)."""
    head_configs = head_configs or HEAD_CONFIGS
    seen: set[tuple] = set()
    cases: list[MatrixCase] = []

    def _add(phase: str, lengths: list[int], note: str, num_heads, num_kv_heads, head_dim):
        _tir, sig = build_attention_tir(
            phase,
            list(lengths),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            nr_lanes=nr_lanes,
        )
        rvv_sig = ("rvv", nr_lanes) + sig
        if rvv_sig in seen:
            return
        seen.add(rvv_sig)
        case_id = make_case_id(
            phase, num_heads, num_kv_heads, head_dim, list(lengths), nr_lanes
        )
        cases.append(
            MatrixCase(
                case_id=case_id,
                phase=phase,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                lengths=tuple(lengths),
                coverage_note=note,
                nr_lanes=nr_lanes,
                signature=rvv_sig,
            )
        )

    if include_decode:
        for num_heads, num_kv_heads, head_dim in head_configs:
            for lengths, note in DECODE_CONTEXT_LENS:
                _add("decode", lengths, note, num_heads, num_kv_heads, head_dim)

    if include_decode_paged:
        for num_heads, num_kv_heads, head_dim in head_configs:
            for lengths, note in DECODE_CONTEXT_LENS:
                _add(
                    "decode_paged",
                    lengths,
                    f"paged KV: {note}",
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )

    if include_prefill:
        for num_heads, num_kv_heads, head_dim in head_configs:
            for lengths, note in PREFILL_SEQ_LENS:
                _add("prefill", lengths, note, num_heads, num_kv_heads, head_dim)

    if include_prefill_paged:
        for num_heads, num_kv_heads, head_dim in head_configs:
            for lengths, note in PREFILL_PAGED_SPECS:
                _add(
                    "prefill_paged",
                    lengths,
                    f"paged prefix cache: {note}",
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )

    if include_linear:
        for lengths, note in LINEAR_SHAPES:
            _add("linear", lengths, f"linear GEMM: {note}", 1, 1, 64)

    if include_layer_ops:
        for lengths, note in RMSNORM_SHAPES:
            _add("rmsnorm", lengths, note, 1, 1, 64)
        for lengths, note in SILU_SHAPES:
            _add("silu_mul", lengths, note, 1, 1, 64)
        for lengths, note in EMBED_SHAPES:
            _add("embedding", lengths, note, 1, 1, 64)
        for lengths, note in ROPE_SHAPES:
            _add("rope", lengths, note, lengths[1], lengths[2], lengths[3])
        for lengths, note in KV_STORE_SHAPES:
            _add("kv_store", lengths, note, lengths[1], lengths[1], lengths[2])

    return cases
