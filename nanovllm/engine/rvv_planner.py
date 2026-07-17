"""
Pre-compile TileLang kernels for an RVV engine session.

Used when ``device="rvv"`` and ``rvv_compile_only=True`` so ``ModelRunner`` can
emit the full decode-stage kernel set without executing on a non-RISC-V host.
Follows the RVV lowering flow in ``usage.md`` (``lower_kernel_rvv`` →
``.tir`` / ``.ll`` / ``.s`` under ``rvv_build_dir``).
"""

from __future__ import annotations

import json
import os

import torch

from nanovllm.backends.tilelang.activation import build_silu_mul_kernel
from nanovllm.backends.tilelang.attention import tilelang_dtype
from nanovllm.backends.tilelang.embedding import build_embedding_kernel
from nanovllm.backends.tilelang.kv_store import build_kv_store_kernel
from nanovllm.backends.tilelang.linear import build_linear_kernel
from nanovllm.backends.tilelang.paged_decode import build_flash_attention_decode_paged_kernel
from nanovllm.backends.tilelang.paged_prefill import build_flash_attention_prefill_paged_kernel
from nanovllm.backends.tilelang.rmsnorm import build_rmsnorm_kernel
from nanovllm.backends.tilelang.rope import build_rope_kernel
from nanovllm.backends.tilelang.rvv_lower import lower_kernel_rvv, rvv_target
from nanovllm.config import Config


def compile_engine_rvv_kernels(config: Config) -> list[str]:
    """Lower representative engine kernels for the configured model.

    Returns kernel names that were lowered. Does not execute kernels.
    """
    hf = config.hf_config
    hidden = hf.hidden_size
    num_heads = hf.num_attention_heads
    num_kv_heads = hf.num_key_value_heads
    head_dim = getattr(hf, "head_dim", hidden // num_heads)
    intermediate = hf.intermediate_size
    vocab = hf.vocab_size
    block_size = config.kvcache_block_size
    max_seqs = min(config.max_num_seqs, 8)
    max_blocks = (config.max_model_len + block_size - 1) // block_size
    dtype = tilelang_dtype(torch.float32)
    target = rvv_target(nr_lanes=config.rvv_nr_lanes)
    compiled: list[str] = []

    def _lower(builder, build_args: tuple, name: str) -> None:
        build_dir = os.path.join(config.rvv_build_dir, name)
        tir = builder.get_tir(*build_args)
        lower_kernel_rvv(
            tir,
            name=name,
            build_dir=build_dir,
            target=target,
            title=name,
            metadata={"build_args": [repr(a) for a in build_args]},
        )
        compiled.append(name)

    _lower(
        build_embedding_kernel,
        (max_seqs, hidden, vocab, 1, 256, dtype),
        "embedding",
    )
    _lower(
        build_linear_kernel,
        (max_seqs, hidden, vocab, False, 64, 64, 64, 128, dtype),
        "lm_head",
    )

    qkv_out = (num_heads + 2 * num_kv_heads) * head_dim
    _lower(
        build_linear_kernel,
        (max_seqs, hidden, qkv_out, False, 64, 64, 64, 128, dtype),
        "qkv_proj",
    )
    _lower(
        build_linear_kernel,
        (max_seqs, num_heads * head_dim, hidden, False, 64, 64, 64, 128, dtype),
        "o_proj",
    )
    _lower(
        build_linear_kernel,
        (max_seqs, hidden, 2 * intermediate, False, 64, 64, 64, 128, dtype),
        "gate_up_proj",
    )
    _lower(
        build_linear_kernel,
        (max_seqs, intermediate, hidden, False, 64, 64, 64, 128, dtype),
        "down_proj",
    )

    _lower(
        build_rmsnorm_kernel,
        (max_seqs, hidden, 1e-6, False, 1, 128, dtype),
        "rmsnorm",
    )
    _lower(
        build_rmsnorm_kernel,
        (max_seqs, hidden, 1e-6, True, 1, 128, dtype),
        "add_rmsnorm",
    )
    _lower(
        build_silu_mul_kernel,
        (max_seqs, intermediate, 64, 128, dtype),
        "silu_mul",
    )
    _lower(
        build_rope_kernel,
        (max_seqs, num_heads, num_kv_heads, head_dim, 64, 128, dtype),
        "rope",
    )

    num_slots = config.num_kvcache_blocks * block_size
    _lower(
        build_kv_store_kernel,
        (max_seqs, num_kv_heads, head_dim, num_slots, 128, dtype),
        "kv_store",
    )

    block_n = min(128, block_size)
    block_h = max(1, num_heads // num_kv_heads)
    assert block_size % block_n == 0
    _lower(
        build_flash_attention_decode_paged_kernel,
        (
            max_seqs,
            config.num_kvcache_blocks,
            block_size,
            max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim ** -0.5,
            block_n,
            block_h,
            2,
            128,
            dtype,
            True,
        ),
        "paged_decode",
    )

    total_q = max_seqs
    _lower(
        build_flash_attention_prefill_paged_kernel,
        (
            max_seqs,
            total_q,
            config.num_kvcache_blocks,
            block_size,
            max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim ** -0.5,
            True,
            64,
            block_n,
            1,
            128,
            dtype,
            True,
        ),
        "paged_prefill",
    )

    os.makedirs(config.rvv_build_dir, exist_ok=True)
    manifest_path = os.path.join(config.rvv_build_dir, "engine_kernels.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "kernels": compiled,
                "model": config.model,
                "nr_lanes": config.rvv_nr_lanes,
                "build_dir": config.rvv_build_dir,
            },
            handle,
            indent=2,
        )
    return compiled
