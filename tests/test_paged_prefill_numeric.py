"""
Numeric validation for paged FlashAttention prefill (prefix-cache path).

Compiles ``build_flash_attention_prefill_paged_kernel`` on host ``llvm`` and
compares against ``paged_prefill_reference``.

Run::

    pytest tests/test_paged_prefill_numeric.py -v
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from tilelang import tvm as _tvm  # noqa: E402

if not _tvm.runtime.enabled("llvm"):
    pytest.skip("TVM build has no LLVM target.", allow_module_level=True)

from nanovllm.backends.tilelang.paged_prefill import (  # noqa: E402
    PAGED_BLOCK_SIZE,
    PREFILL_BLOCK_N,
    build_paged_prefill_inputs,
    flatten_prefill_specs,
    paged_prefill_reference,
    run_tilelang_flash_attn_varlen,
)

ATOL = 1e-2
RTOL = 1e-2

HEAD_CONFIGS = [
    (8, 8, 64),
    (8, 2, 64),
    (16, 8, 128),
]

# (cached, new_q) per sequence
PREFILL_PAGED_SPECS = [
    [(0, 64)],
    [(64, 16)],
    [(32, 48), (128, 32)],
    [(200, 48)],
]


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", HEAD_CONFIGS)
@pytest.mark.parametrize("specs", PREFILL_PAGED_SPECS)
def test_paged_prefill_host_llvm(num_heads, num_kv_heads, head_dim, specs):
    inp = build_paged_prefill_inputs(
        specs,
        block_size=PAGED_BLOCK_SIZE,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        seed=0,
    )
    scale = head_dim ** -0.5
    reference = paged_prefill_reference(
        inp.q,
        inp.k_cache,
        inp.v_cache,
        inp.block_table,
        inp.cu_seqlens_q,
        inp.cu_seqlens_k,
        softmax_scale=scale,
        block_size=inp.block_size,
    )
    out = run_tilelang_flash_attn_varlen(
        inp.q,
        inp.k_cache,
        inp.v_cache,
        inp.cu_seqlens_q,
        inp.cu_seqlens_k,
        inp.max_seqlen_q,
        softmax_scale=scale,
        causal=True,
        block_table=inp.block_table,
        backend="cpu",
        block_N=PREFILL_BLOCK_N,
    )
    torch.testing.assert_close(
        out.float(),
        reference.float(),
        atol=ATOL,
        rtol=RTOL,
    )


def test_paged_prefill_golden_matches_dense_recompute():
    specs = [(64, 32), (128, 16)]
    inp = build_paged_prefill_inputs(
        specs,
        block_size=PAGED_BLOCK_SIZE,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.float32,
        device="cpu",
        seed=1,
    )
    scale = 64 ** -0.5
    ref = paged_prefill_reference(
        inp.q,
        inp.k_cache,
        inp.v_cache,
        inp.block_table,
        inp.cu_seqlens_q,
        inp.cu_seqlens_k,
        softmax_scale=scale,
        block_size=inp.block_size,
    )
    # Dense recompute via gathered K/V per sequence.
    cu_q = inp.cu_seqlens_q.tolist()
    cu_k = inp.cu_seqlens_k.tolist()
    from nanovllm.backends.tilelang.paged_prefill import gather_seq_kv

    out = torch.empty_like(inp.q)
    qf = inp.q.float()
    group = 8 // 2
    for b in range(len(specs)):
        qs, qe = cu_q[b], cu_q[b + 1]
        lk = cu_k[b + 1] - cu_k[b]
        lq = qe - qs
        k_dense, v_dense = gather_seq_kv(
            inp.k_cache, inp.v_cache, inp.block_table, b, lk, inp.block_size
        )
        k_b = k_dense.float().repeat_interleave(group, dim=1).transpose(0, 1)
        v_b = v_dense.float().repeat_interleave(group, dim=1).transpose(0, 1)
        q_b = qf[qs:qe].transpose(0, 1)
        scores = torch.einsum("hid,hjd->hij", q_b, k_b) * scale
        offset = lk - lq
        i_idx = torch.arange(lq).view(1, lq, 1)
        j_idx = torch.arange(lk).view(1, 1, lk)
        causal = j_idx <= (i_idx + offset)
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out_b = torch.einsum("hij,hjd->hid", probs, v_b).transpose(0, 1)
        out[qs:qe] = out_b
    assert (ref - out).abs().max().item() < 1e-6


def test_flatten_specs_roundtrip():
    specs = [(0, 64), (32, 48)]
    flat = flatten_prefill_specs(specs)
    assert flat == [0, 64, 32, 48]
