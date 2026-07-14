"""
Numeric validation for the paged FlashAttention decode kernel.

Compiles ``build_flash_attention_decode_paged_kernel`` on host ``llvm`` and
compares against ``paged_decode_reference`` (PyTorch float32 golden over a
synthetic non-contiguous block pool).

Requires the Path B TileLang dev build on ``PYTHONPATH`` (see ``usage.md``).
Skips cleanly if TileLang / an LLVM-enabled TVM is unavailable.

Run::

    pytest tests/test_paged_decode_numeric.py -v
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from tilelang import tvm as _tvm  # noqa: E402

if not _tvm.runtime.enabled("llvm"):
    pytest.skip("TVM build has no LLVM target; RVV host path unavailable.", allow_module_level=True)

from nanovllm.backends.tilelang.paged_decode import (  # noqa: E402
    build_paged_decode_inputs,
    paged_decode_reference,
    run_tilelang_flash_attn_with_kvcache,
)

ATOL = 1e-2
RTOL = 1e-2

# Engine default block_size; must be a multiple of decode block_N (128 @ 4 lanes).
BLOCK_SIZE = 256
BLOCK_N = 128

HEAD_CONFIGS = [
    (8, 8, 64),
    (8, 2, 64),
    (8, 4, 64),
    (16, 8, 128),
    (16, 2, 64),
]

CONTEXT_LENS = [
    [64],
    [128],
    [64, 128],
    [1, 200],
]


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", HEAD_CONFIGS)
@pytest.mark.parametrize("context_lens", CONTEXT_LENS)
def test_paged_decode_host_llvm(num_heads, num_kv_heads, head_dim, context_lens):
    inp = build_paged_decode_inputs(
        context_lens,
        block_size=BLOCK_SIZE,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        seed=0,
    )
    # Non-contiguous physical ids (builder assigns from the end of the free list).
    assert inp.block_table.max().item() >= 0
    live = inp.block_table[inp.block_table >= 0]
    if live.numel() > 1:
        # At least one sequence should see decreasing/non-identity physical order.
        assert not torch.equal(live, torch.arange(live.numel(), dtype=live.dtype))

    scale = head_dim ** -0.5
    reference = paged_decode_reference(
        inp.q,
        inp.k_cache,
        inp.v_cache,
        inp.block_table,
        inp.cache_seqlens,
        inp.block_size,
        softmax_scale=scale,
    )
    # flash_attn-compatible 4D q
    q4 = inp.q.unsqueeze(1)
    out4 = run_tilelang_flash_attn_with_kvcache(
        q4,
        inp.k_cache,
        inp.v_cache,
        inp.cache_seqlens,
        inp.block_table,
        softmax_scale=scale,
        backend="cpu",
        block_N=BLOCK_N,
        block_H=num_heads // num_kv_heads,
    )
    assert out4.shape == q4.shape
    torch.testing.assert_close(
        out4.squeeze(1).float(),
        reference.float(),
        atol=ATOL,
        rtol=RTOL,
    )


def test_paged_decode_golden_matches_dense_gather():
    """Builder + reference must match a dense gather recompute to 1e-6."""
    inp = build_paged_decode_inputs(
        [64, 128],
        block_size=BLOCK_SIZE,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.float32,
        device="cpu",
        seed=1,
    )
    scale = 64 ** -0.5
    ref = paged_decode_reference(
        inp.q,
        inp.k_cache,
        inp.v_cache,
        inp.block_table,
        inp.cache_seqlens,
        inp.block_size,
        softmax_scale=scale,
    )
    from nanovllm.backends.tilelang.paged_decode import gather_dense_kv

    k_d, v_d = gather_dense_kv(
        inp.k_cache, inp.v_cache, inp.block_table, inp.cache_seqlens, inp.block_size
    )
    out = torch.empty_like(inp.q)
    qf, kf, vf = inp.q.float(), k_d.float(), v_d.float()
    group = 8 // 2
    for b, length in enumerate(inp.context_lens):
        q_b = qf[b]
        k_b = kf[b, :length].repeat_interleave(group, dim=1)
        v_b = vf[b, :length].repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,jhd->hj", q_b, k_b) * scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hj,jhd->hd", probs, v_b)
    assert (ref - out).abs().max().item() < 1e-6
