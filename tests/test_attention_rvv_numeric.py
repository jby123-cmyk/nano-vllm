"""
Numeric validation for the CPU/RVV attention lowering path across many shapes.

Each case compiles the SAME ``@T.prim_func`` FlashAttention kernels used for the
RVV artifacts on the host ``llvm`` backend (``AttentionStage(tilelang_backend=
"cpu")``) and compares against the pure-PyTorch float32 golden.  This is the
runnable proxy for the riscv64 RVV ``.s`` artifacts, which are not host-runnable.

Requires the Path B TileLang dev build on ``PYTHONPATH`` (see ``usage.md``).
Skips cleanly if TileLang / an LLVM-enabled TVM is unavailable.

Run:

    pytest tests/test_attention_rvv_numeric.py -v

Lab-meeting table (max_abs_diff, compile/exec timing, coverage notes)::

    python demos/matrix_report.py --out-dir demos/build_rvv --slide
    python demos/matrix_report.py --slide-only --out-dir demos/build_rvv
    python demos/matrix_report.py --rvv-compile-only --out-dir demos/build_rvv

The matrix is intentionally diverse but bounded (compilation dominates runtime;
TileLang caches per shape).  fp32 host tolerance is loose vs the ~1e-6 observed
diffs so the assertions track real regressions, not float noise.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from tilelang import tvm as _tvm  # noqa: E402

if not _tvm.runtime.enabled("llvm"):
    pytest.skip("TVM build has no LLVM target; RVV host path unavailable.", allow_module_level=True)

from nanovllm.stages.attention import AttentionStage  # noqa: E402

ATOL = 1e-2
RTOL = 1e-2

# (num_heads, num_kv_heads, head_dim) — cover MHA, GQA group 2/4/8, and 2 head dims.
HEAD_CONFIGS = [
    (8, 8, 64),    # MHA (group size 1)
    (8, 2, 64),    # GQA group 4
    (8, 4, 64),    # GQA group 2
    (16, 8, 128),  # GQA group 2, wide head dim (Qwen3-ish)
    (16, 2, 64),   # GQA group 8
]


def _stage(num_heads, num_kv_heads, head_dim, **kwargs):
    return AttentionStage(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        tilelang_backend="cpu",
        seed=0,
        **kwargs,
    )


def _assert_matches(result):
    torch.testing.assert_close(
        result.tilelang_output.float(),
        result.reference_output.float(),
        atol=ATOL,
        rtol=RTOL,
    )


# Decode: single-query GQA over a padded KV cache.  context_lens exercise
# block-aligned, sub-block, and multi-batch ragged cases.
DECODE_CONTEXT_LENS = [
    [64],          # single sequence, sub-block (padded to block_N)
    [128],         # single sequence, exactly one block
    [64, 128],     # ragged batch
    [1, 200],      # min length + spans two blocks
]


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", HEAD_CONFIGS)
@pytest.mark.parametrize("context_lens", DECODE_CONTEXT_LENS)
def test_decode_matches_golden(num_heads, num_kv_heads, head_dim, context_lens):
    stage = _stage(num_heads, num_kv_heads, head_dim)
    result = stage.run_decode(context_lens)
    _assert_matches(result)


# Prefill: varlen causal GQA.  seq_lens exercise block-aligned, non-aligned, and
# multi-sequence packing (cu_seqlens with >1 segment).
PREFILL_SEQ_LENS = [
    [64],          # one block
    [80],          # non block-aligned (padded up to block_M)
    [64, 64],      # two equal sequences
    [48, 96],      # ragged, both non-aligned
]


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", HEAD_CONFIGS)
@pytest.mark.parametrize("seq_lens", PREFILL_SEQ_LENS)
def test_prefill_matches_golden(num_heads, num_kv_heads, head_dim, seq_lens):
    stage = _stage(num_heads, num_kv_heads, head_dim)
    result = stage.run_prefill(seq_lens)
    _assert_matches(result)


def test_decode_invalid_gqa_rejected():
    with pytest.raises(ValueError):
        _stage(num_heads=8, num_kv_heads=3, head_dim=64)


def test_invalid_backend_rejected():
    from nanovllm.backends.tilelang.attention import run_tilelang_attention_decode

    q = torch.randn(1, 8, 64)
    k = torch.randn(1, 128, 2, 64)
    v = torch.randn(1, 128, 2, 64)
    mask = torch.ones(1, 128, 2, dtype=torch.uint8)
    with pytest.raises(ValueError):
        run_tilelang_attention_decode(q, k, v, mask, softmax_scale=0.125, backend="rvv")
