"""
Numeric validation for TileLang linear (F.linear / GEMM).

Compiles ``build_linear_kernel`` on host ``llvm`` and compares against
``torch.nn.functional.linear``.

Run::

    pytest tests/test_linear_rvv_numeric.py -v
"""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
pytest.importorskip("tilelang")

from tilelang import tvm as _tvm  # noqa: E402

if not _tvm.runtime.enabled("llvm"):
    pytest.skip("TVM build has no LLVM target.", allow_module_level=True)

from nanovllm.backends.tilelang.linear import (  # noqa: E402
    DEFAULT_BLOCK_K,
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    run_tilelang_linear,
)
from nanovllm.stages.linear import LinearShape  # noqa: E402

ATOL = 1e-2
RTOL = 1e-2

# Host-llvm compile is slow for full Qwen dims; keep CI-affordable shapes that
# still exercise decode (M=1) and prefill-tile (M=64) plus bias/no-bias.
HOST_SHAPES = [
    LinearShape("small", 128, 256, False),
    LinearShape("qkv_ish", 256, 512, False),
    LinearShape("o_proj_ish", 512, 256, False),
]

TOKEN_COUNTS = [1, 64]


@pytest.mark.parametrize("shape", HOST_SHAPES, ids=lambda s: s.name)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("has_bias", [False, True])
def test_linear_host_llvm(shape, num_tokens, has_bias):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    x = torch.randn(
        num_tokens, shape.in_features, dtype=torch.float32, generator=gen
    )
    weight = torch.randn(
        shape.out_features, shape.in_features, dtype=torch.float32, generator=gen
    )
    bias = None
    if has_bias:
        bias = torch.randn(shape.out_features, dtype=torch.float32, generator=gen)

    reference = F.linear(x, weight, bias)
    out = run_tilelang_linear(
        x,
        weight,
        bias,
        backend="cpu",
        block_M=DEFAULT_BLOCK_M,
        block_N=DEFAULT_BLOCK_N,
        block_K=DEFAULT_BLOCK_K,
    )
    torch.testing.assert_close(
        out.float(),
        reference.float(),
        atol=ATOL,
        rtol=RTOL,
    )


def test_linear_stage_run():
    from nanovllm.stages.linear import LinearStage

    stage = LinearStage(
        in_features=128,
        out_features=256,
        has_bias=True,
        dtype=torch.float32,
        device="cpu",
        tilelang_backend="cpu",
        seed=1,
    )
    result = stage.run(32)
    assert result.max_abs_diff <= ATOL
