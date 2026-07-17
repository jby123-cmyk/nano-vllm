"""
Phase 1 gate: ``run_tilelang_linear`` round-trip on Spike vs host llvm.

Skips when the AraXL Spike toolchain is unavailable.
"""

from __future__ import annotations

import pytest

from nanovllm.backends.spike.config import default_config
from nanovllm.backends.spike.session import reset_spike_session
from nanovllm.backends.tilelang.runtime import _RvvKernelCache, configure_tilelang_runtime

_cfg = default_config()
if _cfg.missing_tools():
    pytest.skip(
        "Spike toolchain unavailable; set ARAXL_ROOT or install tools. "
        f"Missing: {_cfg.missing_tools()[:3]}",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from nanovllm.backends.tilelang.linear import run_tilelang_linear  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_rvv_runtime():
    _RvvKernelCache.reset()
    reset_spike_session()
    yield
    _RvvKernelCache.reset()
    reset_spike_session()


@pytest.mark.slow
def test_spike_linear_execute_matches_cpu():
    configure_tilelang_runtime(
        backend="rvv",
        nr_lanes=4,
        build_dir="demos/build_rvv/spike_session_test",
        compile_only=False,
        execution_backend="spike",
        spike_cache_dir="demos/build_rvv/spike_session_test/exec",
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    x = torch.randn(1, 128, dtype=torch.float32, generator=gen)
    weight = torch.randn(256, 128, dtype=torch.float32, generator=gen)

    ref = run_tilelang_linear(x, weight, None, backend="cpu")
    out = run_tilelang_linear(x, weight, None, backend="rvv")

    max_diff = float((out.float() - ref.float()).abs().max().item())
    assert max_diff <= 1e-2, f"Spike linear max_abs_diff={max_diff}"
