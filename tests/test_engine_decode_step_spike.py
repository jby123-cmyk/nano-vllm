"""Phase 2 gate: one decoder-layer decode step on Spike vs PyTorch reference."""

from __future__ import annotations

import pytest

from nanovllm.backends.spike.config import default_config
from nanovllm.backends.spike.engine_golden import run_engine_decode_step_spike

_cfg = default_config()
if _cfg.missing_tools():
    pytest.skip(
        "Spike toolchain unavailable; set ARAXL_ROOT or install tools. "
        f"Missing: {_cfg.missing_tools()[:3]}",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")


@pytest.mark.slow
def test_engine_decode_step_spike():
    result = run_engine_decode_step_spike(
        batch_size=1,
        context_len=64,
        build_dir="demos/build_rvv/engine_decode_step_test",
    )
    torch.testing.assert_close(
        result.spike_output.float(),
        result.reference_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
