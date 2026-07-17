"""
Spike ISA-sim subset for the RVV attention matrix.

Skips cleanly when the AraXL Spike toolchain is unavailable. Runs a small
subset (1 decode + 1 prefill) so the test stays CI-affordable.

Full matrix::

    python demos/rvv/spike_matrix.py --nr-lanes 4
"""

from __future__ import annotations

import pytest

from nanovllm.backends.spike.config import default_config

_cfg = default_config()
if _cfg.missing_tools():
    pytest.skip(
        "Spike toolchain unavailable; set ARAXL_ROOT or install tools. "
        f"Missing: {_cfg.missing_tools()[:3]}",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from nanovllm.backends.spike.pipeline import run_case  # noqa: E402


@pytest.mark.slow
def test_spike_decode_small():
    result = run_case(
        "decode",
        [64],
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        nr_lanes=4,
        skip_host_sanity=True,
    )
    assert result.passed, result.error or result.spike_result
    assert result.spike_max_abs_diff <= 1e-2


@pytest.mark.slow
def test_spike_decode_paged_small():
    result = run_case(
        "decode_paged",
        [64],
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        nr_lanes=4,
        skip_host_sanity=True,
    )
    assert result.passed, result.error or result.spike_result
    assert result.spike_max_abs_diff <= 1e-2


@pytest.mark.slow
def test_spike_prefill_paged_small():
    result = run_case(
        "prefill_paged",
        [64, 16],
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        nr_lanes=4,
        skip_host_sanity=True,
    )
    assert result.passed, result.error or result.spike_result
    assert result.spike_max_abs_diff <= 1e-2


@pytest.mark.slow
def test_spike_prefill_small():
    result = run_case(
        "prefill",
        [64],
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        nr_lanes=4,
        skip_host_sanity=True,
    )
    assert result.passed, result.error or result.spike_result
    assert result.spike_max_abs_diff <= 1e-2


@pytest.mark.slow
def test_spike_linear_small():
    result = run_case(
        "linear",
        [1, 256, 128],
        num_heads=1,
        num_kv_heads=1,
        head_dim=64,
        nr_lanes=4,
        skip_host_sanity=True,
    )
    assert result.passed, result.error or result.spike_result
    assert result.spike_max_abs_diff <= 1e-2


@pytest.mark.slow
def test_spike_silu_mul_small():
    result = run_case(
        "silu_mul",
        [8, 256],
        num_heads=1,
        num_kv_heads=1,
        head_dim=64,
        nr_lanes=4,
        skip_host_sanity=True,
    )
    assert result.passed, result.error or result.spike_result
    assert result.spike_max_abs_diff <= 1e-2


@pytest.mark.slow
def test_spike_rmsnorm_small():
    result = run_case(
        "rmsnorm",
        [8, 128, 0],
        num_heads=1,
        num_kv_heads=1,
        head_dim=64,
        nr_lanes=4,
        skip_host_sanity=True,
    )
    assert result.passed, result.error or result.spike_result
    assert result.spike_max_abs_diff <= 1e-2


@pytest.mark.slow
def test_spike_rope_small():
    result = run_case(
        "rope",
        [8, 4, 2, 64],
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        nr_lanes=4,
        skip_host_sanity=True,
    )
    assert result.passed, result.error or result.spike_result
    assert result.spike_max_abs_diff <= 1e-2


@pytest.mark.slow
def test_spike_kv_store_small():
    result = run_case(
        "kv_store",
        [8, 2, 64, 512],
        num_heads=2,
        num_kv_heads=2,
        head_dim=64,
        nr_lanes=4,
        skip_host_sanity=True,
    )
    assert result.passed, result.error or result.spike_result
    assert result.spike_max_abs_diff <= 1e-2
