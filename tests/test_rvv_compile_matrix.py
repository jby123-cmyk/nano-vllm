"""
Smoke test for the RVV llvm compiler pathway on attention kernels.

Times ``lower_to_host_device_ir`` + ``host_codegen`` through ``rvv_target()``
(riscv64-unknown-elf + ``+zvl``) without writing artifacts.  Full matrix timing
is available via::

    python demos/rvv/matrix_report.py --rvv-compile-quick --out-dir demos/build_rvv
"""

import pytest

pytest.importorskip("tilelang")

from tilelang import tvm as _tvm  # noqa: E402

if not _tvm.runtime.enabled("llvm"):
    pytest.skip("TVM build has no LLVM target; RVV compile path unavailable.", allow_module_level=True)

from demos.matrix_report import (  # noqa: E402
    DECODE_CONTEXT_LENS,
    HIGHLIGHT_HEADS,
    _collect_rvv_compile_specs,
    _run_rvv_compile_matrix,
)
from nanovllm.backends.tilelang.rvv_lower import time_rvv_lower, rvv_target, assert_target_vlen  # noqa: E402


def test_rvv_compile_one_decode_signature():
    specs = _collect_rvv_compile_specs(nr_lanes=4, head_configs=[HIGHLIGHT_HEADS])
    decode_specs = [s for s in specs if s[2]["phase"] == "decode"]
    assert decode_specs, "expected at least one decode RVV compile spec"
    tir, _sig, meta = decode_specs[0]
    target = rvv_target(nr_lanes=4)
    assert_target_vlen(target, 4)
    timing = time_rvv_lower(tir, target)
    assert timing.compiled, timing.error
    assert timing.vectorized, "expected RVV ops in generated assembly"
    assert timing.compile_ms > 0
    assert meta["coverage_note"] == DECODE_CONTEXT_LENS[0][1]


@pytest.mark.slow
def test_rvv_compile_quick_matrix():
    rows = _run_rvv_compile_matrix(4, head_configs=[HIGHLIGHT_HEADS])
    assert len(rows) == 7  # decode [64] and [128] share padded_kv=128
    assert all(r.rvv_compiled for r in rows), [r.rvv_error for r in rows if not r.rvv_compiled]
    assert all(r.rvv_vectorized for r in rows)
