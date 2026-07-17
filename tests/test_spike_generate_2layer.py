"""Phase 4 gate: 2-layer tiny Qwen, serial prefill, ``max_tokens=8``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nanovllm.backends.spike.config import default_config
from nanovllm.backends.spike.engine_golden import ensure_tiny_fixture_2l, tiny_llm_kwargs
from nanovllm.backends.spike.generate_report import REPORT_JSON_NAME
from nanovllm.backends.spike.report_context import install_collector
from nanovllm.backends.spike.session import reset_spike_session
from nanovllm.backends.tilelang.runtime import _RvvKernelCache

_cfg = default_config()
if _cfg.missing_tools():
    pytest.skip(
        "Spike toolchain unavailable; set ARAXL_ROOT or install tools. "
        f"Missing: {_cfg.missing_tools()[:3]}",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")
pytest.importorskip("transformers")
from nanovllm import LLM, SamplingParams  # noqa: E402

PHASE4_MAX_TOKENS = 8


@pytest.fixture(autouse=True)
def _reset_rvv_runtime():
    from tilelang.cache.kernel_cache import KernelCache

    from nanovllm.backends.tilelang.runtime import configure_tilelang_runtime

    _RvvKernelCache.reset()
    reset_spike_session()
    install_collector(None)
    KernelCache().clear_cache()
    configure_tilelang_runtime(
        backend="cuda",
        compile_only=False,
        execution_backend="compile_only",
    )
    yield
    _RvvKernelCache.reset()
    reset_spike_session()
    install_collector(None)
    KernelCache().clear_cache()


@pytest.fixture(scope="module")
def tiny_fixture_2l() -> str:
    return ensure_tiny_fixture_2l(seed=0)


@pytest.fixture(scope="module")
def prompt_tokens(tiny_fixture_2l: str) -> list[int]:
    meta_path = os.path.join(tiny_fixture_2l, "fixture_meta.json")
    with open(meta_path, encoding="utf-8") as handle:
        return json.load(handle)["prompt_tokens"]


def _generate(
    tiny_fixture: str,
    prompt_tokens: list[int],
    *,
    spike: bool,
    host_cpu_reference: bool,
    build_suffix: str,
    **report_kwargs,
) -> tuple[list[int], Path]:
    build_dir = Path(f"demos/build_rvv/tiny_qwen_2l_{build_suffix}")
    torch.manual_seed(42)
    llm = None
    try:
        llm = LLM(
            tiny_fixture,
            **tiny_llm_kwargs(
                spike=spike,
                host_cpu_reference=host_cpu_reference,
                num_layers=2,
                serial_prefill=True,
                build_dir=str(build_dir),
                **report_kwargs,
            ),
        )
        outputs = llm.generate(
            [prompt_tokens],
            SamplingParams(temperature=1e-9, max_tokens=PHASE4_MAX_TOKENS),
            use_tqdm=False,
        )
        return outputs[0]["token_ids"], build_dir
    finally:
        if llm is not None:
            llm.exit()


def _load_report(build_dir: Path) -> dict:
    report_path = build_dir / REPORT_JSON_NAME
    assert report_path.is_file(), f"missing generate report at {report_path}"
    with open(report_path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.slow
def test_spike_generate_2layer_parity(tiny_fixture_2l: str, prompt_tokens: list[int]):
    ref_build = Path("demos/build_rvv/tiny_qwen_2l_cpu_ref")
    ref_ids, _ = _generate(
        tiny_fixture_2l,
        prompt_tokens,
        spike=False,
        host_cpu_reference=True,
        build_suffix="cpu_ref",
        generate_report=True,
        generate_report_trace=True,
    )
    ref_report = _load_report(ref_build)
    spike_ids, spike_build = _generate(
        tiny_fixture_2l,
        prompt_tokens,
        spike=True,
        host_cpu_reference=False,
        build_suffix="spike_parity",
        generate_report_compare=True,
        generate_report_golden_dir=ref_build,
    )
    assert spike_ids == ref_ids, (
        f"Spike token_ids {spike_ids!r} != host cpu reference {ref_ids!r}"
    )
    spike_report = _load_report(spike_build)
    assert spike_report["tokens"]["golden_output_tokens"] == ref_ids
    assert spike_report["summary"]["tokens_match_golden"] is True
    assert ref_report["tokens"]["output_tokens"] == ref_ids
    if spike_report["summary"]["kernel_max_abs_diff"] is not None:
        assert spike_report["summary"]["kernel_max_abs_diff"] <= 1e-2
    assert spike_report["summary"]["engine_steps"] > 0
    assert Path(spike_build / "generate_report.md").is_file()
