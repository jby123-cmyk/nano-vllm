"""Engine helpers for Spike bring-up: decode-step gate + tiny-model fixture."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nanovllm.backends.spike.session import reset_spike_session
from nanovllm.backends.tilelang.runtime import _RvvKernelCache, configure_tilelang_runtime
from nanovllm.stages.decoder_layer import DecoderLayerResult, DecoderLayerStage

# Engine call site → Spike matrix ``phase`` (see ``KERNEL_ARG_NAMES`` in session.py).
ENGINE_OP_PHASES: dict[str, str] = {
    "VocabParallelEmbedding": "embedding",
    "QKVParallelLinear": "linear",
    "MergedColumnParallelLinear": "linear",
    "RowParallelLinear": "linear",
    "ReplicatedLinear": "linear",
    "ParallelLMHead": "linear",
    "RMSNorm": "rmsnorm",
    "RotaryEmbedding": "rope",
    "store_kvcache": "kv_store",
    "Attention.prefill": "prefill_paged",
    "Attention.decode": "decode_paged",
    "SiluAndMul": "silu_mul",
}

TINY_QWEN_SPEC: dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_hidden_layers": 1,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "float32",
    "attention_bias": False,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_QWEN_PATH = REPO_ROOT / "tests" / "fixtures" / "tiny_qwen"
TINY_QWEN_2L_PATH = REPO_ROOT / "tests" / "fixtures" / "tiny_qwen_2l"
TINY_QWEN_BUILD_DIR = REPO_ROOT / "demos" / "build_rvv" / "tiny_qwen"
TINY_QWEN_2L_BUILD_DIR = REPO_ROOT / "demos" / "build_rvv" / "tiny_qwen_2l"

TINY_QWEN_2L_SPEC: dict[str, Any] = {**TINY_QWEN_SPEC, "num_hidden_layers": 2}


def tiny_fixture_path(*, num_layers: int = 1) -> Path:
    if num_layers == 1:
        return TINY_QWEN_PATH
    if num_layers == 2:
        return TINY_QWEN_2L_PATH
    raise ValueError(f"tiny fixture supports num_layers 1 or 2, got {num_layers}")


def tiny_fixture_spec(*, num_layers: int = 1) -> dict[str, Any]:
    if num_layers == 1:
        return TINY_QWEN_SPEC
    if num_layers == 2:
        return TINY_QWEN_2L_SPEC
    raise ValueError(f"tiny fixture supports num_layers 1 or 2, got {num_layers}")


@dataclass
class EngineDecodeStepResult:
    """Spike vs reference for one composed decoder layer (decode step)."""

    reference_output: torch.Tensor
    spike_output: torch.Tensor
    max_abs_diff: float


def configure_spike_engine_runtime(
    *,
    nr_lanes: int = 4,
    build_dir: str | os.PathLike[str] | None = None,
    spike_cache_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Install RVV + Spike execution for ``LLM(..., device='rvv')``."""
    build_dir = str(build_dir or TINY_QWEN_BUILD_DIR)
    spike_cache_dir = str(spike_cache_dir or os.path.join(build_dir, "spike_exec"))
    _RvvKernelCache.reset()
    reset_spike_session()
    configure_tilelang_runtime(
        backend="rvv",
        nr_lanes=nr_lanes,
        build_dir=build_dir,
        compile_only=False,
        execution_backend="spike",
        spike_cache_dir=spike_cache_dir,
    )


def configure_host_cpu_tilelang_runtime() -> None:
    """Host llvm TileLang reference (parity baseline for tests)."""
    _RvvKernelCache.reset()
    reset_spike_session()
    configure_tilelang_runtime(
        backend="cpu",
        compile_only=False,
        execution_backend="compile_only",
    )


def tiny_llm_kwargs(
    *,
    spike: bool = True,
    host_cpu_reference: bool = False,
    build_dir: str | os.PathLike[str] | None = None,
    num_layers: int = 1,
    serial_prefill: bool = False,
    generate_report: bool | None = None,
    generate_report_compare: bool = False,
    generate_report_trace: bool = False,
    generate_report_golden_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Shared ``LLM`` kwargs for the tiny Qwen fixture."""
    if num_layers == 1:
        default_build_dir = TINY_QWEN_BUILD_DIR
        kvcache_blocks = 8
        max_batched_tokens = 8
    elif num_layers == 2:
        default_build_dir = TINY_QWEN_2L_BUILD_DIR
        kvcache_blocks = 16
        max_batched_tokens = 8
    else:
        raise ValueError(f"tiny fixture supports num_layers 1 or 2, got {num_layers}")

    build_dir = str(build_dir or default_build_dir)
    kwargs: dict[str, Any] = {
        "device": "rvv",
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "rvv_nr_lanes": 4,
        "rvv_build_dir": build_dir,
        "max_model_len": 64,
        "max_num_seqs": 1,
        "max_num_batched_tokens": max_batched_tokens,
        "rvv_num_kvcache_blocks": kvcache_blocks,
    }
    if serial_prefill:
        kwargs["rvv_serial_prefill"] = True
    if host_cpu_reference:
        kwargs["rvv_host_cpu_tilelang"] = True
    elif spike:
        kwargs["rvv_execution_backend"] = "spike"
        kwargs["rvv_compile_only"] = False
    if generate_report is not None:
        kwargs["rvv_generate_report"] = generate_report
    if generate_report_compare:
        kwargs["rvv_generate_report_compare"] = True
    if generate_report_trace:
        kwargs["rvv_generate_report_trace"] = True
    if generate_report_golden_dir is not None:
        kwargs["rvv_generate_report_golden_dir"] = str(generate_report_golden_dir)
    return kwargs


def ensure_tiny_fixture(
    *,
    seed: int = 0,
    force: bool = False,
    num_layers: int = 1,
) -> str:
    """Build tiny Qwen fixture if missing; return its path."""
    path = tiny_fixture_path(num_layers=num_layers)
    weights = path / "model.safetensors"
    if force or not weights.is_file():
        from scripts.build_tiny_qwen_fixture import build_tiny_qwen_fixture

        build_tiny_qwen_fixture(out_dir=str(path), seed=seed, num_layers=num_layers)
    if not weights.is_file():
        raise FileNotFoundError(f"missing tiny fixture weights at {weights}")
    return str(path)


def ensure_tiny_fixture_2l(*, seed: int = 0, force: bool = False) -> str:
    """Build ``tests/fixtures/tiny_qwen_2l`` if missing; return its path."""
    return ensure_tiny_fixture(seed=seed, force=force, num_layers=2)


def run_engine_decode_step_spike(
    *,
    batch_size: int = 1,
    context_len: int = 64,
    hidden_size: int = 256,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    intermediate_size: int = 512,
    nr_lanes: int = 4,
    build_dir: str = "demos/build_rvv/engine_decode_step",
    spike_cache_dir: str | None = None,
    seed: int = 0,
    decode_block_N: int = 64,
) -> EngineDecodeStepResult:
    """Run one decode step on Spike and compare to the stage's PyTorch reference."""
    configure_spike_engine_runtime(
        nr_lanes=nr_lanes,
        build_dir=build_dir,
        spike_cache_dir=spike_cache_dir,
    )
    stage = DecoderLayerStage(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        tilelang_backend="rvv",
        seed=seed,
        decode_block_N=decode_block_N,
    )
    result: DecoderLayerResult = stage.run_decode_step(
        batch_size=batch_size,
        context_len=context_len,
    )
    return EngineDecodeStepResult(
        reference_output=result.reference_output,
        spike_output=result.tilelang_output,
        max_abs_diff=result.max_abs_diff,
    )


def write_tiny_config(path: str | os.PathLike[str], spec: dict[str, Any] | None = None) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "config.json", "w", encoding="utf-8") as handle:
        json.dump(spec or TINY_QWEN_SPEC, handle, indent=2, sort_keys=True)
        handle.write("\n")
