"""Spike ISA-sim harness for RVV TileLang kernels.

Builds ``riscv64-unknown-elf`` ELFs from TileLang RVV ``.ll``, embeds the
PyTorch/AttentionStage golden, runs on Spike (no ``pk``), and parses
``HARNESS_SUMMARY``.

See ``spike_integration_plan.md`` and ``usage.md``.
"""

from nanovllm.backends.spike.config import SpikeConfig, default_config
from nanovllm.backends.spike.pipeline import CaseResult, run_case
from nanovllm.backends.spike.run import SpikeResult, run_spike_elf, smoke_matmul

__all__ = [
    "SpikeConfig",
    "default_config",
    "run_case",
    "CaseResult",
    "run_spike_elf",
    "SpikeResult",
    "smoke_matmul",
]
