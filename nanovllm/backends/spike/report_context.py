"""Active generate-report collector and per-step context for engine runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.backends.spike.generate_report import GenerateReportCollector

_COLLECTOR: GenerateReportCollector | None = None


@dataclass
class ReportContext:
    phase: str = "init"
    step_id: int = -1
    in_warmup: bool = False
    in_generate: bool = False


_CONTEXT = ReportContext()


def install_collector(collector: GenerateReportCollector | None) -> None:
    global _COLLECTOR
    _COLLECTOR = collector
    if collector is None:
        _CONTEXT.phase = "init"
        _CONTEXT.step_id = -1
        _CONTEXT.in_warmup = False
        _CONTEXT.in_generate = False


def active_collector() -> GenerateReportCollector | None:
    return _COLLECTOR


def report_context() -> ReportContext:
    return _CONTEXT


def report_enabled() -> bool:
    return _COLLECTOR is not None


def set_warmup_phase(active: bool) -> None:
    _CONTEXT.in_warmup = active
    _CONTEXT.phase = "warmup" if active else "init"


def begin_generate_run() -> None:
    _CONTEXT.in_generate = True
    _CONTEXT.step_id = -1


def begin_engine_step(*, phase: str) -> None:
    if _CONTEXT.in_warmup:
        _CONTEXT.phase = "warmup"
        return
    _CONTEXT.step_id += 1
    _CONTEXT.phase = phase


def current_report_phase() -> str:
    if _CONTEXT.in_warmup:
        return "warmup"
    return _CONTEXT.phase


def current_step_id() -> int:
    return _CONTEXT.step_id
