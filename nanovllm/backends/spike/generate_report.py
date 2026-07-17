"""Collect and write per-run ``LLM.generate()`` reports under ``rvv_build_dir``."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import torch

from nanovllm.backends.spike.config import DEFAULT_ATOL

GOLDEN_TRACE_NAME = "golden_trace.jsonl"
REPORT_JSON_NAME = "generate_report.json"
REPORT_MD_NAME = "generate_report.md"


def tensor_max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item())


def comparable_output_names(
    kernel_name: str,
    outputs: tuple[str, ...],
    build_args: tuple,
) -> tuple[str, ...]:
    if kernel_name == "rmsnorm" and len(build_args) > 3 and not build_args[3]:
        return ("output",)
    return outputs


def _linear_active_dims(
    x_pad: torch.Tensor,
    weight_pad: torch.Tensor,
) -> tuple[int, int]:
    """Infer unpadded GEMM extents from zero-padded TileLang inputs."""
    active_m = int((x_pad.abs().sum(dim=1) > 0).sum().item())
    active_n = int((weight_pad.abs().sum(dim=1) > 0).sum().item())
    return max(active_m, 1), max(active_n, 1)


def _normalize_output_map(
    outputs: dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    output_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return {output_names[0]: outputs}
    if isinstance(outputs, (tuple, list)):
        return dict(zip(output_names, outputs, strict=True))
    return outputs


def compare_kernel_outputs(
    *,
    kernel_name: str,
    build_args: tuple,
    positional_args: tuple[torch.Tensor | int, ...] | None,
    spike_outputs: dict[str, torch.Tensor],
    reference: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | dict[str, torch.Tensor],
    output_names: tuple[str, ...],
) -> float:
    names = comparable_output_names(kernel_name, output_names, build_args)
    spike_map = {name: spike_outputs[name] for name in names}
    ref_map = _normalize_output_map(reference, names)

    if kernel_name == "linear" and positional_args is not None:
        active_m, active_n = _linear_active_dims(positional_args[0], positional_args[1])
        return tensor_max_abs_diff(
            spike_map[names[0]][:active_m, :active_n],
            ref_map[names[0]][:active_m, :active_n],
        )

    if kernel_name == "kv_store" and positional_args is not None:
        slot_mapping = positional_args[4]
        assert isinstance(slot_mapping, torch.Tensor)
        slots = slot_mapping[slot_mapping >= 0].unique()
        if slots.numel() == 0:
            return 0.0
        peak = 0.0
        for slot in slots.tolist():
            slot = int(slot)
            peak = max(peak, tensor_max_abs_diff(spike_map["k_cache"][slot], ref_map["k_cache"][slot]))
            peak = max(peak, tensor_max_abs_diff(spike_map["v_cache"][slot], ref_map["v_cache"][slot]))
        return peak

    peak = 0.0
    for name in names:
        peak = max(peak, tensor_max_abs_diff(spike_map[name], ref_map[name]))
    return peak


def outputs_max_abs_diff(
    spike_outputs: dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...],
    reference: torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor],
    output_names: tuple[str, ...],
) -> float:
    if isinstance(spike_outputs, torch.Tensor):
        spike_map = {output_names[0]: spike_outputs}
    elif isinstance(spike_outputs, tuple):
        spike_map = dict(zip(output_names, spike_outputs, strict=True))
    else:
        spike_map = spike_outputs

    if isinstance(reference, torch.Tensor):
        ref_map = {output_names[0]: reference}
    elif isinstance(reference, (tuple, list)):
        ref_map = dict(zip(output_names, reference, strict=True))
    else:
        ref_map = reference

    peak = 0.0
    for name in output_names:
        peak = max(peak, tensor_max_abs_diff(spike_map[name], ref_map[name]))
    return peak


@dataclass
class KernelCallRecord:
    call_id: int
    kernel_name: str
    phase: str
    step_id: int
    case_id: str
    case_rel_path: str
    compile_ms: float
    simulate_ms: float
    spike_result: str | None = None
    max_abs_diff_vs_reference: float | None = None
    cumulative_max_abs_diff: float | None = None
    within_atol: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineStepRecord:
    step_id: int
    phase: str
    num_tokens: int
    wall_ms: float
    sampled_token_id: int | None = None
    logits_max_abs_diff: float | None = None
    cumulative_logits_max_abs_diff: float | None = None
    within_atol: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerateReport:
    build_dir: str
    generated_at: str
    model_path: str
    execution_backend: str
    tilelang_backend: str
    rvv_nr_lanes: int
    num_hidden_layers: int
    atol: float
    tokens: dict[str, list[int] | None] = field(
        default_factory=lambda: {
            "prompt_tokens": None,
            "output_tokens": None,
            "golden_output_tokens": None,
        }
    )
    timing_ms: dict[str, Any] = field(default_factory=dict)
    kernel_calls: list[KernelCallRecord] = field(default_factory=list)
    engine_steps: list[EngineStepRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_dir": self.build_dir,
            "generated_at": self.generated_at,
            "model_path": self.model_path,
            "execution_backend": self.execution_backend,
            "tilelang_backend": self.tilelang_backend,
            "rvv_nr_lanes": self.rvv_nr_lanes,
            "num_hidden_layers": self.num_hidden_layers,
            "atol": self.atol,
            "tokens": self.tokens,
            "timing_ms": self.timing_ms,
            "kernel_calls": [row.to_dict() for row in self.kernel_calls],
            "engine_steps": [row.to_dict() for row in self.engine_steps],
            "summary": self.summary,
        }


class GenerateReportCollector:
    def __init__(self, *, build_dir: str, config) -> None:
        self.build_dir = build_dir
        self.config = config
        self.atol = DEFAULT_ATOL
        self._started_at = perf_counter()
        self._init_started_at: float | None = None
        self._init_ended_at: float | None = None
        self._generate_started_at: float | None = None
        self._generate_ended_at: float | None = None
        self._kernel_calls: list[KernelCallRecord] = []
        self._engine_steps: list[EngineStepRecord] = []
        self._kernel_call_id = 0
        self._kernel_cumulative_peak = 0.0
        self._stage_cumulative_peak = 0.0
        self._compile_digest_seen: set[str] = set()
        self._compile_total_raw = 0.0
        self._compile_total_deduped = 0.0
        self._simulate_total = 0.0
        self._phase_compile: dict[str, float] = {}
        self._phase_simulate: dict[str, float] = {}
        self._phase_kernel_calls: dict[str, int] = {}
        self._golden_trace_rows: list[dict[str, Any]] = []
        self._golden_trace_by_step: dict[int, dict[str, Any]] = {}
        self.prompt_tokens: list[int] | None = None
        self.golden_output_tokens: list[int] | None = None
        self.output_tokens: list[int] | None = None
        self._load_golden_trace()

    @classmethod
    def open(cls, *, build_dir: str, config) -> GenerateReportCollector:
        os.makedirs(build_dir, exist_ok=True)
        for name in (REPORT_JSON_NAME, REPORT_MD_NAME):
            path = os.path.join(build_dir, name)
            if os.path.isfile(path):
                os.remove(path)
        trace_path = os.path.join(build_dir, GOLDEN_TRACE_NAME)
        if os.path.isfile(trace_path):
            os.remove(trace_path)
        return cls(build_dir=build_dir, config=config)

    def compare_kernels_enabled(self) -> bool:
        return bool(self.config.rvv_generate_report_compare)

    def trace_logits_enabled(self) -> bool:
        return bool(self.config.rvv_generate_report_trace)

    def golden_dir(self) -> str | None:
        path = getattr(self.config, "rvv_generate_report_golden_dir", None)
        return str(path) if path else None

    def mark_init_start(self) -> None:
        self._init_started_at = perf_counter()

    def mark_init_end(self) -> None:
        self._init_ended_at = perf_counter()

    def begin_generate(self) -> None:
        self._generate_started_at = perf_counter()

    def set_prompt_tokens_if_empty(self, tokens: list[int]) -> None:
        if self.prompt_tokens is None:
            self.prompt_tokens = list(tokens)

    def set_golden_output_tokens(self, tokens: list[int]) -> None:
        self.golden_output_tokens = list(tokens)

    def set_output_tokens(self, tokens: list[int]) -> None:
        self.output_tokens = list(tokens)

    def record_kernel_call(
        self,
        *,
        kernel_name: str,
        phase: str,
        step_id: int,
        case_id: str,
        case_rel_path: str,
        compile_ms: float,
        simulate_ms: float,
        spike_result: str | None = None,
        max_abs_diff_vs_reference: float | None = None,
        build_args_digest: str | None = None,
    ) -> None:
        self._kernel_call_id += 1
        if max_abs_diff_vs_reference is not None:
            self._kernel_cumulative_peak = max(
                self._kernel_cumulative_peak, max_abs_diff_vs_reference
            )
        within_atol = None
        if max_abs_diff_vs_reference is not None:
            within_atol = max_abs_diff_vs_reference <= self.atol

        self._kernel_calls.append(
            KernelCallRecord(
                call_id=self._kernel_call_id,
                kernel_name=kernel_name,
                phase=phase,
                step_id=step_id,
                case_id=case_id,
                case_rel_path=case_rel_path,
                compile_ms=compile_ms,
                simulate_ms=simulate_ms,
                spike_result=spike_result,
                max_abs_diff_vs_reference=max_abs_diff_vs_reference,
                cumulative_max_abs_diff=(
                    self._kernel_cumulative_peak
                    if max_abs_diff_vs_reference is not None
                    else None
                ),
                within_atol=within_atol,
            )
        )

        self._compile_total_raw += compile_ms
        self._simulate_total += simulate_ms
        if build_args_digest is None:
            build_args_digest = case_id
        if build_args_digest not in self._compile_digest_seen:
            self._compile_digest_seen.add(build_args_digest)
            self._compile_total_deduped += compile_ms
        self._phase_compile[phase] = self._phase_compile.get(phase, 0.0) + compile_ms
        self._phase_simulate[phase] = self._phase_simulate.get(phase, 0.0) + simulate_ms
        self._phase_kernel_calls[phase] = self._phase_kernel_calls.get(phase, 0) + 1

    def end_engine_step(
        self,
        *,
        phase: str,
        step_id: int,
        num_tokens: int,
        started_at: float,
        logits: torch.Tensor | None,
        sampled_token_id: int | None,
    ) -> None:
        wall_ms = (perf_counter() - started_at) * 1000.0
        logits_diff = None
        within_atol = None

        if logits is not None and self.trace_logits_enabled():
            row = {
                "step_id": step_id,
                "phase": phase,
                "num_tokens": num_tokens,
                "sampled_token_id": sampled_token_id,
                "logits": logits.detach().float().cpu().tolist(),
            }
            self._golden_trace_rows.append(row)
            self._append_golden_trace_row(row)

        if logits is not None and self._golden_trace_by_step:
            golden = self._golden_trace_by_step.get(step_id)
            if golden is not None:
                golden_logits = torch.tensor(golden["logits"], dtype=torch.float32)
                logits_diff = tensor_max_abs_diff(logits, golden_logits)
                self._stage_cumulative_peak = max(self._stage_cumulative_peak, logits_diff)
                within_atol = logits_diff <= self.atol

        self._engine_steps.append(
            EngineStepRecord(
                step_id=step_id,
                phase=phase,
                num_tokens=num_tokens,
                wall_ms=wall_ms,
                sampled_token_id=sampled_token_id,
                logits_max_abs_diff=logits_diff,
                cumulative_logits_max_abs_diff=(
                    self._stage_cumulative_peak if logits_diff is not None else None
                ),
                within_atol=within_atol,
            )
        )

    def _golden_trace_path(self, build_dir: str | None = None) -> str:
        root = build_dir or self.build_dir
        return os.path.join(root, GOLDEN_TRACE_NAME)

    def _append_golden_trace_row(self, row: dict[str, Any]) -> None:
        path = self._golden_trace_path()
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    def _load_golden_trace(self) -> None:
        golden_dir = self.golden_dir()
        if not golden_dir:
            return
        trace_path = self._golden_trace_path(golden_dir)
        if not os.path.isfile(trace_path):
            return
        with open(trace_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._golden_trace_by_step[int(row["step_id"])] = row

        report_path = os.path.join(golden_dir, REPORT_JSON_NAME)
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            golden_tokens = payload.get("tokens", {}).get("output_tokens")
            if golden_tokens is not None:
                self.golden_output_tokens = list(golden_tokens)

    def _build_timing(self) -> dict[str, Any]:
        now = perf_counter()
        init_ms = None
        if self._init_started_at is not None and self._init_ended_at is not None:
            init_ms = (self._init_ended_at - self._init_started_at) * 1000.0
        generate_ms = None
        if self._generate_started_at is not None:
            end = self._generate_ended_at or now
            generate_ms = (end - self._generate_started_at) * 1000.0
        total_wall_ms = (now - self._started_at) * 1000.0

        by_phase: dict[str, dict[str, float | int]] = {}
        for phase in sorted(set(self._phase_compile) | set(self._phase_simulate)):
            by_phase[phase] = {
                "compile_ms": self._phase_compile.get(phase, 0.0),
                "simulate_ms": self._phase_simulate.get(phase, 0.0),
                "kernel_calls": self._phase_kernel_calls.get(phase, 0),
            }

        return {
            "init_ms": init_ms,
            "generate_ms": generate_ms,
            "total_wall_ms": total_wall_ms,
            "compile_total_raw_ms": self._compile_total_raw,
            "compile_total_deduped_ms": self._compile_total_deduped,
            "simulate_total_ms": self._simulate_total,
            "by_phase": by_phase,
        }

    def _build_summary(self) -> dict[str, Any]:
        compared_calls = [
            row
            for row in self._kernel_calls
            if row.max_abs_diff_vs_reference is not None and row.phase != "warmup"
        ]
        kernel_diffs = [row.max_abs_diff_vs_reference for row in compared_calls]
        stage_diffs = [
            row.logits_max_abs_diff
            for row in self._engine_steps
            if row.logits_max_abs_diff is not None
        ]
        kernel_cumulative_peak = 0.0
        for row in compared_calls:
            kernel_cumulative_peak = max(
                kernel_cumulative_peak, row.max_abs_diff_vs_reference or 0.0
            )
        tokens_match = None
        if self.output_tokens is not None and self.golden_output_tokens is not None:
            tokens_match = self.output_tokens == self.golden_output_tokens
        return {
            "kernel_calls": len(self._kernel_calls),
            "engine_steps": len(self._engine_steps),
            "kernel_max_abs_diff": max(kernel_diffs) if kernel_diffs else None,
            "kernel_cumulative_max_abs_diff": (
                kernel_cumulative_peak if kernel_diffs else None
            ),
            "stage_max_abs_diff": max(stage_diffs) if stage_diffs else None,
            "stage_cumulative_max_abs_diff": (
                self._stage_cumulative_peak if stage_diffs else None
            ),
            "tokens_match_golden": tokens_match,
            "within_atol": (
                (max(kernel_diffs) if kernel_diffs else 0.0) <= self.atol
                and (max(stage_diffs) if stage_diffs else 0.0) <= self.atol
            ),
        }

    def build_report(self) -> GenerateReport:
        hf_config = self.config.hf_config
        return GenerateReport(
            build_dir=self.build_dir,
            generated_at=datetime.now(timezone.utc).isoformat(),
            model_path=self.config.model,
            execution_backend=self.config.rvv_execution_backend,
            tilelang_backend=self.config.tilelang_backend,
            rvv_nr_lanes=self.config.rvv_nr_lanes,
            num_hidden_layers=int(hf_config.num_hidden_layers),
            atol=self.atol,
            tokens={
                "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
                "golden_output_tokens": self.golden_output_tokens,
            },
            timing_ms=self._build_timing(),
            kernel_calls=list(self._kernel_calls),
            engine_steps=list(self._engine_steps),
            summary=self._build_summary(),
        )

    def finalize(self) -> tuple[str, str]:
        self._generate_ended_at = perf_counter()
        if (
            self.golden_output_tokens is None
            and self.output_tokens is not None
            and self.trace_logits_enabled()
        ):
            self.golden_output_tokens = list(self.output_tokens)
        report = self.build_report()
        json_path = os.path.join(self.build_dir, REPORT_JSON_NAME)
        md_path = os.path.join(self.build_dir, REPORT_MD_NAME)
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, json_path)
        write_generate_markdown(md_path, report)
        return json_path, md_path


def write_generate_markdown(path: str, report: GenerateReport) -> None:
    lines = [
        "# Generate report",
        "",
        f"_Generated: {report.generated_at}_",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| model | `{report.model_path}` |",
        f"| execution_backend | {report.execution_backend} |",
        f"| tilelang_backend | {report.tilelang_backend} |",
        f"| num_hidden_layers | {report.num_hidden_layers} |",
        f"| kernel_calls | {report.summary.get('kernel_calls', 0)} |",
        f"| engine_steps | {report.summary.get('engine_steps', 0)} |",
        f"| tokens_match_golden | {report.summary.get('tokens_match_golden')} |",
        f"| kernel_max_abs_diff | {report.summary.get('kernel_max_abs_diff')} |",
        f"| stage_max_abs_diff | {report.summary.get('stage_max_abs_diff')} |",
        f"| within_atol ({report.atol}) | {report.summary.get('within_atol')} |",
        "",
        "## Tokens",
        "",
        f"- prompt: `{report.tokens.get('prompt_tokens')}`",
        f"- output: `{report.tokens.get('output_tokens')}`",
        f"- golden: `{report.tokens.get('golden_output_tokens')}`",
        "",
        "## Timing (ms)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    timing = report.timing_ms
    for key in (
        "init_ms",
        "generate_ms",
        "total_wall_ms",
        "compile_total_raw_ms",
        "compile_total_deduped_ms",
        "simulate_total_ms",
    ):
        lines.append(f"| {key} | {timing.get(key)} |")
    lines.extend(
        [
            "",
            "### By phase",
            "",
            "| phase | compile_ms | simulate_ms | kernel_calls |",
            "|-------|------------|-------------|--------------|",
        ]
    )
    for phase, row in sorted(timing.get("by_phase", {}).items()):
        lines.append(
            f"| {phase} | {row['compile_ms']:.2f} | {row['simulate_ms']:.2f} | {row['kernel_calls']} |"
        )

    if report.kernel_calls:
        compared = [
            row for row in report.kernel_calls if row.max_abs_diff_vs_reference is not None
        ]
        if compared:
            worst = max(compared, key=lambda row: row.max_abs_diff_vs_reference or 0.0)
            lines.extend(
                [
                    "",
                    "## Worst kernel (max_abs_diff_vs_reference)",
                    "",
                    f"- `{worst.kernel_name}` step={worst.step_id} phase={worst.phase}",
                    f"- max_abs_diff={worst.max_abs_diff_vs_reference:.6e}",
                    f"- case `{worst.case_rel_path}`",
                    "",
                ]
            )

    if report.engine_steps:
        lines.extend(
            [
                "## Engine steps",
                "",
                "| step | phase | wall_ms | token_id | logits_max_abs_diff | cumulative |",
                "|------|-------|---------|----------|---------------------|------------|",
            ]
        )
        for row in report.engine_steps:
            diff = (
                f"{row.logits_max_abs_diff:.6e}"
                if row.logits_max_abs_diff is not None
                else "n/a"
            )
            cumulative = (
                f"{row.cumulative_logits_max_abs_diff:.6e}"
                if row.cumulative_logits_max_abs_diff is not None
                else "n/a"
            )
            lines.append(
                f"| {row.step_id} | {row.phase} | {row.wall_ms:.2f} | "
                f"{row.sampled_token_id} | {diff} | {cumulative} |"
            )

    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
