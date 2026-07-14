"""End-to-end: emit IR → golden → harness → build → Spike run for one case."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

from nanovllm.backends.spike.build import build_spike_elf
from nanovllm.backends.spike.config import SpikeConfig, default_config
from nanovllm.backends.spike.emit_ir import emit_kernel_ll
from nanovllm.backends.spike.golden import emit_golden, host_sanity_check
from nanovllm.backends.spike.harness import render_harness
from nanovllm.backends.spike.run import SpikeResult, run_spike_elf


@dataclass
class CaseResult:
    case_id: str
    phase: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lengths: str
    nr_lanes: int
    compiled: bool
    host_max_abs_diff: float | None
    spike_result: str
    spike_max_abs_diff: float
    mismatches: int
    build_ms: float
    run_ms: float
    passed: bool
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def run_case(
    phase: str,
    lengths: list[int],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    nr_lanes: int = 4,
    out_root: str = "demos/build_rvv/spike",
    cfg: SpikeConfig | None = None,
    skip_host_sanity: bool = False,
) -> CaseResult:
    cfg = cfg or default_config(nr_lanes=nr_lanes)
    cfg.assert_available()

    ir = emit_kernel_ll(
        phase,
        lengths,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        nr_lanes=nr_lanes,
        out_root=out_root,
    )
    if not ir.compiled:
        return CaseResult(
            case_id=ir.case_id,
            phase=phase,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            lengths=str(lengths),
            nr_lanes=nr_lanes,
            compiled=False,
            host_max_abs_diff=None,
            spike_result="compile_fail",
            spike_max_abs_diff=float("nan"),
            mismatches=-1,
            build_ms=0.0,
            run_ms=0.0,
            passed=False,
            error=ir.error or "RVV lowering failed",
        )

    golden = emit_golden(
        phase,
        lengths,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        nr_lanes=nr_lanes,
        out_root=out_root,
        atol=cfg.atol,
        case_id=ir.case_id,
    )

    host_diff: float | None = None
    if not skip_host_sanity:
        host_diff = host_sanity_check(golden)
        if host_diff > cfg.atol:
            return CaseResult(
                case_id=ir.case_id,
                phase=phase,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                lengths=str(lengths),
                nr_lanes=nr_lanes,
                compiled=True,
                host_max_abs_diff=host_diff,
                spike_result="host_golden_fail",
                spike_max_abs_diff=float("nan"),
                mismatches=-1,
                build_ms=0.0,
                run_ms=0.0,
                passed=False,
                error=f"host sanity max_abs_diff={host_diff} > atol={cfg.atol}",
            )

    harness = render_harness(golden.manifest_path, golden.case_dir)
    t0 = time.perf_counter()
    build = build_spike_elf(
        ir.ll_path,
        harness.main_c,
        golden.case_dir,
        elf_name=f"{ir.case_id}.spike",
        cfg=cfg,
    )
    build_ms = (time.perf_counter() - t0) * 1000.0

    spike: SpikeResult = run_spike_elf(build.elf_path, cfg=cfg)
    log_path = os.path.join(golden.case_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(spike.stdout)
        handle.write(
            f"\nspike_return_code: {spike.returncode}\n"
            f"spike_result: {spike.spike_result}\n"
            f"max_abs_diff: {spike.max_abs_diff}\n"
            f"host_max_abs_diff: {host_diff}\n"
            f"build_ms: {build_ms}\n"
            f"run_ms: {spike.run_ms}\n"
        )

    return CaseResult(
        case_id=ir.case_id,
        phase=phase,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        lengths=str(lengths),
        nr_lanes=nr_lanes,
        compiled=True,
        host_max_abs_diff=host_diff,
        spike_result=spike.spike_result,
        spike_max_abs_diff=spike.max_abs_diff,
        mismatches=spike.mismatches,
        build_ms=build_ms,
        run_ms=spike.run_ms,
        passed=spike.passed,
        error="" if spike.passed else spike.stdout[-1500:],
    )
