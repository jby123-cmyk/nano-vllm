"""
Compare cold compile time for FlashAttention decode/prefill across three pathways:

  * **host_x86** — TileLang CPU pipeline → native ``llvm`` host codegen (no JIT)
  * **rvv**       — TileLang CPU pipeline → ``riscv64`` RVV llvm/asm codegen
  * **cuda**      — TileLang GPU pipeline → ``.cu`` source generation (no nvcc/launch)

Runs the **8 slide scenarios** (4 decode + 4 prefill) at the representative wide
head config **16 / 8 / 128** (GQA group 2).  Outputs a presentation-ready CSV.

Usage (nanovllm env)::

    python demos/rvv/compile_compare_matrix.py
    python demos/rvv/compile_compare_matrix.py --out-dir demos/build_rvv --nr-lanes 4
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
from tilelang import tvm as tvm

# ``demos/rvv/`` is on sys.path when invoked as ``python demos/rvv/compile_compare_matrix.py``.
from matrix_report import (  # noqa: E402
    COVERAGE_SHORT,
    DECODE_CONTEXT_LENS,
    HIGHLIGHT_HEADS,
    PREFILL_SEQ_LENS,
    _build_rvv_tir,
    _rvv_stage,
)
from nanovllm.backends.tilelang.attention import (  # noqa: E402
    build_flash_attention_decode_kernel,
    build_flash_attention_prefill_kernel,
    tilelang_dtype,
)
from nanovllm.backends.tilelang.rvv_lower import (  # noqa: E402
    assert_target_vlen,
    rvv_target,
    time_rvv_lower,
    time_tilelang_lower,
)
from nanovllm.stages.attention import AttentionStage  # noqa: E402

_HOST_TARGET = tvm.target.Target("llvm")
_CUDA_TARGET = tvm.target.Target("cuda")


@dataclass
class CompileCompareRow:
    phase: str
    scenario: str
    lengths: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    host_x86_compile_ms: float | None
    rvv_compile_ms: float | None
    cuda_compile_ms: float | None
    cuda_nvcc_ms: float | None
    cuda_available: bool
    host_dtype: str
    rvv_dtype: str
    cuda_dtype: str
    host_ok: bool
    rvv_ok: bool
    cuda_ok: bool
    cuda_nvcc_ok: bool
    host_error: str
    rvv_error: str
    cuda_error: str
    cuda_nvcc_error: str

    def as_dict(self) -> dict:
        def _ms(v: float | None) -> str:
            return "" if v is None else f"{v:.1f}"

        return {
            "phase": self.phase,
            "scenario": self.scenario,
            "lengths": self.lengths,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "host_x86_compile_ms": _ms(self.host_x86_compile_ms),
            "rvv_compile_ms": _ms(self.rvv_compile_ms),
            "cuda_compile_ms": _ms(self.cuda_compile_ms),
            "cuda_nvcc_ms": _ms(self.cuda_nvcc_ms),
            "cuda_available": self.cuda_available,
            "host_dtype": self.host_dtype,
            "rvv_dtype": self.rvv_dtype,
            "cuda_dtype": self.cuda_dtype,
            "host_ok": self.host_ok,
            "rvv_ok": self.rvv_ok,
            "cuda_ok": self.cuda_ok,
            "cuda_nvcc_ok": self.cuda_nvcc_ok,
            "host_error": self.host_error,
            "rvv_error": self.rvv_error,
            "cuda_error": self.cuda_error,
            "cuda_nvcc_error": self.cuda_nvcc_error,
        }


def _host_stage(num_heads: int, num_kv_heads: int, head_dim: int) -> AttentionStage:
    return AttentionStage(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        tilelang_backend="cpu",
        seed=0,
    )


def _cuda_stage(num_heads: int, num_kv_heads: int, head_dim: int) -> AttentionStage:
    return AttentionStage(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
        tilelang_backend="cuda",
        seed=0,
    )


def _build_tir(
    stage: AttentionStage, phase: str, lengths: list[int], *, vec_exp2: bool
) -> object:
    from nanovllm.stages.attention import _round_up

    if phase == "decode":
        ctx = stage.prepare_decode_context(lengths)
        build_args = (
            ctx.batch_size,
            ctx.seqlen_kv_padded,
            stage.num_heads,
            stage.num_kv_heads,
            stage.head_dim,
            float(stage.softmax_scale),
            stage.decode_block_N,
            stage.decode_block_H,
            stage.decode_num_stages,
            stage.decode_threads,
            tilelang_dtype(stage.dtype),
            vec_exp2,
        )
        return build_flash_attention_decode_kernel.get_tir(*build_args)

    ctx = stage.prepare_prefill_context(lengths)
    padded_q = _round_up(ctx.total_q, stage.prefill_block_M)
    padded_kv = _round_up(ctx.total_kv, stage.prefill_block_N)
    build_args = (
        len(lengths),
        padded_q,
        padded_kv,
        stage.num_heads,
        stage.num_kv_heads,
        stage.head_dim,
        float(stage.softmax_scale),
        True,
        stage.prefill_block_M,
        stage.prefill_block_N,
        stage.prefill_num_stages,
        stage.prefill_threads,
        tilelang_dtype(stage.dtype),
        vec_exp2,
    )
    return build_flash_attention_prefill_kernel.get_tir(*build_args)


def _timing_result(timing) -> tuple[float, bool, str]:
    return timing.compile_ms, timing.compiled, timing.error or ""


def _time_host_pipeline(
    stage: AttentionStage, phase: str, lengths: list[int]
) -> tuple[float, bool, str]:
    try:
        tir = _build_tir(stage, phase, lengths, vec_exp2=True)
        return _timing_result(
            time_tilelang_lower(tir, _HOST_TARGET, codegen="host", target_host=_HOST_TARGET)
        )
    except Exception as exc:  # noqa: BLE001
        return 0.0, False, str(exc).strip().splitlines()[-1]


def _time_rvv_pipeline(
    stage: AttentionStage, phase: str, lengths: list[int], nr_lanes: int
) -> tuple[float, bool, str]:
    target = rvv_target(nr_lanes=nr_lanes)
    assert_target_vlen(target, nr_lanes)
    try:
        tir, _sig = _build_rvv_tir(stage, phase, lengths)
        return _timing_result(time_rvv_lower(tir, target))
    except Exception as exc:  # noqa: BLE001
        return 0.0, False, str(exc).strip().splitlines()[-1]


def _time_cuda_pipeline(
    stage: AttentionStage, phase: str, lengths: list[int], *, nvcc: bool
) -> tuple[float, bool, str]:
    codegen = "device_nvcc" if nvcc else "device_source"
    try:
        tir = _build_tir(stage, phase, lengths, vec_exp2=False)
        return _timing_result(time_tilelang_lower(tir, _CUDA_TARGET, codegen=codegen))
    except Exception as exc:  # noqa: BLE001
        return 0.0, False, str(exc).strip().splitlines()[-1]


def _iter_scenarios() -> list[tuple[str, list[int], str]]:
    rows: list[tuple[str, list[int], str]] = []
    for lengths, note in DECODE_CONTEXT_LENS:
        rows.append(("decode", lengths, note))
    for lengths, note in PREFILL_SEQ_LENS:
        rows.append(("prefill", lengths, note))
    return rows


def _gpu_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return f"{len(names)} GPU(s): " + ", ".join(names)


def run_compare(nr_lanes: int, *, cuda_enabled: bool) -> list[CompileCompareRow]:
    num_heads, num_kv_heads, head_dim = HIGHLIGHT_HEADS
    host_stage = _host_stage(num_heads, num_kv_heads, head_dim)
    rvv_stage = _rvv_stage(num_heads, num_kv_heads, head_dim, nr_lanes)
    cuda_stage = None
    if cuda_enabled:
        try:
            cuda_stage = _cuda_stage(num_heads, num_kv_heads, head_dim)
        except RuntimeError as exc:
            cuda_enabled = False
            print(f"CUDA unavailable: {exc}", file=sys.stderr)

    results: list[CompileCompareRow] = []
    for phase, lengths, note in _iter_scenarios():
        scenario = COVERAGE_SHORT.get(note, note)
        print(f"\n{phase} {scenario} {lengths}", flush=True)

        host_ms, host_ok, host_err = _time_host_pipeline(host_stage, phase, lengths)
        print(f"  host_x86: {host_ms:.0f} ms  ok={host_ok}", flush=True)

        rvv_ms, rvv_ok, rvv_err = _time_rvv_pipeline(rvv_stage, phase, lengths, nr_lanes)
        print(f"  rvv:      {rvv_ms:.0f} ms  ok={rvv_ok}", flush=True)

        cuda_ms: float | None = None
        cuda_nvcc_ms: float | None = None
        cuda_ok = False
        cuda_nvcc_ok = False
        cuda_err = ""
        cuda_nvcc_err = ""
        if cuda_enabled and cuda_stage is not None:
            cuda_ms, cuda_ok, cuda_err = _time_cuda_pipeline(
                cuda_stage, phase, lengths, nvcc=False
            )
            print(f"  cuda:     {cuda_ms:.0f} ms  ok={cuda_ok} (source, no nvcc)", flush=True)
            cuda_nvcc_ms, cuda_nvcc_ok, cuda_nvcc_err = _time_cuda_pipeline(
                cuda_stage, phase, lengths, nvcc=True
            )
            nvcc_note = "ok" if cuda_nvcc_ok else f"fail: {cuda_nvcc_err[:80]}"
            print(f"  cuda_nvcc: {cuda_nvcc_ms:.0f} ms  {nvcc_note}", flush=True)
        else:
            print("  cuda:     skipped (no GPU)", flush=True)

        results.append(
            CompileCompareRow(
                phase=phase,
                scenario=scenario,
                lengths=str(lengths),
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                host_x86_compile_ms=host_ms if host_ok else None,
                rvv_compile_ms=rvv_ms if rvv_ok else None,
                cuda_compile_ms=cuda_ms if cuda_ok else None,
                cuda_nvcc_ms=cuda_nvcc_ms if cuda_nvcc_ok else None,
                cuda_available=cuda_enabled,
                host_dtype="float32",
                rvv_dtype="float32",
                cuda_dtype="float16",
                host_ok=host_ok,
                rvv_ok=rvv_ok,
                cuda_ok=cuda_ok,
                cuda_nvcc_ok=cuda_nvcc_ok,
                host_error=host_err,
                rvv_error=rvv_err,
                cuda_error=cuda_err,
                cuda_nvcc_error=cuda_nvcc_err,
            )
        )
    return results


def _write_csv(path: str, rows: list[CompileCompareRow]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].as_dict().keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _write_markdown(path: str, rows: list[CompileCompareRow], nr_lanes: int) -> None:
    h, kv, d = HIGHLIGHT_HEADS
    lines = [
        "# Attention compile time comparison (8 scenarios)",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        f"Head config: **{h} / {kv} / {d}** (GQA 2) · RVV `nr_lanes={nr_lanes}` "
        f"(VLEN {1024 * nr_lanes} bits)",
        "",
        "All times are **cold TileLang pipeline** (`lower_to_host_device_ir` + codegen), "
        "not JIT cache hits and not kernel launch.",
        "",
        "Dtypes: host/rvv **fp32** (+ `vec_exp2` on CPU paths); cuda **fp16**.",
        "",
        "**CUDA** column = device source generation (`.cu`, no nvcc). "
        "**cuda_nvcc** = full nvcc/cubin (often blocked by host GCC &lt; C++17).",
        "",
        "| phase | scenario | host x86 (ms) | RVV (ms) | CUDA source (ms) | CUDA nvcc (ms) |",
        "|-------|----------|---------------|----------|------------------|----------------|",
    ]
    for r in rows:
        host = f"{r.host_x86_compile_ms:.0f}" if r.host_x86_compile_ms is not None else "—"
        rvv = f"{r.rvv_compile_ms:.0f}" if r.rvv_compile_ms is not None else "—"
        if r.cuda_available:
            cuda = f"{r.cuda_compile_ms:.0f}" if r.cuda_compile_ms is not None else "fail"
            nvcc = f"{r.cuda_nvcc_ms:.0f}" if r.cuda_nvcc_ms is not None else "fail"
        else:
            cuda = "n/a"
            nvcc = "n/a"
        lines.append(f"| {r.phase} | {r.scenario} | {host} | {rvv} | {cuda} | {nvcc} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="demos/build_rvv")
    parser.add_argument("--nr-lanes", type=int, default=4)
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Do not attempt CUDA compile even if a GPU is present.",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"GPU: {_gpu_summary()}")
    cuda_enabled = not args.skip_cuda and torch.cuda.is_available()
    if not cuda_enabled and not args.skip_cuda:
        print("CUDA not available; cuda columns will be empty.", file=sys.stderr)

    rows = run_compare(args.nr_lanes, cuda_enabled=cuda_enabled)
    csv_path = os.path.join(args.out_dir, "compile_compare.csv")
    md_path = os.path.join(args.out_dir, "COMPILE_COMPARE.md")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, args.nr_lanes)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    ok = all(r.host_ok and r.rvv_ok for r in rows)
    if cuda_enabled:
        ok = ok and all(r.cuda_ok for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
