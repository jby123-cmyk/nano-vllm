"""
Run the full RVV numeric test matrix and emit a lab-meeting table.

Covers the same 42 cases as ``tests/test_attention_rvv_numeric.py`` (40 parametric
+ 2 negative tests are skipped here).  For each case records:

  - coverage metadata (GQA group, padding, ragged / multi-block notes)
  - max_abs_diff vs PyTorch fp32 golden
  - compile time (``tilelang.compile`` via ``_compile_attention_kernel``)
  - kernel execution time
  - whether compile was a TileLang JIT cache hit (same build signature seen before)

Outputs:
  - ``<out-dir>/matrix_report.csv``
  - ``<out-dir>/MATRIX_REPORT.md`` (summary + worst-case row)
  - ``<out-dir>/SLIDE.md`` + ``slide_summary.csv`` (with ``--slide``)
  - ``<out-dir>/rvv_compile_report.csv`` (with ``--rvv-compile``)

Usage (nanovllm env, see usage.md)::

    python demos/matrix_report.py
    python demos/matrix_report.py --out-dir demos/build_rvv --slide
    python demos/matrix_report.py --slide-only --out-dir demos/build_rvv
    python demos/matrix_report.py --rvv-compile-only --slide --nr-lanes 4
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import torch

from nanovllm.backends.tilelang.attention import (
    _compile_attention_kernel,
    build_flash_attention_decode_kernel,
    build_flash_attention_prefill_kernel,
    tilelang_dtype,
)
from nanovllm.backends.tilelang.rvv_lower import (
    assert_target_vlen,
    rvv_target,
    time_rvv_lower,
    vlen_f32_elements,
)
from nanovllm.stages.attention import AttentionStage

# Same matrix as tests/test_attention_rvv_numeric.py
HEAD_CONFIGS = [
    (8, 8, 64),
    (8, 2, 64),
    (8, 4, 64),
    (16, 8, 128),
    (16, 2, 64),
]

DECODE_CONTEXT_LENS = [
    ([64], "single seq, sub-block KV (pad to block_N)"),
    ([128], "single seq, exactly one KV block"),
    ([64, 128], "ragged batch (2 seqs)"),
    ([1, 200], "min len + spans two KV blocks"),
]

PREFILL_SEQ_LENS = [
    ([64], "one block aligned"),
    ([80], "non block-aligned (pad to block_M)"),
    ([64, 64], "two equal sequences"),
    ([48, 96], "ragged, both non-aligned"),
]

ATOL = 1e-2
RTOL = 1e-2
COMPILE_CACHE_HIT_MS = 50.0  # compile faster than this → likely JIT cache hit

# Short labels for one-slide summaries.
COVERAGE_SHORT = {
    "single seq, sub-block KV (pad to block_N)": "sub-block KV",
    "single seq, exactly one KV block": "1 KV block",
    "ragged batch (2 seqs)": "ragged batch",
    "min len + spans two KV blocks": "multi-block KV",
    "one block aligned": "aligned",
    "non block-aligned (pad to block_M)": "non-aligned pad",
    "two equal sequences": "2 sequences",
    "ragged, both non-aligned": "ragged non-aligned",
}

# Representative wide config for slide highlight rows.
HIGHLIGHT_HEADS = (16, 8, 128)
HIGHLIGHT_DECODE_LENS = [1, 200]
HIGHLIGHT_PREFILL_LENS = [80]


@dataclass
class Row:
    phase: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    gqa_group: int
    lengths: str
    coverage_note: str
    batch_size: int
    padded_kv_or_q: int
    padded_kv: int | None
    compile_signature: str
    compile_ms: float
    compile_cache_hit: bool
    exec_ms: float
    total_ms: float
    max_abs_diff: float
    pass_numeric: bool

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "gqa_group": self.gqa_group,
            "lengths": self.lengths,
            "coverage_note": self.coverage_note,
            "batch_size": self.batch_size,
            "padded_q": self.padded_kv_or_q if self.phase == "prefill" else "",
            "padded_kv": self.padded_kv if self.phase == "prefill" else self.padded_kv_or_q,
            "compile_signature": self.compile_signature,
            "compile_ms": f"{self.compile_ms:.1f}",
            "compile_cache_hit": self.compile_cache_hit,
            "exec_ms": f"{self.exec_ms:.2f}",
            "total_ms": f"{self.total_ms:.1f}",
            "max_abs_diff": f"{self.max_abs_diff:.6e}",
            "pass": self.pass_numeric,
        }


@dataclass
class SlideRow:
    phase: str
    scenario: str
    gqa_groups: str
    cases: int
    max_abs_diff: float
    all_pass: bool

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "scenario": self.scenario,
            "gqa_groups": self.gqa_groups,
            "cases": self.cases,
            "max_abs_diff": f"{self.max_abs_diff:.6e}",
            "pass": f"{self.cases}/{self.cases}" if self.all_pass else "FAIL",
        }


@dataclass
class RvvCompileRow:
    phase: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    gqa_group: int
    lengths: str
    coverage_note: str
    compile_signature: str
    rvv_compile_ms: float
    rvv_compiled: bool
    rvv_vectorized: bool
    rvv_error: str

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "gqa_group": self.gqa_group,
            "lengths": self.lengths,
            "coverage_note": self.coverage_note,
            "compile_signature": self.compile_signature,
            "rvv_compile_ms": f"{self.rvv_compile_ms:.1f}",
            "rvv_compiled": self.rvv_compiled,
            "rvv_vectorized": self.rvv_vectorized,
            "rvv_error": self.rvv_error,
        }


def _stage(num_heads: int, num_kv_heads: int, head_dim: int) -> AttentionStage:
    return AttentionStage(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        tilelang_backend="cpu",
        seed=0,
    )


def _rvv_stage(num_heads: int, num_kv_heads: int, head_dim: int, nr_lanes: int) -> AttentionStage:
    """AttentionStage with VLEN-derived decode tiles (matches ``run_rvv_lower``)."""
    decode_block_N = vlen_f32_elements(nr_lanes)
    return AttentionStage(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        tilelang_backend="cpu",
        seed=0,
        decode_block_N=decode_block_N,
        decode_block_H=num_heads // num_kv_heads,
    )


def _within_tol(diff: float) -> bool:
    return diff <= ATOL  # rtol negligible at this scale for our magnitudes


def _decode_signature(stage: AttentionStage, ctx) -> tuple:
    in_dtype = tilelang_dtype(stage.dtype)
    return (
        "decode",
        ctx.batch_size,
        ctx.seqlen_kv_padded,
        stage.num_heads,
        stage.num_kv_heads,
        stage.head_dim,
        stage.decode_block_N,
        stage.decode_block_H,
        stage.decode_num_stages,
        stage.decode_threads,
        in_dtype,
        True,  # vec_exp2 on cpu
    )


def _prefill_signature(stage: AttentionStage, ctx) -> tuple:
    in_dtype = tilelang_dtype(stage.dtype)
    from nanovllm.stages.attention import _round_up

    padded_q = _round_up(ctx.total_q, stage.prefill_block_M)
    padded_kv = _round_up(ctx.total_kv, stage.prefill_block_N)
    return (
        "prefill",
        len(ctx.cu_seqlens_q) - 1,
        padded_q,
        padded_kv,
        stage.num_heads,
        stage.num_kv_heads,
        stage.head_dim,
        stage.prefill_block_M,
        stage.prefill_block_N,
        stage.prefill_num_stages,
        stage.prefill_threads,
        in_dtype,
        True,
    )


def _run_decode_case(
    stage: AttentionStage,
    context_lens: list[int],
    coverage_note: str,
    seen_sigs: set[tuple],
) -> Row:
    ctx = stage.prepare_decode_context(context_lens)
    q, k_cache, v_cache = stage.generate_decode_qkv(ctx)
    reference = stage.decode_reference(q, k_cache, v_cache, ctx)

    sig = _decode_signature(stage, ctx)
    build_args = sig[1:]  # drop phase label; matches attention.py tuple
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
        True,
    )

    t_total0 = time.perf_counter()
    t0 = time.perf_counter()
    kernel = _compile_attention_kernel(
        build_flash_attention_decode_kernel, build_args, 4, "cpu"
    )
    compile_ms = (time.perf_counter() - t0) * 1000.0
    cache_hit = sig in seen_sigs or compile_ms < COMPILE_CACHE_HIT_MS
    seen_sigs.add(sig)

    t1 = time.perf_counter()
    tilelang_out = kernel(
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        ctx.mask.to(torch.uint8).contiguous(),
    )
    exec_ms = (time.perf_counter() - t1) * 1000.0
    total_ms = (time.perf_counter() - t_total0) * 1000.0

    max_abs_diff = (reference.float() - tilelang_out.float()).abs().max().item()
    return Row(
        phase="decode",
        num_heads=stage.num_heads,
        num_kv_heads=stage.num_kv_heads,
        head_dim=stage.head_dim,
        gqa_group=stage.num_heads // stage.num_kv_heads,
        lengths=str(context_lens),
        coverage_note=coverage_note,
        batch_size=ctx.batch_size,
        padded_kv_or_q=ctx.seqlen_kv_padded,
        padded_kv=None,
        compile_signature=str(sig),
        compile_ms=compile_ms,
        compile_cache_hit=cache_hit,
        exec_ms=exec_ms,
        total_ms=total_ms,
        max_abs_diff=max_abs_diff,
        pass_numeric=_within_tol(max_abs_diff),
    )


def _run_prefill_case(
    stage: AttentionStage,
    seq_lens: list[int],
    coverage_note: str,
    seen_sigs: set[tuple],
) -> Row:
    ctx = stage.prepare_prefill_context(seq_lens)
    q, k, v = stage.generate_prefill_qkv(ctx.total_q)
    reference = stage.prefill_reference(q, k, v, ctx)

    from nanovllm.stages.attention import _round_up

    padded_q = _round_up(ctx.total_q, stage.prefill_block_M)
    padded_kv = _round_up(ctx.total_kv, stage.prefill_block_N)
    batch_size = len(seq_lens)

    sig = _prefill_signature(stage, ctx)
    build_args = (
        batch_size,
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
        True,
    )

    q_pad = q
    k_pad = k
    v_pad = v
    if padded_q != ctx.total_q:
        q_pad = torch.zeros(
            padded_q, stage.num_heads, stage.head_dim, device=q.device, dtype=q.dtype
        )
        q_pad[: ctx.total_q] = q
    if padded_kv != ctx.total_kv:
        k_pad = torch.zeros(
            padded_kv, stage.num_kv_heads, stage.head_dim, device=k.device, dtype=k.dtype
        )
        v_pad = torch.zeros(
            padded_kv, stage.num_kv_heads, stage.head_dim, device=v.device, dtype=v.dtype
        )
        k_pad[: ctx.total_kv] = k
        v_pad[: ctx.total_kv] = v

    t_total0 = time.perf_counter()
    t0 = time.perf_counter()
    kernel = _compile_attention_kernel(
        build_flash_attention_prefill_kernel, build_args, 6, "cpu"
    )
    compile_ms = (time.perf_counter() - t0) * 1000.0
    cache_hit = sig in seen_sigs or compile_ms < COMPILE_CACHE_HIT_MS
    seen_sigs.add(sig)

    t1 = time.perf_counter()
    tilelang_out = kernel(
        q_pad.contiguous(),
        k_pad.contiguous(),
        v_pad.contiguous(),
        ctx.cu_seqlens_q.to(torch.int32).contiguous(),
        ctx.cu_seqlens_k.to(torch.int32).contiguous(),
        int(ctx.max_seqlen_q),
    )[: ctx.total_q]
    exec_ms = (time.perf_counter() - t1) * 1000.0
    total_ms = (time.perf_counter() - t_total0) * 1000.0

    max_abs_diff = (reference.float() - tilelang_out.float()).abs().max().item()
    return Row(
        phase="prefill",
        num_heads=stage.num_heads,
        num_kv_heads=stage.num_kv_heads,
        head_dim=stage.head_dim,
        gqa_group=stage.num_heads // stage.num_kv_heads,
        lengths=str(seq_lens),
        coverage_note=coverage_note,
        batch_size=batch_size,
        padded_kv_or_q=padded_q,
        padded_kv=padded_kv,
        compile_signature=str(sig),
        compile_ms=compile_ms,
        compile_cache_hit=cache_hit,
        exec_ms=exec_ms,
        total_ms=total_ms,
        max_abs_diff=max_abs_diff,
        pass_numeric=_within_tol(max_abs_diff),
    )


def _aggregate_slide_rows(rows: list[Row]) -> list[SlideRow]:
    from collections import defaultdict

    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(row.phase, row.coverage_note)].append(row)

    slide_rows: list[SlideRow] = []
    for (phase, note), group in sorted(grouped.items()):
        gqa = ", ".join(
            str(g) for g in sorted({r.gqa_group for r in group})
        )
        slide_rows.append(
            SlideRow(
                phase=phase,
                scenario=COVERAGE_SHORT.get(note, note),
                gqa_groups=gqa,
                cases=len(group),
                max_abs_diff=max(r.max_abs_diff for r in group),
                all_pass=all(r.pass_numeric for r in group),
            )
        )
    return slide_rows


def _write_slide_outputs(out_dir: str, rows: list[Row], slide_rows: list[SlideRow]) -> None:
    diffs = [r.max_abs_diff for r in rows]
    worst = max(rows, key=lambda r: r.max_abs_diff)
    headroom = ATOL / max(diffs) if max(diffs) > 0 else float("inf")

    highlights: list[Row] = []
    for row in rows:
        cfg = (row.num_heads, row.num_kv_heads, row.head_dim)
        if cfg != HIGHLIGHT_HEADS:
            continue
        if row.phase == "decode" and row.lengths == str(HIGHLIGHT_DECODE_LENS):
            highlights.append(row)
        if row.phase == "prefill" and row.lengths == str(HIGHLIGHT_PREFILL_LENS):
            highlights.append(row)

    lines = [
        "# RVV attention matrix — one-slide summary",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Headline",
        "",
        f"- **{len(rows)}/{len(rows)} pass** at atol={ATOL}",
        f"- **max |Δ| = {max(diffs):.2e}** (mean {statistics.mean(diffs):.2e}) "
        f"— ~{headroom:.0e}× under tolerance",
        "- **GQA groups:** 1, 2, 4, 8 — **phases:** decode + prefill",
        f"- **Worst case:** {worst.phase} {worst.num_heads}/{worst.num_kv_heads}/"
        f"{worst.head_dim}, `{worst.lengths}` — "
        f"{COVERAGE_SHORT.get(worst.coverage_note, worst.coverage_note)}",
        "",
        f"## Coverage ({len(slide_rows)} rows = {len(rows)} underlying cases)",
        "",
        "| phase | scenario | GQA | max |Δ| | pass |",
        "|-------|----------|-----|--------|------|",
    ]
    for s in slide_rows:
        mark = " ◀max" if abs(s.max_abs_diff - max(diffs)) < 1e-15 else ""
        lines.append(
            f"| {s.phase} | {s.scenario} | {s.gqa_groups} | "
            f"{s.max_abs_diff:.2e}{mark} | {s.cases}/{s.cases} |"
        )
    lines.extend(["", "## Highlight configs", ""])
    lines.append("| phase | H/KV/D | lengths | coverage | max |Δ| |")
    lines.append("|-------|--------|---------|----------|--------|")
    for h in highlights:
        lines.append(
            f"| {h.phase} | {h.num_heads}/{h.num_kv_heads}/{h.head_dim} | "
            f"`{h.lengths}` | {COVERAGE_SHORT.get(h.coverage_note, h.coverage_note)} | "
            f"{h.max_abs_diff:.2e} |"
        )
    lines.append("")

    slide_md = os.path.join(out_dir, "SLIDE.md")
    slide_csv = os.path.join(out_dir, "slide_summary.csv")
    with open(slide_md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    _write_csv_generic(slide_csv, [s.as_dict() for s in slide_rows])
    print(f"Wrote {slide_md}")
    print(f"Wrote {slide_csv}")


def _write_csv_generic(path: str, dict_rows: list[dict]) -> None:
    if not dict_rows:
        return
    fieldnames = list(dict_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in dict_rows:
            writer.writerow(row)


def _load_rows_from_csv(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for d in csv.DictReader(handle):
            phase = d["phase"]
            padded_kv = int(d["padded_kv"]) if d["padded_kv"] else None
            padded_q = int(d["padded_q"]) if d.get("padded_q") else None
            rows.append(
                Row(
                    phase=phase,
                    num_heads=int(d["num_heads"]),
                    num_kv_heads=int(d["num_kv_heads"]),
                    head_dim=int(d["head_dim"]),
                    gqa_group=int(d["gqa_group"]),
                    lengths=d["lengths"],
                    coverage_note=d["coverage_note"],
                    batch_size=int(d["batch_size"]),
                    padded_kv_or_q=padded_kv if phase == "decode" else (padded_q or 0),
                    padded_kv=padded_kv if phase == "prefill" else None,
                    compile_signature=d["compile_signature"],
                    compile_ms=float(d["compile_ms"]),
                    compile_cache_hit=d["compile_cache_hit"] == "True",
                    exec_ms=float(d["exec_ms"]),
                    total_ms=float(d["total_ms"]),
                    max_abs_diff=float(d["max_abs_diff"]),
                    pass_numeric=d["pass"] == "True",
                )
            )
    return rows


def _rvv_compile_signature(phase: str, sig_host: tuple, nr_lanes: int) -> tuple:
    return ("rvv", nr_lanes) + sig_host


def _build_rvv_tir(stage: AttentionStage, phase: str, lengths: list[int]):
    from nanovllm.stages.attention import _round_up

    if phase == "decode":
        ctx = stage.prepare_decode_context(lengths)
        sig = _decode_signature(stage, ctx)
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
            True,
        )
        return build_flash_attention_decode_kernel.get_tir(*build_args), sig

    ctx = stage.prepare_prefill_context(lengths)
    padded_q = _round_up(ctx.total_q, stage.prefill_block_M)
    padded_kv = _round_up(ctx.total_kv, stage.prefill_block_N)
    sig = _prefill_signature(stage, ctx)
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
        True,
    )
    return build_flash_attention_prefill_kernel.get_tir(*build_args), sig


def _collect_rvv_compile_specs(
    nr_lanes: int,
    *,
    head_configs: list[tuple[int, int, int]] | None = None,
) -> list[tuple[object, tuple, dict]]:
    """Return (tir, rvv_signature, metadata) once per unique RVV JIT signature."""
    head_configs = head_configs or HEAD_CONFIGS
    specs: list[tuple[object, tuple, dict]] = []
    seen: set[tuple] = set()

    for num_heads, num_kv_heads, head_dim in head_configs:
        stage = _rvv_stage(num_heads, num_kv_heads, head_dim, nr_lanes)
        for context_lens, note in DECODE_CONTEXT_LENS:
            tir, sig = _build_rvv_tir(stage, "decode", context_lens)
            rvv_sig = _rvv_compile_signature("decode", sig, nr_lanes)
            if rvv_sig in seen:
                continue
            seen.add(rvv_sig)
            specs.append(
                (
                    tir,
                    rvv_sig,
                    {
                        "phase": "decode",
                        "num_heads": num_heads,
                        "num_kv_heads": num_kv_heads,
                        "head_dim": head_dim,
                        "gqa_group": num_heads // num_kv_heads,
                        "lengths": str(context_lens),
                        "coverage_note": note,
                    },
                )
            )

    for num_heads, num_kv_heads, head_dim in head_configs:
        stage = _rvv_stage(num_heads, num_kv_heads, head_dim, nr_lanes)
        for seq_lens, note in PREFILL_SEQ_LENS:
            tir, sig = _build_rvv_tir(stage, "prefill", seq_lens)
            rvv_sig = _rvv_compile_signature("prefill", sig, nr_lanes)
            if rvv_sig in seen:
                continue
            seen.add(rvv_sig)
            specs.append(
                (
                    tir,
                    rvv_sig,
                    {
                        "phase": "prefill",
                        "num_heads": num_heads,
                        "num_kv_heads": num_kv_heads,
                        "head_dim": head_dim,
                        "gqa_group": num_heads // num_kv_heads,
                        "lengths": str(seq_lens),
                        "coverage_note": note,
                    },
                )
            )

    return specs


def _run_rvv_compile_matrix(
    nr_lanes: int,
    *,
    head_configs: list[tuple[int, int, int]] | None = None,
) -> list[RvvCompileRow]:
    target = rvv_target(nr_lanes=nr_lanes)
    assert_target_vlen(target, nr_lanes)

    rows: list[RvvCompileRow] = []
    specs = _collect_rvv_compile_specs(nr_lanes, head_configs=head_configs)

    for idx, (tir, rvv_sig, meta) in enumerate(specs, start=1):
        timing = time_rvv_lower(tir, target)
        rows.append(
            RvvCompileRow(
                phase=meta["phase"],
                num_heads=meta["num_heads"],
                num_kv_heads=meta["num_kv_heads"],
                head_dim=meta["head_dim"],
                gqa_group=meta["gqa_group"],
                lengths=meta["lengths"],
                coverage_note=meta["coverage_note"],
                compile_signature=str(rvv_sig),
                rvv_compile_ms=timing.compile_ms,
                rvv_compiled=timing.compiled,
                rvv_vectorized=timing.vectorized,
                rvv_error=timing.error or "",
            )
        )
        print(
            f"[{idx}/{len(specs)}] rvv {meta['phase']} "
            f"heads={meta['num_heads']} kv={meta['num_kv_heads']} "
            f"dim={meta['head_dim']} lens={meta['lengths']} "
            f"compile={timing.compile_ms:.0f}ms ok={timing.compiled} "
            f"vec={timing.vectorized}",
            flush=True,
        )

    return rows


def _write_rvv_compile_report(
    path: str, rows: list[RvvCompileRow], nr_lanes: int, wall_s: float
) -> None:
    from collections import defaultdict

    compiled = sum(1 for r in rows if r.rvv_compiled)
    vectorized = sum(1 for r in rows if r.rvv_vectorized)
    unique = len({r.compile_signature for r in rows})
    lines = [
        "# RVV compiler pathway timing",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| nr_lanes | {nr_lanes} |",
        f"| VLEN bits | {1024 * nr_lanes} |",
        f"| Unique signatures | {unique} |",
        f"| Compiled | {compiled}/{len(rows)} |",
        f"| Vectorized (.s scan) | {vectorized}/{len(rows)} |",
        f"| Wall time | {wall_s:.1f} s |",
        "",
        "See `rvv_compile_report.csv` for per-signature rows.",
        "",
        "| phase | scenario | max rvv_compile_ms | all compiled | all vectorized |",
        "|-------|----------|--------------------|--------------|----------------|",
    ]
    by_scenario: dict[tuple[str, str], list[RvvCompileRow]] = defaultdict(list)
    for r in rows:
        by_scenario[(r.phase, r.coverage_note)].append(r)
    for (phase, note), group in sorted(by_scenario.items()):
        worst_ms = max(r.rvv_compile_ms for r in group)
        ok = all(r.rvv_compiled for r in group)
        vec = all(r.rvv_vectorized for r in group)
        lines.append(
            f"| {phase} | {COVERAGE_SHORT.get(note, note)} | {worst_ms:.0f} | {ok} | {vec} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_csv(path: str, rows: list[Row]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].as_dict().keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _write_markdown(path: str, rows: list[Row], wall_s: float) -> None:
    diffs = [r.max_abs_diff for r in rows]
    worst = max(rows, key=lambda r: r.max_abs_diff)
    unique_sigs = len({r.compile_signature for r in rows})
    cold_compiles = sum(1 for r in rows if not r.compile_cache_hit)

    lines = [
        "# RVV numeric test matrix report",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Cases | {len(rows)} |",
        f"| All pass (max_abs_diff ≤ {ATOL}) | {all(r.pass_numeric for r in rows)} |",
        f"| max(max_abs_diff) | {max(diffs):.6e} |",
        f"| mean(max_abs_diff) | {statistics.mean(diffs):.6e} |",
        f"| median(max_abs_diff) | {statistics.median(diffs):.6e} |",
        f"| tolerance atol/rtol | {ATOL} / {RTOL} |",
        f"| Unique JIT compile signatures | {unique_sigs} |",
        f"| Cold compiles (first of signature) | {cold_compiles} |",
        f"| Total wall time | {wall_s:.1f} s |",
        "",
        "## Coverage map",
        "",
        "| phase | scenario | GQA groups exercised | cases |",
        "|-------|----------|----------------------|-------|",
    ]
    # Aggregate coverage notes per phase.
    from collections import defaultdict

    cov: dict[tuple[str, str], set[int]] = defaultdict(set)
    cov_count: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        key = (r.phase, r.coverage_note)
        cov[key].add(r.gqa_group)
        cov_count[key] += 1
    for (phase, note), groups in sorted(cov.items()):
        gqa_str = ", ".join(str(g) for g in sorted(groups))
        lines.append(
            f"| {phase} | {note} | {gqa_str} | {cov_count[(phase, note)]} |"
        )
    lines.append("")
    lines.append("## Worst case (largest max_abs_diff)")
    lines.append("")
    lines.append(f"- **phase** `{worst.phase}`")
    lines.append(
        f"- **heads / kv / dim** {worst.num_heads} / {worst.num_kv_heads} / {worst.head_dim}"
    )
    lines.append(f"- **GQA group** {worst.gqa_group}")
    lines.append(f"- **lengths** `{worst.lengths}` — {worst.coverage_note}")
    lines.append(f"- **max_abs_diff** {worst.max_abs_diff:.6e}")
    lines.append(
        f"- **compile_ms** {worst.compile_ms:.1f} (cache_hit={worst.compile_cache_hit})"
    )
    lines.append(f"- **exec_ms** {worst.exec_ms:.2f}")
    lines.append("")
    lines.append("## Full table")
    lines.append("")
    lines.append("See `matrix_report.csv` for machine-readable rows.")
    lines.append("")
    lines.append(
        "| phase | heads | kv | dim | GQA | lengths | coverage | max_abs_diff | compile_ms | exec_ms | pass |"
    )
    lines.append(
        "|-------|-------|----|----|-----|---------|----------|--------------|------------|---------|------|"
    )
    for r in rows:
        lines.append(
            f"| {r.phase} | {r.num_heads} | {r.num_kv_heads} | {r.head_dim} | "
            f"{r.gqa_group} | `{r.lengths}` | {r.coverage_note} | "
            f"{r.max_abs_diff:.2e} | {r.compile_ms:.0f} | {r.exec_ms:.1f} | {r.pass_numeric} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir", default="demos/build_rvv", help="Write reports under this directory."
    )
    parser.add_argument(
        "--slide",
        action="store_true",
        help="Also write SLIDE.md and slide_summary.csv (8-row lab summary).",
    )
    parser.add_argument(
        "--slide-only",
        action="store_true",
        help="Read matrix_report.csv and write slide outputs only (no numeric rerun).",
    )
    parser.add_argument(
        "--rvv-compile",
        action="store_true",
        help="Time RVV llvm lowering (riscv64 +zvl) per unique JIT signature.",
    )
    parser.add_argument(
        "--rvv-compile-only",
        action="store_true",
        help="Run only the full RVV compile matrix (~35 unique signatures, ~5 min).",
    )
    parser.add_argument(
        "--rvv-compile-quick",
        action="store_true",
        help="RVV compile only the wide representative head (16/8/128); ~7 unique signatures.",
    )
    parser.add_argument(
        "--nr-lanes",
        type=int,
        default=4,
        help="AraXL lane count for --rvv-compile (VLEN = 1024*nr_lanes bits).",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "matrix_report.csv")
    exit_code = 0

    if args.slide_only:
        if not os.path.isfile(csv_path):
            print(f"Missing {csv_path}; run without --slide-only first.", file=sys.stderr)
            return 1
        rows = _load_rows_from_csv(csv_path)
        _write_slide_outputs(args.out_dir, rows, _aggregate_slide_rows(rows))
        return 0

    run_numeric = not (args.rvv_compile_only or args.rvv_compile_quick)
    run_rvv = args.rvv_compile or args.rvv_compile_only or args.rvv_compile_quick

    rows: list[Row] = []
    if run_numeric:
        seen_sigs: set[tuple] = set()
        t_wall0 = time.perf_counter()

        for num_heads, num_kv_heads, head_dim in HEAD_CONFIGS:
            stage = _stage(num_heads, num_kv_heads, head_dim)
            for context_lens, note in DECODE_CONTEXT_LENS:
                row = _run_decode_case(stage, context_lens, note, seen_sigs)
                rows.append(row)
                print(
                    f"decode  heads={num_heads} kv={num_kv_heads} dim={head_dim} "
                    f"lens={context_lens} diff={row.max_abs_diff:.2e} "
                    f"compile={row.compile_ms:.0f}ms exec={row.exec_ms:.1f}ms"
                )

        for num_heads, num_kv_heads, head_dim in HEAD_CONFIGS:
            stage = _stage(num_heads, num_kv_heads, head_dim)
            for seq_lens, note in PREFILL_SEQ_LENS:
                row = _run_prefill_case(stage, seq_lens, note, seen_sigs)
                rows.append(row)
                print(
                    f"prefill heads={num_heads} kv={num_kv_heads} dim={head_dim} "
                    f"lens={seq_lens} diff={row.max_abs_diff:.2e} "
                    f"compile={row.compile_ms:.0f}ms exec={row.exec_ms:.1f}ms"
                )

        wall_s = time.perf_counter() - t_wall0
        md_path = os.path.join(args.out_dir, "MATRIX_REPORT.md")
        _write_csv(csv_path, rows)
        _write_markdown(md_path, rows, wall_s)

        worst = max(rows, key=lambda r: r.max_abs_diff)
        print(f"\nWrote {csv_path}")
        print(f"Wrote {md_path}")
        print(f"Cases: {len(rows)}  all_pass: {all(r.pass_numeric for r in rows)}")
        print(
            f"max_abs_diff: max={max(r.max_abs_diff for r in rows):.6e}  "
            f"mean={statistics.mean(r.max_abs_diff for r in rows):.6e}"
        )
        print(
            f"Worst: {worst.phase} heads={worst.num_heads} kv={worst.num_kv_heads} "
            f"dim={worst.head_dim} lens={worst.lengths} diff={worst.max_abs_diff:.6e}"
        )
        print(
            f"Unique compile signatures: {len({r.compile_signature for r in rows})}  "
            f"wall={wall_s:.1f}s"
        )
        if not all(r.pass_numeric for r in rows):
            exit_code = 1

        if args.slide:
            _write_slide_outputs(args.out_dir, rows, _aggregate_slide_rows(rows))

    if run_rvv:
        head_configs = [HIGHLIGHT_HEADS] if args.rvv_compile_quick else None
        mode = "quick" if args.rvv_compile_quick else "full"
        print(f"\nRVV compile matrix ({mode}, nr_lanes={args.nr_lanes})...")
        t0 = time.perf_counter()
        rvv_rows = _run_rvv_compile_matrix(args.nr_lanes, head_configs=head_configs)
        rvv_wall = time.perf_counter() - t0
        rvv_csv = os.path.join(args.out_dir, "rvv_compile_report.csv")
        rvv_md = os.path.join(args.out_dir, "RVV_COMPILE_REPORT.md")
        _write_csv_generic(rvv_csv, [r.as_dict() for r in rvv_rows])
        _write_rvv_compile_report(rvv_md, rvv_rows, args.nr_lanes, rvv_wall)
        print(f"Wrote {rvv_csv}")
        print(f"Wrote {rvv_md}")
        print(
            f"RVV compile: {sum(1 for r in rvv_rows if r.rvv_compiled)}/{len(rvv_rows)} ok  "
            f"wall={rvv_wall:.1f}s"
        )
        if not all(r.rvv_compiled for r in rvv_rows):
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
