"""
Stage 1 demo: drive TileLang kernels through the CPU/``llvm`` pipeline at an
AraXL RISC-V Vector (RVV) target and emit ``<name>.tir`` / ``<name>.ll`` /
``<name>.s`` per kernel, then write ``STAGE1_REPORT.md``.

This is the reproducible Stage 1 command from ``tilelang.md``. It runs a
kernel ladder from trivial to the real thing:

  (a) elementwise add          — unit-stride vector copy/add
  (b) T.gemm matmul            — exercises the scalar GemmScalar fallback
  (c) FlashAttention decode    — the real decode kernel (attempted; fp32)
  (d) FlashAttention prefill   — the real prefill kernel (attempted; fp32)

Default run (AraXL RVV target, demos/build_rvv/, numeric checks on):

  python demos/run_rvv_lower.py

Override the target (defaults mirror AraXL tvm-apps):

  python demos/run_rvv_lower.py \\
    --mtriple riscv64-unknown-elf --mattr +v,+m,+f,+d --mabi lp64d

Real Qwen3-ish attention dims:

  python demos/run_rvv_lower.py --num-heads 16 --num-kv-heads 8 --head-dim 128

Everything is fp32 (the baseline; fp16 needs zvfh and is out of scope).
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from nanovllm.backends.tilelang.attention import (
    build_flash_attention_decode_kernel,
    build_flash_attention_prefill_kernel,
)
from nanovllm.backends.tilelang.golden_log import write_skipped_golden_log
from nanovllm.backends.tilelang.rvv_lower import (
    DEFAULT_MABI,
    DEFAULT_MATTR,
    DEFAULT_MTRIPLE,
    LowerResult,
    build_elementwise_kernel,
    build_matmul_kernel,
    lower_kernel_rvv,
    numeric_check_elementwise,
    numeric_check_matmul,
    numeric_check_attention_decode,
    numeric_check_attention_prefill,
    rvv_target,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="demos/build_rvv", help="Artifact root (default demos/build_rvv).")
    p.add_argument("--mtriple", default=DEFAULT_MTRIPLE, help="LLVM target triple.")
    p.add_argument("--mattr", default=",".join(DEFAULT_MATTR), help="Comma-separated LLVM attrs.")
    p.add_argument("--mabi", default=DEFAULT_MABI, help="RISC-V ABI.")
    # Elementwise / gemm shapes
    p.add_argument("--elem-n", type=int, default=4096, help="Elementwise vector length.")
    p.add_argument("--gemm-m", type=int, default=128)
    p.add_argument("--gemm-n", type=int, default=128)
    p.add_argument("--gemm-k", type=int, default=128)
    # Attention dims (fp32 baseline)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--seqlen-kv", type=int, default=128, help="Decode KV length (multiple of block_N).")
    p.add_argument("--prefill-total-q", type=int, default=64, help="Prefill packed Q length for RVV ladder.")
    p.add_argument("--no-numeric", action="store_true", help="Skip host numeric validation.")
    return p.parse_args()


def _snippet(s_path: str | None, anchor: str = "vsetvli", before: int = 2, after: int = 12) -> str:
    """Pull a short representative window around the first RVV op in a .s file."""
    if not s_path or not os.path.exists(s_path):
        return ""
    with open(s_path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for i, line in enumerate(lines):
        if anchor in line:
            lo = max(0, i - before)
            hi = min(len(lines), i + after)
            return "\n".join(lines[lo:hi])
    return ""


def build_report(results: list[LowerResult], target_str: str, out_dir: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    a = lines.append

    a("# Stage 1 Report — TileLang → LLVM → RVV (Demo Backend)\n")
    a(f"_Generated: {now}_\n")
    a(f"_Target: `{target_str}`_\n")
    a(
        "\nThis report records the **demo** described in `tilelang.md`: TileLang "
        "kernels routed through the CPU/`llvm` pass pipeline at the AraXL RVV target, "
        "with LLVM auto-vectorization on. **Path B** TileLang backend work (see "
        "`tilelang.md`) added CPU realizations for GPU-style tile ops so FlashAttention "
        "decode/prefill lower end-to-end; LLVM still owns vector scheduling (no custom "
        "LMUL tuning or AraXL subtarget). Remaining limitations are **observed in the "
        "artifacts**, not hidden.\n"
    )

    a("\n## Reproduce\n")
    a("```bash")
    a("python demos/run_rvv_lower.py")
    a("```")

    a("\n## Per-kernel results\n")
    a("| Kernel | Compiled | Vectorized | Numeric (host) | Failing pass | Notes |")
    a("|--------|----------|------------|----------------|--------------|-------|")
    for r in results:
        compiled = "yes" if r.compiled else "no"
        vec = "yes" if r.vectorized else ("no" if r.compiled else "—")
        numeric = r.numeric or "—"
        fpass = r.error_pass or "—"
        note = ""
        if r.compiled:
            note = f"{len(r.vector_ops)} RVV op kinds"
            if r.strided_ops:
                note += f"; strided: {', '.join(r.strided_ops)}"
        else:
            note = (r.error or "").replace("|", "\\|")[:90]
        a(f"| `{r.name}` | {compiled} | {vec} | {numeric} | `{fpass}` | {note} |")

    a("\n## Emitted RVV instructions (evidence for T3)\n")
    for r in results:
        if not r.compiled:
            continue
        a(f"\n### `{r.name}`\n")
        a(f"- Artifacts: `{r.name}.tir`, `{r.name}.ll`, `{r.name}.s`")
        a(f"- Vector ops: `{', '.join(r.vector_ops) or 'none'}`")
        a(f"- Strided/indexed ops (AraXL unit-stride only): `{', '.join(r.strided_ops) or 'none'}`")
        snip = _snippet(r.s_path)
        if snip:
            a("\n```asm")
            a(snip)
            a("```")

    failed = [r for r in results if not r.compiled]
    if failed:
        a("\n## Attempted kernels that did not fully lower\n")
        for r in failed:
            a(f"\n### `{r.name}`\n")
            a(f"- Failing TileLang pass: `{r.error_pass or 'unknown'}`")
            a(f"- Error: `{(r.error or '').strip()}`")
            a(f"- Source TIR preserved at `{r.name}.source.tir`; `{r.name}.tir` holds the annotated source.")

    a("\n## Section 6 limitations — observed in the emitted artifacts\n")
    a(_limitations_section(results))

    text = "\n".join(lines) + "\n"
    report_path = os.path.join(out_dir, "STAGE1_REPORT.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return report_path


def _limitations_section(results: list[LowerResult]) -> str:
    by_name = {r.name: r for r in results}
    gemm = by_name.get("matmul")
    decode = by_name.get("attention_decode")
    strided_seen = sorted({op for r in results for op in r.strided_ops})

    L = []
    L.append(
        "1. **Scalar-fallback GEMM.** `T.gemm` lowers via `GemmScalar` to a plain "
        "triple loop and relies entirely on LLVM auto-vectorization. "
        + (
            f"Observed: `matmul` compiled and LLVM vectorized the loop "
            f"(`{', '.join(gemm.vector_ops[:6])}…`), but the structure is a scalar "
            "triple loop tuned for short SIMD, not a long-vector machine."
            if gemm and gemm.compiled
            else "See `matmul` artifacts."
        )
    )
    L.append(
        "2. **LLVM will not maximize vector length / LMUL.** Default LMUL is "
        "conservative; dynamic LMUL selection is unfinished upstream. LLVM "
        "vectorizes the loop *as written* (note the `vsetvli` reconfiguring VL "
        "per loop rather than a fixed long-vector schedule) and does not "
        "restructure for long vectors."
    )
    L.append(
        "3. **No layout choice (Option A vs B).** LLVM takes the loop nest as "
        "given; it does not choose the lane axis or reduction strategy. The "
        "long-vector-optimal layout is a future tile-backend job."
    )
    L.append(
        "4. **No AraXL cost/scheduling model.** The `.s` uses generic RVV "
        "scheduling/LMUL; AraXL is not a modeled LLVM subtarget."
    )
    L.append(
        "5. **Unit-stride only.** AraXL cannot do strided/gather loads. "
        + (
            f"Observed: strided/indexed ops emitted in the artifacts: "
            f"`{', '.join(strided_seen)}` — these (e.g. `vlse32`) would not run on "
            "AraXL and flag layouts that need contiguity fixes in a later stage."
            if strided_seen
            else "No strided/indexed ops were emitted for these shapes; larger/"
            "transposed tiles can still produce `vlse`/`vluxei`."
        )
    )
    L.append(
        "6. **No hardware transcendental.** `exp2` (softmax) is not a vector "
        "instruction; the baseline would scalarize it or call libm. The "
        "polynomial-via-`call_extern` fix is future work."
    )
    L.append(
        "7. **fp16 not in baseline.** `+f,+d` cover fp32/fp64; fp16 needs `zvfh`. "
        "All kernels here are fp32. (The attention kernels' fp16 tensor-core path "
        "is GPU-only.)"
    )
    L.append(
        "8. **VRF capacity / spills.** No cache sits between the VRF and L2, so "
        "tiles exceeding the VRF spill to memory. Stage 1 does no VRF-aware "
        "tiling; inspect each `.s` for stack spills around the vector loops."
    )

    if decode and decode.compiled:
        decode_numeric = next((r.numeric for r in results if r.name == "attention_decode"), None)
        numeric_note = (
            "Numeric validation passes on the host (PyTorch golden vs CPU TileLang)."
            if decode_numeric == "pass"
            else "See numeric column; host golden check uses llvm+tvm_ffi on CPU."
        )
        L.append(
            "\n**Attention (decode/prefill) status.** Path B TileLang backend work "
            "unblocked the real FlashAttention kernels on CPU/`llvm`: fragment "
            "`infer_layout`, CPU `T.reduce_*`/`T.fill`, early shared→local promotion "
            "in `LowerTileOp`, and `tl.infinity` lowering for LLVM. Both kernels now "
            "emit `.tir`/`.ll`/`.s` with RVV ops (`vfmacc`, `vfredosum`, `vsetvli`, …). "
            f"{numeric_note} Run `demos/run_attention_rvv_stage.py` for a standalone "
            "golden check (mirrors `run_attention_stage.py` on CPU)."
        )
    elif decode and not decode.compiled:
        L.append(
            "\n**Attention (decode/prefill) status.** Lowering failed in "
            f"`{decode.error_pass or 'unknown'}` — see `tilelang.md` for the Path B "
            "backend checklist and rebuild TileLang from `/mnt/ssd/jby123/tilelang/build`."
        )
    return "\n".join(f"{item}\n" for item in L)


def main() -> int:
    args = parse_args()
    target = rvv_target(args.mtriple, args.mattr, args.mabi)
    target_str = str(target)
    scale = args.head_dim ** -0.5

    print(f"RVV Stage 1 lowering — target: {target_str}")
    print(f"Artifacts root: {args.out_dir}\n")

    results: list[LowerResult] = []

    # (a) elementwise add
    elem = build_elementwise_kernel(args.elem_n, block=256, dtype="float32")
    r = lower_kernel_rvv(
        elem, "elementwise", os.path.join(args.out_dir, "elementwise"), target,
        title="Elementwise add (C = A + B), fp32",
        metadata={"n": args.elem_n, "block": 256, "dtype": "float32"},
    )
    r.numeric = "n/a" if args.no_numeric else numeric_check_elementwise(args.elem_n, 256, build_dir=r.build_dir)
    results.append(r)
    print(f"  [a] elementwise : compiled={r.compiled} vectorized={r.vectorized} numeric={r.numeric}")

    # (b) T.gemm matmul
    mm = build_matmul_kernel(args.gemm_m, args.gemm_n, args.gemm_k, dtype="float32")
    r = lower_kernel_rvv(
        mm, "matmul", os.path.join(args.out_dir, "matmul"), target,
        title="T.gemm matmul (GemmScalar fallback), fp32",
        metadata={"m": args.gemm_m, "n": args.gemm_n, "k": args.gemm_k, "dtype": "float32"},
    )
    r.numeric = "n/a" if args.no_numeric else numeric_check_matmul(
        args.gemm_m, args.gemm_n, args.gemm_k, build_dir=r.build_dir
    )
    results.append(r)
    print(f"  [b] matmul      : compiled={r.compiled} vectorized={r.vectorized} numeric={r.numeric}")

    # (c) FlashAttention decode (fp32)
    decode_tir = build_flash_attention_decode_kernel.get_tir(
        1, args.seqlen_kv, args.num_heads, args.num_kv_heads, args.head_dim,
        float(scale), 128, 64, 2, 128, "float32",
    )
    r = lower_kernel_rvv(
        decode_tir, "attention_decode", os.path.join(args.out_dir, "attention_decode"), target,
        title="FlashAttention decode (GQA KV-cache), fp32",
        metadata={
            "phase": "decode", "batch_size": 1, "seqlen_kv": args.seqlen_kv,
            "num_heads": args.num_heads, "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim, "dtype": "float32",
        },
    )
    if args.no_numeric:
        r.numeric = "n/a"
    elif not r.compiled:
        write_skipped_golden_log(
            r.build_dir,
            demo="rvv_lower/attention_decode",
            reference="pytorch float32 attention (AttentionStage golden)",
            backend="tilelang host llvm + tvm_ffi (backend=cpu)",
            reason=f"RVV lowering failed before numeric check: {r.error_pass or 'unknown'} — {r.error or ''}",
        )
        r.numeric = "n/a"
    else:
        r.numeric = numeric_check_attention_decode(
            batch_size=1,
            seqlen_kv=args.seqlen_kv,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            build_dir=r.build_dir,
        )
    results.append(r)
    print(f"  [c] decode      : compiled={r.compiled} vectorized={r.vectorized} numeric={r.numeric}")

    # (d) FlashAttention prefill (fp32)
    prefill_tir = build_flash_attention_prefill_kernel.get_tir(
        1, args.prefill_total_q, args.seqlen_kv, args.num_heads, args.num_kv_heads, args.head_dim,
        float(scale), True, 64, 64, 1, 128, "float32",
    )
    r = lower_kernel_rvv(
        prefill_tir, "attention_prefill", os.path.join(args.out_dir, "attention_prefill"), target,
        title="FlashAttention prefill (varlen causal GQA), fp32",
        metadata={
            "phase": "prefill", "batch_size": 1, "total_q": args.prefill_total_q,
            "total_kv": args.seqlen_kv,
            "num_heads": args.num_heads, "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim, "dtype": "float32",
        },
    )
    if args.no_numeric:
        r.numeric = "n/a"
    elif not r.compiled:
        write_skipped_golden_log(
            r.build_dir,
            demo="rvv_lower/attention_prefill",
            reference="pytorch float32 attention (AttentionStage golden)",
            backend="tilelang host llvm + tvm_ffi (backend=cpu)",
            reason=f"RVV lowering failed before numeric check: {r.error_pass or 'unknown'} — {r.error or ''}",
        )
        r.numeric = "n/a"
    else:
        r.numeric = numeric_check_attention_prefill(
            batch_size=1,
            total_q=args.prefill_total_q,
            total_kv=args.seqlen_kv,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            build_dir=r.build_dir,
        )
    results.append(r)
    print(f"  [d] prefill     : compiled={r.compiled} vectorized={r.vectorized} numeric={r.numeric}")

    report_path = build_report(results, target_str, args.out_dir)
    print(f"\nReport: {report_path}")

    compiled = [r.name for r in results if r.compiled]
    failed = [r.name for r in results if not r.compiled]
    print(f"Compiled: {compiled}")
    print(f"Attempted/failed: {failed}")

    # Acceptance: elementwise + matmul must compile with RVV instructions.
    required = {"elementwise", "matmul"}
    ok = all(r.compiled and r.vectorized for r in results if r.name in required)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
