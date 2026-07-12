"""
Stage 1 demo: drive TileLang kernels through the CPU/``llvm`` pipeline at an
AraXL RISC-V Vector (RVV) target and emit ``<name>.tir`` / ``<name>.ll`` /
``<name>.s`` per kernel, then write ``STAGE1_REPORT.md``.

This is the reproducible Stage 1 command from ``usage.md``. It runs a
kernel ladder from trivial to the real thing:

  (a) elementwise add          — unit-stride vector copy/add
  (b) T.gemm matmul            — GemmVector parallel-N vfmacc nest
  (c) FlashAttention decode    — the real decode kernel (fp32)
  (d) FlashAttention prefill   — the real prefill kernel (fp32)

CPU/``llvm`` pass pipeline (see ``tilelang/cpu/pipeline.py``):

  Simplify → … → SerializeOuterParallel → LayoutInference → LowerTileOp
  → VectorizeLoop → LLVM codegen → RVV assembly

  ``SerializeOuterParallel`` (Phase 4) runs before ``LayoutInference`` and
  restructures multi-axis ``T.Parallel`` elementwise nests for CPU vectorization:
  outer parallel axes become serial, loop-invariant loads are hoisted, and the
  innermost axis becomes the vector lane. GPU pipelines are untouched.

Attention kernels use the natural ``T.Parallel(M, N)`` form (no CPU-specific
loop reshaping in the kernel source). Vectorization comes from the compiler:

  - Phase 2: inline Cephes ``2**x`` polynomial replaces scalar ``exp2f`` (libm)
  - Phase 3: ``acc_s`` feeds ``P @ V`` directly when ``in_dtype == accum_dtype``
  - Phase 4: ``SerializeOuterParallel`` vectorizes softmax exponent, ``acc_o``
    rescale, final divide, and prefill's fp32 mask

GEMM (``GemmVector`` → ``vfmacc``) and reduce (``vle`` + ``vfredmax``/``vfredusum``)
were already vector and are unchanged by Phases 2–4.

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
    DEFAULT_NR_LANES,
    LowerResult,
    assert_target_vlen,
    build_elementwise_kernel,
    build_matmul_kernel,
    lower_kernel_rvv,
    numeric_check_elementwise,
    numeric_check_matmul,
    numeric_check_attention_decode,
    numeric_check_attention_prefill,
    rvv_target,
    vlen_f32_elements,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="demos/build_rvv", help="Artifact root (default demos/build_rvv).")
    p.add_argument("--mtriple", default=DEFAULT_MTRIPLE, help="LLVM target triple.")
    p.add_argument("--mattr", default=",".join(DEFAULT_MATTR), help="Comma-separated LLVM attrs.")
    p.add_argument("--mabi", default=DEFAULT_MABI, help="RISC-V ABI.")
    p.add_argument(
        "--nr-lanes",
        type=int,
        default=DEFAULT_NR_LANES,
        help="AraXL lane count; sole source of VLEN = 1024*nr_lanes bits "
        "(default 4 → VLEN 4096, block_N 128; 8 → VLEN 8192, block_N 256).",
    )
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
        "\nThis report records the **demo** described in `usage.md`: TileLang "
        "kernels routed through the CPU/`llvm` pass pipeline at the AraXL RVV target. "
        "The pipeline includes `SerializeOuterParallel` (Phase 4), which restructures "
        "multi-axis `T.Parallel` elementwise nests before `LayoutInference` so LLVM "
        "can vectorize softmax exponent, rescale, divide, and fp32 mask loops. "
        "GEMM (`GemmVector` → `vfmacc`) and reduce (`vle` + `vfred`) were already "
        "vector and are unchanged. LLVM still owns final VL/LMUL scheduling (no "
        "custom subtarget). Remaining limitations are **observed in the artifacts**, "
        "not hidden.\n"
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
        a(f"- Strided/indexed ops (supported by target sim; informational): `{', '.join(r.strided_ops) or 'none'}`")
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
        "1. **GemmVector for all transpose forms.** `T.gemm` lowers via "
        "`GemmVector` on `llvm`: a `serial i,k` + `parallel j` nest with the "
        "output-`N` axis vectorized (`vfmacc` on the accumulator), including the "
        "transposed `Q @ K^T` in FlashAttention (its `B[j,k]` load becomes a "
        "strided/gather vector load, e.g. `vluxei64`). The target ISA sim "
        "supports strided/indexed loads, so this keeps the score matmul fully "
        "vectorized rather than scalarizing it. "
        + (
            f"Observed: `matmul` vectorized to `{', '.join(gemm.vector_ops[:6])}…`; "
            "LLVM still owns final VL/LMUL selection, so this is not a hand-tuned "
            "long-vector micro-kernel."
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
        "4. **No custom cost/scheduling model.** The `.s` uses generic RVV "
        "scheduling/LMUL; there is no target-specific LLVM subtarget yet."
    )
    L.append(
        "5. **Strided/indexed loads (informational, not a blocker).** The target "
        "ISA sim supports `vlse`/`vluxei`, so these are kept when they preserve "
        "vectorization (e.g. transposed `Q @ K^T`). "
        + (
            f"Observed strided/indexed ops in the artifacts: "
            f"`{', '.join(strided_seen)}` — these keep the corresponding loops "
            "vectorized rather than falling back to scalar element loops."
            if strided_seen
            else "No strided/indexed ops were emitted for these shapes."
        )
    )
    L.append(
        "6. **Softmax `exp2` — vectorized (Phase 2, done).** There is no hardware "
        "transcendental, so the baseline emitted a scalar `call exp2f` (libm) per "
        "element and could not vectorize the softmax loop. The kernels now inline "
        "a Cephes `2**x` polynomial (arithmetic + fp32 exponent bit-trick), so LLVM "
        "emits vector ops (`vfcvt`, `vfmadd`, `vsll.vi`) with **zero `exp2f` "
        "calls**."
    )
    L.append(
        "6b. **Elementwise fragment loops — vectorized (Phase 4, done).** The CPU "
        "pass `SerializeOuterParallel` serializes the outer axis of flattened 2-D "
        "`T.Parallel` nests and hoists loop-invariant loads, so the softmax "
        "exponent, `acc_o` rescale, final divide, and prefill's fp32 mask vectorize "
        "generically (no attention-specific edge case). The decode `uint8`-compare "
        "mask stays scalar — an LLVM mixed-element-width limitation."
    )
    L.append(
        "7. **fp16 not in baseline.** `+f,+d` cover fp32/fp64; fp16 needs `zvfh`. "
        "All kernels here are fp32. (The attention kernels' fp16 tensor-core path "
        "is GPU-only.)"
    )
    L.append(
        "8. **Vector-register capacity / spills.** Tiles exceeding the vector "
        "register file spill to memory. Stage 1 does no capacity-aware tiling; "
        "inspect each `.s` for stack spills around the vector loops."
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
            "emit `.tir`/`.ll`/`.s` with RVV ops (`vfmacc`, `vfredmax`/`vfredusum`, "
            "`vfcvt`/`vfmadd`/`vfmul`, `vsetvli`, …). Phase 2 inlined the softmax "
            "`exp2` polynomial (zero `exp2f` calls); Phase 3 drops the redundant "
            "`acc_s_cast` copy on the fp32 path; Phase 4's `SerializeOuterParallel` "
            "pass vectorizes the elementwise fragment loops generically. Decode "
            "`block_N` is VLEN-derived (`vlen_f32_elements(nr_lanes)`); prefill "
            "tiles remain CUDA-shaped (64×64). "
            f"{numeric_note} Run `demos/run_attention_rvv_stage.py` for a standalone "
            "golden check (mirrors `run_attention_stage.py` on CPU)."
        )
    elif decode and not decode.compiled:
        L.append(
            "\n**Attention (decode/prefill) status.** Lowering failed in "
            f"`{decode.error_pass or 'unknown'}` — see `usage.md` and `background.md`; "
            "rebuild TileLang from `/mnt/ssd/jby123/tilelang/build`."
        )
    return "\n".join(f"{item}\n" for item in L)


def main() -> int:
    args = parse_args()
    target = rvv_target(args.mtriple, args.mattr, args.mabi, nr_lanes=args.nr_lanes)
    target_str = str(target)
    scale = args.head_dim ** -0.5

    # Fail loudly if +zvl / nr_lanes desynced (was a silent footgun).
    vlen_bits = assert_target_vlen(target, args.nr_lanes)

    # Derive decode tiles from target VLEN (Phase 1): block_N = fp32 lanes per
    # LMUL=1 register so reduce/GEMM fill one vector; block_H = GQA group size.
    decode_block_N = vlen_f32_elements(args.nr_lanes)
    decode_block_H = args.num_heads // args.num_kv_heads
    # Decode requires seqlen_kv % block_N == 0; pad the ladder KV length up.
    decode_seqlen_kv = ((args.seqlen_kv + decode_block_N - 1) // decode_block_N) * decode_block_N

    print(f"RVV Stage 1 lowering — target: {target_str}")
    print(f"VLEN: {vlen_bits} bits ({vlen_bits // 32} fp32/cluster, nr_lanes={args.nr_lanes})")
    print(
        f"Decode tiles (VLEN-derived): block_N={decode_block_N} block_H={decode_block_H} "
        f"seqlen_kv={decode_seqlen_kv}"
    )
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
        title="T.gemm matmul (GemmVector + wide VLEN copy), fp32",
        metadata={"m": args.gemm_m, "n": args.gemm_n, "k": args.gemm_k, "dtype": "float32"},
    )
    r.numeric = "n/a" if args.no_numeric else numeric_check_matmul(
        args.gemm_m, args.gemm_n, args.gemm_k, build_dir=r.build_dir
    )
    results.append(r)
    print(f"  [b] matmul      : compiled={r.compiled} vectorized={r.vectorized} numeric={r.numeric}")

    # (c) FlashAttention decode (fp32) — VLEN-derived tiles, vec_exp2 poly,
    # SerializeOuterParallel vectorizes elementwise loops at compile time.
    decode_tir = build_flash_attention_decode_kernel.get_tir(
        1, decode_seqlen_kv, args.num_heads, args.num_kv_heads, args.head_dim,
        float(scale), decode_block_N, decode_block_H, 2, 128, "float32", True,
    )
    r = lower_kernel_rvv(
        decode_tir, "attention_decode", os.path.join(args.out_dir, "attention_decode"), target,
        title="FlashAttention decode (GQA KV-cache), fp32",
        metadata={
            "phase": "decode", "batch_size": 1, "seqlen_kv": decode_seqlen_kv,
            "num_heads": args.num_heads, "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim, "dtype": "float32",
            "block_N": decode_block_N, "block_H": decode_block_H, "vec_exp2": True,
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
            seqlen_kv=decode_seqlen_kv,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_N=decode_block_N,
            block_H=decode_block_H,
            build_dir=r.build_dir,
        )
    results.append(r)
    print(f"  [c] decode      : compiled={r.compiled} vectorized={r.vectorized} numeric={r.numeric}")

    # (d) FlashAttention prefill (fp32) — CUDA-shaped 64×64 tiles (not VLEN-derived);
    # vec_exp2 poly + SerializeOuterParallel (prefill fp32 mask vectorizes).
    prefill_tir = build_flash_attention_prefill_kernel.get_tir(
        1, args.prefill_total_q, args.seqlen_kv, args.num_heads, args.num_kv_heads, args.head_dim,
        float(scale), True, 64, 64, 1, 128, "float32", True,
    )
    r = lower_kernel_rvv(
        prefill_tir, "attention_prefill", os.path.join(args.out_dir, "attention_prefill"), target,
        title="FlashAttention prefill (varlen causal GQA), fp32",
        metadata={
            "phase": "prefill", "batch_size": 1, "total_q": args.prefill_total_q,
            "total_kv": args.seqlen_kv,
            "num_heads": args.num_heads, "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim, "dtype": "float32", "vec_exp2": True,
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
