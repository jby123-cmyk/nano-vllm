"""
Stage 1 RVV lowering harness (DEMO backend).

Drive an existing TileLang ``@T.prim_func`` through TileLang's CPU/``llvm`` pass
pipeline configured for a RISC-V Vector (RVV) target, with LLVM
auto-vectorization enabled, and emit the artifact triple per kernel:

  * ``<name>.tir`` — lowered TensorIR (CPU pipeline output); falls back to the
                     source TIR (annotated) when lowering fails.
  * ``<name>.ll``  — LLVM IR for the RVV target.
  * ``<name>.s``   — RVV assembly.

This is the RVV artifact harness described in ``tilelang.md`` and
``attention.md`` §10. It drives existing TileLang ``@T.prim_func`` kernels
through TileLang's CPU/``llvm`` pass pipeline (with Path B CPU tile-op support)
configured for a RISC-V Vector target, with LLVM auto-vectorization enabled.

Path B extended TileLang so GPU-style fragments, shared tiles, ``T.gemm``,
``T.reduce_*``, and ``tl.infinity`` lower on CPU/``llvm``. LLVM still performs
vector scheduling — this harness does not implement custom LMUL tuning or
AraXL-specific codegen. Remaining limitations are documented in
``STAGE1_REPORT.md`` §6 and ``tilelang.md`` §6.

The RVV target mirrors AraXL's TVM flow
(``AraXL/tvm-apps/kernels/common/common.py``): ``llvm`` backend,
``riscv64-unknown-elf`` triple, ``+v,+m,+f,+d`` attrs, ``lp64d`` ABI. fp16 needs
``zvfh`` and is intentionally out of this baseline — use fp32.

Do NOT add ``from __future__ import annotations`` to this file — it breaks
TileLang's ``@T.prim_func`` type parsing.
"""

import json
import os
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone

import torch
import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.engine.lower import host_codegen, lower_to_host_device_ir

from nanovllm.backends.tilelang.golden_log import (
    compare_and_log,
    log_numeric_exception,
)

# --------------------------------------------------------------------------- #
# RVV target (mirror AraXL tvm-apps/kernels/common/common.py)
# --------------------------------------------------------------------------- #
DEFAULT_MTRIPLE = "riscv64-unknown-elf"
# VLEN = 1024 bits × nr_lanes; default 4 lanes → +zvl4096b (AraXL).
DEFAULT_NR_LANES = 4
DEFAULT_MATTR = (
    "+v",
    "+m",
    "+f",
    "+d",
    f"+zvl{1024 * DEFAULT_NR_LANES}b",
)  # +v=RVV1.0, +m=mul/div, +f/+d=fp32/fp64
DEFAULT_MABI = "lp64d"

# RVV instruction families used to prove vectorization actually happened (T3).
_RVV_VECTOR_RE = re.compile(
    r"\b(vset[i]?vli|vle\d+\.?v?|vse\d+\.?v?|vlm\.v|vsm\.v|"
    r"vf[a-z]+\.[vwf]+|vfmacc|vfmadd|vfredusum|vfredosum|vfredmax|vfredmin|"
    r"vfmv|vmv|vadd|vmul|vfmul|vfadd|vfsub|vfdiv|vfmax|vfmin|vrgather)\b"
)
# Strided / indexed memory ops AraXL CANNOT execute (unit-stride only,
# tilelang.md §6 limitation #5). Recorded as evidence in the report.
_RVV_STRIDED_RE = re.compile(r"\b(vlse\d+|vsse\d+|vl[ou]xei\d+|vs[ou]xei\d+)\b")


def rvv_target(
    mtriple: str = DEFAULT_MTRIPLE,
    mattr=DEFAULT_MATTR,
    mabi: str = DEFAULT_MABI,
    nr_lanes: int = DEFAULT_NR_LANES,
) -> tvm.target.Target:
    """Build the AraXL RVV ``llvm`` target. ``mattr`` may be a list/tuple/str.

    ``nr_lanes`` sets ``+zvl{1024*nr_lanes}b`` when no ``+zvl`` entry is present.
    Pass an explicit ``mattr`` list to override VLEN entirely.
    """
    if isinstance(mattr, str):
        mattr = [tok.strip() for tok in mattr.split(",") if tok.strip()]
    else:
        mattr = list(mattr)
    vlen_bits = 1024 * nr_lanes
    if not any(tok.startswith("+zvl") for tok in mattr):
        mattr = [*mattr, f"+zvl{vlen_bits}b"]
    return tvm.target.Target(
        {
            "kind": "llvm",
            "mtriple": mtriple,
            "mattr": mattr,
            "mabi": mabi,
            "vector-width": vlen_bits,
        }
    )


def scan_rvv_ops(asm: str) -> tuple[list[str], list[str]]:
    """Return (sorted unique vector ops, sorted unique strided/indexed ops)."""
    vector = sorted({m.group(0) for m in _RVV_VECTOR_RE.finditer(asm)})
    strided = sorted({m.group(0) for m in _RVV_STRIDED_RE.finditer(asm)})
    return vector, strided


@dataclass
class LowerResult:
    name: str
    title: str
    build_dir: str
    compiled: bool
    tir_path: str
    ll_path: str | None = None
    s_path: str | None = None
    error: str | None = None
    error_pass: str | None = None
    vector_ops: list[str] = field(default_factory=list)
    strided_ops: list[str] = field(default_factory=list)
    numeric: str | None = None  # "pass" / "fail: ..." / "n/a"

    @property
    def vectorized(self) -> bool:
        return bool(self.vector_ops)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _short_error(exc: Exception) -> tuple[str, str | None]:
    """Return a one-line error message and the failing TileLang pass, if any."""
    msg = str(exc).strip().splitlines()
    one_line = msg[-1] if msg else repr(exc)
    tb = traceback.format_exc()
    pass_match = re.findall(r"tilelang\.transform\.(\w+)\(\)", tb)
    cpu_pass = pass_match[-1] if pass_match else None
    return one_line, cpu_pass


def lower_kernel_rvv(
    func,
    name: str,
    build_dir: str,
    target: tvm.target.Target,
    title: str = "",
    metadata: dict | None = None,
) -> LowerResult:
    """Lower ``func`` (a source ``@T.prim_func``) through the CPU/llvm pipeline
    at ``target`` and emit ``<name>.tir`` / ``<name>.ll`` / ``<name>.s`` plus
    metadata.json + README.txt under ``build_dir``.

    Robust by design: the source TIR is always captured, and any failure in the
    CPU pipeline / codegen is recorded (so the attention kernels can be
    *attempted* and their failures documented, per the Stage 1 plan).
    """
    os.makedirs(build_dir, exist_ok=True)
    title = title or name
    source_tir = func.script()

    tir_path = os.path.join(build_dir, f"{name}.tir")
    ll_path = os.path.join(build_dir, f"{name}.ll")
    s_path = os.path.join(build_dir, f"{name}.s")

    result = LowerResult(
        name=name, title=title, build_dir=build_dir, compiled=False, tir_path=tir_path
    )

    try:
        with target:
            host_mod, device_mod, _params, tgt, tgt_host = lower_to_host_device_ir(
                func, target=target, target_host=target
            )
            # <name>.tir == lowered TensorIR (CPU pipeline output).
            _write_text(tir_path, host_mod.script())

            rt_mod = host_codegen(host_mod, target_host=tgt_host, target=tgt)
            llvm_ir = rt_mod.inspect_source("ll")
            asm = rt_mod.inspect_source("asm")

        _write_text(ll_path, llvm_ir)
        _write_text(s_path, asm)
        result.vector_ops, result.strided_ops = scan_rvv_ops(asm)
        result.compiled = True
        result.ll_path = ll_path
        result.s_path = s_path
    except Exception as exc:  # noqa: BLE001 — Stage 1 attempts every kernel
        one_line, cpu_pass = _short_error(exc)
        result.error = one_line
        result.error_pass = cpu_pass
        # No lowered TIR available; keep the source TIR so <name>.tir exists.
        annotated = (
            f"# NOTE: CPU/llvm (RVV) lowering FAILED for this kernel.\n"
            f"# Failing TileLang pass: {cpu_pass or '<unknown>'}\n"
            f"# Error: {one_line}\n"
            f"# The body below is the SOURCE TensorIR (pre-pipeline), not the\n"
            f"# lowered output. See metadata.json / STAGE1_REPORT.md for details.\n\n"
            f"{source_tir}\n"
        )
        _write_text(tir_path, annotated)

    # Always keep the pristine source TIR alongside the (lowered or annotated) one.
    _write_text(os.path.join(build_dir, f"{name}.source.tir"), source_tir)

    meta = {
        "name": name,
        "title": title,
        "target": str(target),
        "mtriple": target.attrs.get("mtriple"),
        "mattr": list(target.attrs.get("mattr", [])),
        "mabi": target.attrs.get("mabi"),
        "compiled": result.compiled,
        "vectorized": result.vectorized,
        "vector_ops": result.vector_ops,
        "strided_ops": result.strided_ops,
        "error": result.error,
        "error_pass": result.error_pass,
        "llvm_enabled": bool(tvm.runtime.enabled("llvm")),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        meta.update(metadata)
    _write_text(
        os.path.join(build_dir, "metadata.json"),
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
    )

    status = "compiled" if result.compiled else f"FAILED ({result.error_pass})"
    readme = f"""{title}
RVV demo build artifacts — Stage 1 (TileLang -> LLVM -> RVV)
Generated UTC: {meta['generated_utc']}
Target: {meta['target']}
Status: {status}

Files:
  {name}.tir         Lowered TensorIR (CPU/llvm pipeline output).
                     If lowering failed, this holds the annotated SOURCE TIR.
  {name}.source.tir  Pristine source TensorIR from the @T.prim_func.
  {name}.ll          LLVM IR for the RVV target (only if lowering succeeded).
  {name}.s           RVV assembly (only if lowering succeeded).
  metadata.json      Target string, shapes/dtype, vector-op scan, errors.
  golden.log         PyTorch golden comparison (when numeric checks run).

Vectorization (RVV ops detected in {name}.s): {result.vector_ops or 'none'}
Strided/indexed ops (AraXL is unit-stride only): {result.strided_ops or 'none'}

This is a DEMO of what TileLang + LLVM already do today. It performs no custom
vector lowering, LMUL tuning, or AraXL-specific codegen. See STAGE1_REPORT.md
for the observed limitations.
"""
    _write_text(os.path.join(build_dir, "README.txt"), readme)
    return result


# --------------------------------------------------------------------------- #
# Kernel ladder (a) elementwise/copy  and  (b) T.gemm matmul (GemmScalar)
# --------------------------------------------------------------------------- #
def build_elementwise_kernel(n: int, block: int = 256, dtype: str = "float32"):
    """(a) Elementwise: C = A + B. Exercises T.copy-style unit-stride vector
    loads/stores; should auto-vectorize cleanly to vle/vse/vfadd."""

    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
        C: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block), threads=1) as bx:
            for i in T.serial(block):
                idx = bx * block + i
                if idx < n:
                    C[idx] = A[idx] + B[idx]

    return main


def build_matmul_kernel(
    m: int,
    n: int,
    k: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    dtype: str = "float32",
    accum_dtype: str = "float32",
):
    """(b) T.gemm matmul. Uses GemmVector on llvm (i,k + parallel N) and wide
    RVV copy vectorization; GemmScalar on plain ``c`` targets."""

    @T.prim_func
    def main(
        A: T.Tensor((m, k), dtype),
        B: T.Tensor((k, n), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m)) as (bx, by):
            A_local = T.alloc_local((block_m, block_k), dtype)
            B_local = T.alloc_local((block_k, block_n), dtype)
            C_local = T.alloc_local((block_m, block_n), accum_dtype)
            T.clear(C_local)
            for ko in T.serial(T.ceildiv(k, block_k)):
                T.copy(A[by * block_m, ko * block_k], A_local)
                T.copy(B[ko * block_k, bx * block_n], B_local)
                T.gemm(A_local, B_local, C_local)
            T.copy(C_local, C[by * block_m, bx * block_n])

    return main


# --------------------------------------------------------------------------- #
# Numeric validation (T5) — host execution via the CPU backend
# --------------------------------------------------------------------------- #
# The RVV target itself is not natively runnable on an x86 host; bit-exact RVV
# validation requires Spike/Verilator (AraXL tvm-apps). To still gate kernel
# *correctness* we compile the SAME @T.prim_func on the host and compare against
# the PyTorch float32 golden from ``AttentionStage``:
#
#   * elementwise / matmul — native CPU ``c`` backend + cython (fast, simple ops)
#   * attention decode/prefill — host ``llvm`` + tvm_ffi (same CPU tile-op path
#     as RVV: fragments, GemmScalar, skipped software pipeline, tl.infinity)
#
# The RVV ``.s`` is the identical kernel retargeted to riscv64; host execution
# validates the kernel math and lowering plumbing.
def numeric_check_elementwise(
    n: int, block: int = 256, build_dir: str | None = None
) -> str:
    demo = "rvv_lower/elementwise"
    reference = "torch (a + b)"
    backend = "tilelang cpu/c + cython"
    details = {"n": n, "block": block, "dtype": "float32"}
    try:
        func = build_elementwise_kernel(n, block, "float32")
        with tvm.target.Target("c"):
            kernel = tilelang.compile(
                func, out_idx=[2], target="c", target_host="c", execution_backend="cython"
            )
        a = torch.randn(n, dtype=torch.float32)
        b = torch.randn(n, dtype=torch.float32)
        c = kernel(a, b)
        expected = a + b
        if build_dir:
            _, passed = compare_and_log(
                build_dir,
                demo=demo,
                reference=reference,
                backend=backend,
                tilelang_tensor=c,
                reference_tensor=expected,
                atol=1e-5,
                rtol=1e-5,
                details=details,
                raise_on_fail=False,
            )
            return "pass" if passed else "fail: see golden.log"
        torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
        return "pass"
    except Exception as exc:  # noqa: BLE001
        if build_dir:
            log_numeric_exception(
                build_dir, demo=demo, reference=reference, backend=backend, exc=exc, details=details
            )
        return f"fail: {str(exc).strip().splitlines()[-1]}"


def numeric_check_matmul(
    m: int,
    n: int,
    k: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    build_dir: str | None = None,
) -> str:
    demo = "rvv_lower/matmul"
    reference = "torch (a @ b)"
    backend = "tilelang cpu/c + cython"
    details = {"m": m, "n": n, "k": k, "dtype": "float32"}
    try:
        func = build_matmul_kernel(m, n, k, block_m, block_n, block_k, "float32", "float32")
        with tvm.target.Target("c"):
            kernel = tilelang.compile(
                func, out_idx=[2], target="c", target_host="c", execution_backend="cython"
            )
        a = torch.randn(m, k, dtype=torch.float32)
        b = torch.randn(k, n, dtype=torch.float32)
        c = kernel(a, b)
        expected = a @ b
        if build_dir:
            _, passed = compare_and_log(
                build_dir,
                demo=demo,
                reference=reference,
                backend=backend,
                tilelang_tensor=c,
                reference_tensor=expected,
                atol=1e-2,
                rtol=1e-2,
                details=details,
                raise_on_fail=False,
            )
            return "pass" if passed else "fail: see golden.log"
        torch.testing.assert_close(c, expected, atol=1e-2, rtol=1e-2)
        return "pass"
    except Exception as exc:  # noqa: BLE001
        if build_dir:
            log_numeric_exception(
                build_dir, demo=demo, reference=reference, backend=backend, exc=exc, details=details
            )
        return f"fail: {str(exc).strip().splitlines()[-1]}"


def _attention_numeric_check(
    phase: str,
    lengths: list[int],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    stage_kwargs: dict,
    seed: int = 0,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    build_dir: str | None = None,
) -> str:
    """Compare CPU TileLang ``phase`` vs the PyTorch golden in ``AttentionStage``.

    Returns ``"pass"`` or ``"fail: <reason>"`` so the ladder never aborts.
    """
    from nanovllm.stages.attention import AttentionStage

    demo = f"rvv_lower/attention_{phase}"
    reference = "pytorch float32 attention (AttentionStage golden)"
    backend = "tilelang host llvm + tvm_ffi (backend=cpu)"
    details = {
        "phase": phase,
        "lengths": lengths,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "seed": seed,
    }
    try:
        stage = AttentionStage(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.float32,
            device="cpu",
            seed=seed,
            tilelang_backend="cpu",
            **stage_kwargs,
        )
        result = stage.run_prefill(lengths) if phase == "prefill" else stage.run_decode(lengths)
        details["softmax_scale"] = stage.softmax_scale
        if build_dir:
            _, passed = compare_and_log(
                build_dir,
                demo=demo,
                reference=reference,
                backend=backend,
                tilelang_tensor=result.tilelang_output,
                reference_tensor=result.reference_output,
                atol=atol,
                rtol=rtol,
                max_abs_diff=result.max_abs_diff,
                details=details,
                raise_on_fail=False,
            )
            return "pass" if passed else "fail: see golden.log"
        torch.testing.assert_close(
            result.tilelang_output.float(),
            result.reference_output.float(),
            atol=atol,
            rtol=rtol,
        )
        return "pass"
    except Exception as exc:  # noqa: BLE001
        if build_dir:
            log_numeric_exception(
                build_dir, demo=demo, reference=reference, backend=backend, exc=exc, details=details
            )
        return f"fail: {str(exc).strip().splitlines()[-1]}"


def numeric_check_attention_decode(
    batch_size: int = 1,
    seqlen_kv: int = 128,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    block_N: int = 128,
    block_H: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    context_lens: list[int] | None = None,
    seed: int = 0,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    build_dir: str | None = None,
) -> str:
    """Compare CPU TileLang decode vs PyTorch golden (same shapes as RVV ladder)."""
    return _attention_numeric_check(
        "decode",
        context_lens if context_lens is not None else [seqlen_kv] * batch_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        stage_kwargs=dict(
            decode_block_N=block_N,
            decode_block_H=block_H,
            decode_num_stages=num_stages,
            decode_threads=threads,
        ),
        seed=seed,
        atol=atol,
        rtol=rtol,
        build_dir=build_dir,
    )


def numeric_check_attention_prefill(
    batch_size: int = 1,
    total_q: int = 64,
    total_kv: int = 128,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    seq_lens: list[int] | None = None,
    seed: int = 0,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    build_dir: str | None = None,
) -> str:
    """Compare CPU TileLang prefill vs PyTorch golden (same shapes as RVV ladder)."""
    return _attention_numeric_check(
        "prefill",
        seq_lens if seq_lens is not None else [total_q] * batch_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        stage_kwargs=dict(
            prefill_block_M=block_M,
            prefill_block_N=block_N,
            prefill_num_stages=num_stages,
            prefill_threads=threads,
        ),
        seed=seed,
        atol=atol,
        rtol=rtol,
        build_dir=build_dir,
    )
