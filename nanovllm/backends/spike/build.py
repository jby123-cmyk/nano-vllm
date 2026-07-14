"""Compile TileLang RVV ``.ll`` + harness C into a Spike HTIF ELF.

Flags mirror ``AraXL/asm/Makefile`` (clang IR→obj, gcc + ``test.ld`` link).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from nanovllm.backends.spike.config import SpikeConfig, default_config


@dataclass
class BuildResult:
    elf_path: str
    build_dir: str
    compile_ms: float
    log: str


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _ensure_crt_vs(crt_s: str) -> None:
    """Ensure Spike crt0 enables vector state (MSTATUS_VS), matching AraXL asm."""
    with open(crt_s, encoding="utf-8") as handle:
        text = handle.read()
    patched = re.sub(
        r"li t0, MSTATUS_FS \| MSTATUS_XS$",
        "li t0, MSTATUS_FS | MSTATUS_XS | MSTATUS_VS",
        text,
        flags=re.M,
    )
    if patched != text:
        with open(crt_s, "w", encoding="utf-8") as handle:
            handle.write(patched)


def _base_cflags(cfg: SpikeConfig) -> list[str]:
    return [
        f"-march=rv64gcv_zfh_zvfh",
        "-menable-experimental-extensions",
        "-mabi=lp64d",
        "-mno-relax",
        "-mno-implicit-float",
        "-mcmodel=medany",
        "-O3",
        "-ffast-math",
        "-fno-common",
        "-ffunction-sections",
        "-fdata-sections",
        "-fno-builtin-printf",
        f"-DNR_LANES={cfg.nr_lanes}",
        f"-DVLEN={cfg.vlen_bits}",
        "-DNR_CLUSTERS=4",
    ]


def _gcc_spike_flags(cfg: SpikeConfig) -> list[str]:
    return [
        "-mcmodel=medany",
        "-march=rv64gcv",
        "-mabi=lp64d",
        f"-I{cfg.common_dir}",
        f"-I{cfg.runtime_dir}",
        f"-I{os.path.join(cfg.spike_env_dir, 'env')}",
        f"-I{os.path.join(cfg.spike_env_dir, 'benchmarks/common')}",
        "-static",
        "-std=gnu11",
        "-O3",
        "-ffast-math",
        "-fno-common",
        "-fno-builtin-printf",
        f"-DNR_LANES={cfg.nr_lanes}",
        f"-DVLEN={cfg.vlen_bits}",
        "-DNR_CLUSTERS=4",
        "-DPREALLOCATE=1",
        "-DSPIKE=1",
        "-Wunused-variable",
        "-Wall",
        "-Wextra",
        "-Wno-unused-command-line-argument",
    ]


def _compat_ll(src_ll: str, dest_ll: str) -> None:
    with open(src_ll, encoding="utf-8") as handle:
        text = handle.read()
    text = text.replace(" nocreateundeforpoison", "")
    os.makedirs(os.path.dirname(dest_ll) or ".", exist_ok=True)
    with open(dest_ll, "w", encoding="utf-8") as handle:
        handle.write(text)


def _compile_c(
    cfg: SpikeConfig, src: str, obj: str, flags: list[str]
) -> None:
    os.makedirs(os.path.dirname(obj) or ".", exist_ok=True)
    _run([cfg.gcc, *flags, "-c", src, "-o", obj])


def build_spike_elf(
    kernel_ll: str,
    main_c: str,
    build_dir: str,
    *,
    elf_name: str = "kernel.spike",
    extra_c_sources: list[str] | None = None,
    cfg: SpikeConfig | None = None,
) -> BuildResult:
    """Build a Spike ELF from ``kernel_ll`` + ``main_c`` + vendored runtime.

    ``extra_c_sources`` are additional host-side C files (e.g. generated ``data.c``).
    """
    import time

    cfg = cfg or default_config()
    cfg.assert_available()
    _ensure_crt_vs(cfg.crt_s)

    obj_dir = os.path.join(build_dir, "obj")
    bin_dir = os.path.join(build_dir, "bin")
    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    t0 = time.perf_counter()
    logs: list[str] = []

    compat = os.path.join(build_dir, "kernel.compat.ll")
    _compat_ll(kernel_ll, compat)

    kernel_o = os.path.join(obj_dir, "kernel.o")
    ir_flags = [
        *_base_cflags(cfg),
        "-fno-vectorize",
        "-mllvm",
        "-scalable-vectorization=off",
        "-mllvm",
        "-riscv-v-vector-bits-min=0",
        "-x",
        "ir",
        "-c",
        compat,
        "-o",
        kernel_o,
    ]
    proc = _run([cfg.clang, *ir_flags])
    logs.append(proc.stdout + proc.stderr)
    _run([cfg.llvm_objcopy, "--remove-section=.riscv.attributes", kernel_o])

    gcc_flags = _gcc_spike_flags(cfg)
    objects: list[str] = [kernel_o]

    # main harness
    main_o = os.path.join(obj_dir, "main.c.o")
    _compile_c(cfg, main_c, main_o, gcc_flags)
    objects.append(main_o)

    for src in extra_c_sources or []:
        name = os.path.basename(src).replace(".c", ".o")
        obj = os.path.join(obj_dir, name)
        _compile_c(cfg, src, obj, gcc_flags)
        objects.append(obj)

    runtime_sources = [
        ("harness_report.c", os.path.join(cfg.runtime_dir, "harness_report.c")),
        ("tvm_backend_runtime.c", os.path.join(cfg.runtime_dir, "tvm_backend_runtime.c")),
        ("soft_math.c", os.path.join(cfg.runtime_dir, "soft_math.c")),
        ("tvm_ffi_stubs.c", os.path.join(cfg.runtime_dir, "tvm_ffi_stubs.c")),
        ("crt.S", cfg.crt_s),
        ("syscalls.c", cfg.syscalls_c),
        ("util.c", cfg.util_c),
    ]
    for name, src in runtime_sources:
        obj = os.path.join(obj_dir, f"{name}.o")
        if src.endswith(".S"):
            _run([cfg.gcc, *gcc_flags, "-c", src, "-o", obj])
        else:
            _compile_c(cfg, src, obj, gcc_flags)
        objects.append(obj)

    elf_path = os.path.join(bin_dir, elf_name)
    ld_flags = [
        "-static",
        "-nostartfiles",
        "-Wl,--gc-sections",
        "-nostdlib",
        f"-T{cfg.test_ld}",
        "-DSPIKE",
    ]
    link = _run([cfg.gcc, *gcc_flags, "-o", elf_path, *objects, *ld_flags])
    logs.append(link.stdout + link.stderr)

    # Keep a copy next to sources for convenience.
    shutil.copy2(elf_path, os.path.join(build_dir, elf_name))

    compile_ms = (time.perf_counter() - t0) * 1000.0
    return BuildResult(
        elf_path=elf_path,
        build_dir=build_dir,
        compile_ms=compile_ms,
        log="\n".join(logs),
    )


def build_matmul_smoke(
    *,
    out_dir: str = "demos/build_rvv/spike/matmul_smoke",
    kernel_ll: str = "demos/build_rvv/matmul/matmul.ll",
    cfg: SpikeConfig | None = None,
) -> BuildResult:
    """Build the AraXL matmul harness against the Stage-1 matmul ``.ll``."""
    cfg = cfg or default_config()
    main_c = os.path.join(cfg.runtime_dir, "matmul_main.c")
    if not os.path.isfile(kernel_ll):
        raise FileNotFoundError(
            f"Missing {kernel_ll}; run demos/rvv/run_rvv_lower.py first."
        )
    if not os.path.isfile(main_c):
        raise FileNotFoundError(f"Missing vendored harness {main_c}")
    os.makedirs(out_dir, exist_ok=True)
    return build_spike_elf(
        kernel_ll,
        main_c,
        out_dir,
        elf_name="matmul.spike",
        cfg=cfg,
    )
