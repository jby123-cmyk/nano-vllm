"""Run a Spike ELF and parse ``HARNESS_SUMMARY``.

Also exposes ``python -m nanovllm.backends.spike.run --smoke``.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass

from nanovllm.backends.spike.config import SpikeConfig, default_config


_SUMMARY_RE = re.compile(
    r"^HARNESS_SUMMARY\s+"
    r"app=(?P<app>\S+)\s+"
    r"kernel_status=(?P<kernel_status>-?\d+)\s+"
    r"(?:max_abs_diff=(?P<max_abs_diff>[\d.eE+-]+)\s+)?"
    r"(?:max_abs_diff_bits=0x(?P<max_bits>[0-9a-fA-F]+)\s+)?"
    r"max_abs_diff_u6=(?P<max_u6>\d+)\s+"
    r"(?:atol=(?P<atol>[\d.eE+-]+)\s+)?"
    r"(?:atol_bits=0x(?P<atol_bits>[0-9a-fA-F]+)\s+)?"
    r"atol_u6=(?P<atol_u6>\d+)\s+"
    r"elements=(?P<elements>-?\d+)\s+"
    r"mismatches=(?P<mismatches>-?\d+)\s+"
    r"spike_result=(?P<spike_result>\S+)",
    re.M,
)

_EXECUTE_SUMMARY_RE = re.compile(
    r"^HARNESS_EXECUTE_SUMMARY\s+"
    r"app=(?P<app>\S+)\s+"
    r"kernel_status=(?P<kernel_status>-?\d+)\s+"
    r"spike_result=(?P<spike_result>\S+)\s+"
    r"num_outputs=(?P<num_outputs>\d+)",
    re.M,
)

_OUTPUT_BEGIN_RE = re.compile(
    r"^OUTPUT_BEGIN name=(?P<name>\S+) dtype=(?P<dtype>\S+) nbytes=(?P<nbytes>\d+)",
    re.M,
)

_OUTPUT_CHUNK_RE = re.compile(
    r"^OUTPUT_CHUNK offset=(?P<offset>\d+) nbytes=(?P<nbytes>\d+) hex=(?P<hex>[0-9a-fA-F]*)",
    re.M,
)

_OUTPUT_END_RE = re.compile(r"^OUTPUT_END name=(?P<name>\S+)", re.M)


def _bits_to_float(bits_hex: str) -> float:
    bits = int(bits_hex, 16)
    return struct.unpack(">f", struct.pack(">I", bits))[0]


@dataclass
class SpikeResult:
    app: str
    elf_path: str
    returncode: int
    spike_result: str
    max_abs_diff: float
    atol: float
    elements: int
    mismatches: int
    kernel_status: int
    run_ms: float
    stdout: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and self.spike_result == "pass"


@dataclass
class SpikeExecuteResult:
    app: str
    elf_path: str
    returncode: int
    spike_result: str
    kernel_status: int
    num_outputs: int
    output_bytes: dict[str, bytes]
    run_ms: float
    stdout: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and self.spike_result == "pass"


def parse_execute_outputs(stdout: str) -> dict[str, bytes]:
    """Parse OUTPUT_* blocks emitted by ``harness_finish_execute``."""
    outputs: dict[str, bytearray] = {}
    current: str | None = None
    expected = 0
    for line in stdout.splitlines():
        begin = _OUTPUT_BEGIN_RE.match(line)
        if begin:
            current = begin.group("name")
            expected = int(begin.group("nbytes"))
            outputs[current] = bytearray(expected)
            continue
        chunk = _OUTPUT_CHUNK_RE.match(line)
        if chunk and current is not None:
            offset = int(chunk.group("offset"))
            nbytes = int(chunk.group("nbytes"))
            raw = bytes.fromhex(chunk.group("hex") or "")
            if len(raw) != nbytes:
                raise ValueError(
                    f"{current}: chunk nbytes={nbytes} but hex decoded {len(raw)}"
                )
            buf = outputs[current]
            buf[offset : offset + nbytes] = raw
            continue
        end = _OUTPUT_END_RE.match(line)
        if end and current is not None:
            if end.group("name") != current:
                raise ValueError(
                    f"OUTPUT_END name mismatch: {end.group('name')} != {current}"
                )
            if len(outputs[current]) != expected:
                raise ValueError(
                    f"{current}: expected {expected} bytes, got {len(outputs[current])}"
                )
            current = None
    return {name: bytes(data) for name, data in outputs.items()}


def parse_execute_summary(stdout: str) -> dict | None:
    matches = list(_EXECUTE_SUMMARY_RE.finditer(stdout))
    if not matches:
        return None
    m = matches[-1]
    return {
        "app": m.group("app"),
        "kernel_status": int(m.group("kernel_status")),
        "spike_result": m.group("spike_result"),
        "num_outputs": int(m.group("num_outputs")),
    }


def run_spike_execute_elf(
    elf_path: str,
    *,
    cfg: SpikeConfig | None = None,
    timeout_s: float | None = 600.0,
) -> SpikeExecuteResult:
    cfg = cfg or default_config()
    cfg.assert_available()
    if not os.path.isfile(elf_path):
        raise FileNotFoundError(elf_path)

    argv = cfg.spike_argv(elf_path)
    t0 = time.perf_counter()
    proc = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    run_ms = (time.perf_counter() - t0) * 1000.0
    stdout = (proc.stdout or "") + (proc.stderr or "")
    parsed = parse_execute_summary(stdout)
    if parsed is None:
        return SpikeExecuteResult(
            app=os.path.basename(elf_path),
            elf_path=elf_path,
            returncode=proc.returncode,
            spike_result="no_summary",
            kernel_status=-1,
            num_outputs=0,
            output_bytes={},
            run_ms=run_ms,
            stdout=stdout,
        )
    try:
        output_bytes = parse_execute_outputs(stdout)
    except ValueError as exc:
        return SpikeExecuteResult(
            app=parsed["app"],
            elf_path=elf_path,
            returncode=proc.returncode,
            spike_result="output_parse_error",
            kernel_status=parsed["kernel_status"],
            num_outputs=parsed["num_outputs"],
            output_bytes={},
            run_ms=run_ms,
            stdout=stdout + f"\noutput_parse_error: {exc}\n",
        )
    return SpikeExecuteResult(
        app=parsed["app"],
        elf_path=elf_path,
        returncode=proc.returncode,
        spike_result=parsed["spike_result"],
        kernel_status=parsed["kernel_status"],
        num_outputs=parsed["num_outputs"],
        output_bytes=output_bytes,
        run_ms=run_ms,
        stdout=stdout,
    )


def parse_harness_summary(stdout: str) -> dict | None:
    matches = list(_SUMMARY_RE.finditer(stdout))
    if not matches:
        return None
    m = matches[-1]
    if m.group("max_bits") is not None:
        max_abs_diff = _bits_to_float(m.group("max_bits"))
    elif m.group("max_abs_diff") is not None:
        max_abs_diff = float(m.group("max_abs_diff"))
    else:
        max_abs_diff = int(m.group("max_u6")) * 1e-6
    if m.group("atol_bits") is not None:
        atol = _bits_to_float(m.group("atol_bits"))
    elif m.group("atol") is not None:
        atol = float(m.group("atol"))
    else:
        atol = int(m.group("atol_u6")) * 1e-6
    return {
        "app": m.group("app"),
        "kernel_status": int(m.group("kernel_status")),
        "max_abs_diff": max_abs_diff,
        "atol": atol,
        "elements": int(m.group("elements")),
        "mismatches": int(m.group("mismatches")),
        "spike_result": m.group("spike_result"),
    }


def run_spike_elf(
    elf_path: str,
    *,
    cfg: SpikeConfig | None = None,
    timeout_s: float | None = 600.0,
) -> SpikeResult:
    cfg = cfg or default_config()
    cfg.assert_available()
    if not os.path.isfile(elf_path):
        raise FileNotFoundError(elf_path)

    argv = cfg.spike_argv(elf_path)
    t0 = time.perf_counter()
    proc = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    run_ms = (time.perf_counter() - t0) * 1000.0
    stdout = (proc.stdout or "") + (proc.stderr or "")
    parsed = parse_harness_summary(stdout)
    if parsed is None:
        return SpikeResult(
            app=os.path.basename(elf_path),
            elf_path=elf_path,
            returncode=proc.returncode,
            spike_result="no_summary",
            max_abs_diff=float("nan"),
            atol=cfg.atol,
            elements=-1,
            mismatches=-1,
            kernel_status=-1,
            run_ms=run_ms,
            stdout=stdout,
        )
    return SpikeResult(
        app=parsed["app"],
        elf_path=elf_path,
        returncode=proc.returncode,
        spike_result=parsed["spike_result"],
        max_abs_diff=parsed["max_abs_diff"],
        atol=parsed["atol"],
        elements=parsed["elements"],
        mismatches=parsed["mismatches"],
        kernel_status=parsed["kernel_status"],
        run_ms=run_ms,
        stdout=stdout,
    )


def smoke_matmul(
    *,
    out_dir: str = "demos/build_rvv/spike/matmul_smoke",
    kernel_ll: str = "demos/build_rvv/matmul/matmul.ll",
    nr_lanes: int = 4,
) -> SpikeResult:
    from nanovllm.backends.spike.build import build_matmul_smoke

    cfg = default_config(nr_lanes=nr_lanes)
    build = build_matmul_smoke(out_dir=out_dir, kernel_ll=kernel_ll, cfg=cfg)
    print(f"Built {build.elf_path} in {build.compile_ms:.0f} ms")
    result = run_spike_elf(build.elf_path, cfg=cfg)
    log_path = os.path.join(out_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(result.stdout)
        handle.write(
            f"\nspike_return_code: {result.returncode}\n"
            f"spike_result: {result.spike_result}\n"
            f"max_abs_diff: {result.max_abs_diff}\n"
        )
    print(
        f"Spike: result={result.spike_result} max_abs_diff={result.max_abs_diff:.6e} "
        f"rc={result.returncode} run_ms={result.run_ms:.0f}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Build+run Stage-1 matmul.ll with AraXL matmul_main.c on Spike.",
    )
    parser.add_argument("--elf", default="", help="Run an existing Spike ELF.")
    parser.add_argument("--nr-lanes", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        default="demos/build_rvv/spike/matmul_smoke",
        help="Build directory for --smoke.",
    )
    parser.add_argument(
        "--kernel-ll",
        default="demos/build_rvv/matmul/matmul.ll",
        help="LLVM IR for --smoke.",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        result = smoke_matmul(
            out_dir=args.out_dir, kernel_ll=args.kernel_ll, nr_lanes=args.nr_lanes
        )
        return 0 if result.passed else 1

    if args.elf:
        result = run_spike_elf(args.elf, cfg=default_config(nr_lanes=args.nr_lanes))
        print(
            f"Spike: result={result.spike_result} max_abs_diff={result.max_abs_diff:.6e} "
            f"rc={result.returncode}"
        )
        if not result.passed:
            print(result.stdout[-2000:], file=sys.stderr)
        return 0 if result.passed else 1

    parser.error("pass --smoke or --elf")
    return 2


if __name__ == "__main__":
    sys.exit(main())
