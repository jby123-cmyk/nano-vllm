"""Toolchain paths, ISA/varch strings, and geometry for Spike RVV runs.

Defaults mirror AraXL's working ``asm/`` Spike flow. Override via env vars
or ``SpikeConfig`` fields when the install tree moves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# Confirmed AraXL install layout (see spike_integration_plan.md §0).
_DEFAULT_ARAXL_ROOT = "/mnt/ssd/jby123/AraXL"

DEFAULT_NR_LANES = 4
DEFAULT_ATOL = 1e-2
SPIKE_ISA = "rv64gcv_zfh"
SPIKE_ELEN = 64


@dataclass
class SpikeConfig:
    """Resolved toolchain + target geometry for one Spike build/run."""

    arax_root: str = field(
        default_factory=lambda: os.environ.get("ARAXL_ROOT", _DEFAULT_ARAXL_ROOT)
    )
    nr_lanes: int = DEFAULT_NR_LANES
    atol: float = DEFAULT_ATOL
    # Spike --varch vlen is capped at 4096 in the AraXL flow.
    spike_vlen_cap: int = 4096

    @property
    def vlen_bits(self) -> int:
        return 1024 * self.nr_lanes

    @property
    def spike_vlen(self) -> int:
        return min(self.vlen_bits, self.spike_vlen_cap)

    @property
    def clang(self) -> str:
        return os.environ.get(
            "SPIKE_CLANG",
            os.path.join(self.arax_root, "install/riscv-llvm/bin/clang"),
        )

    @property
    def llvm_objcopy(self) -> str:
        return os.environ.get(
            "SPIKE_OBJCOPY",
            os.path.join(self.arax_root, "install/riscv-llvm/bin/llvm-objcopy"),
        )

    @property
    def llvm_objdump(self) -> str:
        return os.environ.get(
            "SPIKE_OBJDUMP",
            os.path.join(self.arax_root, "install/riscv-llvm/bin/llvm-objdump"),
        )

    @property
    def gcc(self) -> str:
        return os.environ.get(
            "SPIKE_GCC",
            os.path.join(
                self.arax_root, "install/riscv-gcc/bin/riscv64-unknown-elf-gcc"
            ),
        )

    @property
    def spike(self) -> str:
        return os.environ.get(
            "SPIKE_BIN",
            os.path.join(self.arax_root, "install/riscv-isa-sim/bin/spike"),
        )

    @property
    def spike_env_dir(self) -> str:
        return os.path.join(self.arax_root, "apps/riscv-tests")

    @property
    def common_dir(self) -> str:
        return os.path.join(self.arax_root, "apps/common")

    @property
    def test_ld(self) -> str:
        return os.path.join(self.spike_env_dir, "benchmarks/common/test.ld")

    @property
    def crt_s(self) -> str:
        return os.path.join(self.spike_env_dir, "benchmarks/common/crt.S")

    @property
    def syscalls_c(self) -> str:
        return os.path.join(self.spike_env_dir, "benchmarks/common/syscalls.c")

    @property
    def util_c(self) -> str:
        return os.path.join(self.common_dir, "util.c")

    @property
    def runtime_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "runtime")

    @property
    def isa_flag(self) -> str:
        return f"--isa={SPIKE_ISA}"

    @property
    def varch_flag(self) -> str:
        return f"--varch=vlen:{self.spike_vlen},elen:{SPIKE_ELEN}"

    def spike_argv(self, elf_path: str) -> list[str]:
        return [self.spike, self.isa_flag, self.varch_flag, elf_path]

    def missing_tools(self) -> list[str]:
        """Return paths of required tools/files that are not present."""
        required = [
            self.clang,
            self.llvm_objcopy,
            self.gcc,
            self.spike,
            self.test_ld,
            self.crt_s,
            self.syscalls_c,
            self.util_c,
        ]
        return [p for p in required if not os.path.exists(p)]

    def assert_available(self) -> None:
        missing = self.missing_tools()
        if missing:
            raise FileNotFoundError(
                "Spike toolchain incomplete; missing:\n  "
                + "\n  ".join(missing)
                + "\nSet ARAXL_ROOT / SPIKE_* env overrides if the install moved."
            )


def default_config(nr_lanes: int = DEFAULT_NR_LANES, atol: float = DEFAULT_ATOL) -> SpikeConfig:
    return SpikeConfig(nr_lanes=nr_lanes, atol=atol)
