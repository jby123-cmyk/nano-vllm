"""
TileLang execution runtime for the nano-vllm engine.

Centralizes the ``cuda`` / ``cpu`` / ``rvv`` backend selection so layer modules
and ``run_tilelang_*`` helpers do not hardcode ``backend="cuda"``.

RVV path (Tier 3):
  * ``rvv_compile_only=True`` — cross-compile kernels to ``rvv_build_dir`` via
    ``lower_kernel_rvv``; execution raises ``RvvExecutionUnavailableError``.
  * ``execution_backend="spike"`` — execute lowered ``.ll`` via ``SpikeKernelSession``.
  * ``execution_backend="native"`` on riscv64 — compile with ``tilelang.compile``
    at ``rvv_target()`` and execute through TVM FFI.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import tilelang
from tilelang import tvm as tvm

from nanovllm.backends.tilelang.rvv_lower import (
    LowerResult,
    assert_target_vlen,
    lower_kernel_rvv,
    rvv_target,
)

TILELANG_BACKENDS = ("cuda", "cpu", "rvv")
RVV_EXECUTION_BACKENDS = ("compile_only", "spike", "native")


class RvvExecutionUnavailableError(RuntimeError):
    """Raised when RVV kernels are compiled but cannot execute on this host."""


@dataclass
class TilelangRuntimeConfig:
    backend: str = "cuda"
    nr_lanes: int = 4
    build_dir: str = "demos/build_rvv/engine"
    compile_only: bool = False
    execution_backend: str = "compile_only"
    spike_cache_dir: str = "demos/build_rvv/engine/spike_exec"


_RUNTIME = TilelangRuntimeConfig()


def configure_tilelang_runtime(
    *,
    backend: str = "cuda",
    nr_lanes: int = 4,
    build_dir: str = "demos/build_rvv/engine",
    compile_only: bool | None = None,
    execution_backend: str = "compile_only",
    spike_cache_dir: str | None = None,
) -> TilelangRuntimeConfig:
    """Install global TileLang execution settings (called from ``ModelRunner``)."""
    if backend not in TILELANG_BACKENDS:
        raise ValueError(
            f"tilelang_backend must be one of {TILELANG_BACKENDS}, got {backend!r}"
        )
    if execution_backend not in RVV_EXECUTION_BACKENDS:
        raise ValueError(
            f"execution_backend must be one of {RVV_EXECUTION_BACKENDS}, "
            f"got {execution_backend!r}"
        )
    if compile_only is None:
        if execution_backend == "spike":
            compile_only = False
        elif execution_backend == "native":
            compile_only = not probe_rvv_native_host()
        else:
            compile_only = backend == "rvv" and not probe_rvv_native_host()
    _RUNTIME.backend = backend
    _RUNTIME.nr_lanes = nr_lanes
    _RUNTIME.build_dir = build_dir
    _RUNTIME.compile_only = compile_only
    _RUNTIME.execution_backend = execution_backend
    _RUNTIME.spike_cache_dir = spike_cache_dir or os.path.join(
        build_dir, "spike_exec"
    )
    if backend == "rvv":
        target = rvv_target(nr_lanes=nr_lanes)
        assert_target_vlen(target, nr_lanes)
        os.makedirs(build_dir, exist_ok=True)
        if execution_backend == "spike":
            from nanovllm.backends.spike.config import default_config
            from nanovllm.backends.spike.session import configure_spike_session

            configure_spike_session(
                cfg=default_config(nr_lanes=nr_lanes),
                cache_dir=_RUNTIME.spike_cache_dir,
            )
    return _RUNTIME


def get_tilelang_runtime() -> TilelangRuntimeConfig:
    return _RUNTIME


def get_tilelang_execution_backend() -> str:
    """Backend string passed to ``run_tilelang_*`` from layer modules."""
    return _RUNTIME.backend


def rvv_execution_enabled() -> bool:
    return (
        _RUNTIME.backend == "rvv"
        and not _RUNTIME.compile_only
        and _RUNTIME.execution_backend in ("spike", "native")
    )


def rvv_model_execution_enabled(config) -> bool:
    """Whether ``device='rvv'`` may run kernels (Spike, native RVV, or host llvm ref)."""
    from nanovllm.engine.device import is_rvv_device

    if not is_rvv_device(config):
        return True
    if config.rvv_host_cpu_tilelang:
        return True
    return rvv_execution_enabled()


def probe_rvv_native_host() -> bool:
    """True when running on riscv64 with LLVM enabled (future native RVV execution)."""
    if platform.machine() not in ("riscv64", "riscv"):
        return False
    return tvm.runtime.enabled("llvm")


def compile_tilelang_kernel(
    builder: Any,
    build_args: tuple,
    out_idx: int | list[int] | None = 0,
    backend: str | None = None,
    *,
    kernel_name: str | None = None,
) -> Callable:
    """Compile a TileLang kernel builder for ``cuda``, ``cpu``, or ``rvv``."""
    backend = backend or _RUNTIME.backend
    if backend == "cuda":
        return builder(*build_args)
    tir_func = builder.get_tir(*build_args)
    compile_out_idx = None if out_idx is None else ([out_idx] if isinstance(out_idx, int) else list(out_idx))
    if backend == "cpu":
        with tvm.target.Target("llvm"):
            kwargs = {
                "target": "llvm",
                "target_host": "llvm",
                "execution_backend": "tvm_ffi",
            }
            if compile_out_idx is not None:
                kwargs["out_idx"] = compile_out_idx
            native = tilelang.compile(tir_func, **kwargs)
        return _maybe_wrap_cpu_reporting(
            native,
            kernel_name=kernel_name or getattr(builder, "__name__", "kernel"),
            build_args=build_args,
            build_args_digest=_RvvKernelCache._cache_key(
                kernel_name or getattr(builder, "__name__", "kernel"),
                build_args,
                compile_out_idx,
            ),
        )
    if backend == "rvv":
        return _RvvKernelCache.get().compile(
            tir_func=tir_func,
            out_idx=compile_out_idx,
            kernel_name=kernel_name or getattr(builder, "__name__", "kernel"),
            build_args=build_args,
        )
    raise ValueError(f"backend must be one of {TILELANG_BACKENDS}, got {backend!r}.")


class _RvvKernelCache:
    """Per-process RVV kernel compile cache (lower artifacts + optional callable)."""

    _instance: _RvvKernelCache | None = None

    def __init__(self) -> None:
        self._cache: dict[str, Callable] = {}

    @classmethod
    def get(cls) -> _RvvKernelCache:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def compile(
        self,
        *,
        tir_func: Any,
        out_idx: list[int] | None,
        kernel_name: str,
        build_args: tuple,
    ) -> Callable:
        key = self._cache_key(kernel_name, build_args, out_idx)
        if key in self._cache:
            return self._cache[key]

        safe_name = f"{kernel_name}_{key[:12]}"
        lower_dir = os.path.join(_RUNTIME.build_dir, safe_name)
        target = rvv_target(nr_lanes=_RUNTIME.nr_lanes)
        lower_result = lower_kernel_rvv(
            tir_func,
            name=safe_name,
            build_dir=lower_dir,
            target=target,
            title=kernel_name,
            metadata={"build_args": [repr(a) for a in build_args], "out_idx": out_idx},
        )

        native_callable = None
        if (
            not _RUNTIME.compile_only
            and _RUNTIME.execution_backend == "native"
            and probe_rvv_native_host()
        ):
            with target:
                kwargs = {
                    "target": target,
                    "target_host": target,
                    "execution_backend": "tvm_ffi",
                }
                if out_idx is not None:
                    kwargs["out_idx"] = out_idx
                native_callable = tilelang.compile(tir_func, **kwargs)

        handle = _RvvKernelHandle(
            kernel_name=kernel_name,
            lower_result=lower_result,
            native_callable=native_callable,
            compile_only=_RUNTIME.compile_only,
            execution_backend=_RUNTIME.execution_backend,
            out_idx=out_idx,
            build_args=build_args,
            tir_func=tir_func,
            build_args_digest=key,
        )
        self._cache[key] = handle
        return handle

    @staticmethod
    def _cache_key(kernel_name: str, build_args: tuple, out_idx: list[int] | None) -> str:
        payload = json.dumps(
            {
                "kernel": kernel_name,
                "args": [repr(a) for a in build_args],
                "out_idx": out_idx,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class _ReportingCpuCallable:
    """Record host-llvm kernel timing into the active generate report."""

    def __init__(
        self,
        callable_: Callable,
        *,
        kernel_name: str,
        build_args: tuple,
        build_args_digest: str,
    ) -> None:
        self._callable = callable_
        self.kernel_name = kernel_name
        self.build_args = build_args
        self.build_args_digest = build_args_digest

    def __call__(self, *args, **kwargs):
        from nanovllm.backends.spike.report_context import (
            active_collector,
            current_report_phase,
            current_step_id,
        )

        started = perf_counter()
        result = self._callable(*args, **kwargs)
        simulate_ms = (perf_counter() - started) * 1000.0
        collector = active_collector()
        if collector is not None:
            collector.record_kernel_call(
                kernel_name=self.kernel_name,
                phase=current_report_phase(),
                step_id=current_step_id(),
                case_id=f"host_llvm_{self.build_args_digest[:12]}",
                case_rel_path="host_llvm",
                compile_ms=0.0,
                simulate_ms=simulate_ms,
                spike_result="host_llvm",
                build_args_digest=self.build_args_digest,
            )
        return result


def _maybe_wrap_cpu_reporting(
    callable_: Callable,
    *,
    kernel_name: str,
    build_args: tuple,
    build_args_digest: str,
) -> Callable:
    from nanovllm.backends.spike.report_context import active_collector

    if active_collector() is None:
        return callable_
    return _ReportingCpuCallable(
        callable_,
        kernel_name=kernel_name,
        build_args=build_args,
        build_args_digest=build_args_digest,
    )


class _RvvKernelHandle:
    """Callable wrapper: lowers to RVV artifacts; executes via Spike or native RVV."""

    def __init__(
        self,
        *,
        kernel_name: str,
        lower_result: LowerResult,
        native_callable: Callable | None,
        compile_only: bool,
        execution_backend: str,
        out_idx: list[int] | None,
        build_args: tuple,
        tir_func: Any,
        build_args_digest: str,
    ) -> None:
        self.kernel_name = kernel_name
        self.lower_result = lower_result
        self._native_callable = native_callable
        self._compile_only = compile_only
        self._execution_backend = execution_backend
        self._out_idx = out_idx
        self._build_args = build_args
        self._tir_func = tir_func
        self._build_args_digest = build_args_digest
        self._cpu_reference_callable: Callable | None = None

    def _cpu_reference(self) -> Callable:
        if self._cpu_reference_callable is None:
            compile_out_idx = self._out_idx
            with tvm.target.Target("llvm"):
                kwargs = {
                    "target": "llvm",
                    "target_host": "llvm",
                    "execution_backend": "tvm_ffi",
                }
                if compile_out_idx is not None:
                    kwargs["out_idx"] = compile_out_idx
                self._cpu_reference_callable = tilelang.compile(self._tir_func, **kwargs)
        return self._cpu_reference_callable

    def _reference_callable_if_enabled(self):
        from nanovllm.backends.spike.report_context import active_collector

        collector = active_collector()
        if collector is None or not collector.compare_kernels_enabled():
            return None
        return self._cpu_reference()

    def __call__(self, *args, **kwargs):
        if self._compile_only:
            raise RvvExecutionUnavailableError(
                f"RVV kernel {self.kernel_name!r} was lowered to "
                f"{self.lower_result.build_dir!r} "
                f"(compiled={self.lower_result.compiled}), but cannot execute on "
                f"{platform.machine()}. Set rvv_execution_backend='spike' or "
                f"rvv_compile_only=False on an RVV host."
            )
        if self._execution_backend == "spike":
            ll_path = self.lower_result.ll_path
            if not ll_path or not os.path.isfile(ll_path):
                raise RvvExecutionUnavailableError(
                    f"RVV kernel {self.kernel_name!r} missing ll_path for Spike execution"
                )
            from nanovllm.backends.spike.session import get_spike_session

            return get_spike_session().execute_call(
                self.kernel_name,
                ll_path,
                args,
                out_idx=self._out_idx,
                build_args=self._build_args,
                reference_callable=self._reference_callable_if_enabled(),
                build_args_digest=self._build_args_digest,
            )
        if self._native_callable is None:
            raise RvvExecutionUnavailableError(
                f"RVV kernel {self.kernel_name!r} has no native callable on "
                f"{platform.machine()}."
            )
        return self._native_callable(*args, **kwargs)
