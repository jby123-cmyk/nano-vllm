"""Spike multi-kernel execution session for live TileLang RVV kernels."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from nanovllm.backends.spike.build import build_spike_elf
from nanovllm.backends.spike.config import SpikeConfig, default_config
from nanovllm.backends.spike.golden import make_tensor_entry, scalar_entry, write_tensor_blob
from nanovllm.backends.spike.harness import render_execute_harness
from nanovllm.backends.spike.generate_report import compare_kernel_outputs
from nanovllm.backends.spike.report_context import active_collector, current_report_phase, current_step_id
from nanovllm.backends.spike.run import SpikeExecuteResult, run_spike_execute_elf

# Packed argument order for each lowered engine kernel.
KERNEL_ARG_NAMES: dict[str, tuple[str, ...]] = {
    "linear": ("x", "weight", "bias", "y"),
    "rmsnorm": ("x", "weight", "residual", "output", "residual_out"),
    "silu_mul": ("x", "output"),
    "embedding": ("input_ids", "weight", "output"),
    "rope": ("q", "k", "cos", "sin", "q_out", "k_out"),
    "kv_store": ("key", "value", "k_cache", "v_cache", "slot_mapping"),
    "decode": ("q", "k_cache", "v_cache", "mask", "output"),
    "decode_paged": (
        "q",
        "k_cache",
        "v_cache",
        "block_table",
        "cache_seqlens",
        "output",
    ),
    "prefill": (
        "q",
        "k",
        "v",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "output",
    ),
    "prefill_paged": (
        "q",
        "k_cache",
        "v_cache",
        "block_table",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "output",
    ),
}

# In-place kernels read outputs from these tensor names after execution.
KERNEL_INPLACE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "kv_store": ("k_cache", "v_cache"),
}


KERNEL_NAME_ALIASES: dict[str, str] = {
    "build_flash_attention_decode_kernel": "decode",
    "build_flash_attention_prefill_kernel": "prefill",
    "build_flash_attention_decode_paged_kernel": "decode_paged",
    "build_flash_attention_prefill_paged_kernel": "prefill_paged",
}


def normalize_kernel_name(kernel_name: str) -> str:
    return KERNEL_NAME_ALIASES.get(kernel_name, kernel_name)


def _reference_outputs_for_compare(
    kernel_name: str,
    positional_args: tuple[torch.Tensor | int, ...],
    reference_callable,
    build_args: tuple = (),
) -> torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | dict[str, torch.Tensor]:
    """Run host-llvm reference on cloned inputs and normalize outputs for diffing."""
    inplace_outputs = KERNEL_INPLACE_OUTPUTS.get(kernel_name)
    arg_names = KERNEL_ARG_NAMES[kernel_name]
    ref_args: list[torch.Tensor | int] = [
        arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in positional_args
    ]
    result = reference_callable(*ref_args)
    if kernel_name == "rmsnorm" and len(build_args) > 3 and not build_args[3]:
        if isinstance(result, (list, tuple)):
            return result[0]
        return result
    if inplace_outputs is None:
        return result
    return {name: ref_args[arg_names.index(name)] for name in inplace_outputs}


def torch_dtype_tag(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.int32:
        return "int32"
    if dtype == torch.uint8:
        return "uint8"
    if dtype == torch.int64:
        return "int64"
    raise ValueError(f"unsupported Spike tensor dtype {dtype}")


def tensor_role(kernel_name: str, arg_name: str, out_idx: int | list[int] | None) -> str:
    if isinstance(out_idx, int):
        out_set = {out_idx}
    elif out_idx is None:
        out_set = set()
    else:
        out_set = set(out_idx)
    names = KERNEL_ARG_NAMES.get(kernel_name)
    if names is None:
        raise ValueError(f"unknown kernel arg layout for {kernel_name!r}")
    if arg_name not in names:
        raise ValueError(f"{arg_name!r} is not in {kernel_name} args {names}")
    if names.index(arg_name) in out_set:
        return "output"
    if kernel_name in KERNEL_INPLACE_OUTPUTS and arg_name in KERNEL_INPLACE_OUTPUTS[kernel_name]:
        return "input"
    return "input"


def output_tensor_names(kernel_name: str, out_idx: int | list[int] | None) -> tuple[str, ...]:
    if kernel_name in KERNEL_INPLACE_OUTPUTS:
        return KERNEL_INPLACE_OUTPUTS[kernel_name]
    names = KERNEL_ARG_NAMES[kernel_name]
    if isinstance(out_idx, int):
        return (names[out_idx],)
    if out_idx is None:
        raise ValueError(f"kernel {kernel_name!r} requires out_idx for execute mode")
    return tuple(names[i] for i in out_idx)


def _output_shape(
    kernel_name: str,
    build_args: tuple,
    output_name: str,
    tensors: dict[str, torch.Tensor | int],
) -> tuple[int, ...]:
    if kernel_name == "linear":
        padded_m, _padded_k, padded_n = build_args[0], build_args[1], build_args[2]
        if output_name == "y":
            return (int(padded_m), int(padded_n))
    if kernel_name == "silu_mul":
        tokens, inter = build_args[0], build_args[1]
        if output_name == "output":
            return (int(tokens), int(inter))
    if kernel_name == "embedding":
        tokens, hidden = build_args[0], build_args[1]
        if output_name == "output":
            return (int(tokens), int(hidden))
    if kernel_name == "rmsnorm":
        tokens, hidden = build_args[0], build_args[1]
        if output_name in ("output", "residual_out"):
            return (int(tokens), int(hidden))
    if kernel_name == "decode":
        if output_name == "output":
            q = tensors["q"]
            assert isinstance(q, torch.Tensor)
            return tuple(q.shape)
    if kernel_name == "rope":
        if output_name == "q_out":
            q = tensors["q"]
            assert isinstance(q, torch.Tensor)
            return tuple(q.shape)
        if output_name == "k_out":
            k = tensors["k"]
            assert isinstance(k, torch.Tensor)
            return tuple(k.shape)
    if output_name == "output" and "q" in tensors:
        q = tensors["q"]
        if isinstance(q, torch.Tensor):
            return tuple(q.shape)
    if output_name in tensors and isinstance(tensors[output_name], torch.Tensor):
        return tuple(tensors[output_name].shape)
    raise ValueError(
        f"cannot infer output shape for {kernel_name}.{output_name} "
        f"with build_args={build_args!r}"
    )


def build_kernel_tensor_map(
    kernel_name: str,
    build_args: tuple,
    positional_args: tuple[torch.Tensor | int, ...],
    out_idx: int | list[int] | None,
) -> dict[str, torch.Tensor | int]:
    """Map positional call arguments to a full kernel tensor dict including outputs."""
    kernel_name = normalize_kernel_name(kernel_name)
    names = KERNEL_ARG_NAMES[kernel_name]
    if isinstance(out_idx, int):
        out_indices = {out_idx}
    elif out_idx is None:
        out_indices = set()
    else:
        out_indices = set(out_idx)
    input_indices = [i for i in range(len(names)) if i not in out_indices]
    if len(positional_args) != len(input_indices):
        raise ValueError(
            f"{kernel_name} expects {len(input_indices)} inputs "
            f"{tuple(names[i] for i in input_indices)}, got {len(positional_args)}"
        )

    tensors: dict[str, torch.Tensor | int] = {}
    for idx, value in zip(input_indices, positional_args, strict=True):
        tensors[names[idx]] = value
    float_inputs = [
        tensor for tensor in tensors.values() if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
    ]
    if float_inputs:
        out_dtype = float_inputs[0].dtype
        ref_device = float_inputs[0].device
    else:
        tensor_inputs = [tensor for tensor in tensors.values() if isinstance(tensor, torch.Tensor)]
        ref = tensor_inputs[0]
        out_dtype = ref.dtype
        ref_device = ref.device
    for idx in sorted(out_indices):
        name = names[idx]
        shape = _output_shape(kernel_name, build_args, name, tensors)
        tensors[name] = torch.zeros(shape, dtype=out_dtype, device=ref_device)
    return tensors


@dataclass
class ExecuteCase:
    case_id: str
    case_dir: str
    manifest_path: str
    manifest: dict[str, Any]


_SESSION: SpikeKernelSession | None = None


class SpikeKernelSession:
    """Execute lowered RVV ``.ll`` kernels on Spike and return PyTorch tensors."""

    def __init__(
        self,
        cfg: SpikeConfig | None = None,
        *,
        cache_dir: str = "demos/build_rvv/engine/spike_exec",
    ) -> None:
        self.cfg = cfg or default_config()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def execute_call(
        self,
        kernel_name: str,
        ll_path: str,
        tensors: tuple[torch.Tensor, ...],
        *,
        out_idx: int | list[int] | None,
        build_args: tuple = (),
        phase: str | None = None,
        reference_callable=None,
        build_args_digest: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute a kernel from positional tensor arguments (``_RvvKernelHandle`` path)."""
        kernel_name = normalize_kernel_name(kernel_name)
        arg_names = KERNEL_ARG_NAMES.get(kernel_name)
        if arg_names is None:
            raise ValueError(f"unknown kernel {kernel_name!r}")
        tensor_map = build_kernel_tensor_map(
            kernel_name, build_args, tensors, out_idx
        )
        outputs = output_tensor_names(kernel_name, out_idx)
        result = self.execute(
            phase or kernel_name,
            ll_path,
            tensor_map,
            outputs=outputs,
            kernel_name=kernel_name,
            out_idx=out_idx,
            positional_args=tensors,
            reference_callable=reference_callable,
            build_args_digest=build_args_digest,
            build_args=build_args,
        )
        values = tuple(result[name] for name in outputs)
        if len(values) == 1:
            return values[0]
        return values

    def execute(
        self,
        phase: str,
        ll_path: str,
        tensors: dict[str, torch.Tensor | int],
        *,
        outputs: list[str] | tuple[str, ...],
        kernel_name: str | None = None,
        out_idx: int | list[int] | None = None,
        exec_id: str | None = None,
        positional_args: tuple[torch.Tensor | int, ...] | None = None,
        reference_callable=None,
        build_args_digest: str | None = None,
        build_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        if not os.path.isfile(ll_path):
            raise FileNotFoundError(ll_path)
        kernel_name = kernel_name or phase
        kernel_name = normalize_kernel_name(kernel_name)
        arg_names = KERNEL_ARG_NAMES.get(kernel_name)
        if arg_names is None:
            raise ValueError(f"unknown kernel {kernel_name!r}")

        exec_case = self._write_execute_case(
            phase=phase,
            kernel_name=kernel_name,
            tensors=tensors,
            arg_names=arg_names,
            outputs=tuple(outputs),
            out_idx=out_idx,
            exec_id=exec_id,
            ll_path=ll_path,
        )
        harness = render_execute_harness(exec_case.manifest_path, exec_case.case_dir)
        elf_name = f"{exec_case.case_id}.spike"
        build = build_spike_elf(
            ll_path,
            harness.main_c,
            exec_case.case_dir,
            elf_name=elf_name,
            cfg=self.cfg,
        )
        spike = run_spike_execute_elf(build.elf_path, cfg=self.cfg)
        log_path = os.path.join(exec_case.case_dir, "run.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(spike.stdout)
            handle.write(
                f"\nspike_return_code: {spike.returncode}\n"
                f"spike_result: {spike.spike_result}\n"
                f"build_ms: {build.compile_ms:.3f}\n"
                f"run_ms: {spike.run_ms:.3f}\n"
            )
        if not spike.passed:
            raise RuntimeError(
                f"Spike execute failed for {exec_case.case_id}: "
                f"{spike.spike_result}\n{spike.stdout[-1500:]}"
            )
        decoded = self._decode_outputs(spike.output_bytes, exec_case.manifest, outputs)
        collector = active_collector()
        if collector is not None:
            max_abs_diff = None
            if (
                reference_callable is not None
                and positional_args is not None
                and current_report_phase() != "warmup"
            ):
                reference_result = _reference_outputs_for_compare(
                    kernel_name,
                    positional_args,
                    reference_callable,
                    build_args,
                )
                max_abs_diff = compare_kernel_outputs(
                    kernel_name=kernel_name,
                    build_args=build_args,
                    positional_args=positional_args,
                    spike_outputs=decoded,
                    reference=reference_result,
                    output_names=tuple(outputs),
                )
            collector.record_kernel_call(
                kernel_name=kernel_name,
                phase=current_report_phase(),
                step_id=current_step_id(),
                case_id=exec_case.case_id,
                case_rel_path=os.path.join("spike_exec", exec_case.case_id),
                compile_ms=build.compile_ms,
                simulate_ms=spike.run_ms,
                spike_result=spike.spike_result,
                max_abs_diff_vs_reference=max_abs_diff,
                build_args_digest=build_args_digest or ll_path,
            )
        return decoded

    def _write_execute_case(
        self,
        *,
        phase: str,
        kernel_name: str,
        tensors: dict[str, torch.Tensor | int],
        arg_names: tuple[str, ...],
        outputs: tuple[str, ...],
        out_idx: int | list[int] | None,
        exec_id: str | None,
        ll_path: str,
    ) -> ExecuteCase:
        case_id = exec_id or self._make_exec_id(
            phase, kernel_name, tensors, arg_names, ll_path
        )
        case_dir = os.path.join(self.cache_dir, case_id)
        blob_dir = os.path.join(case_dir, "blobs")
        os.makedirs(blob_dir, exist_ok=True)

        args: list[dict[str, Any]] = []
        for name in arg_names:
            if name not in tensors:
                raise KeyError(f"missing tensor {name!r} for kernel {kernel_name!r}")
            value = tensors[name]
            if isinstance(value, int):
                args.append(scalar_entry(name, value))
                continue
            tensor = value.detach().cpu().contiguous()
            blob = f"blobs/{name}.bin"
            write_tensor_blob(os.path.join(case_dir, blob), tensor)
            args.append(
                make_tensor_entry(
                    name,
                    torch_dtype_tag(tensor.dtype),
                    list(tensor.shape),
                    blob,
                    role=tensor_role(kernel_name, name, out_idx),
                )
            )

        manifest = {
            "mode": "execute",
            "case_id": case_id,
            "phase": phase,
            "kernel_name": kernel_name,
            "outputs": list(outputs),
            "args": args,
        }
        manifest_path = os.path.join(case_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return ExecuteCase(
            case_id=case_id,
            case_dir=case_dir,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    @staticmethod
    def _make_exec_id(
        phase: str,
        kernel_name: str,
        tensors: dict[str, torch.Tensor | int],
        arg_names: tuple[str, ...],
        ll_path: str,
    ) -> str:
        payload = {
            "phase": phase,
            "kernel": kernel_name,
            "ll": ll_path,
            "args": {
                name: (
                    {"kind": "scalar", "value": int(tensors[name])}
                    if isinstance(tensors[name], int)
                    else {
                        "shape": list(tensors[name].shape),
                        "dtype": str(tensors[name].dtype),
                    }
                )
                for name in arg_names
            },
            "ts": time.time_ns(),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"{phase}_exec_{digest}"

    @staticmethod
    def _decode_outputs(
        raw_outputs: dict[str, bytes],
        manifest: dict[str, Any],
        output_names: list[str] | tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        by_name = {a["name"]: a for a in manifest["args"]}
        decoded: dict[str, torch.Tensor] = {}
        for name in output_names:
            arg = by_name[name]
            raw = raw_outputs[name]
            dtype = arg["dtype"]
            shape = [int(x) for x in arg["shape"]]
            if dtype in ("f32", "float32"):
                tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()
            elif dtype == "int32":
                tensor = torch.frombuffer(bytearray(raw), dtype=torch.int32).reshape(shape).clone()
            else:
                raise ValueError(f"unsupported execute output dtype {dtype!r}")
            decoded[name] = tensor
        return decoded


def configure_spike_session(
    *,
    cfg: SpikeConfig | None = None,
    cache_dir: str = "demos/build_rvv/engine/spike_exec",
) -> SpikeKernelSession:
    """Install the process-global Spike execution session."""
    global _SESSION
    session = SpikeKernelSession(cfg=cfg, cache_dir=cache_dir)
    session.cfg.assert_available()
    _SESSION = session
    return session


def get_spike_session() -> SpikeKernelSession:
    if _SESSION is None:
        raise RuntimeError(
            "SpikeKernelSession is not configured; call configure_spike_session() "
            "or configure_tilelang_runtime(..., execution_backend='spike')."
        )
    return _SESSION


def reset_spike_session() -> None:
    global _SESSION
    _SESSION = None
