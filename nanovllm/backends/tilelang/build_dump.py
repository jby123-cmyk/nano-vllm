"""Dump TileLang compile artifacts (TIR, host code, device code) to a build directory."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from tilelang import tvm as tvm
from tilelang.backend.target import determine_target
from tilelang.engine.lower import (
    canon_target_host,
    device_codegen_without_compile,
    host_codegen,
    lower_to_host_device_ir,
)

from nanovllm.backends.tilelang.attention import (
    build_flash_attention_decode_kernel,
    build_flash_attention_prefill_kernel,
)
from nanovllm.backends.tilelang.embedding import build_embedding_kernel


def make_build_dir(base_dir: str = "demos/build") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    build_dir = os.path.join(base_dir, stamp)
    os.makedirs(build_dir, exist_ok=True)
    return build_dir


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _dump_lowered_artifacts(tir, build_dir: str, kernel_name: str) -> tuple[str, str]:
    """Lower a source TIR to host/device codegen and write all stages to disk.

    Returns (device_target_str, host_target_name). Mirrors the artifact layout
    used for the embedding kernel so every op dumps the same set of files.
    """
    _write_text(os.path.join(build_dir, f"{kernel_name}.tir"), tir.script())

    target = determine_target("auto", return_object=True)
    with target:
        host_mod, device_mod, _params, target_obj, _target_host = lower_to_host_device_ir(tir)
        _write_text(os.path.join(build_dir, "host.tir"), host_mod.script())
        _write_text(os.path.join(build_dir, "device.tir"), device_mod.script())

        device_codegen_mod = device_codegen_without_compile(device_mod, target_obj)
        _write_text(os.path.join(build_dir, "device_kernel.cu"), device_codegen_mod.inspect_source())

        host_target_name = canon_target_host(target_obj, None)
        host_target = tvm.target.Target(host_target_name)
        host_codegen_mod = host_codegen(host_mod, host_target, target=target_obj)
        host_ext = "ll" if host_target_name == "llvm" and tvm.runtime.enabled("llvm") else "c"
        _write_text(os.path.join(build_dir, f"host.{host_ext}"), host_codegen_mod.inspect_source())

    return str(target_obj), host_target_name


def _write_metadata_and_readme(
    build_dir: str,
    kernel_name: str,
    title: str,
    meta: dict,
    host_target_name: str,
) -> None:
    _write_text(
        os.path.join(build_dir, "metadata.json"),
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
    )
    host_ext = "ll" if host_target_name == "llvm" and tvm.runtime.enabled("llvm") else "c"
    readme = f"""{title}
Generated UTC: {datetime.now(timezone.utc).isoformat()}

Files:
  {kernel_name}.tir   Source TensorIR from TileLang @T.prim_func
  host.tir         Host-side TIR after split + lowering passes
  device.tir       Device-side TIR after split + lowering passes
  device_kernel.cu CUDA source for the GPU kernel (executes on device)
  host.{host_ext}            Host launcher that loads/invokes the kernel
  metadata.json    Compile-time constants and target info
  golden.log       PyTorch golden comparison (written by comparison demos)

Note: TVM LLVM enabled = {tvm.runtime.enabled("llvm")}.
When LLVM is disabled, host code is emitted as C (host.c) instead of LLVM IR.
The portable layer is the TIR; CUDA/LLVM are interim targets for bring-up.
"""
    _write_text(os.path.join(build_dir, "README.txt"), readme)


def dump_embedding_build(
    num_tokens: int,
    hidden: int,
    vocab: int,
    token_block: int = 1,
    threads: int = 256,
    in_dtype: str = "float32",
    build_dir: str | None = None,
    metadata: dict | None = None,
) -> str:
    """
    Write compile artifacts for the embedding kernel under demos/build/<timestamp>/.

    Files produced:
      embedding.tir   - TileLang source PrimFunc (from get_tir)
      host.tir        - lowered host-side TIR
      device.tir      - lowered device-side TIR
      device_kernel.cu - CUDA source (GPU device codegen)
      host.c            - host launcher (C; LLVM unavailable in this TVM build)
      metadata.json     - shapes, dtypes, and run context
      README.txt        - brief description of each file
    """
    if build_dir is None:
        build_dir = make_build_dir()

    os.makedirs(build_dir, exist_ok=True)

    tir = build_embedding_kernel.get_tir(
        num_tokens, hidden, vocab, token_block, threads, in_dtype
    )
    _write_text(os.path.join(build_dir, "embedding.tir"), tir.script())

    target = determine_target("auto", return_object=True)
    with target:
        host_mod, device_mod, _params, target_obj, _target_host = lower_to_host_device_ir(tir)
        _write_text(os.path.join(build_dir, "host.tir"), host_mod.script())
        _write_text(os.path.join(build_dir, "device.tir"), device_mod.script())

        device_codegen_mod = device_codegen_without_compile(device_mod, target_obj)
        device_source = device_codegen_mod.inspect_source()
        _write_text(os.path.join(build_dir, "device_kernel.cu"), device_source)

        host_target_name = canon_target_host(target_obj, None)
        host_target = tvm.target.Target(host_target_name)
        host_codegen_mod = host_codegen(host_mod, host_target, target=target_obj)
        host_source = host_codegen_mod.inspect_source()
        host_ext = "ll" if host_target_name == "llvm" and tvm.runtime.enabled("llvm") else "c"
        _write_text(os.path.join(build_dir, f"host.{host_ext}"), host_source)

    meta = {
        "num_tokens": num_tokens,
        "hidden": hidden,
        "vocab": vocab,
        "token_block": token_block,
        "threads": threads,
        "in_dtype": in_dtype,
        "device_target": str(target_obj),
        "host_target": host_target_name,
        "llvm_enabled": bool(tvm.runtime.enabled("llvm")),
    }
    if metadata:
        meta.update(metadata)
    _write_text(
        os.path.join(build_dir, "metadata.json"),
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
    )

    readme = f"""TileLang embedding build artifacts
Generated UTC: {datetime.now(timezone.utc).isoformat()}

Files:
  embedding.tir    Source TensorIR from TileLang @T.prim_func
  host.tir         Host-side TIR after split + lowering passes
  device.tir       Device-side TIR after split + lowering passes
  device_kernel.cu CUDA source for the GPU kernel (executes on device)
  host.{host_ext}            Host launcher that loads/invokes the kernel
  metadata.json    Compile-time constants and target info
  golden.log       PyTorch golden comparison (written by comparison demos)

Note: TVM LLVM enabled = {tvm.runtime.enabled("llvm")}.
When LLVM is disabled, host code is emitted as C (host.c) instead of LLVM IR.
"""
    _write_text(os.path.join(build_dir, "README.txt"), readme)

    print(f"Saved build artifacts to: {build_dir}")
    return build_dir


def dump_attention_prefill_build(
    batch_size: int,
    total_q: int,
    total_kv: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: float,
    is_causal: bool = True,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    in_dtype: str = "float16",
    build_dir: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Write TIR + host/device codegen for the FlashAttention prefill kernel."""
    if build_dir is None:
        build_dir = make_build_dir()
    os.makedirs(build_dir, exist_ok=True)

    tir = build_flash_attention_prefill_kernel.get_tir(
        batch_size,
        total_q,
        total_kv,
        num_heads,
        num_kv_heads,
        head_dim,
        float(softmax_scale),
        is_causal,
        block_M,
        block_N,
        num_stages,
        threads,
        in_dtype,
    )
    device_target, host_target_name = _dump_lowered_artifacts(
        tir, build_dir, "attention_prefill"
    )

    meta = {
        "phase": "prefill",
        "batch_size": batch_size,
        "total_q": total_q,
        "total_kv": total_kv,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "softmax_scale": float(softmax_scale),
        "is_causal": is_causal,
        "block_M": block_M,
        "block_N": block_N,
        "num_stages": num_stages,
        "threads": threads,
        "in_dtype": in_dtype,
        "device_target": device_target,
        "host_target": host_target_name,
        "llvm_enabled": bool(tvm.runtime.enabled("llvm")),
    }
    if metadata:
        meta.update(metadata)
    _write_metadata_and_readme(
        build_dir,
        "attention_prefill",
        "TileLang FlashAttention prefill (varlen causal GQA) build artifacts",
        meta,
        host_target_name,
    )

    print(f"Saved prefill build artifacts to: {build_dir}")
    return build_dir


def dump_attention_decode_build(
    batch_size: int,
    seqlen_kv: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: float,
    block_N: int = 128,
    block_H: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    in_dtype: str = "float16",
    build_dir: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Write TIR + host/device codegen for the FlashAttention decode kernel."""
    if build_dir is None:
        build_dir = make_build_dir()
    os.makedirs(build_dir, exist_ok=True)

    tir = build_flash_attention_decode_kernel.get_tir(
        batch_size,
        seqlen_kv,
        num_heads,
        num_kv_heads,
        head_dim,
        float(softmax_scale),
        block_N,
        block_H,
        num_stages,
        threads,
        in_dtype,
    )
    device_target, host_target_name = _dump_lowered_artifacts(
        tir, build_dir, "attention_decode"
    )

    meta = {
        "phase": "decode",
        "batch_size": batch_size,
        "seqlen_kv": seqlen_kv,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "softmax_scale": float(softmax_scale),
        "block_N": block_N,
        "block_H": block_H,
        "num_stages": num_stages,
        "threads": threads,
        "in_dtype": in_dtype,
        "device_target": device_target,
        "host_target": host_target_name,
        "llvm_enabled": bool(tvm.runtime.enabled("llvm")),
    }
    if metadata:
        meta.update(metadata)
    _write_metadata_and_readme(
        build_dir,
        "attention_decode",
        "TileLang FlashAttention decode (GQA KV-cache) build artifacts",
        meta,
        host_target_name,
    )

    print(f"Saved decode build artifacts to: {build_dir}")
    return build_dir
