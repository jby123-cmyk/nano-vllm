"""Emit AttentionStage inputs + PyTorch golden as binary blobs + manifest.

The Spike harness embeds these blobs so on-target compare uses the same
reference as the host matrix (``demos/rvv/matrix_report.py``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch

from nanovllm.backends.spike.config import DEFAULT_ATOL, DEFAULT_NR_LANES
from nanovllm.backends.tilelang.rvv_lower import vlen_f32_elements
from nanovllm.stages.attention import AttentionStage, _round_up


@dataclass
class GoldenCase:
    case_id: str
    phase: str
    case_dir: str
    manifest_path: str
    manifest: dict[str, Any]


def make_case_id(
    phase: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    lengths: list[int],
    nr_lanes: int,
) -> str:
    if phase == "linear":
        if len(lengths) != 3:
            raise ValueError(
                f"linear case lengths must be [M, N, K], got {lengths}"
            )
        m, n, k = lengths
        return f"linear_m{m}_n{n}_k{k}_nl{nr_lanes}"
    if phase in ("rmsnorm", "silu_mul", "rope", "kv_store", "embedding"):
        lens = "-".join(str(x) for x in lengths)
        return f"{phase}_len{lens}_nl{nr_lanes}"
    lens = "-".join(str(x) for x in lengths)
    return f"{phase}_h{num_heads}_kv{num_kv_heads}_d{head_dim}_len{lens}_nl{nr_lanes}"


def _rvv_stage(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    nr_lanes: int,
) -> AttentionStage:
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


def _write_blob(path: str, tensor: torch.Tensor) -> int:
    """Write contiguous CPU tensor bytes; return element count."""
    arr = tensor.detach().cpu().contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(arr.numpy().tobytes())
    return int(arr.numel())


write_tensor_blob = _write_blob


def _tensor_entry(
    name: str,
    dtype: str,
    shape: list[int],
    blob: str,
    *,
    role: str,
) -> dict[str, Any]:
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    strides.reverse()
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "strides": strides,
        "blob": blob,
        "role": role,  # input | output | golden | scalar
        "kind": "tensor",
    }


make_tensor_entry = _tensor_entry


def _scalar_entry(name: str, value: int) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "int64",
        "value": int(value),
        "role": "scalar",
        "kind": "scalar",
    }


scalar_entry = _scalar_entry


def emit_golden(
    phase: str,
    lengths: list[int],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    nr_lanes: int = DEFAULT_NR_LANES,
    out_root: str = "demos/build_rvv/spike",
    atol: float = DEFAULT_ATOL,
    case_id: str | None = None,
) -> GoldenCase:
    """Write blobs + ``manifest.json`` under ``out_root/<case_id>/``."""
    if phase not in (
        "decode",
        "decode_paged",
        "prefill",
        "prefill_paged",
        "linear",
        "rmsnorm",
        "silu_mul",
        "rope",
        "kv_store",
        "embedding",
    ):
        raise ValueError(
            f"unsupported phase {phase!r}"
        )

    case_id = case_id or make_case_id(
        phase, num_heads, num_kv_heads, head_dim, lengths, nr_lanes
    )
    case_dir = os.path.join(out_root, case_id)
    blob_dir = os.path.join(case_dir, "blobs")
    os.makedirs(blob_dir, exist_ok=True)

    args: list[dict[str, Any]] = []

    if phase == "silu_mul":
        import torch.nn.functional as F

        tokens, inter = lengths[0], lengths[1]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        x = torch.randn(tokens, inter * 2, dtype=torch.float32, generator=gen)
        gate, up = x.chunk(2, -1)
        reference = (F.silu(gate) * up).contiguous()
        output = torch.zeros_like(reference)
        for name, tensor, dtype, role in [
            ("x", x, "f32", "input"),
            ("output", output, "f32", "output"),
            ("golden", reference, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(_tensor_entry(name, dtype, list(tensor.shape), blob, role=role))
        meta = {"num_tokens": tokens, "intermediate": inter, "block_M": 64}
        numel_out = int(reference.numel())
        manifest = {
            "case_id": case_id,
            "phase": phase,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "lengths": list(lengths),
            "nr_lanes": nr_lanes,
            "atol": float(atol),
            "softmax_scale": 1.0,
            "seed": 0,
            "compare_elements": numel_out,
            "meta": meta,
            "args": args,
        }
        manifest_path = os.path.join(case_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return GoldenCase(
            case_id=case_id, phase=phase, case_dir=case_dir,
            manifest_path=manifest_path, manifest=manifest,
        )

    if phase == "embedding":
        tokens, hidden, vocab = lengths
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        weight = torch.randn(vocab, hidden, dtype=torch.float32, generator=gen)
        ids = torch.randint(0, vocab, (tokens,), dtype=torch.int32, generator=gen)
        reference = weight[ids.long()].contiguous()
        output = torch.zeros_like(reference)
        for name, tensor, dtype, role in [
            ("input_ids", ids, "int32", "input"),
            ("weight", weight, "f32", "input"),
            ("output", output, "f32", "output"),
            ("golden", reference, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(_tensor_entry(name, dtype, list(tensor.shape), blob, role=role))
        meta = {"num_tokens": tokens, "hidden": hidden, "vocab": vocab}
        numel_out = int(reference.numel())
        manifest = {
            "case_id": case_id,
            "phase": phase,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "lengths": list(lengths),
            "nr_lanes": nr_lanes,
            "atol": float(atol),
            "softmax_scale": 1.0,
            "seed": 0,
            "compare_elements": numel_out,
            "meta": meta,
            "args": args,
        }
        manifest_path = os.path.join(case_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return GoldenCase(
            case_id=case_id, phase=phase, case_dir=case_dir,
            manifest_path=manifest_path, manifest=manifest,
        )

    if phase == "rmsnorm":
        tokens, hidden = lengths[0], lengths[1]
        fuse = bool(lengths[2]) if len(lengths) > 2 else False
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        x = torch.randn(tokens, hidden, dtype=torch.float32, generator=gen)
        weight = torch.randn(hidden, dtype=torch.float32, generator=gen)
        residual = (
            torch.randn(tokens, hidden, dtype=torch.float32, generator=gen)
            if fuse
            else torch.zeros(tokens, hidden, dtype=torch.float32)
        )
        xf = x + residual if fuse else x
        if fuse:
            residual_out_ref = xf.clone()
        else:
            residual_out_ref = torch.zeros_like(x)
        var = xf.pow(2).mean(-1, keepdim=True)
        reference = xf * torch.rsqrt(var + 1e-6) * weight
        output = torch.zeros_like(reference)
        residual_out = torch.zeros_like(residual_out_ref)
        for name, tensor, dtype, role in [
            ("x", x, "f32", "input"),
            ("weight", weight, "f32", "input"),
            ("residual", residual, "f32", "input"),
            ("output", output, "f32", "output"),
            ("residual_out", residual_out, "f32", "output"),
            ("golden", reference, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(_tensor_entry(name, dtype, list(tensor.shape), blob, role=role))
        meta = {
            "num_tokens": tokens,
            "hidden": hidden,
            "fuse_residual": fuse,
            "eps": 1e-6,
        }
        numel_out = int(reference.numel())
        manifest = {
            "case_id": case_id,
            "phase": phase,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "lengths": list(lengths),
            "nr_lanes": nr_lanes,
            "atol": float(atol),
            "softmax_scale": 1.0,
            "seed": 0,
            "compare_elements": numel_out,
            "meta": meta,
            "args": args,
        }
        manifest_path = os.path.join(case_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return GoldenCase(
            case_id=case_id, phase=phase, case_dir=case_dir,
            manifest_path=manifest_path, manifest=manifest,
        )

    if phase == "linear":
        import torch.nn.functional as F

        from nanovllm.backends.tilelang.linear import (
            DEFAULT_BLOCK_K,
            DEFAULT_BLOCK_M,
            DEFAULT_BLOCK_N,
        )

        if len(lengths) != 3:
            raise ValueError(f"linear lengths must be [M, N, K], got {lengths}")
        m, n, k = lengths
        padded_m = _round_up(max(m, 1), DEFAULT_BLOCK_M)
        padded_n = _round_up(n, DEFAULT_BLOCK_N)
        padded_k = _round_up(k, DEFAULT_BLOCK_K)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        x = torch.randn(m, k, dtype=torch.float32, generator=gen)
        weight = torch.randn(n, k, dtype=torch.float32, generator=gen)
        bias = torch.zeros(n, dtype=torch.float32)
        reference = F.linear(x, weight, None)

        x_pad = torch.zeros(padded_m, padded_k, dtype=torch.float32)
        x_pad[:m, :k] = x
        w_pad = torch.zeros(padded_n, padded_k, dtype=torch.float32)
        w_pad[:n, :k] = weight
        b_pad = torch.zeros(padded_n, dtype=torch.float32)

        output = torch.zeros(padded_m, padded_n, dtype=torch.float32)
        golden_pad = torch.zeros_like(output)
        golden_pad[:m, :n] = reference

        for name, tensor, dtype, role in [
            ("x", x_pad, "f32", "input"),
            ("weight", w_pad, "f32", "input"),
            ("bias", b_pad, "f32", "input"),
            ("output", output, "f32", "output"),
            ("golden", golden_pad, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(
                _tensor_entry(name, dtype, list(tensor.shape), blob, role=role)
            )

        meta = {
            "num_tokens": m,
            "out_features": n,
            "in_features": k,
            "padded_m": padded_m,
            "padded_n": padded_n,
            "padded_k": padded_k,
            "has_bias": False,
            "block_M": DEFAULT_BLOCK_M,
            "block_N": DEFAULT_BLOCK_N,
            "block_K": DEFAULT_BLOCK_K,
            "compare_elements": int(reference.numel()),
        }
        numel_out = int(reference.numel())
        manifest = {
            "case_id": case_id,
            "phase": phase,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "lengths": list(lengths),
            "nr_lanes": nr_lanes,
            "atol": float(atol),
            "softmax_scale": 1.0,
            "seed": 0,
            "compare_elements": numel_out,
            "meta": meta,
            "args": args,
        }
        manifest_path = os.path.join(case_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return GoldenCase(
            case_id=case_id,
            phase=phase,
            case_dir=case_dir,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    if phase == "rope":
        from nanovllm.layers.rotary_embedding import apply_rotary_emb

        tokens, num_heads, num_kv_heads, head_dim = lengths
        half = head_dim // 2
        block_M = 64
        padded_m = _round_up(max(tokens, 1), block_M)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        q = torch.randn(tokens, num_heads, head_dim, dtype=torch.float32, generator=gen)
        k = torch.randn(
            tokens, num_kv_heads, head_dim, dtype=torch.float32, generator=gen
        )
        cos = torch.randn(tokens, 1, half, dtype=torch.float32, generator=gen)
        sin = torch.randn(tokens, 1, half, dtype=torch.float32, generator=gen)

        q_ref = apply_rotary_emb(q, cos, sin)
        k_ref = apply_rotary_emb(k, cos, sin)

        q_pad = q
        k_pad = k
        cos_pad = cos
        sin_pad = sin
        if padded_m != tokens:
            q_pad = torch.zeros(padded_m, num_heads, head_dim, dtype=torch.float32)
            k_pad = torch.zeros(padded_m, num_kv_heads, head_dim, dtype=torch.float32)
            cos_pad = torch.zeros(padded_m, 1, half, dtype=torch.float32)
            sin_pad = torch.zeros(padded_m, 1, half, dtype=torch.float32)
            q_pad[:tokens] = q
            k_pad[:tokens] = k
            cos_pad[:tokens] = cos
            sin_pad[:tokens] = sin

        q_out = torch.zeros_like(q_pad)
        k_out = torch.zeros_like(k_pad)
        golden_q = torch.zeros_like(q_pad)
        golden_q[:tokens] = q_ref
        golden_k = torch.zeros_like(k_pad)
        golden_k[:tokens] = k_ref

        for name, tensor, dtype, role in [
            ("q", q_pad, "f32", "input"),
            ("k", k_pad, "f32", "input"),
            ("cos", cos_pad, "f32", "input"),
            ("sin", sin_pad, "f32", "input"),
            ("q_out", q_out, "f32", "output"),
            ("k_out", k_out, "f32", "output"),
            ("golden", golden_q, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(_tensor_entry(name, dtype, list(tensor.shape), blob, role=role))

        meta = {
            "num_tokens": tokens,
            "padded_m": padded_m,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "block_M": block_M,
            "compare_k_elements": int(k_ref.numel()),
        }
        numel_out = int(q_ref.numel())
        manifest = {
            "case_id": case_id,
            "phase": phase,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "lengths": list(lengths),
            "nr_lanes": nr_lanes,
            "atol": float(atol),
            "softmax_scale": 1.0,
            "seed": 0,
            "compare_elements": numel_out,
            "meta": meta,
            "args": args,
        }
        manifest_path = os.path.join(case_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return GoldenCase(
            case_id=case_id, phase=phase, case_dir=case_dir,
            manifest_path=manifest_path, manifest=manifest,
        )

    if phase == "kv_store":
        tokens, num_kv_heads, head_dim, num_slots = lengths
        D = num_kv_heads * head_dim

        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        key = torch.randn(tokens, num_kv_heads, head_dim, dtype=torch.float32, generator=gen)
        value = torch.randn(tokens, num_kv_heads, head_dim, dtype=torch.float32, generator=gen)
        k_cache = torch.randn(num_slots, D, dtype=torch.float32, generator=gen)
        v_cache = torch.randn(num_slots, D, dtype=torch.float32, generator=gen)
        slot_mapping = torch.tensor(
            [i * 3 % num_slots if i % 3 != 2 else -1 for i in range(tokens)],
            dtype=torch.int32,
        )

        golden_k = k_cache.clone()
        golden_v = v_cache.clone()
        for i in range(tokens):
            slot = int(slot_mapping[i].item())
            if slot >= 0:
                golden_k[slot] = key[i].reshape(-1)
                golden_v[slot] = value[i].reshape(-1)

        for name, tensor, dtype, role in [
            ("key", key, "f32", "input"),
            ("value", value, "f32", "input"),
            ("k_cache", k_cache, "f32", "input"),
            ("v_cache", v_cache, "f32", "input"),
            ("slot_mapping", slot_mapping, "int32", "input"),
            ("golden", golden_k, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(_tensor_entry(name, dtype, list(tensor.shape), blob, role=role))

        meta = {
            "num_tokens": tokens,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_slots": num_slots,
            "D": D,
            "compare_v_elements": int(golden_v.numel()),
        }
        numel_out = int(golden_k.numel())
        manifest = {
            "case_id": case_id,
            "phase": phase,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "lengths": list(lengths),
            "nr_lanes": nr_lanes,
            "atol": float(atol),
            "softmax_scale": 1.0,
            "seed": 0,
            "compare_elements": numel_out,
            "compare_tensor": "k_cache",
            "meta": meta,
            "args": args,
        }
        manifest_path = os.path.join(case_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _write_blob(os.path.join(case_dir, "blobs/golden_v.bin"), golden_v)
        return GoldenCase(
            case_id=case_id, phase=phase, case_dir=case_dir,
            manifest_path=manifest_path, manifest=manifest,
        )

    stage = _rvv_stage(num_heads, num_kv_heads, head_dim, nr_lanes)

    if phase == "decode_paged":
        from nanovllm.backends.tilelang.paged_decode import (
            PAGED_BLOCK_SIZE,
            build_paged_decode_inputs,
            paged_decode_reference,
        )

        inp = build_paged_decode_inputs(
            lengths,
            block_size=PAGED_BLOCK_SIZE,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.float32,
            device="cpu",
            seed=stage.seed,
        )
        reference = paged_decode_reference(
            inp.q,
            inp.k_cache,
            inp.v_cache,
            inp.block_table,
            inp.cache_seqlens,
            inp.block_size,
            softmax_scale=stage.softmax_scale,
        )
        output = torch.zeros_like(reference)
        # Remap unused (-1) slots to 0 for the on-target harness (matches run helper).
        block_table = inp.block_table.clone()
        block_table[block_table < 0] = 0

        for name, tensor, dtype, role in [
            ("q", inp.q, "f32", "input"),
            ("k_cache", inp.k_cache, "f32", "input"),
            ("v_cache", inp.v_cache, "f32", "input"),
            ("block_table", block_table, "int32", "input"),
            ("cache_seqlens", inp.cache_seqlens, "int32", "input"),
            ("output", output, "f32", "output"),
            ("golden", reference, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(
                _tensor_entry(name, dtype, list(tensor.shape), blob, role=role)
            )

        meta = {
            "batch_size": inp.block_table.size(0),
            "num_blocks_pool": inp.num_blocks_pool,
            "block_size": inp.block_size,
            "max_num_blocks": inp.max_num_blocks,
            "decode_block_N": stage.decode_block_N,
            "decode_block_H": stage.decode_block_H,
            "context_lens": list(lengths),
        }
        numel_out = int(reference.numel())
    elif phase == "prefill_paged":
        from nanovllm.backends.tilelang.paged_prefill import (
            PAGED_BLOCK_SIZE,
            build_paged_prefill_inputs,
            paged_prefill_reference,
            prefill_paged_pool_shapes,
            unflatten_prefill_specs,
        )

        specs = unflatten_prefill_specs(lengths)
        inp = build_paged_prefill_inputs(
            specs,
            block_size=PAGED_BLOCK_SIZE,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.float32,
            device="cpu",
            seed=stage.seed,
        )
        reference = paged_prefill_reference(
            inp.q,
            inp.k_cache,
            inp.v_cache,
            inp.block_table,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            softmax_scale=stage.softmax_scale,
            block_size=inp.block_size,
        )
        total_q = int(inp.q.size(0))
        padded_q = _round_up(total_q, stage.prefill_block_M)
        padded_max_q = _round_up(max(inp.max_seqlen_q, 1), stage.prefill_block_M)
        shapes = prefill_paged_pool_shapes(specs, PAGED_BLOCK_SIZE)
        max_ctx = max(cached + new_q for cached, new_q in specs) if specs else 0
        need_blocks = max(1, (max_ctx + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE)
        compile_max_blocks = max(shapes["max_num_blocks"], need_blocks)

        q_pad = inp.q
        if padded_q != total_q:
            q_pad = torch.zeros(
                padded_q, stage.num_heads, stage.head_dim, dtype=inp.q.dtype
            )
            q_pad[:total_q] = inp.q

        block_table = inp.block_table.clone()
        if compile_max_blocks != inp.max_num_blocks:
            padded_bt = torch.zeros(
                (inp.block_table.size(0), compile_max_blocks),
                dtype=torch.int32,
            )
            padded_bt[:, : inp.max_num_blocks] = block_table
            block_table = padded_bt
        block_table[block_table < 0] = 0

        output = torch.zeros(
            padded_q, stage.num_heads, stage.head_dim, dtype=torch.float32
        )
        golden_pad = torch.zeros_like(output)
        golden_pad[:total_q] = reference

        cu_q = inp.cu_seqlens_q.to(torch.int32).contiguous()
        cu_k = inp.cu_seqlens_k.to(torch.int32).contiguous()

        for name, tensor, dtype, role in [
            ("q", q_pad, "f32", "input"),
            ("k_cache", inp.k_cache, "f32", "input"),
            ("v_cache", inp.v_cache, "f32", "input"),
            ("block_table", block_table, "int32", "input"),
            ("cu_seqlens_q", cu_q, "int32", "input"),
            ("cu_seqlens_k", cu_k, "int32", "input"),
            ("output", output, "f32", "output"),
            ("golden", golden_pad, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(
                _tensor_entry(name, dtype, list(tensor.shape), blob, role=role)
            )

        scalar = _scalar_entry("max_seqlen_q", int(padded_max_q))
        by_name = {a["name"]: a for a in args}
        args = [
            by_name["q"],
            by_name["k_cache"],
            by_name["v_cache"],
            by_name["block_table"],
            by_name["cu_seqlens_q"],
            by_name["cu_seqlens_k"],
            scalar,
            by_name["output"],
            by_name["golden"],
        ]
        meta = {
            "batch_size": len(specs),
            "total_q": total_q,
            "padded_q": padded_q,
            "num_blocks_pool": inp.num_blocks_pool,
            "block_size": inp.block_size,
            "max_num_blocks": compile_max_blocks,
            "prefill_block_M": stage.prefill_block_M,
            "prefill_block_N": stage.prefill_block_N,
            "prefill_specs": specs,
            "compare_elements": int(reference.numel()),
        }
        numel_out = int(reference.numel())
    elif phase == "decode":
        ctx = stage.prepare_decode_context(lengths)
        q, k_cache, v_cache = stage.generate_decode_qkv(ctx)
        reference = stage.decode_reference(q, k_cache, v_cache, ctx)
        mask = ctx.mask.to(torch.uint8).contiguous()
        output = torch.zeros_like(reference)

        for name, tensor, dtype, role in [
            ("q", q, "f32", "input"),
            ("k_cache", k_cache, "f32", "input"),
            ("v_cache", v_cache, "f32", "input"),
            ("mask", mask, "uint8", "input"),
            ("output", output, "f32", "output"),
            ("golden", reference, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(
                _tensor_entry(name, dtype, list(tensor.shape), blob, role=role)
            )

        meta = {
            "batch_size": ctx.batch_size,
            "seqlen_kv_padded": ctx.seqlen_kv_padded,
            "decode_block_N": stage.decode_block_N,
            "decode_block_H": stage.decode_block_H,
            "context_lens": list(lengths),
        }
        numel_out = int(reference.numel())
    else:
        ctx = stage.prepare_prefill_context(lengths)
        q, k, v = stage.generate_prefill_qkv(ctx.total_q)
        reference = stage.prefill_reference(q, k, v, ctx)
        padded_q = _round_up(ctx.total_q, stage.prefill_block_M)
        padded_kv = _round_up(ctx.total_kv, stage.prefill_block_N)

        q_pad = q
        k_pad = k
        v_pad = v
        if padded_q != ctx.total_q:
            q_pad = torch.zeros(
                padded_q, stage.num_heads, stage.head_dim, dtype=q.dtype
            )
            q_pad[: ctx.total_q] = q
        if padded_kv != ctx.total_kv:
            k_pad = torch.zeros(
                padded_kv, stage.num_kv_heads, stage.head_dim, dtype=k.dtype
            )
            v_pad = torch.zeros(
                padded_kv, stage.num_kv_heads, stage.head_dim, dtype=v.dtype
            )
            k_pad[: ctx.total_kv] = k
            v_pad[: ctx.total_kv] = v

        # Kernel writes the full padded output; golden is only the real prefix.
        output = torch.zeros(
            padded_q, stage.num_heads, stage.head_dim, dtype=torch.float32
        )
        golden_pad = torch.zeros_like(output)
        golden_pad[: ctx.total_q] = reference

        cu_q = ctx.cu_seqlens_q.to(torch.int32).contiguous()
        cu_k = ctx.cu_seqlens_k.to(torch.int32).contiguous()

        for name, tensor, dtype, role in [
            ("q", q_pad, "f32", "input"),
            ("k", k_pad, "f32", "input"),
            ("v", v_pad, "f32", "input"),
            ("cu_seqlens_q", cu_q, "int32", "input"),
            ("cu_seqlens_k", cu_k, "int32", "input"),
            ("output", output, "f32", "output"),
            ("golden", golden_pad, "f32", "golden"),
        ]:
            blob = f"blobs/{name}.bin"
            _write_blob(os.path.join(case_dir, blob), tensor)
            args.append(
                _tensor_entry(name, dtype, list(tensor.shape), blob, role=role)
            )
        # Packed-arg order: tensors then max_seqlen_q scalar, then output is
        # already in args; insert scalar before output/golden.
        # Kernel call: q,k,v,cu_q,cu_k,max_seqlen_q,(output via out_idx).
        # So scalar goes after cu_seqlens_k and before output.
        scalar = _scalar_entry("max_seqlen_q", int(ctx.max_seqlen_q))
        # Rebuild args in exact packed order.
        by_name = {a["name"]: a for a in args}
        args = [
            by_name["q"],
            by_name["k"],
            by_name["v"],
            by_name["cu_seqlens_q"],
            by_name["cu_seqlens_k"],
            scalar,
            by_name["output"],
            by_name["golden"],
        ]
        meta = {
            "batch_size": len(lengths),
            "total_q": ctx.total_q,
            "total_kv": ctx.total_kv,
            "padded_q": padded_q,
            "padded_kv": padded_kv,
            "prefill_block_M": stage.prefill_block_M,
            "prefill_block_N": stage.prefill_block_N,
            "seq_lens": list(lengths),
            "compare_elements": int(reference.numel()),
        }
        # Compare only the real (unpadded) prefix — see harness compare_offset/count.
        numel_out = int(reference.numel())

    manifest = {
        "case_id": case_id,
        "phase": phase,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "lengths": list(lengths),
        "nr_lanes": nr_lanes,
        "atol": float(atol),
        "softmax_scale": float(stage.softmax_scale),
        "seed": stage.seed,
        "compare_elements": numel_out,
        "meta": meta,
        "args": args,
    }
    manifest_path = os.path.join(case_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return GoldenCase(
        case_id=case_id,
        phase=phase,
        case_dir=case_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def host_sanity_check(golden: GoldenCase) -> float:
    """Compile host CPU kernel and compare to embedded golden; return max_abs_diff."""
    from nanovllm.backends.tilelang.attention import (
        _compile_attention_kernel,
        build_flash_attention_decode_kernel,
        build_flash_attention_prefill_kernel,
        tilelang_dtype,
    )

    m = golden.manifest
    stage = _rvv_stage(
        m["num_heads"], m["num_kv_heads"], m["head_dim"], m["nr_lanes"]
    )
    blob_dir = golden.case_dir

    def load(name: str, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
        path = os.path.join(blob_dir, f"blobs/{name}.bin")
        with open(path, "rb") as handle:
            raw = handle.read()
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()

    if golden.phase == "decode_paged":
        from nanovllm.backends.tilelang.paged_decode import (
            build_flash_attention_decode_paged_kernel,
        )

        by_name = {a["name"]: a for a in m["args"]}
        q = load("q", torch.float32, by_name["q"]["shape"])
        k = load("k_cache", torch.float32, by_name["k_cache"]["shape"])
        v = load("v_cache", torch.float32, by_name["v_cache"]["shape"])
        bt = load("block_table", torch.int32, by_name["block_table"]["shape"])
        cs = load("cache_seqlens", torch.int32, by_name["cache_seqlens"]["shape"])
        golden_t = load("golden", torch.float32, by_name["golden"]["shape"])
        build_args = (
            m["meta"]["batch_size"],
            m["meta"]["num_blocks_pool"],
            m["meta"]["block_size"],
            m["meta"]["max_num_blocks"],
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
        kernel = _compile_attention_kernel(
            build_flash_attention_decode_paged_kernel, build_args, 5, "cpu"
        )
        out = kernel(q, k, v, bt, cs)
        return float((out.float() - golden_t.float()).abs().max().item())

    if golden.phase == "silu_mul":
        from nanovllm.backends.tilelang.activation import (
            _compile_silu_mul_kernel,
        )

        by_name = {a["name"]: a for a in m["args"]}
        x = load("x", torch.float32, by_name["x"]["shape"])
        golden_t = load("golden", torch.float32, by_name["golden"]["shape"])
        meta = m["meta"]
        build_args = (
            meta["num_tokens"],
            meta["intermediate"],
            meta["block_M"],
            128,
            "float32",
        )
        kernel = _compile_silu_mul_kernel(build_args, "cpu")
        out = kernel(x)
        return float((out.float() - golden_t.float()).abs().max().item())

    if golden.phase == "embedding":
        from nanovllm.backends.tilelang.embedding import _compile_embedding_kernel

        by_name = {a["name"]: a for a in m["args"]}
        ids = load("input_ids", torch.int32, by_name["input_ids"]["shape"])
        weight = load("weight", torch.float32, by_name["weight"]["shape"])
        golden_t = load("golden", torch.float32, by_name["golden"]["shape"])
        meta = m["meta"]
        build_args = (
            meta["num_tokens"],
            meta["hidden"],
            meta["vocab"],
            1,
            256,
            "float32",
        )
        kernel = _compile_embedding_kernel(build_args, "cpu")
        out = kernel(ids, weight)
        return float((out.float() - golden_t.float()).abs().max().item())

    if golden.phase == "rmsnorm":
        from nanovllm.backends.tilelang.rmsnorm import _compile_rmsnorm_kernel

        by_name = {a["name"]: a for a in m["args"]}
        x = load("x", torch.float32, by_name["x"]["shape"])
        weight = load("weight", torch.float32, by_name["weight"]["shape"])
        residual = load("residual", torch.float32, by_name["residual"]["shape"])
        golden_t = load("golden", torch.float32, by_name["golden"]["shape"])
        meta = m["meta"]
        build_args = (
            meta["num_tokens"],
            meta["hidden"],
            meta["eps"],
            meta["fuse_residual"],
            1,
            128,
            "float32",
        )
        kernel = _compile_rmsnorm_kernel(build_args, "cpu")
        out, _r = kernel(x, weight, residual)
        return float((out.float() - golden_t.float()).abs().max().item())

    if golden.phase == "linear":
        from nanovllm.backends.tilelang.linear import (
            _compile_linear_kernel,
        )

        by_name = {a["name"]: a for a in m["args"]}
        x = load("x", torch.float32, by_name["x"]["shape"])
        weight = load("weight", torch.float32, by_name["weight"]["shape"])
        bias = load("bias", torch.float32, by_name["bias"]["shape"])
        golden_t = load("golden", torch.float32, by_name["golden"]["shape"])
        meta = m["meta"]
        build_args = (
            meta["padded_m"],
            meta["padded_k"],
            meta["padded_n"],
            meta["has_bias"],
            meta["block_M"],
            meta["block_N"],
            meta["block_K"],
            128,
            "float32",
        )
        kernel = _compile_linear_kernel(build_args, 3, "cpu")
        out = kernel(x, weight, bias)
        mt, nt = meta["num_tokens"], meta["out_features"]
        return float(
            (out[:mt, :nt].float() - golden_t[:mt, :nt].float()).abs().max().item()
        )

    if golden.phase == "rope":
        from nanovllm.backends.tilelang.rope import _compile_rope_kernel

        by_name = {a["name"]: a for a in m["args"]}
        q = load("q", torch.float32, by_name["q"]["shape"])
        k = load("k", torch.float32, by_name["k"]["shape"])
        cos = load("cos", torch.float32, by_name["cos"]["shape"])
        sin = load("sin", torch.float32, by_name["sin"]["shape"])
        golden_q = load("golden", torch.float32, by_name["golden"]["shape"])
        meta = m["meta"]
        tokens = meta["num_tokens"]
        build_args = (
            meta["padded_m"],
            meta["num_heads"],
            meta["num_kv_heads"],
            meta["head_dim"],
            meta["block_M"],
            128,
            "float32",
        )
        kernel = _compile_rope_kernel(build_args, "cpu")
        q_out, k_out = kernel(q, k, cos, sin)
        diff_q = float((q_out[:tokens].float() - golden_q[:tokens].float()).abs().max().item())
        # k_out checked against PyTorch golden recomputed from blobs.
        k_blob = load("k", torch.float32, by_name["k"]["shape"])
        from nanovllm.layers.rotary_embedding import apply_rotary_emb

        k_ref = apply_rotary_emb(k_blob[:tokens], cos[:tokens], sin[:tokens])
        diff_k = float((k_out[:tokens].float() - k_ref.float()).abs().max().item())
        return max(diff_q, diff_k)

    if golden.phase == "kv_store":
        from nanovllm.backends.tilelang.kv_store import _compile_kv_store_kernel

        by_name = {a["name"]: a for a in m["args"]}
        key = load("key", torch.float32, by_name["key"]["shape"])
        value = load("value", torch.float32, by_name["value"]["shape"])
        k_cache = load("k_cache", torch.float32, by_name["k_cache"]["shape"])
        v_cache = load("v_cache", torch.float32, by_name["v_cache"]["shape"])
        slot_mapping = load("slot_mapping", torch.int32, by_name["slot_mapping"]["shape"])
        golden_k = load("golden", torch.float32, by_name["golden"]["shape"])
        with open(os.path.join(blob_dir, "blobs/golden_v.bin"), "rb") as handle:
            golden_v = torch.frombuffer(
                bytearray(handle.read()), dtype=torch.float32
            ).reshape(by_name["v_cache"]["shape"]).clone()
        meta = m["meta"]
        build_args = (
            meta["num_tokens"],
            meta["num_kv_heads"],
            meta["head_dim"],
            meta["num_slots"],
            128,
            "float32",
        )
        kernel = _compile_kv_store_kernel(build_args, "cpu")
        kernel(key, value, k_cache, v_cache, slot_mapping)
        diff_k = float((k_cache.float() - golden_k.float()).abs().max().item())
        diff_v = float((v_cache.float() - golden_v.float()).abs().max().item())
        return max(diff_k, diff_v)

    if golden.phase == "prefill_paged":
        from nanovllm.backends.tilelang.paged_prefill import (
            build_flash_attention_prefill_paged_kernel,
        )

        by_name = {a["name"]: a for a in m["args"]}
        q = load("q", torch.float32, by_name["q"]["shape"])
        k = load("k_cache", torch.float32, by_name["k_cache"]["shape"])
        v = load("v_cache", torch.float32, by_name["v_cache"]["shape"])
        bt = load("block_table", torch.int32, by_name["block_table"]["shape"])
        cu_q = load("cu_seqlens_q", torch.int32, by_name["cu_seqlens_q"]["shape"])
        cu_k = load("cu_seqlens_k", torch.int32, by_name["cu_seqlens_k"]["shape"])
        golden_t = load("golden", torch.float32, by_name["golden"]["shape"])
        max_seqlen_q = next(a["value"] for a in m["args"] if a["name"] == "max_seqlen_q")
        total_q = m["meta"]["total_q"]
        build_args = (
            m["meta"]["batch_size"],
            m["meta"]["padded_q"],
            m["meta"]["num_blocks_pool"],
            m["meta"]["block_size"],
            m["meta"]["max_num_blocks"],
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
        kernel = _compile_attention_kernel(
            build_flash_attention_prefill_paged_kernel, build_args, 7, "cpu"
        )
        out = kernel(q, k, v, bt, cu_q, cu_k, int(max_seqlen_q))[:total_q]
        return float((out.float() - golden_t[:total_q].float()).abs().max().item())

    if golden.phase == "decode":
        q = load("q", torch.float32, m["args"][0]["shape"])
        k = load("k_cache", torch.float32, m["args"][1]["shape"])
        v = load("v_cache", torch.float32, m["args"][2]["shape"])
        mask = load("mask", torch.uint8, m["args"][3]["shape"])
        golden_t = load("golden", torch.float32, m["args"][5]["shape"])
        build_args = (
            m["meta"]["batch_size"],
            m["meta"]["seqlen_kv_padded"],
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
        kernel = _compile_attention_kernel(
            build_flash_attention_decode_kernel, build_args, 4, "cpu"
        )
        out = kernel(q, k, v, mask)
        return float((out.float() - golden_t.float()).abs().max().item())

    q = load("q", torch.float32, next(a["shape"] for a in m["args"] if a["name"] == "q"))
    k = load("k", torch.float32, next(a["shape"] for a in m["args"] if a["name"] == "k"))
    v = load("v", torch.float32, next(a["shape"] for a in m["args"] if a["name"] == "v"))
    cu_q = load(
        "cu_seqlens_q",
        torch.int32,
        next(a["shape"] for a in m["args"] if a["name"] == "cu_seqlens_q"),
    )
    cu_k = load(
        "cu_seqlens_k",
        torch.int32,
        next(a["shape"] for a in m["args"] if a["name"] == "cu_seqlens_k"),
    )
    golden_t = load(
        "golden",
        torch.float32,
        next(a["shape"] for a in m["args"] if a["name"] == "golden"),
    )
    max_seqlen_q = next(a["value"] for a in m["args"] if a["name"] == "max_seqlen_q")
    total_q = m["meta"]["total_q"]
    build_args = (
        m["meta"]["batch_size"],
        m["meta"]["padded_q"],
        m["meta"]["padded_kv"],
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
    kernel = _compile_attention_kernel(
        build_flash_attention_prefill_kernel, build_args, 6, "cpu"
    )
    out = kernel(q, k, v, cu_q, cu_k, int(max_seqlen_q))[:total_q]
    return float((out.float() - golden_t[:total_q].float()).abs().max().item())
