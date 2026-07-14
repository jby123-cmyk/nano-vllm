"""Render ``main.c`` + ``data.c`` from a golden ``manifest.json``."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).with_name("templates") / "tvm_main.c.tmpl"


@dataclass
class HarnessFiles:
    main_c: str
    data_c: str
    case_dir: str


def _c_ident(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def _dtype_bytes(dtype: str) -> int:
    return {"f32": 4, "float32": 4, "int32": 4, "uint8": 1, "int64": 8}[dtype]


def _emit_byte_array(name: str, raw: bytes, per_line: int = 16) -> str:
    lines = [f"static unsigned char {name}[] __attribute__((aligned(64), section(\".l2\"))) = {{"]
    for i in range(0, len(raw), per_line):
        chunk = raw[i : i + per_line]
        hexes = ", ".join(f"0x{b:02x}" for b in chunk)
        comma = "," if i + per_line < len(raw) else ""
        lines.append(f"  {hexes}{comma}")
    lines.append("};")
    return "\n".join(lines)


def _make_tensor_call(dtype: str, data_expr: str, shape_var: str, ndim: int) -> str:
    if dtype in ("f32", "float32"):
        return f"make_tensor((float *){data_expr}, {shape_var}, {ndim})"
    if dtype == "int32":
        return f"make_int32_tensor((int32_t *){data_expr}, {shape_var}, {ndim})"
    if dtype == "uint8":
        return f"make_uint8_tensor((uint8_t *){data_expr}, {shape_var}, {ndim})"
    raise ValueError(f"unsupported tensor dtype {dtype}")


def render_harness(manifest_path: str, case_dir: str | None = None) -> HarnessFiles:
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    case_dir = case_dir or os.path.dirname(manifest_path)

    # Packed args exclude golden (compare-only).
    packed = [a for a in manifest["args"] if a["role"] != "golden"]
    golden = next(a for a in manifest["args"] if a["role"] == "golden")
    output = next(a for a in manifest["args"] if a["role"] == "output")

    data_decls: list[str] = []
    tensor_setup: list[str] = []
    arg_setup: list[str] = []

    # Emit blob arrays for every tensor (including golden/output).
    for arg in manifest["args"]:
        if arg["kind"] != "tensor":
            continue
        blob_path = os.path.join(case_dir, arg["blob"])
        with open(blob_path, "rb") as handle:
            raw = handle.read()
        expected = 1
        for d in arg["shape"]:
            expected *= d
        expected *= _dtype_bytes(arg["dtype"])
        if len(raw) != expected:
            raise ValueError(
                f"{arg['name']}: blob size {len(raw)} != expected {expected}"
            )
        ident = _c_ident(arg["name"]) + "_bytes"
        data_decls.append(_emit_byte_array(ident, raw))
        if arg["shape"]:
            shape_lit = ", ".join(str(int(x)) for x in arg["shape"])
            stride_lit = ", ".join(str(int(x)) for x in arg["strides"])
            ndim = len(arg["shape"])
            data_decls.append(
                f"static int64_t {_c_ident(arg['name'])}_shape[{ndim}] "
                f"__attribute__((aligned(64), section(\".l2\"))) = {{{shape_lit}}};"
            )
            data_decls.append(
                f"static int64_t {_c_ident(arg['name'])}_strides[{ndim}] "
                f"__attribute__((aligned(64), section(\".l2\"))) = {{{stride_lit}}};"
            )

    # Zero the output buffer before the kernel runs (blob may already be zeros).
    out_ident = _c_ident(output["name"])
    tensor_setup.append(
        f"  memset({out_ident}_bytes, 0, sizeof({out_ident}_bytes));"
    )

    for i, arg in enumerate(packed):
        if arg["kind"] == "scalar":
            arg_setup.append(f"  args[{i}].type_index = TVM_ARG_INT64;")
            arg_setup.append(f"  args[{i}].padding = 0;")
            arg_setup.append(f"  args[{i}].value.v_int64 = {int(arg['value'])}LL;")
            continue

        ident = _c_ident(arg["name"])
        ndim = len(arg["shape"])
        make = _make_tensor_call(
            arg["dtype"], f"{ident}_bytes", f"{ident}_shape", ndim
        )
        tensor_setup.append(f"  DLTensor {ident}_tensor = {make};")
        tensor_setup.append(f"  {ident}_tensor.strides = {ident}_strides;")
        arg_setup.append(f"  args[{i}].type_index = TVM_ARG_HANDLE;")
        arg_setup.append(f"  args[{i}].padding = 0;")
        arg_setup.append(f"  args[{i}].value.v_handle = &{ident}_tensor;")

    golden_ident = _c_ident(golden["name"])
    main_c_text = (
        _TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("{{CASE_ID}}", manifest["case_id"])
        .replace("{{PHASE}}", manifest["phase"])
        .replace("{{ATOL}}", f"{float(manifest['atol']):.8g}")
        .replace("{{COMPARE_ELEMENTS}}", str(int(manifest["compare_elements"])))
        .replace("{{NUM_PACKED_ARGS}}", str(len(packed)))
        .replace("{{DATA_DECLS}}", "\n\n".join(data_decls))
        .replace("{{TENSOR_SETUP}}", "\n".join(tensor_setup))
        .replace("{{ARG_SETUP}}", "\n".join(arg_setup))
        .replace("{{OUTPUT_PTR}}", f"{out_ident}_bytes")
        .replace("{{GOLDEN_PTR}}", f"{golden_ident}_bytes")
    )

    main_c = os.path.join(case_dir, "main.c")
    data_c = os.path.join(case_dir, "data.c")
    with open(main_c, "w", encoding="utf-8") as handle:
        handle.write(main_c_text)
    # Empty data.c kept for the build API (optional extra source).
    with open(data_c, "w", encoding="utf-8") as handle:
        handle.write("/* data embedded in main.c */\n")

    return HarnessFiles(main_c=main_c, data_c=data_c, case_dir=case_dir)
