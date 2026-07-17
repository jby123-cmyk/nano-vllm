"""Render ``main.c`` + ``data.c`` from a golden or execute ``manifest.json``."""

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


def _dtype_tag(dtype: str) -> str:
    if dtype in ("f32", "float32"):
        return "f32"
    if dtype == "int32":
        return "int32"
    if dtype == "uint8":
        return "uint8"
    raise ValueError(f"unsupported tensor dtype {dtype}")


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


def _load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def _packed_args(manifest: dict) -> list[dict]:
    return [a for a in manifest["args"] if a["role"] != "golden"]


def _resolve_output_arg(manifest: dict) -> dict:
    compare_name = manifest.get("compare_tensor")
    if compare_name:
        return next(a for a in manifest["args"] if a["name"] == compare_name)
    return next(a for a in manifest["args"] if a["role"] == "output")


def _emit_tensor_decls(
    manifest: dict,
    case_dir: str,
    *,
    include_golden: bool,
) -> list[str]:
    data_decls: list[str] = []
    for arg in manifest["args"]:
        if arg["kind"] != "tensor":
            continue
        if not include_golden and arg["role"] == "golden":
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
    return data_decls


def _emit_tensor_setup(
    manifest: dict,
    packed: list[dict],
    output: dict,
    *,
    output_names: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    tensor_setup: list[str] = []
    arg_setup: list[str] = []

    names_to_zero: list[str]
    if output_names is not None:
        names_to_zero = list(output_names)
    else:
        names_to_zero = [output["name"]]

    by_name = {a["name"]: a for a in packed}
    for name in names_to_zero:
        arg = by_name[name]
        if arg.get("role") == "output":
            ident = _c_ident(name)
            tensor_setup.append(
                f"  memset({ident}_bytes, 0, sizeof({ident}_bytes));"
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

    return tensor_setup, arg_setup


def _validate_post_kernel(manifest: dict, output: dict) -> tuple[str, str]:
    golden = next(a for a in manifest["args"] if a["role"] == "golden")
    out_ident = _c_ident(output["name"])
    golden_ident = _c_ident(golden["name"])
    defines = "\n".join(
        [
            f"#define GOLDEN_ATOL {float(manifest['atol']):.8g}f",
            f"#define COMPARE_ELEMENTS {int(manifest['compare_elements'])}",
        ]
    )
    body = "\n".join(
        [
            "  HarnessCompareStats stats = harness_compare_float(",
            f"      (const float *){out_ident}_bytes, (const float *){golden_ident}_bytes,",
            "      COMPARE_ELEMENTS, GOLDEN_ATOL);",
            f'  return harness_finish("{manifest["case_id"]}", status, stats, '
            "COMPARE_ELEMENTS, GOLDEN_ATOL);",
        ]
    )
    return defines, body


def _execute_post_kernel(manifest: dict, outputs: list[dict]) -> tuple[str, str]:
    specs: list[str] = []
    for out in outputs:
        ident = _c_ident(out["name"])
        nbytes = 1
        for dim in out["shape"]:
            nbytes *= int(dim)
        nbytes *= _dtype_bytes(out["dtype"])
        specs.append(
            "    {"
            f'"{out["name"]}", "{_dtype_tag(out["dtype"])}", '
            f"{ident}_bytes, {nbytes}"
            "},"
        )
    defines = ""
    body = "\n".join(
        [
            "  HarnessOutputSpec output_specs[] = {",
            *specs,
            "  };",
            f'  return harness_finish_execute("{manifest["case_id"]}", status, '
            f"output_specs, {len(outputs)});",
        ]
    )
    return defines, body


def render_harness(manifest_path: str, case_dir: str | None = None) -> HarnessFiles:
    """Render a golden-compare harness (Spike matrix validation path)."""
    manifest = _load_manifest(manifest_path)
    if manifest.get("mode", "validate") == "execute":
        raise ValueError("execute manifest requires render_execute_harness()")
    case_dir = case_dir or os.path.dirname(manifest_path)

    packed = _packed_args(manifest)
    output = _resolve_output_arg(manifest)
    data_decls = _emit_tensor_decls(manifest, case_dir, include_golden=True)
    tensor_setup, arg_setup = _emit_tensor_setup(manifest, packed, output)
    validate_defines, post_kernel_body = _validate_post_kernel(manifest, output)

    main_c_text = (
        _TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("{{CASE_ID}}", manifest["case_id"])
        .replace("{{PHASE}}", manifest["phase"])
        .replace("{{MODE}}", "validate")
        .replace("{{VALIDATE_DEFINES}}", validate_defines)
        .replace("{{NUM_PACKED_ARGS}}", str(len(packed)))
        .replace("{{DATA_DECLS}}", "\n\n".join(data_decls))
        .replace("{{TENSOR_SETUP}}", "\n".join(tensor_setup))
        .replace("{{ARG_SETUP}}", "\n".join(arg_setup))
        .replace("{{POST_KERNEL_BODY}}", post_kernel_body)
    )

    main_c = os.path.join(case_dir, "main.c")
    data_c = os.path.join(case_dir, "data.c")
    with open(main_c, "w", encoding="utf-8") as handle:
        handle.write(main_c_text)
    with open(data_c, "w", encoding="utf-8") as handle:
        handle.write("/* data embedded in main.c */\n")

    return HarnessFiles(main_c=main_c, data_c=data_c, case_dir=case_dir)


def render_execute_harness(manifest_path: str, case_dir: str | None = None) -> HarnessFiles:
    """Render an execute harness that dumps output tensors over HTIF."""
    manifest = _load_manifest(manifest_path)
    if manifest.get("mode") != "execute":
        raise ValueError("validate manifest requires render_harness()")
    case_dir = case_dir or os.path.dirname(manifest_path)

    packed = _packed_args(manifest)
    output_names = list(manifest["outputs"])
    by_name = {a["name"]: a for a in manifest["args"]}
    outputs = [by_name[name] for name in output_names]
    primary = outputs[0]
    data_decls = _emit_tensor_decls(manifest, case_dir, include_golden=False)
    tensor_setup, arg_setup = _emit_tensor_setup(
        manifest, packed, primary, output_names=output_names
    )
    validate_defines, post_kernel_body = _execute_post_kernel(manifest, outputs)

    main_c_text = (
        _TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("{{CASE_ID}}", manifest["case_id"])
        .replace("{{PHASE}}", manifest["phase"])
        .replace("{{MODE}}", "execute")
        .replace("{{VALIDATE_DEFINES}}", validate_defines)
        .replace("{{NUM_PACKED_ARGS}}", str(len(packed)))
        .replace("{{DATA_DECLS}}", "\n\n".join(data_decls))
        .replace("{{TENSOR_SETUP}}", "\n".join(tensor_setup))
        .replace("{{ARG_SETUP}}", "\n".join(arg_setup))
        .replace("{{POST_KERNEL_BODY}}", post_kernel_body)
    )

    main_c = os.path.join(case_dir, "main.c")
    data_c = os.path.join(case_dir, "data.c")
    with open(main_c, "w", encoding="utf-8") as handle:
        handle.write(main_c_text)
    with open(data_c, "w", encoding="utf-8") as handle:
        handle.write("/* data embedded in main.c */\n")

    return HarnessFiles(main_c=main_c, data_c=data_c, case_dir=case_dir)
