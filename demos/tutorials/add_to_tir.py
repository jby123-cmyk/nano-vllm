"""
Reference demo: vector add (C = A + B) compiled through TileLang to TensorIR.

This file has the SAME STRUCTURE as embedding_to_tir.py, but a simpler
operation so you can learn the tools first. Once you understand this file,
swap "vector add" for "embedding lookup" in each section.

Run:
    python demos/tutorials/add_to_tir.py
    python demos/tutorials/add_to_tir.py --size 4096 --dump-tir demos/add.tir
    python demos/tutorials/add_to_tir.py --skip-run   # only print TensorIR, no GPU exec
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# SECTION 1: Imports
# ---------------------------------------------------------------------------
# torch       — create tensors on GPU and compute the "correct answer"
# tilelang    — compiler front-end; turns your kernel description into TIR
# tilelang.language (T) — DSL keywords: @T.prim_func, T.Tensor, T.Kernel, ...
# tilelang.jit — decorator that wraps your builder and exposes .get_tir()
import torch
import tilelang
import tilelang.language as T
from tilelang import jit


# ---------------------------------------------------------------------------
# SECTION 2: Command-line arguments (argparse)
# ---------------------------------------------------------------------------
# Lets you change settings without editing code.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile vector add (C = A + B) to TensorIR via TileLang.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Number of elements in each vector (like --num-tokens in embedding demo).",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=256,
        help="GPU threads per block (tile size for the kernel).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Element type for A, B, C.",
    )
    parser.add_argument(
        "--dump-tir",
        type=str,
        default="",
        help="If set, write TensorIR text to this file (e.g. demos/add.tir).",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only build and print TensorIR; do not compile/run on GPU.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SECTION 3: Problem setup (like loading config + weights in embedding demo)
# ---------------------------------------------------------------------------
# Here we skip HuggingFace config — vector add has no model.
# We just pick a size and dtype, then create random input tensors.
def make_tensors(size: int, dtype_name: str, device: str = "cuda"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo. No GPU found.")

    dtype = getattr(torch, dtype_name)
    a = torch.randn(size, device=device, dtype=dtype)
    b = torch.randn(size, device=device, dtype=dtype)
    c = torch.empty(size, device=device, dtype=dtype)
    return a, b, c


# ---------------------------------------------------------------------------
# SECTION 4: TileLang kernel definition
# ---------------------------------------------------------------------------
# Pattern:
#   @jit outer function  →  returns @T.prim_func inner function
#
# The OUTER function takes compile-time constants (size, block, dtype).
# The INNER @T.prim_func describes the GPU kernel signature and body.
#
# In embedding_to_tir.py you would replace this with:
#   input_ids (N,) + weight (vocab, H) → out (N, H)
@jit
def build_add_kernel(size: int, block: int = 256, in_dtype: str = "float32"):
    # Use in_dtype (not "dtype") in T.Tensor annotations — matches TileLang examples.
    # Do NOT add `from __future__ import annotations` at the top of this file; it breaks
    # TileLang's eager @T.prim_func type parsing (NameError: name 'dtype' is not defined).
    @T.prim_func
    def add_kernel(
        A: T.Tensor((size,), in_dtype),
        B: T.Tensor((size,), in_dtype),
        C: T.Tensor((size,), in_dtype),
    ):
        # Launch one grid of blocks; each block has `block` threads.
        with T.Kernel(T.ceildiv(size, block), threads=block) as bx:
            # Each thread handles one element (when in range).
            for i in T.Parallel(block):
                global_index = bx * block + i
                C[global_index] = A[global_index] + B[global_index]

    return add_kernel


# ---------------------------------------------------------------------------
# SECTION 5: Dump TensorIR (main deliverable for compiler demo)
# ---------------------------------------------------------------------------
# .get_tir() asks TileLang to build the PrimFunc WITHOUT codegen yet.
# .script() turns that PrimFunc into human-readable text (TensorIR).
def dump_tensorir(jit_builder, size: int, block: int, dtype: str, dump_path: str) -> str:
    tir = jit_builder.get_tir(size, block, dtype)
    tir_text = tir.script()

    print("=" * 72)
    print("TensorIR (TIR) — intermediate representation at the TVM layer")
    print("=" * 72)
    print(tir_text)
    print("=" * 72)

    if dump_path:
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            f.write(tir_text)
        print(f"Saved TensorIR to: {dump_path}")

    return tir_text


# ---------------------------------------------------------------------------
# SECTION 6: Compile, run, and verify against PyTorch
# ---------------------------------------------------------------------------
# PyTorch reference = the "golden" answer we trust.
# TileLang kernel   = what we are trying to prove compiles and runs correctly.
def run_and_verify(jit_builder, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                   size: int, block: int, dtype: str) -> None:
    # Calling the @jit builder returns a compiled kernel (first call compiles).
    kernel = jit_builder(size, block, dtype)
    kernel(a, b, c)

    # Golden reference: plain PyTorch element-wise add.
    expected = a + b

    max_diff = (c - expected).abs().max().item()
    print(f"max abs diff vs PyTorch (a + b): {max_diff}")

    atol, rtol = (1e-2, 1e-2) if dtype == "float16" else (1e-5, 1e-5)
    torch.testing.assert_close(c, expected, atol=atol, rtol=rtol)
    print("OK — TileLang output matches PyTorch reference.")


# ---------------------------------------------------------------------------
# SECTION 7: Main entry point — wires all sections together
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    print(f"Demo: vector add  |  size={args.size}  block={args.block}  dtype={args.dtype}")
    print()

    # Step A: create inputs (embedding demo: input_ids + weight table)
    a, b, c = make_tensors(args.size, args.dtype)

    # Step B: show TensorIR before running (embedding demo: same call pattern)
    dump_tensorir(build_add_kernel, args.size, args.block, args.dtype, args.dump_tir)

    if args.skip_run:
        print("--skip-run set; stopping after TensorIR dump.")
        return 0

    # Step C: compile + run + compare (embedding demo: compare to F.embedding)
    run_and_verify(build_add_kernel, a, b, c, args.size, args.block, args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
