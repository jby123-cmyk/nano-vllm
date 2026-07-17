import os

import torch
import tilelang
import tilelang.language as T
from tilelang import jit
from tilelang import tvm as tvm

from nanovllm.backends.tilelang.attention import (
    _ATTENTION_PASS_CONFIGS,
    tilelang_dtype as _attn_tilelang_dtype,
)


def tilelang_dtype(torch_dtype: torch.dtype) -> str:
    return _attn_tilelang_dtype(torch_dtype)


@jit(
    out_idx=[2],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_embedding_kernel(
    num_tokens: int,
    hidden: int,
    vocab: int,
    token_block: int = 1,
    threads: int = 256,
    in_dtype: str = "float32",
):
    @T.prim_func
    def embedding_kernel(
        input_ids: T.Tensor((num_tokens,), "int32"),
        weight: T.Tensor((vocab, hidden), in_dtype),
        out: T.Tensor((num_tokens, hidden), in_dtype),
    ):
        with T.Kernel(num_tokens, threads=threads) as bx:
            row = input_ids[bx]
            for j in T.Parallel(hidden):
                out[bx, j] = weight[row, j]

    return embedding_kernel


def dump_tensorir(
    num_tokens: int,
    hidden: int,
    vocab: int,
    token_block: int = 1,
    threads: int = 256,
    in_dtype: str = "float32",
    dump_path: str = "",
) -> str:
    tir = build_embedding_kernel.get_tir(
        num_tokens, hidden, vocab, token_block, threads, in_dtype
    )
    tir_text = tir.script()

    print("=" * 72)
    print("TensorIR (TIR) — intermediate representation at the TVM layer")
    print("=" * 72)
    print(tir_text)
    print("=" * 72)

    if dump_path:
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as handle:
            handle.write(tir_text)
        print(f"Saved TensorIR to: {dump_path}")

    return tir_text


from nanovllm.backends.tilelang.runtime import compile_tilelang_kernel


def _compile_embedding_kernel(build_args: tuple, backend: str):
    return compile_tilelang_kernel(
        build_embedding_kernel,
        build_args,
        2,
        backend,
        kernel_name="embedding",
    )


def run_tilelang_embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    token_block: int = 1,
    threads: int = 256,
    backend: str = "cuda",
) -> torch.Tensor:
    if input_ids.dim() != 1:
        raise ValueError(f"input_ids must be 1D, got shape {tuple(input_ids.shape)}")

    num_tokens = input_ids.numel()
    vocab, hidden = weight.shape
    in_dtype = tilelang_dtype(weight.dtype)

    build_args = (num_tokens, hidden, vocab, token_block, threads, in_dtype)
    kernel = _compile_embedding_kernel(build_args, backend)
    result = kernel(input_ids.to(dtype=torch.int32).contiguous(), weight.contiguous())
    if out is not None:
        out.copy_(result)
        return out
    return result
