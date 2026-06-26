import os

import torch
import tilelang.language as T
from tilelang import jit


def tilelang_dtype(torch_dtype: torch.dtype) -> str:
    if torch_dtype == torch.float16:
        return "float16"
    if torch_dtype == torch.float32:
        return "float32"
    raise ValueError(
        f"TileLang embedding demo supports float32/float16 weights, got {torch_dtype}. "
        "Cast weights before calling run_tilelang_embedding()."
    )


@jit
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
        with T.Kernel(T.ceildiv(num_tokens, token_block), threads=threads) as bx:
            for ti in T.Parallel(token_block):
                token_idx = bx * token_block + ti
                row = input_ids[token_idx]
                for j in T.serial(hidden):
                    out[token_idx, j] = weight[row, j]

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


def run_tilelang_embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    token_block: int = 1,
    threads: int = 256,
) -> torch.Tensor:
    if input_ids.dim() != 1:
        raise ValueError(f"input_ids must be 1D, got shape {tuple(input_ids.shape)}")

    num_tokens = input_ids.numel()
    vocab, hidden = weight.shape
    in_dtype = tilelang_dtype(weight.dtype)

    if out is None:
        out = torch.empty(num_tokens, hidden, device=weight.device, dtype=weight.dtype)

    kernel = build_embedding_kernel(
        num_tokens, hidden, vocab, token_block, threads, in_dtype
    )
    kernel(input_ids.to(dtype=torch.int32), weight, out)
    return out
