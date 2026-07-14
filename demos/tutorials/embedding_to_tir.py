"""
Low-level TileLang embedding demo (kernel + TIR dump only).

For engine-integrated comparison (tokenize → embed, ref vs TileLang), use:
    python demos/tutorials/compare_tokenize_embed.py

This script remains a minimal compiler tutorial aligned with add_to_tir.py.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

from nanovllm.backends.tilelang.embedding import dump_tensorir, run_tilelang_embedding
from nanovllm.backends.tilelang.weights import load_embedding_weight, load_model_dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile embedding lookup to TensorIR via TileLang (kernel-only demo).",
    )
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--token-block", type=int, default=1)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--random-weights", action="store_true")
    parser.add_argument("--dump-tir", type=str, default="")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vocab, hidden = load_model_dims(args.model, args.vocab, args.hidden)
    dtype = getattr(torch, args.dtype)

    print(
        f"Kernel demo  |  num_tokens={args.num_tokens}  vocab={vocab}  hidden={hidden}  "
        f"dtype={args.dtype}  random_weights={args.random_weights}"
    )

    input_ids = torch.randint(0, vocab, (args.num_tokens,), device="cuda", dtype=torch.int64)
    if args.random_weights:
        weight = torch.randn(vocab, hidden, device="cuda", dtype=dtype)
    else:
        weight = load_embedding_weight(args.model, vocab, hidden, dtype, "cuda")

    dump_tensorir(
        args.num_tokens,
        hidden,
        vocab,
        args.token_block,
        args.threads,
        args.dtype,
        args.dump_tir,
    )

    if args.skip_run:
        print("--skip-run set; stopping after TensorIR dump.")
        return 0

    out = run_tilelang_embedding(
        input_ids,
        weight,
        token_block=args.token_block,
        threads=args.threads,
    )
    expected = F.embedding(input_ids, weight)
    max_diff = (out - expected).abs().max().item()
    print(f"max abs diff vs F.embedding: {max_diff}")

    atol, rtol = (1e-2, 1e-2) if args.dtype == "float16" else (1e-5, 1e-5)
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)
    print("OK — TileLang output matches F.embedding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
