"""
Compare engine tokenization + embedding against the TileLang backend.

This script runs only the first stage of the nano-vllm pipeline:

  prompt  →  tokenize  →  input_ids  →  embed  →  hidden states (N, H)

Side-by-side:
  - reference: F.embedding (same as VocabParallelEmbedding at tp_size=1)
  - tilelang:  compiled gather kernel from nanovllm.backends.tilelang

Run:
  python demos/tutorials/compare_tokenize_embed.py \\
    --random-weights --vocab 512 --hidden 128 \\
    --token-ids 1,2,3,4,5

  python demos/tutorials/compare_tokenize_embed.py \\
    --model ~/huggingface/Qwen3-0.6B/ \\
    --prompt "introduce yourself" \\
    --dump-build

  python demos/tutorials/compare_tokenize_embed.py \\
    --random-weights --vocab 512 --hidden 128 \\
    --token-ids 1,2,3,4,5 \\
    --dump-build demos/build
"""

import argparse
import os
import sys

import torch

from nanovllm.backends.tilelang.build_dump import make_build_dir
from nanovllm.backends.tilelang.golden_log import compare_and_log, golden_log_path
from nanovllm.stages.tokenize_embed import TokenizeEmbedStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare engine tokenize+embed vs TileLang on the same inputs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        help="HuggingFace model directory (tokenizer + embedding weights).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt (tokenized like LLMEngine.add_request).",
    )
    parser.add_argument(
        "--token-ids",
        type=str,
        default="",
        help="Comma-separated token IDs (for --random-weights or bypassing tokenizer).",
    )
    parser.add_argument(
        "--vocab",
        type=int,
        default=None,
        help="Vocab override (required with --random-weights if --model missing).",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="Hidden-size override (required with --random-weights if --model missing).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Weight/compute dtype. float32 is most reliable for TileLang codegen.",
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="Skip loading real embed_tokens.weight (use random table).",
    )
    parser.add_argument(
        "--token-block",
        type=int,
        default=1,
        help="TileLang tokens-per-block tile.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=256,
        help="TileLang threads per CUDA block.",
    )
    parser.add_argument(
        "--dump-tir",
        type=str,
        default="",
        help="Write source TensorIR only to this file (legacy single-file dump).",
    )
    parser.add_argument(
        "--dump-build",
        nargs="?",
        const="demos/build",
        default="",
        metavar="DIR",
        help=(
            "Dump TIR + host/device codegen under DIR/<UTC-timestamp>/ "
            "(default: demos/build when flag is passed with no value)."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance for assert_close (default: 1e-2 fp16, 1e-5 fp32).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance for assert_close (default: 1e-2 fp16, 1e-5 fp32).",
    )
    return parser.parse_args()


def parse_token_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    if not args.prompt and not args.token_ids:
        if args.random_weights:
            args.token_ids = "0,1,2,3,4,5,6,7"
        else:
            args.prompt = "introduce yourself"

    stage = TokenizeEmbedStage(
        model_path=args.model,
        dtype=dtype,
        random_weights=args.random_weights,
        vocab_override=args.vocab,
        hidden_override=args.hidden,
        token_block=args.token_block,
        threads=args.threads,
    )

    dump_build_dir = None
    log_dir = make_build_dir(args.dump_build if args.dump_build else "demos/build")
    if args.dump_build:
        dump_build_dir = log_dir

    token_ids = parse_token_ids(args.token_ids) if args.token_ids else None
    result = stage.run(
        prompt=args.prompt or None,
        token_ids=token_ids,
        dump_tir_path=args.dump_tir,
        dump_build_dir=dump_build_dir,
    )

    atol = args.atol if args.atol is not None else (1e-2 if args.dtype == "float16" else 1e-5)
    rtol = args.rtol if args.rtol is not None else (1e-2 if args.dtype == "float16" else 1e-5)

    print("Tokenize + embed comparison")
    print(f"  model path     : {args.model}")
    print(f"  random weights : {args.random_weights}")
    print(f"  vocab / hidden : {stage.vocab} / {stage.hidden}")
    print(f"  num tokens     : {len(result.token_ids)}")
    print(f"  token ids      : {result.token_ids[:16]}{'...' if len(result.token_ids) > 16 else ''}")
    print(f"  input_ids shape: {tuple(result.input_ids.shape)}")
    print(f"  hidden shape   : {tuple(result.reference_hidden.shape)}")
    print(f"  max abs diff   : {result.max_abs_diff}")
    print(f"  golden log     : {golden_log_path(log_dir)}")
    if dump_build_dir:
        print(f"  build artifacts: {dump_build_dir}")

    compare_and_log(
        log_dir,
        demo="compare_tokenize_embed",
        reference="torch.nn.functional.embedding (VocabParallelEmbedding tp=1)",
        backend="tilelang cuda jit",
        tilelang_tensor=result.tilelang_hidden,
        reference_tensor=result.reference_hidden,
        atol=atol,
        rtol=rtol,
        max_abs_diff=result.max_abs_diff,
        details={
            "dtype": args.dtype,
            "vocab": stage.vocab,
            "hidden": stage.hidden,
            "num_tokens": len(result.token_ids),
            "token_ids": result.token_ids,
        },
    )
    print("OK — TileLang path matches engine reference (F.embedding / tp=1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
