"""
Run the nano-vllm attention stage (prefill or decode) and compare the TileLang
backend against a pure-PyTorch golden reference.

Prefill (default) replaces ``flash_attn_varlen_func``:

  python demos/tutorials/run_attention_stage.py \\
    --num-heads 8 --num-kv-heads 2 --head-dim 64 --seq-lens 5

Decode replaces ``flash_attn_with_kvcache``:

  python demos/tutorials/run_attention_stage.py --decode \\
    --num-heads 8 --num-kv-heads 2 --head-dim 64 --context-lens 17,33

Real Qwen3 dims:

  python demos/tutorials/run_attention_stage.py \\
    --model ~/huggingface/Qwen3-0.6B/ --seq-lens 128,256

Full artifact dump (TIR + host/device codegen) for the chosen phase:

  python demos/tutorials/run_attention_stage.py --seq-lens 128 \\
    --num-heads 8 --num-kv-heads 2 --head-dim 64 --dump-build

Prefill and decode are separate kernels so each can be dumped and benchmarked on
different hardware (e.g. GPU vs vector processor) independently.
"""

import argparse
import sys

import torch

from nanovllm.backends.tilelang.build_dump import make_build_dir
from nanovllm.backends.tilelang.golden_log import compare_and_log, golden_log_path
from nanovllm.stages.attention import AttentionStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the attention stage (prefill/decode): TileLang vs PyTorch golden.",
    )
    parser.add_argument("--model", type=str, default="", help="HF model dir for head dims.")
    parser.add_argument("--num-heads", type=int, default=None, help="Query heads override.")
    parser.add_argument("--num-kv-heads", type=int, default=None, help="KV heads override.")
    parser.add_argument("--head-dim", type=int, default=None, help="Head dim override.")
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Run the decode path (flash_attn_with_kvcache) instead of prefill.",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="128",
        help="Prefill: comma-separated per-sequence lengths (e.g. 128,256).",
    )
    parser.add_argument(
        "--context-lens",
        type=str,
        default="64,128",
        help="Decode: comma-separated per-sequence KV context lengths.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Compute dtype. float16 is the validated tensor-core FlashAttention path.",
    )
    parser.add_argument("--scale", type=float, default=None, help="Softmax scale (default head_dim**-0.5).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for q/k/v generation.")
    parser.add_argument(
        "--dump-tir", type=str, default="", help="Write source TensorIR for the phase to this file."
    )
    parser.add_argument(
        "--dump-build",
        nargs="?",
        const="demos/build",
        default="",
        metavar="DIR",
        help="Dump TIR + host/device codegen under DIR/<UTC-timestamp>/.",
    )
    parser.add_argument("--atol", type=float, default=None, help="assert_close atol override.")
    parser.add_argument("--rtol", type=float, default=None, help="assert_close rtol override.")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    stage = AttentionStage(
        model_path=args.model or None,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        softmax_scale=args.scale,
        seed=args.seed,
    )

    dump_build_dir = make_build_dir(args.dump_build) if args.dump_build else None
    log_dir = dump_build_dir or make_build_dir("demos/build")
    phase = "decode" if args.decode else "prefill"

    if args.decode:
        context_lens = parse_int_list(args.context_lens)
        if dump_build_dir is not None:
            stage.dump_decode_build(
                context_lens, build_dir=dump_build_dir, metadata={"phase": "decode"}
            )
        elif args.dump_tir:
            stage.dump_decode_tir(context_lens, args.dump_tir)
        result = stage.run_decode(context_lens)
        lengths = result.context_lens
    else:
        seq_lens = parse_int_list(args.seq_lens)
        if dump_build_dir is not None:
            stage.dump_prefill_build(
                seq_lens, build_dir=dump_build_dir, metadata={"phase": "prefill"}
            )
        elif args.dump_tir:
            stage.dump_prefill_tir(seq_lens, args.dump_tir)
        result = stage.run_prefill(seq_lens)
        lengths = result.seq_lens

    atol = args.atol if args.atol is not None else (1e-2 if args.dtype == "float16" else 1e-3)
    rtol = args.rtol if args.rtol is not None else (1e-2 if args.dtype == "float16" else 1e-3)

    print(f"Attention stage — {phase}")
    print(f"  model path      : {args.model or '<overrides>'}")
    print(f"  heads / kv / dim: {stage.num_heads} / {stage.num_kv_heads} / {stage.head_dim}")
    print(f"  group size      : {stage.group_size}")
    print(f"  softmax scale   : {stage.softmax_scale}")
    print(f"  dtype           : {args.dtype}")
    print(f"  seq/context lens: {lengths}")
    print(f"  output shape    : {tuple(result.reference_output.shape)}")
    print(f"  max abs diff    : {result.max_abs_diff}")
    print(f"  golden log      : {golden_log_path(log_dir)}")
    if dump_build_dir:
        print(f"  build artifacts : {dump_build_dir}")

    compare_and_log(
        log_dir,
        demo=f"run_attention_stage/{phase}",
        reference="pytorch float32 attention (AttentionStage golden)",
        backend="tilelang cuda jit",
        tilelang_tensor=result.tilelang_output,
        reference_tensor=result.reference_output,
        atol=atol,
        rtol=rtol,
        max_abs_diff=result.max_abs_diff,
        details={
            "phase": phase,
            "dtype": args.dtype,
            "num_heads": stage.num_heads,
            "num_kv_heads": stage.num_kv_heads,
            "head_dim": stage.head_dim,
            "softmax_scale": stage.softmax_scale,
            "lengths": lengths,
        },
    )
    print(f"OK — TileLang {phase} matches PyTorch golden reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
