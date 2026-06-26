"""
Run the attention stage on CPU and compare TileLang (host llvm) vs PyTorch golden.

Mirrors ``demos/run_attention_stage.py`` but uses ``tilelang_backend='cpu'`` so
the same ``@T.prim_func`` kernels exercised by ``run_rvv_lower.py`` are validated
against the reference without a CUDA GPU.

Prefill:

  python demos/run_attention_rvv_stage.py \\
    --num-heads 8 --num-kv-heads 2 --head-dim 64 --seq-lens 64

Decode:

  python demos/run_attention_rvv_stage.py --decode \\
    --num-heads 8 --num-kv-heads 2 --head-dim 64 --context-lens 64,128
"""

import argparse
import sys

import torch

from nanovllm.backends.tilelang.build_dump import make_build_dir
from nanovllm.backends.tilelang.golden_log import compare_and_log, golden_log_path
from nanovllm.stages.attention import AttentionStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attention stage (CPU llvm): TileLang vs PyTorch golden (RVV numeric path).",
    )
    parser.add_argument("--model", type=str, default="", help="HF model dir for head dims.")
    parser.add_argument("--num-heads", type=int, default=8, help="Query heads override.")
    parser.add_argument("--num-kv-heads", type=int, default=2, help="KV heads override.")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dim override.")
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Run the decode path instead of prefill.",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="64",
        help="Prefill: comma-separated per-sequence lengths.",
    )
    parser.add_argument(
        "--context-lens",
        type=str,
        default="64,128",
        help="Decode: comma-separated per-sequence KV context lengths.",
    )
    parser.add_argument("--scale", type=float, default=None, help="Softmax scale (default head_dim**-0.5).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for q/k/v generation.")
    parser.add_argument("--atol", type=float, default=1e-2, help="assert_close atol (fp32 CPU baseline).")
    parser.add_argument("--rtol", type=float, default=1e-2, help="assert_close rtol (fp32 CPU baseline).")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    args = parse_args()

    stage = AttentionStage(
        model_path=args.model or None,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=torch.float32,
        device="cpu",
        tilelang_backend="cpu",
        softmax_scale=args.scale,
        seed=args.seed,
    )

    phase = "decode" if args.decode else "prefill"
    log_dir = make_build_dir("demos/build_rvv/runs")

    if args.decode:
        context_lens = parse_int_list(args.context_lens)
        result = stage.run_decode(context_lens)
        lengths = result.context_lens
    else:
        seq_lens = parse_int_list(args.seq_lens)
        result = stage.run_prefill(seq_lens)
        lengths = result.seq_lens

    print(f"Attention RVV stage — {phase} (CPU llvm golden check)")
    print(f"  model path      : {args.model or '<overrides>'}")
    print(f"  heads / kv / dim: {stage.num_heads} / {stage.num_kv_heads} / {stage.head_dim}")
    print(f"  group size      : {stage.group_size}")
    print(f"  softmax scale   : {stage.softmax_scale}")
    print(f"  dtype           : float32")
    print(f"  tilelang backend: cpu (llvm + tvm_ffi)")
    print(f"  seq/context lens: {lengths}")
    print(f"  output shape    : {tuple(result.reference_output.shape)}")
    print(f"  max abs diff    : {result.max_abs_diff}")
    print(f"  golden log      : {golden_log_path(log_dir)}")

    compare_and_log(
        log_dir,
        demo=f"run_attention_rvv_stage/{phase}",
        reference="pytorch float32 attention (AttentionStage golden)",
        backend="tilelang host llvm + tvm_ffi (backend=cpu)",
        tilelang_tensor=result.tilelang_output,
        reference_tensor=result.reference_output,
        atol=args.atol,
        rtol=args.rtol,
        max_abs_diff=result.max_abs_diff,
        details={
            "phase": phase,
            "dtype": "float32",
            "num_heads": stage.num_heads,
            "num_kv_heads": stage.num_kv_heads,
            "head_dim": stage.head_dim,
            "softmax_scale": stage.softmax_scale,
            "lengths": lengths,
        },
    )
    print(f"OK — CPU TileLang {phase} matches PyTorch golden reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
