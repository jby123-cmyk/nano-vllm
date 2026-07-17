#!/usr/bin/env python3
"""Generate tiny Qwen3-shaped checkpoints under ``tests/fixtures/``."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, Qwen3Config
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nanovllm.backends.spike.engine_golden import (  # noqa: E402
    TINY_QWEN_PATH,
    tiny_fixture_path,
    tiny_fixture_spec,
    write_tiny_config,
)
from nanovllm.models.qwen3 import Qwen3ForCausalLM  # noqa: E402


def _build_tokenizer(path: str, *, vocab_size: int) -> None:
    vocab = {f"tok{i}": i for i in range(3, vocab_size)}
    vocab.update({"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="tok3"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|eos|>",
        pair="$A $B <|eos|>",
        special_tokens=[("<|eos|>", 2)],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        unk_token="tok3",
    )
    fast.save_pretrained(path)


def _export_safetensors(model: Qwen3ForCausalLM, config: Qwen3Config, path: str) -> None:
    q_rows = config.num_attention_heads * config.head_dim
    k_rows = config.num_key_value_heads * config.head_dim
    tensors: dict[str, torch.Tensor] = {}
    for key, weight in model.state_dict().items():
        if key.endswith("qkv_proj.weight"):
            prefix = key[: -len("qkv_proj.weight")]
            tensors[prefix + "q_proj.weight"] = weight[:q_rows].contiguous()
            tensors[prefix + "k_proj.weight"] = weight[q_rows : q_rows + k_rows].contiguous()
            tensors[prefix + "v_proj.weight"] = weight[q_rows + k_rows :].contiguous()
            continue
        if key.endswith("gate_up_proj.weight"):
            prefix = key[: -len("gate_up_proj.weight")]
            half = weight.shape[0] // 2
            tensors[prefix + "gate_proj.weight"] = weight[:half].contiguous()
            tensors[prefix + "up_proj.weight"] = weight[half:].contiguous()
            continue
        tensors[key] = weight.contiguous()
    save_file(tensors, path)


def build_tiny_qwen_fixture(
    *,
    out_dir: str | os.PathLike[str] | None = None,
    seed: int = 0,
    num_layers: int = 1,
) -> str:
    spec = tiny_fixture_spec(num_layers=num_layers)
    out = os.path.abspath(out_dir or str(tiny_fixture_path(num_layers=num_layers)))
    os.makedirs(out, exist_ok=True)
    write_tiny_config(out, spec)

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29501", rank=0, world_size=1)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    config = Qwen3Config.from_pretrained(out)
    torch.set_default_dtype(torch.float32)
    model = Qwen3ForCausalLM(config)
    for param in model.parameters():
        param.data.normal_(mean=0.0, std=0.02, generator=gen)

    _export_safetensors(model, config, os.path.join(out, "model.safetensors"))
    _build_tokenizer(out, vocab_size=int(spec["vocab_size"]))

    meta = {
        "seed": seed,
        "num_layers": num_layers,
        "spec": spec,
        "prompt_tokens": [10, 11, 12],
    }
    with open(os.path.join(out, "fixture_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if dist.is_initialized():
        dist.destroy_process_group()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of decoder layers in the fixture.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Fixture output directory (default: tests/fixtures/tiny_qwen[_2l]).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or str(tiny_fixture_path(num_layers=args.num_layers))
    out = build_tiny_qwen_fixture(out_dir=out_dir, seed=args.seed, num_layers=args.num_layers)
    print(f"Wrote tiny Qwen fixture ({args.num_layers} layers) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
