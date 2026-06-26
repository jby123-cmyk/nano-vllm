import os
from glob import glob

import torch
from safetensors import safe_open
from transformers import AutoConfig

EMBED_WEIGHT_NAME = "model.embed_tokens.weight"


def load_attention_dims(
    model_path: str,
    num_heads_override: int | None = None,
    num_kv_heads_override: int | None = None,
    head_dim_override: int | None = None,
    tensor_parallel_size: int = 1,
) -> tuple[int, int, int]:
    if os.path.isdir(model_path):
        config = AutoConfig.from_pretrained(model_path)
        total_heads = config.num_attention_heads
        total_kv_heads = config.num_key_value_heads
        head_dim = head_dim_override or getattr(
            config, "head_dim", config.hidden_size // total_heads
        )
        if total_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads={total_heads} is not divisible by "
                f"tensor_parallel_size={tensor_parallel_size}."
            )
        if total_kv_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"num_key_value_heads={total_kv_heads} is not divisible by "
                f"tensor_parallel_size={tensor_parallel_size}."
            )
        num_heads = num_heads_override or total_heads // tensor_parallel_size
        num_kv_heads = num_kv_heads_override or total_kv_heads // tensor_parallel_size
        return num_heads, num_kv_heads, head_dim

    if num_heads_override and num_kv_heads_override and head_dim_override:
        return num_heads_override, num_kv_heads_override, head_dim_override

    raise FileNotFoundError(
        f"Model directory not found: {model_path}. "
        "Pass a valid model path or set num_heads, num_kv_heads, and head_dim overrides."
    )


def load_model_dims(
    model_path: str,
    vocab_override: int | None = None,
    hidden_override: int | None = None,
) -> tuple[int, int]:
    if os.path.isdir(model_path):
        config = AutoConfig.from_pretrained(model_path)
        vocab = vocab_override or config.vocab_size
        hidden = hidden_override or config.hidden_size
        return vocab, hidden

    if vocab_override and hidden_override:
        return vocab_override, hidden_override

    raise FileNotFoundError(
        f"Model directory not found: {model_path}. "
        "Pass a valid model path or set both vocab and hidden overrides."
    )


def load_embedding_weight(
    model_path: str,
    vocab: int,
    hidden: int,
    dtype: torch.dtype,
    device: str = "cuda",
) -> torch.Tensor:
    for file in glob(os.path.join(model_path, "*.safetensors")):
        with safe_open(file, "pt", device="cpu") as handle:
            if EMBED_WEIGHT_NAME in handle.keys():
                weight = handle.get_tensor(EMBED_WEIGHT_NAME)
                if weight.shape != (vocab, hidden):
                    raise ValueError(
                        f"{EMBED_WEIGHT_NAME} has shape {tuple(weight.shape)}, "
                        f"expected ({vocab}, {hidden})."
                    )
                return weight.to(device=device, dtype=dtype)
    raise FileNotFoundError(f"Could not find {EMBED_WEIGHT_NAME} under {model_path}")
