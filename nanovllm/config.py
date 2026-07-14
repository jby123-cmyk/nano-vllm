import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Decode attention backend: "flash_attn" (default) or "tilelang"
    # (paged TileLang kernel matching flash_attn_with_kvcache).
    attn_backend: str = "flash_attn"
    # Linear (F.linear / GEMM) backend: "torch" (default) or "tilelang".
    linear_backend: str = "torch"
    # RMSNorm backend: "torch" (default) or "tilelang".
    norm_backend: str = "torch"
    # SiluAndMul backend: "torch" (default) or "tilelang".
    act_backend: str = "torch"
    # RoPE apply backend: "torch" (default) or "tilelang".
    rope_backend: str = "torch"
    # Embedding gather backend: "torch" (default) or "tilelang".
    embed_backend: str = "torch"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.attn_backend in ("flash_attn", "tilelang"), (
            f"attn_backend must be 'flash_attn' or 'tilelang', got {self.attn_backend!r}"
        )
        assert self.linear_backend in ("torch", "tilelang"), (
            f"linear_backend must be 'torch' or 'tilelang', got {self.linear_backend!r}"
        )
        assert self.norm_backend in ("torch", "tilelang"), (
            f"norm_backend must be 'torch' or 'tilelang', got {self.norm_backend!r}"
        )
        assert self.act_backend in ("torch", "tilelang"), (
            f"act_backend must be 'torch' or 'tilelang', got {self.act_backend!r}"
        )
        assert self.rope_backend in ("torch", "tilelang"), (
            f"rope_backend must be 'torch' or 'tilelang', got {self.rope_backend!r}"
        )
        assert self.embed_backend in ("torch", "tilelang"), (
            f"embed_backend must be 'torch' or 'tilelang', got {self.embed_backend!r}"
        )
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
