"""RoPE stage with PyTorch golden (apply path only)."""

from dataclasses import dataclass

import torch

from nanovllm.backends.tilelang.rope import run_tilelang_rope
from nanovllm.layers.rotary_embedding import apply_rotary_emb


@dataclass
class RopeResult:
    reference_q: torch.Tensor
    reference_k: torch.Tensor
    tilelang_q: torch.Tensor
    tilelang_k: torch.Tensor
    max_abs_diff: float


class RopeStage:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        tilelang_backend: str = "cpu",
        seed: int = 0,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.tilelang_backend = tilelang_backend
        self.seed = seed

    def run(self, num_tokens: int) -> RopeResult:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        half = self.head_dim // 2
        q = torch.randn(
            num_tokens,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        k = torch.randn(
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        cos = torch.randn(
            num_tokens, 1, half, device=self.device, dtype=self.dtype, generator=gen
        )
        sin = torch.randn(
            num_tokens, 1, half, device=self.device, dtype=self.dtype, generator=gen
        )
        ref_q = apply_rotary_emb(q, cos, sin)
        ref_k = apply_rotary_emb(k, cos, sin)
        tl_q, tl_k = run_tilelang_rope(
            q, k, cos, sin, backend=self.tilelang_backend
        )
        diff = max(
            float((tl_q.float() - ref_q.float()).abs().max().item()),
            float((tl_k.float() - ref_k.float()).abs().max().item()),
        )
        return RopeResult(
            reference_q=ref_q,
            reference_k=ref_k,
            tilelang_q=tl_q,
            tilelang_k=tl_k,
            max_abs_diff=diff,
        )
