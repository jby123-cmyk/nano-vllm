"""
Linear stage extracted from nano-vllm's ``F.linear`` call sites.

Mirrors ``nanovllm.layers.linear`` compute (``y = x @ weight.T (+ bias)``) with
a pure-PyTorch fp32 golden and the TileLang kernel in
``nanovllm.backends.tilelang.linear``.

Set ``tilelang_backend='cpu'`` and ``device='cpu'`` for RVV numeric validation.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanovllm.backends.tilelang.linear import (
    DEFAULT_BLOCK_K,
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    run_tilelang_linear,
)


@dataclass(frozen=True)
class LinearShape:
    """One engine linear op shape (tp=1)."""

    name: str
    in_features: int
    out_features: int
    has_bias: bool = False


# Qwen3-0.6B tp=1 representative shapes (confirm against hf config when available).
# hidden=1024, intermediate=3072, heads=16, kv_heads=8, head_dim=128
# qkv_out = (16 + 2*8) * 128 = 4096
QWEN3_06B_SHAPES: list[LinearShape] = [
    LinearShape("qkv_proj", 1024, 4096, False),
    LinearShape("o_proj", 2048, 1024, False),
    LinearShape("gate_up_proj", 1024, 6144, False),
    LinearShape("down_proj", 3072, 1024, False),
    # Reduced vocab for CI/Spike affordability (real vocab ~151936).
    LinearShape("lm_head", 1024, 2048, False),
]


@dataclass
class LinearResult:
    shape: LinearShape
    num_tokens: int
    reference_output: torch.Tensor
    tilelang_output: torch.Tensor
    max_abs_diff: float


class LinearStage:
    """Isolated linear GEMM stage with a PyTorch golden ref."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        tilelang_backend: str = "cpu",
        seed: int = 0,
        block_M: int = DEFAULT_BLOCK_M,
        block_N: int = DEFAULT_BLOCK_N,
        block_K: int = DEFAULT_BLOCK_K,
    ):
        if tilelang_backend not in ("cuda", "cpu"):
            raise ValueError(
                f"tilelang_backend must be 'cuda' or 'cpu', got {tilelang_backend!r}."
            )
        if tilelang_backend == "cuda" and device != "cuda":
            raise ValueError("tilelang_backend='cuda' requires device='cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when device='cuda'.")

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.dtype = dtype
        self.device = device
        self.tilelang_backend = tilelang_backend
        self.seed = seed
        self.block_M = block_M
        self.block_N = block_N
        self.block_K = block_K

    def generate_inputs(self, num_tokens: int):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        x = torch.randn(
            num_tokens,
            self.in_features,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        weight = torch.randn(
            self.out_features,
            self.in_features,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        bias = None
        if self.has_bias:
            bias = torch.randn(
                self.out_features,
                device=self.device,
                dtype=self.dtype,
                generator=gen,
            )
        return x, weight, bias

    def reference(self, x, weight, bias=None) -> torch.Tensor:
        """Pure-PyTorch fp32 golden (accurate ground truth)."""
        return F.linear(x.float(), weight.float(), None if bias is None else bias.float())

    def run(self, num_tokens: int) -> LinearResult:
        x, weight, bias = self.generate_inputs(num_tokens)
        reference = self.reference(x, weight, bias)
        tilelang_out = run_tilelang_linear(
            x,
            weight,
            bias,
            backend=self.tilelang_backend,
            block_M=self.block_M,
            block_N=self.block_N,
            block_K=self.block_K,
        )
        max_abs = float((tilelang_out.float() - reference.float()).abs().max().item())
        shape = LinearShape(
            name="custom",
            in_features=self.in_features,
            out_features=self.out_features,
            has_bias=self.has_bias,
        )
        return LinearResult(
            shape=shape,
            num_tokens=num_tokens,
            reference_output=reference.to(self.dtype),
            tilelang_output=tilelang_out,
            max_abs_diff=max_abs,
        )

    @classmethod
    def from_shape(
        cls,
        shape: LinearShape,
        **kwargs,
    ) -> "LinearStage":
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            has_bias=shape.has_bias,
            **kwargs,
        )
