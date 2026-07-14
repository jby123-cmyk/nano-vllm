"""SiluAndMul stage with PyTorch golden."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanovllm.backends.tilelang.activation import run_tilelang_silu_mul


@dataclass
class SiluMulResult:
    reference_output: torch.Tensor
    tilelang_output: torch.Tensor
    max_abs_diff: float


class SiluMulStage:
    def __init__(
        self,
        intermediate: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        tilelang_backend: str = "cpu",
        seed: int = 0,
    ):
        self.intermediate = intermediate
        self.dtype = dtype
        self.device = device
        self.tilelang_backend = tilelang_backend
        self.seed = seed

    def run(self, num_tokens: int) -> SiluMulResult:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        x = torch.randn(
            num_tokens,
            self.intermediate * 2,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        gate, up = x.chunk(2, dim=-1)
        ref = (F.silu(gate.float()) * up.float()).to(self.dtype)
        tl = run_tilelang_silu_mul(x, backend=self.tilelang_backend)
        return SiluMulResult(
            reference_output=ref,
            tilelang_output=tl,
            max_abs_diff=float((tl.float() - ref.float()).abs().max().item()),
        )
