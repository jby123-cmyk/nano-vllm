import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Never mutate ``logits`` in-place: TileLang host-llvm outputs may alias
        # pooled storage reused by later kernels (report capture keeps them alive).
        scaled = logits.float() / temperatures.unsqueeze(dim=1)
        probs = torch.softmax(scaled, dim=-1)
        gumbel = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return (probs / gumbel).argmax(dim=-1)
