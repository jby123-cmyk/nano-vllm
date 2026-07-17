"""
Host-only decoder-layer composition stage.

Chains TileLang ops against a pure-PyTorch fp32 golden for one decode step
(tp=1). Attention uses the dense decode contract (padded KV + mask).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanovllm.backends.tilelang.activation import run_tilelang_silu_mul
from nanovllm.backends.tilelang.attention import run_tilelang_attention_decode
from nanovllm.backends.tilelang.linear import run_tilelang_linear
from nanovllm.backends.tilelang.rmsnorm import run_tilelang_rmsnorm
from nanovllm.backends.tilelang.rope import run_tilelang_rope
from nanovllm.layers.rotary_embedding import apply_rotary_emb
from nanovllm.stages.attention import AttentionStage


@dataclass
class DecoderLayerResult:
    reference_output: torch.Tensor
    tilelang_output: torch.Tensor
    max_abs_diff: float


def _rms(x, weight, eps, residual=None):
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
        residual_out = xf.to(x.dtype)
    else:
        residual_out = None
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = (xf * torch.rsqrt(var + eps)).to(x.dtype) * weight
    if residual is None:
        return y
    return y, residual_out


class DecoderLayerStage:
    """One Qwen3-like decoder layer decode step (fp32, tp=1)."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 512,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        tilelang_backend: str = "cpu",
        seed: int = 0,
        decode_block_N: int = 64,
    ):
        assert hidden_size == num_heads * head_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.eps = eps
        self.dtype = dtype
        self.device = device
        self.tilelang_backend = tilelang_backend
        self.seed = seed
        self.decode_block_N = decode_block_N
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scale = head_dim ** -0.5

    def run_decode_step(
        self, batch_size: int = 1, context_len: int = 64
    ) -> DecoderLayerResult:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        H, I = self.hidden_size, self.intermediate_size
        half = self.head_dim // 2

        def randn(*shape):
            return torch.randn(
                *shape, device=self.device, dtype=self.dtype, generator=gen
            )

        hidden = randn(batch_size, H)
        residual = randn(batch_size, H)
        w_in = randn(H)
        w_qkv = randn(self.q_size + 2 * self.kv_size, H)
        w_qn = randn(self.head_dim)
        w_kn = randn(self.head_dim)
        w_o = randn(H, self.q_size)
        w_post = randn(H)
        w_gu = randn(2 * I, H)
        w_down = randn(H, I)
        cos = randn(batch_size, 1, half)
        sin = randn(batch_size, 1, half)

        # Dense KV cache context (padded); last position held for "current" K/V.
        # AttentionStage only builds dense KV fixtures / PyTorch references here.
        attn_backend = "cpu" if self.tilelang_backend == "rvv" else self.tilelang_backend
        stage = AttentionStage(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
            tilelang_backend=attn_backend,
            seed=self.seed + 1,
            decode_block_N=self.decode_block_N,
            decode_block_H=max(1, self.num_heads // self.num_kv_heads),
        )
        ctx = stage.prepare_decode_context([context_len] * batch_size)
        q_dummy, k_cache, v_cache = stage.generate_decode_qkv(ctx)

        ref = self._run(
            hidden, residual, cos, sin,
            w_in, w_qkv, w_qn, w_kn, w_o, w_post, w_gu, w_down,
            k_cache, v_cache, ctx, tilelang=False,
        )
        tl = self._run(
            hidden, residual, cos, sin,
            w_in, w_qkv, w_qn, w_kn, w_o, w_post, w_gu, w_down,
            k_cache.clone(), v_cache.clone(), ctx, tilelang=True,
        )
        return DecoderLayerResult(
            reference_output=ref,
            tilelang_output=tl,
            max_abs_diff=float((tl.float() - ref.float()).abs().max().item()),
        )

    def _run(
        self,
        hidden, residual, cos, sin,
        w_in, w_qkv, w_qn, w_kn, w_o, w_post, w_gu, w_down,
        k_cache, v_cache, ctx, *, tilelang: bool,
    ):
        b = self.tilelang_backend
        eps = self.eps
        mask = ctx.mask

        if tilelang:
            h, res = run_tilelang_rmsnorm(hidden, w_in, residual, eps=eps, backend=b)
            qkv = run_tilelang_linear(h, w_qkv, None, backend=b)
        else:
            h, res = _rms(hidden, w_in, eps, residual)
            qkv = F.linear(h.float(), w_qkv.float()).to(self.dtype)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if tilelang:
            q = run_tilelang_rmsnorm(q, w_qn, None, eps=eps, backend=b)
            k = run_tilelang_rmsnorm(k, w_kn, None, eps=eps, backend=b)
            q, k = run_tilelang_rope(q, k, cos, sin, backend=b)
        else:
            q = _rms(q, w_qn, eps)
            k = _rms(k, w_kn, eps)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Write current K/V into the last valid cache position per batch row.
        # For simplicity: overwrite position 0 of each sequence's padded cache.
        k_cache = k_cache.clone()
        v_cache = v_cache.clone()
        k_cache[:, 0] = k
        v_cache[:, 0] = v

        if tilelang:
            attn = run_tilelang_attention_decode(
                q, k_cache, v_cache, mask, softmax_scale=self.scale, backend=b,
                block_N=self.decode_block_N,
                block_H=max(1, self.num_heads // self.num_kv_heads),
            )
            o = attn.reshape(attn.size(0), -1)
            h = run_tilelang_linear(o, w_o, None, backend=b)
            h, res = run_tilelang_rmsnorm(h, w_post, res, eps=eps, backend=b)
            gu = run_tilelang_linear(h, w_gu, None, backend=b)
            mid = run_tilelang_silu_mul(gu, backend=b)
            h = run_tilelang_linear(mid, w_down, None, backend=b)
        else:
            attn_stage = AttentionStage(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                device=self.device,
                tilelang_backend="cpu",
                decode_block_N=self.decode_block_N,
                decode_block_H=max(1, self.num_heads // self.num_kv_heads),
            )
            attn = attn_stage.decode_reference(q, k_cache, v_cache, ctx)
            o = attn.reshape(attn.size(0), -1)
            h = F.linear(o.float(), w_o.float()).to(self.dtype)
            h, res = _rms(h, w_post, eps, res)
            gu = F.linear(h.float(), w_gu.float()).to(self.dtype)
            gate, up = gu.chunk(2, -1)
            mid = (F.silu(gate.float()) * up.float()).to(self.dtype)
            h = F.linear(mid.float(), w_down.float()).to(self.dtype)
        return h
