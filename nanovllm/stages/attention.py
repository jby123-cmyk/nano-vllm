"""
Attention stage extracted from the nano-vllm engine, split into independent
prefill and decode paths.

Mirrors the two branches of ``nanovllm.layers.attention.Attention.forward``:

    prefill : flash_attn_varlen_func   -> AttentionStage.run_prefill()
    decode  : flash_attn_with_kvcache  -> AttentionStage.run_decode()

Each path is self-contained so prefill and decode can be compiled, dumped, and
benchmarked separately on different hardware (e.g. GPU vs vector processor).

Set ``tilelang_backend='cpu'`` and ``device='cpu'`` to run the same kernels
through host ``llvm`` for RVV numeric validation (see ``run_attention_rvv_stage.py``).

Golden reference: ``flash_attn`` is the engine op being replaced, but it is not
imported here (it is not required and must not be downloaded). The golden is a
pure-PyTorch attention computed in float32, which is the accurate ground truth
that both ``flash_attn`` and the TileLang kernels approximate.
"""

from dataclasses import dataclass

import torch

from nanovllm.backends.tilelang.attention import (
    dump_decode_tensorir,
    dump_prefill_tensorir,
    run_tilelang_attention_decode,
    run_tilelang_attention_prefill,
    tilelang_dtype,
)
from nanovllm.backends.tilelang.build_dump import (
    dump_attention_decode_build,
    dump_attention_prefill_build,
)
from nanovllm.backends.tilelang.weights import load_attention_dims


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass
class PrefillContext:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    total_q: int
    total_kv: int


@dataclass
class DecodeContext:
    context_lens: torch.Tensor
    mask: torch.Tensor
    batch_size: int
    seqlen_kv_padded: int


@dataclass
class AttentionPrefillResult:
    seq_lens: list[int]
    total_tokens: int
    reference_output: torch.Tensor
    tilelang_output: torch.Tensor
    max_abs_diff: float


@dataclass
class AttentionDecodeResult:
    context_lens: list[int]
    batch_size: int
    reference_output: torch.Tensor
    tilelang_output: torch.Tensor
    max_abs_diff: float


class AttentionStage:
    """
    Isolated FlashAttention stage (prefill + decode) with a PyTorch golden ref.

    Mirrors:
      - model_runner.prepare_prefill() metadata -> prepare_prefill_context()
      - model_runner.prepare_decode() metadata  -> prepare_decode_context()
      - layers.attention.Attention (prefill)    -> run_prefill()
      - layers.attention.Attention (decode)     -> run_decode()
    """

    def __init__(
        self,
        model_path: str | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        softmax_scale: float | None = None,
        seed: int = 0,
        tilelang_backend: str = "cuda",
        prefill_block_M: int = 64,
        prefill_block_N: int = 64,
        prefill_num_stages: int = 1,
        prefill_threads: int = 128,
        decode_block_N: int = 128,
        decode_block_H: int = 64,
        decode_num_stages: int = 2,
        decode_threads: int = 128,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when device='cuda'.")
        if tilelang_backend not in ("cuda", "cpu"):
            raise ValueError(f"tilelang_backend must be 'cuda' or 'cpu', got {tilelang_backend!r}.")
        if tilelang_backend == "cuda" and device != "cuda":
            raise ValueError("tilelang_backend='cuda' requires device='cuda'.")

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.tilelang_backend = tilelang_backend

        self.num_heads, self.num_kv_heads, self.head_dim = load_attention_dims(
            model_path if model_path is not None else "",
            num_heads_override=num_heads,
            num_kv_heads_override=num_kv_heads,
            head_dim_override=head_dim,
        )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be divisible by "
                f"num_kv_heads={self.num_kv_heads}."
            )
        self.group_size = self.num_heads // self.num_kv_heads
        self.softmax_scale = float(softmax_scale or self.head_dim ** -0.5)

        # Validate the dtype eagerly so misuse fails before any compile.
        tilelang_dtype(dtype)

        self.prefill_block_M = prefill_block_M
        self.prefill_block_N = prefill_block_N
        self.prefill_num_stages = prefill_num_stages
        self.prefill_threads = prefill_threads
        self.decode_block_N = decode_block_N
        self.decode_block_H = decode_block_H
        self.decode_num_stages = decode_num_stages
        self.decode_threads = decode_threads

    def _generator(self) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        return gen

    # ------------------------------------------------------------------ #
    # Prefill
    # ------------------------------------------------------------------ #
    def prepare_prefill_context(self, seq_lens: list[int]) -> PrefillContext:
        """Build varlen metadata like model_runner.prepare_prefill (no prefix cache).

        With no prefix cache the per-sequence query and key lengths are equal,
        so cu_seqlens_q == cu_seqlens_k.
        """
        cu = [0]
        for length in seq_lens:
            cu.append(cu[-1] + length)
        cu_tensor = torch.tensor(cu, dtype=torch.int32, device=self.device)
        max_seqlen = max(seq_lens)
        return PrefillContext(
            cu_seqlens_q=cu_tensor,
            cu_seqlens_k=cu_tensor.clone(),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            total_q=cu[-1],
            total_kv=cu[-1],
        )

    def generate_prefill_qkv(
        self, total_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = self._generator()
        q = torch.randn(
            total_tokens, self.num_heads, self.head_dim,
            device=self.device, dtype=self.dtype, generator=gen,
        )
        k = torch.randn(
            total_tokens, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=self.dtype, generator=gen,
        )
        v = torch.randn(
            total_tokens, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=self.dtype, generator=gen,
        )
        return q, k, v

    def prefill_reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: PrefillContext,
    ) -> torch.Tensor:
        """Pure-PyTorch varlen causal GQA in float32 (golden for flash_attn_varlen_func)."""
        cu_q = ctx.cu_seqlens_q.tolist()
        cu_k = ctx.cu_seqlens_k.tolist()
        out = torch.empty_like(q)
        qf, kf, vf = q.float(), k.float(), v.float()
        for b in range(len(cu_q) - 1):
            qs, qe = cu_q[b], cu_q[b + 1]
            ks, ke = cu_k[b], cu_k[b + 1]
            lq, lk = qe - qs, ke - ks
            # (heads, lq, dim) and (heads, lk, dim) after expanding kv heads.
            q_b = qf[qs:qe].transpose(0, 1)
            k_b = kf[ks:ke].repeat_interleave(self.group_size, dim=1).transpose(0, 1)
            v_b = vf[ks:ke].repeat_interleave(self.group_size, dim=1).transpose(0, 1)
            scores = torch.einsum("hid,hjd->hij", q_b, k_b) * self.softmax_scale
            offset = lk - lq
            i_idx = torch.arange(lq, device=self.device).view(1, lq, 1)
            j_idx = torch.arange(lk, device=self.device).view(1, 1, lk)
            causal = j_idx <= (i_idx + offset)
            scores = scores.masked_fill(~causal, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out_b = torch.einsum("hij,hjd->hid", probs, v_b).transpose(0, 1)
            out[qs:qe] = out_b.to(out.dtype)
        return out

    def prefill_tilelang(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: PrefillContext,
    ) -> torch.Tensor:
        return run_tilelang_attention_prefill(
            q, k, v,
            ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
            softmax_scale=self.softmax_scale,
            is_causal=True,
            block_M=self.prefill_block_M,
            block_N=self.prefill_block_N,
            num_stages=self.prefill_num_stages,
            threads=self.prefill_threads,
            backend=self.tilelang_backend,
        )

    def run_prefill(self, seq_lens: list[int]) -> AttentionPrefillResult:
        ctx = self.prepare_prefill_context(seq_lens)
        q, k, v = self.generate_prefill_qkv(ctx.total_q)
        reference = self.prefill_reference(q, k, v, ctx)
        tilelang_out = self.prefill_tilelang(q, k, v, ctx)
        max_abs_diff = (reference.float() - tilelang_out.float()).abs().max().item()
        return AttentionPrefillResult(
            seq_lens=list(seq_lens),
            total_tokens=ctx.total_q,
            reference_output=reference,
            tilelang_output=tilelang_out,
            max_abs_diff=max_abs_diff,
        )

    def dump_prefill_tir(self, seq_lens: list[int], dump_path: str = "") -> str:
        ctx = self.prepare_prefill_context(seq_lens)
        return dump_prefill_tensorir(
            len(seq_lens),
            _round_up(ctx.total_q, self.prefill_block_M),
            _round_up(ctx.total_kv, self.prefill_block_N),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.softmax_scale,
            True,
            self.prefill_block_M,
            self.prefill_block_N,
            self.prefill_num_stages,
            self.prefill_threads,
            tilelang_dtype(self.dtype),
            dump_path,
        )

    def dump_prefill_build(
        self, seq_lens: list[int], build_dir: str | None = None, metadata: dict | None = None
    ) -> str:
        ctx = self.prepare_prefill_context(seq_lens)
        return dump_attention_prefill_build(
            len(seq_lens),
            _round_up(ctx.total_q, self.prefill_block_M),
            _round_up(ctx.total_kv, self.prefill_block_N),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.softmax_scale,
            is_causal=True,
            block_M=self.prefill_block_M,
            block_N=self.prefill_block_N,
            num_stages=self.prefill_num_stages,
            threads=self.prefill_threads,
            in_dtype=tilelang_dtype(self.dtype),
            build_dir=build_dir,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #
    def prepare_decode_context(self, context_lens: list[int]) -> DecodeContext:
        """Build dense-cache decode metadata like model_runner.prepare_decode.

        The KV cache is padded to a multiple of the decode block size; the mask
        marks each sequence's real context length (j < context_len).
        """
        batch_size = len(context_lens)
        max_ctx = max(context_lens)
        seqlen_kv_padded = _round_up(max_ctx, self.decode_block_N)
        context_tensor = torch.tensor(context_lens, dtype=torch.int32, device=self.device)
        positions = torch.arange(seqlen_kv_padded, device=self.device).view(1, seqlen_kv_padded)
        valid = positions < context_tensor.view(batch_size, 1)  # (B, S)
        mask = valid.unsqueeze(-1).expand(batch_size, seqlen_kv_padded, self.num_kv_heads)
        return DecodeContext(
            context_lens=context_tensor,
            mask=mask.to(torch.uint8).contiguous(),
            batch_size=batch_size,
            seqlen_kv_padded=seqlen_kv_padded,
        )

    def generate_decode_qkv(
        self, ctx: DecodeContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = self._generator()
        q = torch.randn(
            ctx.batch_size, self.num_heads, self.head_dim,
            device=self.device, dtype=self.dtype, generator=gen,
        )
        k_cache = torch.randn(
            ctx.batch_size, ctx.seqlen_kv_padded, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=self.dtype, generator=gen,
        )
        v_cache = torch.randn(
            ctx.batch_size, ctx.seqlen_kv_padded, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=self.dtype, generator=gen,
        )
        return q, k_cache, v_cache

    def decode_reference(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        ctx: DecodeContext,
    ) -> torch.Tensor:
        """Pure-PyTorch single-query GQA in float32 (golden for flash_attn_with_kvcache)."""
        context_lens = ctx.context_lens.tolist()
        out = torch.empty_like(q)
        qf, kf, vf = q.float(), k_cache.float(), v_cache.float()
        for b in range(ctx.batch_size):
            length = context_lens[b]
            q_b = qf[b]  # (heads, dim)
            k_b = kf[b, :length].repeat_interleave(self.group_size, dim=1)  # (len, heads, dim)
            v_b = vf[b, :length].repeat_interleave(self.group_size, dim=1)
            scores = torch.einsum("hd,jhd->hj", q_b, k_b) * self.softmax_scale
            probs = torch.softmax(scores, dim=-1)
            out_b = torch.einsum("hj,jhd->hd", probs, v_b)
            out[b] = out_b.to(out.dtype)
        return out

    def decode_tilelang(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        ctx: DecodeContext,
    ) -> torch.Tensor:
        return run_tilelang_attention_decode(
            q, k_cache, v_cache, ctx.mask,
            softmax_scale=self.softmax_scale,
            block_N=self.decode_block_N,
            block_H=self.decode_block_H,
            num_stages=self.decode_num_stages,
            threads=self.decode_threads,
            backend=self.tilelang_backend,
        )

    def run_decode(self, context_lens: list[int]) -> AttentionDecodeResult:
        ctx = self.prepare_decode_context(context_lens)
        q, k_cache, v_cache = self.generate_decode_qkv(ctx)
        reference = self.decode_reference(q, k_cache, v_cache, ctx)
        tilelang_out = self.decode_tilelang(q, k_cache, v_cache, ctx)
        max_abs_diff = (reference.float() - tilelang_out.float()).abs().max().item()
        return AttentionDecodeResult(
            context_lens=list(context_lens),
            batch_size=ctx.batch_size,
            reference_output=reference,
            tilelang_output=tilelang_out,
            max_abs_diff=max_abs_diff,
        )

    def dump_decode_tir(self, context_lens: list[int], dump_path: str = "") -> str:
        ctx = self.prepare_decode_context(context_lens)
        return dump_decode_tensorir(
            ctx.batch_size,
            ctx.seqlen_kv_padded,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.softmax_scale,
            self.decode_block_N,
            self.decode_block_H,
            self.decode_num_stages,
            self.decode_threads,
            tilelang_dtype(self.dtype),
            dump_path,
        )

    def dump_decode_build(
        self, context_lens: list[int], build_dir: str | None = None, metadata: dict | None = None
    ) -> str:
        ctx = self.prepare_decode_context(context_lens)
        return dump_attention_decode_build(
            ctx.batch_size,
            ctx.seqlen_kv_padded,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.softmax_scale,
            block_N=self.decode_block_N,
            block_H=self.decode_block_H,
            num_stages=self.decode_num_stages,
            threads=self.decode_threads,
            in_dtype=tilelang_dtype(self.dtype),
            build_dir=build_dir,
            metadata=metadata,
        )
