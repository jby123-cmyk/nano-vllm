# TileLang Attention Backend

This document explains the TileLang FlashAttention implementation for nano-vllm:
the math, how it is written in TileLang, how it maps onto Qwen3 / the nano-vllm
engine, and how to run and inspect it.

It covers two independently compiled kernels:

| Phase | Engine op replaced | Kernel builder |
|-------|--------------------|----------------|
| **Prefill** | `flash_attn_varlen_func` | `build_flash_attention_prefill_kernel` |
| **Decode** | `flash_attn_with_kvcache` | `build_flash_attention_decode_kernel` |

Source files:

```
nanovllm/backends/tilelang/attention.py   # kernels + run helpers + TIR dumps
nanovllm/stages/attention.py              # engine-aligned stage + PyTorch golden
nanovllm/backends/tilelang/build_dump.py  # TIR + host/device codegen artifacts
nanovllm/backends/tilelang/rvv_lower.py   # CPU/llvm → RVV artifact harness (Path B)
demos/run_attention_stage.py              # CLI: run / compare / dump (CUDA)
demos/run_rvv_lower.py                    # CLI: lower all kernels to RVV asm (CPU)
```

The **attention math is unchanged** between GPU and CPU: the same `@T.prim_func`
kernels in `attention.py` drive both paths. Path B added CPU realizations for the
GPU-style tile primitives (`alloc_fragment`, `alloc_shared`, `T.gemm`,
`T.reduce_*`, `T.infinity`) inside TileLang so those kernels compile through
`target=llvm` without rewriting the algorithm. See `tilelang.md` for backend
details and §10 below for how to run the RVV artifacts.

---

## 1. Background: attention, FlashAttention, GQA

### 1.1 Scaled dot-product attention

For queries `Q (Sq, D)`, keys `K (Sk, D)`, values `V (Sk, D)`:

```
S      = Q @ Kᵀ * scale            # (Sq, Sk) scores,  scale = 1/sqrt(D)
P      = softmax(S, axis=-1)        # row-wise softmax
O      = P @ V                      # (Sq, D) output
```

For autoregressive decoding the softmax is **causal**: query `i` may only attend
to keys `j <= i`.

### 1.2 FlashAttention online softmax

Materializing the full `(Sq, Sk)` score matrix is memory-bound. FlashAttention
streams over keys in blocks of `block_N` and keeps a running softmax so the score
matrix never leaves on-chip memory. For each key block it maintains, per query
row:

- `m` — running row max (`scores_max`)
- `l` — running denominator / sum of exponentials (`logsum`)
- `acc_o` — running unnormalized output accumulator

When a new block arrives with block max `m_new`, the previously accumulated state
is rescaled by `exp(m_old - m_new)` before adding the new block's contribution.
At the end, `O = acc_o / l`. This is exactly the structure both kernels use.

A small but important trick: instead of `exp(x)`, the kernels compute
`exp2(x * log2(e))`, folding `log2(e)` into the softmax scale so the GPU can use a
single fused-multiply for the exponent:

```31:33:nanovllm/backends/tilelang/attention.py
# exp(x) = exp2(x * log2(e)); folding the constant into the scale lets the
# kernel use a single fused-multiply for the softmax exponent.
LOG2E = 1.44269504
```

### 1.3 Grouped-query attention (GQA)

Qwen3 uses GQA: there are more query heads than key/value heads. Several query
heads share one KV head:

```
group_size = num_heads // num_kv_heads
kv_head_idx = q_head_idx // group_size
```

Qwen3-0.6B is typically `num_heads=16`, `num_kv_heads=8`, `head_dim=128`
(`group_size=2`). The kernels take `num_heads` / `num_kv_heads` as compile-time
constants and index the shared KV head directly — no KV duplication in memory.

---

## 2. Prefill kernel (varlen causal GQA)

### 2.1 What it computes

Prefill processes the prompt: many query tokens per sequence, batched across
sequences with **variable lengths**, packed into one flat tensor (no padding
between sequences). This matches `flash_attn_varlen_func` and
`model_runner.prepare_prefill()`.

Tensor contract:

```107:116:nanovllm/backends/tilelang/attention.py
    """Packed (unpadded) varlen FlashAttention, right-aligned causal masking.

    Tensor contract (matches model_runner.prepare_prefill + qwen3 q/k/v views):
      Q_unpad : (total_q,  num_heads,    head_dim)
      K_unpad : (total_kv, num_kv_heads, head_dim)
      V_unpad : (total_kv, num_kv_heads, head_dim)
      cu_seqlens_q / cu_seqlens_k : (batch_size + 1,) int32 prefix sums
      max_seqlen_q : runtime scalar
      -> Output_unpad : (total_q, num_heads, head_dim)
    """
```

`cu_seqlens_*` are prefix sums of per-sequence lengths. For sequences of length
`[5, 7]`, `cu_seqlens = [0, 5, 12]`, and `total_q = 12`. Sequence `b` occupies
rows `cu_seqlens[b] : cu_seqlens[b+1]`.

### 2.2 Grid: one block per (query tile, head, sequence)

```136:138:nanovllm/backends/tilelang/attention.py
        with T.Kernel(
            T.ceildiv(max_seqlen_q, block_M), num_heads, batch_size, threads=threads
        ) as (bx, by, bz):
```

- `bx` — which `block_M`-sized tile of query rows
- `by` — which query head (`head_idx`); the KV head is `head_idx // group_size`
- `bz` — which sequence in the batch

Each program instance owns one `(block_M × head_dim)` query tile for one head of
one sequence, and streams the whole KV sequence past it.

### 2.3 On-chip buffers

`alloc_shared` lives in shared memory (visible to all threads in the block);
`alloc_fragment` lives in registers/tensor-core fragments. Note the accumulators
are **float32** even when inputs are float16 — this is the accuracy-preserving
mixed-precision pattern.

```139:150:nanovllm/backends/tilelang/attention.py
            Q_shared = T.alloc_shared([block_M, head_dim], in_dtype)
            K_shared = T.alloc_shared([block_N, head_dim], in_dtype)
            V_shared = T.alloc_shared([block_N, head_dim], in_dtype)
            O_shared = T.alloc_shared([block_M, head_dim], in_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], in_dtype)
            acc_o = T.alloc_fragment([block_M, head_dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
```

### 2.4 Right-aligned causal masking

The kernel supports prefix/chunked prefill where the KV sequence is longer than
the query slice (`kv_len > q_len`). It right-aligns the causal mask so the newest
query tokens see the full prefix:

```173:181:nanovllm/backends/tilelang/attention.py
            # Right-align the causal mask so prefix-cache prefill (kv longer than
            # q) keeps the newest query tokens attending to the full prefix.
            offset = kv_current_seqlen - q_current_seqlen
            max_visible_k_idx = offset + (bx + 1) * block_M
            loop_range = (
                T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
                if is_causal
                else T.ceildiv(kv_current_seqlen, block_N)
            )
```

`loop_range` is the causal optimization: a query tile only iterates over KV
blocks it can actually see, skipping fully-masked future blocks.

### 2.5 The streaming loop

The pipelined loop over KV blocks is the heart of FlashAttention. Annotated:

```183:230:nanovllm/backends/tilelang/attention.py
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    K_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :],
                    K_shared,
                )

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i + offset < k * block_N + j)
                            or (bx * block_M + i >= q_current_seqlen or k * block_N + j >= kv_current_seqlen),
                            -1e9,
                            0,
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i >= q_current_seqlen or k * block_N + j >= kv_current_seqlen),
                            -1e9,
                            0,
                        )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, head_dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(
                    V_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :],
                    V_shared,
                )

                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

Step by step inside one iteration:

1. **Load K block** into shared memory (`T.copy`).
2. **Initialize scores with the mask** — masked positions get `-1e9` so they
   vanish after `exp`.
3. **`S = Q @ Kᵀ`** via `T.gemm(..., transpose_B=True)` (tensor cores), added on
   top of the mask init.
4. **Update running max** — `reduce_max` over the block, combine with previous.
5. **Rescale factor** `scores_scale = exp2((m_prev - m_new) * scale)`.
6. **Exponentiate scores** `P = exp2(S * scale - m_new * scale)`.
7. **Update denominator** `logsum = logsum * scores_scale + rowsum(P)`.
8. **Rescale the output accumulator** by `scores_scale` (correct the earlier
   blocks for the new max).
9. **Load V block**, then **`acc_o += P @ V`** via the second `T.gemm`.

### 2.6 Finalize and scatter back

```232:241:nanovllm/backends/tilelang/attention.py
            for i, j in T.Parallel(block_M, head_dim):
                # Queries that can see nothing (right-aligned offset) emit zeros.
                acc_o[i, j] = (
                    0 if is_causal and bx * block_M + i + offset < 0 else acc_o[i, j] / logsum[i]
                )

            T.copy(acc_o, O_shared)
            for i, d in T.Parallel(block_M, head_dim):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]
```

Each row is divided by its denominator, then written back to the packed output —
but only for rows that are real query tokens (`< q_current_seqlen`), so padding
tiles are never written.

---

## 3. Decode kernel (single-query GQA over a KV cache)

### 3.1 What it computes

Decode generates one new token per sequence, so each sequence has exactly **one**
query that attends to its entire cached history. This matches
`flash_attn_with_kvcache` and `model_runner.prepare_decode()`.

Tensor contract:

```266:277:nanovllm/backends/tilelang/attention.py
    """One query token per sequence attending to ``seqlen_kv`` cached keys.

    Tensor contract (matches model_runner.prepare_decode):
      Q    : (batch_size, num_heads,    head_dim)
      K    : (batch_size, seqlen_kv, num_kv_heads, head_dim)
      V    : (batch_size, seqlen_kv, num_kv_heads, head_dim)
      mask : (batch_size, seqlen_kv, num_kv_heads) uint8 (1 = valid kv position)
      -> Output : (batch_size, num_heads, head_dim)

    ``seqlen_kv`` is the padded cache length (multiple of ``block_N``); the mask
    encodes each sequence's real ``context_len``. GQA query heads sharing a kv
    head are batched into the ``block_H`` (M) dimension.
    """
```

Two differences from prefill:

- **Dense, padded KV** with a per-position validity **mask**, rather than packed
  varlen with `cu_seqlens`. (Paged block-table KV is a future step — see §8.)
- **GQA heads are batched into the M dimension.** Since there is only one query
  per sequence, the kernel stacks the `group_size` query heads that share a KV
  head into the `block_H` rows of the GEMM, keeping tensor cores busy.

### 3.2 Grid and head batching

```298:315:nanovllm/backends/tilelang/attention.py
        with T.Kernel(batch_size, num_heads // valid_block_H, threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([block_H, head_dim], in_dtype)
            K_shared = T.alloc_shared([block_N, head_dim], in_dtype)
            V_shared = T.alloc_shared([block_N, head_dim], in_dtype)
            O_shared = T.alloc_shared([valid_block_H, head_dim], in_dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], in_dtype)
            mask_local = T.alloc_fragment([block_N], "uint8")
            acc_o = T.alloc_fragment([block_H, head_dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group_num // valid_block_H)
```

`valid_block_H = min(block_H, group_size)` — the rows of the GEMM are the query
heads sharing `cur_kv_head`.

### 3.3 Streaming loop with mask

```322:348:nanovllm/backends/tilelang/attention.py
            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_shared)
                T.copy(mask[bid, k * block_N : (k + 1) * block_N, cur_kv_head], mask_local)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        mask_local[j] != 0, acc_s[i, j], -T.infinity(accum_dtype)
                    )
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_H, head_dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(V[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

The online-softmax body is identical to prefill; the difference is the masking
source — a loaded `mask_local` vector marking valid KV positions (`j <
context_len`) instead of a computed causal predicate.

---

## 4. TileLang primitives used (quick reference)

| Primitive | Role |
|-----------|------|
| `@T.prim_func` | Declares the kernel; tensor params are typed `T.Tensor(shape, dtype)` |
| `T.Kernel(gx, gy, gz, threads=...)` | Launch grid + threads per block; yields block indices |
| `T.alloc_shared(shape, dtype)` | Shared-memory tile (block-wide); on CPU/`llvm` promoted to **stack `local`** |
| `T.alloc_fragment(shape, dtype)` | Register/tensor-core fragment; on CPU/`llvm` → **`local`** with replicated fragment layout |
| `T.copy(src, dst)` | Bulk copy (global↔shared, fragment↔shared) |
| `T.gemm(A, B, C, transpose_B=, policy=)` | Tensor-core matmul `C += A @ Bᵀ`; on CPU/`llvm` → **`GemmVector`** (`i,k` + `T.parallel(N)`, `vfmacc` on `j`) for all transpose forms (transposed operand → indexed/gather load); `GemmScalar` only for `target=c`; `policy=` ignored |
| `T.Pipelined(n, num_stages=)` | Software-pipelined loop on GPU; **pipeline injection skipped** on `llvm` |
| `T.Parallel(...)` | Thread-parallel elementwise loop |
| `T.reduce_max / T.reduce_sum(x, out, dim=)` | Row reductions |
| `T.exp2`, `T.max`, `T.infinity`, `T.if_then_else` | Math / predicates |
| `T.fill / T.clear` | Initialize buffers |

Two project conventions worth calling out:

- **No `from __future__ import annotations`** in kernel files — it breaks
  TileLang's `@T.prim_func` type parsing (noted at the top of `attention.py`).
- **dtype strings** (`"float16"`, `"float32"`) are accepted everywhere a dtype is
  needed; `tilelang_dtype()` validates and converts from `torch.dtype`.

---

## 5. JIT, compile-time constants, and the run helpers

### 5.1 `@jit` and what is "compile-time"

Each kernel is wrapped with `@jit(out_idx=[...], pass_configs=...)`. The builder's
arguments (shapes, head counts, `softmax_scale`, dtype, tile sizes) are **baked
into the kernel as constants** at compile time; calling the builder with new
values triggers a recompile (cached per argument tuple). `out_idx` marks which
parameter is the output tensor that the helper returns.

```88:106:nanovllm/backends/tilelang/attention.py
@jit(
    out_idx=[6],
    pass_configs=_ATTENTION_PASS_CONFIGS,
)
def build_flash_attention_prefill_kernel(
    batch_size: int,
    total_q: int,
    total_kv: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: float,
    is_causal: bool = True,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    in_dtype: str = "float16",
):
```

### 5.2 Prefill run helper: padding + execute

Because tile sizes are compile-time constants, the packed lengths are rounded up
to a multiple of the block size so the last tile never reads out of bounds. Pad
rows belong to no sequence range (excluded by `cu_seqlens`), so results are
unchanged; the helper slices the real rows back off at the end.

```387:428:nanovllm/backends/tilelang/attention.py
    # Pad packed lengths up to the block size so the final per-sequence block
    # never reads past the allocation. Padding rows belong to no sequence range
    # and are excluded by cu_seqlens, so results are unaffected.
    padded_q = _round_up(total_q, block_M)
    padded_kv = _round_up(total_kv, block_N)

    q_pad = q
    k_pad = k
    v_pad = v
    if padded_q != total_q:
        q_pad = torch.zeros(padded_q, num_heads, head_dim, device=q.device, dtype=q.dtype)
        q_pad[:total_q] = q
    if padded_kv != total_kv:
        k_pad = torch.zeros(padded_kv, num_kv_heads, head_dim, device=k.device, dtype=k.dtype)
        v_pad = torch.zeros(padded_kv, num_kv_heads, head_dim, device=v.device, dtype=v.dtype)
        k_pad[:total_kv] = k
        v_pad[:total_kv] = v

    kernel = build_flash_attention_prefill_kernel(
        batch_size,
        padded_q,
        padded_kv,
        num_heads,
        num_kv_heads,
        head_dim,
        float(softmax_scale),
        is_causal,
        block_M,
        block_N,
        num_stages,
        threads,
        in_dtype,
    )
    out = kernel(
        q_pad,
        k_pad,
        v_pad,
        cu_seqlens_q.to(torch.int32),
        cu_seqlens_k.to(torch.int32),
        int(max_seqlen_q),
    )
    return out[:total_q]
```

The decode helper (`run_tilelang_attention_decode`) is analogous, but requires
`seqlen_kv` to already be a multiple of `block_N` and passes the validity mask.

---

## 6. The TMA / warp-specialization pass config (important)

On TMA-capable GPUs (Hopper sm_90, Blackwell sm_100+/sm_120 — e.g. RTX 5090),
TileLang's warp-specialization pass routes producer global→shared loads through
TMA bulk copies, which need a 64-byte-aligned `CUtensorMap` descriptor. On this
toolchain that descriptor is stack-allocated only 8-byte aligned, so the kernel
**compiles fine but aborts at launch**:

```
Invalid TMA descriptor arguments for __tvm_tensormap_create_tiled:
  tensorMap address must be 64-byte aligned, but got ... mod64=8
```

The fix is a shared pass-config applied to both kernels that disables warp
specialization (removes the TMA *load* path) and TMA-lower (removes the TMA
*store* path), forcing portable cp.async/synchronous copies:

```62:66:nanovllm/backends/tilelang/attention.py
_ATTENTION_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
```

This changes performance only, not numerics. See the full rationale comment in
`attention.py` (lines 35–61).

---

## 7. Running and validating

The stage (`nanovllm/stages/attention.py`) wires the kernels to an
engine-aligned interface and a **pure-PyTorch float32 golden reference**.
`flash_attn` is the engine op being replaced but is intentionally **not**
imported; the fp32 PyTorch math is the accurate ground truth both `flash_attn`
and TileLang approximate.

### 7.1 Environment

```bash
conda activate nanovllm
cd /mnt/ssd/jby123/nano-vllm

# sanity check: CUDA + TileLang available
python -c "import torch; from tilelang import tvm; print('cuda:', torch.cuda.is_available(), 'llvm:', tvm.runtime.enabled('llvm'))"
```

A CUDA GPU is required. `flash_attn` is **not** required.

### 7.2 Compare against the golden (prefill)

```bash
python demos/run_attention_stage.py \
  --num-heads 8 --num-kv-heads 2 --head-dim 64 \
  --seq-lens 5,7
```

`--seq-lens 5,7` means a batch of two sequences (lengths 5 and 7). Expected tail:

```
  max abs diff    : 0.0009765625
OK — TileLang prefill matches PyTorch golden reference.
```

### 7.3 Compare against the golden (decode)

```bash
python demos/run_attention_stage.py --decode \
  --num-heads 8 --num-kv-heads 2 --head-dim 64 \
  --context-lens 17,33
```

`--context-lens 17,33` means two sequences with 17 and 33 cached tokens, each
generating one new token.

### 7.4 Real Qwen3 head dimensions

Point `--model` at a Qwen3 checkpoint to pull `num_heads/num_kv_heads/head_dim`
from its config (Q/K/V are still randomly generated for the kernel test):

```bash
python demos/run_attention_stage.py \
  --model ~/huggingface/Qwen3-0.6B/ --seq-lens 128,256
```

### 7.5 Useful flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--decode` | off (prefill) | Select the decode kernel |
| `--seq-lens a,b,...` | `128` | Prefill per-sequence query lengths |
| `--context-lens a,b,...` | `64,128` | Decode per-sequence KV lengths |
| `--dtype float16\|float32` | `float16` | Compute dtype |
| `--scale` | `head_dim**-0.5` | Softmax scale |
| `--seed` | `0` | Reproducible random Q/K/V |
| `--atol` / `--rtol` | 1e-2 fp16 / 1e-3 fp32 | `assert_close` tolerance |
| `--dump-tir PATH` | off | Write source TIR for the phase |
| `--dump-build [DIR]` | off | Write TIR + host/device codegen tree |

### 7.6 Understanding the numerical difference

A `max abs diff` of `0.0009765625 = 2⁻¹⁰` is exactly **one float16 ULP** for an
output near magnitude 1 — the smallest possible nonzero fp16 error. It comes from
fp16 inputs/output vs the fp32 golden, the `exp2` fast-math identity, and the
streaming-softmax reduction order. To confirm the kernel logic independent of
fp16 rounding, run in float32 (difference should drop to ~1e-6):

```bash
python demos/run_attention_stage.py --decode --dtype float32 \
  --num-heads 8 --num-kv-heads 2 --head-dim 64 --context-lens 17,33
```

---

## 8. Dumping TIR and codegen artifacts

Every kernel can dump its compiler artifacts, which is the primary debugging and
retargeting workflow (prefill and decode are separate kernels, so each dumps its
own tree).

Source TIR only:

```bash
python demos/run_attention_stage.py \
  --num-heads 8 --num-kv-heads 2 --head-dim 64 --seq-lens 128 \
  --dump-tir demos/attention_prefill.tir
```

Full lowered + codegen tree under `demos/build/<UTC-timestamp>/`:

```bash
python demos/run_attention_stage.py \
  --num-heads 8 --num-kv-heads 2 --head-dim 64 --seq-lens 128 \
  --dump-build
```

Produced files:

| File | Meaning |
|------|---------|
| `attention_prefill.tir` / `attention_decode.tir` | Source TensorIR from `@T.prim_func` |
| `host.tir` / `device.tir` | Lowered host/device TIR |
| `device_kernel.cu` | CUDA device source |
| `host.ll` or `host.c` | Host launcher (LLVM IR if TVM built with LLVM, else C) |
| `metadata.json` | Shapes, targets, tile sizes, dtype |
| `README.txt` | Human-readable index |

The portable layer is the **TIR**; CUDA/LLVM are interim targets for bring-up. A
future custom backend adds a new device codegen path from `device.tir` rather
than replacing it.

Programmatic equivalents on the stage:

```python
from nanovllm.stages.attention import AttentionStage

stage = AttentionStage(num_heads=8, num_kv_heads=2, head_dim=64)

# prefill
stage.dump_prefill_tir([128], "demos/attention_prefill.tir")
stage.dump_prefill_build([128], build_dir="demos/build/manual_prefill")
res = stage.run_prefill([128, 256])
print(res.max_abs_diff)

# decode
stage.dump_decode_tir([64, 128], "demos/attention_decode.tir")
res = stage.run_decode([64, 128])
print(res.max_abs_diff)
```

---

## 9. Mapping to the engine, limitations, next steps

### 9.1 Engine mapping

| Engine component | Stage / backend |
|------------------|-----------------|
| `model_runner.prepare_prefill()` varlen metadata | `AttentionStage.prepare_prefill_context()` |
| `model_runner.prepare_decode()` metadata | `AttentionStage.prepare_decode_context()` |
| `Attention` prefill (`flash_attn_varlen_func`) | `AttentionStage.run_prefill()` → prefill kernel |
| `Attention` decode (`flash_attn_with_kvcache`) | `AttentionStage.run_decode()` → decode kernel |

### 9.2 Compatibility / limitations

The attention **math** matches Qwen3 (GQA, causal, head_dim 128, tp=1), but this
is not yet a drop-in engine replacement:

- **Dense padded KV** in decode, not paged `block_tables` + `slot_mapping`.
- **No KV-cache store** (the engine's Triton `store_kvcache` is separate).
- **No prefix/chunked-prefill cache reads** through block tables.
- **tp = 1 only**; **fp16/fp32** only (engine often runs bf16).
- Kernels **recompile per shape** (JIT cache), vs the engine's dynamic batching.

### 9.3 Next steps

1. Paged KV cache store + gather (replace dense decode cache with block tables).
2. Hook `Attention.forward` to the TileLang path behind a flag, preserving the
   `flash_attn` tensor contract and a PyTorch fallback.
3. Remaining Qwen3 ops (RoPE, RMSNorm, SwiGLU MLP, LM head), then `ModelRunner`
   integration with CUDA-graph bypass for compiled ops.

See `handoff.md` §7 for the full roadmap.

---

## 10. CPU / LLVM / RVV backend (Path B)

The FlashAttention kernels were written for CUDA (shared memory, fragments, warp
GEMM policies, software pipelining). **Path B** teaches TileLang's existing
CPU/`llvm` pipeline to lower the *same* kernels to RISC-V Vector assembly — no
rewrite of the online-softmax algorithm. A **wide-vectorization follow-up**
added explicit `GemmVector` loop nests and VLEN-aware `T.copy` widening; see
`tilelang.md` §6 for the full TileLang → TVM → LLVM pipeline.

### 10.1 What changes on CPU (not the math)

| GPU concept | CPU/`llvm` lowering (Path B + wide vectorization) |
|-------------|---------------------------------------------------|
| `T.alloc_shared` (Q/K/V/O tiles) | Promoted to **stack `local`** buffers before tile-op lowering |
| `T.alloc_fragment` (acc_s, acc_o, softmax state) | **`local`** buffers + **fragment layouts** for `LayoutInference` |
| `T.gemm(..., policy=FullRow)` | **`GemmVector`** on `llvm`: `for i,k` + `for j in T.parallel(N)`; LLVM emits `vfmacc` on `j`. Both the non-transposed `P @ V` and the transposed `Q @ K^T` use this nest; `Q @ K^T`'s `B[j,k]` operand lowers to an indexed/gather load (`vluxei`). `GemmScalar` only on `target=c` |
| `T.copy` (K/V/Q tile loads) | **`VectorizePlanner`** widens copy loops using target VLEN (e.g. 64 fp32/step at `block_N=64`, VLEN=4096) — not a fixed 4-wide cap |
| `T.reduce_max` / `T.reduce_sum` | Serial CPU reduce lowering (`src/cpu/op/reduce.cc`) |
| `T.fill(-T.infinity(...))` | CPU fill for `local.fragment` scopes |
| `T.Pipelined(..., num_stages=2)` | Present in source TIR; **pipeline injection disabled** for `llvm` |
| `T.exp2` (softmax) | LLVM/libm `exp2`; not a native RVV transcendental |
| `threads=128` launch | Degenerate 1-wide logical thread; **RVV lanes** from LLVM `VectorizeLoop` + `GemmVector` parallel loops |

Default demo dtype is **`float32`** (`+f,+d` on the RVV target). fp16 needs
`zvfh` and remains GPU-oriented in this repo.

### 10.1b Two vectorization stages (why `+zvl4096b` alone was not enough)

RVV code quality depends on **two** passes, not just LLVM:

1. **TileLang `VectorizePlanner`** (`loop_vectorize.cc`) — runs in the CPU pipeline
   *before* LLVM. It vectorizes `T.copy` loops using `vector-width` /
   `llvm_get_vector_width` (4096 bits for AraXL 4-lane). Without `with rvv:`
   during lowering, this pass falls back to **128 bits** → 4 fp32 copy slices.
2. **LLVM `VectorizeLoop`** — vectorizes elementwise loops and the
   `T.parallel(N)` body from `GemmVector` into scalable
   `<vscale x 4 x float>` and `llvm.fmuladd.nxv4f32`.

`rvv_target(nr_lanes=4)` must set **both** `+zvl4096b` and `vector-width: 4096`.
Use `demos/run_rvv_lower.py --nr-lanes N` to retarget other lane counts.

### 10.2 Prerequisites

```bash
conda activate nanovllm
cd /mnt/ssd/jby123/nano-vllm

# TileLang dev build with Path B patches (see tilelang.md)
export PYTHONPATH=/mnt/ssd/jby123/tilelang/build/python:/mnt/ssd/jby123/tilelang:$PYTHONPATH

python -c "from tilelang import tvm; print('llvm:', tvm.runtime.enabled('llvm'))"
```

A CUDA GPU is **not** required for RVV lowering — only LLVM + the patched
TileLang build.

### 10.3 Lower attention kernels to RVV (recommended)

The Stage 1 ladder script lowers elementwise, matmul, **decode**, and **prefill**
in one shot and writes `demos/build_rvv/STAGE1_REPORT.md`:

```bash
python demos/run_rvv_lower.py
```

With explicit VLEN (default 4 lanes → `+zvl4096b`):

```bash
python demos/run_rvv_lower.py --nr-lanes 4
```

Qwen3-ish dimensions (fp32):

```bash
python demos/run_rvv_lower.py \
  --num-heads 16 --num-kv-heads 8 --head-dim 128 --seqlen-kv 128
```

Artifacts per kernel under `demos/build_rvv/<name>/`:

| File | Contents |
|------|----------|
| `<name>.source.tir` | Pristine `@T.prim_func` from `attention.py` |
| `<name>.tir` | Lowered host TIR after the CPU/`llvm` pipeline |
| `<name>.ll` | LLVM IR (`riscv64-unknown-elf`, `+v`) |
| `<name>.s` | RVV assembly (~7k lines for decode at default dims) |
| `metadata.json` | Target, shapes, detected RVV op families |

Observed RVV families in decode/prefill asm include `vsetvli`, `vle32`, `vse32`,
`vfmacc`, `vfmul`, `vfredosum`, and (in some builds) strided `vlse32` — see
`STAGE1_REPORT.md` for the full scan. After wide vectorization, the default
decode build reports **`strided_ops: []`** in `metadata.json` (unit-stride GEMM
+ wide copies); matmul shows `vfmacc.vv` in the hot loop.

**Lowered attention GEMM shape (decode, `block_N=64`):** both `Q@Kᵀ` and
`P@V` appear in `.tir` as `for i,k` + `for j in T.parallel(64)` after
`GemmVector` lowering — the same nest documented in `tilelang.md` §6.

### 10.4 Lower a single kernel programmatically

```python
from tilelang import tvm
from tilelang.engine.lower import lower_to_host_device_ir, host_codegen
from nanovllm.backends.tilelang.attention import build_flash_attention_decode_kernel
from nanovllm.backends.tilelang.rvv_lower import rvv_target, lower_kernel_rvv

rvv = rvv_target(nr_lanes=4)  # riscv64-unknown-elf, +v,+m,+f,+d,+zvl4096b, vector-width=4096

decode_tir = build_flash_attention_decode_kernel.get_tir(
    1, 128, 8, 2, 64, 0.125, 128, 64, 2, 128, "float32"
)

# Artifact triple + metadata.json
result = lower_kernel_rvv(
    decode_tir, "attention_decode", "demos/build_rvv/attention_decode", rvv
)
print(result.compiled, result.vector_ops)

# Or inspect lowered TIR / LLVM directly:
with rvv:
    host_mod, _, _, tgt, tgt_host = lower_to_host_device_ir(decode_tir, target=rvv, target_host=rvv)
    rt = host_codegen(host_mod, target_host=tgt_host, target=tgt)
    print(rt.inspect_source("asm")[:2000])
```

Use `build_flash_attention_prefill_kernel.get_tir(...)` for prefill; pad lengths
to multiples of `block_M` / `block_N` the same way as the CUDA run helpers (§5.2).

### 10.5 CUDA vs CPU workflows

| Goal | Command / module |
|------|------------------|
| Numeric check vs PyTorch golden (CUDA JIT) | `demos/run_attention_stage.py` |
| Numeric check vs PyTorch golden (CPU llvm, RVV path) | `demos/run_attention_rvv_stage.py` |
| Dump CUDA `.cu` + host codegen | `run_attention_stage.py --dump-build` |
| Lower to RVV `.tir`/`.ll`/`.s` + ladder numeric checks | `demos/run_rvv_lower.py` or `rvv_lower.lower_kernel_rvv` |

RVV artifacts are not executed on RISC-V hardware here. **Numeric validation**
compiles the same `@T.prim_func` on the host via `llvm` + `tvm_ffi` (the
`run_tilelang_attention_{prefill,decode}(..., backend="cpu")` helpers in
`attention.py`) and compares against the same PyTorch float32 golden as the CUDA
stage (`AttentionStage` with `tilelang_backend='cpu'`). `run_rvv_lower.py` runs
this automatically for decode/prefill alongside RVV asm emission.

### 10.6 Known RVV limitations (still true after wide vectorization)

Path B fixed **lowering** (fragments, shared tiles, reduce, infinity). Wide
vectorization fixed the worst **auto-vec gaps** (4-wide copies, scalar GEMM
inner loops). Still open:

- **Not hand-tuned** — LLVM LMUL/VL selection, not AraXL `apps/gemm`-style kernels.
- **`block_N=64` vs VLEN** — tiles smaller than full 256-fp32 vector width per op.
- **Softmax reductions still serial** — `reduce_max`/`reduce_sum` lower to
  horizontal `vfslide1down` shuffles, not lane-parallel reductions. The GEMMs
  vectorize; the reduction axis does not (AraXL Option-A is future work,
  `memory.md` §4.4–4.5).
- **No software `exp2` polynomial** — softmax `exp2` still uses libm/scalar paths.
- **fp16** needs `zvfh`.
- **Strided/gather ops** — the transposed `Q @ K^T` operand load lowers to
  `vluxei64` (in `strided_ops`); expected with AraXL's unit-stride constraint out
  of scope. Layouts may add more with different tile shapes — re-scan after changes.

See `demos/build_rvv/STAGE1_REPORT.md` §6 and `tilelang.md` §7 for follow-ups.
