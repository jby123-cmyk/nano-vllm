# TileLang Backend Handoff

Branch: `jby123/tilelang-demo`  
Last updated: 2026-06-25

This document is the entry point for future agent sessions working on the
hardware-agnostic TileLang backend for nano-vllm. Read this before changing
compiler integration, demos, or engine hooks.

> **Companion docs (read these for the attention / RVV work):**
> - `attention.md` — FlashAttention kernels, CUDA workflow, **§10 CPU/RVV run guide**
> - `tilelang.md` — **Path B** TileLang backend changes (CPU/`llvm` + RVV lowering)
> - `memory.md` — AraXL memory-hierarchy mapping strategy (design rationale)
> - `demos/README.md` — runnable commands for every demo

---

## 1. Project goal

Replace nano-vllm’s **CUDA-targeted PyTorch/Triton/FlashAttention execution path**
with a **compiler-driven backend** built on TileLang → TVM → TensorIR (TIR) →
target-specific codegen, with a long-term target of **hardware-agnostic LLVM**
for host orchestration and retargetable device lowering.

**Milestones done:**
- embedding lookup (`input_ids` + `embed_tokens.weight` → hidden states)
- FlashAttention **prefill** (varlen causal GQA — `flash_attn_varlen_func`)
- FlashAttention **decode** (single-query GQA over a KV cache — `flash_attn_with_kvcache`)
- **CPU / RVV backend (Path B)** — the *same* attention `@T.prim_func` kernels lower
  through TileLang's CPU/`llvm` pipeline to RISC-V Vector (RVV) assembly, with host
  **numeric validation** against the PyTorch golden (see `tilelang.md`, `attention.md` §10)

Prefill and decode are separate kernels/stages so each can be lowered, dumped,
and benchmarked independently across hardware (e.g. GPU vs vector processor).
The CUDA path (`backend="cuda"`) and the CPU/RVV path (`backend="cpu"`) share one
implementation in `backends/tilelang/attention.py` — the kernels never diverge.

**Next milestones:** paged KV cache store/gather (replace the dense decode cache
with block tables), then additional Qwen3 layers until the full inference path
can run through compiled backends. On the RVV side: AraXL-specific layout / unit-
stride copies and Spike/Verilator execution (see `tilelang.md` §7).

---

## 2. What has been built

### 2.1 Repository layout

```
nano-vllm/
├── handoff.md                          # this file (entry point)
├── attention.md                        # FlashAttention kernels + §10 CPU/RVV run guide
├── tilelang.md                         # Path B TileLang CPU/llvm/RVV backend changes
├── memory.md                           # AraXL memory-hierarchy mapping strategy
├── demos/
│   ├── README.md                       # how to run demos
│   ├── add_to_tir.py                   # tutorial: vector add → TIR
│   ├── embedding_to_tir.py             # kernel-only embedding → TIR
│   ├── compare_tokenize_embed.py       # embedding integration demo (ref vs TileLang)
│   ├── run_attention_stage.py          # attention demo: prefill / --decode (CUDA, ref vs TileLang)
│   ├── run_attention_rvv_stage.py      # attention golden check on CPU/llvm (RVV numeric path)
│   ├── run_rvv_lower.py                # Stage 1: lower all kernels to RVV .tir/.ll/.s + numeric
│   ├── build/<timestamp>/              # generated CUDA artifacts (gitignored)
│   └── build_rvv/                      # generated RVV artifacts + STAGE1_REPORT.md (gitignored)
├── nanovllm/
│   ├── engine/                         # unchanged production inference path
│   ├── models/                         # Qwen3 PyTorch model
│   ├── layers/                         # PyTorch ops (attention, embed, …)
│   ├── stages/
│   │   ├── tokenize_embed.py           # isolated tokenize + embed stage
│   │   └── attention.py                # isolated attention stage (prefill + decode; cuda/cpu backend)
│   └── backends/
│       └── tilelang/
│           ├── embedding.py            # TileLang embedding kernel + run helper
│           ├── attention.py            # prefill + decode kernels + run_*(backend="cuda"|"cpu")
│           ├── rvv_lower.py            # RVV target + lower_kernel_rvv() + numeric checks
│           ├── build_dump.py           # TIR + codegen artifact dump (per op/phase)
│           └── weights.py              # model dims + safetensors load
├── tests/
│   └── test_attention_rvv_numeric.py   # parametrized CPU/RVV numeric matrix (pytest)
└── pyproject.toml
```

**Design rule:** library code lives under `nanovllm/`; runnable experiments live
under `demos/`. Do not put compiler backends inside `nanovllm/engine/` until
they are stable and wired into `ModelRunner`.

### 2.2 Engine slice: tokenize + embed

`nanovllm/stages/tokenize_embed.py` extracts the **first stage** of inference
without starting the full engine:

| Step | nano-vllm engine (reference) | Stage method |
|------|------------------------------|--------------|
| Tokenize prompt | `LLMEngine.add_request()` → `AutoTokenizer.encode` | `tokenize()` |
| Build GPU tensor | `ModelRunner.prepare_prefill()` → flat `input_ids` | `prepare_input_ids()` |
| Embedding (default) | `Qwen3Model.forward()` → `VocabParallelEmbedding` | `embed_reference()` |
| Embedding (TileLang) | *(new)* | `embed_tilelang()` |

**Golden reference:** `torch.nn.functional.embedding(input_ids, weight)` — equivalent
to `VocabParallelEmbedding.forward()` when `tensor_parallel_size == 1`.

The stage does **not** run `LLMEngine`, scheduler, KV cache, or attention.

### 2.3 TileLang backend: embedding

`nanovllm/backends/tilelang/embedding.py`:

- `@jit` builder: `build_embedding_kernel(num_tokens, hidden, vocab, …)`
- Kernel semantics: `out[t, j] = weight[input_ids[t], j]` (gather / lookup)
- `run_tilelang_embedding(input_ids, weight)` — JIT compile + execute on GPU
- `dump_tensorir(...)` — source TIR only via `.get_tir()`

Weight loading: `weights.py` reads `model.embed_tokens.weight` from safetensors.

### 2.4 Build artifact dump (TIR-first workflow)

`nanovllm/backends/tilelang/build_dump.py` implements **`dump_embedding_build()`**,
called from the stage when `--dump-build` is passed.

**Policy:** lowering pipelines are **tapped out at the TIR stage for inspection**.
Every new kernel backend should expose TIR dumps at minimum; full builds should
follow the same artifact layout.

Generated folder: `demos/build/<UTC-timestamp>/`

| File | Meaning |
|------|---------|
| `embedding.tir` | Source TensorIR from TileLang `@T.prim_func` |
| `host.tir` | Lowered host-side TIR (launcher, checks) |
| `device.tir` | Lowered device-side TIR (compute) |
| `host.ll` | Host codegen via **LLVM** (when TVM built with `USE_LLVM=ON`) |
| `host.c` | Host codegen fallback when LLVM unavailable |
| `device_kernel.cu` | Device codegen for **CUDA** (current GPU target) |
| `metadata.json` | Shapes, targets, prompt/token_ids for reproducibility |
| `README.txt` | Human-readable index of artifacts |

**Important distinction:**

- **Portable / hardware-agnostic layer:** `embedding.tir`, `host.tir`, `device.tir`
- **Host target today:** LLVM IR (`host.ll`) — CPU launcher that validates tensors and invokes the device kernel
- **Device target today:** CUDA (`device_kernel.cu`) — still required for NVIDIA GPU execution during bring-up

Future custom hardware should add a **new device codegen path from `device.tir`**, not replace TIR.

### 2.5 Comparison demo

`demos/compare_tokenize_embed.py`:

1. Builds `TokenizeEmbedStage` from CLI args
2. Tokenizes prompt (or accepts `--token-ids`)
3. Optionally dumps TIR / full build tree (`--dump-tir`, `--dump-build`)
4. Runs **reference** (`F.embedding`) and **TileLang** on the same tensors
5. Reports `max abs diff` and `torch.testing.assert_close`

This is the **correctness gate**: compiled output must match the golden PyTorch op.

### 2.6 Production engine status

**Not yet integrated.** `ModelRunner`, `Qwen3ForCausalLM`, scheduler, and KV cache
are unchanged. The TileLang path is validated in isolation via `compare_tokenize_embed.py`.

Future hook point: replace or branch inside `VocabParallelEmbedding.forward()` in
`nanovllm/layers/embed_head.py`, preserving the same tensor contract
`(input_ids) → (num_tokens, hidden_size)`.

---

## 3. Architecture

### 3.1 Data flow (embedding milestone)

```
prompt or token_ids
       │
       ▼
  TokenizeEmbedStage.tokenize()          [CPU, HuggingFace — same as engine]
       │
       ▼
  prepare_input_ids()  →  input_ids (N,) int64 on CUDA
       │
       ├──────────────────────────────┐
       ▼                              ▼
  embed_reference()            embed_tilelang()
  F.embedding (golden)         TileLang JIT kernel
       │                              │
       └──────── assert_close ────────┘
                    │
                    ▼
            hidden states (N, H)
```

### 3.2 Compiler flow (TileLang backend)

```
TileLang Python (@T.prim_func)
       │
       ▼
  embedding.tir          ← source TIR (.get_tir) — PRIMARY DEBUG ARTIFACT
       │
       ▼  lower_to_host_device_ir (inside Target context)
  host.tir  +  device.tir
       │
       ├─ host_codegen (LLVM)  →  host.ll
       └─ device_codegen (CUDA) →  device_kernel.cu
       │
       ▼
  TVM runtime / JIT adapter  →  GPU execution
```

### 3.3 Host vs device

| | Host | Device |
|---|------|--------|
| **Hardware** | CPU (x86) | GPU (CUDA today) |
| **Role** | Arg validation, launch, driver API | Embedding gather math |
| **Artifact** | `host.ll` | `device_kernel.cu` |
| **Future** | LLVM stays on host | Retarget `device.tir` to vector cluster / other ISA |

### 3.4 Mapping to nano-vllm inference (full stack)

```
[done in demo]     tokenize → embed
[done in demo]     attention prefill + decode (FlashAttention; dense KV cache)
[done in demo]     same attention kernels → CPU/llvm → RVV asm + host numeric validation
[pending]          paged KV cache store/gather (block tables, Triton store today)
[pending]          MLP, RMSNorm, RoPE, LM head, sampling
[pending]          ModelRunner integration + CUDA graph bypass for compiled ops
[pending]          RVV: AraXL layout/unit-stride copies, Spike/Verilator execution
```

---

## 4. How to run

### 4.1 Environment

Conda env: `nanovllm`

Activate hooks (see `$CONDA_PREFIX/etc/conda/activate.d/`):

- `dev.sh` — gcc-toolset-14, CUDA_HOME, flash-attn build vars
- `tilelang.sh` — `PYTHONPATH=/mnt/ssd/jby123/tilelang`, conda `LD_LIBRARY_PATH`
  (for libstdc++/LLVM), `LLVM_CONFIG`, `CUDAToolkit_ROOT`

TileLang dev build: `/mnt/ssd/jby123/tilelang/build` (LLVM-enabled rebuild)

Verify:

```bash
conda activate nanovllm
python -c "from tilelang import tvm; print('llvm enabled:', tvm.runtime.enabled('llvm'))"
```

Requires `LD_LIBRARY_PATH` including `$CONDA_PREFIX/lib` (set by activate hook).

### 4.2 Commands

```bash
cd /mnt/ssd/jby123/nano-vllm

# Quick correctness check (no model on disk)
python demos/compare_tokenize_embed.py \
  --random-weights --vocab 512 --hidden 128 \
  --token-ids 1,2,3,4,5

# Real Qwen weights + prompt
python demos/compare_tokenize_embed.py \
  --model ~/huggingface/Qwen3-0.6B/ \
  --prompt "introduce yourself"

# Full artifact dump (TIR + host.ll + device_kernel.cu + metadata)
python demos/compare_tokenize_embed.py \
  --model ~/huggingface/Qwen3-0.6B/ \
  --prompt "introduce yourself" \
  --dump-build
```

Use **`float32`** for reliable TileLang CUDA codegen unless explicitly testing fp16.

Rebuild TileLang with `/usr/bin/cmake` (not conda cmake 4.x):

```bash
cd /mnt/ssd/jby123/tilelang/build
/usr/bin/cmake .. -G Ninja \
  -DUSE_CUDA=ON \
  -DUSE_LLVM="$CONDA_PREFIX/bin/llvm-config" \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DTILELANG_USE_CUDA_STUBS=ON
ninja -j$(nproc)
```

---

## 5. Coding practices (follow for new kernels)

These patterns are intentional in the current embedding implementation. Extend them,
do not bypass them.

### 5.1 Directory and documentation

- **One backend per subdirectory:** `nanovllm/backends/tilelang/<op>.py`
- **Stages mirror engine slices:** `nanovllm/stages/<stage>.py` — thin orchestration, no kernel logic
- **Demos are entry points:** `demos/compare_<stage>.py` for integration tests
- **Markdown where it helps:** `demos/README.md` for run commands; `handoff.md` for architecture;
  auto-generated `demos/build/<ts>/README.txt` per compile dump
- **Do not** add `from __future__ import annotations` in TileLang `@T.prim_func` files — breaks type parsing

### 5.2 Separation of concerns

| Module | Responsibility |
|--------|----------------|
| `embedding.py` | Kernel definition + `run_*` + `dump_tensorir` |
| `build_dump.py` | Full lowering artifact dump (TIR stages + codegen text) |
| `weights.py` | Checkpoint I/O only |
| `tokenize_embed.py` | Engine-aligned stage orchestration + golden reference |
| `compare_*.py` | CLI, argparse, pass/fail reporting |

Avoid monolithic demo files that duplicate kernel code.

### 5.3 Correctness-first design

- **Golden reference is always PyTorch** for the op being replaced (`F.embedding` today)
- Run both paths on **identical** `input_ids` and `weight` tensors
- Use `torch.testing.assert_close` with explicit `atol`/`rtol` (fp16 vs fp32)
- Report `max abs diff` before assert for debugging

Do not compare against a second compiled path without a PyTorch ground truth.

### 5.4 TIR dump policy

**Every new lowering entry point must support TIR inspection:**

1. Source TIR: `.get_tir()` on the `@jit` builder
2. Lowered TIR: `host.tir` + `device.tir` via `lower_to_host_device_ir`
3. CLI flags: `--dump-tir <file>` and/or `--dump-build [dir]`

Extend `build_dump.py` or add parallel `build_dump_<op>.py` — do not scatter ad-hoc
print statements in demos.

### 5.5 Robustness without one-off conditionals

Prefer structural checks over scattered special cases:

- **`load_model_dims()`** — single place for config resolution; clear error if model
  path missing and overrides not provided
- **`tilelang_dtype()`** — explicit supported dtypes with one clear `ValueError`
- **Idempotent `PYTHONPATH` in conda activate** — avoid duplicate path stacking
- **Shape validation in kernel builder** — compile-time constants (`num_tokens`, `hidden`,
  `vocab`) baked into TIR; runtime checks in host codegen (see `host.ll` strings)
- **Use dataclasses for stage results** (`TokenizeEmbedResult`) — typed outputs, no tuple soup

When adding attention/KV kernels, encode invariants in data structures (sequence lengths,
block tables, head dims) rather than inline `if` branches copied from call sites.

### 5.6 Engine contract preservation

Compiled ops must match existing tensor layouts:

- `input_ids`: 1D flattened batch (prefill), `int64` in engine, `int32` in TileLang kernel
- `weight`: `(vocab_size, hidden_size)`
- output: `(num_tokens, hidden_size)`

Match `model_runner.prepare_prefill()` / `prepare_decode()` semantics before
integrating into `ModelRunner.run_model()`.

### 5.7 Lazy package imports

`nanovllm/__init__.py` uses lazy exports so `import nanovllm.stages...` does not
pull in `flash_attn` and the full engine.

---

## 6. Known limitations (embedding milestone)

| Limitation | Notes |
|------------|-------|
| `tensor_parallel_size > 1` | Not implemented; reference uses tp=1 `F.embedding` only |
| Dynamic batch size | Kernel recompiles per fixed `num_tokens` at JIT time |
| fp16 device codegen | May fail on some GPU/toolchain combos; fp32 is default for demos |
| No `ModelRunner` hook | Demo-only validation |
| Device codegen still CUDA | `device.tir` is portable; `device_kernel.cu` is interim GPU target |
| Tokenization stays on CPU | Not a TileLang target; keep HuggingFace tokenizer |

---

## 7. Next steps (recommended order)

### 7.1 Integrate embedding into engine (small)

- Add `nanovllm/backends/tilelang/` hook in `VocabParallelEmbedding.forward()` behind a flag
- Preserve fallback to `F.embedding`
- Unit test via existing `compare_tokenize_embed` logic

### 7.2 FlashAttention kernel (large) — DONE (isolated stage)

Reference: `nanovllm/layers/attention.py`

- Prefill (`flash_attn_varlen_func`) → `nanovllm/backends/tilelang/attention.py`
  `build_flash_attention_prefill_kernel` (varlen causal GQA), adapted from
  TileLang `examples/flash_attention/example_gqa_fwd_varlen.py`.
- Decode (`flash_attn_with_kvcache`) → `build_flash_attention_decode_kernel`
  (single-query GQA over a dense padded KV cache + validity mask), adapted from
  TileLang `examples/flash_decoding/example_gqa_decode.py` (no-split variant).
- Metadata mirrors `Context`: prefill builds `cu_seqlens_q/k`; decode builds a
  per-sequence `context_len` mask. (Paged `block_tables`/`slot_mapping` deferred
  to §7.3.)
- Golden reference: pure-PyTorch float32 attention (`flash_attn` NOT imported;
  it is the op being replaced, the PyTorch path is the accurate ground truth).
- TIR + host/device dumps per phase via `dump_attention_prefill_build` /
  `dump_attention_decode_build` and the `--dump-build` / `--dump-tir` flags.
- Validation: `demos/run_attention_stage.py` (prefill default, `--decode` flag).
- Default dtype is **float16** here (tensor-core FlashAttention), unlike the
  embedding milestone where fp32 is default. Tolerances: 1e-2 fp16 / 1e-3 fp32.

### 7.2b CPU / RVV backend (Path B) — DONE

The *same* prefill/decode `@T.prim_func` kernels lower through TileLang's CPU/`llvm`
pipeline to RISC-V Vector assembly (no algorithm rewrite). Full details in
`tilelang.md`; run guide in `attention.md` §10.

- `run_tilelang_attention_{prefill,decode}(..., backend="cpu")` compile via host
  `llvm` + `tvm_ffi`; `AttentionStage(tilelang_backend="cpu", device="cpu")` drives
  the engine-aligned path without a GPU.
- `demos/run_rvv_lower.py` emits the `.tir`/`.ll`/`.s` artifact triple per kernel
  (elementwise, matmul, attention decode/prefill) and writes `demos/build_rvv/STAGE1_REPORT.md`.
- **Numeric validation:** the riscv64 RVV `.s` is not host-runnable, so correctness
  is gated on the identical host-`llvm` lowering vs the PyTorch golden —
  `demos/run_attention_rvv_stage.py` and `tests/test_attention_rvv_numeric.py`.
- Everything here is **fp32** (fp16 needs `zvfh`). Requires the Path B TileLang dev
  build (see `tilelang.md` §5).

### 7.3 Paged KV cache store + gather (medium)

Reference: Triton `store_kvcache_kernel` in `layers/attention.py`

- Replace the decode stage's dense padded KV cache with paged block tables.
- Map slot indices → physical KV blocks (`block_manager.py`, `model_runner.py`).
- Store kernel (write k/v at `slot_mapping`) likely separate from the matmul;
  decode kernel then gathers KV via `block_tables` (see TileLang
  `examples/blocksparse_attention/example_tilelang_sparse_gqa_decode_paged.py`).
- Validate against Triton reference output.

### 7.4 Remaining Qwen3 layers

RoPE, RMSNorm, SwiGLU MLP, LM head — prioritize by isolation difficulty and time on GPU.

### 7.5 Full hardware-agnostic backend

End state:

```
TileLang kernels  →  device.tir  →  target-specific codegen
                                   ├─ CUDA (interim NVIDIA)
                                   ├─ HIP / Metal / …
                                   └─ custom vector cluster (future)
Host LLVM launcher  →  host.ll    →  CPU orchestration (stable)
Golden PyTorch ops  →  correctness gate per kernel
ModelRunner         →  dispatches compiled backends instead of PyTorch modules
```

---

## 8. Files to read first (new agent checklist)

1. `handoff.md` (this file)
2. `demos/README.md`
3. `attention.md` — FlashAttention math + §10 CPU/RVV run guide
4. `tilelang.md` — Path B TileLang CPU/llvm/RVV backend (read before touching lowering)
5. `nanovllm/stages/attention.py` — prefill + decode stage (cuda/cpu backend)
6. `nanovllm/backends/tilelang/attention.py` — prefill + decode kernels + run helpers
7. `nanovllm/backends/tilelang/rvv_lower.py` — RVV target + lower + numeric checks
8. `nanovllm/backends/tilelang/build_dump.py`
9. `demos/run_attention_stage.py`, `demos/run_attention_rvv_stage.py`, `demos/run_rvv_lower.py`
10. `tests/test_attention_rvv_numeric.py` — numeric matrix across shapes
11. `memory.md` — AraXL hardware-mapping strategy (future RVV tuning)
12. `nanovllm/layers/attention.py` / `embed_head.py` — integration targets
13. `nanovllm/engine/model_runner.py` — eventual wiring point

---

## 9. Success criteria (per kernel)

Before merging new backend work:

- [ ] Golden PyTorch reference identified and documented
- [ ] TileLang kernel matches reference (`max abs diff` within tolerance)
- [ ] Source + lowered TIR dumped (`--dump-build` or equivalent)
- [ ] Host artifact generated (`host.ll` with LLVM build)
- [ ] Device artifact generated (CUDA or new target)
- [ ] `metadata.json` captures compile-time constants
- [ ] Demo script under `demos/compare_*.py`
- [ ] No regression to lazy `nanovllm` imports
- [ ] `demos/README.md` updated with run commands

---

## 10. Related paths outside this repo

| Path | Purpose |
|------|---------|
| `/mnt/ssd/jby123/tilelang` | TileLang + TVM dev source and `build/` |
| `~/huggingface/Qwen3-0.6B/` | Default model checkpoint for demos |
| `$CONDA_PREFIX/etc/conda/activate.d/` | Runtime env for LLVM + TileLang |

---

*This handoff reflects the embedding + FlashAttention (CUDA) + CPU/RVV (Path B)
milestones on branch `jby123/tilelang-demo`. Update this file when adding kernels,
changing artifact layout, or integrating into `ModelRunner`.*
