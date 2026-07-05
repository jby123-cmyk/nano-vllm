# Demos

Scripts for learning TileLang and comparing compiled backends against the
nano-vllm engine. These do **not** replace full inference (`example.py`).

## Layout

```
demos/
  add_to_tir.py              # Tutorial: vector add → TensorIR
  embedding_to_tir.py        # Tutorial: embedding kernel → TensorIR (random input_ids)
  compare_tokenize_embed.py  # Engine stage: tokenize + embed, ref vs TileLang
  run_attention_stage.py     # Engine stage: FlashAttention prefill / decode, ref vs TileLang (CUDA)
  run_attention_rvv_stage.py # Same golden check on CPU llvm (RVV numeric validation path)
  run_rvv_lower.py           # Stage 1: TileLang → LLVM → RVV (.tir/.ll/.s) demo

nanovllm/
  stages/tokenize_embed.py   # Extracted engine stage (tokenize → embed)
  stages/attention.py        # Extracted engine stage (FlashAttention prefill + decode)
  backends/tilelang/         # TileLang kernels + weight loading

tests/
  test_attention_rvv_numeric.py  # pytest: CPU/RVV numeric matrix vs PyTorch golden
```

## Which script to run

| Goal | Command |
|------|---------|
| Learn TileLang basics | `python demos/add_to_tir.py --skip-run` |
| Dump embedding TensorIR only | `python demos/embedding_to_tir.py --random-weights --vocab 512 --hidden 128 --skip-run --dump-tir demos/embedding.tir` |
| Dump TIR + codegen to `demos/build/<timestamp>/` | `python demos/compare_tokenize_embed.py ... --dump-build` |
| **Compare engine vs TileLang** | `python demos/compare_tokenize_embed.py --random-weights --vocab 512 --hidden 128 --token-ids 1,2,3,4,5` |
| **Attention prefill** | `python demos/run_attention_stage.py --num-heads 8 --num-kv-heads 2 --head-dim 64 --seq-lens 128,256` |
| **Attention decode** | `python demos/run_attention_stage.py --decode --num-heads 8 --num-kv-heads 2 --head-dim 64 --context-lens 64,128` |
| **Dump attention TIR + codegen** | `python demos/run_attention_stage.py --num-heads 8 --num-kv-heads 2 --head-dim 64 --seq-lens 128 --dump-build` |
| **Attention golden on CPU (RVV path)** | `python demos/run_attention_rvv_stage.py --seq-lens 64` or `--decode --context-lens 64` |
| **RVV Stage 1 lowering (`.tir`/`.ll`/`.s`)** | `python demos/run_rvv_lower.py` |
| **CPU/RVV numeric matrix (pytest)** | `pytest tests/test_attention_rvv_numeric.py` |

Each comparison run writes a per-build **`golden.log`** (pass/fail, max abs diff,
atol/rtol, and the full error on failure). Paths:

- `demos/build/<timestamp>/golden.log` — embedding / CUDA attention demos
- `demos/build_rvv/<kernel>/golden.log` — each kernel in `run_rvv_lower.py`
- `demos/build_rvv/runs/<timestamp>/golden.log` — `run_attention_rvv_stage.py`

> The pytest matrix and the RVV demos require the Path B TileLang dev build on
> `PYTHONPATH` (see `tilelang.md` §5); they skip cleanly if LLVM/TileLang is absent.

## Build artifact dump (`--dump-build`)

`compare_tokenize_embed.py --dump-build` writes a timestamped folder:

```
demos/build/20250621_132000/
  embedding.tir      # source TensorIR
  host.tir           # lowered host TIR
  device.tir         # lowered device TIR
  device_kernel.cu   # CUDA device source
  host.c             # host launcher (C; LLVM not enabled in this TVM build)
  golden.log         # PyTorch golden comparison result for this run
  metadata.json
  README.txt
```

If TVM is built with LLVM, `host.ll` is written instead of `host.c`.

## RVV Stage 1 lowering (`run_rvv_lower.py`)

Drives TileLang kernels through the **CPU/`llvm` pass pipeline** configured for
an AraXL **RISC-V Vector (RVV)** target (mirroring `AraXL/tvm-apps`:
`riscv64-unknown-elf`, `+v,+m,+f,+d`, `lp64d`), with LLVM auto-vectorization
on, and emits the **`.tir` / `.ll` / `.s`** artifact triple per kernel.

**Path B** TileLang backend work (see `tilelang.md`) added CPU realizations for
GPU-style tile ops so FlashAttention decode/prefill lower end-to-end. LLVM still
owns vector scheduling — no custom LMUL tuning or AraXL subtarget. Everything
runs in **fp32** (fp16 needs `zvfh`).

```bash
python demos/run_rvv_lower.py                       # defaults: AraXL RVV target
python demos/run_rvv_lower.py --mattr +v,+m,+f,+d   # override target attrs
python demos/run_rvv_lower.py --num-heads 16 --num-kv-heads 8 --head-dim 128
```

Kernel ladder, written under `demos/build_rvv/<name>/`:

| Kernel | Status | Notes |
|--------|--------|-------|
| `elementwise` | compiles + vectorizes | unit-stride `vle`/`vse`/`vfadd` |
| `matmul` (`T.gemm`) | compiles + vectorizes | `GemmVector` (`i,k` + parallel `j`) → `vfmacc` |
| `attention_decode` | compiles + vectorizes | FlashAttention decode; both GEMMs (incl. `Q@Kᵀ`) via `GemmVector`; numeric pass (host llvm) |
| `attention_prefill` | compiles + vectorizes | FlashAttention prefill; both GEMMs (incl. `Q@Kᵀ`) via `GemmVector`; numeric pass (host llvm) |

```
demos/build_rvv/
  <name>/<name>.tir         # lowered TensorIR (annotated source TIR if lowering failed)
  <name>/<name>.source.tir  # pristine source TIR
  <name>/<name>.ll          # LLVM IR for the RVV target (if compiled)
  <name>/<name>.s           # RVV assembly (if compiled)
  <name>/metadata.json
  <name>/golden.log           # PyTorch golden comparison for this kernel
  <name>/README.txt
  STAGE1_REPORT.md          # per-kernel table + observed limitations
```

The attention kernels use the **same** `@T.prim_func` sources as the CUDA path
(`attention.md`); Path B taught TileLang to lower their GPU tile primitives on
CPU/`llvm`. `STAGE1_REPORT.md` records per-kernel asm evidence and remaining
RVV limitations (`vlse32`, scalar `exp2`, etc.). See `tilelang.md` for backend
details.

## Engine mapping

| Engine component | Stage / backend |
|------------------|-----------------|
| `LLMEngine.add_request()` tokenization | `TokenizeEmbedStage.tokenize()` |
| `ModelRunner.prepare_prefill()` → `input_ids` | `TokenizeEmbedStage.prepare_input_ids()` |
| `VocabParallelEmbedding` (tp=1) | `TokenizeEmbedStage.embed_reference()` |
| TileLang backend (new) | `TokenizeEmbedStage.embed_tilelang()` |
| `ModelRunner.prepare_prefill()` varlen metadata | `AttentionStage.prepare_prefill_context()` |
| `ModelRunner.prepare_decode()` metadata | `AttentionStage.prepare_decode_context()` |
| `layers.attention.Attention` prefill (`flash_attn_varlen_func`) | `AttentionStage.run_prefill()` |
| `layers.attention.Attention` decode (`flash_attn_with_kvcache`) | `AttentionStage.run_decode()` |

The attention golden reference is pure PyTorch (float32); `flash_attn` is the
engine op being replaced but is not imported here. Decode uses a dense, padded
KV cache (per-sequence `context_len` mask) rather than the paged block table —
paged KV cache store/gather is a separate follow-on (handoff §7.3).

Full inference (scheduler, sampling) is unchanged and lives under `nanovllm/engine/`
and `nanovllm/models/`.
