# TileLang CPU / LLVM / RVV backend (Path B)

This document records **Path B** work: extending the TileLang fork at
`/mnt/ssd/jby123/tilelang` so GPU-style tile kernels (FlashAttention in
nano-vllm) lower through the existing **CPU/`llvm` pass pipeline** and emit
RISC-V Vector (RVV) assembly via LLVM auto-vectorization.

The **attention algorithm in nano-vllm is unchanged** — the same `@T.prim_func`
kernels in `nanovllm/backends/tilelang/attention.py` drive both CUDA and CPU
paths. Path B added CPU realizations for GPU primitives inside TileLang.

For how to run the RVV artifacts from nano-vllm, see `attention.md` §10 and
`demos/run_rvv_lower.py`.

---

## 1. Problem (before Path B)

Elementwise and simple matmul kernels already compiled to RVV. FlashAttention
decode/prefill failed in a chain of missing CPU support:

| Stage | Failure |
|-------|---------|
| `LayoutInference` | `T.alloc_fragment` accumulators (`acc_s_cast`, …) had **no CPU layout** — `GemmScalar.infer_layout()` returned `{}` |
| `LowerTileOp` | `T.fill` on `local.fragment` unsupported; `T.reduce_*` had no CPU lowering |
| Software pipeline | `T.Pipelined(..., num_stages=2)` turned 2D shared tiles into **3D** buffers; `GemmScalar` indexing broke |
| `MakePackedAPI` | `shared`/`shared.dyn` buffers remapped to `local` **after** copy/gemm lowering left dangling `DeclBuffer` refs |
| LLVM codegen | `tl.infinity` had no `llvm` intrinsic lowering |

---

## 2. Solution overview

```
GPU kernel source (unchanged)
    │
    ▼
CPUPassPipeline (tilelang/cpu/pipeline.py)
    │  skip PipelinePlanning/InjectSoftwarePipeline when target=llvm
    ▼
LayoutInference
    │  fragment layouts from cpu/layout_utils + gemm infer_layout
    ▼
LowerTileOp
    │  early shared/fragment → local remap (lower_tile_op.cc)
    │  T.gemm → GemmScalar / cpu.vector
    │  T.reduce_* → CPU reduce lowering
    │  T.fill → fragment-aware fill
    ▼
VectorizeLoop → LLVM RVV backend → .ll / .s
```

**Design principle:** one logical thread per fragment tile (`replicate=1`); LLVM
`VectorizeLoop` + the RVV backend emit `vle`/`vfmacc`/`vfred*` on innermost
contiguous loops. No custom LMUL scheduler or AraXL subtarget in Path B.

---

## 3. Files changed (TileLang repo)

### 3.1 Python — fragment layouts

| File | Change |
|------|--------|
| `tilelang/cpu/layout_utils.py` | **NEW** — `make_cpu_fragment_layout`, `make_lane_major_fragment`, `infer_fragment_operand_layouts` |
| `tilelang/cpu/op/gemm/gemm_scalar.py` | `infer_layout()` returns fragment layouts via `infer_fragment_operand_layouts` |
| `tilelang/cpu/op/gemm/gemm_vector.py` | **NEW** — lane-major `infer_layout`; `lower()` delegates to `GemmScalar` |
| `tilelang/cpu/op/gemm/__init__.py` | Register `cpu.scalar` (target `c`) and `cpu.vector` (target `llvm`) |

### 3.2 Python — CPU pass pipeline

| File | Change |
|------|--------|
| `tilelang/cpu/pipeline.py` | Skip `PipelinePlanning` + `InjectSoftwarePipeline` when `target.kind.name == "llvm"` |
| `tilelang/cpu/promote_shared_to_local.py` | **NEW** utility pass (substitute-based); **not wired** — C++ early remap is authoritative |

### 3.3 C++ — tile op lowering

| File | Change |
|------|--------|
| `src/cpu/op/gemm.cc` | `SelectInst`: `llvm` → `cpu.vector`, `c` → `cpu.scalar` |
| `src/cpu/op/reduce.cc` | **NEW** — `RegisterReduceImpl` for CPU; uses `ReduceLowerer` |
| `src/tl_templates/cpu/reduce.h` | **NEW** — serial `AllReduce` stub for host templates |
| `src/cpu/op/copy.cc` | `InferLayout` propagates fragment src→dst layout |
| `src/cpu/op/fill.cc` | Uses `backend::Fill::Lower` (supports `local.fragment`) |
| `src/transform/lower_tile_op.cc` | **Critical:** promote `shared`/`fragment` → `local` **before** visiting block body; skip layout `Forward()` when remapped buffer has no layout; tolerate missing layout on BufferLoad/Store |
| `src/op/math.cc` | Register `tl.infinity` for `llvm.FLowerIntrinsic` and `c.FLowerIntrinsic` |

---

## 4. Semantic mapping (GPU → CPU/`llvm`)

| GPU primitive | CPU/`llvm` behavior after Path B |
|---------------|----------------------------------|
| `T.alloc_shared` | Stack **`local`** buffer (no block shared memory) |
| `T.alloc_fragment` | **`local`** + replicated or lane-major **fragment layout** |
| `T.gemm(..., policy=FullRow)` | `GemmScalar` triple loop; warp policy ignored; LLVM vectorizes |
| `T.reduce_max` / `T.reduce_sum` | Serial reduce via `reduce.cc` + `reduce.h` |
| `T.fill(-T.infinity(...))` | Fragment-aware fill lowering |
| `T.Pipelined(..., num_stages=N)` | Still in source TIR; **not injected** on `llvm` |
| `T.Kernel(..., threads=128)` | Degenerate 1-wide logical thread; RVV lanes from LLVM |
| `T.exp2` | libm / scalarized — not a hardware RVV transcendental |

---

## 5. Build and verify

```bash
# From the TileLang repo (CUDA optional for cmake; reduce.cc may need manual ninja entry)
cd /mnt/ssd/jby123/tilelang/build
cmake .. && ninja -j$(nproc)

# Point nano-vllm at the dev build
export PYTHONPATH=/mnt/ssd/jby123/tilelang/build/python:/mnt/ssd/jby123/tilelang:$PYTHONPATH
conda activate nanovllm
cd /mnt/ssd/jby123/nano-vllm
python demos/run_rvv_lower.py
```

Expected ladder after a successful Path B build:

| Kernel | Compiled | Vectorized | Numeric (host) |
|--------|----------|------------|----------------|
| `elementwise` | yes | yes | pass |
| `matmul` | yes | yes | pass |
| `attention_decode` | yes | yes | pass |
| `attention_prefill` | yes | yes | pass |

Artifacts: `demos/build_rvv/<name>/{source.tir,tir,ll,s,metadata.json}` and
`demos/build_rvv/STAGE1_REPORT.md`.

**Compound target note:** `lower_to_host_device_ir` builds
`Target(rvv, target_host=rvv)`. Copy vectorization requires
`with rvv:` so `Target::Current()` is set during the CPU pipeline.

---

## 6. What Path B did *not* fix

These remain **observed limitations** (documented in `STAGE1_REPORT.md` §6):

1. **Scalar-fallback GEMM** — `T.gemm` is a triple loop + LLVM auto-vec, not a tuned RVV micro-kernel.
2. **Conservative LMUL** — LLVM picks VL per loop; no long-vector restructuring.
3. **No layout optimizer** — lane axis and reduction strategy are not chosen for RVV.
4. **No AraXL LLVM subtarget** — generic RVV scheduling only.
5. **Strided loads** — `vlse32` appears in matmul/attention asm; AraXL is unit-stride only.
6. **No hardware `exp2`** — softmax still uses libm/scalar paths.
7. **fp32 baseline** — fp16 needs `zvfh`; GPU tensor-core paths unchanged.
8. **No VRF-aware tiling** — large tiles may spill to stack in `.s`.

Path B **did** fix **compiler plumbing** so the real FlashAttention kernels emit
~7k lines of RVV asm each. Host golden checks (`run_attention_rvv_stage.py`,
`run_rvv_lower.py` numeric column) validate kernel math via native `llvm`
execution; bit-exact RVV execution on RISC-V sim/hardware is future work.

---

## 7. Follow-ups

| Item | Notes |
|------|-------|
| Execute RVV `.s` on Spike/Verilator | Bit-exact RISC-V hardware/sim validation |
| `GemmVector.lower()` with explicit K-innermost parallel nest | May reduce `vlse32` in GEMM |
| Wire or remove `promote_shared_to_local.py` | C++ remap in `lower_tile_op.cc` is sufficient today |
| Register `reduce.cc` in CMake reliably | Manual `build.ninja` patch if CUDA cmake fails |
| TileLang unit tests | e.g. `testing/python/llvm/test_tilelang_llvm_fragment_gemm.py` |
| AraXL-specific layout / unit-stride copies | Eliminate `vlse32` in attention tiles |

---

## 8. Related docs (nano-vllm)

| Doc | Contents |
|-----|----------|
| `attention.md` | FlashAttention algorithm, CUDA workflow, **§10 CPU/RVV run instructions** |
| `demos/README.md` | Demo scripts including `run_rvv_lower.py` |
| `demos/build_rvv/STAGE1_REPORT.md` | Auto-generated per-kernel results and asm evidence |
| `nanovllm/backends/tilelang/rvv_lower.py` | RVV target helper + `lower_kernel_rvv()` harness |
| `demos/run_attention_rvv_stage.py` | Standalone CPU golden check (mirrors the CUDA stage) |
| `tests/test_attention_rvv_numeric.py` | Parametrized CPU/RVV numeric matrix (pytest) |
