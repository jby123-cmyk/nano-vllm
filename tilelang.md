# TileLang CPU / LLVM / RVV backend (Path B)

This document records **Path B** work: extending the TileLang fork at
`/mnt/ssd/jby123/tilelang` so GPU-style tile kernels (FlashAttention in
nano-vllm) lower through the existing **CPU/`llvm` pass pipeline** and emit
RISC-V Vector (RVV) assembly via LLVM auto-vectorization, with explicit
**GemmVector** loop nests and **VLEN-aware `T.copy` vectorization** for AraXL
(VLEN=4096, 4 lanes).

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
    │  T.gemm → GemmVector (llvm) / GemmScalar (c)
    │  T.reduce_* → CPU reduce lowering
    │  T.fill → fragment-aware fill
    ▼
VectorizePlanner (wide VLEN for T.copy on llvm+RVV)
    ▼
VectorizeLoop → LLVM RVV backend → .ll / .s
```

**Design principle:** one logical thread per fragment tile (`replicate=1`). RVV
performance comes from **two vectorization stages**:

1. **TileLang `VectorizePlanner`** (`loop_vectorize.cc`) — runs inside the CPU
   pipeline *before* LLVM. It widens `T.copy` loop vectorization using the target
   VLEN (`vector-width` / `+zvl*b`), not a hardcoded 128-bit cap.
2. **LLVM `VectorizeLoop` + RVV backend** — vectorizes serial loops (elementwise)
   and the **`GemmVector` `T.parallel(N)`** nest into scalable
   `<vscale x 4 x float>` / `vfmacc` on unit-stride axes.

No custom LMUL scheduler or AraXL LLVM subtarget yet — LLVM still picks VL/LMUL.

---

## 3. Files changed (TileLang repo)

### 3.1 Python — fragment layouts

| File | Change |
|------|--------|
| `tilelang/cpu/layout_utils.py` | **NEW** — `make_cpu_fragment_layout`, `make_lane_major_fragment`, `infer_fragment_operand_layouts` |
| `tilelang/cpu/op/gemm/gemm_scalar.py` | `infer_layout()` returns fragment layouts via `infer_fragment_operand_layouts` |
| `tilelang/cpu/op/gemm/gemm_vector.py` | Lane-major `infer_layout`; **`lower()` emits `i,k` serial + `j` parallel** nest (not scalar `i,j,k`) for **all** transpose forms (transposed operand → strided vector load, no scalar fallback) |
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

### 3.4 C++ — wide RVV vectorization (Path B follow-up)

| File | Change |
|------|--------|
| `src/cpu/target_utils.{h,cc}` | **NEW** — `TargetIsLLVMRVV(target)` (`llvm` + `+v`); `TargetLLVMVectorWidthBits(target)` via TVM `llvm_get_vector_width` |
| `src/transform/loop_vectorize.cc` | `VectorizePlanner::Plan()`: when `TargetIsLLVMRVV`, set `vector_load_bits_max_` from VLEN bits (e.g. 4096 for AraXL 4-lane) instead of 128 |

### 3.5 nano-vllm harness

| File | Change |
|------|--------|
| `nanovllm/backends/tilelang/rvv_lower.py` | `rvv_target(nr_lanes=4)` sets `vector-width: 1024*nr_lanes` and `+zvl*b`; default `+zvl4096b` |
| `demos/run_rvv_lower.py` | `--nr-lanes` flag; matmul ladder title documents GemmVector + wide copy |

---

## 4. Semantic mapping (GPU → CPU/`llvm`)

| GPU primitive | CPU/`llvm` behavior after Path B |
|---------------|----------------------------------|
| `T.alloc_shared` | Stack **`local`** buffer (no block shared memory) |
| `T.alloc_fragment` | **`local`** + replicated or lane-major **fragment layout** |
| `T.gemm(..., policy=FullRow)` | **`GemmVector`** on `llvm` (`i,k` + `T.parallel(N)`) for all transpose forms incl. `Q @ K^T`; `GemmScalar` only on target `c`; LLVM emits `vfmacc` on `j` |
| `T.copy` (tile loads/stores) | `VectorizePlanner` widens copy loops to VLEN/element-size (e.g. 64 fp32/step at VLEN=4096 for 64-wide tiles) |
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
`Target(rvv, target_host=rvv)`. Wide copy vectorization requires
`with rvv:` so `Target::Current()` is set during the CPU pipeline (otherwise
`VectorizePlanner` falls back to 128-bit and `T.copy` stays 4-wide).

**Matmul evidence (128³, `block=64`, VLEN=4096):** lowered TIR uses 64-wide copy
slices and `for i,k` + `parallel j` GEMM; `.ll` has `llvm.fmuladd.nxv4f32`;
`.s` hot loop uses `vfmacc.vv` / `vl2re32` (numeric pass).

---

## 6. How wide vectorization works (TileLang → TVM → LLVM)

Understanding this pipeline explains why `+zvl4096b` alone did **not** fix
4-wide copies before the follow-up patches.

### Stage A — Tile op lowering (`LowerTileOp`)

- `src/cpu/op/gemm.cc` selects **`cpu.vector`** (`GemmVector`) when
  `target.kind.name == "llvm"`.
- `GemmVector.lower()` (Python) replaces the scalar `i → j → k` nest with:

  ```python
  for i in T.serial(M):
    for k in T.serial(K):
      a_val = A[i, k]
      for j in T.parallel(N):
        C[i, j] += a_val * B[k, j]
  ```

  The contraction axis `k` is outside the vectorized `j` loop so `C[i,:]` is
  **unit-stride** and LLVM emits `vfmacc` on the accumulator. The same nest
  handles the transposed forms — only the *indexing* into A/B changes, e.g. for
  `transpose_B` (FlashAttention's `Q @ K^T`) the operand load is `B[j,k]`, which
  LLVM vectorizes as a strided/gather load (`vluxei`/`vlse`) while keeping the
  `j` accumulation vectorized. (Strided loads are fine here — AraXL's
  unit-stride-only constraint is out of scope for this lowering demo.)

- `T.copy` lowers to nested loops over tile rows/cols; those loops are input to
  Stage B.

### Stage B — TileLang `VectorizePlanner` (`loop_vectorize.cc`)

Runs in the CPU pass pipeline **before** LLVM sees the IR.

1. Reads `Target::Current()` inside `with rvv:`.
2. If `TargetIsLLVMRVV` (`llvm` CPU target with `+v`):
   - Calls `TargetLLVMVectorWidthBits` → TVM `llvm_get_vector_width`, which
     honors `vector-width` and `+zvl*b` on the target dict.
   - Sets `vector_load_bits_max_ = initial_vector_size_ = vlen_bits` (4096 for
     default AraXL 4-lane config) instead of **128**.
3. When vectorizing copy/store loops, divides `vector_load_bits_max_` by
   `dtype.bits()` to get elements per vector step (up to 128 fp32 at VLEN=4096).
   Actual slice width is also bounded by tile extents (e.g. 64 for `block_N=64`).

**Symptom fixed:** `T.copy` TIR went from `*4:*4+4` slices to `*64:*64+64`;
asm went from fixed `vsetivli ..., 4` to `vsetvli` with runtime VL.

### Stage C — LLVM codegen

1. **`VectorizeLoop`** (TVM/LLVM pass) vectorizes:
   - Simple unit-stride loops (elementwise add).
   - The `T.parallel(N)` body from `GemmVector` into scalable vectors
     (`<vscale x 4 x float>` at VLEN=4096 → up to 256 fp32 per op).
2. **RVV backend** lowers to `vsetvli`, `vle32`, `vfmacc.vv`, `vfred*`, etc.

### Target knobs (nano-vllm)

`rvv_target(nr_lanes=4)` in `rvv_lower.py` must set **both**:

- `mattr`: `+zvl4096b` (LLVM feature / VLEN hint)
- `vector-width`: `4096` (bits — used by `llvm_get_vector_width`)

Missing either leaves Stage B at 128-bit or LLVM without scalable VLEN info.

---

## 7. What is still *not* fixed

These remain **observed limitations** (see also `STAGE1_REPORT.md` §6):

1. **Not a hand-tuned micro-kernel** — `GemmVector` + LLVM auto-vec, not an
   explicit LMUL/VRF-scheduled GEMM like AraXL `apps/gemm`.
2. **Tile size vs VLEN** — `block_N=64` uses fewer lanes per op than VLEN=4096
   allows (up to ~256 fp32); increase `block_N` to fill wider vectors.
3. **Conservative LMUL** — LLVM picks VL/LMUL per loop; no TileLang LMUL planner.
4. **No layout optimizer** — lane axis for softmax reductions not chosen for RVV
   (see `memory.md` Option A — still design, not wired).
5. **No AraXL LLVM subtarget** — generic RVV scheduling only.
6. **No hardware `exp2`** — softmax still uses libm/scalar paths.
7. **fp32 baseline** — fp16 needs `zvfh`; GPU tensor-core paths unchanged.
8. **No VRF-aware tiling** — large tiles may spill to stack in `.s`.

Path B + wide vectorization **did** fix compiler plumbing and the worst
auto-vec gaps (4-wide copies, scalar GEMM inner loops — both the non-transposed
`P @ V` and the transposed `Q @ K^T` now vectorize via `GemmVector`).
FlashAttention kernels emit RVV asm with `vfmacc` on both GEMMs; the transposed
`Q @ K^T` operand load lowers to an indexed/gather vector load (`vluxei64` in
`strided_ops`), which is expected once AraXL's unit-stride constraint is out of
scope (`memory.md` covers the unit-stride layout choice as future work). The
softmax `reduce_max`/`reduce_sum` are still serial horizontal reductions
(`vfslide1down` shuffles), not lane-parallel — see limitation #4. Host golden
checks validate math via native `llvm` execution; Spike/Verilator on `.s` is
future work.

---

## 8. Follow-ups

| Item | Notes |
|------|-------|
| Execute RVV `.s` on Spike/Verilator | Bit-exact RISC-V hardware/sim validation |
| Larger `block_N` / `block_M` aligned to VLEN | Fill `<vscale x 4>` vectors on attention/matmul tiles |
| Wire or remove `promote_shared_to_local.py` | C++ remap in `lower_tile_op.cc` is sufficient today |
| Register `reduce.cc` in CMake reliably | Manual `build.ninja` patch if CUDA cmake fails |
| TileLang unit tests | e.g. `testing/python/llvm/test_tilelang_llvm_fragment_gemm.py` |
| AraXL Option-A softmax reductions + software `exp2` | See `memory.md` §4.4–4.5 |
| VRF-aware tile sizing | See `memory.md` §5 |

---

## 9. Related docs (nano-vllm)

| Doc | Contents |
|-----|----------|
| `attention.md` | FlashAttention algorithm, CUDA workflow, **§10 CPU/RVV run instructions** |
| `demos/README.md` | Demo scripts including `run_rvv_lower.py` |
| `demos/build_rvv/STAGE1_REPORT.md` | Auto-generated per-kernel results and asm evidence |
| `nanovllm/backends/tilelang/rvv_lower.py` | RVV target helper + `lower_kernel_rvv()` harness |
| `demos/run_attention_rvv_stage.py` | Standalone CPU golden check (mirrors the CUDA stage) |
| `tests/test_attention_rvv_numeric.py` | Parametrized CPU/RVV numeric matrix (pytest) |
