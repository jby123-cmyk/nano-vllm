# From GPU to RVV: how TileLang FlashAttention is compiled for a vector CPU

This document explains, in depth, how the **same** TileLang kernels that target
NVIDIA GPUs in this repository are re-compiled into **RISC-V Vector (RVV)**
assembly for a standard long-vector CPU (the AraXL configuration: VLEN = 4096
bits, 4 lanes). It is written to be read start-to-finish: every piece of jargon
is defined the first time it appears, and each GPU construct is shown next to
its RVV analogue with real code excerpts from the dumped artifacts and the
TileLang fork.

It complements the other docs:

- `tilelang.md` — terse changelog of the "Path B" backend work.
- `attention.md` — the FlashAttention math and the CUDA/CPU run workflow.
- `memory.md` — the AraXL memory-hierarchy design (future work).

Everything here is **fp32** and validated on the host `llvm` backend; the
RISC-V `.s` is emitted but executed on Spike/Verilator only as future work.

---

## Table of contents

1. [The one-sentence mental model](#1-the-one-sentence-mental-model)
2. [RVV architecture primer (all the jargon)](#2-rvv-architecture-primer-all-the-jargon)
3. [SIMT vs SIMD: how a GPU's threads become a CPU's lanes](#3-simt-vs-simd-how-a-gpus-threads-become-a-cpus-lanes)
4. [How we tell TileLang to target RVV instead of a GPU](#4-how-we-tell-tilelang-to-target-rvv-instead-of-a-gpu)
5. [The two compiler pipelines, side by side](#5-the-two-compiler-pipelines-side-by-side)
6. [GPU-specific constructs and their RVV analogues](#6-gpu-specific-constructs-and-their-rvv-analogues)
7. [How TileLang actually vectorizes the loops](#7-how-tilelang-actually-vectorizes-the-loops)
8. [Why this maps efficiently onto a standard RVV core](#8-why-this-maps-efficiently-onto-a-standard-rvv-core)
9. [The change made in this session](#9-the-change-made-in-this-session)
10. [Inventory: new / changed directories, pipelines, and files](#10-inventory-new--changed-directories-pipelines-and-files)
11. [What is still scalar (honest limitations)](#11-what-is-still-scalar-honest-limitations)

---

## 1. The one-sentence mental model

> A GPU gets its speed from **dedicated silicon** (tensor cores, shared memory,
> warp schedulers, async copy engines). A vector CPU has none of those, so
> TileLang's job on RVV is to **rewrite the same tile operations into ordinary
> loop nests whose shape lets LLVM's RISC-V auto-vectorizer emit long vector
> instructions** — and to configure the target so LLVM knows how wide those
> vectors are.

The kernel source never changes. What changes is the *target object*, which
selects a *different pass pipeline*, which lowers the GPU tile primitives
(`T.gemm`, `T.copy`, `T.reduce_*`, fragments, shared memory) into CPU-friendly
loops instead of GPU intrinsics.

```
             ┌─────────────────────── same @T.prim_func ───────────────────────┐
             │                                                                  │
     Target("cuda")                                                     Target("llvm", riscv)
             │                                                                  │
   CUDAPassPipelineBody                                              CPUPassPipelineBody
   • warp specialization                                            • (skipped)
   • software pipeline (mbarrier/TMA)                               • (skipped)
   • T.gemm → mma_sync  (tensor core)                               • T.gemm → GemmVector loop
   • shared memory tiles                                            • stack "local" tiles
   • warp-shuffle reductions                                        • serial reductions
             │                                                                  │
   CodeGenCUDA → .cu → nvcc → SASS                                  CodeGenLLVM → .ll → RVV .s
```

---

## 2. RVV architecture primer (all the jargon)

RVV is the **RISC-V Vector extension**. It is "SIMD" — Single Instruction,
Multiple Data — meaning one instruction operates on many numbers at once. Its
defining twist is that the vector length is **not baked into the instructions**;
the same binary runs on chips with different vector widths.

Here is every term you need, in dependency order.

### Vector registers (`v0`–`v31`)
There are 32 vector registers. Each holds *a bag of numbers*, not a single
scalar. A `float` register `fa0` holds one number; a vector register `v8` holds
many.

### VLEN — the physical width, in bits
`VLEN` is how many bits fit in one vector register. This is a property of the
**hardware**:
- a phone-class core might have `VLEN = 128` → 4 × fp32 per register,
- the AraXL config used here has `VLEN = 4096` → **128 × fp32** per register.

In our target this is announced two ways (both are required, see §4):
`+zvl4096b` (an LLVM feature flag) and `"vector-width": 4096`.

### `vsetvli` — "set vector length for this iteration"
Because VLEN is not known at compile time, before a vector loop the code asks
the hardware: *"I have N elements of type e32 (32-bit); how many can you do at
once?"* The instruction `vsetvli` answers with `vl` (the granted length) and
sets `vtype` (element width + grouping). You then process `vl` elements and loop
until the data is consumed. This is why RVV assembly is peppered with `vsetvli`
— it is the "how wide am I right now" handshake and is the essence of
**vector-length-agnostic (VLA)** code.

```asm
vsetvli t0, a2, e32, m1, ta, ma   ; t0 = min(a2, VLEN/32); e32=fp32; m1=LMUL 1
```

### LMUL — "length multiplier"
You can gang **2, 4, or 8** vector registers together to process more elements
per instruction (`m2`, `m4`, `m8`) at the cost of having fewer independent
register groups. `m1` = one register; `mf2` = half a register. Choosing LMUL
well is real performance tuning; today LLVM picks it conservatively (a listed
limitation).

### `vscale` and "scalable vectors"
In LLVM IR, a VLA vector is written `<vscale x 4 x float>`. `vscale` is a
runtime constant = VLEN / 128. So at VLEN = 4096, `vscale = 32`, and
`<vscale x 4 x float>` = 128 fp32. This IR type is how a *single* compiled
program adapts to any VLEN. Seeing `nxv4f32` (short for `<vscale x 4 x float>`)
in a `.ll` file is the sign that LLVM genuinely vectorized a loop.

### The memory-access vocabulary (this decides efficiency)
- **unit-stride** (`vle32.v`, `vse32.v`): load/store *consecutive* elements.
  Cheapest — the hardware streams them.
- **strided** (`vlse32`, `vsse32`): elements a *fixed distance* apart. More
  expensive.
- **indexed / gather-scatter** (`vluxei`, `vsuxei`): elements at *arbitrary*
  offsets given by an index vector. Most expensive.

A well-mapped kernel keeps its hot loop **unit-stride**.

### The compute vocabulary
- **`vfmacc.vv`** — vector fused multiply-accumulate: `acc += a * b`,
  element-wise, one instruction. The workhorse of any matmul.
- **`vfmul.vf` / `vfmsub.vf`** — vector × *scalar* (`.vf` = "vector, float
  scalar"). Used for `x * scale`.
- **reductions** (`vfredusum`, `vfredmax`) — collapse a whole vector into one
  number (a dot-product sum or a row max). Necessary, but **horizontal** — they
  fight the grain of a vector machine and are slow relative to element-wise ops.
- **slides / splats** (`vfslide1down.vf`, `vfmv.f.s`, `vslidedown.vx`) — move
  single lanes in/out of a vector. A swarm of these means the compiler is
  *scalarizing* — pulling elements out one at a time (e.g. to call a scalar
  math function, or to do a manual reduction).

### Masking and tail handling
`ta, ma` in `vsetvli` mean "tail-agnostic, mask-agnostic." Vector loops that
don't divide evenly, or that are predicated (`if` inside a parallel loop), use a
mask register `v0` to disable lanes. This is the RVV analogue of GPU threads
taking divergent branches.

---

## 3. SIMT vs SIMD: how a GPU's threads become a CPU's lanes

GPUs use **SIMT** — Single Instruction, Multiple *Threads*. You write code as
if for one thread; the hardware runs 32 of them in lockstep as a **warp**, and
thousands of them across the chip. Parallelism is expressed as *many threads*.

RVV uses **SIMD** — one thread, but each instruction chews through a *vector* of
elements. Parallelism is expressed as *wide instructions*.

The translation TileLang performs:

| GPU (SIMT) | RVV (SIMD) |
|---|---|
| `threadIdx.x` (up to 1024 threads/block) | collapses to **one logical thread** (`replicate=1`) |
| a **warp** of 32 threads in lockstep | one `vsetvli` loop iteration processing `vl` lanes |
| `blockIdx.x/y/z` grid | plain serial `for` loops over the block indices |
| thread cooperates via shared memory | one thread owns the whole tile in registers/stack |
| warp-shuffle reduction across 32 threads | serial reduction (or a `vfred*`) over lanes |

The key realization: **a warp and a vector are two ways to spend the same
parallelism.** Where the GPU spreads 128 score columns across 128 threads, the
RVV code puts those 128 columns into one 128-wide vector register and updates
them with a single `vfmacc`. The trick to making SIMD "as effective as SIMT" is
to pick which loop axis becomes the vector so that (a) it is long enough to fill
`VLEN` and (b) its memory accesses are unit-stride. That is exactly what the
`GemmVector` loop nest does (§6.1).

TileLang encodes "one logical thread" as a fully-replicated fragment layout:

```python
# tilelang/cpu/layout_utils.py
def cpu_logical_threads() -> int:
    """One owner per fragment tile; physical T.Kernel thread bindings are
    lowered separately and LLVM vectorizes across the RVV lanes."""
    return 1
```

---

## 4. How we tell TileLang to target RVV instead of a GPU

### 4.1 Build an `llvm` target with a RISC-V triple and the `+v` extension

`nanovllm/backends/tilelang/rvv_lower.py`:

```python
def rvv_target(mtriple="riscv64-unknown-elf", mattr=(...,"+v","+zvl4096b"),
               mabi="lp64d", nr_lanes=4):
    vlen_bits = 1024 * nr_lanes             # 4 lanes -> 4096
    return tvm.target.Target({
        "kind": "llvm",                     # <-- routes to the CPU pipeline
        "mtriple": mtriple,                 # riscv64-unknown-elf
        "mattr": mattr,                     # +v (RVV1.0), +m, +f, +d, +zvl4096b
        "mabi": mabi,                       # lp64d
        "vector-width": vlen_bits,          # 4096 — read by LLVM's vectorizer
    })
```

There is **no CUDA target anywhere in the RVV path**, so the tensor-core code
is simply unreachable. Two fields on this object do all the steering:

- `"kind": "llvm"` selects the CPU pass pipeline (next).
- `+v` + `+zvl4096b` + `"vector-width": 4096` tell LLVM this is a vector machine
  and how wide it is.

### 4.2 Lower inside `with target:` and dump each stage

```python
# rvv_lower.py — lower_kernel_rvv()
with target:
    host_mod, device_mod, _params, tgt, tgt_host = lower_to_host_device_ir(
        func, target=target, target_host=target)
    _write_text(tir_path, host_mod.script())           # <name>.tir  (lowered TIR)
    rt_mod = host_codegen(host_mod, target_host=tgt_host, target=tgt)
    _write_text(ll_path, rt_mod.inspect_source("ll"))  # <name>.ll   (LLVM IR)
    _write_text(s_path,  rt_mod.inspect_source("asm")) # <name>.s    (RVV asm)
```

The `with target:` scope matters: the vectorizer reads `Target::Current()` to
learn the VLEN. Without it, copy vectorization silently falls back to 128 bits.

### 4.3 The target kind picks the pipeline

TileLang registers the CPU pipeline for target kinds `c` and `llvm`:

```python
# tilelang/cpu/pipeline.py
for _kind in ("c", "llvm"):
    register_pipeline(PassPipeline(_kind, CPUPassPipelineBody))
```

CUDA registers its own:

```python
# tilelang/cuda/pipeline.py
cuda_pipeline = PassPipeline("cuda", CUDAPassPipelineBody)
register_pipeline(cuda_pipeline)
```

So "target RVV instead of GPU" reduces to: *the target's kind is `llvm`, so
`CPUPassPipelineBody` runs.* Everything below is teaching that pipeline to cope
with GPU-authored kernels.

---

## 5. The two compiler pipelines, side by side

Both pipelines share a spine (bind target → simplify → **LayoutInference** →
**LowerTileOp** → vectorize → split host/device → make API). The differences are
where the GPU-vs-CPU story lives. Here are the two bodies with the divergences
marked.

### GPU: `CUDAPassPipelineBody` (excerpt, `tilelang/cuda/pipeline.py`)

```python
if allow_warp_specialized(target=target):
    mod = ProducerConsumerWarpSpecialized()(mod)   # split producer/consumer warps
mod = LowerBlackwell2SM()(mod)                      # 2-SM tensor-core MMA
mod = IfStmtBinding()(mod)
mod = PipelinePlanning()(mod)                       # plan software pipeline
mod = InjectSoftwarePipeline()(mod)                 # cp.async / mbarrier double-buffer
mod = LayoutInference()(mod)                        # MMA fragment + swizzled shared layouts
mod = LowerTileOp()(mod)                            # T.gemm -> mma_sync
...
mod = LowerSharedBarrier()(mod)                     # hardware mbarrier (sm_90+)
mod = LowerHopperIntrin()(mod); LowerLDGSTG()(mod)  # TMA / ldmatrix intrinsics
mod = VectorizeLoop(...)(mod)
mod = SplitHostDevice()(mod)                        # host launcher + device kernel
```

### RVV: `CPUPassPipelineBody` (excerpt, `tilelang/cpu/pipeline.py`)

```python
mod = IfStmtBinding()(mod)
# Software pipelining merges shared tiles into 3-D buffers the CPU GEMM can't
# index; skip it on llvm — RVV latency hiding is left to LLVM's scheduler.
if target.kind.name != "llvm":
    mod = PipelinePlanning()(mod)
    mod = InjectSoftwarePipeline()(mod)
mod = LayoutInference()(mod)                         # CPU fragment layouts (replicate=1)
mod = LowerTileOp()(mod)                             # T.gemm -> GemmVector loop nest
...
mod = VectorizeLoop(enable_vectorize=...)(mod)       # widen copies to VLEN
mod = SplitHostDevice()(mod)                         # one fused llvm function
```

The CPU body **deletes** every `@CUDA-specific` line: no warp specialization, no
`InjectSoftwarePipeline`, no `LowerSharedBarrier`, no `LowerHopperIntrin` /
`LowerLDGSTG` (TMA / ldmatrix), no persistent-threadblock pass. What remains is
the generic tile → loop → vectorize → codegen path, with `LayoutInference` and
`LowerTileOp` taught to accept CPU realizations of the GPU tile ops (§6).

---

## 6. GPU-specific constructs and their RVV analogues

This is the heart of the port. Each subsection shows the GPU lowering (from the
dumped `demos/build/<ts>/device_kernel.cu`) next to the RVV lowering (from
`demos/build_rvv/<kernel>/*.tir` / `*.s`).

### 6.1 Tensor cores (`mma_sync`) → the `GemmVector` `vfmacc` loop

A **tensor core** is a hardware block that computes a small matrix multiply
`D = A·B + C` (e.g. 16×8×16) in one instruction, consuming operands from
**fragments** (per-thread register shards produced by `ldmatrix`). It also
handles `A·Bᵀ` natively — the transpose is *free*.

**GPU** (`device_kernel.cu`): the `Q @ Kᵀ` score matmul is a tensor-core call.
Note the template flag `..., false, true>` — the `true` is `transpose_B`,
absorbed by the hardware:

```cpp
for (int ki = 0; ki < 4; ++ki) {
  tl::ptx_ldmatrix_x4(&Q_shared[...], &A_local[0]);      // load fragment
  for (int i_3 = 0; i_3 < 8; ++i_3)
    tl::ptx_ldmatrix_x4(&K_shared[...], &B_local[i_3*8]);
  for (int j = 0; j < 8; ++j) {
    tl::mma_sync<half,half,float,16,8,16, false, true>(  // <-- tensor core, Bᵀ
        (float*)(acc_s + j*8), (unsigned*)A_local, (unsigned*)(B_local + j*8));
  }
}
```

There is no visible reduction over the contraction axis — the tensor core
swallows it.

**RVV** (`attention_decode.tir`): there is no tensor core, so `T.gemm` becomes
an ordinary triple loop whose innermost axis is `T.parallel(N)`:

```python
for i, k_1 in T.grid(64, 64):
    a_val: T.float32 = Q_shared_1[i * 64 + k_1]           # one scalar
    for j in T.parallel(128):                             # <-- the vector axis
        acc_s_1[i * 128 + j] = acc_s_1[i * 128 + j] + a_val * K_shared_1[j * 64 + k_1]
```

Why this shape? The contraction axis `k` is the *serial outer* loop, and the
output columns `j` are the *innermost parallel* loop. That makes `acc_s[i, :]`
(a whole row of scores) unit-stride, so LLVM turns the `j` loop into
`<vscale x 4 x float>` `llvm.fmuladd` → **`vfmacc.vv`**. This is the SIMD
equivalent of the tensor core: instead of 32 threads each holding a fragment,
one thread holds the score row in a vector register and accumulates it with one
FMA per contraction step. The transposed operand `K_shared_1[j*64 + k_1]` walks
across `j` with stride 64, so *that one load* becomes a gather (`vluxei64`) —
the price of the transpose that the tensor core got for free.

The `GemmVector` lowering that produces this nest:

```python
# tilelang/cpu/op/gemm/gemm_vector.py
for i in T.serial(M):
  for k in T.serial(K):
    a_val = A_buf[a0 + (k if trans_A else i), a1 + (i if trans_A else k)]
    for j in T.parallel(N):
      C_buf[c0 + i, c1 + j] += T.cast(
          a_val * B_buf[b0 + (j if trans_B else k), b1 + (k if trans_B else j)],
          accum_dtype)
```

Selection is by target kind, in C++:

```cpp
// src/cpu/op/gemm.cc — SelectInst
if (target.defined() && target->kind->name == "llvm")
  return "cpu.vector";   // GemmVector
return "cpu.scalar";     // GemmScalar (plain C target only)
```

### 6.2 Specialized / async-copy engines (TMA, `cp.async`, warp specialization, `mbarrier`) → plain unit-stride copies

Modern NVIDIA GPUs have a **Tensor Memory Accelerator (TMA)** and `cp.async`:
dedicated engines that stream tiles from global to shared memory *asynchronously*
while the math units keep working. **Warp specialization** dedicates some warps
to "produce" (copy) and others to "consume" (compute), and `mbarrier`
(memory barriers) synchronize the handoff. This is how a GPU hides memory
latency.

**GPU** (`device_kernel.cu`): TMA loads gated by mbarriers:

```cpp
mbarrier[8].arrive_and_expect_tx(8192);
tl::tma_load(Q_desc, mbarrier[8], &Q_shared[0], 0, blockIdx.y*4, blockIdx.x);
...
mbarrier[0].wait(0);                     // consumer waits for the copy
```

**RVV**: a scalar CPU has no async copy engine and no warps to specialize, so
the entire pipeline machinery is **skipped** (§5) and `T.copy` becomes a plain
loop of unit-stride vector loads/stores:

```python
# attention_decode.tir — the K tile load
for i_j_fused in range(128):
    K_shared_1[i_j_fused*64 : i_j_fused*64 + 64] = K_2[i_j_fused*128 + cur_kv_head*64 :
                                                       i_j_fused*128 + cur_kv_head*64 + 64]
```

Each 64-element slice is a `vle32.v` / `vse32.v`. Latency hiding is left to
LLVM's instruction scheduler and the CPU's own memory system rather than an
explicit software pipeline. The `T.Pipelined(num_stages=2)` annotation survives
in the IR but is inert:

```python
for k in T.serial(1, annotations={"num_stages": 2}):   # annotation ignored on llvm
```

This is the reason the pipeline explicitly branches on `target.kind.name`:

```python
# tilelang/cpu/pipeline.py
if target.kind.name != "llvm":
    mod = PipelinePlanning()(mod)
    mod = InjectSoftwarePipeline()(mod)
```

(Skipping it is not just an optimization choice: `InjectSoftwarePipeline`
reshapes a shared tile `K_shared` into a 3-D `[stages, block_N, D]` buffer that
the CPU GEMM indexing cannot handle.)

### 6.3 Shared memory → stack `local` buffers

GPU **shared memory** is a fast on-chip scratchpad shared by all threads in a
block (`scope="shared.dyn"`). A CPU has no such tier — the equivalent of
"on-chip, close to the ALUs" is just registers and the stack.

**GPU** (`device.tir`): `buf_dyn_shmem` is allocated in `shared.dyn`:

```python
buf_dyn_shmem = T.alloc_buffer((74240,), "uint8", scope="shared.dyn")
Q_shared: T.handle("float16", "shared.dyn") = ...
```

**RVV**: `LowerTileOp` was patched to promote `shared` (and `fragment`) buffers
to `scope="local"` *before* the copy/gemm ops are lowered, so nothing downstream
still references a shared buffer:

```cpp
// src/transform/lower_tile_op.cc  (promotion runs BEFORE visiting the body)
if (IsFragmentBuffer(buffer) ||
    (TargetIsCPU(target_) && IsSharedBuffer(buffer))) {
  Type new_type = PointerType(ptr_type->element_type, "local");
  ...
  buffer_remap_.Set(buffer, new_buf);
}
```

In the lowered TIR you see `K_shared_1 = T.decl_buffer(..., scope="local")`.

### 6.4 Warp-shuffle reductions → serial (or `vfred`) reductions

The softmax needs a row-max and a row-sum across the score columns. On a GPU
these are **warp reductions**: 32 threads exchange values via register shuffles
(`tl::AllReduce`) in log₂(32) = 5 steps.

**GPU** (`device_kernel.cu`):

```cpp
for (int rv = 0; rv < 32; ++rv)
    scores_max_clear[i_5] = max(scores_max_clear[i_5], acc_s[...]);
scores_max_clear[i_5] = tl::AllReduce<tl::MaxOp, 4, 1, 128, ...>::run(scores_max_clear[i_5]);
```

**RVV**: with one logical thread there is no cross-thread exchange; the reduction
is a serial loop over the lane axis:

```python
# attention_decode.tir
for i in T.unroll(64):
    scores_max_clear[i] = T.float32("-inf")
    for rv in T.unroll(128):
        scores_max_clear[i] = T.max(scores_max_clear[i], acc_s_1[i * 128 + rv])
```

LLVM lowers this into horizontal `vfslide1down`/`vfmv.f` shuffles (or a
`vfredmax`). This is the least vector-friendly part of the kernel (see §11); the
CPU reduce lowering that makes it *legal* lives in the new files:

```cpp
// src/cpu/op/reduce.cc + src/tl_templates/cpu/reduce.h
// A serial AllReduce stub; with replicate=1 it degenerates to a serial loop.
```

### 6.5 Fragments / `ldmatrix` → CPU fragment layouts

A GPU **fragment** is the per-thread shard of a matrix that a tensor core
consumes; `ldmatrix` loads it with a special swizzled layout. On CPU there are
no fragments, but `LayoutInference` still needs *a* layout for every
`T.alloc_fragment` buffer or it fails. The fork supplies a trivial one: a single
owner thread with a dense row-major index, so reductions/GEMMs over the last
axis are unit-stride.

```python
# tilelang/cpu/layout_utils.py — make_lane_major_fragment
def forward_thread_fn(*vars):   # everything owned by logical thread 0
    return 0
def forward_index_fn(*vars):    # dense row-major -> last axis contiguous (RVV lane axis)
    idx = vars[0] * strides[0]
    for d in range(1, ndim):
        idx = idx + vars[d] * strides[d]
    return idx
return Fragment(shape, forward_thread_fn, forward_index_fn, replicate=1)
```

`GemmScalar.infer_layout` and `Copy::InferLayout` were also updated to hand out
these layouts (and to propagate a fragment layout across a dtype-cast copy such
as `acc_s → acc_s_cast`).

### 6.6 Device math intrinsics (`exp2f`) → libm scalar calls

`exp2` (used by the online-softmax exponent) is a fast hardware intrinsic on a
GPU. RVV has **no vector transcendental instruction**, so LLVM scalarizes it.

**GPU** (`device_kernel.cu`):

```cpp
acc_s[i_8] = exp2f((acc_s[i_8] * 0x1.7154...p-3f) - (scores_max[...] * 0x1.7154...p-3f));
```

**RVV** (`attention_decode.s`): the affine part (`x*scale - max*scale`)
vectorizes (`vfmul.vf`, `vfmsub.vf`), then each lane is extracted, sent to libm,
and reinserted:

```asm
vfmul.vf  v8, v8, fa5
vfmsub.vf v9, fa5, v8
vfmv.f.s  fa0, v9        ; extract lane 0
call      exp2f          ; scalar libm call
vfslide1down.vf v8, v8, fa0   ; reinsert; repeat per lane (66 calls in decode.s)
```

### 6.7 `tl.infinity` → a real `-inf` constant

The causal/padding mask fills disallowed scores with `-inf` via
`-T.infinity()`. This intrinsic only had CUDA/HIP lowering; the fork registered
it for `llvm` and `c`:

```cpp
// src/op/math.cc
TVM_REGISTER_OP("tl.infinity")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", infinity_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", infinity_op)
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic", infinity_op)   // added
    .set_attr<FLowerIntrinsic>("c.FLowerIntrinsic", infinity_op);     // added
```

It becomes a `-inf` float immediate (a `lui` of `0xFF800000`) in the `.s`.

### Summary table

| GPU construct | Hardware role | RVV analogue | Where handled |
|---|---|---|---|
| `mma_sync` (tensor core) | matrix-multiply unit, `A·Bᵀ` free | `GemmVector` `parallel-N` `vfmacc` loop (transpose → gather) | `gemm.cc`, `gemm_vector.py` |
| TMA / `cp.async` / warp spec / `mbarrier` | async copy + latency hiding | plain unit-stride `vle`/`vse`; pipeline skipped | `pipeline.py` |
| shared memory (`shared.dyn`) | on-chip scratchpad | stack `local` buffers | `lower_tile_op.cc` |
| warp-shuffle `AllReduce` | cross-thread reduction | serial reduce (→ `vfred`/slides) | `reduce.cc`, `reduce.h` |
| fragments / `ldmatrix` | tensor-core operand shards | replicate-1 lane-major fragment layout | `layout_utils.py`, `gemm_scalar.py`, `copy.cc` |
| `exp2f` device intrinsic | 1-cycle transcendental | per-lane libm `exp2f` call | (LLVM scalarization) |
| `tl.infinity` | mask sentinel | `-inf` float immediate | `math.cc` |
| `T.Pipelined(num_stages)` | software pipeline | inert annotation | `pipeline.py` |
| `threadIdx` / warps | SIMT lanes | one logical thread; LLVM vector lanes | `layout_utils.py` |

---

## 7. How TileLang actually vectorizes the loops

There are **two** vectorizers, and it is important to know which does what.

### Stage 1 — TileLang's `VectorizePlanner` (vectorizes the *copies*)

TileLang's own pass (`src/transform/loop_vectorize.cc`) widens `T.copy` loops.
By default it caps at 128/256 bits; the fork added an RVV branch that reads the
real VLEN from the target:

```cpp
Target target = Target::Current(false);
if (TargetIsLLVMRVV(target)) {                        // llvm + "+v"
  int vlen_bits = TargetLLVMVectorWidthBits(target);  // 4096 (from "vector-width"/+zvl)
  if (vlen_bits <= 0) vlen_bits = 128;
  vector_load_bits_max_ = initial_vector_size_ = loop_extent_vector_size_ = vlen_bits;
}
```

`TargetIsLLVMRVV` / `TargetLLVMVectorWidthBits` are the new helpers in
`src/cpu/target_utils.cc`:

```cpp
bool TargetIsLLVMRVV(Target target) {
  if (!target.defined() || target->GetTargetDeviceType() != kDLCPU) return false;
  if (target->kind->name != "llvm") return false;
  return target_has_feature_fn("v", target).cast<bool>();     // the +v extension
}
int TargetLLVMVectorWidthBits(Target target) {
  return llvm_get_vector_width_fn(target).cast<int>();         // honors "vector-width"/+zvl*b
}
```

Effect: a `T.copy` that used to emit four 4-wide slices now emits VLEN-wide
slices (e.g. `A_local_1[i*64 : i*64+64] = ...` in `matmul.tir`), which the
downstream `VectorizeLoop` pass turns into TIR vector ramps → `vle32.v`/`vse32.v`.

### Stage 2 — LLVM's own loop vectorizer (vectorizes the *math*)

The `GemmVector` `for j in T.parallel(N)` loop and the elementwise loops stay
**scalar loops in the dumped `.tir`**. They are vectorized by **LLVM's**
LoopVectorize + RISC-V backend when `host_codegen` invokes LLVM with `+v` and
`vector-width` set. That is what produces `<vscale x 4 x float>`
`llvm.fmuladd.nxv4f32` in the `.ll` and `vfmacc.vv` in the `.s`.

So the honest answer to "how does TileLang know how to vectorize the loops":

- For **data movement** (`T.copy`), TileLang vectorizes directly, using the VLEN
  you configured.
- For **arithmetic** (`T.gemm`, elementwise), TileLang does **not** vectorize
  directly — it (a) emits a clean unit-stride parallel loop and (b) sets the
  target features so **LLVM's** vectorizer can. TileLang's job is to hand LLVM a
  loop that is *legal and profitable* to vectorize.

---

## 8. Why this maps efficiently onto a standard RVV core

Pulling it together, the strategy that makes SIMD approach SIMT efficiency:

1. **Retarget, don't rewrite.** One kernel source; the `llvm`/RISC-V target
   picks the CPU pipeline and the `cpu.vector` GEMM. GPU concepts collapse:
   shared → stack local, warps/threads → one logical thread, pipeline → skipped.

2. **Choose the vector axis deliberately.** `GemmVector` makes the *output-N*
   axis the innermost parallel loop and keeps the contraction *serial and
   outside*. That guarantees the accumulator `C[i, :]` is unit-stride, so LLVM
   emits `vfmacc` on it — the SIMD stand-in for a tensor core's MMA. This is the
   single most important decision for efficiency.

3. **Feed the real VLEN.** `TargetLLVMVectorWidthBits` + the RVV branch in
   `VectorizePlanner` widen copies to the full 4096-bit vector, and the target's
   `vector-width`/`+zvl4096b` let LLVM emit *scalable* (`vscale`) vectors — code
   that fills whatever VLEN the hardware has, up to 128 fp32 per instruction.

4. **Keep the hot loop unit-stride; localize the unavoidable gather.** The
   transposed `Q @ Kᵀ` needs one strided/gather operand (`vluxei`); everything
   else — the accumulator, the copies, the PV matmul — stays unit-stride.

The result: the two matmuls that dominate attention compile to the same kind of
long-vector `vfmacc` loop as a standalone matmul, which is the RVV analogue of
the tensor-core path. A warp's worth of parallel work now lives in a vector
register instead of across 32 threads.

---

## 9. The change made in this session

Almost all of the above pre-existed as "Path B." The change made in *this*
session was narrow and specific: **`GemmVector` used to bail out to the scalar
`GemmScalar` for any transposed GEMM**, so FlashAttention's `Q @ Kᵀ` (which uses
`transpose_B=True`) was compiled as a scalar `for i, j, k` dot product — LLVM
could not turn it into `vfmacc`, and the prefill `.s` had a single `vfmacc.vv`
dominated by reduction shuffles.

Original (removed) guard:

```python
def lower(self, ...):
    if self.trans_A or self.trans_B:
        return GemmScalar(self.gemm_node).lower(...)   # <-- scalar fallback
    ...
```

Now the transpose flags only change the *indexing* into A/B, never the loop
shape (§6.1), so `Q @ Kᵀ` uses the same `parallel-N` `vfmacc` nest as the
untransposed matmul. The transposed operand becomes a gather load (`vluxei64`),
which is fine because the AraXL unit-stride constraint is explicitly out of
scope for this lowering demo. Verified: both attention GEMMs now emit
`fmuladd.nxv4f32`, and the 42-case CPU numeric matrix (`pytest
tests/test_attention_rvv_numeric.py`) still passes.

---

## 10. Inventory: new / changed directories, pipelines, and files

All backend changes live in the TileLang fork at `/mnt/ssd/jby123/tilelang`
(nano-vllm consumes it via `PYTHONPATH`). The nano-vllm side only *drives* the
backend; it contains no lowering logic.

### Directory that is effectively new: `tilelang/cpu/` (+ `src/cpu/`)

The CPU/LLVM tile-op backend. GPU support lives in `tilelang/cuda/` + `src/cuda/`;
this is its CPU sibling, extended by Path B to handle GPU-authored kernels.

### Pipelines

| Pipeline | File | Status | Why |
|---|---|---|---|
| `CUDAPassPipelineBody` | `tilelang/cuda/pipeline.py` | unchanged (reference) | tensor cores, TMA, warp specialization, software pipeline |
| `CPUPassPipelineBody` | `tilelang/cpu/pipeline.py` | **changed** | skip software-pipeline injection on `llvm`; otherwise the shared→3-D reshape breaks the CPU GEMM |

### Files (TileLang fork)

| File | New/Changed | Why it is necessary |
|---|---|---|
| `src/cpu/target_utils.cc` / `.h` | changed | Add `TargetIsLLVMRVV` and `TargetLLVMVectorWidthBits` so passes can detect an RVV target and read its VLEN |
| `src/transform/loop_vectorize.cc` | changed | Widen `T.copy` vectorization to the full VLEN on RVV instead of 128/256 bits |
| `src/cpu/op/gemm.cc` | changed | `SelectInst`: pick `cpu.vector` (GemmVector) for `llvm`, `cpu.scalar` for `c` |
| `tilelang/cpu/op/gemm/__init__.py` | changed | Register both GEMM impls and their target matchers |
| `tilelang/cpu/op/gemm/gemm_vector.py` | **new** (+ edited this session) | The `serial i,k` + `parallel j` `vfmacc` nest; now handles all transpose forms |
| `tilelang/cpu/op/gemm/gemm_scalar.py` | changed | `infer_layout` returns CPU fragment layouts so fragment-using kernels pass `LayoutInference` |
| `tilelang/cpu/layout_utils.py` | **new** | CPU fragment layouts: replicate-1, lane-major (last axis = RVV lane axis) |
| `src/transform/lower_tile_op.cc` | changed | Promote `shared`/`fragment` → `local` *before* lowering; tolerate buffers without a layout |
| `src/cpu/op/copy.cc` | changed | Propagate a fragment layout across dtype-cast copies (`acc_s → acc_s_cast`) |
| `src/cpu/op/fill.cc` | changed | Route `T.fill`/`T.clear` through the shared filler that supports `local.fragment` |
| `src/cpu/op/reduce.cc` | **new** | CPU lowering for `T.reduce_max`/`T.reduce_sum` |
| `src/tl_templates/cpu/reduce.h` | **new** | Serial `AllReduce` template used by the CPU reduce lowering |
| `src/op/math.cc` | changed | Register `tl.infinity` for `llvm` and `c` targets |
| `tilelang/cpu/promote_shared_to_local.py` | **new (unused)** | Alternative Python promotion pass; the C++ path in `lower_tile_op.cc` is authoritative |

### Files (nano-vllm side — drivers, not backend)

| File | Role |
|---|---|
| `nanovllm/backends/tilelang/rvv_lower.py` | Builds the RVV target, runs the lower→codegen→dump harness, scans emitted RVV ops |
| `nanovllm/backends/tilelang/attention.py` | The FlashAttention `@T.prim_func` kernels (shared with the CUDA path) |
| `demos/run_rvv_lower.py` | Runs the elementwise → matmul → decode → prefill ladder, writes `STAGE1_REPORT.md` |
| `tests/test_attention_rvv_numeric.py` | 42-case CPU numeric matrix vs the PyTorch golden |

---

## 11. What is still scalar (honest limitations)

The GEMMs are vectorized; two things are not, both inherent to attention rather
than to matmul:

1. **Softmax reductions** (`reduce_max` / `reduce_sum`) lower to serial
   horizontal loops → `vfslide1down` shuffles, not a lane-parallel reduction.
   The AraXL "Option A" layout (lanes = KV columns) that would make these
   parallel is design-only in `memory.md` §4.4.

2. **`exp2`** scalarizes to per-lane libm `exp2f` calls (no RVV vector
   transcendental). A polynomial `exp2` via `call_extern` is future work
   (`memory.md` §4.5).

Additional non-blocking notes: LLVM picks LMUL/VL conservatively (no TileLang
LMUL planner); `block_N=64` underfills a 4096-bit register; the transposed
score matmul uses a gather (`vluxei`) that a strict unit-stride machine (AraXL)
could not run; fp16 needs `zvfh`; and the `.s` is validated numerically on the
host but not yet executed on Spike/Verilator.
