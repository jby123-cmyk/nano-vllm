# RVV pipeline — background

How the nano-vllm RVV lowering path works, how it differs from standard
TileLang (CUDA), and where vectorization actually happens.

See `usage.md` for commands and `implementation_plan.md` for next steps.

---

## 1. What this pipeline is

The **RVV pipeline** takes the same TileLang `@T.prim_func` kernels used for CUDA
FlashAttention and lowers them through a **CPU / LLVM** pass pipeline configured
for **RISC-V Vector (RVV)**, emitting:

```
@T.prim_func  →  CPU TensorIR  →  LLVM IR  →  RVV assembly (.s)
```

It is a **cross-compile and validation harness**, not a production inference
runtime. Host execution uses the same tile-op lowering on x86 `llvm` for a fast
golden check; Spike executes the RVV `.ll` with the PyTorch golden embedded
(`demos/rvv/spike_matrix.py`). Verilator remains future work.

The algorithm in `nanovllm/backends/tilelang/attention.py` (dense decode /
prefill) and `nanovllm/backends/tilelang/paged_decode.py` (paged decode matching
`flash_attn_with_kvcache`) and `nanovllm/backends/tilelang/paged_prefill.py`
(paged prefill matching `flash_attn_varlen_func` with `block_table`) and
`nanovllm/backends/tilelang/linear.py` (batched `F.linear` / GEMM with PyTorch
`weight[out, in]` layout via `transpose_B`) is **unchanged** between CUDA and
RVV — only the backend realization differs. The linear kernel shares the same
`GemmVector` → `vfmacc` path as the attention score/context GEMMs. Additional
per-op kernels (`rmsnorm`, `activation` / SiluAndMul, `rope`, `kv_store`,
`embedding`) follow the same pattern; a full decoder layer is **composed** from
these kernels on the host (`DecoderLayerStage`), not lowered as one prim_func. The paged kernel reads the engine's
`[num_blocks, block_size, num_kv_heads, head_dim]` pool via `block_table`
indirection; gather/strided loads (`vluxei` / `vlse`) are expected and keep the
GEMM / softmax loops vectorized.

**Current direction.** The eventual target is a **strided-load-capable** vector
ISA studied through **instruction-set-level simulation**. Optimization goal:
**maximize vectorization / efficiency**, not satisfy a unit-stride memory
constraint. Strided/indexed loads (`vlse`, `vluxei`) are therefore allowed and
preferred when they keep a loop vectorized (see §5, §9). RVV target parameters
still come from the AraXL config for convenience.

---

## 2. Standard TileLang (CUDA) vs Path B (CPU/RVV)

| Aspect | CUDA (standard) | CPU/RVV (Path B) |
|--------|-----------------|------------------|
| Target | `cuda` | `llvm` + `+v` + `+zvl*b` |
| `T.alloc_shared` | GPU shared SRAM | Promoted to stack `local` |
| `T.alloc_fragment` | Tensor-core registers | Stack `local` + fragment layout metadata |
| `T.gemm` | WMMA / tensor cores | `GemmVector`: serial `i,k` + parallel `j` |
| `T.reduce_*` | Warp shuffle / CUB | `vfredmax` / `vfredusum` via LLVM intrinsics |
| `T.Pipelined` | Software pipeline (stages) | **Skipped** on `llvm` target |
| `threads=N` | Real CUDA block | Degenerate 1 logical thread; RVV lanes from LLVM |
| Vectorization | Hand-tuned CUDA + tensor cores | **Two-stage** TileLang planner + LLVM auto-vec |
| `T.exp2` | libdevice / fast math | inline Cephes `2**x` polynomial → vector `vfcvt`/`vfmadd`/`vsll.vi` (Phase 2; no libm call) |

Path B added CPU realizations inside the TileLang fork
(`/mnt/ssd/jby123/tilelang`): fragment layouts, `GemmVector`, CPU reduce
lowering, wide `VectorizePlanner`, `tl.infinity` for LLVM, early shared→local
promotion in `LowerTileOp`.

---

## 3. End-to-end flow

```
GPU kernel source (attention.py — unchanged)
    │
    ▼
CPUPassPipeline (tilelang/cpu/pipeline.py)
    │  skip PipelinePlanning / InjectSoftwarePipeline when target=llvm
    ▼
LayoutInference
    │  CPU fragment layouts (lane-major for GemmVector)
    ▼
LowerTileOp
    │  shared / fragment → local (stack)
    │  T.gemm → GemmVector (llvm) or GemmScalar (c)
    │  T.reduce_* → ReduceLowerer (reduce.h) → LLVM RVV intrinsics
    │  T.fill → fragment-aware fill
    ▼
VectorizePlanner (loop_vectorize.cc)     ← Stage B
    │  VLEN-aware copy widening when TargetIsLLVMRVV
    ▼
VectorizeLoop + LLVM RVV backend          ← Stage C
    ▼
.tir / .ll / .s
```

**Design principle:** one logical thread per fragment tile (`replicate=1`).
Performance comes from **vectorizing the loop nests LLVM sees**, not from
emulating GPU warps.

---

## 4. Two stages of vectorization

### Stage B — TileLang `VectorizePlanner`

Runs in the CPU pipeline **before** LLVM.

- Reads `TargetLLVMVectorWidthBits(target)` → TVM `llvm_get_vector_width`
  (from `+zvl*b` on RISC-V).
- Sets `vector_load_bits_max_` to full VLEN bits (e.g. 4096), not 128.
- Widens `T.copy` loop vectorization: `elements_per_step = VLEN_bits / dtype.bits`,
  capped by tile extent.

**Effect:** copy loops go from 4-wide slices to 64-wide (or more) at VLEN=4096.

### Stage C — LLVM `VectorizeLoop` + RVV backend

- Vectorizes simple unit-stride loops (elementwise).
- Vectorizes `T.parallel(N)` body from `GemmVector` into scalable vectors
  (`<vscale x k x float>`).
- Lowers to `vsetvli`, `vle32`, `vfmacc.vv`, `vfred*`, etc.

LLVM picks LMUL/VL per loop conservatively; there is no custom AraXL subtarget or
hand-tuned LMUL scheduler yet.

---

## 5. GemmVector — how GEMM vectorizes

`GemmVector` (`tilelang/cpu/op/gemm/gemm_vector.py`) reorders the contraction:

```python
for i in T.serial(M):
    for k in T.serial(K):
        a_val = A[i, k]   # or transposed indexing
        for j in T.parallel(N):
            C[i, j] += a_val * B[...]
```

The **output-N (`j`) axis is innermost and parallel**, so `C[i,:]` is unit-stride
and LLVM emits `vfmacc` on the accumulator row.

For `Q @ K^T` (`transpose_B=True`), the B load is strided/gather along `j`;
LLVM emits `vluxei64` while keeping `vfmacc` on the accumulator. This **keeps the
score matmul fully vectorized**. The target is a strided-load-capable vector ISA
(evaluated via instruction-set-level simulation), so this gather is desirable —
it avoids scalarizing the transpose. Optimizing for **vectorization/efficiency**
takes priority over any unit-stride legality constraint (see the direction note
in §9).

---

## 6. CPU reduce — RVV `vfred` path

`T.reduce_max` / `T.reduce_sum` on the inner axis use `ReduceLowerer` in
`tilelang/src/backend/common/op/reduce.h`.

### Unified `can_pack` modes (RVV)

| Mode | When | Behavior |
|------|------|----------|
| **A** | RVV + single chunk + non-fragment src | `vle` from source + `vfred` |
| **B** | Fragment src (`acc_s`) or multi-chunk | Vertical accumulate → `vle`+`vfred` from pack buffer |
| **C** | Non-RVV | Tree fold fallback |

Attention uses **Mode B** because `acc_s` is `local.fragment` (logical indices
are not direct memory addresses before layout lowering).

### Key properties

- Pack width `vsize` = largest power-of-2 dividing reduce extent, capped by
  `VLEN_bits / dtype.bits`.
- Scalable types use `vscale()`; `vfred` VL matches tile width (e.g. 128 for
  `block_N=128`), not necessarily full hardware VLEN.
- Init vectors use `llvm.riscv.vfmv.v.f`, hoisted outside row loops via `Bind`.
- No separate `_fold` / `_pack` spill buffers in the final attention IR when
  single-chunk fragment path fuses to direct `vle`+`vfred` from `acc_s`.

---

## 7. Softmax `exp2` — software vector polynomial (Phase 2)

Standard TileLang emits `T.exp2` for the softmax exponent. On GPU this is a fast
hardware/libdevice instruction, but on the CPU/RVV path it lowers to a scalar
`call exp2f` (libm), which has no RVV vector variant — a hard barrier that leaves
the whole softmax loop scalar.

**Fix.** On the vector path (`vec_exp2`, set for `backend == "cpu"`) the kernel
replaces `T.exp2` with an inline Cephes single-precision `2**x` polynomial
(`_exp2_poly` in `attention.py`):

- Range-reduce `2**x = 2**n · 2**r`, `n = round(x)`, `r ∈ [-0.5, 0.5]`.
- `2**r` from a degree-6 Horner polynomial (~1 ulp).
- `2**n` built directly in the fp32 exponent field:
  `reinterpret((int(n) + 127) << 23)` — no libm, all arithmetic + a shift.
- Inputs are clamped to `≥ -127` so masked / `-inf` scores flush `2**x → 0`,
  matching `exp2`.

The polynomial removes the libm barrier but does **not** by itself vectorize the
loop: TileLang flattens the `T.Parallel(block_H, block_N)` exponent into a 1-D
loop indexed as `scores_max[i // block_N]`, and that integer-division index
blocks LLVM's loop vectorizer. That flattening problem is generic to every 2-D
elementwise fragment loop, so its fix lives in the compiler (`SerializeOuterParallel`,
§8) rather than the kernel. The kernel keeps the plain 2-D form:

```python
for i, j in T.Parallel(block_H, block_N):
    acc_s[i, j] = _exp2(acc_s[i, j] * scale - scores_max[i] * scale)
```

`vec_exp2` now only selects poly (CPU) vs hardware `exp2` (CUDA) — a genuine
algorithm choice — and no longer controls loop shape.

Result: 0 `exp2f` calls in the decode/prefill `.s`; with the §8 pass the exponent
runs as vector `vfcvt` (float↔int) / `vfmadd` (Horner) / `vsll.vi` (`2**n`).
CUDA is untouched (hardware `exp2`).

---

## 8. Generic elementwise vectorization — `SerializeOuterParallel` (Phase 4)

`T.exp2` was only one symptom. A multi-axis `T.Parallel(M, N)` is an elementwise
SIMT loop: on GPU every point is a thread, but on **CPU there are no threads**,
so TileLang fuses the nest into one flat loop and per-row / per-column broadcast
operands become non-affine indices — `row[i // N]`, `mask[i % N]`. LLVM's loop
vectorizer rejects `//` / `%` indices, so the body falls back to scalar
`flw`/`fsw`. This scalarized **every** 2-D elementwise fragment loop (softmax
exponent, mask apply, `acc_o` rescale, final divide), not just attention.

**Fix — a CPU-pipeline pass**, `tilelang/transform/serialize_outer_parallel.py`,
run right before `LayoutInference` in `tilelang/cpu/pipeline.py` (GPU pipelines
untouched). Two correctness-preserving rewrites per parallel nest:

1. **Serialize outer axes.** Any parallel loop that contains another parallel
   loop becomes serial; only the innermost axis stays parallel (the vector lane).
   `kParallel → kSerial` cannot change results — on CPU the outer axes never had
   hardware threads.
2. **Hoist loop-invariant loads.** The inner body may still read a fragment
   indexed only by the (now serial) outer axes, e.g. `scores_max[i]`. Such
   invariant value-loads are pulled into a scalar `Bind` in the enclosing serial
   scope and broadcast into the lane, so nothing memory-dependent blocks the
   vectorizer.

```
for i, j in T.Parallel(M, N):          for i in T.serial(M):
    acc[i,j] = f(acc[i,j], row[i])  =>     t = row[i]              # hoisted
                                           for j in T.Parallel(N): # vector lane
                                               acc[i,j] = f(acc[i,j], t)
```

Nests carrying `parallel_loop_layout` (author-chosen layout) or `reducer_info`
(cross-lane reduction) annotations are skipped, so `T.copy`/`T.reduce`/`T.gemm`
lowering is untouched — the pass only rewrites raw elementwise `T.Parallel`.
Kernels therefore stay in the GPU-optimal 2-D form; the compiler adapts them for
CPU. This is why the Phase 2 hand-written serial-row/parallel-column `exp2`
restructure could be reverted (§7).

**Result:** softmax exponent, `acc_o` rescale, and the final divide vectorize
(`vfcvt`/`vfmadd`/`vfmul`, invariant loads hoisted); prefill's fp32 causal mask
vectorizes too. Only the **decode** mask stays scalar — it mixes a `uint8`
compare with an `fp32` select and LLVM does not auto-vectorize that mixed-width
masked-select (an LLVM codegen limit, not the flattening this pass fixes; the
same mask *pattern* vectorizes for same-width fp32 operands). 42/42 numeric
shapes pass.

---

## 9. Target direction — ISA-level, strided-load capable, vectorization-first

**Direction (current).** The work targets a **strided-load-capable** vector ISA,
evaluated via **instruction-set-level simulation** (functional, not cycle- or
hardware-accurate). The earlier AraXL **unit-stride-only** constraint no longer
drives design decisions:

- **Strided / indexed loads (`vlse`, `vluxei`) are allowed and preferred** when
  they keep a loop vectorized. E.g. the transposed `Q @ K^T` gather (`vluxei64`)
  is kept because it preserves a fully-vectorized `vfmacc` matmul; the
  alternative (an on-chip scalar transpose) is *slower*, not faster.
- **Optimize for vectorization/efficiency, not memory-legality.** The metric is
  how much of the kernel runs in wide vector instructions, not whether every
  memory op is unit-stride.
- A specific microarchitecture (VRF size, cache tiers) is **not yet fixed**;
  capacity-aware tiling is deferred until the architecture is chosen.

The RVV target parameters (VLEN = `1024 × nr_lanes`, `+zvl*b`) still originate
from the AraXL TVM flow and remain a convenient long-vector configuration.

**Placement (unchanged).** Path B maps `T.alloc_shared` and `T.alloc_fragment`
to **stack `local` arrays**. After GEMM, `acc_s` lives on the stack; reduce and
`exp2` reload from memory each phase. The **algorithm** is FlashAttention; the
**placement** is host-stack. Optimizing placement waits on the target arch.

---

## 10. VLEN scalability

Vectorization logic is **parameterized by target VLEN**, not hardcoded to 4096:

- `TargetLLVMVectorWidthBits()` → `llvm_get_vector_width()` from `+zvl*b`
- `VectorizePlanner`, `GetPreferedVectorizedSize`, `MakeRVVPackSpec` all use it
- `rvv_target(nr_lanes=N)` sets `vector-width = 1024*N`

Caveats:

- **Compile-time VLEN** — one binary per `+zvl*b`; not runtime-adaptive.
- **Tile sizes bound VL** — the RVV ladder now derives decode `block_N` from the
  target VLEN (`vlen_f32_elements(nr_lanes)`: 128 @ 4 lanes, 256 @ 8 lanes), so
  reduce/GEMM fill the hardware vector (VL 128→256 as lanes grow). Prefill tiles
  are still fixed (deferred).
- **`+zvl` / `nr_lanes` stay in sync** — `nr_lanes` is now the single source of
  VLEN: `rvv_target()` strips any `+zvl*` from `--mattr` and injects
  `+zvl{1024*nr_lanes}b`, and the harness asserts `llvm_get_vector_width() ==
  1024 * nr_lanes` at startup (see `usage.md`).

---

## 11. Current limitations (observed in artifacts)

1. **GPU-shaped tiles (prefill)** — decode `block_N` is now VLEN-derived and
   `block_H` is the GQA group size; prefill still uses CUDA-shaped `block_M/N=64`.
2. **Stack-backed fragments** — fragments map to stack; capacity-aware placement
   waits on a chosen target microarchitecture.
3. **Decode mask stays scalar** — the softmax exponent, `acc_o` rescale, final
   divide, and prefill's fp32 causal mask are now vectorized generically by
   `SerializeOuterParallel` (§8). The remaining scalar loop is the **decode**
   mask (`uint8` compare + `fp32` select); LLVM does not auto-vectorize that
   mixed-element-width masked-select. Widening the mask to fp32 would fix it but
   is an attention-specific edge case, so it is left as-is.
4. **Conservative LMUL** — dynamic `vsetvli` per loop; LLVM backend overhead (`vl1r` spills).
5. **fp32 baseline** — fp16 needs `zvfh`; GPU tensor-core paths unchanged.
6. **Spike ISA-sim** — the RVV matrix can execute on Spike via
   `demos/rvv/spike_matrix.py` / `nanovllm/backends/spike/` (PyTorch golden
   embedded in the ELF). Verilator / RTL sim remains future work.

(`vluxei64` on `Q @ K^T` is **no longer listed as a limitation** — it is the
desired fully-vectorized path on a strided-capable target; see §5/§9. The
`acc_s_cast` copy is also gone on the fp32 path — Phase 3 drops it whenever
`in_dtype == accum_dtype`.)

---

## 12. File map

| Location | Role |
|----------|------|
| `nanovllm/backends/tilelang/attention.py` | FlashAttention kernel source (shared CUDA/RVV) |
| `nanovllm/backends/tilelang/paged_decode.py` | Paged decode (`flash_attn_with_kvcache` contract) |
| `nanovllm/backends/tilelang/linear.py` | Batched `F.linear` / GEMM (`weight[out, in]`) |
| `nanovllm/backends/tilelang/rvv_lower.py` | RVV target + `lower_kernel_rvv()` harness |
| `demos/rvv/run_rvv_lower.py` | Stage 1 ladder + `STAGE1_REPORT.md` |
| `demos/rvv/matrix_report.py` | Host-proxy numeric + RVV compile matrix |
| `demos/rvv/spike_matrix.py` | Full matrix → Spike ISA-sim execution |
| `nanovllm/backends/spike/` | Spike build/run harness (embed PyTorch golden) |
| `/mnt/ssd/jby123/tilelang/tilelang/transform/serialize_outer_parallel.py` | CPU elementwise-loop vectorization pass (§8) |
| `/mnt/ssd/jby123/tilelang/tilelang/cpu/pipeline.py` | CPU pass pipeline (registers the pass) |
| `/mnt/ssd/jby123/tilelang/src/transform/loop_vectorize.cc` | VLEN-aware copy planner |
| `/mnt/ssd/jby123/tilelang/tilelang/cpu/op/gemm/gemm_vector.py` | GemmVector nest |
| `/mnt/ssd/jby123/tilelang/src/backend/common/op/reduce.h` | RVV `vfred` reduce lowering |
| `/mnt/ssd/jby123/tilelang/src/cpu/target_utils.cc` | `TargetIsLLVMRVV`, VLEN helpers |
