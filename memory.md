# Memory Hierarchy: Mapping FlashAttention Tiles onto AraXL

This document describes the memory-system challenges of running the nano-vllm
FlashAttention kernels (see `attention.md`) on the **AraXL** RISC-V vector
processor, and the strategy for mapping FlashAttention tiles onto AraXL's
hierarchy efficiently.

It is the memory-system companion to `attention.md`: that document explains the
TileLang FlashAttention kernels for GPU; this one explains why the GPU memory
model does **not** transfer directly to a long-vector machine, and how we adapt.

> **Implementation status (read first).** This document is the *target* AraXL
> mapping **strategy**. The Path B backend (`tilelang.md`) lowers the GPU kernels
> to RVV via LLVM auto-vectorization. As of the wide-vectorization follow-up:
>
> - **Implemented today:** `GemmVector` (`i,k` + parallel `j`) for `T.gemm` on
>   `llvm`; VLEN-aware `VectorizePlanner` for wide `T.copy` (64 fp32/step at
>   `block_N=64`, VLEN=4096); unit-stride GEMM loads (`vfmacc` in asm; current
>   decode build has `strided_ops: []`).
> - **Still future work:** Option-A reductions (§4.4), `exp2` polynomial
>   (§4.5), explicit lane-major layout for softmax (§4.4), VRF-aware tile sizing
>   (§5), double-buffer prefetch budgeting, and Spike/Verilator execution.

---

## 1. The AraXL memory hierarchy (what we actually target)

AraXL is the scaled-up Ara2 design: multiple lane-based vector clusters
coprocessing for a CVA6 scalar core, interconnected to a shared L2. The
memory-relevant facts, taken from the RTL (`hardware/src/ara_soc.sv`,
`config/*.mk`):

| Tier | What it is | Access path | Speed |
|------|-----------|-------------|-------|
| **VRF** (vector register file) | `32 × VLEN`-bit architectural registers | direct from lanes | fast (register) |
| **"L2"** | a single flat on-chip SRAM bank | VLSU → AXI crossbar → `axi_to_mem` | main-memory latency |
| **Off-chip DRAM** | **does not exist yet** | — | — |

Two facts dominate everything below:

### 1.1 The VRF is huge — and it is the *only* fast on-chip tier

`VLEN` scales with lane count (`VLEN = 1024 bits × nr_lanes`):

| Config | `nr_lanes` | `VLEN` (bits) | VRF total (`32 × VLEN`) |
|--------|-----------|---------------|--------------------------|
| `2_lanes`  | 2  | 2048   | 8 KB  |
| `4_lanes`  | 4  | 4096   | 16 KB |
| `8_lanes`  | 8  | 8192   | 32 KB |
| `16_lanes` (default) | 16 | 16384 | **64 KB** |

64 KB of register file at the default config is comparable to a GPU SM's
*entire shared-memory budget* — except here it is registers feeding the lanes
directly. This is the architectural thesis of a long-vector machine: put the
on-chip working storage into a large VRF.

### 1.2 "L2" is not a cache — it is flat SRAM main memory over AXI

Despite the comment in `ara_soc.sv` calling it an "L2 cache," it is instantiated
as a plain memory (`i_dram`) behind `axi_to_mem`, mapped at `DRAMBase`
(`0x8000_0000`). The vector load/store unit (VLSU) streams to it across the AXI
crossbar. There is **no cache tier between the VRF and L2**, and **no off-chip
DRAM controller** — the on-chip SRAM *is* the top of the memory system today.

**Consequence:** from the compute's point of view, AraXL has a **two-tier**
hierarchy — VRF (fast) and L2/AXI (main-memory speed) — with **no middle
scratchpad tier**. This is fundamentally different from a GPU's
registers → shared-memory SRAM → L2 → global.

---

## 2. Why the GPU memory model does not transfer

The GPU FlashAttention kernels (`nanovllm/backends/tilelang/attention.py`) lean
on three things AraXL does not have:

1. **A fast software-managed scratchpad** (`T.alloc_shared`) — used to stage
   coalesced loads and to share/reuse K/V blocks across the threads of a block.
2. **Tensor cores** (`T.gemm` → `mma`) — dense matrix-multiply units.
3. **Hardware transcendentals** (`T.exp2` → one `MUFU.EX2` instruction).

On AraXL, the natural answer to "where does shared memory go?" is **nowhere
new** — its roles collapse into the VRF (for working/staged tiles) and L2 (for
reuse via streaming). Trying to use **L2 as a scratchpad** (the naive
`shared → L2` mapping) is the trap: L2 sits behind the AXI crossbar and the
VLSU, so frequent small accesses to it pay main-memory latency every time. That
defeats the purpose of a scratchpad.

The correct mental model is the classic Cray-style vector hierarchy: a large
register file plus a high-bandwidth, decoupled path to banked main memory, with
latency hidden by **long vectors**, not by a scratchpad or by thread
oversubscription. This model is well proven for regular, high-arithmetic-
intensity, unit-stride kernels (GEMM, conv, FFT — exactly what Ara/AraXL
benchmark). It is weaker for irregular / gather-heavy / low-intensity workloads.

---

## 3. The challenges (enumerated)

| # | Challenge | Why it matters for FlashAttention |
|---|-----------|-----------------------------------|
| **C1** | **No fast scratchpad tier.** The GPU `shared` memory role has no hardware home. | Q/K/V staging and cross-row reuse must live in the **VRF**, not in an SRAM scratchpad. |
| **C2** | **VRF capacity is the binding constraint.** 64 KB (at 16 lanes), and tiles that don't fit spill to L2 — with no cache to catch them. | Tile sizes (`block_N`, `head_dim`, accumulator precision) must be chosen so the FlashAttention working set fits in the VRF. |
| **C3** | **L2 is main-memory speed over AXI.** Latency is only hidden by *long, contiguous* streaming loads + the decoupled VLSU. | K/V blocks must be loaded as **long unit-stride streams**, reused from VRF, never re-fetched per element. |
| **C4** | **Unit-stride loads only — no gather/strided.** (`FUNCTIONALITIES.md`.) | A paged/strided KV cache cannot be gathered directly; KV must be **contiguous** in the streamed dimension, or pre-staged. |
| **C5** | **No off-chip memory yet.** Everything must fit in on-chip SRAM. | Fine for simulation and small shapes; a real multi-GB KV cache needs an off-chip controller before this is production-viable. |
| **C6** | **No hardware transcendental.** Only `vfrec7`/`vfrsqrt7` estimates exist. | The softmax `exp2` becomes a **software polynomial** (e.g. AraXL's `apps/exp`), routed via `call_extern`. |
| **C7** | **No tensor cores.** | `Q@Kᵀ` and `P@V` become **vectorized FMA loops** (`vfmacc`); the GEMM layout (lane axis) must be chosen explicitly. |

---

## 4. How we leverage AraXL to map the tiles

### 4.1 Memory-scope mapping

| TileLang scope | GPU realization | **AraXL realization** |
|----------------|-----------------|------------------------|
| `T.alloc_fragment` | tensor-core / register fragment | **VRF** (lane-major) |
| `T.alloc_shared` | shared-memory scratchpad | **VRF** (no separate scratchpad; `shared` collapses into registers) |
| `global` tensors | DRAM via L2 | **L2 SRAM**, reached by long unit-stride VLSU streams |

The key decision: **both** `fragment` and `shared` tiles target the VRF. There
is no `shared → L2` mapping; L2 is global memory only.

### 4.2 Keep the FlashAttention working set VRF-resident

FlashAttention's online softmax already **never materializes the full
`(Sq, Sk)` score matrix** — it streams over KV blocks and keeps only a small
running state. This is exactly the property that makes it fit the VRF: the
per-iteration working set is one K block, one V block, the current score tile,
the output accumulator, and the softmax statistics — all small and bounded.

We exploit this directly: the streaming-KV loop loads each `block_N` slice of K
and V from L2 into the VRF **once**, computes the block's contribution entirely
in registers, and discards it. No round-trip to L2 as a scratchpad.

### 4.3 Stream KV with long, unit-stride, decoupled loads

C3 + C4 dictate the access pattern:

- Load K/V blocks as **contiguous unit-stride** vectors so the VLSU issues long
  bursts over AXI; the crossbar/SRAM latency is amortized over the whole block.
- The Path B `VectorizePlanner` now widens `T.copy` to VLEN/element-size (not
  a fixed 128-bit / 4-fp32 cap); with `block_N=64` and VLEN=4096, copies move
  64 fp32 per slice in lowered TIR.
- Lean on the **decoupled VLSU**: prefetch the next K/V block while the current
  block is being consumed in the VRF (the vector analog of `T.Pipelined` /
  cp.async double-buffering). This requires budgeting VRF space for two K/V
  blocks (see §5).
- Lay out the KV cache so the streamed dimension is contiguous. Paged/strided
  KV (block tables) must be resolved to contiguous streams *before* the kernel,
  since AraXL cannot gather.

### 4.4 Reductions: Option A (lanes = KV columns)

The per-row softmax reduction (`reduce over the block_N columns`) has two
layouts (see the discussion in `attention.md`'s companion notes):

- **Option A — lanes = the reduction axis (columns):** strip-mine the row with
  `vsetvl`, combine partials with `vfmax`/`vfadd`, finish with a horizontal
  `vfredmax`/`vfredusum`. This is exactly AraXL's hand-written
  `apps/softmax/kernel/softmax_reduction.c`.
- **Option B — lanes = rows:** lane-private accumulation, no horizontal reduce,
  but needs a *column* of the score tile per step → **strided/gather access**,
  which AraXL does not support.

Because of **C4 (unit-stride only)**, we use **Option A**: it keeps all memory
access contiguous and matches the existing, validated AraXL softmax kernel. The
horizontal `vfred*` cost is acceptable and lands once per row per KV block.

### 4.5 `exp2` via a software polynomial

Per **C6**, the softmax `exp2` is lowered to a vectorized polynomial routine
(AraXL ships one under `apps/exp` / `apps/softmax/lib/exp.h`) called via
`call_extern`, rather than relying on a hardware instruction or a scalarized
libm path. The folded `scale × log2(e)` constant from the kernels is preserved.

### 4.6 GEMM as vectorized FMA

Per **C7**, `Q@Kᵀ` and `P@V` become vectorized FMA loops over the contraction
axis. The lane axis is chosen to keep operand access unit-stride and to feed the
horizontal-reduction layout from §4.4 (the `S = Q@Kᵀ` output is consumed
column-wise by the softmax, so the score tile is laid out columns-as-lanes).

**Implemented (llvm target):** TileLang `GemmVector` lowers each `T.gemm` to
`for i,k` serial + `for j in T.parallel(N)`, so `B[k,:]` and `C[i,:]` are
unit-stride and LLVM emits `vfmacc` on the `j` loop. This matches the access
pattern described here but does **not** yet pick softmax-specific lane layouts
or VRF-sized tiles — see `tilelang.md` §6–7.

---

## 5. VRF capacity budget (worked example)

The binding question (C2): **do the tiles fit in 64 KB?** Using `head_dim = 128`,
fp16 inputs, fp32 accumulators, and `M` = number of query rows processed
together (decode packs a GQA group; prefill uses a query tile):

Per-iteration VRF residents:

| Buffer | Shape | dtype | Size |
|--------|-------|-------|------|
| K block | `block_N × 128` | fp16 | `block_N × 256` B |
| V block | `block_N × 128` | fp16 | `block_N × 256` B |
| Score tile `acc_s` | `M × block_N` | fp32 | `M × block_N × 4` B |
| Q tile (persistent) | `M × 128` | fp16 | `M × 256` B |
| Output `acc_o` (persistent) | `M × 128` | fp32 | `M × 512` B |
| softmax stats | `~5 × M` | fp32 | `~M × 20` B |

For `M = 8`:

| `block_N` | K+V (single) | K+V (double-buffered) | score tile | Q+O+stats | **Total (double-buf)** | Fits 64 KB? |
|-----------|--------------|------------------------|------------|-----------|------------------------|-------------|
| 128 | 64 KB | 128 KB | 4 KB | ~6 KB | **138 KB** | ✗ |
| 64  | 32 KB | 64 KB  | 2 KB | ~6 KB | **72 KB**  | ✗ (tight) |
| 32  | 16 KB | 32 KB  | 1 KB | ~6 KB | **39 KB**  | ✓ |
| 16  | 8 KB  | 16 KB  | 0.5 KB | ~6 KB | **22.5 KB** | ✓ |

**Takeaways:**

- `block_N` (the KV streaming block) is the dominant VRF consumer and the main
  tuning knob.
- For `head_dim = 128`, fp16 K/V, **`block_N = 32` (or 16) with double-buffering
  fits comfortably** in the 64 KB VRF; `block_N = 64` only fits single-buffered.
- Larger `M` (packing more query rows / heads) trades against `block_N`.
- The GPU decode kernel's `block_H = 64` (batching the GQA group into the M
  dimension to feed tensor cores) is a GPU-specific choice; on AraXL we set
  `M ≈ group_size` and recover parallelism along the **lane** dimension instead,
  which shrinks the M-dimension tiles dramatically.

These numbers are illustrative; final tiling is tuned on Verilator against cycle
counts and L2 bandwidth.

---

## 6. Open risks and future work

1. **L2 bandwidth scaling.** A single SRAM bank over AXI is a bandwidth ceiling
   for `nr_lanes = 16+`. Decode is memory-bound on KV; needs measurement of
   sustained VLSU throughput vs. lane FLOP capacity.
2. **Paged KV cache (C4).** The engine's block-table KV is strided/gathered;
   AraXL needs a contiguous-stream layout or a pre-gather stage. This is the
   biggest blocker for a drop-in attention replacement.
3. **Off-chip memory (C5).** Real KV caches exceed on-chip SRAM. An off-chip
   DRAM controller turns "L2" into a genuine staging/cache tier — at which point
   a software-managed scratchpad (and the associated ISA/LLVM/TileLang work)
   becomes worth revisiting.
4. **Double-buffering VRF pressure.** Decoupled prefetch needs 2× K/V buffers;
   §5 shows this roughly halves the usable `block_N`. Whether the VLSU's
   decoupling already hides enough latency *without* explicit double-buffering
   is an empirical question.
5. **Spill behavior.** With no cache between VRF and L2, any register spill goes
   straight to main-memory latency. Tile sizing must leave margin; verify no
   spills in the generated `vsetvli`/LMUL code.

---

## 7. Summary

- AraXL has a **two-tier** memory hierarchy: a large **VRF** (64 KB at 16 lanes)
  and **L2 SRAM over AXI** (main-memory speed). There is **no scratchpad tier**
  and **no off-chip DRAM** yet.
- The GPU "shared memory" concept **collapses into the VRF**; L2 is global
  memory only. Do **not** map `shared → L2`.
- FlashAttention is a natural fit *because* its online softmax keeps a small,
  VRF-resident working set and never materializes the full score matrix.
- Strategy: **fragment + shared → VRF**, **global → L2**; stream KV in
  **long unit-stride** blocks with **decoupled prefetch**; reductions via
  **Option A** (`vfredmax`, unit-stride); `exp2` via a **software polynomial**;
  GEMM via **vectorized FMA**.
- The binding constraint is **VRF capacity**: for `head_dim = 128` fp16,
  `block_N ≈ 32` fits double-buffered in 64 KB. `block_N` is the main knob.
- This hierarchy is **proven for regular, long, unit-stride, high-intensity**
  vector kernels. The real risks are **KV gather (C4)**, **L2 bandwidth**, and
  **off-chip scaling (C5)** — all to be measured on Verilator.
