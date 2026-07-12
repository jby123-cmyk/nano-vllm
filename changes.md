# Changes — VLEN sync, VLEN-derived tiles, vector `exp2`, `acc_s_cast` drop, generic CPU parallel-loop vectorization

Phases 1–3 of `implementation_plan.md` plus Phase 4 (generic elementwise
vectorization). Phase 1 = target VLEN sync + VLEN-derived decode tiles. Phase 2
= vectorized software softmax `exp2`. Phase 3 = drop the redundant `acc_s_cast`
copy on the fp32 path. **Phase 4 = a generic TileLang CPU-pipeline pass
(`SerializeOuterParallel`) that makes *every* elementwise `T.Parallel(M, N, …)`
fragment loop vectorize on CPU — softmax, mask, rescale, and the final divide —
without any per-kernel edit.** Phase 4 supersedes the hand-written serial-row /
parallel-column `exp2` restructure from Phase 2: the kernels are now written in
the plain `T.Parallel(M, N)` form and the compiler does the restructuring.

## Direction

The target is a **strided-load-capable** vector ISA, studied via
**instruction-set-level simulation**. The optimization goal is **maximum
vectorization / efficiency**, *not* unit-stride memory legality. Strided/indexed
loads (`vlse`, `vluxei`) are allowed and preferred when they keep a loop
vectorized. (Earlier drafts optimized for AraXL's unit-stride-only VLSU; that
constraint no longer applies.)

## Summary

This changeset delivers two efficiency wins and documents one rollback.

**Kept:**

1. **Target VLEN sync** — `nr_lanes` is now the single source of VLEN, so LLVM's
   effective VLEN can never desync from `vector-width`. Fixes a silent footgun
   where a stale `+zvl4096b` in `DEFAULT_MATTR` outranked `--nr-lanes 8`.
2. **VLEN-derived decode tiles** — decode `block_N` is derived from the target
   VLEN and `block_H` from the GQA group size, so the reduce/GEMM vectors fill
   the hardware and scale with lanes (VL 128 @ 4 lanes → 256 @ 8 lanes).

**Rolled back:**

3. **Unit-stride `Q @ K^T` staging (`stage_kt`)** — briefly added to remove the
   `vluxei64` gather, but it replaced a fully-vectorized `vfmacc` score matmul
   with a **scalar** on-chip transpose. Under the vectorization-first goal that
   is a regression, so it was removed. Decode keeps `transpose_B=True` (vectorized
   gather).

The **CUDA / engine path is unchanged** throughout.

---

## Results (verified, `run_rvv_lower.py`)

| Metric | Baseline | Now |
|--------|----------|-----|
| `llvm_get_vector_width()` vs `1024*nr_lanes` | mismatch possible | equal (asserted at startup) |
| decode reduce `vle`/`vfred` VL | 128 (fixed) | 128 @ 4 lanes, **256 @ 8 lanes** |
| score matmul | `vfmacc` + `vluxei64` gather (vectorized) | same — kept vectorized |
| softmax `exp2f` (libm) calls, decode/prefill | 5 / 5 (scalar) | **0 / 0** (vector poly) |
| softmax exponent (elementwise) | scalar loop | **vector** `vfcvt`/`vfmadd`/`vsll.vi` |
| `acc_o` rescale + final divide (elementwise) | scalar loop | **vector** `vfmul` (invariant load hoisted) |
| prefill causal mask (fp32) | scalar loop | **vector** compare/merge |
| decode mask (uint8 compare) | scalar | scalar — LLVM mixed-width limit (documented) |
| `acc_s_cast` copy (fp32 P@V input) | full-tile copy per KV block | **removed** (reads `acc_s`) |
| numeric golden | pass | pass — ladder @ 4 & 8 lanes; 42/42 pytest shapes |

Reduce VL now scales with the target (decode, `nr_lanes=8`, `block_N=256`):

```asm
li      a7, 256
vsetvli zero, a7, e32, m1, tu, ma
...
vle32.v v8, (a1)
vfredusum.vs v9, v8, v11
```

---

## File-by-file changes

### `nanovllm/backends/tilelang/rvv_lower.py`

`DEFAULT_MATTR` is base ISA only; `rvv_target()` makes `nr_lanes` authoritative:

```python
DEFAULT_MATTR = ("+v", "+m", "+f", "+d")   # +zvl injected from nr_lanes

def rvv_target(mtriple=..., mattr=DEFAULT_MATTR, mabi=..., nr_lanes=DEFAULT_NR_LANES):
    ...
    vlen_bits = 1024 * nr_lanes
    mattr = [tok for tok in mattr if not tok.startswith("+zvl")]  # strip stale VLEN
    mattr.append(f"+zvl{vlen_bits}b")                             # inject correct VLEN
    return tvm.target.Target({..., "mattr": mattr, "vector-width": vlen_bits})
```

Two helpers:

```python
def vlen_f32_elements(nr_lanes: int) -> int:
    """fp32 elements per cluster at LMUL=1 (VLEN_bits / 32): 128 @ 4, 256 @ 8."""
    return (1024 * nr_lanes) // 32

def assert_target_vlen(target, nr_lanes) -> int:
    """Fail loudly if LLVM's effective VLEN (+zvl*b) != 1024*nr_lanes."""
    from tvm.target.codegen import llvm_get_vector_width
    expected = 1024 * nr_lanes
    with target:
        actual = llvm_get_vector_width()
    if actual != expected:
        raise ValueError(f"RVV target VLEN mismatch: {actual} vs expected {expected} ...")
    return actual
```

The strided-op scan (`_RVV_STRIDED_RE` / report labels) was reframed from
"AraXL cannot execute" to informational — strided ops are supported and fine.

### `demos/run_rvv_lower.py`

`--nr-lanes` defaults to `DEFAULT_NR_LANES` (4). `main()` asserts the VLEN and
derives the decode tiles:

```python
vlen_bits = assert_target_vlen(target, args.nr_lanes)
decode_block_N = vlen_f32_elements(args.nr_lanes)          # 128 @ 4, 256 @ 8
decode_block_H = args.num_heads // args.num_kv_heads       # GQA group size
decode_seqlen_kv = ceil(args.seqlen_kv / decode_block_N) * decode_block_N
```

These derived tiles feed both the decode lowering (`get_tir`) and the numeric
check, so the artifact and its golden use the same shapes. The report's
limitation narrative was updated (strided loads are the desired vectorized path;
softmax `exp2` is the remaining scalar cost).

### `nanovllm/backends/tilelang/attention.py`, `nanovllm/stages/attention.py`

The temporary `stage_kt` build flag and its plumbing (kernel builder,
`run_tilelang_attention_decode`, `AttentionStage.decode_stage_kt`,
`numeric_check_attention_decode`) were **removed**. The decode score GEMM is back
to the original one-liner:

```python
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
```

which keeps the matmul fully vectorized (`vfmacc` + `vluxei64` gather).

---

## Phase 2 — vectorized software softmax `exp2`

**Problem.** `T.exp2` lowers to a scalar `call exp2f` (libm) per element on the
CPU/RVV path. libm has no RVV vector variant, so the call is a hard barrier: the
softmax exponent loop stays scalar (5 `exp2f` call sites in the decode `.s`).

**Fix.** Inline a vectorizable `2**x` in pure arithmetic so no call remains and
LLVM can SIMD the loop. `_exp2_poly` (in `attention.py`) uses the Cephes
single-precision `exp2f` polynomial with range reduction `2**x = 2**n · 2**r`,
`n = round(x)`, `r ∈ [-0.5, 0.5]`; `2**n` is built directly in the fp32 exponent
field via an integer shift + `reinterpret`:

```python
def _exp2_poly(x):
    x = T.max(x, T.float32(-127.0))          # flush 2**x→0 for masked/-inf inputs
    n = T.floor(x + T.float32(0.5))
    r = x - n
    q = T.float32(_EXP2_POLY[0])
    for coeff in _EXP2_POLY[1:]:             # Horner, degree-6, ~1 ulp
        q = q * r + T.float32(coeff)
    two_r = q * r + T.float32(1.0)
    pow2n = T.reinterpret(T.shift_left(T.Cast("int32", n) + 127, 23), "float32")
    return two_r * pow2n
```

The polynomial alone was not enough: TileLang flattens `T.Parallel(block_H,
block_N)` into a 1-D loop indexed as `scores_max[i // block_N]`, and that
integer-division index blocks LLVM auto-vectorization. This flattening problem
is **not** `exp2`-specific — it hits every 2-D elementwise fragment loop — so
its fix moved out of the kernel and into the compiler (Phase 4). The kernel just
selects the polynomial:

```python
def _exp2(v):
    return _exp2_poly(v) if vec_exp2 else T.exp2(v)   # poly on CPU, HW exp2 on CUDA
...
for i, j in T.Parallel(block_H, block_N):             # plain 2-D loop
    acc_s[i, j] = _exp2(acc_s[i, j] * scale - scores_max[i] * scale)
```

`vec_exp2` (set by the run helpers to `backend == "cpu"`) only chooses poly vs
hardware `exp2` — a genuine CPU/CUDA algorithm choice — so **CUDA keeps hardware
`exp2`**. It no longer controls loop structure; Phase 4 handles that generically.

**Result:** decode/prefill `.s` go from 5 → **0** `exp2f` calls; with Phase 4 the
exponent emits vector `vfcvt` / `vfmadd` / `vsll.vi` (decode: 12 `vfcvt`, 6
`vsll.vi`). Numeric golden passes (42/42 shapes in
`tests/test_attention_rvv_numeric.py`; poly error ~1e-7 ≪ 1e-2 tolerance).

## Phase 3 — drop the redundant `acc_s_cast` copy (fp32)

The `P @ V` GEMM input must match its `in_dtype`. On CUDA (`in_dtype=float16`,
`accum=float32`) the fp32 scores are copied into an fp16 `acc_s_cast` buffer. On
the fp32 vector path `in_dtype == accum_dtype`, so that copy is pure overhead —
a `block_H×block_N` (resp. `block_M×block_N`) stack round-trip per KV block.
Gate it on the dtypes and feed `acc_s` straight in when they match:

```python
need_cast = in_dtype != accum_dtype
...
if need_cast:
    acc_s_cast = T.alloc_fragment([block_H, block_N], in_dtype)
...
if need_cast:
    T.copy(acc_s, acc_s_cast)
...
T.gemm(acc_s_cast if need_cast else acc_s, V_shared, acc_o, policy=...)
```

Purely dtype-driven (no flag): fp16 CUDA still casts; fp32 skips. `acc_s_cast`
no longer appears in the fp32 decode/prefill TIR.

### `nanovllm/backends/tilelang/attention.py` (Phase 2/3)

- `_EXP2_POLY` coefficients + `_exp2_poly` helper (module level).
- Decode and prefill builders take `vec_exp2: bool = False`; inner `_exp2`
  selects poly vs `T.exp2` (poly on CPU, hardware `exp2` on CUDA); the exponent
  stays a plain `T.Parallel(block_*, block_N)` loop (Phase 4 vectorizes it);
  `acc_s_cast` alloc/copy/GEMM-input gated on `need_cast`.
- `run_tilelang_attention_decode`/`_prefill` append `backend == "cpu"` to
  `build_args`, so the CPU numeric path validates exactly what the RVV artifact
  emits.

## Phase 4 — generic CPU parallel-loop vectorization (`SerializeOuterParallel`)

**Problem (general, not attention-specific).** A multi-axis `T.Parallel(M, N)`
is an elementwise SIMT loop. On GPU each of the `M·N` points maps to a thread.
On CPU there are no threads, so TileLang fuses the nest into one flat loop and
the per-row / per-column broadcast operands become non-affine indices —
`row[i // N]`, `mask[i % N]`. LLVM's loop vectorizer cannot handle those `//` /
`%` indices, so the whole body falls back to scalar `flw`/`fsw`. This is why the
softmax exponent, the mask apply, the `acc_o` rescale, and the final divide were
all scalar. Any kernel written with 2-D `T.Parallel` elementwise loops hits it.

**Fix — a CPU-pipeline pass, so kernels stay in the natural form.** New pass
`tilelang/transform/serialize_outer_parallel.py`, wired into
`tilelang/cpu/pipeline.py` right before `LayoutInference` (GPU pipelines are
untouched). It applies two correctness-preserving rewrites to every parallel
nest:

1. **Serialize outer axes.** Any parallel loop that *contains another parallel
   loop* becomes serial, leaving only the innermost axis parallel (the CPU
   vector lane). `kParallel → kSerial` never changes results — it only changes
   which axis is the lane, and on CPU the outer axes had no hardware threads
   anyway.
2. **Hoist loop-invariant loads.** After (1) the inner body may still read a
   fragment indexed only by the (now serial) outer axes — e.g. `scores_max[i]`.
   A fragment load left inside the vector loop keeps LLVM from vectorizing, so
   each invariant value-load is pulled into a scalar `Bind` in the enclosing
   serial scope and referenced as a broadcast. This is the generic form of the
   hand-written `neg_i = scores_max[i]` hoist that Phase 2 used.

```
for i, j in T.Parallel(M, N):                for i in T.serial(M):
    acc[i, j] = f(acc[i, j], row[i])   ==>       t = row[i]          # hoisted
                                                 for j in T.Parallel(N):
                                                     acc[i, j] = f(acc[i, j], t)
```

Nests carrying an explicit `parallel_loop_layout` (author-chosen layout,
validated as a whole nest) or a `reducer_info` (cross-lane reduction) annotation
are skipped, so `T.copy`/`T.reduce`/`T.gemm` lowering is unaffected — the pass
only touches raw elementwise `T.Parallel` loops.

**Why a compiler pass and not a kernel helper.** The `@T.prim_func` is AST-parsed
(TVMScript), so a `body`-callable Python helper can't emit loops; and the
serial/parallel split must be **CPU-only** (on GPU it would serialize `M` and
throw away thread parallelism). A CPU-pipeline pass satisfies both: kernels keep
the GPU-optimal `T.Parallel(M, N)` form, and the compiler restructures for CPU.

**Result (decode `.s`, fp32, 4 lanes).** softmax exponent, `acc_o` rescale, and
the final divide now vectorize (`vfcvt` 0→12, vector `vfmul` 0→8, plus
`vfmadd`/`vmerge`/`vmflt`); `exp2f` calls stay 0. Prefill likewise (index-based
causal mask included). All four ladder kernels report `vectorized=True`,
`numeric=pass`, and **42/42** `test_attention_rvv_numeric.py` shapes pass.

**Known residual.** The *decode* mask (`mask_local[j] != 0 ? acc_s : -inf`)
still lowers scalar: it mixes a `uint8` compare with an `fp32` select, and LLVM
does not auto-vectorize that mixed-element-width masked-select. This is an LLVM
codegen limitation orthogonal to the flattening the pass fixes — the *pattern*
(mask apply) vectorizes for same-width operands (prefill's fp32 mask does).
Forcing it would require widening the mask to fp32, i.e. an attention-specific
edge case, so it is left as-is and documented.

### `tilelang/transform/serialize_outer_parallel.py`, `tilelang/cpu/pipeline.py`, `tilelang/transform/__init__.py`

New CPU pass + registration + one pipeline line (before `LayoutInference`).

### `nanovllm/backends/tilelang/attention.py` (Phase 4)

The hand-written serial-row / parallel-column `exp2` restructure (decode and
prefill) was reverted to the plain `T.Parallel(block_*, block_N)` loop. No other
kernel edits: mask, rescale, and divide were already plain `T.Parallel` and now
vectorize via the pass. Result: the attention kernels contain zero CPU-specific
loop-shape special-casing.

### Docs

- `background.md` — §1/§5/§7 reframed to the strided-capable, vectorization-first
  direction; the AraXL unit-stride constraint and the `stage_kt` fix removed.
  §6/§8/§9 updated: softmax `exp2` now vectorized (poly), `acc_s_cast` dropped;
  new §8 documents the generic `SerializeOuterParallel` pass.
- `implementation_plan.md` — Phase 1 target sync + VLEN tiles (done); Phase 2
  vector `exp2` (done); Phase 3 `acc_s_cast` drop (done); autotune deferred.
- `usage.md` — §6 flags **lost vectorization** (scalar loops), not strided loads;
  notes `exp2f` should now be absent.

- `background.md` — §1/§5/§7 reframed to the strided-capable, vectorization-first
  direction; the AraXL unit-stride constraint and the `stage_kt` fix removed.
- `implementation_plan.md` — Phase 1 = target sync + VLEN tiles (done);
  `stage_kt` noted as rolled back; next priority is vector `exp2` (Phase 2).
- `usage.md` — §6 flags **lost vectorization** (scalar loops), not strided loads.

---

## Appendix — decode assembly, before vs after

Isolating the two changes on the decode kernel (fp32, VLEN=4096, `block_N=128`,
`block_H=4`) shows they are independent, and that the **compiler pass**, not the
polynomial, is what vectorizes the loops:

| variant | `exp2f` calls | exp2 / rescale / divide loops |
|---|---|---|
| pre-change: hardware `exp2` + no pass | **5** | scalar |
| Phase 2 only: poly `exp2`, no pass | 0 | **still scalar** (poly via scalar `fcvt`/`fmadd`) |
| current: poly `exp2` + `SerializeOuterParallel` | 0 | **vector** (`vfcvt`/`vfmul`/`vfdiv`/`vmerge`) |

Vector-op counts (`.s`): before `{vfmacc 2 (gemm only), vsll 2, vfredmax/sum 4/4}`
→ after `{vfcvt 12, vfmacc 22 (gemm + Horner), vfmadd 4, vfmul 12, vfdiv 4,
vmerge 4, vsll 6}`.

### Softmax exponent (the 4×128 `acc_s` loop)

**Before** — one flat 512-iteration scalar loop with a libm call per element. Note
`srliw a0, s4, 7` = `i // 128`: the fused iterator divided to recover the per-row
`scores_max` index — exactly the non-affine broadcast that blocks the vectorizer:

```asm
.LBB0_146:
    srliw   a0, s4, 7            # i // 128  (row index from fused iterator)
    slli    a0, a0, 2
    ld      a1, 232(sp)
    add     a0, a1, a0
    flw     fa5, 0(a0)           # scores_max[i // 128]   (broadcast operand)
    flw     fa4, 0(s6)           # acc_s[fused]
    fmul.s  fa5, fa5, fs1
    fmsub.s fa0, fa4, fs1, fa5
    call    exp2f                # scalar libm call, per element
    fsw     fa0, 0(s6)
    addi    s4, s4, 1
    addi    s6, s6, 4
    li      a0, 512
    bne     s4, a0, .LBB0_146    # 512 = 4 rows × 128 cols, scalar
```

**After** — the pass serializes the row axis and hoists `scores_max[i]` (`fa5`),
so each row is one straight-line 128-lane (`m2`) vector evaluation of the Cephes
`2^x` polynomial; no loop, no libm:

```asm
    vsetvli   a7, zero, e32, m2, ta, ma
    vfmv.v.f  v10, ft1
    vfmsac.vf v10, fa5, v8          # x = acc_s*scale - max*scale  (max broadcast)
    vmfgt.vf  v0, v10, fs1
    vmerge.vvm v8, v8, v10, v0       # clamp max(x, -127)
    vfcvt.x.f.v v14, v10             # floor  (round-to-int)
    vfmacc.vf v12, fa3, v8           # ┐
    vfmacc.vv v10, v8, v12           # │ degree-6 Horner (2^r)
    vfmacc.vv v16, v8, v10           # │
    ...                              # ┘
    vsll.vi   v12, v14, 23           # 2^n via exponent bit-field
    vadd.vx   v12, v12, a7
    vfmul.vv  v8, v10, v12           # 2^r · 2^n
    vs2r.v    v8, (a6)
```

### `acc_o` rescale and final divide (proof the pass is generic, not exp2-specific)

**Before** (Phase-2 build, pass still off) — the final divide is a 256-iteration
scalar loop; `srli a2, a0, 6` = `i // 64` is again the flattened broadcast index:

```asm
.LBB0_174:
    flw     fa5, 0(a1)           # acc_o[fused]
    srli    a2, a0, 6            # i // 64   (row index)
    ...
    flw     fa4, 0(a2)           # logsum[i // 64]  (broadcast operand)
    fdiv.s  fa5, fa5, fa4        # scalar divide
    fsw     fa5, 0(a1)
    addi    a0, a0, 1
    li      a2, 256
    bne     a0, a2, .LBB0_174    # 256 = 4 × 64, scalar
```

**After** — `logsum[i]` is hoisted to a scalar (`fa5`, the pass's `licm_2`) and
broadcast; the 64-wide row is one vector op. The `acc_o` rescale is identical with
`vfmul.vf`:

```asm
.LBB0_237:
    vl2re32.v v8, (a1)
    vfdiv.vf  v8, v8, fa5         # divide by broadcast logsum[i]
    vs2r.v    v8, (a1)
```

The mask apply is the one loop that stays scalar after the pass (decode's `uint8`
compare + `fp32` select); see the Phase 4 "known residual" note.

### GEMM and reduction — **unchanged** (assembly for reference)

Neither GEMM nor reduce is touched by this work, so there is no before/after —
the assembly below is both. `SerializeOuterParallel` runs *before* `LowerTileOp`,
where `T.gemm`/`T.reduce` are still opaque tile intrinsics (not yet loop nests),
and it additionally skips any `reducer_info` nest. The GEMMs use the pre-existing
`GemmVector` lowering (emitted as `__tvm_parallel_lambda` bodies) and the reduce
uses the pre-existing RVV `vfred` path. Both were already vector before Phase 2/4.

**`Q @ Kᵀ` (transposed B → strided gather + `vfmacc`), loop `LBB2_11`:**

```asm
.LBB2_11:
    vsll.vi     v12, v8, 6         # j * 64                 (column-major K index)
    vadd.vx     v12, v12, a2
    vsetvli     zero, zero, e64, m4, ta, ma
    vsext.vf2   v16, v12           # widen indices to i64
    vsll.vi     v12, v16, 2        # * 4 bytes
    vsetvli     zero, zero, e32, m2, ta, ma
    vluxei64.v  v16, (a0), v12     # GATHER Kᵀ column (desired on strided target)
    vl2re32.v   v12, (a5)          # acc_s row
    vfmacc.vv   v12, v10, v16      # acc_s += a_val * K   (a_val splat in v10)
    vs2r.v      v12, (a5)
    vadd.vx     v8, v8, t1
    sub         t2, t2, t1
    add         a5, a5, a6
    bnez        t2, .LBB2_11
```

**`P @ V` (unit-stride V → no gather + `vfmacc`), loop `LBB3_9`:**

```asm
.LBB3_9:
    vl2re32.v   v10, (t3)          # V row      (unit stride — whole-register load)
    vl2re32.v   v12, (a4)          # acc_o row
    vfmacc.vv   v12, v8, v10       # acc_o += a_val * V   (a_val splat in v8)
    vs2r.v      v12, (a4)
    sub         t2, t2, t1
    add         a4, a4, a6
    add         t3, t3, a6
    bnez        t2, .LBB3_9
```

**`reduce_max` — one `vle` + one horizontal `vfredmax` per row (VL = block_N):**

```asm
    vle32.v     v8, (a0)           # load the whole 128-wide score row
    vsetvli     zero, zero, e32, m1, ta, ma
    vl1r.v      v9, (init)         # v9 = splat(-inf)
    vfredmax.vs v9, v8, v9         # horizontal max, seeded with v9[0]
    vfmv.f.s    fa5, v9            # extract scalar
```

**`reduce_sum` — identical with `vfredusum` and a `0.0` seed:**

```asm
    vle32.v     v8, (a0)           # load the whole 128-wide P row
    vsetvli     zero, zero, e32, m1, ta, ma
    vl1r.v      v11, (init)        # v11 = splat(0.0)
    vfredusum.vs v9, v8, v11       # horizontal sum, seeded with v11[0]
    vfmv.f.s    fa5, v9            # extract scalar
```

The score row is exactly one `m1` register at VLEN=4096 (`block_N = VLEN/32 =
128`), so the reduce is a single `vle`+`vfred` with no accumulate loop; a
`block_N` wider than one register would add the Mode-B serial `vfadd.vv`/`vfmax.vv`
accumulate before the final fold (see `background.md` §6). The `vl1r.v` init reload
is RVV backend spill overhead, not part of the reduce algorithm.

---

## How to reproduce

```bash
conda activate nanovllm
export PYTHONPATH=/mnt/ssd/jby123/tilelang:/mnt/ssd/jby123/tilelang/build/python:$PYTHONPATH
export LD_LIBRARY_PATH=/mnt/ssd/jby123/miniconda3/envs/nanovllm/lib:$LD_LIBRARY_PATH

python demos/run_rvv_lower.py                # 4 lanes: block_N=128
python demos/run_rvv_lower.py --nr-lanes 8   # 8 lanes: block_N=256
```
