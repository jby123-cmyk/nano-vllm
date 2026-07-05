# Matmul vectorization: TileLang vs LLVM

Side-by-side examples from dumped matmul artifacts at `/tmp/rvv_demo/matmul/` (regenerate with `python demos/run_rvv_lower.py` → `demos/build_rvv/matmul/`).

---

## Part 1: TileLang vectorization (in `matmul.tir`)

TileLang vectorization shows up as **64-wide slice copies** and **64-wide broadcasts** — not as `vle`/`vfmacc` (those come later from LLVM).

### Before: source TIR (`matmul.source.tir`)

You wrote high-level `T.copy` over whole 64×64 regions:

```python
                            for ko in range(2):
                                T.copy(T.region(A[by * 64, ko * 64], 1, 64, 64), T.region(A_local[0, 0], 2, 64, 64))
                                T.copy(T.region(B[ko * 64, bx * 64], 1, 64, 64), T.region(B_local[0, 0], 2, 64, 64))
                                T.gemm(T.region(A_local[0, 0], 1, 64, 64), T.region(B_local[0, 0], 1, 64, 64), T.region(C_local[0, 0], 3, 64, 64), ...)
```

No explicit vector width yet — just "copy this 64×64 tile."

### After: lowered TIR (`matmul.tir`) — TileLang widened the copies

**Clear `C_local` — 64-wide broadcast per chunk:**

```python
                for i in T.unroll(64):
                    C_local_1[i * 64:i * 64 + 64] = T.Broadcast(T.float32(0.0), 64)
```

**`A` copy — 64 floats per iteration instead of one-at-a-time:**

```python
                    for i_j_fused in range(64):
                        A_2 = T.Buffer((16384,), data=A)
                        A_local_1[i_j_fused * 64:i_j_fused * 64 + 64] = A_2[by * 8192 + i_j_fused * 128 + ko * 64:by * 8192 + i_j_fused * 128 + ko * 64 + 64]
```

**`B` copy — same pattern:**

```python
                    for i_j_fused in range(64):
                        B_2 = T.Buffer((16384,), data=B)
                        B_local_1[i_j_fused * 64:i_j_fused * 64 + 64] = B_2[ko * 8192 + i_j_fused * 128 + bx * 64:ko * 8192 + i_j_fused * 128 + bx * 64 + 64]
```

**`C` writeback — same 64-wide slices:**

```python
                for i_j_fused in range(64):
                    C_2 = T.Buffer((16384,), data=C)
                    C_2[by * 8192 + i_j_fused * 128 + bx * 64:by * 8192 + i_j_fused * 128 + bx * 64 + 64] = C_local_1[i_j_fused * 64:i_j_fused * 64 + 64]
```

**What to look for:** the `[start : start+64]` slice notation. That is TileLang saying "move 64 contiguous floats per loop trip." 64 trips × 64 floats = 4096 elements. The width 64 matches your tile row size (not the hardware max of 128).

### Same file — what TileLang did NOT vectorize

The GEMM compute is still **scalar element access** in TIR:

```python
                    for i, k in T.grid(64, 64):
                        a_val: T.float32 = A_local_1[i * 64 + k]
                        for j in T.parallel(64):
                            C_local_1[i * 64 + j] = C_local_1[i * 64 + j] + a_val * B_local_1[k * 64 + j]
```

One `j` at a time, one float per side. `T.parallel` marks parallelism for the compiler; it is not SIMD yet.

---

## Part 2: LLVM vectorization (TIR → `.ll` → `.s`)

LLVM vectorizes **both** the copy loops (from TileLang's widened slices) **and** the GEMM `j` loop (which TileLang left scalar).

### A. GEMM math — TIR input (still scalar)

Same loop as above — this is what LLVM receives for the multiply:

```python
                    for i, k in T.grid(64, 64):
                        a_val: T.float32 = A_local_1[i * 64 + k]
                        for j in T.parallel(64):
                            C_local_1[i * 64 + j] = C_local_1[i * 64 + j] + a_val * B_local_1[k * 64 + j]
```

Conceptually: `C[i,j] += a_val * B[k,j]` with `a_val` fixed across `j`.

### B. GEMM math — LLVM IR (`matmul.ll`)

LLVM's loop vectorizer created a `vector.body` with scalable vectors:

```llvm
  %broadcast.splatinsert = insertelement <vscale x 4 x float> poison, float %a_val, i64 0, !dbg !71
  %broadcast.splat = shufflevector <vscale x 4 x float> %broadcast.splatinsert, <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer, !dbg !71
  br label %vector.body, !dbg !71

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %offset.idx = add i64 %index, %smin, !dbg !71
  %41 = getelementptr float, ptr %invariant.gep, i64 %offset.idx, !dbg !71
  %wide.load = load <vscale x 4 x float>, ptr %41, align 4, !dbg !71, !tbaa !52, !alias.scope !75
  %42 = getelementptr float, ptr %invariant.gep2, i64 %offset.idx, !dbg !71
  %wide.load7 = load <vscale x 4 x float>, ptr %42, align 4, !dbg !71, !tbaa !52, !alias.scope !78, !noalias !75
  %43 = tail call <vscale x 4 x float> @llvm.fmuladd.nxv4f32(<vscale x 4 x float> %broadcast.splat, <vscale x 4 x float> %wide.load, <vscale x 4 x float> %wide.load7), !dbg !71
  store <vscale x 4 x float> %43, ptr %42, align 4, !dbg !71, !tbaa !52, !alias.scope !78, !noalias !75
  %index.next = add nuw i64 %index, %40
  %44 = icmp eq i64 %index.next, %n.vec
  br i1 %44, label %middle.block, label %vector.body, !prof !80, !llvm.loop !81
```

Read this as:

- `%broadcast.splat` = `a_val` replicated to all lanes
- `%wide.load` = vector load of `B[k, j:j+lanes]`
- `%wide.load7` = vector load of `C[i, j:j+lanes]`
- `@llvm.fmuladd.nxv4f32` = `C += a_val * B` across lanes
- `!"llvm.loop.isvectorized", i32 1` metadata confirms vectorization

### C. GEMM math — RVV assembly (`matmul.s`)

The LLVM IR lowers to real RVV FMA instructions:

```asm
	vsetvli	t3, zero, e32, m2, ta, ma
	vfmv.v.f	v8, fa5
	slli	t3, a4, 2
	add	a4, a0, a5
	add	a4, a4, t3
	slli	a6, a6, 1
	add	t4, a3, a2
	add	t3, t4, t3
.LBB2_9:
	vl2re32.v	v10, (t3)
	vl2re32.v	v12, (a4)
	vfmacc.vv	v12, v8, v10
	vs2r.v	v12, (a4)
	sub	t2, t2, t1
	add	a4, a4, a6
	add	t3, t3, a6
	bnez	t2, .LBB2_9
```

| Instruction | Role |
|-------------|------|
| `vfmv.v.f v8, fa5` | broadcast scalar `a_val` into vector register `v8` |
| `vl2re32.v v10, (t3)` | vector load `B[k, :]` chunk |
| `vl2re32.v v12, (a4)` | vector load `C[i, :]` accumulator chunk |
| `vfmacc.vv v12, v8, v10` | `v12 += v8 * v10` (vector FMA) |
| `vs2r.v v12, (a4)` | vector store `C` back |

That is LLVM vectorization of the **compute** loop.

---

### D. Copy loops — TIR (TileLang) → ASM (LLVM)

**TIR** (TileLang already widened to 64-wide slices):

```python
                    for i_j_fused in range(64):
                        A_2 = T.Buffer((16384,), data=A)
                        A_local_1[i_j_fused * 64:i_j_fused * 64 + 64] = A_2[by * 8192 + i_j_fused * 128 + ko * 64:by * 8192 + i_j_fused * 128 + ko * 64 + 64]
```

**ASM** (LLVM turned each 64-wide slice into `vle`/`vse`):

```asm
.LBB0_57:
	li	a2, 64
	vsetvli	zero, a2, e32, mf2, ta, ma
	vle32.v	v8, (a0)
	vse32.v	v8, (a1)
.Ltmp99:
	addi	a1, a1, 256
.Ltmp100:
	addi	a0, a0, 512
	ld	a2, 112(sp)
	bne	a1, a2, .LBB0_57
```

| Instruction | Role |
|-------------|------|
| `vsetvli zero, a2, e32, mf2` | set vector length to 64, e32 |
| `vle32.v v8, (a0)` | vector load 64 floats from global `A` |
| `vse32.v v8, (a1)` | vector store 64 floats to local `A_local` |
| loop with `+256` / `+512` byte offsets | next 64-float chunk |

So for copies: **TileLang widens in TIR** (`[i*64 : i*64+64]`), **LLVM emits the actual RVV load/store** (`vle32`/`vse32`).

---

## Quick map: who did what

| Location | TileLang | LLVM |
|----------|----------|------|
| `matmul.tir` lines 85–86, 90–95, 101–103 | 64-wide slice copies/broadcasts | — |
| `matmul.tir` lines 97–100 | left scalar (`for j in T.parallel(64)`) | — |
| `matmul.ll` `vector.body` + `fmuladd.nxv4f32` | — | vectorized GEMM |
| `matmul.s` `vfmacc.vv` loop | — | vectorized GEMM |
| `matmul.s` `vle32`/`vse32` loops | widened the TIR slices | emitted RVV instructions |

**One-line summary:** look for `[*64 : *64+64]` slices in `.tir` for TileLang; look for `vfmacc`/`vle32` in `.s` (or `<vscale x 4 x float>` in `.ll`) for LLVM.
