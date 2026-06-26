# Stage 1 Report — TileLang → LLVM → RVV (Demo Backend)

_Generated: 2026-06-25T21:42:09.325024+00:00_

_Target: `{"kind":"llvm","tag":"","keys":["cpu"],"mabi":"lp64d","mtriple":"riscv64-unknown-elf","mattr":["+v","+m","+f","+d"]}`_


This report records the **demo** described in `tilelang.md`: TileLang kernels routed through the CPU/`llvm` pass pipeline at the AraXL RVV target, with LLVM auto-vectorization on. **Path B** TileLang backend work (see `tilelang.md`) added CPU realizations for GPU-style tile ops so FlashAttention decode/prefill lower end-to-end; LLVM still owns vector scheduling (no custom LMUL tuning or AraXL subtarget). Remaining limitations are **observed in the artifacts**, not hidden.


## Reproduce

```bash
python demos/run_rvv_lower.py
```

## Per-kernel results

| Kernel | Compiled | Vectorized | Numeric (host) | Failing pass | Notes |
|--------|----------|------------|----------------|--------------|-------|
| `elementwise` | yes | yes | pass | `—` | 5 RVV op kinds |
| `matmul` | yes | yes | pass | `—` | 10 RVV op kinds; strided: vlse32 |
| `attention_decode` | yes | yes | pass | `—` | 15 RVV op kinds; strided: vlse32 |
| `attention_prefill` | yes | yes | pass | `—` | 15 RVV op kinds; strided: vlse32 |

## Emitted RVV instructions (evidence for T3)


### `elementwise`

- Artifacts: `elementwise.tir`, `elementwise.ll`, `elementwise.s`
- Vector ops: `vfadd.vv, vmv, vse64.v, vsetivli, vsetvli`
- Strided/indexed ops (AraXL unit-stride only): `none`

```asm
.Ltmp80:
	xori	a7, t1, 256
	vsetvli	t2, zero, e32, m2, ta, ma
	mv	t2, a2
	mv	t3, a1
	mv	t4, a3
	mv	t5, a7
.Ltmp81:
.LBB0_50:
	vl2re32.v	v8, (t2)
	vl2re32.v	v10, (t3)
	vfadd.vv	v8, v8, v10
	vs2r.v	v8, (t4)
	sub	t5, t5, a0
```

### `matmul`

- Artifacts: `matmul.tir`, `matmul.ll`, `matmul.s`
- Vector ops: `vfmul.vv, vfmv, vfmv.f, vfredosum, vle32.v, vmv, vse32.v, vse64.v, vsetivli, vsetvli`
- Strided/indexed ops (AraXL unit-stride only): `vlse32`

```asm
	andi	t2, t1, 64
	xori	t1, t2, 64
	vsetvli	t3, zero, e32, m1, ta, ma
	vfmv.s.f	v8, fa5
	mv	t3, a7
	mv	t4, a2
	mv	t5, t1
.Ltmp114:
.LBB0_66:
	vl2re32.v	v10, (t4)
	vsetvli	t6, zero, e32, m2, ta, ma
	vlse32.v	v12, (t3), s4
	vfmul.vv	v10, v10, v12
	vfredosum.vs	v8, v10, v8
```

### `attention_decode`

- Artifacts: `attention_decode.tir`, `attention_decode.ll`, `attention_decode.s`
- Vector ops: `vfmacc.vv, vfmsub.vf, vfmul.vf, vfmul.vv, vfmv, vfmv.f, vfmv.v, vfredosum, vle32.v, vmv, vse32.v, vse64.v, vse8.v, vsetivli, vsetvli`
- Strided/indexed ops (AraXL unit-stride only): `vlse32`

```asm
	addi	a1, a0, 64
	li	a2, 32
	vsetvli	zero, a2, e8, m2, ta, ma
	vlseg2e8.v	v8, (a0)
	vlseg2e8.v	v10, (a1)
	ld	a0, 1376(sp)
	vse8.v	v8, (a0)
	ld	a0, 40(sp)
	vse8.v	v10, (a0)
	li	a1, 64
	j	.LBB0_121
.Ltmp223:
.LBB0_120:
	li	a1, 0
```

### `attention_prefill`

- Artifacts: `attention_prefill.tir`, `attention_prefill.ll`, `attention_prefill.s`
- Vector ops: `vadd, vfmacc.vv, vfmsub.vf, vfmul.vf, vfmul.vv, vfmv, vfmv.f, vfmv.v, vfredosum, vle32.v, vmv, vse32.v, vse64.v, vsetivli, vsetvli`
- Strided/indexed ops (AraXL unit-stride only): `vlse32`

```asm
	sd	a0, 664(sp)
	vmv.s.x	v16, zero
	vsetvli	a0, zero, e32, m2, ta, ma
	vid.v	v8
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, sp, a0
	addi	a0, a0, 720
	vs2r.v	v8, (a0)
	addi	a0, t3, 160
	sd	a0, 456(sp)
	addi	a0, t3, 176
	sd	a0, 448(sp)
	addi	a0, t3, 192
```

## Section 6 limitations — observed in the emitted artifacts

1. **Scalar-fallback GEMM.** `T.gemm` lowers via `GemmScalar` to a plain triple loop and relies entirely on LLVM auto-vectorization. Observed: `matmul` compiled and LLVM vectorized the loop (`vfmul.vv, vfmv, vfmv.f, vfredosum, vle32.v, vmv…`), but the structure is a scalar triple loop tuned for short SIMD, not a long-vector machine.

2. **LLVM will not maximize vector length / LMUL.** Default LMUL is conservative; dynamic LMUL selection is unfinished upstream. LLVM vectorizes the loop *as written* (note the `vsetvli` reconfiguring VL per loop rather than a fixed long-vector schedule) and does not restructure for long vectors.

3. **No layout choice (Option A vs B).** LLVM takes the loop nest as given; it does not choose the lane axis or reduction strategy. The long-vector-optimal layout is a future tile-backend job.

4. **No AraXL cost/scheduling model.** The `.s` uses generic RVV scheduling/LMUL; AraXL is not a modeled LLVM subtarget.

5. **Unit-stride only.** AraXL cannot do strided/gather loads. Observed: strided/indexed ops emitted in the artifacts: `vlse32` — these (e.g. `vlse32`) would not run on AraXL and flag layouts that need contiguity fixes in a later stage.

6. **No hardware transcendental.** `exp2` (softmax) is not a vector instruction; the baseline would scalarize it or call libm. The polynomial-via-`call_extern` fix is future work.

7. **fp16 not in baseline.** `+f,+d` cover fp32/fp64; fp16 needs `zvfh`. All kernels here are fp32. (The attention kernels' fp16 tensor-core path is GPU-only.)

8. **VRF capacity / spills.** No cache sits between the VRF and L2, so tiles exceeding the VRF spill to memory. Stage 1 does no VRF-aware tiling; inspect each `.s` for stack spills around the vector loops.


**Attention (decode/prefill) status.** Path B TileLang backend work unblocked the real FlashAttention kernels on CPU/`llvm`: fragment `infer_layout`, CPU `T.reduce_*`/`T.fill`, early shared→local promotion in `LowerTileOp`, and `tl.infinity` lowering for LLVM. Both kernels now emit `.tir`/`.ll`/`.s` with RVV ops (`vfmacc`, `vfredosum`, `vsetvli`, …). Numeric validation passes on the host (PyTorch golden vs CPU TileLang). Run `demos/run_attention_rvv_stage.py` for a standalone golden check (mirrors `run_attention_stage.py` on CPU).

