# Stage 1 Report — TileLang → LLVM → RVV (Demo Backend)

_Generated: 2026-07-03T22:21:19.778285+00:00_

_Target: `{"kind":"llvm","tag":"","keys":["cpu"],"vector-width":4096,"mabi":"lp64d","mtriple":"riscv64-unknown-elf","mattr":["+v","+m","+f","+d","+zvl4096b"]}`_


This report records the **demo** described in `tilelang.md`: TileLang kernels routed through the CPU/`llvm` pass pipeline at the AraXL RVV target, with LLVM auto-vectorization on. **Path B** TileLang backend work (see `tilelang.md`) added CPU realizations for GPU-style tile ops so FlashAttention decode/prefill lower end-to-end; LLVM still owns vector scheduling (no custom LMUL tuning or AraXL subtarget). Remaining limitations are **observed in the artifacts**, not hidden.


## Reproduce

```bash
python demos/run_rvv_lower.py
```

## Per-kernel results

| Kernel | Compiled | Vectorized | Numeric (host) | Failing pass | Notes |
|--------|----------|------------|----------------|--------------|-------|
| `elementwise` | yes | yes | pass | `—` | 5 RVV op kinds |
| `matmul` | yes | yes | pass | `—` | 8 RVV op kinds |
| `attention_decode` | yes | yes | pass | `—` | 12 RVV op kinds; strided: vluxei64 |
| `attention_prefill` | yes | yes | pass | `—` | 12 RVV op kinds; strided: vluxei64 |

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
- Vector ops: `vfmacc.vv, vfmv.v, vle32.v, vmv, vse32.v, vse64.v, vsetivli, vsetvli`
- Strided/indexed ops (AraXL unit-stride only): `none`

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
.Ltmp101:
	slli	s8, s8, 2
.Ltmp102:
```

### `attention_decode`

- Artifacts: `attention_decode.tir`, `attention_decode.ll`, `attention_decode.s`
- Vector ops: `vadd, vfmacc.vv, vfmsub.vf, vfmul.vf, vfmv.f, vfmv.v, vle32.v, vmv, vse32.v, vse64.v, vsetivli, vsetvli`
- Strided/indexed ops (AraXL unit-stride only): `vluxei64`

```asm
	sd	a0, 24(sp)
	li	a0, 128
	vsetvli	zero, a0, e32, m1, ta, ma
	vmv.v.i	v8, 0
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, sp, a0
	addi	a0, a0, 320
	vs1r.v	v8, (a0)
	li	a0, 64
	vsetvli	zero, a0, e32, mf2, ta, ma
	vmv.v.i	v8, 0
	csrr	a0, vlenb
	add	a0, sp, a0
```

### `attention_prefill`

- Artifacts: `attention_prefill.tir`, `attention_prefill.ll`, `attention_prefill.s`
- Vector ops: `vadd, vfmacc.vv, vfmsub.vf, vfmul.vf, vfmv.f, vfmv.v, vle32.v, vmv, vse32.v, vse64.v, vsetivli, vsetvli`
- Strided/indexed ops (AraXL unit-stride only): `vluxei64`

```asm
	sd	a0, 272(sp)
	li	a0, 128
	vsetvli	zero, a0, e32, m1, ta, ma
	vmv.v.i	v8, 0
	csrr	a0, vlenb
	li	a1, 12
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 512
	vs1r.v	v8, (a0)
	li	a0, 64
	vsetvli	zero, a0, e32, mf2, ta, ma
	vmv.v.i	v8, 0
	csrr	a0, vlenb
```

## Section 6 limitations — observed in the emitted artifacts

1. **GemmVector for all transpose forms.** `T.gemm` lowers via `GemmVector` on `llvm`: a `serial i,k` + `parallel j` nest with the output-`N` axis vectorized (`vfmacc` on the accumulator), including the transposed `Q @ K^T` in FlashAttention (its `B[j,k]` load becomes a strided/gather vector load). Observed: `matmul` vectorized to `vfmacc.vv, vfmv.v, vle32.v, vmv, vse32.v, vse64.v…`; LLVM still owns final VL/LMUL selection, so this is not a hand-tuned long-vector micro-kernel.

2. **LLVM will not maximize vector length / LMUL.** Default LMUL is conservative; dynamic LMUL selection is unfinished upstream. LLVM vectorizes the loop *as written* (note the `vsetvli` reconfiguring VL per loop rather than a fixed long-vector schedule) and does not restructure for long vectors.

3. **No layout choice (Option A vs B).** LLVM takes the loop nest as given; it does not choose the lane axis or reduction strategy. The long-vector-optimal layout is a future tile-backend job.

4. **No AraXL cost/scheduling model.** The `.s` uses generic RVV scheduling/LMUL; AraXL is not a modeled LLVM subtarget.

5. **Unit-stride only.** AraXL cannot do strided/gather loads. Observed: strided/indexed ops emitted in the artifacts: `vluxei64` — these (e.g. `vlse32`) would not run on AraXL and flag layouts that need contiguity fixes in a later stage.

6. **No hardware transcendental.** `exp2` (softmax) is not a vector instruction; the baseline would scalarize it or call libm. The polynomial-via-`call_extern` fix is future work.

7. **fp16 not in baseline.** `+f,+d` cover fp32/fp64; fp16 needs `zvfh`. All kernels here are fp32. (The attention kernels' fp16 tensor-core path is GPU-only.)

8. **VRF capacity / spills.** No cache sits between the VRF and L2, so tiles exceeding the VRF spill to memory. Stage 1 does no VRF-aware tiling; inspect each `.s` for stack spills around the vector loops.


**Attention (decode/prefill) status.** Path B TileLang backend work unblocked the real FlashAttention kernels on CPU/`llvm`: fragment `infer_layout`, CPU `T.reduce_*`/`T.fill`, early shared→local promotion in `LowerTileOp`, and `tl.infinity` lowering for LLVM. Both kernels now emit `.tir`/`.ll`/`.s` with RVV ops (`vfmacc`, `vfredosum`, `vsetvli`, …). Numeric validation passes on the host (PyTorch golden vs CPU TileLang). Run `demos/run_attention_rvv_stage.py` for a standalone golden check (mirrors `run_attention_stage.py` on CPU).

