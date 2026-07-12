# RVV pipeline — setup and testing

This guide covers how to build the TileLang fork, configure the RVV target
correctly, run the lowering ladder, and validate results.

For architecture and design rationale, see `background.md`. For what to optimize
next, see `implementation_plan.md`.

---

## Prerequisites

| Component | Location / version |
|-----------|-------------------|
| nano-vllm repo | This tree |
| TileLang fork (Path B) | `/mnt/ssd/jby123/tilelang` |
| Python env | `nanovllm` conda env recommended |
| LLVM | Bundled with TVM in the TileLang build (LLVM 18 in current setup) |

Path B patches live in the TileLang repo, not in nano-vllm. nano-vllm supplies
the harness (`nanovllm/backends/tilelang/rvv_lower.py`) and demo scripts.

---

## 1. Build TileLang

```bash
cd /mnt/ssd/jby123/tilelang/build
cmake .. && cmake --build . -j$(nproc)
```

If `reduce.cc` is missing from the ninja graph after a fresh cmake, add it
manually to `build.ninja` or touch `src/cpu/op/reduce.cc` and rebuild.

---

## 2. Environment

```bash
conda activate nanovllm
export PYTHONPATH=/mnt/ssd/jby123/tilelang:/mnt/ssd/jby123/tilelang/build/python:$PYTHONPATH
export LD_LIBRARY_PATH=/mnt/ssd/jby123/miniconda3/envs/nanovllm/lib:$LD_LIBRARY_PATH
cd /mnt/ssd/jby123/nano-vllm
```

`LD_LIBRARY_PATH` is required on some hosts so TVM can load `libLLVM.so` from the
conda env.

Verify TileLang loads from the dev build:

```
Loading tilelang libs from dev root: /mnt/ssd/jby123/tilelang/build
```

---

## 3. RVV target configuration

### VLEN model

AraXL uses **VLEN = 1024 bits × nr_lanes per cluster**. Logical RVV vectors can
span multiple clusters; TileLang's `vector-width` and `+zvl*b` model **per-cluster**
VLEN for the compiler.

| `nr_lanes` | `+zvl*b` | `vector-width` | fp32 elements per cluster (LMUL=1) |
|------------|----------|----------------|-------------------------------------|
| 2 | `+zvl2048b` | 2048 | 64 |
| 4 | `+zvl4096b` | 4096 | 128 |
| 8 | `+zvl8192b` | 8192 | 256 |
| 16 | `+zvl16384b` | 16384 | 512 |

### `nr_lanes` is the single source of VLEN

`rvv_target(nr_lanes=N)` sets `vector-width = 1024 * N`, **strips any `+zvl*`
already in `mattr`, and injects `+zvl{1024*N}b`**. So `--mattr` carries only the
base ISA (`+v,+m,+f,+d`) — you cannot (and should not) hand-tune `+zvl`; change
VLEN with `--nr-lanes`. `DEFAULT_MATTR` no longer embeds any `+zvl`, so the old
"split target" footgun (stale `+zvl4096b` outranking `--nr-lanes 8`) is gone.

The harness also **asserts** `llvm_get_vector_width() == 1024 * nr_lanes` at
startup (`assert_target_vlen`) and prints the effective VLEN, so a mismatch fails
loudly instead of silently mis-measuring. Verify manually if you like:

```python
from nanovllm.backends.tilelang.rvv_lower import rvv_target, assert_target_vlen

t = rvv_target(nr_lanes=8)          # mattr defaults to base ISA
print(assert_target_vlen(t, 8))     # prints 8192, raises on mismatch
```

### Recommended invocations

**Default AraXL 4-lane (VLEN 4096, decode `block_N`=128):**

```bash
python demos/run_rvv_lower.py --nr-lanes 4
```

**8-lane (VLEN 8192, decode `block_N`=256):**

```bash
python demos/run_rvv_lower.py --nr-lanes 8
```

**Qwen3-ish attention shapes:**

```bash
python demos/run_rvv_lower.py \
  --num-heads 16 --num-kv-heads 8 --head-dim 128 --seqlen-kv 128 \
  --nr-lanes 4
```

The decode ladder now derives `block_N` from VLEN (`vlen_f32_elements(nr_lanes)`),
`block_H` from the GQA group size, and pads `seqlen_kv` up to a multiple of
`block_N` automatically, so `seqlen_kv % block_N == 0` is handled for you.

---

## 4. Stage 1 lowering ladder

```bash
python demos/run_rvv_lower.py [options]
```

Kernel ladder (simple → complex):

| Step | Kernel | What it exercises |
|------|--------|-------------------|
| (a) | `elementwise` | Unit-stride `vle` / `vfadd` / `vse` |
| (b) | `matmul` | `GemmVector` → `vfmacc` + wide `T.copy` |
| (c) | `attention_decode` | Full FlashAttention decode, fp32 |
| (d) | `attention_prefill` | Full FlashAttention prefill, fp32 |

### Artifacts

Written under `demos/build_rvv/<kernel>/`:

| File | Contents |
|------|----------|
| `<kernel>.source.tir` | Pristine `@T.prim_func` source |
| `<kernel>.tir` | Lowered CPU/LLVM TensorIR |
| `<kernel>.ll` | LLVM IR for the RVV target |
| `<kernel>.s` | RISC-V assembly |
| `metadata.json` | Target string, vector-op scan, errors |
| `golden.log` | Numeric check vs PyTorch (when run) |

Report: `demos/build_rvv/STAGE1_REPORT.md` (regenerated each run).

### Useful flags

```bash
python demos/run_rvv_lower.py --no-numeric          # skip golden checks (faster)
python demos/run_rvv_lower.py --out-dir /tmp/rvv    # custom artifact root
python demos/run_rvv_lower.py --elem-n 8192         # longer elementwise vector
```

---

## 5. Numeric validation

The RVV `.s` is cross-compiled for `riscv64-unknown-elf` and is **not executed
on the host**. Correctness is checked by running the **same** `@T.prim_func` on
host `llvm` (same CPU tile-op path) and comparing to PyTorch.

### Ladder (built into `run_rvv_lower.py`)

- **elementwise / matmul** — host `c` + cython backend
- **attention decode / prefill** — host `llvm` + `tvm_ffi`, `tilelang_backend=cpu`

### Standalone attention golden

```bash
python demos/run_attention_rvv_stage.py --seq-lens 64
python demos/run_attention_rvv_stage.py --decode --context-lens 128
```

### Pytest matrix

```bash
pytest tests/test_attention_rvv_numeric.py -v
```

Tests skip cleanly if TileLang/LLVM is not on `PYTHONPATH`.

---

## 6. Inspecting vectorization

### Quick scan

```bash
grep -oE 'vsetvli|vle32|vfmacc|vfredmax|vfredusum|vluxei|vfcvt|vsll' \
  demos/build_rvv/attention_decode/attention_decode.s | sort | uniq -c
grep -c 'exp2f' demos/build_rvv/attention_decode/attention_decode.s   # expect 0
```

### What “good” looks like (attention decode)

- **GEMM:** `vfmacc.vv` in hot loops
- **Score `Q @ K^T`:** `vfmacc` + a `vluxei64` gather on the K operand — this is
  the **desired** fully-vectorized path (the target supports strided/indexed loads)
- **Reduce:** one `vle32` + one `vfredmax` / `vfredusum` per score row (VL = `block_N`)
- **Softmax `exp2` (Phase 2):** **no `call exp2f`** — the exponent is an inline
  `2**x` polynomial emitting vector `vfcvt` / `vfmadd` / `vsll.vi`. A `call exp2f`
  reappearing means the vector path (`vec_exp2`/`backend=="cpu"`) is not active.
- **Elementwise fragment loops (Phase 4):** the softmax exponent, `acc_o` rescale,
  final divide, and prefill's fp32 causal mask are vectorized by the CPU pass
  `SerializeOuterParallel` (`tilelang/transform/serialize_outer_parallel.py`,
  wired into `tilelang/cpu/pipeline.py`). It serializes outer parallel axes and
  hoists loop-invariant loads so the innermost `T.Parallel` axis becomes the
  vector lane. Kernels stay in the plain 2-D `T.Parallel(M, N)` form — no
  CPU-specific loop reshaping in the kernel source.
- **Known scalar residual:** the **decode** mask (`uint8` compare + `fp32`
  select) stays scalar — LLVM does not auto-vectorize that mixed-element-width
  masked-select. This is expected, not a regression.

### What to flag

| Pattern | Meaning |
|---------|---------|
| **Scalar** `flw`/`fsw` loops where a vector op would do | Lost vectorization — the thing to fix (except the documented decode `uint8` mask) |
| `call exp2f` in the softmax loop | Vector `exp2` polynomial not applied (should be 0) |
| Scalar exponent/rescale/divide loops | `SerializeOuterParallel` not applied — check the CPU pipeline registration |
| `li reg, 64` before `vsetvli` when VLEN supports 128+ | Tile size, not hardware, is the cap |
| `+zvl` not matching `1024*nr_lanes` | Can't happen via the harness (asserted); indicates a hand-built target |

> Note: `vluxei`/`vlse` strided/indexed loads are **not** a problem — the target
> supports them, and they keep loops vectorized. Prioritize vectorization.

### LLVM IR spot-check

```bash
grep -E 'vfred|vle\.|nxv[0-9]+f32|i64 128|i64 256' \
  demos/build_rvv/attention_decode/attention_decode.ll | head
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Loading tilelang libs from dev root` missing | Set `PYTHONPATH` to TileLang build |
| `GLIBCXX_3.4.30 not found` | Set `LD_LIBRARY_PATH` to conda `lib` |
| Attention compile fails on `tirx.undef` | Rebuild TileLang; reduce path uses `Broadcast(0)` passthrough |
| `seqlen_kv must be multiple of block_N` | Pad cache length or change `block_N` in kernel builder |
| Wider VLEN shows no asm change | Increase tile sizes (`block_N`, `head_dim`) toward `VLEN/32` f32 elements |
| Lowering fails | Read `metadata.json` `error` / `error_pass`; check `*.tir` annotation at top |

---

## 8. Related scripts

| Script | Purpose |
|--------|---------|
| `demos/run_rvv_lower.py` | Full Stage 1 ladder + report |
| `demos/run_attention_rvv_stage.py` | Standalone CPU attention golden |
| `demos/run_attention_stage.py` | CUDA attention reference |
| `nanovllm/backends/tilelang/rvv_lower.py` | Target helper + `lower_kernel_rvv()` |
