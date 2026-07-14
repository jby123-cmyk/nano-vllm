"""
Run the full RVV attention matrix on Spike (ISA-sim).

For each unique RVV JIT signature:
  emit kernel.ll → embed PyTorch golden → generate harness → build ELF → spike run

Usage (nanovllm env, see usage.md)::

    python demos/rvv/spike_matrix.py --limit 2
    python demos/rvv/spike_matrix.py --nr-lanes 4
    python demos/rvv/spike_matrix.py --only decode_h8_kv2_d64_len64_nl4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure repo root is importable when run as a script.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nanovllm.backends.spike.cases import HEAD_CONFIGS, enumerate_cases  # noqa: E402
from nanovllm.backends.spike.config import default_config  # noqa: E402
from nanovllm.backends.spike.pipeline import CaseResult, run_case  # noqa: E402
from nanovllm.backends.spike.report import write_reports  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nr-lanes", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        default="demos/build_rvv",
        help="Report root; per-case artifacts under <out-dir>/spike/<case_id>/",
    )
    parser.add_argument(
        "--spike-root",
        default="",
        help="Override per-case artifact root (default <out-dir>/spike).",
    )
    parser.add_argument("--only", default="", help="Run a single case_id.")
    parser.add_argument("--limit", type=int, default=0, help="Run at most N cases.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only the wide representative head config (16/8/128).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel case workers (Spike runs included; keep small).",
    )
    parser.add_argument(
        "--host-sanity",
        action="store_true",
        help="Also compile/run host llvm golden before Spike (slower).",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after failures (default: continue anyway; exit code reflects fails).",
    )
    parser.add_argument(
        "--decode-only", action="store_true", help="Skip prefill cases."
    )
    parser.add_argument(
        "--prefill-only", action="store_true", help="Skip decode cases."
    )
    parser.add_argument(
        "--attention-only",
        action="store_true",
        help="Skip linear GEMM cases (decode/prefill only).",
    )
    parser.add_argument(
        "--linear-only",
        action="store_true",
        help="Only linear GEMM cases (skip attention).",
    )
    args = parser.parse_args(argv)

    cfg = default_config(nr_lanes=args.nr_lanes)
    try:
        cfg.assert_available()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    spike_root = args.spike_root or os.path.join(args.out_dir, "spike")
    head_configs = [(16, 8, 128)] if args.quick else None
    include_attn = not args.linear_only
    cases = enumerate_cases(
        args.nr_lanes,
        head_configs=head_configs,
        include_decode=include_attn and not args.prefill_only,
        include_decode_paged=include_attn and not args.prefill_only,
        include_prefill=include_attn and not args.decode_only,
        include_prefill_paged=include_attn and not args.decode_only,
        include_linear=not args.attention_only,
        include_layer_ops=not args.attention_only,
    )
    if args.only:
        cases = [c for c in cases if c.case_id == args.only]
        if not cases:
            print(f"No case matching --only {args.only}", file=sys.stderr)
            return 2
    if args.limit > 0:
        cases = cases[: args.limit]

    print(
        f"Spike matrix: {len(cases)} cases, nr_lanes={args.nr_lanes}, "
        f"workers={args.workers}, spike_root={spike_root}"
    )
    t0 = time.perf_counter()
    rows: list[CaseResult] = []

    def _one(case):
        return run_case(
            case.phase,
            list(case.lengths),
            num_heads=case.num_heads,
            num_kv_heads=case.num_kv_heads,
            head_dim=case.head_dim,
            nr_lanes=case.nr_lanes,
            out_root=spike_root,
            cfg=cfg,
            skip_host_sanity=not args.host_sanity,
        )

    if args.workers <= 1:
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case.case_id} ...", flush=True)
            result = _one(case)
            rows.append(result)
            print(
                f"  -> {result.spike_result} max_abs_diff={result.spike_max_abs_diff:.6e} "
                f"build={result.build_ms:.0f}ms run={result.run_ms:.0f}ms",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_one, c): c for c in cases}
            done = 0
            for fut in as_completed(futs):
                done += 1
                case = futs[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    result = CaseResult(
                        case_id=case.case_id,
                        phase=case.phase,
                        num_heads=case.num_heads,
                        num_kv_heads=case.num_kv_heads,
                        head_dim=case.head_dim,
                        lengths=str(list(case.lengths)),
                        nr_lanes=case.nr_lanes,
                        compiled=False,
                        host_max_abs_diff=None,
                        spike_result="exception",
                        spike_max_abs_diff=float("nan"),
                        mismatches=-1,
                        build_ms=0.0,
                        run_ms=0.0,
                        passed=False,
                        error=str(exc),
                    )
                rows.append(result)
                print(
                    f"[{done}/{len(cases)}] {result.case_id} -> {result.spike_result} "
                    f"max_abs_diff={result.spike_max_abs_diff:.6e}",
                    flush=True,
                )

    # Stable order for the report.
    order = {c.case_id: i for i, c in enumerate(cases)}
    rows.sort(key=lambda r: order.get(r.case_id, 10**9))

    wall_s = time.perf_counter() - t0
    csv_path, md_path = write_reports(
        args.out_dir, rows, nr_lanes=args.nr_lanes, wall_s=wall_s
    )
    n_pass = sum(1 for r in rows if r.passed)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Spike matrix: {n_pass}/{len(rows)} pass  wall={wall_s:.1f}s")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    # Silence unused import warning for HEAD_CONFIGS in --help context.
    _ = HEAD_CONFIGS
    sys.exit(main())
