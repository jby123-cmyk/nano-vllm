"""Aggregate Spike matrix results into CSV + markdown report."""

from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from nanovllm.backends.spike.pipeline import CaseResult


def write_csv(path: str, rows: list[CaseResult]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].as_dict().keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            d = row.as_dict()
            # Make floats readable.
            for k, v in list(d.items()):
                if isinstance(v, float):
                    d[k] = f"{v:.6e}" if k.endswith("diff") else f"{v:.2f}"
            writer.writerow(d)


def write_markdown(path: str, rows: list[CaseResult], *, nr_lanes: int, wall_s: float) -> None:
    passed = [r for r in rows if r.passed]
    failed = [r for r in rows if not r.passed]
    diffs = [r.spike_max_abs_diff for r in rows if r.passed]
    lines = [
        "# Spike RVV matrix report",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| nr_lanes | {nr_lanes} |",
        f"| VLEN bits | {1024 * nr_lanes} |",
        f"| Cases | {len(rows)} |",
        f"| Passed | {len(passed)}/{len(rows)} |",
        f"| All pass | {len(failed) == 0} |",
    ]
    if diffs:
        lines.extend(
            [
                f"| max(spike_max_abs_diff) | {max(diffs):.6e} |",
                f"| mean(spike_max_abs_diff) | {statistics.mean(diffs):.6e} |",
            ]
        )
    lines.extend(
        [
            f"| Wall time | {wall_s:.1f} s |",
            "",
            "## Coverage map",
            "",
            "| phase | scenario | cases | pass |",
            "|-------|----------|-------|------|",
        ]
    )

    by_note: dict[tuple[str, str], list[CaseResult]] = defaultdict(list)
    # coverage_note is not on CaseResult — recover from lengths/phase grouping only.
    for r in rows:
        by_note[(r.phase, r.lengths)].append(r)
    for (phase, lengths), group in sorted(by_note.items()):
        ok = sum(1 for r in group if r.passed)
        lines.append(f"| {phase} | `{lengths}` | {len(group)} | {ok}/{len(group)} |")

    if failed:
        lines.extend(["", "## Failures", ""])
        for r in failed:
            lines.append(
                f"- `{r.case_id}` spike_result=`{r.spike_result}` "
                f"diff={r.spike_max_abs_diff} error={r.error[:120]!r}"
            )

    if passed:
        worst = max(passed, key=lambda r: r.spike_max_abs_diff)
        lines.extend(
            [
                "",
                "## Worst passed case (largest spike_max_abs_diff)",
                "",
                f"- **case_id** `{worst.case_id}`",
                f"- **spike_max_abs_diff** {worst.spike_max_abs_diff:.6e}",
                f"- **host_max_abs_diff** {worst.host_max_abs_diff}",
                "",
            ]
        )

    lines.extend(
        [
            "## Full table",
            "",
            "| case_id | phase | H/KV/D | lengths | spike_result | spike_max_abs_diff | host_max_abs_diff | pass |",
            "|---------|-------|--------|---------|--------------|--------------------|-------------------|------|",
        ]
    )
    for r in rows:
        host = (
            f"{r.host_max_abs_diff:.6e}"
            if isinstance(r.host_max_abs_diff, float)
            else "n/a"
        )
        lines.append(
            f"| `{r.case_id}` | {r.phase} | {r.num_heads}/{r.num_kv_heads}/{r.head_dim} | "
            f"`{r.lengths}` | {r.spike_result} | {r.spike_max_abs_diff:.6e} | {host} | {r.passed} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_reports(
    out_dir: str,
    rows: list[CaseResult],
    *,
    nr_lanes: int,
    wall_s: float,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "spike_matrix.csv")
    md_path = os.path.join(out_dir, "SPIKE_MATRIX_REPORT.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, nr_lanes=nr_lanes, wall_s=wall_s)
    return csv_path, md_path
