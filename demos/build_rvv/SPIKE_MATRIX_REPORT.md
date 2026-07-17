# Spike RVV matrix report

_Generated: 2026-07-15T23:27:30.164481+00:00_

## Summary

| Metric | Value |
|--------|-------|
| nr_lanes | 4 |
| VLEN bits | 4096 |
| Cases | 1 |
| Passed | 1/1 |
| All pass | True |
| max(spike_max_abs_diff) | 1.144409e-05 |
| mean(spike_max_abs_diff) | 1.144409e-05 |
| Wall time | 4.5 s |

## Coverage map

| phase | scenario | cases | pass |
|-------|----------|-------|------|
| linear | `[1, 256, 128]` | 1 | 1/1 |

## Worst passed case (largest spike_max_abs_diff)

- **case_id** `linear_m1_n256_k128_nl4`
- **spike_max_abs_diff** 1.144409e-05
- **host_max_abs_diff** None

## Full table

| case_id | phase | H/KV/D | lengths | spike_result | spike_max_abs_diff | host_max_abs_diff | pass |
|---------|-------|--------|---------|--------------|--------------------|-------------------|------|
| `linear_m1_n256_k128_nl4` | linear | 1/1/64 | `[1, 256, 128]` | pass | 1.144409e-05 | n/a | True |
