# Spike RVV matrix report

_Generated: 2026-07-13T01:20:33.722768+00:00_

## Summary

| Metric | Value |
|--------|-------|
| nr_lanes | 4 |
| VLEN bits | 4096 |
| Cases | 1 |
| Passed | 1/1 |
| All pass | True |
| max(spike_max_abs_diff) | 0.000000e+00 |
| mean(spike_max_abs_diff) | 0.000000e+00 |
| Wall time | 0.9 s |

## Coverage map

| phase | scenario | cases | pass |
|-------|----------|-------|------|
| embedding | `[8, 64, 128]` | 1 | 1/1 |

## Worst passed case (largest spike_max_abs_diff)

- **case_id** `embedding_len8-64-128_nl4`
- **spike_max_abs_diff** 0.000000e+00
- **host_max_abs_diff** 0.0

## Full table

| case_id | phase | H/KV/D | lengths | spike_result | spike_max_abs_diff | host_max_abs_diff | pass |
|---------|-------|--------|---------|--------------|--------------------|-------------------|------|
| `embedding_len8-64-128_nl4` | embedding | 1/1/64 | `[8, 64, 128]` | pass | 0.000000e+00 | 0.000000e+00 | True |
