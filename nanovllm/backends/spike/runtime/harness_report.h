/* Vendored from AraXL/asm/harness_report.h — source of truth for TVM Spike harness glue. */
#ifndef ARAXL_ASM_HARNESS_REPORT_H_
#define ARAXL_ASM_HARNESS_REPORT_H_

typedef struct HarnessCompareStats {
  float max_abs_diff;
  int mismatches;
} HarnessCompareStats;

HarnessCompareStats harness_compare_float(const float *out, const float *golden, int count, float atol);

// Print HARNESS_SUMMARY line (Spike forwards via HTIF) and return process exit code.
int harness_finish(const char *app, int kernel_status, HarnessCompareStats stats, int count, float atol);

#endif
