/* Vendored from AraXL/asm/harness_report.c — source of truth for TVM Spike harness glue. */
#include "harness_report.h"

extern int printf(const char *fmt, ...);

static unsigned float_to_u6(float x) {
  if (x < 0.0f) {
    x = -x;
  }
  return (unsigned)(x * 1000000.0f);
}

static unsigned float_to_bits(float x) {
  union {
    float f;
    unsigned u;
  } bits;
  bits.f = x;
  return bits.u;
}

HarnessCompareStats harness_compare_float(const float *out, const float *golden, int count, float atol) {
  HarnessCompareStats stats;
  stats.max_abs_diff = 0.0f;
  stats.mismatches = 0;

  for (int i = 0; i < count; ++i) {
    float diff = out[i] - golden[i];
    if (diff < 0.0f) {
      diff = -diff;
    }
    if (diff > stats.max_abs_diff) {
      stats.max_abs_diff = diff;
    }
    if (diff > atol) {
      stats.mismatches++;
    }
  }

  return stats;
}

int harness_finish(const char *app, int kernel_status, HarnessCompareStats stats, int count, float atol) {
  if (kernel_status != 0) {
    printf(
        "HARNESS_SUMMARY app=%s kernel_status=%d max_abs_diff_bits=0x00000000 max_abs_diff_u6=0 "
        "atol_bits=0x%08x atol_u6=%u elements=%d mismatches=-1 spike_result=kernel_error\n",
        app, kernel_status, float_to_bits(atol), float_to_u6(atol), count);
    return 10 + kernel_status;
  }

  printf(
      "HARNESS_SUMMARY app=%s kernel_status=0 max_abs_diff_bits=0x%08x max_abs_diff_u6=%u "
      "atol_bits=0x%08x atol_u6=%u elements=%d mismatches=%d spike_result=%s\n",
      app, float_to_bits(stats.max_abs_diff), float_to_u6(stats.max_abs_diff),
      float_to_bits(atol), float_to_u6(atol), count, stats.mismatches,
      (stats.mismatches == 0) ? "pass" : "golden_mismatch");

  return (stats.mismatches == 0) ? 0 : 1;
}

static void harness_dump_output_hex(const HarnessOutputSpec *output) {
  const unsigned char *data = output->data;
  int nbytes = output->nbytes;
  const int chunk = 256;
  printf(
      "OUTPUT_BEGIN name=%s dtype=%s nbytes=%d\n",
      output->name,
      output->dtype_tag,
      nbytes);
  for (int off = 0; off < nbytes; off += chunk) {
    int n = nbytes - off;
    if (n > chunk) {
      n = chunk;
    }
    printf("OUTPUT_CHUNK offset=%d nbytes=%d hex=", off, n);
    for (int i = 0; i < n; ++i) {
      printf("%02x", data[off + i]);
    }
    printf("\n");
  }
  printf("OUTPUT_END name=%s\n", output->name);
}

int harness_finish_execute(
    const char *app, int kernel_status, const HarnessOutputSpec *outputs, int num_outputs) {
  if (kernel_status != 0) {
    printf(
        "HARNESS_EXECUTE_SUMMARY app=%s kernel_status=%d spike_result=kernel_error "
        "num_outputs=%d\n",
        app,
        kernel_status,
        num_outputs);
    return 10 + kernel_status;
  }

  printf(
      "HARNESS_EXECUTE_SUMMARY app=%s kernel_status=0 spike_result=pass num_outputs=%d\n",
      app,
      num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    harness_dump_output_hex(&outputs[i]);
  }
  return 0;
}
