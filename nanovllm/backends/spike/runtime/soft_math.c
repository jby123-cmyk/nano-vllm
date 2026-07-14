/* Vendored from AraXL/asm/soft_math.c — source of truth for TVM Spike harness glue. */
#include <stdint.h>

// Minimal exp2f for Spike ELFs: avoids linking newlib libm (medany relocation issues).
float exp2f(float x) {
  if (x <= -126.0f) {
    return 0.0f;
  }
  if (x >= 128.0f) {
    return 3.402823466e38f;
  }

  int ix = (int)x;
  float fx = x - (float)ix;
  float y = 1.0f + fx * (0.69314718056f + fx * 0.24022650695f);

  union {
    float f;
    uint32_t u;
  } scale;
  scale.u = (uint32_t)((ix + 127) << 23);
  return y * scale.f;
}

// expf via exp2f (log2(e) scale) — needed by SiluAndMul / sigmoid lowering.
float expf(float x) {
  return exp2f(x * 1.4426950408889634f);
}
