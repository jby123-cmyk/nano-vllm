/* Vendored from AraXL/asm/matmul_main.c — source of truth for TVM Spike harness glue. */
#include "tvm_harness.h"
#include "harness_report.h"
#include <math.h>

extern int __tvm_ffi_main(void *self_handle, TVMArg *args, int num_args, void *result);
extern void *(*__TVMBackendAllocWorkspace)(int, int, uint64_t, int, int);
extern int (*__TVMBackendFreeWorkspace)(int, int, void *);
typedef int (*TVMParallelLambda)(int task_id, void *penv, void *cdata);
extern int (*__TVMBackendParallelLaunch)(TVMParallelLambda flambda, void *cdata, int num_task);
extern int printf(const char *fmt, ...);
extern void tohost_exit(uintptr_t code);

#define M 128
#define K 128
#define N 128
#define GOLDEN_ATOL 3e-3f

static float a_data[M][K] __attribute__((aligned(64), section(".l2")));
static float b_data[K][N] __attribute__((aligned(64), section(".l2")));
static float c_data[M][N] __attribute__((aligned(64), section(".l2")));
static float golden_data[M][N] __attribute__((aligned(64), section(".l2")));
static unsigned char tvm_ws[2 * 1024 * 1024] __attribute__((aligned(64), section(".l2")));
static uint64_t tvm_ws_top = 0;

static int64_t shape_a[2] __attribute__((aligned(64), section(".l2"))) = {M, K};
static int64_t shape_b[2] __attribute__((aligned(64), section(".l2"))) = {K, N};
static int64_t shape_c[2] __attribute__((aligned(64), section(".l2"))) = {M, N};
static int64_t strides_a[2] __attribute__((aligned(64), section(".l2"))) = {K, 1};
static int64_t strides_b[2] __attribute__((aligned(64), section(".l2"))) = {N, 1};
static int64_t strides_c[2] __attribute__((aligned(64), section(".l2"))) = {N, 1};

static void *tvm_alloc(int device_type, int device_id, uint64_t nbytes, int dtype_code, int dtype_bits) {
  (void)device_type;
  (void)device_id;
  (void)dtype_code;
  (void)dtype_bits;
  uintptr_t base = (uintptr_t)(tvm_ws + tvm_ws_top);
  uintptr_t aligned = (base + 63u) & ~(uintptr_t)63u;
  uint64_t next = (uint64_t)(aligned - (uintptr_t)tvm_ws) + nbytes;
  if (next > sizeof(tvm_ws)) {
    return 0;
  }
  tvm_ws_top = next;
  return (void *)aligned;
}

static int tvm_free(int device_type, int device_id, void *ptr) {
  (void)device_type;
  (void)device_id;
  (void)ptr;
  return 0;
}

typedef struct {
  void *sync_handle;
  int num_task;
} TVMParallelGroupEnv;

static int tvm_parallel_launch(TVMParallelLambda flambda, void *cdata, int num_task) {
  TVMParallelGroupEnv env;
  env.sync_handle = 0;
  env.num_task = (num_task > 0) ? num_task : 1;
  for (int task_id = 0; task_id < env.num_task; ++task_id) {
    int rc = flambda(task_id, &env, cdata);
    if (rc != 0) {
      return rc;
    }
  }
  return 0;
}

uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  (void)regs;
  printf("matmul trap cause=0x%lx epc=0x%lx\n", (unsigned long)cause, (unsigned long)epc);
  tohost_exit(1337);
  return epc;
}

int main(void) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      a_data[i][j] = ((float)((i * 17 + j * 7) % 97) - 48.0f) * 0.03125f;
    }
  }
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      b_data[i][j] = ((float)((i * 13 + j * 11) % 89) - 44.0f) * 0.046875f;
    }
  }
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      c_data[i][j] = 0.0f;
      golden_data[i][j] = 0.0f;
    }
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += a_data[i][k] * b_data[k][j];
      }
      golden_data[i][j] = acc;
    }
  }

  DLTensor a = make_tensor(&a_data[0][0], shape_a, 2);
  DLTensor b = make_tensor(&b_data[0][0], shape_b, 2);
  DLTensor c = make_tensor(&c_data[0][0], shape_c, 2);
  a.strides = strides_a;
  b.strides = strides_b;
  c.strides = strides_c;

  TVMArg args[3];
  args[0].type_index = TVM_ARG_HANDLE;
  args[0].padding = 0;
  args[0].value.v_handle = &a;
  args[1].type_index = TVM_ARG_HANDLE;
  args[1].padding = 0;
  args[1].value.v_handle = &b;
  args[2].type_index = TVM_ARG_HANDLE;
  args[2].padding = 0;
  args[2].value.v_handle = &c;

  __TVMBackendAllocWorkspace = tvm_alloc;
  __TVMBackendFreeWorkspace = tvm_free;
  __TVMBackendParallelLaunch = tvm_parallel_launch;

  int status = __tvm_ffi_main(0, args, 3, 0);
  HarnessCompareStats stats =
      harness_compare_float(&c_data[0][0], &golden_data[0][0], M * N, GOLDEN_ATOL);
  return harness_finish("matmul", status, stats, M * N, GOLDEN_ATOL);
}
