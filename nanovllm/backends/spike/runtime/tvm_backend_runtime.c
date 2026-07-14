/* Vendored from AraXL/asm/tvm_backend_runtime.c — source of truth for TVM Spike harness glue. */
#include <stdint.h>

typedef int (*TVMParallelLambda)(int task_id, void *penv, void *cdata);

/* Provide the TVM backend hooks the kernel .ll references. */
void *(*__TVMBackendAllocWorkspace)(int, int, uint64_t, int, int) = 0;
int (*__TVMBackendFreeWorkspace)(int, int, void *) = 0;
int (*__TVMBackendParallelLaunch)(TVMParallelLambda flambda, void *cdata, int num_task) = 0;

static unsigned char tvm_ws[16 * 1024 * 1024] __attribute__((aligned(64), section(".l2")));
static uint64_t tvm_ws_top = 0;

void *tvm_alloc(int device_type, int device_id, uint64_t nbytes, int dtype_code, int dtype_bits) {
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

int tvm_free(int device_type, int device_id, void *ptr) {
  (void)device_type;
  (void)device_id;
  (void)ptr;
  return 0;
}

typedef struct {
  void *sync_handle;
  int num_task;
} TVMParallelGroupEnv;

int tvm_parallel_launch(TVMParallelLambda flambda, void *cdata, int num_task) {
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

void tvm_backend_runtime_init(void) {
  __TVMBackendAllocWorkspace = tvm_alloc;
  __TVMBackendFreeWorkspace = tvm_free;
  __TVMBackendParallelLaunch = tvm_parallel_launch;
}
