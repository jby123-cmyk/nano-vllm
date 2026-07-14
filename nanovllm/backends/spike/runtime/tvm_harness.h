/* Vendored from AraXL/asm/tvm_harness.h — source of truth for TVM Spike harness glue. */
#ifndef ARAXL_ASM_TVM_HARNESS_H_
#define ARAXL_ASM_TVM_HARNESS_H_

#include <stdint.h>

#define TVM_ARG_HANDLE 7
#define TVM_ARG_INT64 2
#define TVM_DEVICE_CPU 1

typedef union {
  void *v_handle;
  int64_t v_int64;
  double v_float64;
} TVMValue;

typedef struct {
  int32_t type_index;
  int32_t padding;
  TVMValue value;
} TVMArg;

typedef struct {
  int32_t device_type;
  int32_t device_id;
} DLDevice;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
} DLTensor;

// TVM-generated code may call this on argument validation failures.
static inline void TVMFFIErrorSetRaisedFromCStrParts(const char *kind, const char **parts, int num_parts) {
  (void)kind;
  (void)parts;
  (void)num_parts;
}

static inline DLTensor make_tensor(float *data, int64_t *shape, int32_t ndim) {
  DLTensor tensor;
  tensor.data = data;
  tensor.device.device_type = TVM_DEVICE_CPU;
  tensor.device.device_id = 0;
  tensor.ndim = ndim;
  tensor.dtype.code = 2;   // float
  tensor.dtype.bits = 32;  // float32
  tensor.dtype.lanes = 1;
  tensor.shape = shape;
  tensor.strides = 0;      // packed contiguous
  tensor.byte_offset = 0;
  return tensor;
}

static inline DLTensor make_int32_tensor(int32_t *data, int64_t *shape, int32_t ndim) {
  DLTensor tensor;
  tensor.data = data;
  tensor.device.device_type = TVM_DEVICE_CPU;
  tensor.device.device_id = 0;
  tensor.ndim = ndim;
  tensor.dtype.code = 0;   // int
  tensor.dtype.bits = 32;
  tensor.dtype.lanes = 1;
  tensor.shape = shape;
  tensor.strides = 0;
  tensor.byte_offset = 0;
  return tensor;
}

static inline DLTensor make_uint8_tensor(uint8_t *data, int64_t *shape, int32_t ndim) {
  DLTensor tensor;
  tensor.data = data;
  tensor.device.device_type = TVM_DEVICE_CPU;
  tensor.device.device_id = 0;
  tensor.ndim = ndim;
  tensor.dtype.code = 1;   // uint
  tensor.dtype.bits = 8;
  tensor.dtype.lanes = 1;
  tensor.shape = shape;
  tensor.strides = 0;
  tensor.byte_offset = 0;
  return tensor;
}

#endif
