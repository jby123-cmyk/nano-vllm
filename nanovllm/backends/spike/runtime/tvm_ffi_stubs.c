/* Vendored from AraXL/asm/tvm_ffi_stubs.c — source of truth for TVM Spike harness glue. */
// TVM-generated kernels may reference this symbol for error propagation.
void TVMFFIErrorSetRaisedFromCStrParts(const char *kind, const char **parts, int num_parts) {
  extern int printf(const char *fmt, ...);
  printf("TVM error kind: %s\n", kind ? kind : "<null>");
  for (int i = 0; i < num_parts; ++i) {
    const char *p = (parts && parts[i]) ? parts[i] : "<null>";
    printf("TVM error part[%d]: %s\n", i, p);
  }
}
