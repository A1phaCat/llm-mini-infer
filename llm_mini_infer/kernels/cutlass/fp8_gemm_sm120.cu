// The extern "C" launchers of the block-FP8 GEMM (fp8_gemm_sm120.cuh). Each configuration has its own translation
// unit (fp8_gemm_sm120_c{0,1,2,3}.cu).
#include "fp8_gemm_sm120.cuh"

// config: 0 swap-AB 128x32x128, 1 ping-pong 64x128x128, 2 cooperative 128x128x128, 3 swap-AB 128x64x128 (SGLang's).
// pdl != 0: launch with programmatic dependent launch. Returns 0, a cudaError_t, or -1 (shape not implementable),
// -2 (workspace too small), -3 (CUTLASS failed to initialise or launch), -4 (unknown config).
LMK_API int lmk_fp8_gemm_sm120(const void* a, const void* b, const void* sa, const void* sb, void* d, int m, int n,
                               int k, int config, int swizzle, int sm_count, int device, void* workspace,
                               int64_t workspace_bytes, int pdl, cudaStream_t stream) {
  if (m <= 0) return 0;
  switch (config) {
    case 0:
      return lmk_fp8::run_c0(a, b, sa, sb, d, m, n, k, swizzle, sm_count, device, workspace, workspace_bytes,
                               pdl != 0, stream);
    case 1:
      return lmk_fp8::run_c1(a, b, sa, sb, d, m, n, k, swizzle, sm_count, device, workspace, workspace_bytes,
                               pdl != 0, stream);
    case 2:
      return lmk_fp8::run_c2(a, b, sa, sb, d, m, n, k, swizzle, sm_count, device, workspace, workspace_bytes,
                               pdl != 0, stream);
    case 3:
      return lmk_fp8::run_c3(a, b, sa, sb, d, m, n, k, swizzle, sm_count, device, workspace, workspace_bytes,
                               pdl != 0, stream);
    default:
      return -4;
  }
}

// The workspace size a call needs (in bytes), or -4 for an unknown config.
LMK_API int64_t lmk_fp8_gemm_sm120_workspace(int m, int n, int k, int config, int swizzle, int sm_count, int device) {
  switch (config) {
    case 0:
      return lmk_fp8::workspace_c0(m, n, k, swizzle, sm_count, device);
    case 1:
      return lmk_fp8::workspace_c1(m, n, k, swizzle, sm_count, device);
    case 2:
      return lmk_fp8::workspace_c2(m, n, k, swizzle, sm_count, device);
    case 3:
      return lmk_fp8::workspace_c3(m, n, k, swizzle, sm_count, device);
    default:
      return -4;
  }
}
