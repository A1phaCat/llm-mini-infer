// The swapab64 configuration of the block-FP8 GEMM (fp8_gemm_sm120.cuh), in its own translation unit.
#include "fp8_gemm_sm120.cuh"

namespace lmk_fp8 {

int run_c3(const void* a, const void* b, const void* sa, const void* sb, void* d, int m, int n, int k, int swizzle,
           int sm_count, int device, void* workspace, int64_t workspace_bytes, bool pdl, cudaStream_t s) {
  return run<config_swapab64<cutlass::bfloat16_t>::Gemm>(a, b, sa, sb, d, m, n, k, swizzle, sm_count, device,
      workspace, workspace_bytes, pdl, s);
}

int64_t workspace_c3(int m, int n, int k, int swizzle, int sm_count, int device) {
  return workspace_bytes<config_swapab64<cutlass::bfloat16_t>::Gemm>(m, n, k, swizzle, sm_count, device);
}

}  // namespace lmk_fp8
