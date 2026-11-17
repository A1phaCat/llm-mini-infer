// The GDN prefill inputs in one kernel (models/qwen3_5/backend.py `GDN.prefill_rows` before FlashInfer's chunked
// kernel): q and k L2-normalised in fp32 and rounded to BF16, v copied contiguous, and the gates alpha = exp(g),
// g = -exp(A_log) * softplus(a + dt_bias), and beta = sigmoid(b), in fp32 with torch's formulas and rounding
// (softplus threshold 20). Only the reduction order of the L2 sum differs from torch.
//
// Programmatic dependent launch (common.cuh): the kernel waits for the kernel before it as its first statement. Then
// it lets the next kernel launch. The launcher adds the attribute while the library's flag is on.
#include "common.cuh"

constexpr int kD = 128;  // head dim of q, k and v

// x [T, 2*HK*128 + HV*128] at row stride xs (the conv output); a, b [T, HV] at row strides as, bs; A_log [HV] fp32;
// dt_bias [HV] BF16. Outputs: qn, kn [T, HK, 128] BF16; v [T, HV, 128] BF16; alpha, beta [T, HV] fp32.
__global__ void gdn_prep_kernel(const bf16* __restrict__ x, int64_t xs, const bf16* __restrict__ a, int64_t as,
                                const bf16* __restrict__ b, int64_t bs, const float* __restrict__ a_log,
                                const bf16* __restrict__ dt_bias, bf16* __restrict__ qn, bf16* __restrict__ kn,
                                bf16* __restrict__ v, float* __restrict__ alpha, float* __restrict__ beta, int HK,
                                int HV) {
  lmk_pdl_wait();  // PDL: nothing a predecessor writes is read before this (common.cuh)
  lmk_pdl_trigger();
  const int64_t t = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const bf16* xr = x + t * xs;
  for (int h = warp; h < 2 * HK + HV; h += nw) {
    const bf16* src = xr + (int64_t)h * kD;
    if (h < 2 * HK) {
      float vals[4];
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        vals[i] = bf2f(src[lane + 32 * i]);
        ss = fadd(ss, fmul(vals[i], vals[i]));
      }
      ss = warp_sum(ss);
      const float r = rsqrtf(fadd(ss, 1e-6f));
      bf16* dst = (h < HK ? qn + (t * HK + h) * kD : kn + (t * HK + (h - HK)) * kD);
#pragma unroll
      for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = f2bf(fmul(vals[i], r));
    } else {
      bf16* dst = v + (t * HV + (h - 2 * HK)) * kD;
#pragma unroll
      for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = src[lane + 32 * i];
    }
  }
  for (int hv = threadIdx.x; hv < HV; hv += blockDim.x) {
    const float xa = fadd(bf2f(a[t * as + hv]), bf2f(dt_bias[hv]));
    const float sp = xa > 20.0f ? xa : log1pf(expf(xa));
    const float g = fmul(-expf(a_log[hv]), sp);
    alpha[t * HV + hv] = expf(g);
    beta[t * HV + hv] = t_sigmoid(bf2f(b[t * bs + hv]));
  }
}

LMK_API int lmk_gdn_prep(const void* x, int64_t x_stride, const void* a, int64_t a_stride, const void* b,
                         int64_t b_stride, const void* a_log, const void* dt_bias, void* qn, void* kn, void* v,
                         void* alpha, void* beta, int64_t T, int HK, int HV, cudaStream_t stream) {
  if (T <= 0) return 0;
  return lmk_launch(gdn_prep_kernel, dim3((unsigned)T), dim3(256), 0, stream, (const bf16*)x, x_stride, (const bf16*)a,
                    a_stride, (const bf16*)b, b_stride, (const float*)a_log, (const bf16*)dt_bias, (bf16*)qn,
                    (bf16*)kn, (bf16*)v, (float*)alpha, (float*)beta, HK, HV);
}
