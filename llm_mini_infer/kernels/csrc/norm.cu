// Fused elementwise and norm kernels for Qwen3.5/3.8 in BF16. They round where the torch reference does
// (models/qwen3_5/model.py): all intermediates in fp32, one rounding to BF16 at the end. Only the order of the
// sum of squares differs from torch, which can move a result by one BF16 ulp.
//
// Programmatic dependent launch (PDL, common.cuh): each kernel waits for its predecessor first, then lets the next
// kernel launch. The launchers add the attribute while the library's flag is on.
#include "common.cuh"

// Gemma RMSNorm with an optional fused residual add (RMSNorm.forward / forward_add):
//   s = x (+ residual);  residual_out = bf16(s);  out = bf16((s * rsqrt(sum(s^2) * (1/D) + eps)) * (1 + w)).
// One block per row. x has row stride `xs`; residual and the outputs have row stride D.
__global__ void gemma_rmsnorm_kernel(const bf16* __restrict__ x, int64_t xs, const bf16* __restrict__ res,
                                     const bf16* __restrict__ w, bf16* __restrict__ out, bf16* __restrict__ res_out,
                                     int D, float eps) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  extern __shared__ float row[];  // D floats: s
  __shared__ float red[32];
  const int64_t r = blockIdx.x;
  const bf16* xr = x + r * xs;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float s = bf2f(xr[i]);
    if (res != nullptr) {
      s = fadd(s, bf2f(res[r * D + i]));
      res_out[r * D + i] = f2bf(s);
    }
    row[i] = s;
    ss = fadd(ss, fmul(s, s));
  }
  ss = block_sum(ss, red);
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    out[r * D + i] = f2bf(fmul(fmul(row[i], inv), fadd(1.0f, bf2f(w[i]))));
  }
}

LMK_API int lmk_gemma_rmsnorm(const void* x, int64_t x_stride, const void* residual, const void* weight, void* out,
                              void* residual_out, int64_t rows, int D, float eps, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const int threads = D >= 4096 ? 1024 : D >= 1024 ? 512 : 256;
  return lmk_launch(gemma_rmsnorm_kernel, dim3((unsigned)rows), dim3(threads), D * sizeof(float), stream,
                    (const bf16*)x, x_stride, (const bf16*)residual, (const bf16*)weight, (bf16*)out,
                    (bf16*)residual_out, D, eps);
}

// GDN output norm (GatedRMSNorm), per (token, head) of D values (D = 128 in GDN; a multiple of 32, at most 256):
//   out = bf16(((x * rsqrt(sum(x^2) * (1/D) + eps)) * w) * silu(z)).
// One warp per (token, head). x is [T, H, D] contiguous; z has token stride `zs` and head stride D; out is [T, H, D].
__global__ void gated_rmsnorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ z, int64_t zs,
                                     const bf16* __restrict__ w, bf16* __restrict__ out, int64_t n, int H, int D,
                                     float eps) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  const int64_t item = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (item >= n) return;
  const int64_t t = item / H;
  const int h = (int)(item % H);
  const bf16* xr = x + item * D;
  const bf16* zr = z + t * zs + (int64_t)h * D;
  float v[8];
  float ss = 0.0f;
  const int per = D / 32;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    if (k < per) {
      v[k] = bf2f(xr[lane + 32 * k]);
      ss = fadd(ss, fmul(v[k], v[k]));
    }
  }
  ss = warp_sum(ss);
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    if (k < per) {
      const int i = lane + 32 * k;
      const float y = fmul(fmul(v[k], inv), bf2f(w[i]));
      out[item * D + i] = f2bf(fmul(y, t_silu(bf2f(zr[i]))));
    }
  }
}

LMK_API int lmk_gated_rmsnorm(const void* x, const void* z, int64_t z_stride, const void* weight, void* out,
                              int64_t tokens, int H, int D, float eps, cudaStream_t stream) {
  const int64_t n = tokens * H;
  if (n <= 0) return 0;
  if (D % 32 || D > 256) return -1;
  const int warps = 8;
  return lmk_launch(gated_rmsnorm_kernel, dim3((unsigned)((n + warps - 1) / warps)), dim3(warps * 32), 0, stream,
                    (const bf16*)x, (const bf16*)z, z_stride, (const bf16*)weight, (bf16*)out, n, H, D, eps);
}

// MLP activation. gu is [T, 2I] (gate, then up) at row stride `gs`; out [T, I] = bf16(silu(g) * u).
__global__ void silu_mul_kernel(const bf16* __restrict__ gu, int64_t gs, bf16* __restrict__ out, int64_t T, int I) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * I) return;
  const int64_t t = idx / I;
  const int i = (int)(idx % I);
  const float g = bf2f(gu[t * gs + i]);
  const float u = bf2f(gu[t * gs + I + i]);
  out[idx] = f2bf(fmul(t_silu(g), u));
}

LMK_API int lmk_silu_mul(const void* gu, int64_t gu_stride, void* out, int64_t T, int I, cudaStream_t stream) {
  const int64_t n = T * I;
  if (n <= 0) return 0;
  return lmk_launch(silu_mul_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, stream, (const bf16*)gu,
                    gu_stride, (bf16*)out, T, I);
}

// Attention output gate. o is [T, H*D] contiguous; gate is [T, H, D] at token stride `gs` and head stride `hs`;
// out [T, H*D] = bf16(o * sigmoid(gate)).
__global__ void sigmoid_gate_kernel(const bf16* __restrict__ o, const bf16* __restrict__ g, int64_t gs, int64_t hs,
                                    bf16* __restrict__ out, int64_t T, int H, int D) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t HD = (int64_t)H * D;
  if (idx >= T * HD) return;
  const int64_t t = idx / HD;
  const int64_t rem = idx % HD;
  const int h = (int)(rem / D), d = (int)(rem % D);
  const float gv = bf2f(g[t * gs + h * hs + d]);
  out[idx] = f2bf(fmul(bf2f(o[idx]), t_sigmoid(gv)));
}

LMK_API int lmk_sigmoid_gate(const void* o, const void* gate, int64_t g_stride, int64_t g_head_stride, void* out,
                             int64_t T, int H, int D, cudaStream_t stream) {
  const int64_t n = T * H * D;
  if (n <= 0) return 0;
  return lmk_launch(sigmoid_gate_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, stream, (const bf16*)o,
                    (const bf16*)gate, g_stride, g_head_stride, (bf16*)out, T, H, D);
}
