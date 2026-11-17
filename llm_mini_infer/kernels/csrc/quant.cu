// Dynamic FP8 activation quantization for the block-FP8 linear layers (docs/architecture.md, "FP8"): e4m3 values
// with one FP32 scale per token and per group of 128 values along K. The numerics are
//   scale = max(absmax, 1e-10) / 448;   q = e4m3_rn(clamp(x * (1 / scale), -448, 448))
// with IEEE divisions (1 / scale once a group), a product rounded to nearest and round-to-nearest-even, as vLLM's
// standalone quantizer computes it; SiLU-and-mul divides, q = e4m3_rn(clamp(x / scale, -448, 448)) (below). The torch
// reference is `quant_ref` in kernels/fp8.py.
//
// Layouts: q is [T, K] row-major. The scales are "MN-major" [K / 128, ld]: the scale of token t and group g is at
// s[g * ld + t], with ld >= T. The CUTLASS blockwise GEMM reads the activation scales this way.
//
// The fused producers round to BF16 where the unfused BF16 path rounds (the norms, the attention gate), then quantize
// the rounded values. The result equals the BF16 op followed by the standalone quantization, bit for bit.
// SiLU-and-mul quantizes the fp32 product without rounding it, and divides by the scale, as vLLM's fused
// silu_and_mul_per_block_quant does.
// vLLM's activation-quant fusion pass substitutes that kernel for the two ops (its log shows the match on
// Qwen3.8-27B-FP8). That kernel floors the scale at 1 / (448 * 512), and its clamp turns a NaN into +448. Ours floors
// the absmax at 1e-10, as vLLM's standalone quantization does, and turns a NaN into -448. The two differ only for a
// group with an absmax below 1/512, and on a NaN.
//
// Programmatic dependent launch (common.cuh): every kernel here, except the eager weight dequantization, first waits
// for the previous kernel, then lets the next one launch. The launchers add the attribute while the library's flag
// is on (the runner's `dev.runner.cuda.pdl`).
#include "quant.cuh"

namespace {

using namespace lmk_quant;

// Standalone: x [T, K] BF16 with row stride xs (in elements, a multiple of 4) -> q [T, K], s [K / 128, ld].
__global__ void quant_fp8_kernel(const bf16* __restrict__ x, int64_t xs, uint8_t* __restrict__ q,
                                 float* __restrict__ s, int64_t T, int K, int64_t ld) {
  lmk_pdl_wait();  // PDL: nothing a predecessor writes is read before it (common.cuh)
  lmk_pdl_trigger();
  const int G = K / kGroup;
  const int64_t item = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (item >= T * G) return;
  const int lane = threadIdx.x & 31;
  const int64_t t = item / G;
  const int g = (int)(item % G);
  float v[4];
  load_bf16x4(x + t * xs + g * kGroup + 4 * lane, v);
  quant_store4(v, q + t * K + g * kGroup, s + (int64_t)g * ld + t, lane);
}

// Gemma RMSNorm (+ residual), then quantization, with the arithmetic of the BF16 kernel (norm.cu,
// gemma_rmsnorm_kernel). Each output rounds to BF16 (and goes to `out` when given), then quantizes per group of 128.
// One block per row.
__global__ void gemma_rmsnorm_quant_kernel(const bf16* __restrict__ x, int64_t xs, const bf16* __restrict__ res,
                                           const bf16* __restrict__ w, bf16* __restrict__ out,
                                           bf16* __restrict__ res_out, uint8_t* __restrict__ q,
                                           float* __restrict__ s, int64_t ld, int D, float eps) {
  lmk_pdl_wait();  // PDL: nothing a predecessor writes is read before it (common.cuh)
  lmk_pdl_trigger();
  extern __shared__ float row[];  // D floats: the input (summed with the residual)
  __shared__ float red[32];
  const int64_t r = blockIdx.x;
  const bf16* xr = x + r * xs;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float v = bf2f(xr[i]);
    if (res != nullptr) {
      v = fadd(v, bf2f(res[r * D + i]));
      res_out[r * D + i] = f2bf(v);
    }
    row[i] = v;
    ss = fadd(ss, fmul(v, v));
  }
  ss = block_sum(ss, red);  // its barriers also publish row[] to every warp
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int g = warp; g < D / kGroup; g += nw) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = g * kGroup + 4 * lane + j;
      const bf16 y = f2bf(fmul(fmul(row[i], inv), fadd(1.0f, bf2f(w[i]))));
      if (out != nullptr) out[r * D + i] = y;
      v[j] = bf2f(y);
    }
    quant_store4(v, q + r * D + g * kGroup, s + (int64_t)g * ld + r, lane);
  }
}

// The GDN output norm, then quantization, for heads of D = 128 (one group per head). The arithmetic is that of the
// BF16 kernel (norm.cu, gated_rmsnorm_kernel), rounded to BF16. One warp per (token, head). x is [T, H, 128]
// contiguous, z has token stride zs, q is [T, H * 128], s is [H, ld].
__global__ void gated_rmsnorm_quant_kernel(const bf16* __restrict__ x, const bf16* __restrict__ z, int64_t zs,
                                           const bf16* __restrict__ w, uint8_t* __restrict__ q, float* __restrict__ s,
                                           int64_t ld, int64_t n, int H, float eps) {
  lmk_pdl_wait();  // PDL: nothing a predecessor writes is read before it (common.cuh)
  lmk_pdl_trigger();
  const int64_t item = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (item >= n) return;
  const int64_t t = item / H;
  const int h = (int)(item % H);
  const bf16* xr = x + item * kGroup;
  const bf16* zr = z + t * zs + (int64_t)h * kGroup;
  // The lane-to-element mapping (lane + 32 k) of the BF16 kernel, so the sum of squares adds in the same order.
  float v[4];
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    v[k] = bf2f(xr[lane + 32 * k]);
    ss = fadd(ss, fmul(v[k], v[k]));
  }
  ss = warp_sum(ss);
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)kGroup), eps));
  float y[4];
  float am = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int i = lane + 32 * k;
    const float o = fmul(fmul(v[k], inv), bf2f(w[i]));
    y[k] = bf2f(f2bf(fmul(o, t_silu(bf2f(zr[i])))));
    am = fmaxf(am, fabsf(y[k]));
  }
  am = warp_max(am);
  const float sc = group_scale(am);
  const float qinv = group_inv(sc);
  uint8_t* qr = q + t * (int64_t)H * kGroup + h * kGroup;
#pragma unroll
  for (int k = 0; k < 4; ++k) qr[lane + 32 * k] = static_cast<uint8_t>(fp8_byte(y[k], qinv));
  if (lane == 0) s[(int64_t)h * ld + t] = sc;
}

// SiLU-and-mul, then quantization of the fp32 product by division (see the top of the file). gu is [T, 2I] (gate,
// then up) with row stride gs -> q [T, I], s [I / 128, ld]. One warp per (token, group).
__global__ void silu_mul_quant_kernel(const bf16* __restrict__ gu, int64_t gs, uint8_t* __restrict__ q,
                                      float* __restrict__ s, int64_t T, int I, int64_t ld) {
  lmk_pdl_wait();  // PDL: nothing a predecessor writes is read before it (common.cuh)
  lmk_pdl_trigger();
  const int G = I / kGroup;
  const int64_t item = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (item >= T * G) return;
  const int lane = threadIdx.x & 31;
  const int64_t t = item / G;
  const int g = (int)(item % G);
  float gv[4], uv[4], v[4];
  load_bf16x4(gu + t * gs + g * kGroup + 4 * lane, gv);
  load_bf16x4(gu + t * gs + I + g * kGroup + 4 * lane, uv);
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = fmul(t_silu(gv[j]), uv[j]);
  quant_store4<true>(v, q + t * I + g * kGroup, s + (int64_t)g * ld + t, lane);
}

// The attention output gate, then quantization. The product of o [T, H*D] (contiguous) and sigmoid(gate [T, H, D],
// token stride gs, head stride hs) rounds to BF16, as in the BF16 kernel, then quantizes per group of 128 along H*D.
// D is a multiple of 128. One warp per (token, group).
__global__ void sigmoid_gate_quant_kernel(const bf16* __restrict__ o, const bf16* __restrict__ gate, int64_t gs,
                                          int64_t hs, uint8_t* __restrict__ q, float* __restrict__ s, int64_t T, int H,
                                          int D, int64_t ld) {
  lmk_pdl_wait();  // PDL: nothing a predecessor writes is read before it (common.cuh)
  lmk_pdl_trigger();
  const int K = H * D;
  const int G = K / kGroup;
  const int64_t item = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (item >= T * G) return;
  const int lane = threadIdx.x & 31;
  const int64_t t = item / G;
  const int g = (int)(item % G);
  const int e = g * kGroup + 4 * lane;  // first of this lane's 4 elements (all in one head, as D % 128 == 0)
  const int h = e / D, d = e % D;
  float ov[4], gv[4], v[4];
  load_bf16x4(o + t * K + e, ov);
  load_bf16x4(gate + t * gs + h * hs + d, gv);
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = bf2f(f2bf(fmul(ov[j], t_sigmoid(gv[j]))));
  quant_store4(v, q + t * K + g * kGroup, s + (int64_t)g * ld + t, lane);
}

// Dequantizes a block-FP8 weight to BF16 (the precise prefix's BF16 path; kernels/fp8.py `linear_bf16`):
// out[n, k] = bf16(float(w[n, k]) * ws[n / 128, k / 128]), the product in fp32, as the torch reference
// (`fp8.dequant_weight`, then a cast). 16 values per thread; K is a multiple of 128, so they share one row and block.
__global__ void dequant_fp8_kernel(const uint8_t* __restrict__ w, const float* __restrict__ ws, bf16* __restrict__ out,
                                   int64_t total, int K) {
  const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 16;
  if (i >= total) return;
  const int64_t n = i / K;
  const int k = (int)(i % K);
  const float sc = ws[(n / kGroup) * (K / kGroup) + k / kGroup];
  const uint4 raw = *reinterpret_cast<const uint4*>(w + i);
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
  __align__(16) bf16 o[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    __nv_fp8_e4m3 f;
    f.__x = b[j];
    o[j] = f2bf(fmul(static_cast<float>(f), sc));
  }
  reinterpret_cast<uint4*>(out + i)[0] = reinterpret_cast<const uint4*>(o)[0];
  reinterpret_cast<uint4*>(out + i)[1] = reinterpret_cast<const uint4*>(o)[1];
}

constexpr int kWarps = 8;  // warps per block in the one-warp-per-group kernels

inline unsigned blocks_for(int64_t items) { return (unsigned)((items + kWarps - 1) / kWarps); }

inline bool aligned4(const void* p, int64_t stride) {
  return (reinterpret_cast<uintptr_t>(p) % 8 == 0) && stride % 4 == 0;
}

}  // namespace

// Every launcher returns a cudaError_t (0 = success), or -1 for an unsupported shape or alignment. The Python wrappers
// (kernels/ops.py) check shapes first and otherwise fall back to the reference.

LMK_API int lmk_quant_fp8(const void* x, int64_t x_stride, void* q, void* s, int64_t ld, int64_t T, int K,
                          cudaStream_t stream) {
  if (T <= 0) return 0;
  if (K % kGroup || !aligned4(x, x_stride) || ld < T) return -1;
  return lmk_launch(quant_fp8_kernel, dim3(blocks_for(T * (K / kGroup))), dim3(kWarps * 32), 0, stream, (const bf16*)x,
                    x_stride, (uint8_t*)q, (float*)s, T, K, ld);
}

LMK_API int lmk_gemma_rmsnorm_quant(const void* x, int64_t x_stride, const void* residual, const void* weight,
                                    void* out, void* residual_out, void* q, void* s, int64_t ld, int64_t rows, int D,
                                    float eps, cudaStream_t stream) {
  if (rows <= 0) return 0;
  if (D % kGroup || ld < rows) return -1;
  const int threads = D >= 4096 ? 1024 : D >= 1024 ? 512 : 256;
  return lmk_launch(gemma_rmsnorm_quant_kernel, dim3((unsigned)rows), dim3(threads), D * sizeof(float), stream,
                    (const bf16*)x, x_stride, (const bf16*)residual, (const bf16*)weight, (bf16*)out,
                    (bf16*)residual_out, (uint8_t*)q, (float*)s, ld, D, eps);
}

LMK_API int lmk_gated_rmsnorm_quant(const void* x, const void* z, int64_t z_stride, const void* weight, void* q,
                                    void* s, int64_t ld, int64_t tokens, int H, int D, float eps,
                                    cudaStream_t stream) {
  const int64_t n = tokens * H;
  if (n <= 0) return 0;
  if (D != kGroup || ld < tokens) return -1;
  return lmk_launch(gated_rmsnorm_quant_kernel, dim3(blocks_for(n)), dim3(kWarps * 32), 0, stream, (const bf16*)x,
                    (const bf16*)z, z_stride, (const bf16*)weight, (uint8_t*)q, (float*)s, ld, n, H, eps);
}

LMK_API int lmk_silu_mul_quant(const void* gu, int64_t gu_stride, void* q, void* s, int64_t ld, int64_t T, int I,
                               cudaStream_t stream) {
  if (T <= 0) return 0;
  if (I % kGroup || !aligned4(gu, gu_stride) || ld < T) return -1;
  return lmk_launch(silu_mul_quant_kernel, dim3(blocks_for(T * (I / kGroup))), dim3(kWarps * 32), 0, stream,
                    (const bf16*)gu, gu_stride, (uint8_t*)q, (float*)s, T, I, ld);
}

LMK_API int lmk_dequant_fp8(const void* w, const void* ws, void* out, int64_t N, int K, cudaStream_t stream) {
  const int64_t total = N * (int64_t)K;
  if (total <= 0) return 0;
  if (K % kGroup || N % kGroup || reinterpret_cast<uintptr_t>(w) % 16 || reinterpret_cast<uintptr_t>(out) % 16)
    return -1;
  const int64_t threads = total / 16;
  dequant_fp8_kernel<<<(unsigned)((threads + 255) / 256), 256, 0, stream>>>(
      (const uint8_t*)w, (const float*)ws, (bf16*)out, total, K);
  return lmk_status();
}

LMK_API int lmk_sigmoid_gate_quant(const void* o, const void* gate, int64_t g_stride, int64_t g_head_stride, void* q,
                                   void* s, int64_t ld, int64_t T, int H, int D, cudaStream_t stream) {
  if (T <= 0) return 0;
  if (D % kGroup || !aligned4(gate, g_stride) || g_head_stride % 4 || !aligned4(o, (int64_t)H * D) || ld < T)
    return -1;
  return lmk_launch(sigmoid_gate_quant_kernel, dim3(blocks_for(T * ((int64_t)H * D / kGroup))), dim3(kWarps * 32), 0,
                    stream, (const bf16*)o, (const bf16*)gate, g_stride, g_head_stride, (uint8_t*)q, (float*)s, T, H,
                    D, ld);
}
