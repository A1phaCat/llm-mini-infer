// The vision tower's attention input in one kernel (models/qwen3_5/vision.py). q and k take the 2-D rotary of their
// patch: fp32 angles, `rotate_half` over the whole head (HF's `apply_rotary_pos_emb_vision`). The two products and the
// sum each round, as torch does with separate kernels, and the result rounds once to BF16. The kernel writes q, k and
// v as [n, h, dp] with zeros past the head dim d, for the ragged attention kernel (dp >= d: FlashInfer's head dims).
// qkv is [n, 3, h, d] with row stride `qs`. cos and sin are [n, d] fp32.
#include "common.cuh"

__global__ void vision_qkv_rope_kernel(const bf16* __restrict__ qkv, int64_t qs, const float* __restrict__ cos,
                                       const float* __restrict__ sin, bf16* __restrict__ q, bf16* __restrict__ k,
                                       bf16* __restrict__ v, int64_t n, int h, int d, int dp) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * h * dp) return;
  const int i = (int)(idx % dp);
  const int64_t th = idx / dp;
  const int hh = (int)(th % h);
  const int64_t t = th / h;
  if (i >= d) {
    q[idx] = f2bf(0.0f);
    k[idx] = f2bf(0.0f);
    v[idx] = f2bf(0.0f);
    return;
  }
  const int half = d / 2;
  const bf16* row = qkv + t * qs;
  const float c = cos[t * d + i], s = sin[t * d + i];
#pragma unroll
  for (int which = 0; which < 2; ++which) {
    const bf16* x = row + (int64_t)which * h * d + (int64_t)hh * d;
    const float xi = bf2f(x[i]);
    const float r = i < half ? -bf2f(x[i + half]) : bf2f(x[i - half]);
    const float o = fadd(fmul(xi, c), fmul(r, s));
    (which == 0 ? q : k)[idx] = f2bf(o);
  }
  v[idx] = row[(int64_t)2 * h * d + (int64_t)hh * d + i];
}

LMK_API int lmk_vision_qkv_rope(const void* qkv, int64_t qkv_stride, const void* cos, const void* sin, void* q,
                                void* k, void* v, int64_t n, int h, int d, int dp, cudaStream_t stream) {
  if (n <= 0) return 0;
  if (d % 2 || dp < d) return -1;
  const int64_t total = n * h * dp;
  vision_qkv_rope_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
      (const bf16*)qkv, qkv_stride, (const float*)cos, (const float*)sin, (bf16*)q, (bf16*)k, (bf16*)v, n, h, d, dp);
  return lmk_status();
}

// Gemma 4's vision tower: its attention input in one kernel (reference: `vision_qkv` in models/gemma4/vision.py). q =
// rope_2d(rms(q) ⊙ q_norm), k = rope_2d(rms(k) ⊙ k_norm), v = rms(v). Each runs in fp32 and rounds once to BF16. The
// kernel writes [n, h, dp] with zeros past the head dim d (dp: FlashInfer's head dims).
//
// One warp handles one (patch, head, q | k | v). Lane l holds the head's elements l, l + 32 and l + 64 (d <= 96),
// squares them, sums them in that order, then sums across the warp (`warp_sum`'s xor tree). This is the order of
// torch's (2.14.1, the pinned version) CUDA mean over a last dimension below 128 with at least 16 outputs: one warp per
// output, each lane accumulating its strided elements, then shuffles at 16, 8, 4, 2, 1. With fewer outputs, torch
// widens its block to 64 lanes and adds l + 64 before l + 32 when d > 64, and the caller runs the torch ops instead. So
// the norm is torch's bit for bit. Then come × `factor` (torch's mean: output count over element count, in fp32), + eps
// and rsqrtf, each step rounding as in torch's separate kernels.
//
// The 2-D RoPE follows HF's `apply_multidimensional_rope`. Each half of the head has its own NeoX rotation with pairs
// (j, j + d / 4) inside the half. The first half uses the patch's column angles, the second its row angles. It rounds
// each product, then the sum, as torch's separate kernels do.
__global__ void g4_vision_qkv_kernel(const bf16* __restrict__ qkv, int64_t qs, const bf16* __restrict__ qw,
                                     const bf16* __restrict__ kw, const float* __restrict__ cos,
                                     const float* __restrict__ sin, float eps, float factor, bf16* __restrict__ q,
                                     bf16* __restrict__ k, bf16* __restrict__ v, int64_t n, int h, int d, int dp) {
  __shared__ float xs[4][96];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int64_t w = (int64_t)blockIdx.x * 4 + wid;  // (patch, head, which)
  if (w >= n * h * 3) return;
  const int which = (int)(w % 3);
  const int64_t th = w / 3;
  const int hh = (int)(th % h);
  const int64_t t = th / h;
  const bf16* x = qkv + t * qs + ((int64_t)which * h + hh) * d;
  float xv[3];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int e = lane + 32 * j;
    xv[j] = e < d ? bf2f(x[e]) : 0.0f;
    if (e < d) ss = fadd(ss, fmul(xv[j], xv[j]));
  }
  ss = warp_sum(ss);
  const float inv = rsqrtf(fadd(fmul(ss, factor), eps));
  bf16* out = (which == 0 ? q : which == 1 ? k : v) + (t * h + hh) * (int64_t)dp;
  if (which == 2) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int e = lane + 32 * j;
      if (e < d) out[e] = f2bf(fmul(xv[j], inv));
    }
  } else {
    const bf16* wt = which == 0 ? qw : kw;
    float* sx = xs[wid];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int e = lane + 32 * j;
      if (e < d) sx[e] = fmul(fmul(xv[j], inv), bf2f(wt[e]));
    }
    __syncwarp();
    const int half = d / 2, quarter = d / 4;
    const float* cr = cos + t * half;
    const float* sr = sin + t * half;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int e = lane + 32 * j;
      if (e >= d) continue;
      const int a = e / half, p = e % half, i = a * quarter + p % quarter;
      const float c = cr[i], s = sr[i];
      const float o = p < quarter ? fsub(fmul(sx[e], c), fmul(sx[e + quarter], s))
                                  : fadd(fmul(sx[e], c), fmul(sx[e - quarter], s));
      out[e] = f2bf(o);
    }
  }
  for (int e = d + lane; e < dp; e += 32) out[e] = f2bf(0.0f);
}

// qkv [n, 3, h, d] has row stride `qkv_stride` (elements; inner dims contiguous). q_w and k_w are [d]. cos and sin are
// [n, d / 2] fp32 contiguous. q, k and v are [n, h, dp]. d is a multiple of 4, at most 96. `factor` is torch's mean
// factor for [n, h, d] (fp32 n·h / (n·h·d)).
LMK_API int lmk_g4_vision_qkv(const void* qkv, int64_t qkv_stride, const void* q_w, const void* k_w, const void* cos,
                              const void* sin, float eps, float factor, void* q, void* k, void* v, int64_t n, int h,
                              int d, int dp, cudaStream_t stream) {
  if (n <= 0) return 0;
  if (d % 4 || d > 96 || dp < d || h <= 0) return -1;
  const int64_t warps = n * h * 3;
  g4_vision_qkv_kernel<<<(unsigned)((warps + 3) / 4), 128, 0, stream>>>(
      (const bf16*)qkv, qkv_stride, (const bf16*)q_w, (const bf16*)k_w, (const float*)cos, (const float*)sin, eps,
      factor, (bf16*)q, (bf16*)k, (bf16*)v, n, h, d, dp);
  return lmk_status();
}
